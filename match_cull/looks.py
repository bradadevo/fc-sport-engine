from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

DEFAULT_LUT_DIR = Path.home() / "Library/Application Support/ProApps/Custom LUTs"
# Pixels processed per slice by the numpy LUT fallback, to bound memory
_CHUNK_PIXELS = 1_000_000
# A LUT may not darken pixels below this brightness (fully protected at black)
SHADOW_GUARD_END = 130
# Grass: the furthest a look may turn green toward yellow or teal (Lab hue
# degrees), and the range it may scale grass colorfulness, before the look is
# automatically applied at reduced strength
MAX_GRASS_HUE_SHIFT = 5.0
GRASS_CHROMA_RANGE = (0.75, 1.3)
MIN_STRENGTH = 0.3
# Field greens from shade to full sun (sunlit grass is already yellow-green)
GRASS_SAMPLES = [(70, 140, 60), (90, 160, 80), (60, 120, 50), (120, 180, 90),
                 (140, 175, 70), (165, 190, 85), (150, 185, 100), (175, 195, 95), (155, 180, 60)]


@dataclass
class Lut:
    name: str
    table: np.ndarray  # (N, N, N, 3) indexed [blue, green, red]
    domain_min: np.ndarray
    domain_max: np.ndarray
    strength: float = 1.0  # how much of the look to apply (see auto_strength)

    @property
    def size(self) -> int:
        return self.table.shape[0]


def parse_cube(path: str) -> Lut:
    """Reads an Adobe/Resolve .cube 3D LUT (red varies fastest in the data)."""
    size = None
    domain_min = np.zeros(3, dtype=np.float32)
    domain_max = np.ones(3, dtype=np.float32)
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split()[0].upper()
        if key == "LUT_3D_SIZE":
            size = int(line.split()[1])
        elif key == "DOMAIN_MIN":
            domain_min = np.array(line.split()[1:4], dtype=np.float32)
        elif key == "DOMAIN_MAX":
            domain_max = np.array(line.split()[1:4], dtype=np.float32)
        elif key in ("TITLE", "LUT_1D_SIZE", "LUT_1D_INPUT_RANGE", "LUT_3D_INPUT_RANGE"):
            if key == "LUT_1D_SIZE":
                raise ValueError(f"{path} is a 1D LUT; only 3D LUTs are supported")
        else:
            try:
                rows.append([float(v) for v in line.split()[:3]])
            except ValueError:
                continue
    if not size or len(rows) != size ** 3:
        raise ValueError(f"{path} is not a valid 3D .cube LUT")
    table = np.array(rows, dtype=np.float32).reshape(size, size, size, 3)
    return Lut(Path(path).stem, table, domain_min, domain_max)


def apply_lut(img: Image.Image, lut: Lut) -> Image.Image:
    """Applies the LUT with Pillow's built-in 3D LUT filter (trilinear,
    implemented in C). .cube data and Pillow both list red fastest, so the
    table passes straight through."""
    if not (np.allclose(lut.domain_min, 0) and np.allclose(lut.domain_max, 1)):
        return _apply_lut_any_domain(img, lut)
    table = ImageFilter.Color3DLUT(lut.size, lut.table.reshape(-1).tolist())
    return img.convert("RGB").filter(table)


def _apply_lut_any_domain(img: Image.Image, lut: Lut) -> Image.Image:
    """Fallback for the rare .cube with a DOMAIN other than 0-1, which
    Pillow's filter cannot express: trilinear interpolation in numpy."""
    src = np.asarray(img.convert("RGB"))
    h, w, _ = src.shape
    flat = src.reshape(-1, 3)
    out = np.empty_like(flat)
    n = lut.size
    scale = (n - 1) / (lut.domain_max - lut.domain_min)
    t = lut.table
    for start in range(0, flat.shape[0], _CHUNK_PIXELS):
        x = flat[start:start + _CHUNK_PIXELS].astype(np.float32) / 255.0
        idx = np.clip((x - lut.domain_min) * scale, 0, n - 1)
        i0 = np.minimum(idx.astype(np.int32), n - 2)
        f = idx - i0
        r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
        fr, fg, fb = f[:, 0:1], f[:, 1:2], f[:, 2:3]
        c00 = t[b0, g0, r0] + (t[b0, g0, r0 + 1] - t[b0, g0, r0]) * fr
        c01 = t[b0, g0 + 1, r0] + (t[b0, g0 + 1, r0 + 1] - t[b0, g0 + 1, r0]) * fr
        c10 = t[b0 + 1, g0, r0] + (t[b0 + 1, g0, r0 + 1] - t[b0 + 1, g0, r0]) * fr
        c11 = t[b0 + 1, g0 + 1, r0] + (t[b0 + 1, g0 + 1, r0 + 1] - t[b0 + 1, g0 + 1, r0]) * fr
        c0 = c00 + (c01 - c00) * fg
        c1 = c10 + (c11 - c10) * fg
        res = c0 + (c1 - c0) * fb
        out[start:start + _CHUNK_PIXELS] = np.clip(res * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(out.reshape(h, w, 3))


def _luma(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def apply_look(img: Image.Image, lut: Lut) -> Image.Image:
    """Applies a creative LUT without letting it crush shadows: wherever the
    LUT makes a dark pixel darker, the result is blended back toward the
    pre-look pixel, fully at black and fading out by SHADOW_GUARD_END. Dark
    skin and dark kits keep their detail; brighter tones get the full look.
    (A blend rather than a brightness scale, because a LUT that outputs pure
    black cannot be scaled back up.)"""
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    looked = np.asarray(apply_lut(img, lut), dtype=np.float32)
    looked = base + (looked - base) * lut.strength
    lum_in, lum_out = _luma(base), _luma(looked)
    guard = np.clip(1 - lum_in / SHADOW_GUARD_END, 0, 1) ** 0.5
    weight = np.where(lum_out < lum_in, guard, 0.0)[..., None]
    out = looked + (base - looked) * weight
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def discover_luts(folder: Path = DEFAULT_LUT_DIR) -> dict[str, str]:
    """Maps LUT name -> .cube path for every LUT under the folder. When the
    same name appears twice, the first one found (by path) wins."""
    found: dict[str, str] = {}
    if not Path(folder).is_dir():
        return found
    for path in sorted(Path(folder).rglob("*.cube")):
        if path.name.startswith("."):
            continue
        found.setdefault(path.stem, str(path))
    return found


def describe_lut(lut: Lut) -> str:
    """A few words on what the LUT does (warmth, contrast, color, brightness),
    measured by running test colors through it. Gives the model something
    to go on beyond the LUT's name."""
    def run(rgb):
        img = Image.fromarray(np.array(rgb, dtype=np.uint8).reshape(1, -1, 3))
        return np.asarray(apply_lut(img, lut), dtype=np.float32).reshape(-1, 3)

    grays = run([[v, v, v] for v in (64, 128, 192)])
    shadow, mid, highlight = grays
    words = []

    warmth = mid[0] - mid[2]
    words.append("warm" if warmth > 8 else "cool" if warmth < -8 else "neutral color balance")

    contrast = (highlight.mean() - shadow.mean()) / 128.0
    words.append("high contrast" if contrast > 1.12 else "soft contrast" if contrast < 0.88 else "normal contrast")

    colors = [[200, 60, 60], [60, 160, 60], [60, 90, 200], [210, 160, 120]]
    before = np.array(colors, dtype=np.float32)
    after = run(colors)
    sat = lambda c: (c.max(axis=1) - c.min(axis=1)).mean()
    ratio = sat(after) / sat(before)
    words.append("punchy colors" if ratio > 1.12 else "muted colors" if ratio < 0.85 else "natural saturation")

    shift = mid.mean() - 128
    if shift > 10:
        words.append("brighter")
    elif shift < -10:
        words.append("darker")
    grass_shift, _ = grass_effect(lut)
    if grass_shift < -MAX_GRASS_HUE_SHIFT:
        words.append("turns grass yellow")
    elif grass_shift > MAX_GRASS_HUE_SHIFT:
        words.append("turns grass teal")
    if lut.strength < 0.99:
        words.append(f"applied at {round(lut.strength * 100)}% to keep grass natural")
    return ", ".join(words)


def _hue_chroma(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    return np.degrees(np.arctan2(b, a)), np.hypot(a, b)


def grass_effect(lut: Lut) -> tuple[float, float]:
    """(hue shift in degrees, chroma ratio) the full-strength look gives field
    greens. Negative hue shift = toward yellow, positive = toward teal."""
    greens = np.array(GRASS_SAMPLES, dtype=np.uint8).reshape(1, -1, 3)
    full = Lut(lut.name, lut.table, lut.domain_min, lut.domain_max)
    out = np.asarray(apply_lut(Image.fromarray(greens), full))
    h0, c0 = _hue_chroma(greens)
    h1, c1 = _hue_chroma(out)
    return float((h1 - h0).mean()), float(c1.mean() / c0.mean())


def auto_strength(lut: Lut) -> float:
    """The strongest amount of the look that keeps grass natural: within
    MAX_GRASS_HUE_SHIFT of its hue and GRASS_CHROMA_RANGE of its color.
    (A blend is close to linear in these measures, so the limits scale.)"""
    shift, ratio = grass_effect(lut)
    strength = 1.0
    if abs(shift) > MAX_GRASS_HUE_SHIFT:
        strength = min(strength, MAX_GRASS_HUE_SHIFT / abs(shift))
    lo, hi = GRASS_CHROMA_RANGE
    if ratio > hi:
        strength = min(strength, (hi - 1) / (ratio - 1))
    elif ratio < lo:
        strength = min(strength, (1 - lo) / (1 - ratio))
    return max(MIN_STRENGTH, strength)
