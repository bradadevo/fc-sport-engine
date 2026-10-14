"""The photographer's own look, learned from a folder of their favorite photos.

learn() measures the references (brightness, contrast, saturation, warmth,
highlight handling) and asks Gemini once to describe their taste in moments
and composition. apply_style() then nudges every finished edit toward those
measurements, so a heavy LUT is pulled back toward how the photographer
actually edits. All adjustments are partial and capped, and shadows are
never darkened.
"""

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from google.genai import types
from PIL import Image

from match_cull import images

STYLE_PATH = Path.home() / ".config" / "match-cull" / "style.json"
# Below this average color, a reference is black and white and is left out of
# the color measurements
BW_CHROMA = 3.0
# How far each edit moves toward the learned look (0 = not at all, 1 = fully)
STRENGTH = 0.35
# Caps on how much any single edit may change
MAX_CHROMA_SCALE = (0.75, 1.15)
MAX_WARMTH_SHIFT = 6.0
MAX_TINT_SHIFT = 4.0
MAX_BRIGHTNESS_SHIFT = 2.0
# Lightness (0-100) above which highlights are rolled off when an edit clips
# more than the references do
HIGHLIGHT_KNEE = 90.0
# Share of the learned look that comes from the photographer's very best
# photos (the rest comes from all the others), when any are named
FEATURED_SHARE = 0.5
# Reference photos sent to Gemini to describe the photographer's taste
TASTE_SAMPLES = 10
TASTE_EDGE = 768

FEATURED_NOTE = """The first {n} photo(s) are the photographer's very best work: weight them most
heavily, and where they differ from the others, follow them.
"""

TASTE_PROMPT = """These are a sports photographer's favorite photos from youth soccer matches.
In 5 to 8 short bullet points, describe what this photographer values, so another editor could
pick and edit photos the way they would:
- which kinds of moments and subjects they keep (action, emotion, team photos, coaches, portraits...)
- how they compose and frame
- their editing look (brightness, contrast, color, warmth, black and white use)
Answer with the bullet points only."""


@dataclass
class Style:
    source: str
    count: int
    lightness: float  # median L*, 0-100
    contrast: float  # L* spread between the 5th and 95th percentile
    chroma: float  # mean colorfulness
    warmth: float  # mean b* (yellow-blue)
    tint: float  # mean a* (magenta-green)
    highlight_clip: float  # percent of pixels at L* > 98
    notes: str = ""
    featured: list | None = None  # file names of the photographer's very best photos


def _lab(img: Image.Image) -> np.ndarray:
    """Lab with L* 0-100 and a*, b* centered on 0. Letterbox bars are removed
    so frames grabbed from video measure correctly."""
    arr = np.asarray(img.convert("RGB"))
    rows = arr.mean(axis=(1, 2)) > 8
    cols = arr.mean(axis=(0, 2)) > 8
    if rows.sum() > 16 and cols.sum() > 16:
        arr = arr[rows][:, cols]
    lab = cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    return lab


def measure(img: Image.Image) -> dict:
    lab = _lab(img)
    lightness, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    return {
        "lightness": float(np.median(lightness)),
        "contrast": float(np.percentile(lightness, 95) - np.percentile(lightness, 5)),
        "chroma": float(np.hypot(a, b).mean()),
        "warmth": float(b.mean()),
        "tint": float(a.mean()),
        "highlight_clip": float((lightness > 98).mean() * 100),
    }


def _taste_notes(client, paths: list[str], model: str, usage: list | None, featured: int = 0) -> str:
    parts = []
    for path in paths[:TASTE_SAMPLES]:
        img = images.load_image(path)
        img.thumbnail((TASTE_EDGE, TASTE_EDGE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    prompt = (FEATURED_NOTE.format(n=featured) if featured else "") + TASTE_PROMPT
    response = client.models.generate_content(model=model, contents=parts + [prompt],
                                              config=types.GenerateContentConfig(temperature=0.2))
    if usage is not None:
        meta = getattr(response, "usage_metadata", None)
        usage.append(getattr(meta, "total_token_count", None) or 0)
    return (response.text or "").strip()


def learn(folder: str, client=None, model: str = "", usage: list | None = None,
          featured: list[str] | None = None) -> Style:
    """Measures every readable photo in the folder (color stats skip black and
    white ones) and, with a client, asks Gemini to describe the taste.
    featured names the photographer's very best photos (file names in the
    folder): they make up FEATURED_SHARE of the learned look and are shown to
    Gemini first."""
    best = {f.strip() for f in featured or []}
    ordered = sorted(images.find_photos(folder), key=lambda p: Path(p).name not in best)
    missing = best - {Path(p).name for p in ordered}
    if missing:
        raise ValueError(f"Not found in {folder}: {', '.join(sorted(missing))}")
    paths, stats, best_stats = [], [], []
    for path in ordered:
        try:
            img = images.load_image(path)
        except Exception:
            continue
        paths.append(path)
        img.thumbnail((1200, 1200))
        m = measure(img)
        if m["chroma"] >= BW_CHROMA:
            (best_stats if Path(path).name in best else stats).append(m)
    if not stats and not best_stats:
        raise ValueError(f"No readable color photos in {folder}")
    keys = (stats or best_stats)[0].keys()
    median = {k: float(np.median([s[k] for s in stats or best_stats])) for k in keys}
    if best_stats and stats:
        for k in keys:
            best_value = float(np.median([s[k] for s in best_stats]))
            median[k] = FEATURED_SHARE * best_value + (1 - FEATURED_SHARE) * median[k]
    notes = _taste_notes(client, paths, model, usage, featured=len(best)) if client is not None else ""
    return Style(source=str(folder), count=len(paths), notes=notes, featured=sorted(best) or None, **median)


def save(style: Style, path: Path | None = None) -> None:
    path = Path(path or STYLE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(style), indent=1))


def load(path: Path | None = None) -> Style | None:
    try:
        return Style(**json.loads(Path(path or STYLE_PATH).read_text()))
    except (OSError, ValueError, TypeError):
        return None


def apply_style(img: Image.Image, style: Style, strength: float = STRENGTH) -> Image.Image:
    """Moves one finished edit part of the way toward the learned look:
    colorfulness, warmth/tint, brightness (shadows never darkened) and, if the
    edit clips more highlights than the references, a highlight roll-off."""
    arr = np.asarray(img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness = lab[..., 0] * 100 / 255
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    current = measure(img)

    # Colorfulness: scale a*/b* toward the reference chroma
    if current["chroma"] > 1:
        scale = np.clip(style.chroma / current["chroma"], *MAX_CHROMA_SCALE)
        scale = 1 + (scale - 1) * strength
        a, b = a * scale, b * scale
    # Warmth and tint: shift toward the reference averages
    b = b + np.clip((style.warmth - b.mean()) * strength, -MAX_WARMTH_SHIFT, MAX_WARMTH_SHIFT)
    a = a + np.clip((style.tint - a.mean()) * strength, -MAX_TINT_SHIFT, MAX_TINT_SHIFT)

    # Brightness: lift the midtones, or pull down only the bright tones
    shift = float(np.clip((style.lightness - current["lightness"]) * strength,
                          -MAX_BRIGHTNESS_SHIFT, MAX_BRIGHTNESS_SHIFT))
    if shift > 0:
        lightness = lightness + shift * np.sin(np.pi * np.clip(lightness / 100, 0, 1))
    elif shift < 0:
        lightness = lightness + shift * np.clip((lightness - 40) / 60, 0, 1)

    # Highlights: roll off above the knee when the edit clips more than the references
    if current["highlight_clip"] > style.highlight_clip + 0.3:
        over = np.clip(lightness - HIGHLIGHT_KNEE, 0, None)
        lightness = np.where(lightness > HIGHLIGHT_KNEE,
                             HIGHLIGHT_KNEE + (100 - HIGHLIGHT_KNEE) * (1 - np.exp(-over / (100 - HIGHLIGHT_KNEE))),
                             lightness)

    lab[..., 0] = np.clip(lightness * 255 / 100, 0, 255)
    lab[..., 1] = np.clip(a + 128, 0, 255)
    lab[..., 2] = np.clip(b + 128, 0, 255)
    rgb = cv2.cvtColor(lab.round().astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)
