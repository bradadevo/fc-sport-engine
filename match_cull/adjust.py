"""Version D: Gemini looks at the finished A edit and suggests slider
adjustments, like an editor at Lightroom's basic panel. Our code applies
them faithfully; nothing is generated or redrawn, so faces and details stay
real. Every slider is clamped to a safe range, and the same rules as the
rest of the editing hold: shadows are never crushed, and any brightening of
the players goes through a feathered mask.
"""

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from match_cull import enhance
from sports_reel.vision import generate_structured, image_to_jpeg

CACHE_FILENAME = ".match_cull_adjustments.json"

# Slider ranges: (lowest, highest). Values outside are clamped.
RANGES = {
    "exposure": (-1.0, 1.0),
    "contrast": (-30, 30),
    "highlights": (-50, 50),
    "shadows": (-15, 50),
    "warmth": (-30, 30),
    "tint": (-30, 30),
    "vibrance": (-30, 30),
    "clarity": (-30, 50),
    "subject_brightness": (0, 40),
}

PROMPT = """You are a sports photo editor at Lightroom's basic panel. This is an edit of a youth
soccer photo that has already been cropped and given a natural correction. Suggest small slider
adjustments that would make it look its best for this photographer. Be conservative: 0 means
"leave as is", and most photos need only a few sliders moved.

Sliders and ranges:
- exposure: -1.0 to +1.0 stops (whole photo)
- contrast: -30 to +30
- highlights: -50 to +50 (negative recovers bright areas such as white kits and sky)
- shadows: -15 to +50 (positive opens up dark areas)
- warmth: -30 (cooler) to +30 (warmer)
- tint: -30 (greener) to +30 (more magenta)
- vibrance: -30 to +30 (boosts muted colors more than strong ones)
- clarity: -30 to +50 (crispness of detail)
- subject_brightness: 0 to +40 (brightens only the players, e.g. faces in shadow)
Also give reason: one short sentence on what you changed and why.
{taste}"""

TASTE_SECTION = """
The photographer's taste, learned from their favorite photos:
{notes}
"""


class Adjustments(BaseModel):
    exposure: float = Field(description="Stops, -1.0 to +1.0")
    contrast: float = Field(description="-30 to +30")
    highlights: float = Field(description="-50 to +50")
    shadows: float = Field(description="-15 to +50")
    warmth: float = Field(description="-30 to +30")
    tint: float = Field(description="-30 to +30")
    vibrance: float = Field(description="-30 to +30")
    clarity: float = Field(description="-30 to +50")
    subject_brightness: float = Field(description="0 to +40, brightens only the players")
    reason: str = Field(description="One short sentence on what changed and why")

    def clamped(self) -> "Adjustments":
        values = {k: float(min(max(getattr(self, k), lo), hi)) for k, (lo, hi) in RANGES.items()}
        return Adjustments(reason=self.reason, **values)

    def summary(self) -> str:
        """Compact text of the sliders that moved, e.g. 'exposure +0.3, highlights -25'."""
        moved = []
        for key in RANGES:
            value = getattr(self, key)
            if abs(value) >= (0.05 if key == "exposure" else 1):
                moved.append(f"{key.replace('_', ' ')} {value:+.1f}" if key == "exposure"
                             else f"{key.replace('_', ' ')} {value:+.0f}")
        return ", ".join(moved) or "no changes"


def no_changes(reason: str) -> Adjustments:
    return Adjustments(exposure=0, contrast=0, highlights=0, shadows=0, warmth=0, tint=0,
                       vibrance=0, clarity=0, subject_brightness=0, reason=reason)


def suggest(client, img: Image.Image, model: str, taste_notes: str = "",
            usage: list | None = None) -> Adjustments:
    prompt = PROMPT.format(taste=TASTE_SECTION.format(notes=taste_notes) if taste_notes else "")
    return generate_structured(client, image_to_jpeg(img), prompt, Adjustments,
                               model=model, usage=usage).clamped()


def _luma(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def _set_luma(arr: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Changes brightness to target while keeping each pixel's color."""
    lum = _luma(arr)
    return arr * ((np.clip(target, 0, 255) + 1) / (lum + 1))[..., None]


def _protect_shadows(before: np.ndarray, after: np.ndarray, end: float = 140) -> np.ndarray:
    """Undoes darkening in the shadows (almost fully through the darker skin
    and navy range, fading out by `end`)."""
    guard = np.clip(1 - before / end, 0, 1) ** 0.3
    return np.where(after < before, after + (before - after) * guard, after)


def apply(img: Image.Image, adj: Adjustments, subject_box: tuple[int, int, int, int] | None = None) -> Image.Image:
    """Applies the (clamped) sliders. subject_box is in this image's pixels."""
    adj = adj.clamped()
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)

    # Exposure: brighten with a gamma lift; darken through the highlights only
    if adj.exposure > 0:
        arr = 255.0 * (np.clip(arr, 0, 255) / 255.0) ** (1 / (1 + 0.6 * adj.exposure))
    elif adj.exposure < 0:
        lum = _luma(arr)
        darker = lum * (1 + 0.35 * adj.exposure)
        weight = np.clip((lum - 100) / 155, 0, 1)
        arr = _set_luma(arr, lum + (darker - lum) * weight)

    # Contrast around mid-gray; extra contrast may not crush shadows
    if adj.contrast:
        lum = _luma(arr)
        target = 128 + (lum - 128) * (1 + adj.contrast / 100)
        arr = _set_luma(arr, _protect_shadows(lum, target))

    # Highlights: move tones above 150 (negative = recover)
    if adj.highlights:
        lum = _luma(arr)
        t = np.clip((lum - 150) / 105, 0, 1)
        arr = _set_luma(arr, lum + adj.highlights / 100 * 40 * t)

    # Shadows: open up (or gently deepen) tones below 110
    if adj.shadows:
        lum = _luma(arr)
        t = np.clip(1 - lum / 110, 0, 1) ** 1.5
        arr = _set_luma(arr, lum + adj.shadows / 100 * 40 * t)

    # Warmth and tint in Lab
    if adj.warmth or adj.tint:
        lab = cv2.cvtColor(np.clip(arr + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[..., 2] += adj.warmth * 0.25
        lab[..., 1] += adj.tint * 0.2
        arr = cv2.cvtColor(np.clip(lab, 0, 255).round().astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    # Vibrance: boosts muted colors more, keeps brightness
    if adj.vibrance:
        before = _luma(arr)
        hsv = cv2.cvtColor(np.clip(arr + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[..., 1]
        hsv[..., 1] = np.clip(sat + adj.vibrance / 100 * sat * (1 - sat / 255), 0, 255)
        arr = cv2.cvtColor(hsv.round().astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        arr = _set_luma(arr, before)

    # Clarity: more detail (OpenCV detailEnhance), or a gentle softening
    if adj.clarity > 0:
        arr = enhance.clarity(arr, amount=adj.clarity / 100 * enhance.CLARITY_AMOUNT * 1.5)
    elif adj.clarity < 0:
        soft = cv2.GaussianBlur(arr, (0, 0), 1.2)
        arr = arr + (soft - arr) * (-adj.clarity / 100)

    # Subject brightness: lift only the players' darker tones, feathered
    if adj.subject_brightness > 0 and subject_box is not None:
        mask = enhance.subject_mask(arr.shape[:2], subject_box)
        lum = _luma(arr)
        t = np.clip(1 - lum / 170, 0, 1)
        lifted = lum + adj.subject_brightness * t
        arr = _set_luma(arr, lum + (lifted - lum) * mask)

    return Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8))
