"""Automatic photo corrections applied before the creative look.

Built on standard tools: OpenCV's gray-world white balance, CLAHE
(contrast-limited adaptive histogram equalization) for local contrast and
detailEnhance for clarity. Exposure is set for the whole scene and only ever
darkens highlights, so it cannot crush darker skin; players who are still dark
get a gentle, feathered local lift (like a Lightroom subject mask) instead of
the whole photo being brightened for them.
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter

# White balance: how much of OpenCV's gray-world correction to apply, and
# the largest change it may make to any one color channel
WB_STRENGTH = 0.5
WB_MAX_GAIN = 0.12
# Levels: where the brightest 0.5% of pixels land; the dark end never moves,
# and the bright end is stretched by at most LEVELS_MAX_STRETCH
WHITE_TARGET = 248
LEVELS_MAX_STRETCH = 1.2
# Exposure: photos whose median brightness is inside this range are left as
# shot; darker or brighter ones are moved only to the edge of the range,
# within these gamma limits
EXPOSURE_OK_RANGE = (0.28, 0.66)
GAMMA_LIMITS = (0.7, 1.5)
# Darkening only touches tones brighter than this
HIGHLIGHT_START = 100
# CLAHE on the lightness channel: strength and tile grid
CLAHE_CLIP = 1.6
CLAHE_TILES = (8, 8)
# CLAHE may not darken lightness below this (0-255 scale)
CLAHE_SHADOW_END = 110
# Subject lift (a soft local "dodge" on the players, like a Lightroom subject
# mask): only applied when the players are darker than SUBJECT_TARGET, adds at
# most SUBJECT_MAX_LIFT at black, fading to nothing by SUBJECT_LIFT_END so
# bright grass inside the box is barely touched. The mask edge is feathered
# by SUBJECT_FEATHER x the box's short side.
SUBJECT_TARGET = 105
SUBJECT_MAX_LIFT = 30
SUBJECT_LIFT_END = 150
SUBJECT_FEATHER = 0.25
# The softer B version has this much less contrast than A
SOFT_CONTRAST = 0.10
# Punch (the C version): vibrance amount, and extra contrast above midtones
PUNCH_VIBRANCE = 0.3
PUNCH_CONTRAST = 0.12
# Clarity: OpenCV's edge-preserving detail enhancement, blended in at this
# amount; sigma_s is the neighborhood size, sigma_r how strong an edge must be
# to be kept sharp (low = fewer halos)
CLARITY_AMOUNT = 0.6
# Clarity is at full strength on crops of CLARITY_FULL_MP or more, easing to
# CLARITY_SMALL_FACTOR x on crops of CLARITY_SMALL_MP or less, where it would
# mostly sharpen blur and compression noise
CLARITY_FULL_MP = 4.0
CLARITY_SMALL_MP = 1.0
CLARITY_SMALL_FACTOR = 0.4
CLARITY_SIGMA_S = 10
CLARITY_SIGMA_R = 0.15
# Clarity may not darken tones below this brightness (0-255)
CLARITY_SHADOW_END = 100

Box = tuple[int, int, int, int]


def _luma(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def _region(arr: np.ndarray, box: Box | None) -> np.ndarray:
    """The subject's pixels, or the whole image if there is no usable box."""
    if box is None:
        return arr
    left, top, right, bottom = box
    h, w = arr.shape[:2]
    left, right = max(0, left), min(w, right)
    top, bottom = max(0, top), min(h, bottom)
    if right - left < 8 or bottom - top < 8:
        return arr
    return arr[top:bottom, left:right]


def auto_white_balance(arr: np.ndarray) -> np.ndarray:
    """OpenCV's gray-world estimate of the color cast, turned into per-channel
    gains that are capped at WB_MAX_GAIN and applied at WB_STRENGTH. Gray-world
    assumes the scene averages to gray, which a frame full of grass does not,
    so its raw output can swing far toward green; the caps keep it to a nudge."""
    src = np.clip(arr, 0, 255).astype(np.uint8)
    wb = cv2.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.95)
    balanced = cv2.cvtColor(wb.balanceWhite(cv2.cvtColor(src, cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB)
    lum = _luma(arr)
    mid = (lum > 40) & (lum < 215)
    if mid.sum() < 100:
        return arr
    gains = balanced[mid].astype(np.float32).mean(axis=0) / np.maximum(src[mid].astype(np.float32).mean(axis=0), 1)
    gains = gains / gains.mean()  # correct color only, not brightness
    gains = np.clip(gains, 1 - WB_MAX_GAIN, 1 + WB_MAX_GAIN)
    return arr * (1 + (gains - 1) * WB_STRENGTH)


def auto_levels(arr: np.ndarray) -> np.ndarray:
    """Stretches the bright end so the image uses the full tonal range. The
    darkest tones stay where they are."""
    lum = _luma(arr)
    lo, hi = np.percentile(lum, [0.5, 99.5])
    if hi - lo < 10:
        return arr  # flat image (fog, solid color): leave it alone
    scale = min((WHITE_TARGET - lo) / (hi - lo), LEVELS_MAX_STRETCH)
    return (arr - lo) * scale + lo


def auto_exposure(arr: np.ndarray, subject: np.ndarray | None = None) -> np.ndarray:
    """Brightens dark subjects with a gamma curve; tames bright ones by
    pulling down only the bright tones, so shadows are never darkened.
    Judged on the subject so a sunny field does not darken the players."""
    region = arr if subject is None else subject
    median = np.median(_luma(np.clip(region, 0, 255))) / 255.0
    if median <= 0.01 or median >= 0.99:
        return arr
    low, high = EXPOSURE_OK_RANGE
    if low <= median <= high:
        return arr  # already a reasonable exposure: leave the photographer's choice alone
    target = low if median < low else high
    gamma = np.log(target) / np.log(median)
    gamma = min(max(gamma, GAMMA_LIMITS[0]), GAMMA_LIMITS[1])
    base = np.clip(arr, 0, 255)
    curved = 255.0 * (base / 255.0) ** gamma
    if gamma <= 1:
        return curved
    weight = np.clip((_luma(base) - HIGHLIGHT_START) / (255 - HIGHLIGHT_START), 0, 1)[..., None]
    return base * (1 - weight) + curved * weight


def local_contrast(arr: np.ndarray) -> np.ndarray:
    """OpenCV CLAHE on the L channel of Lab: adds clarity and opens up detail
    in shadows and highlights without changing colors. CLAHE can darken a dark
    area that sits next to a bright one (a dark face against a sunny field),
    so any darkening is undone in the shadows, fully at black and fading out
    by CLAHE_SHADOW_END."""
    src = np.clip(arr + 0.5, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
    before = lab[..., 0].astype(np.float32)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES)
    after = clahe.apply(lab[..., 0]).astype(np.float32)
    guard = np.clip(1 - before / CLAHE_SHADOW_END, 0, 1)
    lightness = np.where(after < before, after + (before - after) * guard, after)
    lab[..., 0] = np.clip(lightness + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)


def clarity_amount(megapixels: float) -> float:
    span = CLARITY_FULL_MP - CLARITY_SMALL_MP
    t = min(1.0, max(0.0, (megapixels - CLARITY_SMALL_MP) / span))
    return CLARITY_AMOUNT * (CLARITY_SMALL_FACTOR + (1 - CLARITY_SMALL_FACTOR) * t)


def clarity(arr: np.ndarray, amount: float | None = None) -> np.ndarray:
    """Crisp detail in faces, jerseys and the ball: OpenCV detailEnhance (an
    edge-preserving filter, so no halos around players), blended in at
    CLARITY_AMOUNT. Like every step, it may not darken shadows."""
    src = np.clip(arr + 0.5, 0, 255).astype(np.uint8)
    detail = cv2.detailEnhance(np.ascontiguousarray(src[..., ::-1]),
                               sigma_s=CLARITY_SIGMA_S, sigma_r=CLARITY_SIGMA_R)
    detail = detail[..., ::-1].astype(np.float32)
    amount = clarity_amount(arr.shape[0] * arr.shape[1] / 1e6) if amount is None else amount
    out = arr + (detail - arr) * amount
    lum_in, lum_out = _luma(arr), _luma(out)
    guard = np.clip(1 - lum_in / CLARITY_SHADOW_END, 0, 1) ** 0.5
    weight = np.where(lum_out < lum_in, guard, 0.0)[..., None]
    return out + (arr - out) * weight


def subject_mask(shape: tuple[int, int], box: Box) -> np.ndarray:
    """1 inside the subject box, fading smoothly to 0 outside it."""
    h, w = shape
    left, top, right, bottom = box
    left, right = max(0, left), min(w, right)
    top, bottom = max(0, top), min(h, bottom)
    mask = np.zeros((h, w), dtype=np.float32)
    if right - left < 8 or bottom - top < 8:
        return mask
    mask[top:bottom, left:right] = 1
    sigma = max(2.0, SUBJECT_FEATHER * min(right - left, bottom - top))
    return cv2.GaussianBlur(mask, (0, 0), sigma)


def lift_subject(arr: np.ndarray, box: Box | None) -> np.ndarray:
    """Brightens the darker tones of the players only (dark skin, navy kits)
    when they are darker than SUBJECT_TARGET, through a feathered mask, so the
    rest of the photo keeps its exposure."""
    if box is None:
        return arr
    region = _region(arr, box)
    if region is arr:
        return arr
    median = float(np.median(_luma(np.clip(region, 0, 255))))
    if median >= SUBJECT_TARGET:
        return arr
    amount = SUBJECT_MAX_LIFT * min(1.0, (SUBJECT_TARGET - median) / SUBJECT_TARGET * 2)
    lum = np.clip(_luma(arr), 0, 255)
    t = np.clip(1 - lum / SUBJECT_LIFT_END, 0, 1)
    lifted = lum + amount * t ** 1.5
    ratio = (lifted + 1) / (lum + 1)
    mask = subject_mask(arr.shape[:2], box)[..., None]
    return arr * (1 + (ratio[..., None] - 1) * mask)


def enhance(img: Image.Image, subject: Box | None = None) -> Image.Image:
    """White balance, levels, exposure for the whole scene, local contrast, a
    gentle lift on the players if they are dark (subject box in this image's
    pixel coordinates), then clarity."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr = auto_white_balance(arr)
    arr = auto_levels(arr)
    arr = auto_exposure(arr)
    arr = local_contrast(arr)
    arr = lift_subject(arr, subject)
    arr = clarity(arr)
    return Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8))


def punch(img: Image.Image) -> Image.Image:
    """The punchier natural edit (C): vibrance (boosts muted colors more than
    already-strong ones, so skin and kits do not go neon) and extra contrast
    in the upper tones only, so shadows and darker skin are not darkened."""
    arr = np.asarray(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1]
    hsv[..., 1] = np.clip(sat + PUNCH_VIBRANCE * sat * (1 - sat / 255), 0, 255)
    out = cv2.cvtColor(hsv.round().astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    # More saturation lowers the brightness of blues (navy kits): restore each
    # pixel's original brightness so vibrance changes color only
    before = _luma(arr.astype(np.float32))
    out = out * ((before + 1) / (_luma(out) + 1))[..., None]
    lum = _luma(out)
    lifted = np.where(lum > 128, lum + PUNCH_CONTRAST * (lum - 128), lum)
    out = out * ((np.clip(lifted, 0, 255) + 1) / (lum + 1))[..., None]
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def soften_contrast(img: Image.Image, amount: float = SOFT_CONTRAST) -> Image.Image:
    """Reduces contrast by `amount` (0.1 = 10%) around mid-gray: blacks lift
    and highlights come down a touch, color is kept. Used for the softer B
    version so A and B are clearly different."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    lum = _luma(arr)
    target = 128 + (lum - 128) * (1 - amount)
    out = arr * ((np.clip(target, 0, 255) + 1) / (lum + 1))[..., None]
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def output_sharpen(img: Image.Image) -> Image.Image:
    """Light sharpening after resizing to the final size."""
    return img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
