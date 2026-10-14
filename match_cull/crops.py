from dataclasses import dataclass

# Crops never go below this many pixels on their short side (or the photo's
# own short side, if it is smaller), so small sources are not zoomed into mush
MIN_CROP_SHORT_SIDE = 1080
# How much room to leave around the subject in the tight crop
SUBJECT_MARGIN = 1.6
# The tight crop never zooms in past this fraction of the widest crop,
# so it keeps enough resolution
MIN_TIGHT_FRACTION = 0.55


@dataclass(frozen=True)
class OutputSize:
    key: str
    label: str
    aspect: float | None  # width / height; None keeps each photo's own shape
    pixels: tuple[int, int] | None  # final size; None keeps the crop's full resolution
    match_source: bool = False  # final size = the original photo's pixel size
    ai_framing: bool = False  # the model chooses shape and framing per photo


SIZES = {
    "instagram": OutputSize("instagram", "Instagram post (4:5)", 4 / 5, (1080, 1350)),
    "story": OutputSize("story", "Instagram Reel / Story (9:16)", 9 / 16, (1080, 1920)),
    "youtube": OutputSize("youtube", "YouTube / full size (16:9)", 16 / 9, None),
    "original": OutputSize("original", "Original size (cropped and zoomed, same dimensions)", None, None,
                           match_source=True),
    "ai": OutputSize("ai", "Best crop (AI decides shape and framing)", None, None, ai_framing=True),
}

# Shapes the model may choose for an AI crop
AI_ASPECTS = {"4:5": 4 / 5, "1:1": 1.0, "3:2": 3 / 2, "2:3": 2 / 3, "16:9": 16 / 9, "9:16": 9 / 16}
# An AI crop never zooms in past this fraction of the widest crop of its shape
MIN_AI_FRACTION = 0.4

Box = tuple[int, int, int, int]  # left, top, right, bottom in pixels


def _min_height(max_h: float, aspect: float) -> float:
    """Smallest crop height allowed for this shape: short side at least
    MIN_CROP_SHORT_SIDE, but never more than the largest crop that fits."""
    short_from_h = min(1.0, aspect)  # short side = height * min(1, aspect)
    return min(max_h, MIN_CROP_SHORT_SIDE / short_from_h)


def crop_box(img_w: int, img_h: int, subject: tuple[float, float, float, float],
             aspect: float | None, tight: bool) -> Box:
    """Crop rectangle of the given aspect centered on the subject.

    subject is (xmin, ymin, xmax, ymax) normalized 0-1. The wide crop is the
    largest rectangle of that shape that fits the photo; the tight crop
    closes in so the subject fills more of the frame.
    """
    aspect = aspect or img_w / img_h
    xmin, ymin, xmax, ymax = (min(1.0, max(0.0, v)) for v in subject)
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    cx = (xmin + xmax) / 2 * img_w
    cy = (ymin + ymax) / 2 * img_h

    max_h = min(img_h, img_w / aspect)
    h = max_h
    if tight:
        sub_w = (xmax - xmin) * img_w
        sub_h = (ymax - ymin) * img_h
        needed = max(sub_h, sub_w / aspect) * SUBJECT_MARGIN
        h = min(max_h, max(needed, max_h * MIN_TIGHT_FRACTION, _min_height(max_h, aspect)))
    w = h * aspect

    left = min(max(cx - w / 2, 0), img_w - w)
    top = min(max(cy - h / 2, 0), img_h - h)
    return round(left), round(top), round(left + w), round(top + h)


def ai_crop_box(img_w: int, img_h: int, suggested: tuple[float, float, float, float],
                shape: str) -> Box:
    """Turns the model's suggested crop (xmin, ymin, xmax, ymax normalized)
    into a clean rectangle of the chosen shape: the smallest one of that
    shape containing the suggestion, centered on it, kept inside the photo
    and never zoomed in so far that it loses too much resolution."""
    aspect = AI_ASPECTS.get(shape.strip(), img_w / img_h)
    xmin, ymin, xmax, ymax = (min(1.0, max(0.0, v)) for v in suggested)
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    cx = (xmin + xmax) / 2 * img_w
    cy = (ymin + ymax) / 2 * img_h

    max_h = min(img_h, img_w / aspect)
    wanted = max((ymax - ymin) * img_h, (xmax - xmin) * img_w / aspect)
    h = min(max_h, max(wanted, max_h * MIN_AI_FRACTION, _min_height(max_h, aspect)))
    w = h * aspect

    left = min(max(cx - w / 2, 0), img_w - w)
    top = min(max(cy - h / 2, 0), img_h - h)
    return round(left), round(top), round(left + w), round(top + h)
