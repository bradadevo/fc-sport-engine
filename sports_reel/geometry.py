from PIL import Image

from sports_reel.vision import SoccerSubjectDetection

TARGET_ASPECT = 9.0 / 16.0
SAFE_ZONE_TOP = 0.20
SAFE_ZONE_BOTTOM = 0.70
ZOOM_SCALE = 0.85

# EXIF orientation values that rotate the image a quarter turn
_QUARTER_TURN_ORIENTATIONS = (5, 6, 7, 8)

Box = tuple[float, float, float, float]


def upright_size(image_path: str) -> tuple[int, int]:
    """Returns (width, height) as displayed, after EXIF auto-rotation."""
    with Image.open(image_path) as img:
        w, h = img.size
        if img.getexif().get(0x0112) in _QUARTER_TURN_ORIENTATIONS:
            w, h = h, w
    return w, h


def compute_pan_zoom_offsets(img_w: int, img_h: int, detection: SoccerSubjectDetection) -> tuple[Box, Box]:
    """
    Returns (start_box, end_box) for the Ken Burns move. Each box is
    (left, top, right, bottom) insets as a percentage of the image height,
    which is the unit FCPXML's pan-rect uses for all four edges.
    """
    center_x = min(1.0, max(0.0, (detection.xmin + detection.xmax) / 2.0)) * img_w
    center_y = min(1.0, max(0.0, (detection.ymin + detection.ymax) / 2.0)) * img_h

    # Safe-zone clamp for Instagram header / caption overlay areas
    center_y = max(img_h * SAFE_ZONE_TOP, min(center_y, img_h * SAFE_ZONE_BOTTOM))

    # Maximal 9:16 rectangle that fits inside the photo
    crop_h_base = img_h
    crop_w_base = crop_h_base * TARGET_ASPECT

    if crop_w_base > img_w:
        crop_w_base = img_w
        crop_h_base = crop_w_base / TARGET_ASPECT

    start_scale = 1.00 if detection.motion_type == "push_in" else ZOOM_SCALE
    end_scale = ZOOM_SCALE if detection.motion_type == "push_in" else 1.00

    def get_box(scale):
        w = crop_w_base * scale
        h = crop_h_base * scale
        left = max(0.0, min(center_x - (w / 2.0), img_w - w))
        top = max(0.0, min(center_y - (h / 2.0), img_h - h))
        right = max(0.0, img_w - (left + w))
        bottom = max(0.0, img_h - (top + h))
        return tuple(round(px / img_h * 100.0, 4) for px in (left, top, right, bottom))

    return get_box(start_scale), get_box(end_scale)
