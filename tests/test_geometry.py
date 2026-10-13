import pytest
from PIL import Image

from sports_reel.geometry import ZOOM_SCALE, compute_pan_zoom_offsets, upright_size
from sports_reel.vision import SoccerSubjectDetection, center_detection


def detection(xmin, ymin, xmax, ymax, motion_type="push_in"):
    return SoccerSubjectDetection(subject_description="test", xmin=xmin, ymin=ymin,
                                  xmax=xmax, ymax=ymax, motion_type=motion_type)


def to_pixels(box, img_h):
    """Converts a percent-of-height inset box back to pixel insets."""
    return tuple(v * img_h / 100.0 for v in box)


def crop_rect(box, img_w, img_h):
    left, top, right, bottom = to_pixels(box, img_h)
    return left, top, img_w - left - right, img_h - top - bottom  # x, y, w, h


SIZES = [(6000, 4000), (4000, 6000), (1080, 1920), (1920, 1080), (3000, 3000)]
SUBJECTS = [(0.4, 0.4, 0.6, 0.6), (0.0, 0.0, 0.1, 0.1), (0.9, 0.9, 1.0, 1.0), (0.8, 0.1, 1.3, 0.2)]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("subject", SUBJECTS)
@pytest.mark.parametrize("motion", ["push_in", "pull_out"])
def test_boxes_are_9x16_and_inside_the_image(size, subject, motion):
    img_w, img_h = size
    for box in compute_pan_zoom_offsets(img_w, img_h, detection(*subject, motion_type=motion)):
        assert all(v >= 0 for v in box)
        x, y, w, h = crop_rect(box, img_w, img_h)
        assert w > 0 and h > 0
        assert x + w <= img_w + 0.5 and y + h <= img_h + 0.5
        assert w / h == pytest.approx(9 / 16, rel=1e-3)


def test_push_in_zooms_from_full_crop_to_tighter_crop():
    start, end = compute_pan_zoom_offsets(6000, 4000, detection(0.4, 0.4, 0.6, 0.6, "push_in"))
    _, _, _, start_h = crop_rect(start, 6000, 4000)
    _, _, _, end_h = crop_rect(end, 6000, 4000)
    assert start_h == pytest.approx(4000, abs=1)
    assert end_h == pytest.approx(4000 * ZOOM_SCALE, abs=1)


def test_pull_out_is_the_reverse_of_push_in():
    subject = (0.3, 0.3, 0.5, 0.6)
    push = compute_pan_zoom_offsets(6000, 4000, detection(*subject, "push_in"))
    pull = compute_pan_zoom_offsets(6000, 4000, detection(*subject, "pull_out"))
    assert pull == (push[1], push[0])


def test_crop_is_centered_on_the_subject_horizontally():
    _, end = compute_pan_zoom_offsets(6000, 4000, detection(0.2, 0.4, 0.4, 0.6))
    x, _, w, _ = crop_rect(end, 6000, 4000)
    assert x + w / 2 == pytest.approx(0.3 * 6000, abs=1)


def test_safe_zone_clamps_vertical_center():
    # Portrait source wider than 9:16 crops, so the tight box can move vertically
    img_w, img_h = 4000, 6000
    _, low = compute_pan_zoom_offsets(img_w, img_h, detection(0.4, 0.9, 0.6, 1.0))
    _, high = compute_pan_zoom_offsets(img_w, img_h, detection(0.4, 0.0, 0.6, 0.05))
    _, y_low, _, h = crop_rect(low, img_w, img_h)
    _, y_high, _, _ = crop_rect(high, img_w, img_h)
    # A subject at the very bottom is treated as sitting at 70% of the height
    assert y_low + h / 2 == pytest.approx(min(0.70 * img_h, img_h - h / 2), abs=1)
    # A subject at the very top is treated as sitting at 20% of the height
    assert y_high + h / 2 == pytest.approx(max(0.20 * img_h, h / 2), abs=1)


def test_center_fallback_gives_a_centered_crop():
    start, _ = compute_pan_zoom_offsets(6000, 4000, center_detection())
    x, _, w, _ = crop_rect(start, 6000, 4000)
    assert x + w / 2 == pytest.approx(3000, abs=1)


def test_upright_size_applies_exif_rotation(tmp_path):
    plain = tmp_path / "plain.jpg"
    rotated = tmp_path / "rotated.jpg"
    img = Image.new("RGB", (300, 200))
    img.save(plain)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated 90 degrees: displays as portrait
    img.save(rotated, exif=exif)

    assert upright_size(str(plain)) == (300, 200)
    assert upright_size(str(rotated)) == (200, 300)
