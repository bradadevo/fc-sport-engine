import numpy as np
import pytest
from PIL import Image

from match_cull import adjust


def adj(**values):
    return adjust.no_changes("test").model_copy(update=values)


def luma(img):
    a = np.asarray(img, dtype=np.float32)
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


@pytest.fixture
def photo():
    rng = np.random.default_rng(2)
    arr = np.zeros((120, 160, 3), dtype=np.float32)
    arr[:] = (110, 160, 90)  # grass
    arr[:30] = (200, 210, 230)  # sky
    arr[40:110, 60:100] = (60, 45, 40)  # dark-skinned player
    arr[40:60, 60:100] = (235, 235, 235)  # white jersey
    arr += rng.normal(0, 5, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def test_no_changes_is_a_no_op(photo):
    out = adjust.apply(photo, adjust.no_changes("none"))
    assert np.abs(np.asarray(out, dtype=int) - np.asarray(photo, dtype=int)).max() <= 1


def test_values_are_clamped():
    wild = adj(exposure=5, contrast=-200, shadows=-90, subject_brightness=500)
    c = wild.clamped()
    assert (c.exposure, c.contrast, c.shadows, c.subject_brightness) == (1.0, -30, -15, 40)


@pytest.mark.parametrize("slider,value,region,direction", [
    ("exposure", 0.8, (slice(None), slice(None)), +1),
    ("highlights", -50, (slice(40, 60), slice(60, 100)), -1),
    ("shadows", 40, (slice(60, 110), slice(60, 100)), +1),
])
def test_sliders_move_the_right_tones(photo, slider, value, region, direction):
    before = luma(photo)[region].mean()
    after = luma(adjust.apply(photo, adj(**{slider: value})))[region].mean()
    assert (after - before) * direction > 2


def test_warmth_and_vibrance(photo):
    warm = np.asarray(adjust.apply(photo, adj(warmth=30)), dtype=np.float32).mean(axis=(0, 1))
    base = np.asarray(photo, dtype=np.float32).mean(axis=(0, 1))
    assert warm[0] - warm[2] > base[0] - base[2]
    sat = lambda im: np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1].mean()
    assert sat(adjust.apply(photo, adj(vibrance=30))) > sat(photo)


def test_contrast_cannot_crush_the_dark_player(photo):
    player = (slice(60, 110), slice(60, 100))
    before = np.percentile(luma(photo)[player], 10)
    after = np.percentile(luma(adjust.apply(photo, adj(contrast=30)))[player], 10)
    assert after >= before - 3


def test_subject_brightness_only_touches_the_players(photo):
    box = (60, 60, 100, 110)
    out = luma(adjust.apply(photo, adj(subject_brightness=40), subject_box=box))
    base = luma(photo)
    assert out[70:100, 70:90].mean() > base[70:100, 70:90].mean() + 8  # player lifted
    assert abs(out[:, :20].mean() - base[:, :20].mean()) < 1  # far edge untouched


def test_summary_lists_only_moved_sliders():
    assert adj(exposure=0.3, highlights=-25).summary() == "exposure +0.3, highlights -25"
    assert adjust.no_changes("x").summary() == "no changes"
