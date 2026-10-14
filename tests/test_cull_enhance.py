import numpy as np
import pytest
from PIL import Image

from match_cull import enhance


def luma(img):
    a = np.asarray(img, dtype=np.float32)
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def gradient(lo, hi, tint=(0, 0, 0)):
    ramp = np.linspace(lo, hi, 200, dtype=np.float32)
    arr = np.stack([np.tile(ramp, (100, 1)) + t for t in tint], axis=-1)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def test_underexposed_photo_gets_brighter():
    dark = gradient(5, 90)
    assert np.median(luma(enhance.enhance(dark))) > np.median(luma(dark)) + 30


def test_overexposed_photo_gets_darker():
    bright = gradient(170, 252)
    # Pulled down through the highlights only, so less than a plain gamma would
    assert np.median(luma(enhance.enhance(bright))) < np.median(luma(bright)) - 10


def test_flat_photo_gets_more_contrast():
    flat = gradient(100, 160)
    out = luma(enhance.enhance(flat))
    # Only the bright end is stretched (shadows stay put), so 60 -> 75+
    assert out.max() - out.min() > 75


def test_color_cast_is_reduced():
    blue = gradient(40, 220, tint=(-20, 0, 25))
    before = np.asarray(blue, dtype=np.float32).mean(axis=(0, 1))
    after = np.asarray(enhance.enhance(blue), dtype=np.float32).mean(axis=(0, 1))
    # capped gains: a clear but gentle correction
    assert (after[2] - after[0]) < (before[2] - before[0]) * 0.85


def test_well_exposed_photo_changes_little():
    good = gradient(6, 248)
    diff = np.abs(luma(enhance.enhance(good)) - luma(good))
    assert np.median(diff) < 12


def test_solid_color_does_not_crash():
    out = enhance.enhance(Image.new("RGB", (50, 50), (30, 120, 40)))
    assert out.size == (50, 50)


def sunny_field_with_dark_player():
    """Bright grass and sky everywhere, with a dark-skinned player in a navy
    kit in the middle: the case that used to get crushed to black."""
    arr = np.zeros((300, 300, 3), dtype=np.float32)
    arr[:] = (150, 205, 120)  # sunlit grass
    arr[:80] = (215, 225, 240)  # bright sky
    rng = np.random.default_rng(3)
    player = rng.normal(0, 6, (120, 60, 3)) + (70, 48, 38)  # dark skin tones
    player[50:] = rng.normal(0, 6, (70, 60, 3)) + (22, 28, 52)  # navy kit
    arr[120:240, 120:180] = player
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), (120, 120, 180, 240)


def test_dark_skin_and_navy_kit_are_not_crushed_on_a_sunny_field():
    img, box = sunny_field_with_dark_player()
    before = luma(img.crop(box))
    after = luma(enhance.enhance(img, subject=box).crop(box))
    assert (after < 20).mean() <= (before < 20).mean() + 0.01
    assert np.percentile(after, 5) >= np.percentile(before, 5) - 2
    assert np.median(after) >= np.median(before) - 3


def test_no_step_makes_the_deepest_shadows_darker():
    img, box = sunny_field_with_dark_player()
    arr = np.asarray(img, dtype=np.float32)
    lo_before = np.percentile(luma(img), 0.5)
    for step in (enhance.auto_white_balance, enhance.auto_levels):
        out = step(arr)
        assert np.percentile(luma(np.clip(out, 0, 255)), 0.5) >= lo_before - 3


def test_exposure_is_judged_on_the_subject():
    img, box = sunny_field_with_dark_player()
    whole = luma(enhance.enhance(img).crop(box))
    on_subject = luma(enhance.enhance(img, subject=box).crop(box))
    assert np.median(on_subject) > np.median(whole)


def test_clarity_adds_detail_without_crushing_shadows():
    rng = np.random.default_rng(5)
    base = np.full((120, 120, 3), 140, dtype=np.float32)
    base += rng.normal(0, 6, base.shape)  # fine texture (jersey, grass)
    base[40:80, 40:80] = (60, 45, 40) + rng.normal(0, 4, (40, 40, 3))  # dark skin patch
    out = enhance.clarity(base)
    texture = lambda a: np.abs(np.diff(luma(np.clip(a, 0, 255)), axis=1))[:30, :30].mean()
    assert texture(out) > texture(base) * 1.1
    dark = lambda a: luma(np.clip(a, 0, 255))[45:75, 45:75].mean()
    assert dark(out) >= dark(base) - 1


def test_grass_filled_frame_does_not_turn_green():
    rng = np.random.default_rng(7)
    arr = np.zeros((200, 300, 3), dtype=np.float32)
    arr[:] = (95, 150, 70)  # grass everywhere
    arr[20:60, 130:170] = (220, 120, 40)  # small orange keeper
    arr += rng.normal(0, 5, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    before = np.asarray(img, dtype=np.float32).mean(axis=(0, 1))
    after = enhance.auto_white_balance(np.asarray(img, dtype=np.float32)).mean(axis=(0, 1))
    ratio = after / before
    assert np.all(np.abs(ratio / ratio.mean() - 1) <= enhance.WB_MAX_GAIN * enhance.WB_STRENGTH + 0.01)


def test_one_dark_player_does_not_brighten_the_whole_photo():
    img, box = sunny_field_with_dark_player()
    arr = luma(img)
    outside = np.ones(arr.shape, dtype=bool)
    outside[box[1] - 40:box[3] + 40, box[0] - 40:box[2] + 40] = False  # beyond the feathered edge
    out = luma(enhance.enhance(img, subject=box))
    assert abs(np.median(out[outside]) - np.median(arr[outside])) < 12  # background exposure kept
    player = lambda a: a[box[1]:box[3], box[0]:box[2]]
    assert np.percentile(player(out), 10) > np.percentile(player(arr), 10) + 5  # player lifted locally


def test_clarity_is_gentler_on_small_crops():
    assert enhance.clarity_amount(12) == enhance.CLARITY_AMOUNT
    assert enhance.clarity_amount(0.3) == enhance.CLARITY_AMOUNT * enhance.CLARITY_SMALL_FACTOR
    assert enhance.clarity_amount(0.3) < enhance.clarity_amount(2.5) < enhance.clarity_amount(5)


def test_punch_adds_color_and_never_darkens_shadows():
    img, box = sunny_field_with_dark_player()
    out = enhance.punch(img)
    sat = lambda im: np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1].mean()
    assert sat(out) > sat(img)
    dark = lambda im: luma(im)[box[1]:box[3], box[0]:box[2]]
    assert np.percentile(dark(out), 10) >= np.percentile(dark(img), 10) - 1


def test_soften_contrast_reduces_contrast_by_ten_percent():
    img = gradient(20, 235)
    before, after = luma(img), luma(enhance.soften_contrast(img))
    spread = lambda l: np.percentile(l, 95) - np.percentile(l, 5)
    assert spread(after) == pytest.approx(spread(before) * 0.9, rel=0.03)
    assert np.percentile(after, 5) > np.percentile(before, 5)  # blacks lift, never crush
