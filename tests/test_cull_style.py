import numpy as np
from PIL import Image

from match_cull import style


def reference_style(**overrides):
    base = dict(source="Gold", count=10, lightness=72.0, contrast=78.0, chroma=16.7,
                warmth=11.8, tint=-3.8, highlight_clip=0.1)
    base.update(overrides)
    return style.Style(**base)


def scene(seed=0, saturation=1.0, warm=0):
    rng = np.random.default_rng(seed)
    arr = np.zeros((120, 160, 3), dtype=np.float32)
    arr[:] = (90, 160, 80)  # grass
    arr[:40] = (190, 205, 230)  # sky
    arr[50:110, 60:100] = (60, 45, 40)  # dark-skinned player
    arr += rng.normal(0, 4, arr.shape)
    gray = arr.mean(axis=2, keepdims=True)
    arr = gray + (arr - gray) * saturation
    arr[..., 0] += warm
    arr[..., 2] -= warm
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def test_oversaturated_edit_is_pulled_toward_the_reference():
    loud = scene(saturation=1.8)
    out = style.apply_style(loud, reference_style())
    assert style.measure(out)["chroma"] < style.measure(loud)["chroma"]


def test_cool_edit_is_warmed_toward_the_reference():
    cool = scene(warm=-15)
    out = style.apply_style(cool, reference_style())
    assert style.measure(out)["warmth"] > style.measure(cool)["warmth"] + 1


def test_changes_are_capped():
    img = scene()
    out = style.apply_style(img, reference_style(warmth=60, chroma=80, lightness=100))
    before, after = style.measure(img), style.measure(out)
    # capped shift, plus the existing warmth growing with the capped color scaling
    assert after["warmth"] - before["warmth"] <= style.MAX_WARMTH_SHIFT + abs(before["warmth"]) * (style.MAX_CHROMA_SCALE[1] - 1) + 1
    # the warmth and tint shifts add a little color on top of the capped scaling
    assert after["chroma"] <= before["chroma"] * style.MAX_CHROMA_SCALE[1] + style.MAX_WARMTH_SHIFT + style.MAX_TINT_SHIFT


def test_style_never_crushes_the_dark_player():
    img = scene()
    out = style.apply_style(img, reference_style(lightness=40))  # reference darker than the edit
    lum = lambda im: np.asarray(im, dtype=np.float32)[50:110, 60:100].mean()
    assert lum(out) >= lum(img) - 2


def test_blown_highlights_are_rolled_off():
    arr = np.full((100, 100, 3), 120, dtype=np.uint8)
    arr[:30] = 255
    img = Image.fromarray(arr)
    out = style.apply_style(img, reference_style())
    assert style.measure(out)["highlight_clip"] < style.measure(img)["highlight_clip"]


def test_learn_measures_color_photos_and_skips_black_and_white(tmp_path):
    scene(saturation=1.0).save(tmp_path / "a.jpg")
    scene(seed=1, saturation=1.2).save(tmp_path / "b.png")
    Image.new("L", (100, 100), 128).convert("RGB").save(tmp_path / "bw.jpg")
    learned = style.learn(str(tmp_path))
    assert learned.count == 3
    assert learned.chroma > style.BW_CHROMA
    style.save(learned, tmp_path / "style.json")
    assert style.load(tmp_path / "style.json") == learned


def test_letterbox_bars_are_ignored():
    img = scene()
    arr = np.asarray(img)
    boxed = np.zeros((arr.shape[0], arr.shape[1] + 80, 3), dtype=np.uint8)
    boxed[:, 40:-40] = arr
    a, b = style.measure(img), style.measure(Image.fromarray(boxed))
    assert abs(a["lightness"] - b["lightness"]) < 1


def test_best_photos_outweigh_the_rest(tmp_path):
    for i in range(6):
        Image.new("RGB", (200, 150), (210, 200, 170)).save(tmp_path / f"bright{i}.jpg")
    Image.new("RGB", (200, 150), (90, 80, 60)).save(tmp_path / "best.jpg")
    plain = style.learn(str(tmp_path))
    weighted = style.learn(str(tmp_path), featured=["best.jpg"])
    assert weighted.lightness < plain.lightness - 10
    assert weighted.featured == ["best.jpg"]


def test_unknown_best_photo_is_an_error(tmp_path):
    scene().save(tmp_path / "a.jpg")
    import pytest
    with pytest.raises(ValueError, match="nope.jpg"):
        style.learn(str(tmp_path), featured=["nope.jpg"])
