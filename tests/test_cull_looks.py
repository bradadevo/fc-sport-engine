import numpy as np
import pytest
from PIL import Image

from match_cull import looks


def write_cube(path, size, fn, header=""):
    lines = [f"LUT_3D_SIZE {size}", header]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rgb = np.array([r, g, b]) / (size - 1)
                lines.append(" ".join(f"{v:.6f}" for v in fn(rgb)))
    path.write_text("\n".join(lines))
    return path


@pytest.fixture
def photo():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8))


def test_identity_lut_leaves_the_photo_unchanged(tmp_path, photo):
    lut = looks.parse_cube(write_cube(tmp_path / "id.cube", 17, lambda c: c))
    out = np.asarray(looks.apply_lut(photo, lut), dtype=int)
    assert np.abs(out - np.asarray(photo, dtype=int)).max() <= 1


def test_channels_are_not_swapped(tmp_path):
    # A LUT that zeroes blue must leave red and green alone
    lut = looks.parse_cube(write_cube(tmp_path / "noblue.cube", 9, lambda c: [c[0], c[1], 0]))
    img = Image.new("RGB", (2, 2), (200, 100, 50))
    assert looks.apply_lut(img, lut).getpixel((0, 0)) == (200, 100, 0)


def test_interpolates_between_lattice_points(tmp_path):
    lut = looks.parse_cube(write_cube(tmp_path / "inv.cube", 2, lambda c: 1 - c))
    img = Image.new("RGB", (1, 1), (64, 128, 192))
    assert looks.apply_lut(img, lut).getpixel((0, 0)) == (191, 127, 63)


def test_big_images_are_processed_in_slices(tmp_path, monkeypatch):
    monkeypatch.setattr(looks, "_CHUNK_PIXELS", 7)
    lut = looks.parse_cube(write_cube(tmp_path / "inv.cube", 5, lambda c: 1 - c))
    img = Image.new("RGB", (10, 3), (10, 20, 30))
    pixels = np.asarray(looks.apply_lut(img, lut)).reshape(-1, 3)
    assert {tuple(int(v) for v in p) for p in pixels} == {(245, 235, 225)}


def test_rejects_broken_files(tmp_path):
    bad = tmp_path / "bad.cube"
    bad.write_text("LUT_3D_SIZE 4\n0 0 0\n")
    with pytest.raises(ValueError):
        looks.parse_cube(bad)


def test_discover_finds_nested_luts_once_per_name(tmp_path):
    write_cube(tmp_path / "MVP.cube", 2, lambda c: c)
    (tmp_path / "Pack").mkdir()
    write_cube(tmp_path / "Pack" / "MVP.cube", 2, lambda c: c)
    write_cube(tmp_path / "Pack" / "Team Pride.cube", 2, lambda c: c)
    found = looks.discover_luts(tmp_path)
    assert sorted(found) == ["MVP", "Team Pride"]


def test_describe_tells_warm_from_cool(tmp_path):
    warm = looks.parse_cube(write_cube(tmp_path / "w.cube", 9, lambda c: np.clip(c * [1.15, 1, 0.85], 0, 1)))
    cool = looks.parse_cube(write_cube(tmp_path / "c.cube", 9, lambda c: np.clip(c * [0.85, 1, 1.15], 0, 1)))
    assert looks.describe_lut(warm).startswith("warm")
    assert looks.describe_lut(cool).startswith("cool")


def test_pillow_engine_matches_reference_interpolation(tmp_path, photo):
    fn = lambda c: np.clip([c[0] ** 0.8, c[1] * 0.9 + 0.05, c[2] ** 1.3], 0, 1)
    lut = looks.parse_cube(write_cube(tmp_path / "curve.cube", 17, fn))
    fast = np.asarray(looks.apply_lut(photo, lut), dtype=int)
    reference = np.asarray(looks._apply_lut_any_domain(photo, lut), dtype=int)
    assert np.abs(fast - reference).max() <= 2


def test_look_cannot_crush_shadows(tmp_path):
    crusher = looks.parse_cube(write_cube(tmp_path / "crush.cube", 17, lambda c: c ** 2.2))
    dark = Image.fromarray(np.full((10, 10, 3), (60, 42, 35), dtype=np.uint8))  # dark skin tone
    bright = Image.fromarray(np.full((10, 10, 3), (220, 220, 220), dtype=np.uint8))
    lum = lambda im: float(np.asarray(im, dtype=np.float32).mean())
    assert lum(looks.apply_lut(dark, crusher)) < lum(dark) * 0.5  # the raw LUT really does crush
    assert lum(looks.apply_look(dark, crusher)) >= lum(dark) * 0.75  # the guarded look does not
    assert lum(looks.apply_look(bright, crusher)) < lum(bright) * 0.85  # bright tones get the full look


def test_look_brightening_shadows_is_left_alone(tmp_path):
    lifter = looks.parse_cube(write_cube(tmp_path / "lift.cube", 9, lambda c: c ** 0.6))
    dark = Image.fromarray(np.full((4, 4, 3), 40, dtype=np.uint8))
    assert np.array_equal(np.asarray(looks.apply_look(dark, lifter)), np.asarray(looks.apply_lut(dark, lifter)))


def test_look_that_turns_grass_yellow_is_toned_down(tmp_path):
    yellowing = looks.parse_cube(write_cube(tmp_path / "yellow.cube", 17,
                                            lambda c: np.clip([c[0] * 1.35 + 0.05, c[1], c[2] * 0.6], 0, 1)))
    shift, _ = looks.grass_effect(yellowing)
    assert shift < -looks.MAX_GRASS_HUE_SHIFT
    yellowing.strength = looks.auto_strength(yellowing)
    assert looks.MIN_STRENGTH <= yellowing.strength < 1
    assert "turns grass yellow" in looks.describe_lut(yellowing)

    grass = Image.fromarray(np.array(looks.GRASS_SAMPLES, dtype=np.uint8).reshape(1, -1, 3))
    h0, _ = looks._hue_chroma(np.asarray(grass))
    h1, _ = looks._hue_chroma(np.asarray(looks.apply_look(grass, yellowing)))
    # within the limit, unless the look is so strong it is already at the minimum strength
    assert abs((h1 - h0).mean()) <= max(looks.MAX_GRASS_HUE_SHIFT, abs(shift) * looks.MIN_STRENGTH) + 1.5


def test_gentle_look_keeps_full_strength(tmp_path):
    gentle = looks.parse_cube(write_cube(tmp_path / "gentle.cube", 9, lambda c: np.clip(c * 1.03, 0, 1)))
    assert looks.auto_strength(gentle) == 1.0
