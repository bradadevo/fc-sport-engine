import csv
import os

import numpy as np
import pytest
from PIL import Image

from match_cull import analysis, cli
from tests.test_cull_looks import write_cube


@pytest.fixture
def match(tmp_path, monkeypatch):
    folder = tmp_path / "Match"
    folder.mkdir()
    rng = np.random.default_rng(1)
    for i in range(6):
        arr = rng.integers(0, 256, (400, 600, 3), dtype=np.uint8)
        Image.fromarray(arr).save(folder / f"DSC{i:04d}.JPG")
    (folder / "notes.txt").write_text("not a photo")
    luts = tmp_path / "LUTs"
    luts.mkdir()
    write_cube(luts / "MVP.cube", 5, lambda c: c)
    write_cube(luts / "Team Pride.cube", 5, lambda c: np.clip(c * [1.1, 1, 0.9], 0, 1))

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cli.vision, "create_client", lambda: object())
    # Never read or write the real learned style during tests
    monkeypatch.setattr(cli.style, "STYLE_PATH", tmp_path / "style.json")
    return folder, luts


class FakeGemini:
    def __init__(self, monkeypatch, scores, failing=(), shape="1:1"):
        self.scores, self.failing, self.calls, self.shape = scores, set(failing), 0, shape
        self.order = iter(range(10_000))
        monkeypatch.setattr(cli.analysis, "analyze", self)

    def __call__(self, client, img, look_descriptions, model, usage=None, taste_notes=""):
        self.taste_notes = taste_notes
        self.calls += 1
        i = next(self.order)
        if i in self.failing:
            raise RuntimeError("503 high demand")
        return analysis.PhotoAssessment(
            score=self.scores[i % len(self.scores)], reason="test", subject_description="striker",
            ymin=0.3, xmin=0.4, ymax=0.6, xmax=0.6, crop_xmin=0.3, crop_ymin=0.2, crop_xmax=0.7, crop_ymax=0.8, crop_shape=self.shape, best_look="team pride", second_look="Made Up")


def run(folder, luts, *extra):
    cli.main([str(folder), "--luts", str(luts), *extra])


def test_keeps_top_photos_with_two_versions_and_leaves_originals_alone(match, monkeypatch):
    folder, luts = match
    before = {p: os.stat(folder / p).st_mtime_ns for p in os.listdir(folder)}
    FakeGemini(monkeypatch, scores=[5, 9, 3, 8, 7, 6])
    run(folder, luts, "--size", "instagram", "--keep", "3")

    out = folder / "Culled - Instagram post"
    made = sorted(p for p in os.listdir(out) if p.endswith(".jpg"))
    assert made == [f"DSC{n}_{v}.jpg" for n in ("0001", "0003", "0004") for v in "ABCO"]
    with Image.open(out / "DSC0001_A.jpg") as img:
        assert img.size == (1080, 1350)
    assert {p: os.stat(folder / p).st_mtime_ns for p in before} == before

    rows = list(csv.DictReader(open(out / "cull_report.csv")))
    assert len(rows) == 6
    kept = {r["photo"]: r for r in rows if r["kept"] == "yes"}
    assert sorted(kept) == ["DSC0001.JPG", "DSC0003.JPG", "DSC0004.JPG"]
    assert kept["DSC0001.JPG"]["look_a"] == "Team Pride"
    assert kept["DSC0001.JPG"]["look_b"] == "MVP, 10% less contrast"  # "Made Up" is not installed


def test_asks_for_size_and_keep_when_not_given(match, monkeypatch, capsys):
    folder, luts = match
    FakeGemini(monkeypatch, scores=[9, 8, 7, 6, 5, 4])
    answers = iter(["9", "3", "bad", "50%"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    run(folder, luts)

    out = folder / "Culled - YouTube"
    assert out.is_dir()
    with Image.open(out / "DSC0000_A.jpg") as img:
        assert img.size[0] / img.size[1] == pytest.approx(16 / 9, rel=0.01)
    printed = capsys.readouterr().out
    assert "What are these photos for?" in printed
    assert "Please type a number" in printed
    assert "enter a number like 30" in printed
    assert len([p for p in os.listdir(out) if p.endswith("_A.jpg")]) == 3


def test_rerun_only_rates_photos_that_failed(match, monkeypatch):
    folder, luts = match
    first = FakeGemini(monkeypatch, scores=[9, 8, 7, 6, 5, 4], failing={2})
    run(folder, luts, "--size", "story", "--keep", "100%")
    assert first.calls == 6

    second = FakeGemini(monkeypatch, scores=[7])
    run(folder, luts, "--size", "story", "--keep", "100%")
    assert second.calls == 1


def test_no_luts_still_enhances(match, monkeypatch, tmp_path):
    folder, _ = match
    FakeGemini(monkeypatch, scores=[9])
    empty = tmp_path / "none"
    empty.mkdir()
    run(folder, empty, "--size", "original", "--keep", "2")
    out = folder / "Culled - Original size"
    assert len([p for p in os.listdir(out) if p.endswith(".jpg")]) == 8  # 2 keepers x O, A, B, C


def test_keep_all_edits_every_photo(match, monkeypatch, capsys):
    folder, luts = match
    FakeGemini(monkeypatch, scores=[2, 9, 3, 8, 1, 6])
    run(folder, luts, "--size", "instagram", "--keep", "all")
    out = folder / "Culled - Instagram post"
    assert len([p for p in os.listdir(out) if p.endswith("_A.jpg")]) == 6
    assert "Editing every photo" in capsys.readouterr().out


def test_ratings_are_reused_for_a_different_size(match, monkeypatch):
    folder, luts = match
    first = FakeGemini(monkeypatch, scores=[9, 8, 7, 6, 5, 4])
    run(folder, luts, "--size", "instagram", "--keep", "2")
    assert first.calls == 6
    second = FakeGemini(monkeypatch, scores=[1])
    run(folder, luts, "--size", "youtube", "--keep", "all")
    assert second.calls == 0
    assert (folder / "Culled - YouTube" / "DSC0005_A.jpg").exists()


def test_original_size_keeps_pixel_dimensions(match, monkeypatch):
    folder, luts = match
    FakeGemini(monkeypatch, scores=[9])
    run(folder, luts, "--size", "original", "--keep", "1")
    out = folder / "Culled - Original size"
    made = sorted(out.glob("*.jpg"))
    assert len(made) == 4
    for path in made:
        with Image.open(path) as img:
            assert img.size == (600, 400)


def test_best_crop_uses_the_ai_shape_and_full_frame_alternate(match, monkeypatch, capsys):
    folder, luts = match
    FakeGemini(monkeypatch, scores=[9], shape="1:1")
    run(folder, luts, "--size", "ai", "--keep", "1")
    out = folder / "Culled - Best crop"
    (file_a,), (file_b,) = list(out.glob("*_A.jpg")), list(out.glob("*_B.jpg"))
    with Image.open(file_a) as a, Image.open(file_b) as b:
        assert a.size[0] == a.size[1]
        assert b.size == (600, 400)
    assert "A = AI crop 1:1" in capsys.readouterr().out


def test_estimate_only_spends_nothing_and_writes_nothing(match, monkeypatch, capsys):
    folder, luts = match
    fake = FakeGemini(monkeypatch, scores=[9])
    run(folder, luts, "--size", "instagram", "--keep", "30%", "--estimate")
    printed = capsys.readouterr().out
    assert fake.calls == 0
    assert not any(p.startswith("Culled") for p in os.listdir(folder))
    assert "6 to rate with Gemini" in printed
    assert "up to 2 keepers" in printed
    assert "Tokens:  about 19,200" in printed


def test_estimate_counts_remembered_photos_and_uses_their_token_average(match, monkeypatch, capsys):
    folder, luts = match
    FakeGemini(monkeypatch, scores=[9], failing={1})
    run(folder, luts, "--size", "instagram", "--keep", "2")
    cache = cli.vision.DetectionCache(str(folder), analysis.CACHE_FILENAME, analysis.PhotoAssessment)
    for name, entry in cache.entries.items():
        entry["tokens"] = 1000
    cache.path.write_text(__import__("json").dumps(cache.entries))
    capsys.readouterr()
    run(folder, luts, "--size", "instagram", "--keep", "all", "--estimate")
    printed = capsys.readouterr().out
    assert "5 already rated, 1 to rate" in printed
    assert "about 1,000 (1 ratings" in printed


def test_declining_the_confirmation_changes_nothing(match, monkeypatch, capsys):
    folder, luts = match
    fake = FakeGemini(monkeypatch, scores=[9])
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    run(folder, luts, "--size", "instagram", "--keep", "2")
    assert fake.calls == 0
    assert "Cancelled" in capsys.readouterr().out
    assert not any(p.startswith("Culled") for p in os.listdir(folder))


def test_learned_style_is_applied_and_taste_reaches_the_rating(match, monkeypatch, tmp_path, capsys):
    folder, luts = match
    refs = tmp_path / "Gold"
    refs.mkdir()
    for i, color in enumerate([(200, 170, 120), (180, 160, 110)]):
        Image.new("RGB", (300, 200), color).save(refs / f"gold{i}.jpg")
    monkeypatch.setattr(cli.style, "_taste_notes", lambda client, paths, model, usage, featured=0: "- loves goalkeepers")
    cli.main([str(refs), "--learn-style"])
    assert (tmp_path / "style.json").exists()
    assert "loves goalkeepers" in capsys.readouterr().out

    fake = FakeGemini(monkeypatch, scores=[9])
    run(folder, luts, "--size", "instagram", "--keep", "1")
    assert fake.taste_notes == "- loves goalkeepers"
    assert "Using your look learned from Gold (2 photos)" in capsys.readouterr().out

    fake = FakeGemini(monkeypatch, scores=[9])
    run(folder, luts, "--size", "story", "--keep", "1", "--no-style", "--refresh")
    assert fake.taste_notes == ""


def test_original_version_is_cropped_but_never_color_edited(match, monkeypatch):
    folder, luts = match
    FakeGemini(monkeypatch, scores=[9])
    run(folder, luts, "--size", "original", "--keep", "1")
    out = folder / "Culled - Original size"
    (o_file,) = list(out.glob("*_O.jpg"))
    stem = o_file.name[:-6]
    with Image.open(folder / f"{stem}.JPG") as src, Image.open(o_file) as o:
        box = cli.crops.crop_box(*src.size, (0.4, 0.3, 0.6, 0.6), None, tight=True)
        expected = src.convert("RGB").crop(box).resize(src.size, Image.Resampling.LANCZOS)
        o_diff = np.abs(np.asarray(o, dtype=int) - np.asarray(expected, dtype=int)).mean()
        assert o_diff < 6  # only JPEG re-compression of the noisy test image, no color changes
        with Image.open(out / f"{stem}_A.jpg") as a:
            a_diff = np.abs(np.asarray(a, dtype=int) - np.asarray(expected, dtype=int)).mean()
        assert a_diff > o_diff + 2  # A really is edited, O is not
    rows = list(csv.DictReader(open(out / "cull_report.csv")))
    kept = [r for r in rows if r["kept"] == "yes"][0]
    assert kept["original_crop"].endswith("_O.jpg") and kept["option_c"].endswith("_C.jpg")


def test_low_resolution_photos_are_flagged(match, monkeypatch, capsys):
    folder, luts = match  # the fixture photos are 600x400, well under 4 MP
    FakeGemini(monkeypatch, scores=[9])
    run(folder, luts, "--size", "instagram", "--keep", "1")
    printed = capsys.readouterr().out
    assert "[low resolution: 0.2 MP]" in printed
    assert "photos are low resolution" in printed
    rows = list(csv.DictReader(open(folder / "Culled - Instagram post" / "cull_report.csv")))
    assert all("low resolution" in r["note"] for r in rows)


def test_default_is_natural_edits_without_luts(match, monkeypatch, capsys):
    folder, luts = match
    FakeGemini(monkeypatch, scores=[9])
    cli.main([str(folder), "--size", "instagram", "--keep", "1"])  # no --luts
    out = folder / "Culled - Instagram post"
    printed = capsys.readouterr().out
    assert "Looks available" not in printed
    assert "C = " in printed and "punchy" in printed
    kept = [r for r in csv.DictReader(open(out / "cull_report.csv")) if r["kept"] == "yes"][0]
    assert kept["look_a"] == "natural" and kept["look_b"] == "natural, 10% less contrast"
    stem = kept["option_a"][:-6]
    with Image.open(out / f"{stem}_A.jpg") as a, Image.open(out / f"{stem}_C.jpg") as c:
        sat = lambda im: np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1].mean()
        assert sat(c) > sat(a)  # C is the punchier version
