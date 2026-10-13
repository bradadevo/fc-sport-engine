import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from sports_reel import cli


@pytest.fixture
def folder(tmp_path):
    for i in range(3):
        Image.new("RGB", (600, 400)).save(tmp_path / f"IMG_{i}.JPG")
    return tmp_path


def add_track(folder, name="song.wav", seconds=4):
    sf.write(folder / name, np.zeros(22050 * seconds, dtype=np.float32), 22050)
    return folder / name


def test_folder_alone_finds_photos_and_the_music_track(folder, capsys):
    add_track(folder)
    cli.main([str(folder), "--no-vision"])

    assert (folder / "AI_Sports_Reel.fcpxml").exists()
    assert "Using music track: song.wav" in capsys.readouterr().out


def test_photos_and_audio_flags_still_work(folder, tmp_path_factory):
    track = add_track(tmp_path_factory.mktemp("music"))
    cli.main(["--photos", str(folder), "--audio", str(track), "--no-vision"])
    assert (folder / "AI_Sports_Reel.fcpxml").exists()


def test_no_music_track_is_a_clear_error(folder):
    with pytest.raises(SystemExit, match="No music track found"):
        cli.main([str(folder), "--no-vision"])


def test_several_music_tracks_asks_you_to_choose(folder):
    add_track(folder, "a.wav")
    add_track(folder, "b.wav")
    with pytest.raises(SystemExit, match="Found 2 music tracks"):
        cli.main([str(folder), "--no-vision"])


def test_audio_flag_picks_between_several_tracks(folder):
    add_track(folder, "a.wav")
    chosen = add_track(folder, "b.wav")
    cli.main([str(folder), "--audio", str(chosen), "--no-vision"])
    assert "b.wav" in (folder / "AI_Sports_Reel.fcpxml").read_text()


def test_no_folder_is_a_usage_error(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--no-vision"])
    assert "point me at a folder" in capsys.readouterr().err


class CountingVision:
    """Replaces the Gemini call; fails for the named photos."""

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.calls = []

    def __call__(self, client, image_path, model=None):
        name = image_path.rsplit("/", 1)[-1]
        self.calls.append(name)
        if name in self.failing:
            raise RuntimeError("503 high demand")
        return cli.vision.SoccerSubjectDetection(
            subject_description="shot", ymin=0.2, xmin=0.3, ymax=0.6, xmax=0.7, motion_type="push_in")


@pytest.fixture
def fake_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cli.vision, "create_client", lambda: object())

    def install(failing=()):
        fake = CountingVision(failing)
        monkeypatch.setattr(cli.vision, "analyze_photo", fake)
        return fake

    return install


def test_rerun_only_retries_photos_that_failed(folder, fake_gemini, capsys):
    add_track(folder)
    first = fake_gemini(failing={"IMG_1.JPG"})
    cli.main([str(folder)])
    assert sorted(first.calls) == ["IMG_0.JPG", "IMG_1.JPG", "IMG_2.JPG"]
    assert "1 of 3 photos fell back" in capsys.readouterr().out

    second = fake_gemini()
    cli.main([str(folder)])
    assert second.calls == ["IMG_1.JPG"]
    out = capsys.readouterr().out
    assert "Reused 2 remembered detections" in out
    assert "fell back" not in out

    third = fake_gemini()
    cli.main([str(folder)])
    assert third.calls == []


def test_refresh_asks_about_every_photo_again(folder, fake_gemini):
    add_track(folder)
    fake_gemini()
    cli.main([str(folder)])

    again = fake_gemini()
    cli.main([str(folder), "--refresh"])
    assert len(again.calls) == 3


def test_cache_file_is_not_mistaken_for_media(folder, fake_gemini):
    add_track(folder)
    fake_gemini()
    cli.main([str(folder)])
    assert (folder / cli.vision.CACHE_FILENAME).exists()
    assert len(cli.find_photos(str(folder))) == 3
    assert len(cli.find_audio(str(folder))) == 1


def test_unreadable_photo_is_skipped_with_a_warning(folder, capsys):
    add_track(folder)
    (folder / "IMG_broken.jpg").write_bytes(b"this is not an image")
    cli.main([str(folder), "--no-vision"])

    assert "skipping IMG_broken.jpg" in capsys.readouterr().out
    xml = (folder / "AI_Sports_Reel.fcpxml").read_text()
    assert "IMG_broken" not in xml
    assert xml.count("<video ") == 3


def test_png_and_heic_photos_are_included(folder):
    import pillow_heif
    add_track(folder, seconds=10)
    Image.new("RGB", (400, 600)).save(folder / "IMG_7.png")
    pillow_heif.from_pillow(Image.new("RGB", (400, 600))).save(folder / "IMG_8.heic")
    cli.main([str(folder), "--no-vision"])

    xml = (folder / "AI_Sports_Reel.fcpxml").read_text()
    assert "IMG_7.png" in xml and "IMG_8.heic" in xml
    assert xml.count("<video ") == 5
