import shutil
import subprocess

import librosa
import numpy as np
import pytest
import soundfile as sf

from sports_reel.audio import FALLBACK_INTERVAL_SEC, extract_beat_grid, plan_cuts

SR = 22050


def write_click_track(path, seconds=12, interval=0.5):
    times = np.arange(0, seconds, interval)
    y = librosa.clicks(times=times, sr=SR, length=SR * seconds)
    sf.write(path, y, SR)


def test_beat_grid_follows_tempo_and_min_interval(tmp_path):
    wav = tmp_path / "clicks.wav"
    write_click_track(wav)

    beats, duration = extract_beat_grid(str(wav), min_interval_sec=0.8)

    assert duration == pytest.approx(12.0, abs=0.01)
    assert beats[0] == 0.0
    assert len(beats) > 5
    gaps = np.diff(beats)
    assert gaps.min() >= 0.8
    # 120 bpm clicks filtered at 0.8s leave every other beat: ~1.0s apart
    assert np.median(gaps) == pytest.approx(1.0, abs=0.1)


def test_silence_yields_no_beats(tmp_path):
    wav = tmp_path / "silence.wav"
    sf.write(wav, np.zeros(SR * 5, dtype=np.float32), SR)

    beats, duration = extract_beat_grid(str(wav))

    assert beats == [0.0]
    assert duration == pytest.approx(5.0, abs=0.01)


def test_missing_audio_path():
    assert extract_beat_grid(None) == ([0.0], 0.0)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_m4a_decodes_through_ffmpeg(tmp_path):
    wav = tmp_path / "clicks.wav"
    m4a = tmp_path / "clicks.m4a"
    write_click_track(wav)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), str(m4a)], check=True)

    beats, duration = extract_beat_grid(str(m4a))

    assert duration == pytest.approx(12.0, abs=0.2)
    assert len(beats) > 5


def test_plan_cuts_uses_beats_when_photos_are_scarce():
    cuts = plan_cuts([0.0, 1.0, 2.0, 3.0, 4.0], total_duration=10.0, num_photos=3)
    assert cuts == [0.0, 1.0, 2.0, 3.0]


def test_plan_cuts_falls_back_to_constant_pacing_without_beats():
    cuts = plan_cuts([0.0], total_duration=5.0, num_photos=10)
    assert cuts == [0.0, 1.25, 2.5, 3.75, 5.0]


def test_plan_cuts_continues_pacing_after_last_beat_until_music_ends():
    cuts = plan_cuts([0.0, 1.0, 2.0], total_duration=6.0, num_photos=10)
    assert cuts == [0.0, 1.0, 2.0, 3.25, 4.5, 6.0]


def test_plan_cuts_absorbs_short_tail_into_last_clip():
    # 0.3s left after the last beat is below the minimum, so the clip stretches
    cuts = plan_cuts([0.0, 1.0, 2.0], total_duration=2.3, num_photos=10)
    assert cuts == [0.0, 1.0, 2.3]


def test_plan_cuts_without_audio_paces_every_photo():
    cuts = plan_cuts([0.0], total_duration=0.0, num_photos=3)
    assert cuts == [0.0, FALLBACK_INTERVAL_SEC, 2 * FALLBACK_INTERVAL_SEC, 3 * FALLBACK_INTERVAL_SEC]


def test_plan_cuts_clips_respect_min_interval():
    beats = [0.0, 0.9, 1.8, 2.7]
    cuts = plan_cuts(beats, total_duration=7.0, num_photos=50, min_interval_sec=0.8)
    assert min(np.diff(cuts)) >= 0.8
    assert cuts[-1] == 7.0
