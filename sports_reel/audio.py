import os
import shutil
import subprocess
import tempfile

import librosa

FALLBACK_INTERVAL_SEC = 1.25


def _load_audio(audio_path: str):
    """Loads audio at its native sample rate, decoding through ffmpeg for
    formats libsndfile cannot read (.m4a, .aac)."""
    try:
        return librosa.load(audio_path, sr=None)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                f"Could not decode {audio_path}. Install ffmpeg (brew install ffmpeg) "
                "to read .m4a/.aac files."
            )
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "decoded.wav")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", audio_path, "-vn", wav_path],
            check=True,
        )
        return librosa.load(wav_path, sr=None)


def extract_beat_grid(audio_path: str | None, min_interval_sec: float = 0.8) -> tuple[list[float], float]:
    """
    Extracts audio beats using librosa's dynamic programming beat tracker.
    Enforces a minimum duration per photo to prevent seizure-inducing cut rates.

    Returns (beat_times, total_duration). beat_times always starts at 0.0; if
    no beats are found it contains only that, and plan_cuts() falls back to
    constant pacing. A missing audio_path returns ([0.0], 0.0).
    """
    if not audio_path:
        return [0.0], 0.0

    y, sr = _load_audio(audio_path)
    duration = float(librosa.get_duration(y=y, sr=sr))

    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        raw_beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    except Exception:
        raw_beat_times = []

    # Filter beats that are clustered too tightly for still photo pacing
    filtered_beats = [0.0]
    for b in raw_beat_times:
        if b - filtered_beats[-1] >= min_interval_sec:
            filtered_beats.append(float(b))

    return filtered_beats, duration


def plan_cuts(
    beat_times: list[float],
    total_duration: float,
    num_photos: int,
    min_interval_sec: float = 0.8,
    fallback_interval_sec: float = FALLBACK_INTERVAL_SEC,
) -> list[float]:
    """
    Turns a beat grid into cut boundaries: clip i runs cuts[i] -> cuts[i+1].

    Cuts follow the beats, then continue at constant fallback pacing until the
    music ends (this also covers tracks with no detectable beats). Photos that
    would land after the end of the music are dropped, so the result may hold
    fewer than num_photos clips. With no audio (total_duration <= 0), every
    photo gets the fallback interval.
    """
    if num_photos <= 0:
        return [0.0]

    if total_duration <= 0:
        return [i * fallback_interval_sec for i in range(num_photos + 1)]

    cuts = [b for b in beat_times if b < total_duration] or [0.0]
    while cuts[-1] + fallback_interval_sec < total_duration:
        cuts.append(cuts[-1] + fallback_interval_sec)

    # Close the final clip at the end of the music, absorbing a too-short tail
    if total_duration - cuts[-1] >= min_interval_sec or len(cuts) == 1:
        cuts.append(total_duration)
    else:
        cuts[-1] = total_duration

    return cuts[: num_photos + 1]
