import shutil
import subprocess
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from sports_reel.fcpxml import assemble_fcpxml, cuts_to_frame_durations
from sports_reel.geometry import compute_pan_zoom_offsets
from sports_reel.vision import center_detection

FCP_DTD = Path("/Applications/Final Cut Pro.app/Contents/Frameworks/Interchange.framework"
               "/Versions/A/Resources/FCPXMLv1_11.dtd")


def seconds(time_str):
    return Fraction(time_str.rstrip("s"))


@pytest.fixture
def reel(tmp_path):
    """Builds a small project in a folder whose name contains a space."""
    photos_dir = tmp_path / "match photos"
    photos_dir.mkdir()
    audio_path = tmp_path / "song.wav"
    sf.write(audio_path, np.zeros((44100 * 6, 2), dtype=np.float32), 44100)

    photos = []
    for i, size in enumerate([(600, 400), (400, 600), (600, 400)]):
        path = photos_dir / f"IMG_{i:04d}.JPG"
        Image.new("RGB", size).save(path)
        start_box, end_box = compute_pan_zoom_offsets(*size, center_detection())
        photos.append({"filepath": str(path), "width": size[0], "height": size[1],
                       "start_box": start_box, "end_box": end_box})

    def build(fps=29.97, with_audio=True, transition_sec=0.0, title=None):
        out = tmp_path / f"reel_{fps}_{with_audio}_{transition_sec}_{bool(title)}.fcpxml"
        assemble_fcpxml(photos, [0.0, 1.0, 2.5, 4.0], str(out),
                        audio_path=str(audio_path) if with_audio else None,
                        total_audio_duration=6.0 if with_audio else 0.0, fps=fps,
                        transition_sec=transition_sec, title=title)
        return out

    return build


@pytest.mark.skipif(not (FCP_DTD.exists() and shutil.which("xmllint")),
                    reason="Final Cut Pro DTD or xmllint not available")
@pytest.mark.parametrize("fps", [29.97, 60.0])
@pytest.mark.parametrize("with_audio", [True, False])
@pytest.mark.parametrize("effects", [{}, {"transition_sec": 0.5, "title": "MATCH DAY & <more>"}])
def test_output_validates_against_final_cut_dtd(reel, tmp_path, fps, with_audio, effects):
    out = reel(fps=fps, with_audio=with_audio, **effects)
    dtd = tmp_path / "fcpxml.dtd"  # xmllint cannot load a DTD from a path with spaces
    shutil.copy(FCP_DTD, dtd)

    result = subprocess.run(["xmllint", "--noout", "--dtdvalid", str(dtd), str(out)],
                            capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_clips_are_contiguous_and_on_the_cuts(reel):
    root = ET.parse(reel()).getroot()
    clips = root.findall("./library/event/project/sequence/spine/video")

    assert len(clips) == 3
    position = Fraction(0)
    for clip in clips:
        assert seconds(clip.get("offset")) == position
        position += seconds(clip.get("duration"))
    assert seconds(root.find(".//sequence").get("duration")) == position
    assert float(position) == pytest.approx(4.0, abs=1 / 29.97)
    # Second cut lands within half a frame of 1.0s
    assert float(seconds(clips[1].get("offset"))) == pytest.approx(1.0, abs=0.5 / 29.97)


def test_every_clip_has_two_pan_rects(reel):
    root = ET.parse(reel()).getroot()
    for clip in root.findall(".//spine/video"):
        crop = clip.find("adjust-crop")
        assert crop.get("mode") == "pan"
        assert len(crop.findall("pan-rect")) == 2


def test_assets_use_absolute_file_uris(reel):
    root = ET.parse(reel()).getroot()
    sources = [rep.get("src") for rep in root.findall("./resources/asset/media-rep")]

    assert len(sources) == 4  # audio + 3 photos
    for src in sources:
        assert src.startswith("file:///")
        assert " " not in src
    assert any("match%20photos" in src for src in sources)


def test_audio_is_connected_under_first_clip_and_trimmed_to_timeline(reel):
    root = ET.parse(reel()).getroot()
    clips = root.findall(".//spine/video")
    audio_clips = root.findall(".//asset-clip")

    assert len(audio_clips) == 1
    assert clips[0].find("asset-clip") is audio_clips[0]
    assert audio_clips[0].get("lane") == "-1"
    assert audio_clips[0].get("duration") == root.find(".//sequence").get("duration")


def test_no_audio_omits_audio_asset_and_clip(reel):
    root = ET.parse(reel(with_audio=False)).getroot()
    assert root.find(".//asset[@id='r_audio']") is None
    assert root.find(".//asset-clip") is None


def test_time_strings_use_the_frame_rate_denominator(reel):
    for fps, den in [(29.97, "30000s"), (60.0, "6000s")]:
        root = ET.parse(reel(fps=fps)).getroot()
        for clip in root.findall(".//spine/video"):
            assert clip.get("duration").endswith("/" + den)


def test_frame_snapping_does_not_drift():
    cuts = [i * 0.517 for i in range(200)]
    durations = cuts_to_frame_durations(cuts, (1001, 30000))
    assert sum(durations) == round(cuts[-1] * 30000 / 1001)


def test_rejects_mismatched_cut_count(tmp_path):
    with pytest.raises(ValueError):
        assemble_fcpxml([], [0.0, 1.0], str(tmp_path / "x.fcpxml"))


def test_dissolves_are_centered_on_every_cut(reel):
    plain = ET.parse(reel()).getroot()
    cuts = [seconds(c.get("offset")) for c in plain.findall(".//spine/video")][1:]

    root = ET.parse(reel(transition_sec=0.3)).getroot()
    transitions = root.findall(".//spine/transition")

    assert len(transitions) == len(cuts) == 2
    for transition, cut in zip(transitions, cuts):
        start, duration = seconds(transition.get("offset")), seconds(transition.get("duration"))
        assert start + duration / 2 == cut
        assert float(duration) == pytest.approx(0.3, abs=1 / 29.97)
        assert transition.find("filter-video").get("ref") == "r_dissolve"
    assert root.find("./resources/effect[@id='r_dissolve']") is not None


def test_dissolves_do_not_move_or_resize_the_clips(reel):
    plain = ET.parse(reel()).getroot().findall(".//spine/video")
    root = ET.parse(reel(transition_sec=0.3)).getroot()
    spine = list(root.find(".//spine"))

    assert [el.tag for el in spine] == ["video", "transition", "video", "transition", "video"]
    clips = [el for el in spine if el.tag == "video"]
    for clip, original in zip(clips, plain):
        assert clip.get("offset") == original.get("offset")
        assert clip.get("duration") == original.get("duration")


def test_stills_start_an_hour_in_and_children_follow(reel):
    for fps, hour in [(29.97, Fraction(108108000, 30000)), (60.0, Fraction(3600))]:
        root = ET.parse(reel(fps=fps, title="PSG 10U")).getroot()
        clips = root.findall(".//spine/video")
        assert all(seconds(c.get("start")) == hour for c in clips)
        assert seconds(clips[0].find("asset-clip").get("offset")) == hour
        assert seconds(clips[0].find("title").get("offset")) == hour


def test_dissolve_is_capped_on_short_clips(tmp_path):
    photo = {"filepath": str(tmp_path / "a.jpg"), "width": 600, "height": 400,
             "start_box": (0.0, 0.0, 0.0, 0.0), "end_box": (0.0, 0.0, 0.0, 0.0)}
    out = tmp_path / "short.fcpxml"
    assemble_fcpxml([photo] * 3, [0.0, 0.2, 0.4, 0.6], str(out), transition_sec=2.0)

    root = ET.parse(out).getroot()
    offsets = [seconds(t.get("offset")) for t in root.findall(".//spine/transition")]
    durations = [seconds(t.get("duration")) for t in root.findall(".//spine/transition")]
    assert offsets[0] + durations[0] <= offsets[1]


def test_no_dissolves_by_default_or_for_a_single_photo(reel, tmp_path):
    assert ET.parse(reel()).getroot().find(".//transition") is None
    photo = {"filepath": str(tmp_path / "a.jpg"), "width": 600, "height": 400,
             "start_box": (0.0, 0.0, 0.0, 0.0), "end_box": (0.0, 0.0, 0.0, 0.0)}
    out = tmp_path / "one.fcpxml"
    assemble_fcpxml([photo], [0.0, 1.0], str(out), transition_sec=0.3)
    root = ET.parse(out).getroot()
    assert root.find(".//transition") is None
    assert root.find("./resources/effect") is None


def test_title_is_connected_above_the_first_clip(reel):
    root = ET.parse(reel(title="PSG 10U")).getroot()
    clips = root.findall(".//spine/video")
    titles = root.findall(".//title")

    assert len(titles) == 1
    assert clips[0].find("title") is titles[0]
    assert titles[0].get("lane") == "1"
    assert titles[0].find("text/text-style").text == "PSG 10U"
    assert titles[0].get("ref") == "r_title"
    assert root.find("./resources/effect[@id='r_title']").get("uid").endswith("Basic Title.moti")
    assert ET.parse(reel()).getroot().find(".//title") is None
