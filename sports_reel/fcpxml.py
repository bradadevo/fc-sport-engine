from pathlib import Path
import xml.etree.ElementTree as ET

import soundfile as sf

CROSS_DISSOLVE_UID = "FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265"
BASIC_TITLE_UID = ".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti"
TITLE_DURATION_SEC = 2.5

# Supported timeline rates -> exact frame duration as (numerator, denominator) seconds
FRAME_DURATIONS = {
    29.97: (1001, 30000),
    60.0: (100, 6000),
}


def frames_to_time(frames: int, frame_duration: tuple[int, int]) -> str:
    """Formats a frame count as an FCPXML rational time string."""
    num, den = frame_duration
    return f"{frames * num}/{den}s"


def cuts_to_frame_durations(cut_times: list[float], frame_duration: tuple[int, int]) -> list[int]:
    """Snaps each cut to a frame, so clips stay on the beat instead of
    drifting from per-clip rounding. Returns one frame count per clip."""
    num, den = frame_duration
    cut_frames = [round(t * den / num) for t in cut_times]
    return [max(1, b - a) for a, b in zip(cut_frames, cut_frames[1:])]


def assemble_fcpxml(
    photos_data: list[dict],
    cut_times: list[float],
    output_xml: str,
    audio_path: str | None = None,
    total_audio_duration: float = 0.0,
    fps: float = 29.97,
    transition_sec: float = 0.0,
    title: str | None = None,
) -> None:
    """
    Writes an FCPXML 1.11 project: a 1080x1920 timeline with one Ken Burns
    still per cut and the music connected underneath.

    photos_data items need: filepath, width, height, start_box, end_box
    (boxes as produced by geometry.compute_pan_zoom_offsets). cut_times needs
    one more entry than photos_data.

    transition_sec > 0 puts a cross dissolve of about that length on every
    cut, centered on the beat. title adds an opening Basic Title.
    """
    if len(cut_times) != len(photos_data) + 1:
        raise ValueError("cut_times must have exactly one more entry than photos_data")

    frame_duration = FRAME_DURATIONS[fps]
    num, den = frame_duration

    fcpxml = ET.Element("fcpxml", version="1.11")
    resources = ET.SubElement(fcpxml, "resources")

    # Vertical 1080x1920 timeline format
    ET.SubElement(resources, "format", id="r1", frameDuration=frames_to_time(1, frame_duration),
                  width="1080", height="1920", colorSpace="1-1-1 (Rec. 709)")

    # Audio Asset
    audio_frames = int(total_audio_duration * den / num)
    if audio_path:
        audio_attrs = {}
        try:
            info = sf.info(audio_path)
            audio_attrs = {"audioChannels": str(info.channels), "audioRate": str(info.samplerate)}
        except Exception:
            pass  # Final Cut reads these from the file if they are missing
        audio_asset = ET.SubElement(resources, "asset", id="r_audio", name=Path(audio_path).name,
                                    start="0s", duration=frames_to_time(audio_frames, frame_duration),
                                    hasAudio="1", audioSources="1", **audio_attrs)
        ET.SubElement(audio_asset, "media-rep", kind="original-media", src=Path(audio_path).resolve().as_uri())

    # Image Assets (stills need a format carrying their own pixel dimensions)
    image_formats = {}
    for idx, item in enumerate(photos_data):
        size = (item["width"], item["height"])
        if size not in image_formats:
            fmt_id = f"r_fmt_{len(image_formats)}"
            image_formats[size] = fmt_id
            ET.SubElement(resources, "format", id=fmt_id, name="FFVideoFormatRateUndefined",
                          width=str(size[0]), height=str(size[1]), colorSpace="1-13-1")
        asset = ET.SubElement(resources, "asset", id=f"r_img_{idx}", name=Path(item["filepath"]).name,
                              start="0s", duration="0s", hasVideo="1", videoSources="1",
                              format=image_formats[size])
        ET.SubElement(asset, "media-rep", kind="original-media", src=Path(item["filepath"]).resolve().as_uri())

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name="AI Sports Reels")
    project = ET.SubElement(event, "project", name="Soccer_Autocut_Reel")

    clip_durations = cuts_to_frame_durations(cut_times, frame_duration)
    total_frames = sum(clip_durations)

    # Half-length of each dissolve in frames, capped so neighbouring
    # dissolves on a short clip can never run into each other
    half = 0
    if transition_sec > 0 and len(photos_data) > 1:
        half = min(max(1, round(transition_sec * den / num / 2)), min(clip_durations) // 3)
    if half:
        ET.SubElement(resources, "effect", id="r_dissolve", name="Cross Dissolve", uid=CROSS_DISSOLVE_UID)
    if title:
        ET.SubElement(resources, "effect", id="r_title", name="Basic Title", uid=BASIC_TITLE_UID)

    sequence = ET.SubElement(project, "sequence", format="r1",
                             duration=frames_to_time(total_frames, frame_duration),
                             tcStart="0s", tcFormat="NDF", audioLayout="stereo", audioRate="48k")
    spine = ET.SubElement(sequence, "spine")

    # Stills start one hour into their own (endless) media, as Final Cut does,
    # which leaves room on both sides for a dissolve to borrow frames from
    still_start = frames_to_time(3600 * round(den / num), frame_duration)

    # Append Image Clips with Ken Burns Pan-and-Zoom. Clips butt up against
    # each other on the cut; a dissolve sits across the cut, centered on it.
    cut_frame = 0
    first_clip = None
    last_idx = len(photos_data) - 1
    for idx, item in enumerate(photos_data):
        clip = ET.SubElement(spine, "video", ref=f"r_img_{idx}", name=Path(item["filepath"]).name,
                             offset=frames_to_time(cut_frame, frame_duration), start=still_start,
                             duration=frames_to_time(clip_durations[idx], frame_duration))

        crop = ET.SubElement(clip, "adjust-crop", mode="pan")
        for box in (item["start_box"], item["end_box"]):
            left, top, right, bottom = box
            ET.SubElement(crop, "pan-rect", left=str(left), top=str(top), right=str(right), bottom=str(bottom))

        if first_clip is None:
            first_clip = clip
        cut_frame += clip_durations[idx]

        if half and idx < last_idx:
            transition = ET.SubElement(spine, "transition", name="Cross Dissolve",
                                       offset=frames_to_time(cut_frame - half, frame_duration),
                                       duration=frames_to_time(2 * half, frame_duration))
            ET.SubElement(transition, "filter-video", ref="r_dissolve", name="Cross Dissolve")

    # Connect Audio Bed underneath the primary storyline (anchored to the first clip)
    if audio_path and first_clip is not None:
        ET.SubElement(first_clip, "asset-clip", ref="r_audio", lane="-1", offset=still_start,
                      name=Path(audio_path).name,
                      duration=frames_to_time(min(total_frames, audio_frames), frame_duration),
                      audioRole="music")

    # Opening title above the primary storyline, kept inside the social-media safe zone
    if title and first_clip is not None:
        title_frames = min(total_frames, round(TITLE_DURATION_SEC * den / num))
        title_clip = ET.SubElement(first_clip, "title", ref="r_title", lane="1", offset=still_start,
                                   name=title, start=still_start,
                                   duration=frames_to_time(title_frames, frame_duration))
        text = ET.SubElement(title_clip, "text")
        ET.SubElement(text, "text-style", ref="ts1").text = title
        style_def = ET.SubElement(title_clip, "text-style-def", id="ts1")
        ET.SubElement(style_def, "text-style", font="Helvetica Neue", fontSize="96", fontFace="Bold",
                      fontColor="1 1 1 1", shadowColor="0 0 0 0.75", shadowOffset="4 315",
                      alignment="center")
        ET.SubElement(title_clip, "adjust-transform", position="0 25")

    ET.indent(fcpxml, space="  ", level=0)
    with open(output_xml, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n')
        f.write(ET.tostring(fcpxml, encoding="unicode"))
        f.write("\n")
