import argparse
import os
import sys
from pathlib import Path

from sports_reel import audio, fcpxml, geometry, vision

PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".heif")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".aac", ".flac")
PROTECTED_FOLDERS = ("Downloads", "Desktop", "Documents")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sports-reel",
        description="Build a beat-synced 9:16 Final Cut Pro timeline from sports stills and a music track.",
    )
    parser.add_argument("folder", nargs="?",
                        help="Folder holding the photos and one music track")
    parser.add_argument("--photos", help="Path to photo directory (instead of the folder argument)")
    parser.add_argument("--audio", help="Path to music track (default: the one audio file in the photo folder)")
    parser.add_argument("--out", help="Path to output .fcpxml (default: <photos_dir>/AI_Sports_Reel.fcpxml)")
    parser.add_argument("--fps", type=float, choices=sorted(fcpxml.FRAME_DURATIONS), default=29.97,
                        help="Target framerate (default: 29.97)")
    parser.add_argument("--min-duration", type=float, default=0.8,
                        help="Minimum clip duration in seconds (default: 0.8)")
    parser.add_argument("--transition", type=float, default=0.5,
                        help="Cross dissolve length in seconds on every cut; 0 for hard cuts (default: 0.5)")
    parser.add_argument("--title", help="Opening title text (default: no title)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse audio and vision without writing XML")
    parser.add_argument("--model", default=vision.DEFAULT_MODEL,
                        help=f"Gemini model for subject detection (default: {vision.DEFAULT_MODEL})")
    parser.add_argument("--refresh", action="store_true",
                        help="Ask Gemini about every photo again, ignoring remembered detections")
    parser.add_argument("--no-vision", action="store_true",
                        help="Skip Gemini and center-crop every photo (no API key needed)")
    args = parser.parse_args(argv)
    if args.folder and args.photos:
        parser.error("give the folder either on its own or with --photos, not both")
    args.photos = args.photos or args.folder
    if not args.photos:
        parser.error("point me at a folder of photos, e.g.  sports-reel ~/Movies/match")
    return args


def protected_folder(path: str) -> str | None:
    """Returns the name of the macOS privacy-protected folder (Downloads,
    Desktop, Documents) containing path, if any. Final Cut Pro is often not
    allowed to read media there, and has crashed on import when it cannot."""
    resolved = Path(path).resolve()
    for name in PROTECTED_FOLDERS:
        if resolved.is_relative_to(Path.home() / name):
            return name
    return None


def find_photos(photos_dir: str) -> list[str]:
    """Photos in the folder, sorted by name to respect bursts / camera order."""
    return sorted(
        str(p) for p in Path(photos_dir).iterdir()
        if p.is_file() and p.suffix.lower() in PHOTO_EXTENSIONS and not p.name.startswith(".")
    )


def find_audio(photos_dir: str) -> list[str]:
    """Audio files sitting in the photo folder, sorted by name."""
    return sorted(
        str(p) for p in Path(photos_dir).iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS and not p.name.startswith(".")
    )


def main(argv=None) -> None:
    args = parse_args(argv)

    if not os.path.isdir(args.photos):
        sys.exit(f"Error: Photo directory not found: {args.photos}")
    if not args.audio:
        tracks = find_audio(args.photos)
        if not tracks:
            sys.exit(f"Error: No music track found in {args.photos}. "
                     "Put one audio file in the folder, or pass --audio.")
        if len(tracks) > 1:
            names = ", ".join(os.path.basename(t) for t in tracks)
            sys.exit(f"Error: Found {len(tracks)} music tracks in the folder ({names}). "
                     "Keep just one there, or choose with --audio.")
        args.audio = tracks[0]
        print(f"Using music track: {os.path.basename(args.audio)}")
    if not os.path.isfile(args.audio):
        sys.exit(f"Error: Audio track not found at {args.audio}")
    if not args.no_vision and not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Error: GEMINI_API_KEY is not set. Export it in your terminal, "
                 "or pass --no-vision to center-crop every photo.")

    # Read every photo's size up front, so one damaged file is skipped
    # instead of leaving a hole in the timeline
    photo_paths = []
    sizes = {}
    for path in find_photos(args.photos):
        try:
            sizes[path] = geometry.upright_size(path)
            photo_paths.append(path)
        except Exception:
            print(f"Warning: skipping {os.path.basename(path)}: it could not be read as an image.")
    if not photo_paths:
        sys.exit(f"Error: No photos (JPEG, PNG or HEIC) found in folder: {args.photos}")

    output_xml = args.out or str(Path(args.photos) / "AI_Sports_Reel.fcpxml")

    for label, path in (("Photos", args.photos), ("Audio", args.audio)):
        folder = protected_folder(path)
        if folder:
            print(f"Warning: {label} are in your {folder} folder, which Final Cut Pro may not be "
                  f"allowed to read (it can crash on import). Move them to ~/Movies or ~/Pictures, "
                  f"or allow Final Cut Pro under System Settings > Privacy & Security > Files and Folders.")

    print(f"Found {len(photo_paths)} photos. Extracting musical rhythm grid...")
    beat_times, total_duration = audio.extract_beat_grid(args.audio, min_interval_sec=args.min_duration)
    if len(beat_times) > 1:
        print(f"Detected {len(beat_times)} sync beats across {total_duration:.2f} seconds.")
    else:
        print(f"No beats detected across {total_duration:.2f} seconds; "
              f"using constant {audio.FALLBACK_INTERVAL_SEC}s pacing.")

    cut_times = audio.plan_cuts(beat_times, total_duration, len(photo_paths),
                                min_interval_sec=args.min_duration)
    num_clips = len(cut_times) - 1
    if num_clips < len(photo_paths):
        print(f"Warning: the music fits {num_clips} clips; "
              f"dropping the last {len(photo_paths) - num_clips} photos.")
    photo_paths = photo_paths[:num_clips]

    client = None if args.no_vision else vision.create_client()
    if client is None:
        print("\nVision skipped (--no-vision): center-cropping every photo.")
    else:
        print(f"\nStarting Vision Pipeline analysis with {args.model}...")

    cache = None if client is None else vision.DetectionCache(args.photos)

    processed_photos = []
    fallbacks = 0
    remembered = 0
    for idx, img_path in enumerate(photo_paths):
        name = os.path.basename(img_path)
        w, h = sizes[img_path]

        detection = cache.get(img_path) if cache and not args.refresh else None
        if detection:
            remembered += 1
            print(f"  [{idx + 1}/{num_clips}] {name}: {detection.motion_type} - "
                  f"{detection.subject_description} (remembered)")
        else:
            detection, error = vision.analyze_photo_or_center(client, img_path, model=args.model)
            if error:
                fallbacks += 1
                print(f"  [{idx + 1}/{num_clips}] {name}: center crop (vision failed: {error[:120]})")
            elif cache:
                cache.put(img_path, detection)
                print(f"  [{idx + 1}/{num_clips}] {name}: {detection.motion_type} - {detection.subject_description}")

        start_box, end_box = geometry.compute_pan_zoom_offsets(w, h, detection)
        processed_photos.append({
            "filepath": img_path,
            "width": w,
            "height": h,
            "start_box": start_box,
            "end_box": end_box,
        })

    if remembered:
        print(f"Reused {remembered} remembered detections (use --refresh to redo them).")
    if fallbacks:
        print(f"Warning: {fallbacks} of {num_clips} photos fell back to a center crop. "
              f"Run the same command again to retry just those.")

    if args.dry_run:
        print(f"\n[Dry run] {num_clips} clips, {cut_times[-1]:.2f}s timeline. No XML written.")
        for item, start, end in zip(processed_photos, cut_times, cut_times[1:]):
            print(f"  {start:7.2f}s -> {end:7.2f}s  {os.path.basename(item['filepath'])}")
        return

    fcpxml.assemble_fcpxml(processed_photos, cut_times, output_xml, audio_path=args.audio,
                           total_audio_duration=total_duration, fps=args.fps,
                           transition_sec=args.transition, title=args.title)
    print(f"\n[FCPXML Exported] -> {output_xml}")


if __name__ == "__main__":
    main()
