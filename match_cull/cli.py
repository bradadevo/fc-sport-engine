import argparse
import csv
import os
import sys
from pathlib import Path

from PIL import Image

import time

from match_cull import analysis, crops, enhance, estimate, images, looks, select, style
from sports_reel import vision

SIZE_MENU = list(crops.SIZES.values())
# Photos smaller than this are probably phone/WhatsApp copies or screenshots
LOW_RES_MP = 4.0
OUTPUT_FOLDER_NAMES = {
    "instagram": "Instagram post",
    "story": "Reel - Story",
    "youtube": "YouTube",
    "original": "Original size",
    "ai": "Best crop",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="match-cull",
        description="Pick the best shots from a match, crop them, enhance them and apply a look. "
                    "Originals are never changed; edits go into a new folder.",
    )
    parser.add_argument("folder", help="Folder of match photos (JPEG, HEIC, PNG or camera RAW); "
                                       "with --learn-style, a folder of your favorite photos")
    parser.add_argument("--learn-style", action="store_true",
                        help="Learn your look from the folder (your favorite photos) and use it on every run")
    parser.add_argument("--best", action="append", metavar="FILE",
                        help="With --learn-style: one of your very best photos in that folder (repeatable); "
                             f"together they make up {round(style.FEATURED_SHARE * 100)}%% of the learned look")
    parser.add_argument("--no-style", action="store_true", help="Ignore your learned look for this run")
    parser.add_argument("--size", choices=sorted(crops.SIZES),
                        help="Skip the size question: " + ", ".join(sorted(crops.SIZES)))
    parser.add_argument("--keep", help="Skip the keep question: a number (30), a percentage (25%%), or all")
    parser.add_argument("--luts", nargs="?", const=str(looks.DEFAULT_LUT_DIR), default=None,
                        help="Use creative LUTs for A and B (off by default: all edits are natural). "
                             "Give a folder of .cube files, or no folder for Final Cut's Custom LUTs")
    parser.add_argument("--out", help="Output folder (default: '<folder>/Culled - <size>')")
    parser.add_argument("--model", default=vision.DEFAULT_MODEL,
                        help=f"Gemini model (default: {vision.DEFAULT_MODEL})")
    parser.add_argument("--estimate", action="store_true",
                        help="Only show how long the run would take and how many tokens it would use")
    parser.add_argument("--yes", action="store_true", help="Start without asking for confirmation")
    parser.add_argument("--refresh", action="store_true",
                        help="Ask Gemini about every photo again, ignoring remembered results")
    return parser.parse_args(argv)


def ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        sys.exit("\nError: no answer given. Use --size and --keep to run without questions.")


def ask_size() -> crops.OutputSize:
    print("\nWhat are these photos for?")
    for i, size in enumerate(SIZE_MENU, 1):
        print(f"  {i}) {size.label}")
    while True:
        answer = ask(f"Choose 1-{len(SIZE_MENU)} [1]: ").strip() or "1"
        if answer.isdigit() and 1 <= int(answer) <= len(SIZE_MENU):
            return SIZE_MENU[int(answer) - 1]
        print(f"  Please type a number from 1 to {len(SIZE_MENU)}.")


def ask_keep(total: int) -> int:
    print(f"\nHow many of the {total} photos do you want to keep?")
    print("  Type a number (e.g. 30), a percentage (e.g. 25%), or 'all' to edit every photo.")
    while True:
        answer = ask("Keep [25%]: ").strip() or "25%"
        try:
            return select.parse_keep(answer, total)
        except ValueError as e:
            print(f"  {e}")


def load_looks(folder: str) -> tuple[dict[str, looks.Lut], dict[str, str]]:
    """Parses every LUT once; returns (name -> Lut, name -> description)."""
    parsed, described = {}, {}
    for name, path in looks.discover_luts(Path(folder)).items():
        try:
            lut = looks.parse_cube(path)
        except Exception as e:
            print(f"Warning: skipping LUT {name}: {e}")
            continue
        lut.strength = looks.auto_strength(lut)
        parsed[name] = lut
        described[name] = looks.describe_lut(lut)
    return parsed, described


def render(img: Image.Image, box: crops.Box, size: crops.OutputSize, lut: looks.Lut | None,
           subject: tuple[float, float, float, float] | None = None,
           my_style: style.Style | None = None, punchy: bool = False, softer: bool = False) -> Image.Image:
    """Crop -> enhance -> look -> final size -> sharpen. subject is the
    normalized (xmin, ymin, xmax, ymax) box of the players, used to judge
    exposure on them rather than on the field or sky."""
    subject_px = None
    if subject:
        w, h = img.size
        subject_px = (round(subject[0] * w) - box[0], round(subject[1] * h) - box[1],
                      round(subject[2] * w) - box[0], round(subject[3] * h) - box[1])
    out = enhance.enhance(img.crop(box), subject_px)
    if lut is not None:
        out = looks.apply_look(out, lut)
    if punchy:
        out = enhance.punch(out)
    if softer:
        out = enhance.soften_contrast(out)
    if my_style is not None:
        out = style.apply_style(out, my_style)
    if size.pixels:
        out = out.resize(size.pixels, Image.Resampling.LANCZOS)
    elif size.match_source and out.size != img.size:
        out = out.resize(img.size, Image.Resampling.LANCZOS)
    return enhance.output_sharpen(out)


def learn_style_command(args) -> None:
    photos = images.find_photos(args.folder)
    print(f"Learning your look from {len(photos)} photos in {args.folder}.")
    print("  Measuring brightness, contrast, color and highlights (free), then one Gemini call\n"
          f"  to describe your taste: about {estimate.TASTE_TOKENS:,} tokens, under a minute.")
    client = vision.create_client() if os.environ.get("GEMINI_API_KEY") else None
    if client is None:
        print("  GEMINI_API_KEY is not set: learning the look only, without taste notes.")
    usage: list[int] = []
    try:
        learned = style.learn(args.folder, client, args.model, usage, featured=[Path(b).name for b in args.best or []])
    except ValueError as e:
        sys.exit(f"Error: {e}")
    if learned.featured:
        print(f"  Your best: {', '.join(learned.featured)} "
              f"({round(style.FEATURED_SHARE * 100)}% of the learned look comes from these)")
    style.save(learned)
    print(f"\nSaved to {style.STYLE_PATH}")
    print(f"  Brightness {learned.lightness:.0f}/100, contrast {learned.contrast:.0f}, "
          f"color {learned.chroma:.1f}, warmth {learned.warmth:+.1f}, "
          f"blown highlights {learned.highlight_clip:.1f}%")
    if learned.notes:
        print("\nYour taste, as Gemini describes it:\n" + learned.notes)
    if usage:
        print(f"\nGemini tokens used: {sum(usage):,}")
    print("\nEvery run now nudges edits toward this look (turn off with --no-style). Photos rated before\n"
          "today keep their old scores; add --refresh to re-rate them with your taste in mind.")


def render_original(img: Image.Image, box: crops.Box, size: crops.OutputSize) -> Image.Image:
    """The O version: the crop (and resize, if the size needs one) only. No
    color, tone, clarity or sharpening changes."""
    out = img.crop(box)
    if size.pixels:
        out = out.resize(size.pixels, Image.Resampling.LANCZOS)
    elif size.match_source and out.size != img.size:
        out = out.resize(img.size, Image.Resampling.LANCZOS)
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.learn_style:
        if not os.path.isdir(args.folder):
            sys.exit(f"Error: Folder not found: {args.folder}")
        return learn_style_command(args)

    if not os.path.isdir(args.folder):
        sys.exit(f"Error: Folder not found: {args.folder}")
    photo_paths = images.find_photos(args.folder)
    if not photo_paths:
        sys.exit(f"Error: No photos found in {args.folder}")

    print(f"Found {len(photo_paths)} photos in {args.folder}.")
    size = crops.SIZES[args.size] if args.size else ask_size()
    try:
        keep = select.parse_keep(args.keep, len(photo_paths)) if args.keep else ask_keep(len(photo_paths))
    except ValueError as e:
        sys.exit(f"Error: --keep {args.keep}: {e}")

    cache = vision.DetectionCache(args.folder, analysis.CACHE_FILENAME, analysis.PhotoAssessment)
    remembered = set() if args.refresh else {p for p in photo_paths if cache.get(p) is not None}
    keep_count = len(photo_paths) if keep == select.KEEP_ALL else min(keep, len(photo_paths))
    plan = estimate.estimate(photo_paths, remembered, keep_count, cache.average_tokens())
    print("\nEstimate:")
    print("\n".join(plan.lines()))
    if args.estimate:
        return
    if plan.to_rate and not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Error: GEMINI_API_KEY is not set. Open a new Terminal window and try again.")
    if not args.yes and sys.stdin.isatty():
        if ask("\nStart? [Y/n]: ").strip().lower() in ("n", "no"):
            print("Cancelled. Nothing was changed.")
            return
    started = time.monotonic()

    out_dir = Path(args.out or Path(args.folder) / f"Culled - {OUTPUT_FOLDER_NAMES[size.key]}")
    out_dir.mkdir(parents=True, exist_ok=True)

    my_style = None if args.no_style else style.load()
    if my_style:
        print(f"\nUsing your look learned from {Path(my_style.source).name} ({my_style.count} photos).")
    taste_notes = my_style.notes if my_style else ""

    lut_table, lut_descriptions = load_looks(args.luts) if args.luts else ({}, {})
    look_names = list(lut_table)
    if look_names:
        print(f"\nLooks available: {', '.join(look_names)}")
    elif args.luts:
        print(f"\nWarning: no .cube LUTs found in {args.luts}; all edits are natural.")

    client = vision.create_client()

    print(f"\nRating {len(photo_paths)} photos with {args.model}...")
    candidates: list[select.Candidate] = []
    assessments: dict[str, analysis.PhotoAssessment] = {}
    failures: dict[str, str] = {}
    tokens_used: list[int] = []
    low_res: dict[str, float] = {}
    for idx, path in enumerate(photo_paths, 1):
        name = os.path.basename(path)
        try:
            img = images.load_image(path)
        except Exception:
            failures[path] = "could not be opened"
            print(f"  [{idx}/{len(photo_paths)}] {name}: skipped (could not be opened)")
            continue

        assessment = None if args.refresh else cache.get(path)
        remembered = assessment is not None
        if assessment is None:
            try:
                usage: list[int] = []
                assessment = analysis.analyze(client, img, lut_descriptions, args.model, usage=usage,
                                              taste_notes=taste_notes)
                cache.put(path, assessment, tokens=sum(usage))
                tokens_used.extend(usage)
            except Exception as e:
                failures[path] = f"rating failed: {str(e)[:100]}"
                print(f"  [{idx}/{len(photo_paths)}] {name}: not rated ({str(e)[:100]})")
                continue

        assessments[path] = assessment
        candidates.append(select.Candidate(path, assessment.score, images.sharpness(img),
                                           images.dhash(img), images.capture_time(path)))
        note = " (remembered)" if remembered else ""
        megapixels = img.size[0] * img.size[1] / 1e6
        if megapixels < LOW_RES_MP:
            low_res[path] = megapixels
            note += f" [low resolution: {megapixels:.1f} MP]"
        print(f"  [{idx}/{len(photo_paths)}] {name}: {assessment.score}/10 - {assessment.reason}{note}")

    keepers = select.choose_keepers(candidates, keep)
    bursts = len({c.burst for c in candidates})
    print(f"\n{len(candidates)} photos rated in {bursts} distinct moments. Keeping {len(keepers)}.")
    if keep == select.KEEP_ALL:
        print("  (Editing every photo: duplicates and low scores included.)")
    elif len(keepers) < keep:
        print(f"  (Asked for {keep}, but only {len(keepers)} distinct moments scored "
              f"{select.MIN_KEEP_SCORE}/10 or better.)")

    print(f"\nEditing keepers into {out_dir} ...")
    outputs: dict[str, tuple[str, str, str, str, str | None, str | None]] = {}
    for idx, cand in enumerate(keepers, 1):
        a = assessments[cand.path]
        best, second = analysis.resolve_looks(a, look_names)
        img = images.load_image(cand.path)
        exif = images.exif_for_output(cand.path)
        subject = (a.xmin, a.ymin, a.xmax, a.ymax)
        stem = Path(cand.path).stem
        file_o, file_a, file_b, file_c = (f"{stem}_{v}.jpg" for v in "OABC")

        if size.ai_framing:
            suggested = (a.crop_xmin, a.crop_ymin, a.crop_xmax, a.crop_ymax)
            box_a = crops.ai_crop_box(*img.size, suggested, a.crop_shape)
            shape_a = a.crop_shape if a.crop_shape.strip() in crops.AI_ASPECTS else "original shape"
            label_a, label_b = f"AI crop {shape_a}", "full frame"
        else:
            box_a = crops.crop_box(*img.size, subject, size.aspect, tight=True)
            label_a, label_b = "tight crop", "wide crop"
        box_b = crops.crop_box(*img.size, subject, size.aspect, tight=False)
        images.save_jpeg(render_original(img, box_a, size), str(out_dir / file_o), exif)
        images.save_jpeg(render(img, box_a, size, lut_table.get(best), subject, my_style),
                         str(out_dir / file_a), exif)
        images.save_jpeg(render(img, box_b, size, lut_table.get(second), subject, my_style, softer=True),
                         str(out_dir / file_b), exif)
        # With LUTs, C is the natural alternative; without, it is the punchier natural edit
        images.save_jpeg(render(img, box_a, size, None, subject, my_style, punchy=not look_names),
                         str(out_dir / file_c), exif)
        best, second = best or "natural", (second or "natural") + ", 10% less contrast"
        c_label = "natural" if look_names else "punchy"
        outputs[cand.path] = (file_o, file_a, file_b, file_c, best, second)
        print(f"  [{idx}/{len(keepers)}] {stem}: O = original, {label_a}; A = {label_a}, {best}; "
              f"B = {label_b}, {second}; C = {label_a}, {c_label}")

    report = out_dir / "cull_report.csv"
    with open(report, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["photo", "kept", "score", "moment", "reason", "subject",
                         "original_crop", "option_a", "look_a", "option_b", "look_b", "option_c", "note"])
        by_path = {c.path: c for c in candidates}
        for path in photo_paths:
            name = os.path.basename(path)
            if path in failures:
                writer.writerow([name, "no", "", "", failures[path], "", "", "", "", "", "", "", ""])
                continue
            a, cand = assessments[path], by_path[path]
            file_o, file_a, file_b, file_c, best, second = outputs.get(path, ("", "", "", "", "", ""))
            writer.writerow([name, "yes" if path in outputs else "no", a.score, cand.burst + 1,
                             a.reason, a.subject_description, file_o, file_a, best or "",
                             file_b, second or "", file_c,
                             f"low resolution ({low_res[path]:.1f} MP): use the original file"
                             if path in low_res else ""])

    print(f"\nDone. {len(keepers)} photos kept, each as O (original crop) + A, B, C edits, in:\n  {out_dir}")
    print(f"Scores and reasons for every photo: {report.name}")
    print(f"Took {estimate.format_duration(time.monotonic() - started)}; "
          f"Gemini tokens used this run: {sum(tokens_used):,} for {len(tokens_used)} ratings.")
    if low_res:
        print(f"Note: {len(low_res)} photos are low resolution (under {LOW_RES_MP:.0f} MP), likely phone, "
              f"WhatsApp or screenshot copies. Edits of those will be soft; the camera originals "
              f"will look much better.")
    if failures:
        print(f"Warning: {len(failures)} photos could not be rated. Run the same command again "
              f"to retry just those.")


if __name__ == "__main__":
    main()
