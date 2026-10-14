# Sports Reel Engine

Two command-line tools for sports photographers, both using Gemini to find the
action in each photo:

- **sports-reel** turns a folder of stills and a music track into a
  beat-synced, vertical (1080x1920) Final Cut Pro timeline, with a Ken Burns
  move on each photo, cross dissolves and an optional title.
- **match-cull** picks the best shots from a match, crops them and edits them
  into a new folder, leaving the originals untouched (see
  [Match Cull](#match-cull)).

## Setup

Requires macOS, [uv](https://docs.astral.sh/uv/), and ffmpeg (only needed for
`.m4a` / `.aac` music).

```bash
brew install ffmpeg
uv sync
```

Both tools use the Gemini API. Create a key at
<https://aistudio.google.com/apikey> and export it:

```bash
export GEMINI_API_KEY="your-key-here"
```

Add that line to `~/.zshrc` to keep it across terminal sessions.

## Sports Reel

```bash
uv run sports-reel ~/Movies/match
```

Point it at one folder holding the photos (JPEG, PNG or HEIC) and a single
music track. It writes
`AI_Sports_Reel.fcpxml` into that folder.

| Flag | Meaning |
|---|---|
| `--audio` | Music track (wav, mp3, flac, m4a, aac) when it is not in the photo folder, or the folder holds several. |
| `--photos` | Same as giving the folder on its own. Photos are used in filename order. |
| `--out` | Output path. Default: `<photos>/AI_Sports_Reel.fcpxml`. |
| `--fps` | `29.97` (default) or `60`. |
| `--min-duration` | Shortest allowed clip in seconds. Default `0.8`. |
| `--transition` | Cross dissolve length in seconds on every cut. Default `0.5`; `0` for hard cuts. |
| `--title` | Opening title text, e.g. `--title "MATCH DAY"`. Default: no title. |
| `--dry-run` | Analyze audio and photos and print the cut list; write no timeline. |
| `--refresh` | Ask Gemini about every photo again, ignoring remembered detections. |
| `--model` | Gemini model. Default `gemini-3.8-flash`. |
| `--no-vision` | Skip Gemini and center-crop every photo. No API key needed. |

To check the Final Cut import before spending API calls, run once with
`--no-vision`.

## How the timeline is built

- **Cuts** land on detected beats, never closer together than `--min-duration`.
  If the track has no detectable beats, or the beats stop early, cuts continue
  every 1.25s until the music ends.
- **Photos beyond the end of the music are dropped**, with a warning. Fewer
  photos than beats just makes a shorter reel.
- **Framing**: Gemini returns a box around the action and a motion type.
  `push_in` zooms from the full 9:16 crop to 85%; `pull_out` does the reverse.
  The subject is kept between 20% and 70% of the frame height, clear of
  Instagram / TikTok overlays.
- **Failures**: rate limits and outages (429 / 500 / 503) are retried three times
  with backoff. A photo that still fails gets a center crop instead of being
  dropped.
- **Reruns**: successful detections are remembered in a hidden file in the
  photo folder (`.sports_reel_detections.json`). Running the command again only
  asks Gemini about photos that are new, changed, or failed last time.

## Importing into Final Cut Pro

1. Open Final Cut Pro and select the library to import into.
2. Choose **File > Import > XML...** and pick `AI_Sports_Reel.fcpxml`.
3. Open the **AI Sports Reels** event and the **Soccer_Autocut_Reel** project.

The file points at your photos and music by absolute path, so leave them where
they were when you ran the command. Moving them means relinking in Final Cut.

**Keep media out of Downloads, Desktop and Documents.** macOS blocks Final Cut
Pro from reading those folders unless you grant it access, and Final Cut has
crashed on import when it could not open the referenced files. Put photos and
music under `~/Movies` or `~/Pictures`, or allow Final Cut Pro under System
Settings > Privacy & Security > Files and Folders. The command warns when it
sees media in one of those folders.

## Match Cull

Picks the best shots from a match, crops them, enhances them and applies a
look. Your originals are never changed or deleted.

```bash
uv run match-cull ~/Movies/"Saturday Match"
```

It asks two questions, then rates every photo with Gemini:

1. **What the photos are for**: Instagram post (4:5, 1080x1350), Reel/Story
   (9:16, 1080x1920), YouTube / full size (16:9), original size (crops and
   zooms in, but the file keeps the original's pixel dimensions), or best crop
   (Gemini chooses the shape and framing for each photo).
2. **How many to keep**: a number (`30`), a percentage (`25%`), or `all`.
   `100%` keeps every distinct, usable moment; `all` edits literally every
   photo, including burst duplicates and low scores.

Skip the questions with `--size instagram|story|youtube|original|ai` and
`--keep 30` (or `--keep all`).

Before starting, it shows an estimate and asks to confirm:

```
Estimate:
  Photos:  74 found, 0 already rated, 74 to rate with Gemini
  Tokens:  about 236,800 (74 ratings x ~3,200)
  Rating:  6 min to 21 min (depends on how busy Gemini is)
  Editing: about 1 min for up to 60 keepers x 4 versions (O, A, B, C)
  Total:   about 7 min to 22 min
```

Gemini tokens depend on how many photos still need rating, not on how many
you keep: every photo has to be rated to find the best ones. The keep amount
changes the editing time. `--estimate` shows this and stops without using any
tokens; `--yes` starts without the confirmation. At the end it reports the
time taken and the tokens actually used.

How it decides and edits:

- **Rating**: Gemini scores each photo 1-10 for a soccer highlight gallery
  (peak action, sharp focus, faces, composition) and explains why.
- **Bursts**: near-identical frames shot in quick succession count as one
  moment; only the best frame of each is kept.
- **Keeping**: the top-scoring moments, up to the number you asked for. Photos
  scored 3 or lower are never kept.
- **Four files for each keeper**, side by side in one folder:
  - `_O`: the original, cropped only (same crop as A). No color, tone,
    clarity or sharpening changes.
  - `_A`: tight crop on the action (or Gemini's own framing with best crop),
    with a natural edit.
  - `_B`: the widest crop (the full frame with best crop), same natural edit
    with 10% less contrast, for a softer alternative.
  - `_C`: A's crop with a punchier natural edit: vibrance and extra contrast in
    the bright tones (shadows untouched).
  No LUTs are used unless you add `--luts`; then A and B get the two looks
  Gemini picks from your LUT folder and C is the natural edit.
- **Enhancement**: each crop gets OpenCV gray-world white balance (capped, so
  grass-filled frames do not turn green), a gentle highlight stretch, and OpenCV
  CLAHE local contrast. Exposure is left as shot when it is reasonable and only
  corrected when a photo is genuinely too dark or too bright. Players who are
  darker than the scene (darker skin, navy kits) get a soft local lift through
  a feathered mask, like a Lightroom subject mask, instead of the whole photo
  being brightened for them. No step may push shadows toward black.
- **Clarity**: OpenCV's edge-preserving detail enhancement crisps faces,
  jerseys and the ball on every edit (A, B and C) without halos.
- **Look strength** (with `--luts`): a LUT that would turn grass yellow, teal,
  dull or neon is automatically applied at reduced strength.
- **Small photos**: crops never zoom below 1080 pixels on the short side, clarity
  is gentler on small crops, and photos under 4 MP (phone, WhatsApp or
  screenshot copies) are flagged so you can use the originals.
- **Your look**: run once on a folder of your favorite photos:

  ```bash
  uv run match-cull ~/Downloads/"Gold NAC" --learn-style --best "my best shot.heic"
  ```

  The tool measures their brightness, contrast, color, warmth and highlights,
  and asks Gemini to describe your taste in moments and framing (one call,
  about 15,000 tokens). `--best` (repeatable) names your very best photos:
  together they make up half of the learned look and Gemini weighs them most.
  The look is saved to `~/.config/match-cull/style.json`. Every run then gives
  edits a light nudge toward it (color, warmth, highlights) and rates new photos
  with your taste in mind; photos rated earlier need `--refresh` to be re-rated.
  `--no-style` turns it off for a run.
- **Looks (optional)**: with `--luts`, chosen from the `.cube` LUTs in Final
  Cut's Custom LUTs folder (your Sports LUT Pack); `--luts <folder>` for a
  different set.

Everything goes into a new folder inside the photo folder, such as
`Culled - Instagram post` or `Culled - Best crop` (or `--out <folder>`),
together with `cull_report.csv` listing every photo's score, reason, whether it
was kept, its files and any note (such as low resolution). Reads JPEG, HEIC, PNG and
camera RAW (Sony `.ARW` and others). Ratings are remembered (in a hidden
`.match_cull_analysis.json` in the photo folder), so rerunning
(for example with a different size) only asks Gemini about new or failed
photos; `--refresh` redoes them all.

## Development

```bash
uv run pytest
```

The tests never call Gemini (it is replaced with a fake) and never touch your
learned look. They validate generated reel XML against the FCPXML 1.11 DTD that
ships inside Final Cut Pro, when it is installed.

```text
match_cull/
  cli.py        Questions, rating, editing, report
  analysis.py   Gemini scoring, crop and look choice
  select.py     Bursts and keeper selection
  crops.py      Output sizes and crop rectangles
  enhance.py    Exposure, white balance, local contrast, subject lift, clarity
  style.py      Learning your look and nudging edits toward it
  estimate.py   Time and token estimate before a run
  looks.py      .cube LUT reading and applying
  images.py     Loading (incl. RAW, HEIC), hashing, saving
sports_reel/
  cli.py        Entrypoint and flags
  audio.py      Beat tracking and cut planning
  vision.py     Gemini subject detection
  geometry.py   9:16 safe-zone crop math
  fcpxml.py     FCPXML 1.11 writer
```
