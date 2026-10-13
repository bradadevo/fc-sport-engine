# Sports Reel Engine

Turns a folder of sports stills and a music track into a beat-synced, vertical
(1080x1920) Final Cut Pro timeline. Each photo gets a Ken Burns move aimed at
the action, found by Gemini.

## Setup

Requires macOS, [uv](https://docs.astral.sh/uv/), and ffmpeg (only needed for
`.m4a` / `.aac` music).

```bash
brew install ffmpeg
uv sync
```

Subject detection uses the Gemini API. Create a key at
<https://aistudio.google.com/apikey> and export it:

```bash
export GEMINI_API_KEY="your-key-here"
```

Add that line to `~/.zshrc` to keep it across terminal sessions.

## Usage

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

## Development

```bash
uv run pytest
```

The tests validate generated XML against the FCPXML 1.11 DTD that ships inside
Final Cut Pro, when it is installed.

```text
sports_reel/
  cli.py        Entrypoint and flags
  audio.py      Beat tracking and cut planning
  vision.py     Gemini subject detection
  geometry.py   9:16 safe-zone crop math
  fcpxml.py     FCPXML 1.11 writer
```
