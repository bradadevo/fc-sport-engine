"""Compares Gemini models on the photographer's taste: do they score the
photographer's favorite photos above ordinary match photos? Uses match-cull's
production rating prompt (with the learned taste notes, no LUTs).

    uv run python studies/2026-10-14-gemini-model-comparison/run_comparison.py set.json out.json

set.json: {"gold": [paths to favorite photos], "ordinary": [paths to other photos]}.
All models run in parallel; each rates every photo once, plus 4 repeats to
measure consistency. Needs GEMINI_API_KEY."""

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from match_cull import analysis, images, style
from sports_reel import vision

SET_FILE, OUT_FILE = Path(sys.argv[1]), Path(sys.argv[2])
MODELS = ["gemini-3.8-flash", "gemini-pro-latest", "gemini-3.1-pro-preview", "gemini-4-argon"]
photo_set = json.loads(SET_FILE.read_text())
jobs = [("gold", p) for p in photo_set["gold"]] + [("ordinary", p) for p in photo_set["ordinary"]]
# Repeat 4 photos to measure consistency
jobs += [("gold", photo_set["gold"][0]), ("gold", photo_set["gold"][1]),
         ("ordinary", photo_set["ordinary"][0]), ("ordinary", photo_set["ordinary"][5])]

taste = (style.load() or style.Style("", 0, 0, 0, 0, 0, 0, 0)).notes
prompt = analysis.build_prompt({}, taste)
jpegs = {p: vision.image_to_jpeg(images.load_image(p)) for _, p in jobs}
lock = threading.Lock()
results = {m: [] for m in MODELS}


def save():
    OUT_FILE.write_text(json.dumps(results, indent=1))


def run_model(model):
    client = vision.create_client()
    for group, path in jobs:
        usage: list[int] = []
        started = time.monotonic()
        row = {"group": group, "photo": Path(path).name}
        try:
            a = vision.generate_structured(client, jpegs[path], prompt, analysis.PhotoAssessment,
                                           model=model, usage=usage)
            row.update(score=max(1, min(10, a.score)), reason=a.reason,
                       crop=[a.crop_xmin, a.crop_ymin, a.crop_xmax, a.crop_ymax], shape=a.crop_shape,
                       subject=[a.xmin, a.ymin, a.xmax, a.ymax])
        except Exception as e:
            row["error"] = str(e)[:200]
        row["seconds"] = round(time.monotonic() - started, 1)
        row["tokens"] = sum(usage)
        with lock:
            results[model].append(row)
            save()
            print(f"{model:24} {row['photo'][:30]:30} {row.get('score', 'ERR')!s:>3} "
                  f"{row['seconds']:5.1f}s {row['tokens']:>6} tok", flush=True)


with ThreadPoolExecutor(len(MODELS)) as pool:
    list(pool.map(run_model, MODELS))
print("done", flush=True)
