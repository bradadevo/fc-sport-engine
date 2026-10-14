# Gemini model comparison for match-cull (2026-10-14)

**Question:** is there a Gemini model that rates and crops match photos better
than `gemini-3.8-flash`, the default?

Only the rating and cropping depend on the model. All editing (exposure, white
balance, clarity, the A/B/C versions) is done by match-cull's own code and is
identical whichever model is used.

## Method

- **Photos:** 8 of the photographer's favorites (from their "Gold NAC"
  reference folder, including their two best) and 8 ordinary match photos
  (WhatsApp copies and camera files from two other shoots). File names are in
  [`photo_set.json`](photo_set.json); the photos themselves are not in the repo.
- **Prompt:** match-cull's production rating prompt, including the taste notes
  learned from the reference folder, with no LUTs.
- **Models:** `gemini-3.8-flash` (baseline), `gemini-pro-latest`,
  `gemini-3.1-pro-preview`, `gemini-4-argon`, all run in parallel.
- **Measures:**
  - *Ranking:* pick one favorite and one ordinary photo at random; how often
    does the model score the favorite higher? (50% = coin flip, 100% = perfect;
    ties count half.)
  - *Consistency:* 4 photos were rated twice; average score difference.
  - *Speed and tokens* per photo, as reported by the API.
  - *Crops:* compared by eye on the two best photos and one ordinary photo.

Script: [`run_comparison.py`](run_comparison.py). Per-photo results:
[`results.json`](results.json) (scores, crops, timings and tokens; Gemini's
free-text reasons are left out because they describe the children in the
photos). Totals: [`summary.json`](summary.json).

## Results

| Model | Ranking | Favorites (avg) | Ordinary (avg) | Gap | Repeat difference | Seconds per photo | Tokens per photo |
|---|---|---|---|---|---|---|---|
| **gemini-3.8-flash** (default) | 84% | 6.5 | 4.5 | 2.0 | **0.00** | 9.8 | 2,870 |
| gemini-pro-latest | 82% | 6.8 | 4.4 | 2.4 | 0.25 | 17.2 | 3,608 |
| **gemini-3.1-pro-preview** | **90%** | **7.4** | 4.4 | **3.0** | 0.25 | 17.8 | 3,768 |
| gemini-4-argon | 82% | 6.8 | 5.0 | 1.8 | 0.50 | **5.8** | 2,715 |

- The photographer's two best photos: 7 and 7 from Flash, 8 and 9 from
  3.1-pro-preview.
- Crops were similar across models. On the second best photo (a player
  stretching for the ball), 3.1-pro-preview kept the full extension and the
  ball; 4-argon cropped the ball out.
- Total cost of the study: about 259,000 tokens.

## Caveats

- 16 photos is a small sample: 84% vs 90% is 4 of 64 favorite-vs-ordinary
  pairs.
- All models were given taste notes learned from the same favorites, which
  flatters every model equally.
- "Preview" models can change or be withdrawn without notice.
- Pro models cost more per token than Flash; account prices were not checked.

## Decision

Keep `gemini-3.8-flash` as the default: twice as fast, about 25% fewer tokens
at a lower per-token price, perfectly repeatable scores, and only slightly
behind on ranking. For an important shoot, use the Pro model with
`--model gemini-3.1-pro-preview`; no code change is needed.

Image-generation models (such as `gemini-3-pro-image`) were ruled out for
editing: they redraw photos, which can change faces and details. Version D
instead has Gemini *suggest* slider adjustments that match-cull applies
faithfully.
