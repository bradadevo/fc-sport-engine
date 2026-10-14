import math
import re
from dataclasses import dataclass

from match_cull.images import hash_distance

# Frames this close in time are the same burst whatever they look like
BURST_SECONDS = 0.7
# Frames up to this far apart in time are a burst if they also look alike
BURST_SIMILAR_SECONDS = 3.0
# How alike two frames must look (bits differing out of 64)
BURST_HASH_DISTANCE = 12
# Photos scored this low are never kept, even to fill the requested count
MIN_KEEP_SCORE = 4


@dataclass
class Candidate:
    path: str
    score: int
    sharpness: float
    hash: int
    time: float | None
    burst: int = 0


# parse_keep's answer for "edit every photo": no burst merging, no score floor
KEEP_ALL = -1


def parse_keep(text: str, total: int) -> int:
    """'30' -> 30 photos, '25%' -> a quarter of them (rounded up), 'all' ->
    KEEP_ALL (every photo, including duplicates and rejects)."""
    text = text.strip()
    if text.lower() == "all":
        return KEEP_ALL
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        pct = float(match.group(1))
        if not 0 < pct <= 100:
            raise ValueError("percentage must be between 0 and 100")
        return max(1, math.ceil(total * pct / 100))
    if text.isdigit() and int(text) > 0:
        return int(text)
    raise ValueError("enter a number like 30, a percentage like 25%, or all")


def same_burst(a: Candidate, b: Candidate) -> bool:
    if a.time is not None and b.time is not None:
        gap = abs(b.time - a.time)
        if gap <= BURST_SECONDS:
            return True
        if gap > BURST_SIMILAR_SECONDS:
            return False
    return hash_distance(a.hash, b.hash) <= BURST_HASH_DISTANCE


def group_bursts(candidates: list[Candidate]) -> None:
    """Numbers bursts of consecutive near-identical frames (in name order)."""
    burst = 0
    for i, cand in enumerate(candidates):
        if i and not same_burst(candidates[i - 1], cand):
            burst += 1
        cand.burst = burst


def choose_keepers(candidates: list[Candidate], keep: int) -> list[Candidate]:
    """Best frame from each burst, then the top `keep` by score (sharper
    wins ties), returned in camera order. KEEP_ALL returns every photo."""
    group_bursts(candidates)
    if keep == KEEP_ALL:
        return list(candidates)
    best_of_burst: dict[int, Candidate] = {}
    for cand in candidates:
        current = best_of_burst.get(cand.burst)
        if current is None or (cand.score, cand.sharpness) > (current.score, current.sharpness):
            best_of_burst[cand.burst] = cand
    eligible = [c for c in best_of_burst.values() if c.score >= MIN_KEEP_SCORE]
    ranked = sorted(eligible, key=lambda c: (c.score, c.sharpness), reverse=True)[:keep]
    order = {c.path: i for i, c in enumerate(candidates)}
    return sorted(ranked, key=lambda c: order[c.path])
