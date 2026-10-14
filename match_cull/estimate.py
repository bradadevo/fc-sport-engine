"""Up-front estimate of how long a run takes and how many Gemini tokens it uses.

Numbers measured on 2026-10-14 with gemini-3.8-flash; the tool replaces the
token figure with the average of this folder's own past ratings when it has
them.
"""

from dataclasses import dataclass

from PIL import Image
import rawpy

# Tokens per photo rating (image + prompt + the model's thinking + answer)
DEFAULT_TOKENS_PER_PHOTO = 3200
# Tokens for the one-time 'learn my style' call (10 sample photos + answer)
TASTE_TOKENS = 15000
# Tokens and seconds for version D's suggested adjustments, one call per keeper
ADJUST_TOKENS = 2500
# Seconds per Gemini rating: when it is quiet, and when it is busy and retrying
RATE_SECONDS = (5, 17)
# Seconds to make all versions of a keeper (O, A, B, C), per megapixel of the original
EDIT_SECONDS_PER_MEGAPIXEL = 1.0


@dataclass
class Estimate:
    total: int
    remembered: int
    to_rate: int
    keep: int
    tokens: int
    rate_seconds: tuple[float, float]
    edit_seconds: float
    adjust_needed: int = 0  # keepers that still need D suggestions (upper bound)

    def lines(self) -> list[str]:
        lo, hi = self.rate_seconds
        d_tokens = self.adjust_needed * ADJUST_TOKENS
        lo, hi = lo + self.adjust_needed * RATE_SECONDS[0], hi + self.adjust_needed * RATE_SECONDS[1]
        rating_tokens = (f"{self.to_rate} ratings x ~{self.tokens // max(self.to_rate, 1):,}"
                         if self.to_rate else "")
        d_part = f"up to {self.adjust_needed} D suggestions x ~{ADJUST_TOKENS:,}" if self.adjust_needed else ""
        parts = " + ".join(p for p in (rating_tokens, d_part) if p)
        rows = [
            f"  Photos:  {self.total} found, {self.remembered} already rated, {self.to_rate} to rate with Gemini",
            f"  Tokens:  about {self.tokens + d_tokens:,} ({parts})"
            if parts else "  Tokens:  none (every photo is already rated and suggested)",
            f"  Gemini:  {format_duration(lo)} to {format_duration(hi)} (depends on how busy Gemini is)"
            if parts else "  Gemini:  instant",
            f"  Editing: about {format_duration(self.edit_seconds)} for up to {self.keep} keepers x 5 versions (O, A, B, C, D)",
            f"  Total:   about {format_duration(lo + self.edit_seconds)} to {format_duration(hi + self.edit_seconds)}",
        ]
        return rows


def megapixels(path: str) -> float:
    """Pixel count of a photo from its header, without decoding it."""
    try:
        with Image.open(path) as img:
            w, h = img.size
    except Exception:
        try:
            with rawpy.imread(path) as raw:
                w, h = raw.sizes.width, raw.sizes.height
        except Exception:
            return 0.0
    return w * h / 1e6


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    if seconds < 60:
        return f"{seconds} sec"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def estimate(photo_paths: list[str], remembered: set[str], keep: int,
             tokens_per_photo: float | None = None, adjust_needed: int = 0) -> Estimate:
    """keep is the number of keepers asked for (already capped to the photo
    count). Rating cost depends on the photos not yet rated, whatever the keep
    amount; editing time depends on the keepers' size."""
    to_rate = [p for p in photo_paths if p not in remembered]
    per_photo = tokens_per_photo or DEFAULT_TOKENS_PER_PHOTO
    sizes = [megapixels(p) for p in photo_paths]
    avg_mp = sum(sizes) / len(sizes) if sizes else 0.0
    return Estimate(
        total=len(photo_paths),
        remembered=len(photo_paths) - len(to_rate),
        to_rate=len(to_rate),
        keep=keep,
        tokens=round(len(to_rate) * per_photo),
        rate_seconds=(len(to_rate) * RATE_SECONDS[0], len(to_rate) * RATE_SECONDS[1]),
        edit_seconds=keep * avg_mp * EDIT_SECONDS_PER_MEGAPIXEL * 1.25,
        adjust_needed=adjust_needed,
    )
