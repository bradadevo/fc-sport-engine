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

    def lines(self) -> list[str]:
        lo, hi = self.rate_seconds
        rows = [
            f"  Photos:  {self.total} found, {self.remembered} already rated, {self.to_rate} to rate with Gemini",
            f"  Tokens:  about {self.tokens:,} ({self.to_rate} ratings x ~{self.tokens // max(self.to_rate, 1):,})"
            if self.to_rate else "  Tokens:  none (every photo is already rated)",
            f"  Rating:  {format_duration(lo)} to {format_duration(hi)} (depends on how busy Gemini is)"
            if self.to_rate else "  Rating:  instant",
            f"  Editing: about {format_duration(self.edit_seconds)} for up to {self.keep} keepers x 4 versions (O, A, B, C)",
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
             tokens_per_photo: float | None = None) -> Estimate:
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
        edit_seconds=keep * avg_mp * EDIT_SECONDS_PER_MEGAPIXEL,
    )
