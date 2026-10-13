import io
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from google import genai
from google.genai import errors, types

DEFAULT_MODEL = "gemini-3.8-flash"
MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = (429, 500, 503)
REQUEST_TIMEOUT_MS = 60_000
# Longest edge sent to the model; full-resolution DSLR files are needlessly large
INFERENCE_MAX_EDGE = 1536
CACHE_FILENAME = ".sports_reel_detections.json"

PROMPT = """
Locate the primary athletic focal point in this soccer match photograph:
- Target the player on the ball, the point of contact on a strike, a tackle, or emotional celebration.
- Return precise normalized bounding box coordinates (0.0 to 1.0).
- Classify motion_type: 'push_in' for shots/strikes/celebrations, 'pull_out' for counter-attacks/runs.
"""


class SoccerSubjectDetection(BaseModel):
    subject_description: str = Field(description="Description of action (e.g., striker shot, diving save, celebration)")
    ymin: float = Field(description="Normalized top bounding edge 0.0-1.0")
    xmin: float = Field(description="Normalized left bounding edge 0.0-1.0")
    ymax: float = Field(description="Normalized bottom bounding edge 0.0-1.0")
    xmax: float = Field(description="Normalized right bounding edge 0.0-1.0")
    motion_type: str = Field(description="'push_in' for strikes/celebrations, 'pull_out' for open runs")


def center_detection() -> SoccerSubjectDetection:
    """Standard center crop, used when a photo cannot be analyzed."""
    return SoccerSubjectDetection(
        subject_description="center crop (fallback)",
        ymin=0.25, xmin=0.25, ymax=0.75, xmax=0.75,
        motion_type="push_in",
    )


def create_client() -> genai.Client:
    """Creates a Gemini client; the SDK reads GEMINI_API_KEY from the environment."""
    return genai.Client(http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))


def _inference_jpeg(image_path: str) -> bytes:
    """Auto-rotates per EXIF and downsizes, so the model sees the photo
    upright and its coordinates match the displayed image."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((INFERENCE_MAX_EDGE, INFERENCE_MAX_EDGE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def analyze_photo(
    client: genai.Client,
    image_path: str,
    model: str = DEFAULT_MODEL,
    sleep=time.sleep,
) -> SoccerSubjectDetection:
    """Asks Gemini for the subject box, retrying rate limits and outages
    (429 / 500 / 503) with exponential backoff. Raises once retries are spent."""
    image_bytes = _inference_jpeg(image_path)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    PROMPT,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SoccerSubjectDetection,
                    temperature=0.1,
                ),
            )
            return SoccerSubjectDetection.model_validate_json(response.text)
        except errors.APIError as e:
            if e.code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES:
                raise
            sleep(2 ** attempt)


def analyze_photo_or_center(
    client: genai.Client | None,
    image_path: str,
    model: str = DEFAULT_MODEL,
) -> tuple[SoccerSubjectDetection, str | None]:
    """Returns (detection, error). On any failure the detection is a center
    crop and error says why; with no client it is a center crop with no error."""
    if client is None:
        return center_detection(), None
    try:
        return analyze_photo(client, image_path, model=model), None
    except Exception as e:
        return center_detection(), str(e)


class DetectionCache:
    """Remembers successful detections in a hidden file in the photo folder,
    so a rerun only asks Gemini about photos that are new, changed, or failed
    last time. Center-crop fallbacks are never stored."""

    def __init__(self, photos_dir: str):
        self.path = Path(photos_dir) / CACHE_FILENAME
        self.entries: dict[str, dict] = {}
        try:
            self.entries = json.loads(self.path.read_text())
        except (OSError, ValueError):
            pass  # no cache yet, or an unreadable one: start fresh

    @staticmethod
    def _fingerprint(image_path: str) -> list[int]:
        stat = os.stat(image_path)
        return [stat.st_size, stat.st_mtime_ns]

    def get(self, image_path: str) -> SoccerSubjectDetection | None:
        entry = self.entries.get(os.path.basename(image_path))
        if not entry or entry.get("fingerprint") != self._fingerprint(image_path):
            return None
        try:
            return SoccerSubjectDetection.model_validate(entry["detection"])
        except Exception:
            return None

    def put(self, image_path: str, detection: SoccerSubjectDetection) -> None:
        """Stores a detection and saves straight away, so progress survives
        an interrupted run."""
        self.entries[os.path.basename(image_path)] = {
            "fingerprint": self._fingerprint(image_path),
            "detection": detection.model_dump(),
        }
        try:
            self.path.write_text(json.dumps(self.entries, indent=1))
        except OSError:
            pass  # a read-only folder just means nothing is remembered
