from datetime import datetime
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageOps

RAW_EXTENSIONS = (".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2")
PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".heif") + RAW_EXTENSIONS
JPEG_QUALITY = 92

_EXIF_ORIENTATION = 0x0112
_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 0x9003
_SUBSEC_ORIGINAL = 0x9291


def find_photos(folder: str) -> list[str]:
    """Photos directly inside the folder (not subfolders), sorted by name to
    keep camera order."""
    return sorted(
        str(p) for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in PHOTO_EXTENSIONS and not p.name.startswith(".")
    )


def load_image(path: str) -> Image.Image:
    """Opens any supported photo as an upright RGB image. Files Pillow cannot
    read (camera RAW, including RAW files saved with a .jpg name) are
    developed with the camera's white balance."""
    try:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        with rawpy.imread(path) as raw:
            return Image.fromarray(raw.postprocess(use_camera_wb=True, output_bps=8))


def exif_for_output(path: str) -> bytes | None:
    """The original EXIF (camera, lens, capture time) to copy onto edits, with
    orientation reset because the pixels are already upright."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            exif[_EXIF_ORIENTATION] = 1
            return exif.tobytes()
    except Exception:
        return None


def capture_time(path: str) -> float | None:
    """Capture time in seconds (with sub-second precision when the camera
    records it), or None if the file has no usable EXIF date."""
    try:
        with Image.open(path) as img:
            ifd = img.getexif().get_ifd(_EXIF_IFD)
    except Exception:
        return None
    stamp = ifd.get(_DATETIME_ORIGINAL)
    if not stamp:
        return None
    try:
        seconds = datetime.strptime(str(stamp).strip(), "%Y:%m:%d %H:%M:%S").timestamp()
    except ValueError:
        return None
    subsec = str(ifd.get(_SUBSEC_ORIGINAL, "")).strip()
    if subsec.isdigit():
        seconds += int(subsec) / (10 ** len(subsec))
    return seconds


def _gray(img: Image.Image, max_edge: int) -> np.ndarray:
    small = img.convert("L")
    small.thumbnail((max_edge, max_edge))
    return np.asarray(small, dtype=np.float32)


def sharpness(img: Image.Image) -> float:
    """Variance of the Laplacian on a downsized copy: higher means sharper.
    Used to break ties between frames the model scores the same."""
    g = _gray(img, 1024)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def dhash(img: Image.Image) -> int:
    """64-bit difference hash: near-identical frames get nearby hashes."""
    small = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hash_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def save_jpeg(img: Image.Image, path: str, exif: bytes | None = None) -> None:
    kwargs = {"quality": JPEG_QUALITY, "subsampling": 0}
    if exif:
        kwargs["exif"] = exif
    img.save(path, "JPEG", **kwargs)
