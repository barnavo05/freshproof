"""Finding and opening photos, with phone rotation handled.

Phones often store a photo sideways and add a hidden rotation flag (EXIF).
Browsers and viewers apply that flag, so you see the photo upright. Code that
ignores it sees a rotated image, and defect masks drawn on the upright photo
would no longer line up. Everything in FreshProof opens photos through
load_oriented so masks and images always agree.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
_EXIF_ORIENTATION = 0x0112
_SWAPS_SIDES = {5, 6, 7, 8}


def list_photos(raw_dir: str | Path) -> dict[str, Path]:
    """Map photo_id (file name without extension) to its path, sorted by id."""
    raw = Path(raw_dir)
    if not raw.is_dir():
        return {}
    found: dict[str, Path] = {}
    for ext in IMAGE_EXTENSIONS:
        for path in sorted(raw.iterdir()):
            if path.is_file() and path.suffix.lower() == ext:
                found.setdefault(path.stem, path)
    return dict(sorted(found.items()))


def find_image(raw_dir: str | Path, photo_id: str) -> Path:
    photos = list_photos(raw_dir)
    if photo_id not in photos:
        raise FileNotFoundError(f"no image found for photo_id {photo_id} in {raw_dir}")
    return photos[photo_id]


def load_oriented(path: str | Path) -> Image.Image:
    """Open a photo upright, as RGB."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def oriented_size(path: str | Path) -> tuple[int, int]:
    """(width, height) of the photo as it looks upright, without decoding pixels."""
    with Image.open(path) as img:
        width, height = img.size
        orientation = img.getexif().get(_EXIF_ORIENTATION, 1)
    return (height, width) if orientation in _SWAPS_SIDES else (width, height)
