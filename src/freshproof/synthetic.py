"""Synthetic produce photos for smoke testing the pipeline.

These are drawings, not real produce. Use them to check that loading, splitting,
training, calibration and the API all work end to end before real photos exist.
Never report results on synthetic data as results of the real system.

    PYTHONPATH=src python -m freshproof.synthetic --out data_synth --items 8 --days 6
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data.schema import Grade, LightCondition

PRODUCE_COLORS = {
    "tomato": (200, 40, 40),
    "banana": (230, 200, 50),
    "potato": (160, 120, 80),
}
DEFECT_COLOR = (60, 35, 20)
LIGHT_FACTOR = {
    LightCondition.DAYLIGHT: 1.0,
    LightCondition.SHADE: 0.8,
    LightCondition.TUBE_LIGHT: 0.95,
    LightCondition.SHOP_LIGHT: 1.1,
    LightCondition.EVENING: 0.7,
}
COLUMNS = [
    "photo_id", "produce_type", "grade", "light", "phone_model",
    "item_id", "defect_mask_path", "days_until_spoiled", "labeler",
]


def grade_from_fraction(fraction: float) -> Grade:
    if fraction <= 0:
        return Grade.A
    if fraction < 0.05:
        return Grade.B
    if fraction < 0.15:
        return Grade.C
    return Grade.REJECT


def _ellipse_mask(size: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    return np.asarray(img) > 0


def _render_item(rng, size, produce_type, n_days, light):
    """Yield (image, mask or None, defect_fraction) for each day of one item."""
    cx = size * rng.uniform(0.45, 0.55)
    cy = size * rng.uniform(0.45, 0.55)
    rx = size * rng.uniform(0.28, 0.38)
    ry = size * rng.uniform(0.28, 0.38)
    produce_mask = _ellipse_mask(size, cx, cy, rx, ry)

    background = rng.integers(170, 215, size=3)
    blobs = [
        (
            cx + rng.uniform(-0.5, 0.5) * rx,
            cy + rng.uniform(-0.5, 0.5) * ry,
            rng.uniform(0.25, 0.5) * size * 0.3,
        )
        for _ in range(rng.integers(1, 4))
    ]
    base_color = np.array(PRODUCE_COLORS[produce_type]) + rng.integers(-15, 16, size=3)

    for day in range(n_days):
        progress = day / max(n_days - 1, 1)
        image = np.tile(background, (size, size, 1)).astype(float)
        image[produce_mask] = base_color

        defect = np.zeros((size, size), dtype=bool)
        if progress > 0:
            for bx, by, br in blobs:
                radius = br * progress
                defect |= _ellipse_mask(size, bx, by, radius, radius)
            defect &= produce_mask
        image[defect] = DEFECT_COLOR

        image = image * LIGHT_FACTOR[light] + rng.normal(0, 4, image.shape)
        image = np.clip(image, 0, 255).astype(np.uint8)
        fraction = defect.sum() / produce_mask.sum()
        yield Image.fromarray(image), (defect if defect.any() else None), float(fraction)


def generate_dataset(
    out_dir: str | Path,
    items_per_type: int = 8,
    days: int = 6,
    size: int = 128,
    seed: int = 0,
    produce_types: tuple[str, ...] = tuple(PRODUCE_COLORS),
) -> Path:
    """Write raw/, masks/ and labels.csv under out_dir and return labels.csv."""
    out = Path(out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lights = list(LIGHT_FACTOR)

    rows = []
    for produce in produce_types:
        for item_index in range(items_per_type):
            item_id = f"{produce}_{item_index:03d}"
            light = lights[int(rng.integers(len(lights)))]
            frames = _render_item(rng, size, produce, days, light)
            for day, (image, mask, fraction) in enumerate(frames):
                photo_id = f"{item_id}_d{day}"
                image.save(out / "raw" / f"{photo_id}.jpg", quality=92)
                mask_path = ""
                if mask is not None:
                    Image.fromarray((mask * 255).astype(np.uint8)).save(
                        out / "masks" / f"{photo_id}.png"
                    )
                    mask_path = f"masks/{photo_id}.png"
                rows.append(
                    {
                        "photo_id": photo_id,
                        "produce_type": produce,
                        "grade": grade_from_fraction(fraction).value,
                        "light": light.value,
                        "phone_model": "synthetic",
                        "item_id": item_id,
                        "defect_mask_path": mask_path,
                        "days_until_spoiled": days - 1 - day,
                        "labeler": "synthetic",
                    }
                )

    labels_path = out / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data_synth")
    parser.add_argument("--items", type=int, default=8, help="items per produce type")
    parser.add_argument("--days", type=int, default=6)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = generate_dataset(args.out, args.items, args.days, args.size, args.seed)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
