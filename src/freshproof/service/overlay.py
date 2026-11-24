"""Turn a heatmap into a picture a person can read: the photo with the evidence in red."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def render_overlay(image: Image.Image, heatmap: np.ndarray, strength: float = 0.6) -> bytes:
    """Return PNG bytes of the image with the heatmap tinted red on top."""
    heat = np.asarray(heatmap, dtype=float)
    if heat.ndim != 2:
        raise ValueError("heatmap must be 2D")
    peak = heat.max()
    if peak > 0:
        heat = heat / peak
    heat = np.clip(heat, 0.0, 1.0)

    base = image.convert("RGB")
    if heat.shape != (base.height, base.width):
        resized = Image.fromarray((heat * 255).astype(np.uint8)).resize(
            base.size, Image.BILINEAR
        )
        heat = np.asarray(resized, dtype=float) / 255.0

    pixels = np.asarray(base, dtype=float)
    alpha = (heat * strength)[..., None]
    red = np.array([255.0, 0.0, 0.0])
    blended = pixels * (1 - alpha) + red * alpha

    out = io.BytesIO()
    Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)).save(out, format="PNG")
    return out.getvalue()
