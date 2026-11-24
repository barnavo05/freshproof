"""Evidence metrics: does the highlighted region match the real defect?

A saliency map (for example Grad-CAM) is often shown as proof that a model
looked at the right place. These functions measure that claim against a
ground truth defect mask instead of trusting it.

All functions take two arrays of the same 2D shape:
    saliency: non-negative floats, any scale
    mask:     binary defect mask (True or 1 where the defect is)
"""

from __future__ import annotations

import numpy as np


def _check(saliency: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    saliency = np.asarray(saliency, dtype=float)
    mask = np.asarray(mask).astype(bool)
    if saliency.ndim != 2 or mask.ndim != 2:
        raise ValueError("saliency and mask must be 2D arrays")
    if saliency.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: saliency {saliency.shape} vs mask {mask.shape}"
        )
    if np.any(saliency < 0):
        raise ValueError("saliency must be non-negative")
    return saliency, mask


def energy_in_mask(saliency: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of total saliency that falls inside the defect mask (0 to 1).

    Returns nan when the mask is empty, since there is no defect to find.
    Returns 0.0 when the saliency map is all zeros.
    """
    saliency, mask = _check(saliency, mask)
    if not mask.any():
        return float("nan")
    total = saliency.sum()
    if total == 0:
        return 0.0
    return float(saliency[mask].sum() / total)


def pointing_game(saliency: np.ndarray, mask: np.ndarray) -> float:
    """1.0 if the single strongest saliency pixel lies inside the mask, else 0.0.

    Returns nan when the mask is empty.
    """
    saliency, mask = _check(saliency, mask)
    if not mask.any():
        return float("nan")
    idx = np.unravel_index(np.argmax(saliency), saliency.shape)
    return float(mask[idx])


def iou_at_threshold(
    saliency: np.ndarray, mask: np.ndarray, threshold: float = 0.5
) -> float:
    """Intersection over union after binarizing the saliency map.

    The map is first scaled to 0..1 by its maximum, then cut at `threshold`.
    Returns nan when the mask is empty and 0.0 when nothing passes the threshold.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    saliency, mask = _check(saliency, mask)
    if not mask.any():
        return float("nan")
    peak = saliency.max()
    if peak == 0:
        return 0.0
    binary = (saliency / peak) >= threshold
    union = np.logical_or(binary, mask).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(binary, mask).sum() / union)


def evidence_report(
    saliency: np.ndarray, mask: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """All evidence metrics for one image in a single dict."""
    return {
        "energy_in_mask": energy_in_mask(saliency, mask),
        "pointing_game": pointing_game(saliency, mask),
        "iou": iou_at_threshold(saliency, mask, threshold),
    }
