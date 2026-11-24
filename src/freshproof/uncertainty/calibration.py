"""Calibration and abstention for the "check by hand" flag.

A model that says 90 percent should be right about 90 percent of the time.
These helpers measure that gap, fix it with temperature scaling, and pick a
confidence threshold below which the app asks for a manual check.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _as_pair(confidences, correct) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct).astype(bool)
    if conf.ndim != 1 or conf.shape != ok.shape:
        raise ValueError("confidences and correct must be 1D arrays of equal length")
    if conf.size == 0:
        raise ValueError("need at least one prediction")
    return conf, ok


def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    """Average gap between confidence and accuracy, weighted by bin size."""
    conf, ok = _as_pair(confidences, correct)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (conf > lo) & (conf <= hi)
        if i == 0:
            in_bin |= conf == 0.0
        if in_bin.any():
            ece += in_bin.mean() * abs(ok[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def negative_log_likelihood(logits, labels, temperature: float = 1.0) -> float:
    probs = softmax(logits, temperature)
    labels = np.asarray(labels, dtype=int)
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))


def fit_temperature(
    logits, labels, low: float = 0.05, high: float = 10.0, steps: int = 200
) -> float:
    """Grid search for the temperature that minimises validation NLL.

    Fit this on the validation set only, never on the holdout set.
    """
    grid = np.geomspace(low, high, steps)
    losses = [negative_log_likelihood(logits, labels, t) for t in grid]
    return float(grid[int(np.argmin(losses))])


def selective_accuracy(confidences, correct, threshold: float) -> tuple[float, float]:
    """(accuracy on accepted predictions, share of predictions accepted)."""
    conf, ok = _as_pair(confidences, correct)
    accepted = conf >= threshold
    if not accepted.any():
        return float("nan"), 0.0
    return float(ok[accepted].mean()), float(accepted.mean())


class AbstainChoice(NamedTuple):
    threshold: float
    accuracy: float
    coverage: float


def choose_abstain_threshold(
    confidences, correct, target_accuracy: float, min_coverage: float = 0.0
) -> Optional[AbstainChoice]:
    """Lowest confidence threshold whose accepted predictions reach the target.

    Lower threshold means the app answers more often. Returns None if no
    threshold reaches the target accuracy with at least min_coverage.
    """
    conf, ok = _as_pair(confidences, correct)
    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be between 0 and 1")
    for threshold in np.unique(conf):
        accuracy, coverage = selective_accuracy(conf, ok, threshold)
        if coverage >= min_coverage and accuracy >= target_accuracy:
            return AbstainChoice(float(threshold), accuracy, coverage)
    return None
