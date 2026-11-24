import math

import numpy as np
import pytest

from freshproof.evaluation import (
    energy_in_mask,
    evidence_report,
    iou_at_threshold,
    pointing_game,
)


def _mask():
    m = np.zeros((10, 10), dtype=bool)
    m[2:5, 2:5] = True
    return m


def test_perfect_saliency_scores_one():
    mask = _mask()
    sal = mask.astype(float)
    assert energy_in_mask(sal, mask) == pytest.approx(1.0)
    assert pointing_game(sal, mask) == 1.0
    assert iou_at_threshold(sal, mask) == pytest.approx(1.0)


def test_saliency_far_from_defect_scores_zero():
    mask = _mask()
    sal = np.zeros((10, 10))
    sal[8:10, 8:10] = 1.0
    assert energy_in_mask(sal, mask) == 0.0
    assert pointing_game(sal, mask) == 0.0
    assert iou_at_threshold(sal, mask) == 0.0


def test_partial_overlap_energy():
    mask = _mask()
    sal = np.zeros((10, 10))
    sal[2:5, 2:5] = 1.0  # 9 pixels inside
    sal[7:10, 7:10] = 1.0  # 9 pixels outside
    assert energy_in_mask(sal, mask) == pytest.approx(0.5)


def test_peak_inside_mask_but_energy_mostly_outside():
    mask = _mask()
    sal = np.full((10, 10), 0.3)
    sal[3, 3] = 1.0
    assert pointing_game(sal, mask) == 1.0
    assert energy_in_mask(sal, mask) < 0.2


def test_empty_mask_returns_nan():
    sal = np.ones((10, 10))
    mask = np.zeros((10, 10), dtype=bool)
    assert math.isnan(energy_in_mask(sal, mask))
    assert math.isnan(pointing_game(sal, mask))
    assert math.isnan(iou_at_threshold(sal, mask))


def test_all_zero_saliency_scores_zero():
    mask = _mask()
    sal = np.zeros((10, 10))
    assert energy_in_mask(sal, mask) == 0.0
    assert iou_at_threshold(sal, mask) == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        energy_in_mask(np.ones((4, 4)), np.ones((5, 5), dtype=bool))


def test_negative_saliency_raises():
    with pytest.raises(ValueError):
        energy_in_mask(-np.ones((4, 4)), np.ones((4, 4), dtype=bool))


def test_bad_threshold_raises():
    with pytest.raises(ValueError):
        iou_at_threshold(np.ones((4, 4)), np.ones((4, 4), dtype=bool), threshold=1.5)


def test_report_has_all_keys():
    mask = _mask()
    rep = evidence_report(mask.astype(float), mask)
    assert set(rep) == {"energy_in_mask", "pointing_game", "iou"}
