import numpy as np
import pytest

from freshproof.uncertainty import (
    choose_abstain_threshold,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    selective_accuracy,
    softmax,
)


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_higher_temperature_flattens():
    logits = np.array([[4.0, 1.0, 0.0]])
    assert softmax(logits, 5.0).max() < softmax(logits, 1.0).max()


def test_softmax_rejects_bad_temperature():
    with pytest.raises(ValueError):
        softmax(np.ones((1, 2)), 0)


def test_ece_zero_when_confident_and_right():
    assert expected_calibration_error([1.0, 1.0, 1.0], [1, 1, 1]) == pytest.approx(0.0)


def test_ece_large_when_overconfident():
    conf = np.full(100, 0.9)
    correct = np.array([1] * 50 + [0] * 50)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.4)


def test_ece_input_validation():
    with pytest.raises(ValueError):
        expected_calibration_error([], [])
    with pytest.raises(ValueError):
        expected_calibration_error([0.5, 0.6], [1])


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=5000)
    labels = np.array([rng.choice(3, p=p) for p in probs])
    overconfident = np.log(probs) * 3.0  # sharper than the truth

    temperature = fit_temperature(overconfident, labels)
    assert temperature > 2.0

    before = softmax(overconfident)
    after = softmax(overconfident, temperature)
    ece_before = expected_calibration_error(before.max(1), before.argmax(1) == labels)
    ece_after = expected_calibration_error(after.max(1), after.argmax(1) == labels)
    assert ece_after < ece_before
    assert negative_log_likelihood(overconfident, labels, temperature) < (
        negative_log_likelihood(overconfident, labels)
    )


def test_selective_accuracy():
    conf = np.array([0.9, 0.8, 0.6, 0.5])
    correct = np.array([1, 1, 0, 0])
    acc, cov = selective_accuracy(conf, correct, 0.7)
    assert acc == 1.0
    assert cov == 0.5
    acc, cov = selective_accuracy(conf, correct, 0.95)
    assert np.isnan(acc) and cov == 0.0


def test_choose_threshold_meets_target():
    conf = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.5])
    correct = np.array([1, 1, 1, 0, 1, 0])
    choice = choose_abstain_threshold(conf, correct, target_accuracy=1.0)
    assert choice is not None
    assert choice.accuracy == 1.0
    assert choice.threshold == pytest.approx(0.85)
    assert choice.coverage == pytest.approx(0.5)


def test_choose_threshold_none_when_impossible():
    conf = np.array([0.9, 0.8])
    correct = np.array([0, 0])
    assert choose_abstain_threshold(conf, correct, target_accuracy=0.9) is None


def test_choose_threshold_respects_min_coverage():
    conf = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.5])
    correct = np.array([1, 1, 1, 0, 1, 0])
    choice = choose_abstain_threshold(conf, correct, 1.0, min_coverage=0.9)
    assert choice is None
