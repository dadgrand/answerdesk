"""Calibration metrics, against hand-computed values.

# AT-ESC-07  # AT-EVL-06
"""

from __future__ import annotations

import pytest

from answerdesk.policy.metrics import (
    aurc,
    brier_score,
    clopper_pearson_upper,
    expected_calibration_error,
    reliability_bins,
    risk_at_coverage,
    risk_coverage_curve,
    wilson_interval,
    wilson_upper,
)


def test_wilson_interval_matches_the_closed_form() -> None:
    """n=100, k=10 → p=0.1, interval ≈ [0.0554, 0.1739] (textbook Wilson)."""
    low, high = wilson_interval(10, 100)
    assert low == pytest.approx(0.0554, abs=5e-4)
    assert high == pytest.approx(0.1739, abs=5e-4)
    assert wilson_upper(0, 100) == pytest.approx(0.0370, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_clopper_pearson_upper_is_exact_for_a_known_case() -> None:
    """0 events in 10 trials: upper = 1 − 0.05^(1/10) = 0.2589."""
    assert clopper_pearson_upper(0, 10) == pytest.approx(0.2589, abs=1e-3)
    assert clopper_pearson_upper(10, 10) == 1.0
    assert clopper_pearson_upper(0, 0) == 1.0


def test_reliability_bins_and_ece_by_hand() -> None:
    """Two bins: (0.9, 0.9, 0.9→correct once) and (0.1, 0.1→wrong twice)."""
    probs = [0.9, 0.9, 0.1, 0.1]
    labels = [True, False, False, False]
    bins = reliability_bins(probs, labels, bins=10)
    assert len(bins) == 2
    low_bin = next(bin_row for bin_row in bins if bin_row.index == 1)
    high_bin = next(bin_row for bin_row in bins if bin_row.index == 9)
    assert low_bin.count == 2 and low_bin.accuracy == pytest.approx(0.0)
    assert high_bin.count == 2 and high_bin.accuracy == pytest.approx(0.5)
    # ECE = 0.5*|0.0 − 0.1| + 0.5*|0.5 − 0.9| = 0.05 + 0.2 = 0.25
    assert expected_calibration_error(probs, labels, bins=10) == pytest.approx(0.25)


def test_ece_equal_mass_mode_can_differ_from_equal_width() -> None:
    probs = [0.05, 0.15, 0.25, 0.95]
    labels = [False, False, True, True]
    width = expected_calibration_error(probs, labels, bins=2, mode="width")
    mass = expected_calibration_error(probs, labels, bins=2, mode="mass")
    assert width != pytest.approx(mass)
    assert 0.0 <= min(width, mass) <= max(width, mass) <= 1.0


def test_perfectly_calibrated_predictions_have_zero_ece() -> None:
    probs = [0.5, 0.5, 0.5, 0.5]
    labels = [True, True, False, False]
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)
    assert brier_score(probs, labels) == pytest.approx(0.25)


def test_brier_score_hand_computed() -> None:
    assert brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)
    assert brier_score([1.0, 0.0], [False, True]) == pytest.approx(1.0)
    assert brier_score([0.9, 0.1], [True, True]) == pytest.approx((0.01 + 0.81) / 2)


def test_risk_coverage_curve_and_aurc() -> None:
    probs = [0.9, 0.8, 0.2, 0.1]
    labels = [True, False, False, True]
    curve = risk_coverage_curve(probs, labels)
    assert [point.coverage for point in curve] == [0.25, 0.5, 0.75, 1.0]
    assert [point.risk for point in curve] == [0.0, 0.5, pytest.approx(2 / 3), 0.5]
    area = aurc(curve)
    assert area == pytest.approx(
        0.5 * 0.25 * 0 + 0.25 * (0 + 0.5) / 2 + 0.25 * (0.5 + 2 / 3) / 2 + 0.25 * (2 / 3 + 0.5) / 2
    )
    assert risk_at_coverage(probs, labels, 0.25) == pytest.approx(0.0)
    assert risk_at_coverage(probs, labels, 1.0) == pytest.approx(0.5)


def test_degenerate_inputs_are_safe() -> None:
    assert expected_calibration_error([], []) == 0.0
    assert brier_score([], []) == 0.0
    assert risk_coverage_curve([], []) == []
    assert aurc([]) == 0.0
    assert risk_at_coverage([], [], 0.5) == 0.0
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [True, False])
    with pytest.raises(ValueError):
        reliability_bins([0.5], [True], bins=0)
    with pytest.raises(ValueError):
        reliability_bins([0.5], [True], mode="nope")
