"""Calibration tests: ECE (two binnings), Brier, risk–coverage, AURC (AT-EVL-06).

# AT-EVL-06
"""

from __future__ import annotations

import pytest

from answerdesk.eval.calibration import (
    aurc,
    brier_score,
    coverage_at_risk,
    expected_calibration_error,
    reliability_bins,
    risk_at_coverage,
    risk_coverage_curve,
    summarise,
)


def test_ece_hand_computed_equal_width() -> None:
    # two bins: (0.0-0.1) with confidence 0.0 accuracy 1.0 -> gap 1.0
    #           (0.9-1.0) with confidence 1.0 accuracy 1.0 -> gap 0.0
    confidences = [0.0, 1.0]
    correct = [True, True]
    ece = expected_calibration_error(confidences, correct, bins=2, mode="equal_width")
    assert ece == pytest.approx(0.5)


def test_ece_perfect_calibration_is_zero() -> None:
    confidences = [1.0, 1.0, 0.0, 0.0]
    correct = [True, True, False, False]
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.0)


def test_ece_equal_mass_bins_are_not_equal_width() -> None:
    confidences = [0.05, 0.1, 0.2, 0.9]
    correct = [False, False, True, True]
    width_bins = reliability_bins(confidences, correct, bins=2, mode="equal_width")
    mass_bins = reliability_bins(confidences, correct, bins=2, mode="equal_mass")
    assert sum(bin.count for bin in width_bins) == 4
    assert sum(bin.count for bin in mass_bins) == 4
    assert [bin.count for bin in mass_bins] == [2, 2]
    assert expected_calibration_error(
        confidences, correct, bins=2, mode="equal_mass"
    ) != pytest.approx(expected_calibration_error(confidences, correct, bins=2, mode="equal_width"))


def test_brier_closed_form() -> None:
    assert brier_score([1.0, 0.0], [True, True]) == pytest.approx(0.5)
    assert brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)
    assert brier_score([], []) == 0.0


def test_risk_coverage_curve_and_aurc() -> None:
    confidences = [0.9, 0.8, 0.2, 0.1]
    correct = [True, False, True, False]
    curve = risk_coverage_curve(confidences, correct)
    assert curve[0].coverage == 0.0
    assert curve[1].risk == pytest.approx(0.0)  # top-1 was right
    assert curve[2].risk == pytest.approx(0.5)  # 1 error in 2 answered
    assert curve[3].risk == pytest.approx(1 / 3)  # 1 error in 3 answered
    assert curve[4].risk == pytest.approx(0.5)  # 2 errors in 4
    # trapezoid over equal coverage steps with risks 0, 0, .5, 1/3, .5
    expected = (
        0.25 * (0.0 + 0.0) / 2
        + 0.25 * (0.0 + 0.5) / 2
        + 0.25 * (0.5 + 1 / 3) / 2
        + 0.25 * (1 / 3 + 0.5) / 2
    )
    assert aurc(curve) == pytest.approx(expected)


def test_risk_at_coverage_and_coverage_at_risk() -> None:
    confidences = [0.9, 0.8, 0.7, 0.6]
    correct = [True, True, False, False]
    risk, threshold = risk_at_coverage(confidences, correct, 0.5)
    assert risk == pytest.approx(0.0)
    assert threshold == pytest.approx(0.8)
    coverage, achieved = coverage_at_risk(confidences, correct, 0.0)
    assert coverage == pytest.approx(0.5)
    assert achieved == pytest.approx(0.0)
    # an impossible target falls back to "answer nothing"
    coverage, achieved = coverage_at_risk(confidences, correct, -1.0)
    assert coverage == 0.0 and achieved == 0.0


def test_summarise_reports_both_eces_and_buckets() -> None:
    confidences = [0.95, 0.9, 0.85, 0.2, 0.15]
    correct = [True, True, False, False, True]
    report = summarise(
        confidences, correct, bins=5, target_risk=0.0, per_bucket={"informational": {"n": 5}}
    )
    assert report.n == 5
    assert 0.0 <= report.ece_equal_width <= 1.0
    assert 0.0 <= report.ece_equal_mass <= 1.0
    assert 0.0 <= report.brier <= 1.0
    assert report.curve and len(report.curve) == 6
    assert report.risk_interval is not None
    payload = report.to_dict()
    assert payload["n"] == 5
    assert payload["per_bucket"] == {"informational": {"n": 5}}
    assert len(payload["curve"]) == 6


def test_summarise_handles_empty_input() -> None:
    report = summarise([], [])
    assert report.n == 0
    assert report.to_dict()["curve"] == []


def test_calibration_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        summarise([0.5], [True, False])
