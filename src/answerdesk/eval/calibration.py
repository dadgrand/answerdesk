"""Calibration and risk–coverage analysis (FR-EVL-06, §8.5 E-1/E-5/E-6).

What is computed and why the report needs all of it:

* **ECE, equal-width and equal-mass.** ECE is binning-dependent `[A]`: equal-width bins can
  hold two items and report a confident zero. Both variants are published so the reader can
  see whether a "calibrated" number is an artefact of one binning choice.
* **Brier score.** A proper scoring rule: unlike ECE it cannot be gamed by binning, which is
  why E-5 requires both.
* **Risk–coverage curve and AURC.** Selective prediction is the actual product behaviour:
  the system answers when ``s(x) <= q̂`` and escalates otherwise. The curve shows the whole
  trade-off, and AURC summarises it *without* becoming a target to tune (E-6 is explicitly
  a diagnostic, not a gate).

Convention: ``confidence = p̂(correct)``; a turn is auto-answered when
``confidence >= threshold``. ``scores`` are therefore confidences, and the risk at coverage
``c`` is the error rate among the ``c`` fraction of turns the system was most confident about.

    # AT-EVL-06  # AT-ESC-07
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from answerdesk.eval.stats import Interval, wilson_interval

BinningMode = Literal["equal_width", "equal_mass"]


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.accuracy - self.mean_confidence)


@dataclass(frozen=True)
class RiskCoveragePoint:
    """One operating point of the selective-prediction curve."""

    coverage: float
    risk: float
    threshold: float
    n_answered: int


@dataclass(frozen=True)
class CalibrationReport:
    """The artifact schema FR-EVL-06 requires, with per-bucket detail."""

    n: int
    ece_equal_width: float
    ece_equal_mass: float
    brier: float
    aurc: float
    mean_confidence: float
    accuracy: float
    bins_equal_width: list[ReliabilityBin] = field(default_factory=list)
    bins_equal_mass: list[ReliabilityBin] = field(default_factory=list)
    curve: list[RiskCoveragePoint] = field(default_factory=list)
    risk_at_target: float | None = None
    coverage_at_target: float | None = None
    risk_interval: Interval | None = None
    per_bucket: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "ece_equal_width": round(self.ece_equal_width, 6),
            "ece_equal_mass": round(self.ece_equal_mass, 6),
            "brier": round(self.brier, 6),
            "aurc": round(self.aurc, 6),
            "mean_confidence": round(self.mean_confidence, 6),
            "accuracy": round(self.accuracy, 6),
            "bins_equal_width": [
                {
                    "lower": round(bin.lower, 6),
                    "upper": round(bin.upper, 6),
                    "count": bin.count,
                    "mean_confidence": round(bin.mean_confidence, 6),
                    "accuracy": round(bin.accuracy, 6),
                }
                for bin in self.bins_equal_width
            ],
            "bins_equal_mass": [
                {
                    "lower": round(bin.lower, 6),
                    "upper": round(bin.upper, 6),
                    "count": bin.count,
                    "mean_confidence": round(bin.mean_confidence, 6),
                    "accuracy": round(bin.accuracy, 6),
                }
                for bin in self.bins_equal_mass
            ],
            "curve": [
                {
                    "coverage": round(point.coverage, 6),
                    "risk": round(point.risk, 6),
                    "threshold": round(point.threshold, 6),
                    "n_answered": point.n_answered,
                }
                for point in self.curve
            ],
            "risk_at_target": self.risk_at_target,
            "coverage_at_target": self.coverage_at_target,
            "risk_interval": (
                None
                if self.risk_interval is None
                else {
                    "low": round(self.risk_interval.low, 6),
                    "high": round(self.risk_interval.high, 6),
                    "n": self.risk_interval.n,
                }
            ),
            "per_bucket": self.per_bucket,
        }


def _validate(
    confidences: Sequence[float], correct: Sequence[bool]
) -> tuple[list[float], list[bool]]:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correctness flags must have the same length")
    values = [float(min(1.0, max(0.0, value))) for value in confidences]
    return values, [bool(flag) for flag in correct]


def reliability_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = 10,
    mode: BinningMode = "equal_width",
) -> list[ReliabilityBin]:
    """Equal-width or equal-mass reliability bins; empty bins are omitted."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    values, flags = _validate(confidences, correct)
    if not values:
        return []
    if mode == "equal_width":
        edges = [index / bins for index in range(bins + 1)]
        groups: list[list[int]] = [[] for _ in range(bins)]
        for position, value in enumerate(values):
            index = min(bins - 1, int(value * bins))
            groups[index].append(position)
        result: list[ReliabilityBin] = []
        for index, members in enumerate(groups):
            if not members:
                continue
            result.append(_bin(edges[index], edges[index + 1], members, values, flags))
        return result
    order = sorted(range(len(values)), key=lambda position: (values[position], position))
    result = []
    size = len(order) / bins
    for index in range(bins):
        start = round(index * size)
        end = round((index + 1) * size)
        members = order[start:end]
        if not members:
            continue
        lower = values[members[0]]
        upper = values[members[-1]]
        result.append(_bin(lower, upper, members, values, flags))
    return result


def _bin(
    lower: float, upper: float, members: list[int], values: list[float], flags: list[bool]
) -> ReliabilityBin:
    mean_confidence = sum(values[position] for position in members) / len(members)
    accuracy = sum(1.0 for position in members if flags[position]) / len(members)
    return ReliabilityBin(
        lower=lower,
        upper=upper,
        count=len(members),
        mean_confidence=mean_confidence,
        accuracy=accuracy,
    )


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = 10,
    mode: BinningMode = "equal_width",
) -> float:
    """``Σ (n_b / n) · |accuracy_b - confidence_b|`` (§8.5 E-5)."""
    values, flags = _validate(confidences, correct)
    if not values:
        return 0.0
    error = 0.0
    for bin in reliability_bins(values, flags, bins=bins, mode=mode):
        error += (bin.count / len(values)) * bin.gap
    return error


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error of the confidence forecast (E-5)."""
    values, flags = _validate(confidences, correct)
    if not values:
        return 0.0
    return sum(
        (value - (1.0 if flag else 0.0)) ** 2 for value, flag in zip(values, flags, strict=True)
    ) / len(values)


def risk_coverage_curve(
    confidences: Sequence[float], correct: Sequence[bool]
) -> list[RiskCoveragePoint]:
    """Risk at every achievable coverage, sweeping the answer threshold downwards."""
    values, flags = _validate(confidences, correct)
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda position: (-values[position], position))
    points: list[RiskCoveragePoint] = [
        RiskCoveragePoint(coverage=0.0, risk=0.0, threshold=1.0 + 1e-12, n_answered=0)
    ]
    errors = 0
    for answered, position in enumerate(order, start=1):
        if not flags[position]:
            errors += 1
        points.append(
            RiskCoveragePoint(
                coverage=answered / len(values),
                risk=errors / answered,
                threshold=values[position],
                n_answered=answered,
            )
        )
    return points


def aurc(curve: Sequence[RiskCoveragePoint]) -> float:
    """Area under the risk–coverage curve (trapezoid over the sweep above)."""
    if len(curve) < 2:
        return 0.0
    area = 0.0
    for left, right in itertools.pairwise(curve):
        width = right.coverage - left.coverage
        area += width * (left.risk + right.risk) / 2.0
    return area


def coverage_at_risk(
    confidences: Sequence[float], correct: Sequence[bool], target_risk: float
) -> tuple[float, float]:
    """Largest coverage whose realised risk stays at or below ``target_risk``.

    Returns ``(coverage, risk)``; a target that even the single most confident turn cannot
    meet yields ``(0.0, 0.0)``.
    """
    values, flags = _validate(confidences, correct)
    if not values:
        return 0.0, 0.0
    order = sorted(range(len(values)), key=lambda position: (-values[position], position))
    best_coverage, best_risk = 0.0, 0.0
    errors = 0
    for answered, position in enumerate(order, start=1):
        if not flags[position]:
            errors += 1
        risk = errors / answered
        if risk <= target_risk:
            best_coverage, best_risk = answered / len(values), risk
    return best_coverage, best_risk


def risk_at_coverage(
    confidences: Sequence[float], correct: Sequence[bool], coverage: float
) -> tuple[float, float]:
    """Risk and the operative threshold at a fixed coverage (the operating point)."""
    values, flags = _validate(confidences, correct)
    if not values:
        return 0.0, 1.0
    order = sorted(range(len(values)), key=lambda position: (-values[position], position))
    answered = max(1, min(len(values), math.ceil(coverage * len(values))))
    kept = order[:answered]
    errors = sum(1 for position in kept if not flags[position])
    return errors / answered, values[kept[-1]]


def summarise(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = 10,
    target_risk: float = 0.05,
    per_bucket: dict[str, Any] | None = None,
) -> CalibrationReport:
    """Everything E-5/E-6 asks for, in one artifact."""
    values, flags = _validate(confidences, correct)
    if not values:
        return CalibrationReport(
            n=0,
            ece_equal_width=0.0,
            ece_equal_mass=0.0,
            brier=0.0,
            aurc=0.0,
            mean_confidence=0.0,
            accuracy=0.0,
            per_bucket=per_bucket or {},
        )
    curve = risk_coverage_curve(values, flags)
    coverage, risk = coverage_at_risk(values, flags, target_risk)
    answered_at_target = round(coverage * len(values))
    order = sorted(range(len(values)), key=lambda position: (-values[position], position))
    kept = order[:answered_at_target]
    errors_at_target = sum(1 for position in kept if not flags[position])
    return CalibrationReport(
        n=len(values),
        ece_equal_width=expected_calibration_error(values, flags, bins=bins, mode="equal_width"),
        ece_equal_mass=expected_calibration_error(values, flags, bins=bins, mode="equal_mass"),
        brier=brier_score(values, flags),
        aurc=aurc(curve),
        mean_confidence=sum(values) / len(values),
        accuracy=sum(1.0 for flag in flags if flag) / len(flags),
        bins_equal_width=reliability_bins(values, flags, bins=bins, mode="equal_width"),
        bins_equal_mass=reliability_bins(values, flags, bins=bins, mode="equal_mass"),
        curve=curve,
        risk_at_target=risk,
        coverage_at_target=coverage,
        risk_interval=(
            wilson_interval(errors_at_target, answered_at_target) if answered_at_target else None
        ),
        per_bucket=per_bucket or {},
    )


def curve_arrays(curve: Sequence[RiskCoveragePoint]) -> tuple[np.ndarray, np.ndarray]:
    """``(coverage, risk)`` arrays for plotting."""
    if not curve:
        return np.asarray([]), np.asarray([])
    return (
        np.asarray([point.coverage for point in curve], dtype=np.float64),
        np.asarray([point.risk for point in curve], dtype=np.float64),
    )


__all__ = [
    "BinningMode",
    "CalibrationReport",
    "ReliabilityBin",
    "RiskCoveragePoint",
    "aurc",
    "brier_score",
    "coverage_at_risk",
    "curve_arrays",
    "expected_calibration_error",
    "reliability_bins",
    "risk_at_coverage",
    "risk_coverage_curve",
    "summarise",
]
