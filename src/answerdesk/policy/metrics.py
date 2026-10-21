"""Calibration and selective-prediction metrics, implemented rather than imported.

These are the numbers the escalation artifact is judged by (requirements §8.5): realised
risk with a Wilson interval (E-1), ECE and Brier (E-5), the risk–coverage curve and its
area (E-6). They live here, dependency-free and unit-tested against hand-computed values,
so a report can be reproduced without a plotting or ML stack.

Definitions (docs/escalation.md §4):

* ``wilson_interval(k, n)`` — Wilson score interval for a binomial proportion; the
  ``clopper_pearson_upper`` companion is used where a conservative bound is required.
* ``expected_calibration_error`` — ``Σ_b (n_b/n)·|acc_b − conf_b|`` over equal-width (or
  equal-mass) bins; ECE is binning-dependent, which is why the bins are always reported
  alongside it (the spec's caveat, not ours to hide).
* ``brier_score`` — ``mean((p̂ − y)²)``, a proper scoring rule that does not depend on bins.
* ``risk_coverage_curve`` — for every threshold on ``p̂``: coverage = fraction answered,
  risk = error rate among answered.
* ``aurc`` — trapezoidal ``∫ risk dcoverage``; a diagnostic, never a gate (E-6).

    # AT-ESC-07  # AT-EVL-06
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence

from pydantic import Field

from answerdesk.contracts import Contract

#: z for a two-sided 95% interval.
Z_95 = 1.959963984540054


def clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def wilson_interval(successes: int, n: int, *, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for ``successes / n``; ``(0, 1)`` when ``n == 0``."""
    if n <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > n:
        raise ValueError("successes must be within [0, n]")
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = (z / denominator) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (clamp01(centre - half), clamp01(centre + half))


def wilson_upper(successes: int, n: int, *, z: float = Z_95) -> float:
    """Upper bound of the Wilson interval — the E-1 / K-1 "UCB" column."""
    return wilson_interval(successes, n, z=z)[1]


def clopper_pearson_upper(successes: int, n: int, *, alpha: float = 0.05) -> float:
    """Exact one-sided Clopper–Pearson upper bound (used where a conservative bound matters).

    Solves ``P(X ≤ successes; n, p) = alpha`` for ``p`` by bisection on the regularised
    incomplete beta, so the module needs no SciPy. Returns ``1.0`` when ``successes >= n``.
    """
    if n <= 0:
        return 1.0
    if successes >= n:
        return 1.0
    if successes < 0:
        raise ValueError("successes must be >= 0")

    def cdf(p: float) -> float:
        # P(X <= k | n, p) = I_{1-p}(n - k, k + 1)
        return _regularised_incomplete_beta(1.0 - p, n - successes, successes + 1)

    low, high = successes / n, 1.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if cdf(mid) > alpha:
            low = mid
        else:
            high = mid
    return high


def _regularised_incomplete_beta(x: float, a: int, b: int) -> float:
    """``I_x(a, b)`` for positive integer parameters, via the Bernoulli/Binomial identity.

    For integer ``a, b`` the regularised incomplete beta equals
    ``Σ_{j=a}^{a+b-1} C(a+b-1, j) x^j (1-x)^(a+b-1-j)``.
    """
    total = a + b - 1
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_terms: list[float] = []
    for j in range(a, total + 1):
        log_terms.append(
            math.lgamma(total + 1)
            - math.lgamma(j + 1)
            - math.lgamma(total - j + 1)
            + j * math.log(x)
            + (total - j) * math.log1p(-x)
        )
    return min(1.0, math.fsum(math.exp(term) for term in log_terms))


class ReliabilityBin(Contract):
    """One calibration bin: what was predicted vs what happened."""

    index: int = Field(ge=0)
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)

    @property
    def gap(self) -> float:
        return abs(self.accuracy - self.mean_confidence)


def _check_inputs(probs: Sequence[float], labels: Sequence[int]) -> None:
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length")


def reliability_bins(
    probs: Sequence[float],
    labels: Sequence[bool | int],
    *,
    bins: int = 10,
    mode: str = "width",
) -> list[ReliabilityBin]:
    """Bin ``probs`` and report mean confidence vs accuracy per bin.

    ``mode="width"`` uses equal-width bins on ``[0, 1]`` (the E-5 default);
    ``mode="mass"`` uses equal-count bins (reported alongside, because ECE depends on it).
    Empty bins are omitted from the returned list but ``count`` still shows in the ECE
    normalisation via the total.
    """
    _check_inputs(probs, labels)
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if mode not in {"width", "mass"}:
        raise ValueError("mode must be 'width' or 'mass'")
    n = len(probs)
    if n == 0:
        return []

    entries = list(zip(probs, labels, strict=True))
    grouped: dict[int, list[tuple[float, float]]] = {}
    if mode == "width":
        for prob, label in entries:
            index = min(int(clamp01(prob) * bins), bins - 1)
            grouped.setdefault(index, []).append((clamp01(prob), float(label)))
        bounds = {index: (index / bins, (index + 1) / bins) for index in range(bins)}
    else:
        ordered = sorted(entries, key=lambda item: item[0])
        for position, (prob, label) in enumerate(ordered):
            index = min(int(position * bins / n), bins - 1)
            grouped.setdefault(index, []).append((clamp01(prob), float(label)))
        bounds = {}
        for index in range(bins):
            start = math.floor(index * n / bins)
            stop = math.floor((index + 1) * n / bins)
            if stop > start:
                bounds[index] = (
                    clamp01(ordered[start][0]),
                    clamp01(ordered[stop - 1][0]),
                )

    result: list[ReliabilityBin] = []
    for index in sorted(grouped):
        rows = grouped[index]
        lower, upper = bounds.get(index, (0.0, 1.0))
        result.append(
            ReliabilityBin(
                index=index,
                lower=lower,
                upper=upper,
                count=len(rows),
                mean_confidence=sum(prob for prob, _ in rows) / len(rows),
                accuracy=sum(label for _, label in rows) / len(rows),
            )
        )
    return result


def expected_calibration_error(
    probs: Sequence[float],
    labels: Sequence[bool | int],
    *,
    bins: int = 10,
    mode: str = "width",
) -> float:
    """``Σ_b (n_b / n)·|acc_b − conf_b|`` (0 is perfect calibration)."""
    _check_inputs(probs, labels)
    n = len(probs)
    if n == 0:
        return 0.0
    total = 0.0
    for bin_row in reliability_bins(probs, labels, bins=bins, mode=mode):
        total += (bin_row.count / n) * bin_row.gap
    return total


def brier_score(probs: Sequence[float], labels: Sequence[bool | int]) -> float:
    """``mean((p̂ − y)²)`` — proper, bin-free, and the E-5 companion to ECE."""
    _check_inputs(probs, labels)
    if not probs:
        return 0.0
    return math.fsum(
        (clamp01(prob) - float(label)) ** 2 for prob, label in zip(probs, labels, strict=True)
    ) / len(probs)


class RiskCoveragePoint(Contract):
    """One point of the risk–coverage curve (selective prediction)."""

    coverage: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    n_selected: int = Field(ge=0)


def risk_coverage_curve(
    probs: Sequence[float], labels: Sequence[bool | int]
) -> list[RiskCoveragePoint]:
    """Risk and coverage for every distinct threshold on ``p̂`` (descending confidence)."""
    _check_inputs(probs, labels)
    n = len(probs)
    if n == 0:
        return []
    ordered = sorted(
        ((clamp01(prob), float(label)) for prob, label in zip(probs, labels, strict=True)),
        key=lambda item: item[0],
        reverse=True,
    )
    points: list[RiskCoveragePoint] = []
    errors = 0
    index = 0
    while index < n:
        threshold = ordered[index][0]
        while index < n and ordered[index][0] == threshold:
            errors += 1 if ordered[index][1] == 0.0 else 0
            index += 1
        points.append(
            RiskCoveragePoint(
                coverage=index / n,
                risk=errors / index,
                threshold=threshold,
                n_selected=index,
            )
        )
    return points


def aurc(curve: Sequence[RiskCoveragePoint]) -> float:
    """Area under the risk–coverage curve (trapezoidal over coverage)."""
    if not curve:
        return 0.0
    ordered = sorted(curve, key=lambda point: point.coverage)
    area = 0.0
    for previous, current in itertools.pairwise(ordered):
        width = current.coverage - previous.coverage
        area += width * (current.risk + previous.risk) / 2.0
    return area


def risk_at_coverage(
    probs: Sequence[float], labels: Sequence[bool | int], coverage: float
) -> float:
    """Risk of the most confident ``coverage`` fraction (linear interpolation of the curve)."""
    curve = risk_coverage_curve(probs, labels)
    if not curve:
        return 0.0
    if coverage <= curve[0].coverage:
        return curve[0].risk
    for previous, current in itertools.pairwise(curve):
        if previous.coverage <= coverage <= current.coverage:
            span = current.coverage - previous.coverage
            if span == 0:
                return current.risk
            weight = (coverage - previous.coverage) / span
            return previous.risk + weight * (current.risk - previous.risk)
    return curve[-1].risk


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean that returns 0.0 for an empty iterable (report-friendly)."""
    items = list(values)
    return 0.0 if not items else math.fsum(items) / len(items)


__all__ = [
    "Z_95",
    "ReliabilityBin",
    "RiskCoveragePoint",
    "aurc",
    "brier_score",
    "clamp01",
    "clopper_pearson_upper",
    "expected_calibration_error",
    "mean",
    "reliability_bins",
    "risk_at_coverage",
    "risk_coverage_curve",
    "wilson_interval",
    "wilson_upper",
]
