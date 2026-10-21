"""Statistics the report is not allowed to omit (FR-EVL-07, §6.3, §10.4).

Three tools, each with the edge case that makes it honest:

* **Wilson score interval** for proportions — never the normal approximation, which
  produces impossible bounds and collapses to zero width at 0/1 (exactly where abstention
  metrics live);
* **paired bootstrap over queries** (``B = 1000``, 2.5/97.5 percentiles) for "is arm A
  better than arm B on the same queries", which is the only comparison the report is
  allowed to call an improvement (§6.3 step 5);
* **McNemar's test** on the discordant pairs of a binary outcome (any-hit@k), with the
  exact-binomial fallback when ``b + c < 25`` because the χ² approximation is not
  trustworthy at that size.

Plus Cohen's κ, which gates whether a judged metric may be reported for a language at all
(§10.5, F-10).

    # AT-EVL-07
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
from numpy.typing import NDArray

#: Bootstrap resamples required by §10.4 / FR-EVL-07.
DEFAULT_BOOTSTRAP = 1000
#: Below this many discordant pairs, McNemar uses the exact binomial test.
MCNEMAR_EXACT_BELOW = 25


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided interval and the method that produced it."""

    value: float
    low: float
    high: float
    n: int
    method: Literal["wilson", "bootstrap", "none"] = "wilson"
    confidence: float = 0.95

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class Difference:
    """A paired difference with its bootstrap interval."""

    mean_a: float
    mean_b: float
    diff: float
    low: float
    high: float
    n: int
    bootstrap: int = DEFAULT_BOOTSTRAP
    seed: int = 0

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


@dataclass(frozen=True)
class McNemarResult:
    """McNemar's test on the discordant cells (b: A right/B wrong, c: the reverse)."""

    b: int
    c: int
    statistic: float
    p_value: float
    method: Literal["chi2", "exact_binomial", "n/a"]

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05 and self.method != "n/a"


def z_for(confidence: float = 0.95) -> float:
    """Two-sided normal quantile for a confidence level (1.96 at 95%)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def proportion(successes: float, n: int) -> float:
    return (successes / n) if n else 0.0


def wilson_interval(successes: float, n: int, *, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion (no continuity correction)."""
    if n <= 0:
        return Interval(value=0.0, low=0.0, high=1.0, n=0, method="none", confidence=confidence)
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} outside [0, {n}]")
    z = z_for(confidence)
    p_hat = successes / n
    denominator = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    low = max(0.0, centre - margin)
    high = min(1.0, centre + margin)
    return Interval(
        value=p_hat,
        low=low,
        high=high,
        n=n,
        method="wilson",
        confidence=confidence,
    )


def _as_array(values: Iterable[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    array = (
        np.asarray(values, dtype=np.float64)
        if isinstance(values, np.ndarray)
        else np.asarray(list(values), dtype=np.float64)
    )
    if array.ndim != 1:
        raise ValueError("expected a one-dimensional sequence")
    return array


def bootstrap_interval(
    values: Iterable[float] | NDArray[np.float64],
    *,
    samples: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    confidence: float = 0.95,
) -> Interval:
    """Percentile bootstrap CI of the mean (§10.4).

    Deterministic for a fixed ``seed``: ``numpy.random.default_rng`` is the PCG64 stream,
    whose output is stable across numpy releases, so the same seed reproduces the same
    interval (NFR-DET-02).
    """
    array = _as_array(values)
    n = int(array.size)
    if n == 0:
        return Interval(value=0.0, low=0.0, high=1.0, n=0, method="none", confidence=confidence)
    mean = float(array.mean())
    if n == 1 or float(array.std()) == 0.0:
        return Interval(
            value=mean, low=mean, high=mean, n=n, method="bootstrap", confidence=confidence
        )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, n, size=(samples, n))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return Interval(
        value=mean,
        low=float(min(low, mean)),
        high=float(max(high, mean)),
        n=n,
        method="bootstrap",
        confidence=confidence,
    )


def paired_bootstrap(
    a: Iterable[float] | NDArray[np.float64],
    b: Iterable[float] | NDArray[np.float64],
    *,
    samples: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    confidence: float = 0.95,
) -> Difference:
    """Paired bootstrap of ``mean(a) - mean(b)`` over the *same* items (§6.3 step 5).

    Resampling is by item, not by arm: the pairing is the whole point, and an unpaired
    interval is wider than the data deserves.
    """
    left = _as_array(a)
    right = _as_array(b)
    if left.shape != right.shape:
        raise ValueError(f"paired samples must have equal length, got {left.size} and {right.size}")
    n = int(left.size)
    if n == 0:
        return Difference(0.0, 0.0, 0.0, 0.0, 0.0, 0, samples, seed)
    delta = left - right
    mean_diff = float(delta.mean())
    if n == 1 or float(delta.std()) == 0.0:
        return Difference(
            mean_a=float(left.mean()),
            mean_b=float(right.mean()),
            diff=mean_diff,
            low=mean_diff,
            high=mean_diff,
            n=n,
            bootstrap=samples,
            seed=seed,
        )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, n, size=(samples, n))
    means = delta[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return Difference(
        mean_a=float(left.mean()),
        mean_b=float(right.mean()),
        diff=mean_diff,
        low=float(min(low, mean_diff)),
        high=float(max(high, mean_diff)),
        n=n,
        bootstrap=samples,
        seed=seed,
    )


def _binomial_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial test of ``b ~ Binom(b + c, 0.5)``."""
    n = b + c
    if n == 0:
        return 1.0
    smaller = min(b, c)
    tail = sum(math.comb(n, k) for k in range(0, smaller + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def mcnemar(b: int, c: int) -> McNemarResult:
    """McNemar's test with the χ² approximation and its exact-binomial fallback."""
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return McNemarResult(b=b, c=c, statistic=0.0, p_value=1.0, method="n/a")
    if n < MCNEMAR_EXACT_BELOW:
        return McNemarResult(
            b=b,
            c=c,
            statistic=float((b - c) ** 2 / n),
            p_value=_binomial_two_sided(b, c),
            method="exact_binomial",
        )
    statistic = (abs(b - c) - 1) ** 2 / n
    p_value = math.erfc(math.sqrt(max(0.0, statistic) / 2.0))
    return McNemarResult(b=b, c=c, statistic=statistic, p_value=p_value, method="chi2")


def cohen_kappa(
    first: Iterable[bool] | NDArray[np.bool_], second: Iterable[bool] | NDArray[np.bool_]
) -> float:
    """Cohen's κ between two binary labellings (§10.5 human calibration)."""
    left = [bool(item) for item in first]
    right = [bool(item) for item in second]
    if len(left) != len(right):
        raise ValueError("labellings must have the same length")
    n = len(left)
    if n == 0:
        return 0.0
    observed = sum(1 for a, b in zip(left, right, strict=True) if a == b) / n
    p_left = sum(left) / n
    p_right = sum(right) / n
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    if abs(1.0 - expected) < 1e-12:
        # Both labellers used a single class: agreement is undefined, not perfect.
        return 0.0
    return (observed - expected) / (1.0 - expected)


def quantile(values: Iterable[float] | NDArray[np.float64], q: float) -> float:
    """Linear-interpolation quantile used for p50/p95 latency reporting."""
    array = _as_array(values)
    if array.size == 0:
        return 0.0
    return float(np.percentile(array, 100.0 * q))


def mean(values: Iterable[float] | NDArray[np.float64]) -> float:
    array = _as_array(values)
    if not array.size:
        return 0.0
    return float(np.mean(array))


__all__ = [
    "DEFAULT_BOOTSTRAP",
    "MCNEMAR_EXACT_BELOW",
    "Difference",
    "Interval",
    "McNemarResult",
    "bootstrap_interval",
    "cohen_kappa",
    "mcnemar",
    "mean",
    "paired_bootstrap",
    "proportion",
    "quantile",
    "wilson_interval",
    "z_for",
]
