"""Conformal Risk Control / Learn-then-Test with a finite-sample certificate.

Nothing here is a nonconformity quantile: the threshold is the largest calibration-set
acceptance level whose **upper confidence bound** on the selective risk stays below ``alpha``.
Two bounds are implemented, both valid for a bounded loss in [0, 1]:

* **Hoeffding** — ``R_hat + sqrt(log(1/delta) / (2 n))``. Simple, always valid, loose.
* **Bentkus** — ``R_hat + (sqrt(R_hat * x / n) + x / n)`` with ``x = log(1/delta)``. Tight for
  small risks and the bound used for the shipped certificate.

The certificate asserts only what the bound gives: with probability at least ``1 - delta`` over
the calibration draw, the selective risk of the *chosen* threshold is at most ``alpha`` **if**
the threshold is chosen by this rule on that same calibration split (a fixed-sequence /
Learn-then-Test procedure: one pass over the grid, most permissive first, stop at the first
feasible level). Test-set risk is reported separately and is *not* covered by the certificate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Bound = Literal["bentkus", "hoeffding"]


def bentkus_ucb(risk: float, n: int, delta: float) -> float:
    """Bentkus-style upper confidence bound for a bounded mean, clipped to [0, 1]."""
    if n <= 0:
        return 1.0
    if not 0.0 <= risk <= 1.0:
        raise ValueError("risk must be in [0, 1]")
    x = math.log(1.0 / delta)
    return min(1.0, risk + math.sqrt(risk * x / n) + x / n)


def hoeffding_ucb(risk: float, n: int, delta: float) -> float:
    """Hoeffding upper confidence bound, clipped to [0, 1]."""
    if n <= 0:
        return 1.0
    if not 0.0 <= risk <= 1.0:
        raise ValueError("risk must be in [0, 1]")
    return min(1.0, risk + math.sqrt(math.log(1.0 / delta) / (2.0 * n)))


def ucb(risk: float, n: int, delta: float, bound: Bound) -> float:
    if bound == "bentkus":
        return bentkus_ucb(risk, n, delta)
    if bound == "hoeffding":
        return hoeffding_ucb(risk, n, delta)
    raise ValueError(f"unknown bound {bound!r}")


@dataclass(frozen=True)
class Certificate:
    alpha: float
    delta: float
    bound: Bound
    n_calibration: int
    threshold: float
    calibration_risk: float
    calibration_coverage: float
    calibration_ucb: float
    feasible: bool
    grid: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "delta": self.delta,
            "bound": self.bound,
            "n_calibration": self.n_calibration,
            "threshold": round(float(self.threshold), 6),
            "calibration_risk": round(float(self.calibration_risk), 6),
            "calibration_coverage": round(float(self.calibration_coverage), 6),
            "calibration_ucb": round(float(self.calibration_ucb), 6),
            "feasible": bool(self.feasible),
            "rule": (
                "select the largest calibration acceptance level (most permissive threshold) "
                "whose UCB(R_hat) <= alpha, scanning the grid once from most to least "
                "permissive and stopping at the first feasible level (Learn-then-Test / "
                "fixed-sequence); the guarantee covers this selection rule, not any threshold "
                "chosen after looking at test data"
            ),
            "claims": (
                "with probability >= 1-delta over the calibration draw, the selective risk of "
                "the returned threshold is <= alpha (bounded loss, i.i.d. calibration)"
                if self.feasible
                else "no threshold on the grid satisfies the bound; the certificate is ABSENT and "
                "the policy must abstain rather than claim risk control"
            ),
            "grid": [dict(point) for point in self.grid],
        }


def certify(
    *,
    scores: Sequence[float],
    labels: Sequence[int],
    alpha: float = 0.05,
    delta: float = 0.1,
    bound: Bound = "bentkus",
    grid: Sequence[float] | None = None,
) -> Certificate:
    """Learn-then-Test selection of the acceptance threshold with a finite-sample bound.

    ``scores`` and ``labels`` are the *calibration* (dev) split; ``labels == 1`` means correct.
    The grid is scanned from the most permissive threshold (answer everything) to the strictest,
    and the first level whose UCB is within ``alpha`` is taken — by construction this is the
    largest coverage that survives the certificate.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be aligned")
    n = len(scores)
    if n == 0:
        return Certificate(alpha, delta, bound, 0, 1.0, 1.0, 0.0, 1.0, False, tuple())
    thresholds = sorted(set(float(score) for score in scores), reverse=True)
    if grid is not None:
        thresholds = sorted({*thresholds, *[float(value) for value in grid]}, reverse=True)
    levels: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = [
            (score, label)
            for score, label in zip(scores, labels, strict=True)
            if score >= threshold
        ]
        if not accepted:
            continue
        risk = sum(1 for _score, label in accepted if label == 0) / len(accepted)
        coverage = len(accepted) / n
        bound_value = ucb(risk, len(accepted), delta, bound)
        levels.append(
            {
                "threshold": round(threshold, 6),
                "coverage": round(coverage, 6),
                "risk": round(risk, 6),
                "ucb": round(bound_value, 6),
                "n_accepted": len(accepted),
                "feasible": bound_value <= alpha,
            }
        )
    feasible_levels = [level for level in levels if level["feasible"]]
    if not feasible_levels:
        strictest = max(thresholds) if thresholds else 1.0
        return Certificate(
            alpha=alpha,
            delta=delta,
            bound=bound,
            n_calibration=n,
            threshold=strictest,
            calibration_risk=1.0,
            calibration_coverage=0.0,
            calibration_ucb=1.0,
            feasible=False,
            grid=tuple(levels),
        )
    chosen = max(feasible_levels, key=lambda level: level["coverage"])
    return Certificate(
        alpha=alpha,
        delta=delta,
        bound=bound,
        n_calibration=n,
        threshold=float(chosen["threshold"]),
        calibration_risk=float(chosen["risk"]),
        calibration_coverage=float(chosen["coverage"]),
        calibration_ucb=float(chosen["ucb"]),
        feasible=True,
        grid=tuple(levels),
    )


__all__ = [
    "Bound",
    "Certificate",
    "bentkus_ucb",
    "certify",
    "hoeffding_ucb",
    "ucb",
]
