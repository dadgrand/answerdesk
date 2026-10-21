"""Split conformal calibration with the finite-sample (n+1) correction.

The object of the game (requirements §8.1): with a target risk ``α``, decide below which
risk score ``s = 1 − p̂`` an answer may be shown automatically. The conformal quantile is

```
rank = ⌈(n + 1)(1 − α)⌉                     # the +1 correction: finite-sample, distribution-free
q̂    = the rank-th smallest of {s_1 … s_n}  # inf when rank > n (not enough calibration data)
```

Two facts are kept strictly apart here, because conflating them is how "calibrated" systems
ship uncalibrated (this distinction is derived in docs/escalation.md §1):

1. **Coverage (distribution-free).** With exchangeable calibration and test points,
   ``P(s_new ≤ q̂) ≥ 1 − α``. That is the split-conformal guarantee, and it is what the
   ``n+1`` rank buys. It says nothing about *correctness* yet.
2. **Risk (needs a calibrated scorer).** If ``p̂`` is calibrated, then among the selected
   points ``E[error | selected] = E[1 − p̂ | selected] ≤ 1 − threshold``. The decision layer
   therefore answers only above ``1 − α_final`` (``thresholds.py``), and the artifact
   reports ECE/Brier so the calibration half of the argument is measured, not assumed.

When ``rank > n`` the certificate is **infeasible**: there is not enough data to place the
quantile, and the honest behaviour is to answer nothing at that α (the decision layer
escalates). ``minimum_calibration_size`` states how many points each α needs.

    # AT-ESC-02  # AT-ESC-06
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import Field

from answerdesk.contracts import Contract, IntentBucket
from answerdesk.policy.errors import CalibrationError

#: Refuse an empty or absurd calibration set early with an actionable message.
MIN_CALIBRATION_FOR_ANY_ALPHA = 1


def conformal_rank(n: int, alpha: float) -> int:
    """``⌈(n + 1)(1 − α)⌉`` — 1-based rank of the conformal quantile."""
    if not 0.0 < alpha < 1.0:
        raise CalibrationError(f"alpha must be in (0, 1), got {alpha}")
    if n < 0:
        raise CalibrationError("n must be >= 0")
    return math.ceil((n + 1) * (1.0 - alpha))


def minimum_calibration_size(alpha: float) -> int:
    """Smallest ``n`` for which ``conformal_rank(n, alpha) <= n`` (α=0.05 → 19, 0.01 → 99)."""
    if not 0.0 < alpha < 1.0:
        raise CalibrationError(f"alpha must be in (0, 1), got {alpha}")
    n = 1
    while conformal_rank(n, alpha) > n:
        n += 1
    return n


class ConformalQuantile(Contract):
    """The quantile itself, with the arithmetic that produced it."""

    q_hat: float = Field(description="threshold on the risk score; +inf when infeasible")
    rank: int = Field(ge=1)
    n: int = Field(ge=0)
    alpha: float = Field(gt=0.0, lt=1.0)
    feasible: bool = Field(description="False when rank > n: not enough calibration data")


def conformal_quantile(risk_scores: Sequence[float], alpha: float) -> ConformalQuantile:
    """Compute ``q̂`` from risk scores, returning ``inf`` when the rank is unreachable."""
    if not risk_scores:
        raise CalibrationError("cannot calibrate on an empty score set")
    values = sorted(float(score) for score in risk_scores)
    for score in values:
        if not 0.0 <= score <= 1.0:
            raise CalibrationError(f"risk scores must be in [0, 1], got {score}")
    n = len(values)
    rank = conformal_rank(n, alpha)
    if rank > n:
        return ConformalQuantile(q_hat=math.inf, rank=rank, n=n, alpha=alpha, feasible=False)
    return ConformalQuantile(q_hat=values[rank - 1], rank=rank, n=n, alpha=alpha, feasible=True)


class ConformalCertificate(Contract):
    """One bucket's calibration result, bound to the data it was fit on."""

    bucket: IntentBucket
    alpha: float = Field(gt=0.0, lt=1.0)
    n: int = Field(ge=0)
    rank: int = Field(ge=1)
    q_hat: float
    feasible: bool
    coverage_lower_bound: float = Field(ge=0.0, le=1.0)
    calibration_hash: str = Field(min_length=8)
    scorer_name: str = ""
    scorer_version: str = ""
    notes: str = ""

    @property
    def description(self) -> str:
        state = "feasible" if self.feasible else "INFEASIBLE (insufficient calibration data)"
        return (
            f"{self.bucket.value}: alpha={self.alpha}, n={self.n}, rank={self.rank}, "
            f"q_hat={self.q_hat:.6f} — {state}"
        )


class ConformalSet(Contract):
    """Certificates for every calibrated bucket (the policy artifact's core)."""

    certificates: list[ConformalCertificate] = Field(default_factory=list)

    def for_bucket(self, bucket: IntentBucket) -> ConformalCertificate:
        for certificate in self.certificates:
            if certificate.bucket is bucket:
                return certificate
        raise CalibrationError(
            f"no conformal certificate for bucket {bucket.value!r}; calibrate every bucket "
            "the policy can route to (requirements §8.2)"
        )

    def conforming(self, risk_score: float, bucket: IntentBucket) -> bool:
        """Whether ``s = 1 − p̂`` is at or below ``q̂`` for this bucket."""
        certificate = self.for_bucket(bucket)
        if not certificate.feasible:
            return False
        return risk_score <= certificate.q_hat

    def buckets(self) -> list[IntentBucket]:
        return [certificate.bucket for certificate in self.certificates]

    def infeasible_buckets(self) -> list[IntentBucket]:
        return [c.bucket for c in self.certificates if not c.feasible]


def fit_certificate(
    risk_scores: Sequence[float],
    *,
    bucket: IntentBucket,
    alpha: float,
    calibration_hash: str,
    scorer_name: str = "",
    scorer_version: str = "",
    notes: str = "",
) -> ConformalCertificate:
    """Calibrate one bucket: quantile + provenance."""
    quantile = conformal_quantile(risk_scores, alpha)
    return ConformalCertificate(
        bucket=bucket,
        alpha=alpha,
        n=quantile.n,
        rank=quantile.rank,
        q_hat=quantile.q_hat,
        feasible=quantile.feasible,
        coverage_lower_bound=1.0 - alpha,
        calibration_hash=calibration_hash,
        scorer_name=scorer_name,
        scorer_version=scorer_version,
        notes=notes
        or (
            "conformal quantile on 1 - p_hat; risk guarantee additionally requires a "
            "calibrated scorer (see docs/escalation.md §1)"
        ),
    )


def fit_certificates(
    scores_by_bucket: Mapping[IntentBucket, Sequence[float]],
    alphas: Mapping[IntentBucket, float],
    *,
    calibration_hash: str,
    scorer_name: str = "",
    scorer_version: str = "",
) -> ConformalSet:
    """Calibrate every bucket present in ``scores_by_bucket`` at its own α."""
    certificates: list[ConformalCertificate] = []
    for bucket in IntentBucket:
        scores = scores_by_bucket.get(bucket)
        if not scores:
            continue
        alpha = alphas[bucket]
        certificates.append(
            fit_certificate(
                scores,
                bucket=bucket,
                alpha=alpha,
                calibration_hash=calibration_hash,
                scorer_name=scorer_name,
                scorer_version=scorer_version,
            )
        )
    if not certificates:
        raise CalibrationError("no bucket had any calibration score")
    return ConformalSet(certificates=certificates)


__all__ = [
    "MIN_CALIBRATION_FOR_ANY_ALPHA",
    "ConformalCertificate",
    "ConformalQuantile",
    "ConformalSet",
    "conformal_quantile",
    "conformal_rank",
    "fit_certificate",
    "fit_certificates",
    "minimum_calibration_size",
]
