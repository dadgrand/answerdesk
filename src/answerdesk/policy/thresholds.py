"""Composing the three axes into one threshold table.

Three constraints meet at the answer decision (requirements §8.2–§8.4):

* the **conformal axis** — ``s = 1 − p̂ ≤ q̂`` (the coverage certificate);
* the **cost axis** — ``p̂ ≥ 1 − C_delay/C_wrong`` for an answer and
  ``p̂ ≥ 1 − C_abstain/C_wrong`` for a clarifying question (§8.3);
* the **risk-tier axis** — a stricter bucket ``α`` must tighten the answer threshold, because
  only an *answer* carries risk; clarifying and escalating are safe actions.

So, per bucket (``α_final = min(α_bucket, C_abstain/C_wrong)``):

```
answer_min_p  = max(1 − C_delay/C_wrong, 1 − α_final)
clarify_min_p = 1 − C_abstain/C_wrong
```

At the reference defaults both answer edges coincide (0.95), which is the coincidence
requirements §8.3 notes ("Setting the wrong/abstain ratio to 10 makes the operating point
p̂ ≥ 0.90 on the conformal axis"). For the account/legal bucket (α = 0.01) the α axis is the
binding one (0.99), exactly as §8.2 demands ("the cost model can tighten α but never
loosen it"). The derivation is in docs/escalation.md §3.

    # AT-ESC-03  # AT-ESC-06
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field

from answerdesk.contracts import Contract, IntentBucket
from answerdesk.policy.conformal import ConformalCertificate, ConformalSet
from answerdesk.policy.costs import CostSet


class PolicyThresholds(Contract):
    """Every threshold one bucket uses, with the inputs that produced them."""

    bucket: IntentBucket
    alpha_bucket: float = Field(gt=0.0, lt=1.0)
    alpha_final: float = Field(gt=0.0, lt=1.0)
    q_hat: float
    coverage_feasible: bool
    # 1.0 is a legitimate degenerate value: `C_delay = 0` means “clarifying is free”, so
    # nothing is ever confident enough to answer outright. The band then never fires.
    answer_min_probability: float = Field(gt=0.0, le=1.0)
    clarify_min_probability: float = Field(gt=0.0, le=1.0)
    conformal_coverage_lower_bound: float = Field(ge=0.0, le=1.0)

    @property
    def answer_min_risk_score(self) -> float:
        return 1.0 - self.answer_min_probability

    @property
    def clarify_min_risk_score(self) -> float:
        return 1.0 - self.clarify_min_probability


class ThresholdTable(Contract):
    """Per-bucket thresholds plus the provenance a decision cites as ``threshold_ref``."""

    thresholds: list[PolicyThresholds] = Field(default_factory=list)
    cost_hash: str = Field(min_length=8)
    calibration_hash: str = Field(min_length=8)
    scorer_name: str = ""
    scorer_version: str = ""

    def for_bucket(self, bucket: IntentBucket) -> PolicyThresholds:
        for entry in self.thresholds:
            if entry.bucket is bucket:
                return entry
        raise KeyError(f"no thresholds for bucket {bucket.value!r}")

    def threshold_ref(self, artifact_hash: str) -> str:
        """The string a :class:`answerdesk.contracts.Decision` records (artifact-addressed)."""
        return f"cal-{artifact_hash[:12]}"

    def buckets(self) -> list[IntentBucket]:
        return [entry.bucket for entry in self.thresholds]


def derive_thresholds(
    *, certificate: ConformalCertificate, cost_set: CostSet, alpha_bucket: float
) -> PolicyThresholds:
    """One bucket: conformal quantile + cost bands + the bucket's risk ceiling."""
    alpha_final = min(alpha_bucket, cost_set.alpha_from_costs)
    cost_answer_edge = cost_set.answer_min_probability
    cost_clarify_edge = cost_set.clarify_min_probability
    alpha_answer_edge = 1.0 - alpha_final
    answer_min = max(cost_answer_edge, alpha_answer_edge)
    clarify_min = cost_clarify_edge
    if clarify_min >= answer_min:
        # Degenerate configuration: the clarify band would be empty. Keep it monotone and
        # let validation surface the cause (cost or alpha), never silently widen "answer".
        clarify_min = answer_min
    return PolicyThresholds(
        bucket=certificate.bucket,
        alpha_bucket=alpha_bucket,
        alpha_final=alpha_final,
        q_hat=certificate.q_hat,
        coverage_feasible=certificate.feasible,
        answer_min_probability=round(answer_min, 12),
        clarify_min_probability=round(clarify_min, 12),
        conformal_coverage_lower_bound=certificate.coverage_lower_bound,
    )


def build_threshold_table(
    *,
    conformal_set: ConformalSet,
    cost_set: CostSet,
    alpha_by_bucket: Mapping[IntentBucket, float],
    calibration_hash: str,
) -> ThresholdTable:
    """Compose every calibrated bucket into the table the decision engine consumes."""
    thresholds = [
        derive_thresholds(
            certificate=certificate,
            cost_set=cost_set,
            alpha_bucket=alpha_by_bucket[certificate.bucket],
        )
        for certificate in conformal_set.certificates
    ]
    return ThresholdTable(
        thresholds=thresholds,
        cost_hash=cost_set.hash(),
        calibration_hash=calibration_hash,
        scorer_name=conformal_set.certificates[0].scorer_name,
        scorer_version=conformal_set.certificates[0].scorer_version,
    )


__all__ = [
    "PolicyThresholds",
    "ThresholdTable",
    "build_threshold_table",
    "derive_thresholds",
]
