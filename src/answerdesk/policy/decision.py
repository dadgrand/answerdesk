"""The decision engine: guards → staleness → generation signal → conformity → cost band.

This is the composition point the whole product's safety story rests on, so the order is
fixed and tested (requirements §8.4):

```
1. PII guard / injection guard        -> refuse or escalate
2. hard-policy intent                 -> escalate (HIGH_RISK_INTENT)
3. stale calibration artifact         -> escalate (STALE_CALIBRATION) — §8.6
4. generation could not produce a
   verified, cited answer             -> escalate with the generator's reason code
5. conformity check  s = 1 − p̂ ≤ q̂    -> escalate (LOW_CONFIDENCE)
6. cost-sensitive band                -> answer / clarify / escalate
7. otherwise                          -> answer
```

Two properties are enforced by construction rather than by discipline:

* guards and the generator can only *block* an answer — the engine never upgrades a
  blocked turn to an answer (there is no code path that does);
* every ``Decision`` carries ``alpha = α_final`` and a ``threshold_ref`` that resolves to the
  sealed calibration artifact, so a decision can be reproduced from disk (FR-ESC-06).

The generator signal is consumed through a two-attribute Protocol, not by importing the
generation layer's concrete classes (the t6 seam).

    # AT-ESC-03  # AT-ESC-04  # AT-ESC-05  # AT-ESC-06  # AT-ESC-10
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable

from answerdesk.contracts import (
    Action,
    Contract,
    Decision,
    EscalationDecision,
    IntentBucket,
    Priority,
    ReasonCode,
    RetrievedPassage,
)
from answerdesk.logging import get_logger
from answerdesk.policy.artifact import CalibrationArtifact
from answerdesk.policy.conformal import ConformalSet
from answerdesk.policy.costs import CostSet
from answerdesk.policy.errors import ScorerError, StaleCalibrationError
from answerdesk.policy.guards import (
    GuardConfig,
    GuardVerdict,
    TurnSignals,
    evaluate_guards,
    scan,
)
from answerdesk.policy.thresholds import PolicyThresholds, ThresholdTable

_LOGGER = get_logger(__name__)


@runtime_checkable
class GenerationSignalLike(Protocol):
    """The two fields the policy needs from ``generate``'s signal (structural seam)."""

    action: Action
    reason_code: ReasonCode
    detail: str


#: Generator reason codes that must survive into the policy decision unchanged.
_PASS_THROUGH = frozenset(
    {
        ReasonCode.NO_EVIDENCE,
        ReasonCode.VERIFIER_REJECT,
        ReasonCode.CITATION_FAILED,
        ReasonCode.INJECTION_SUSPECTED,
        ReasonCode.PROVIDER_ERROR,
        ReasonCode.OUT_OF_SCOPE,
        ReasonCode.UNSUPPORTED_LANGUAGE,
        ReasonCode.CORPUS_MISSING,
        ReasonCode.PII_DETECTED,
        ReasonCode.HIGH_RISK_INTENT,
        ReasonCode.LOW_CONFIDENCE,
    }
)

StaleAction = Literal["escalate", "refuse", "warn"]


class PolicyOutcome(Contract):
    """The decision plus everything the trace and the console need about it."""

    decision: Decision
    thresholds: PolicyThresholds
    signals: TurnSignals
    redacted_query: str = ""
    guard_reason: ReasonCode | None = None
    notes: str = ""

    @property
    def answered(self) -> bool:
        return self.decision.action is Action.ANSWER


class DecisionEngine:
    """Turns ``p̂`` and the guards into a contract :class:`Decision`."""

    def __init__(
        self,
        *,
        thresholds: ThresholdTable,
        conformal: ConformalSet,
        cost_set: CostSet,
        guards: GuardConfig | None = None,
        artifact_hash: str = "",
        stale: bool = False,
        stale_action: StaleAction = "escalate",
        corpus_version: str = "",
    ) -> None:
        self.thresholds = thresholds
        self.conformal = conformal
        self.cost_set = cost_set
        self.guards = guards or GuardConfig()
        self.artifact_hash = artifact_hash
        self.stale = stale
        self.stale_action = stale_action
        self.corpus_version = corpus_version

    # -- construction -------------------------------------------------------------
    @classmethod
    def from_artifact(
        cls,
        artifact: CalibrationArtifact,
        *,
        guards: GuardConfig | None = None,
        now: datetime | None = None,
        corpus_version: str | None = None,
        stale_action: StaleAction | None = None,
        max_age_days: int | None = None,
    ) -> DecisionEngine:
        """Rebuild the engine from a sealed artifact, computing staleness (§8.6)."""
        artifact.verify()
        default_action: StaleAction = (
            guards.stale_calibration.action if guards is not None else "escalate"
        )
        resolved_stale_action: StaleAction = stale_action or default_action
        limit = max_age_days or (guards.stale_calibration.max_age_days if guards else 7)
        stale = artifact.is_stale(now=now, max_age_days=limit, corpus_version=corpus_version)
        if stale:
            _LOGGER.warning(
                "calibration artifact is stale",
                extra={
                    "threshold_ref": artifact.threshold_ref(),
                    "age_days": round(artifact.age_days(now), 2),
                    "max_age_days": limit,
                    "corpus_version_mismatch": bool(
                        corpus_version is not None and artifact.corpus_version != corpus_version
                    ),
                },
            )
        return cls(
            thresholds=artifact.thresholds,
            conformal=artifact.conformal,
            cost_set=artifact.costs,
            guards=guards,
            artifact_hash=artifact.artifact_hash,
            stale=stale,
            stale_action=resolved_stale_action,
            corpus_version=artifact.corpus_version,
        )

    # -- decision -----------------------------------------------------------------
    def evaluate(
        self,
        *,
        p_correct: float,
        bucket: IntentBucket,
        query: str = "",
        passages: Sequence[RetrievedPassage] = (),
        intent_label: str | None = None,
        generation: GenerationSignalLike | None = None,
        signals: TurnSignals | None = None,
    ) -> PolicyOutcome:
        """Full outcome (decision + signals + thresholds); the trace reads this."""
        probability = _validated_probability(p_correct)
        entry = self.thresholds.for_bucket(bucket)
        resolved_signals = signals or scan(
            query,
            passages,
            bucket=bucket,
            intent_label=intent_label,
            config=self.guards,
        )
        notes: list[str] = []

        def decide(
            action: Action, reason: ReasonCode, detail: str, guard_reason: ReasonCode | None = None
        ) -> PolicyOutcome:
            return PolicyOutcome(
                decision=Decision(
                    action=action,
                    reason_code=reason,
                    score=probability,
                    alpha=entry.alpha_final,
                    threshold_ref=self.threshold_ref(),
                ),
                thresholds=entry,
                signals=resolved_signals,
                redacted_query=resolved_signals.redacted_query,
                guard_reason=guard_reason,
                notes="; ".join([*notes, detail]).strip("; "),
            )

        verdict: GuardVerdict | None = evaluate_guards(resolved_signals, self.guards)
        if verdict is not None:
            return decide(verdict.action, verdict.reason_code, verdict.detail, verdict.reason_code)

        if self.stale and self.stale_action != "warn":
            action = Action.ESCALATE if self.stale_action == "escalate" else Action.REFUSE
            return decide(
                action,
                ReasonCode.STALE_CALIBRATION,
                "calibration artifact is stale: refusing to answer as if calibrated (§8.6)",
            )
        if self.stale:
            notes.append("stale calibration artifact: warning only (configured)")

        if generation is not None and generation.action is not Action.ANSWER:
            reason = (
                generation.reason_code
                if generation.reason_code in _PASS_THROUGH
                else ReasonCode.LOW_CONFIDENCE
            )
            return decide(
                generation.action,
                reason,
                f"generator declined: {generation.detail or reason.value}",
            )

        if generation is None and not passages:
            return decide(
                Action.ESCALATE,
                ReasonCode.NO_EVIDENCE,
                "no passages and no generator signal: nothing supports an answer",
            )

        if not entry.coverage_feasible:
            return decide(
                Action.ESCALATE,
                ReasonCode.LOW_CONFIDENCE,
                "conformal certificate is infeasible at this alpha (not enough calibration data)",
            )

        risk_score = 1.0 - probability
        if not self.conformal.conforming(risk_score, bucket):
            return decide(
                Action.ESCALATE,
                ReasonCode.LOW_CONFIDENCE,
                f"risk score {risk_score:.4f} exceeds q_hat {entry.q_hat:.4f} "
                f"(alpha={entry.alpha_bucket})",
            )

        if probability >= entry.answer_min_probability:
            return decide(
                Action.ANSWER,
                ReasonCode.ANSWERED,
                f"p_hat {probability:.4f} >= answer threshold {entry.answer_min_probability:.4f}",
            )
        if probability >= entry.clarify_min_probability:
            return decide(
                Action.CLARIFY,
                ReasonCode.CLARIFY_AMBIGUOUS,
                f"p_hat {probability:.4f} in clarify band "
                f"[{entry.clarify_min_probability:.4f}, {entry.answer_min_probability:.4f})",
            )
        return decide(
            Action.ESCALATE,
            ReasonCode.LOW_CONFIDENCE,
            f"p_hat {probability:.4f} < escalate threshold {entry.clarify_min_probability:.4f}",
        )

    # -- helpers ------------------------------------------------------------------
    def threshold_ref(self) -> str:
        if self.artifact_hash:
            return f"cal-{self.artifact_hash[:12]}"
        return f"cal-{self.thresholds.calibration_hash[:12]}"

    def to_escalation(
        self,
        decision: Decision,
        *,
        priority: Priority = Priority.NORMAL,
        detail: str | None = None,
    ) -> EscalationDecision:
        """Wrap a non-answer decision in the operator-facing escalation record."""
        return EscalationDecision(
            decision=decision,
            priority=priority,
            detail=detail or f"{decision.reason_code.value}: {decision.threshold_ref}",
        )

    def refresh_staleness(
        self,
        artifact: CalibrationArtifact,
        *,
        now: datetime | None = None,
        corpus_version: str | None = None,
        max_age_days: int = 7,
    ) -> bool:
        """Recompute staleness against a fresh artifact (used by the weekly refresh job)."""
        self.stale = artifact.is_stale(
            now=now or datetime.now(UTC),
            max_age_days=max_age_days,
            corpus_version=corpus_version,
        )
        return self.stale


def guards_stale_action(guards: GuardConfig | None) -> StaleAction:
    if guards is None:
        return "escalate"
    return guards.stale_calibration.action


def _validated_probability(value: float) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ScorerError(f"p_correct must be a number, got {value!r}") from exc
    if math.isnan(probability) or math.isinf(probability):
        raise ScorerError(f"p_correct must be finite, got {probability}")
    return min(1.0, max(0.0, probability))


def assert_calibrated_engine(engine: DecisionEngine) -> None:
    """Raise ``StaleCalibrationError`` when the engine would answer without a fresh artifact."""
    if engine.stale and engine.stale_action != "warn":
        raise StaleCalibrationError(
            f"calibration artifact {engine.threshold_ref()} is stale "
            f"(corpus_version={engine.corpus_version!r}); refresh it before answering"
        )


__all__ = [
    "DecisionEngine",
    "GenerationSignalLike",
    "PolicyOutcome",
    "StaleAction",
    "assert_calibrated_engine",
    "guards_stale_action",
]
