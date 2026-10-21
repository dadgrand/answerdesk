"""Model cascade: route a turn by confidence and margin, deterministically.

The v2 latency plan is not "use a smaller model everywhere" — that trades quality for speed
blindly. It is "use the smaller model where the answer is easy to be sure about, and the large
model everywhere else", with the decision made from signals the system already computes
*before* generation:

* ``p_hat`` — the calibrated answerability score (policy layer),
* ``margin`` — how far ``p_hat`` sits from the decision band edge (uncertainty),
* the intent bucket/label — a hard-policy turn (account/legal, fraud) always takes the large model,
  because that is where a cheap wrong answer costs the most.

Every rule is data (thresholds live in the policy object, not in the router), the decision is a
total function of its inputs (no clocks, no randomness, no model), and it returns the reason, so
a trace can explain why a turn went to the small arm. `tests/unit/llm/test_cascade.py` pins the
routing table, including the boundaries.

    # AT-LAT-03  # AT-ESC-02
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from answerdesk.contracts import Contract, IntentBucket

ArmMode = Literal["full", "budget"]


class CascadeArm(Contract):
    """One routing target: a model plus the generation mode it is allowed to use."""

    name: str = Field(min_length=1)
    model: str = ""
    preset: str = ""
    mode: ArmMode = "full"
    max_output_tokens: int = Field(default=512, ge=1)
    note: str = ""


class CascadePolicy(Contract):
    """Thresholds for the route. Defaults are the ones measured in docs/latency.md."""

    small: CascadeArm = Field(
        default_factory=lambda: CascadeArm(
            name="small", mode="budget", max_output_tokens=160, note="budget arm"
        )
    )
    large: CascadeArm = Field(
        default_factory=lambda: CascadeArm(name="large", mode="full", max_output_tokens=512)
    )
    #: ``p_hat`` at or above this is "easy": a confident turn may take the small arm.
    easy_threshold: float = Field(default=0.90, gt=0.0, le=1.0)
    #: The small arm is only used when ``p_hat`` exceeds the band edge by at least this margin.
    min_margin: float = Field(default=0.02, ge=0.0, lt=1.0)
    #: Hard-policy buckets never take the small arm.
    hard_buckets: tuple[IntentBucket, ...] = (IntentBucket.ACCOUNT_LEGAL,)
    #: Labels that force the large arm (same vocabulary as the policy guard config).
    hard_labels: tuple[str, ...] = (
        "account",
        "legal",
        "fraud",
        "complaint",
        "kyc",
        "payment",
        "refund",
    )


class CascadeDecision(Contract):
    """The chosen arm and a machine-readable reason (traced, loggable, testable)."""

    arm: CascadeArm
    reason: Literal[
        "hard_policy",
        "low_confidence",
        "narrow_margin",
        "easy",
        "forced",
    ]
    p_hat: float = 0.0
    margin: float = 0.0
    threshold: float = 0.0
    detail: str = ""

    def is_arm(self, arm: CascadeArm) -> bool:
        return self.arm.name == arm.name


class CascadeRouter:
    """Deterministic router: same inputs, same arm, same reason."""

    def __init__(self, policy: CascadePolicy | None = None) -> None:
        self.policy = policy or CascadePolicy()

    # -- helpers ------------------------------------------------------------------
    def _is_hard(self, bucket: IntentBucket, intent_label: str | None) -> bool:
        if bucket in self.policy.hard_buckets:
            return True
        if intent_label is None:
            return False
        label = intent_label.strip().lower().replace("-", "_")
        return label in {entry.strip().lower() for entry in self.policy.hard_labels}

    # -- public -------------------------------------------------------------------
    def route(
        self,
        p_hat: float,
        *,
        bucket: IntentBucket = IntentBucket.INFORMATIONAL,
        intent_label: str | None = None,
        margin: float | None = None,
        force: str | None = None,
    ) -> CascadeDecision:
        """Pick an arm. ``margin`` defaults to ``p_hat - easy_threshold``.

        ``force`` is the operator override (``"small"``/``"large"``); it is honoured last and
        recorded as the reason, so a forced arm is never mistaken for a measured one.
        """
        policy = self.policy
        value = max(0.0, min(1.0, float(p_hat)))
        resolved_margin = (value - policy.easy_threshold) if margin is None else float(margin)

        if force is not None:
            arm = policy.small if force == "small" else policy.large
            return CascadeDecision(
                arm=arm,
                reason="forced",
                p_hat=value,
                margin=round(resolved_margin, 6),
                threshold=policy.easy_threshold,
                detail=f"operator forced arm {force!r}",
            )

        if self._is_hard(bucket, intent_label):
            return CascadeDecision(
                arm=policy.large,
                reason="hard_policy",
                p_hat=value,
                margin=round(resolved_margin, 6),
                threshold=policy.easy_threshold,
                detail=(
                    f"bucket={bucket.value}"
                    + (f", label={intent_label}" if intent_label else "")
                    + " always takes the large arm"
                ),
            )

        if value < policy.easy_threshold:
            return CascadeDecision(
                arm=policy.large,
                reason="low_confidence",
                p_hat=value,
                margin=round(resolved_margin, 6),
                threshold=policy.easy_threshold,
                detail=f"p_hat {value:.4f} < easy threshold {policy.easy_threshold:.2f}",
            )

        if resolved_margin < policy.min_margin:
            return CascadeDecision(
                arm=policy.large,
                reason="narrow_margin",
                p_hat=value,
                margin=round(resolved_margin, 6),
                threshold=policy.easy_threshold,
                detail=(
                    f"margin {resolved_margin:.4f} < {policy.min_margin:.2f}; "
                    "a turn at the band edge keeps the large arm"
                ),
            )

        return CascadeDecision(
            arm=policy.small,
            reason="easy",
            p_hat=value,
            margin=round(resolved_margin, 6),
            threshold=policy.easy_threshold,
            detail=(
                f"p_hat {value:.4f} >= {policy.easy_threshold:.2f} with margin "
                f"{resolved_margin:.4f} >= {policy.min_margin:.2f}"
            ),
        )


__all__ = [
    "ArmMode",
    "CascadeArm",
    "CascadeDecision",
    "CascadePolicy",
    "CascadeRouter",
]
