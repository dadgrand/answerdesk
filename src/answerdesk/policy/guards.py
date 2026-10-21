"""Guard composition: deterministic, total, and incapable of forcing an answer.

Requirements §8.4 fixes the precedence and forbids the obvious failure modes:

```
1. PII guard / injection guard   -> refuse or escalate   (PII_DETECTED / INJECTION_SUSPECTED)
2. Hard-policy intent            -> escalate             (HIGH_RISK_INTENT)
3. No usable evidence / verifier reject                  (handled by the decision engine)
4. Conformity check                                       (LOW_CONFIDENCE)
5. Cost-sensitive band                                    (answer / clarify / escalate)
```

This module owns steps 1–2 (and the PII redaction that must happen before anything leaves
the process). It returns a :class:`GuardVerdict` — never an answer — so "a guard cannot
force an answer" is a type-level fact, not a comment.

The guard *policy* (which action each signal forces) lives in ``configs/policy/guards.yaml``
and is validated here; a config that cannot be honoured is a startup error, not a silent
downgrade.

    # AT-ESC-04  # AT-ESC-08  # AT-ESC-09
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from answerdesk.contracts import (
    Action,
    Contract,
    IntentBucket,
    ReasonCode,
    RetrievedPassage,
)
from answerdesk.generate.guards import InjectionFinding
from answerdesk.policy.errors import GuardError
from answerdesk.policy.injection import scan_turn
from answerdesk.policy.pii import PIIFinding, RedactionResult, redact

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Default guard policy file (a code constant, not a gated config key).
DEFAULT_GUARDS_PATH = Path("configs") / "policy" / "guards.yaml"


class GuardActionConfig(Contract):
    """What a specific guard forces when it fires."""

    action: Literal["escalate", "refuse"] = "escalate"
    redact_in_answer: bool = True


class HighRiskConfig(Contract):
    """Which intents are hard-policy (§8.4 step 2)."""

    intent_labels: list[str] = Field(default_factory=list)
    buckets: list[IntentBucket] = Field(default_factory=lambda: [IntentBucket.ACCOUNT_LEGAL])

    def normalised_labels(self) -> set[str]:
        return {label.strip().lower() for label in self.intent_labels if label.strip()}


class StaleCalibrationConfig(Contract):
    """What happens when the calibration artifact is stale (§8.6)."""

    action: Literal["escalate", "refuse", "warn"] = "escalate"
    max_age_days: int = Field(default=7, ge=1)


class GuardConfig(Contract):
    """The whole guard set."""

    pii: GuardActionConfig = Field(default_factory=GuardActionConfig)
    injection: GuardActionConfig = Field(default_factory=GuardActionConfig)
    high_risk: HighRiskConfig = Field(default_factory=HighRiskConfig)
    stale_calibration: StaleCalibrationConfig = Field(default_factory=StaleCalibrationConfig)


def default_guards_path() -> Path:
    return _REPO_ROOT / DEFAULT_GUARDS_PATH


def load_guard_config(path: str | Path | None = None) -> GuardConfig:
    file_path = Path(path) if path is not None else default_guards_path()
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GuardError(f"cannot read guard config {file_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise GuardError(f"guard config {file_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise GuardError(f"guard config {file_path} must be a mapping")
    try:
        return GuardConfig.model_validate(raw)
    except Exception as exc:
        raise GuardError(f"guard config {file_path} is invalid: {exc}") from exc


class TurnSignals(Contract):
    """What the guards found in one turn, plus the redacted query for the trace."""

    pii: list[PIIFinding] = Field(default_factory=list)
    injection: list[InjectionFinding] = Field(default_factory=list)
    high_risk: bool = False
    redacted_query: str = ""

    @property
    def any_pii(self) -> bool:
        return bool(self.pii)

    @property
    def any_injection(self) -> bool:
        return bool(self.injection)

    def pii_kinds(self) -> list[str]:
        return sorted({finding.kind.value for finding in self.pii})


class GuardVerdict(Contract):
    """A guard's forced outcome. ``action`` can only be ``escalate`` or ``refuse``."""

    action: Action
    reason_code: ReasonCode
    detail: str = ""

    @model_validator(mode="after")
    def _cannot_answer(self) -> GuardVerdict:
        if self.action not in (Action.ESCALATE, Action.REFUSE):
            raise ValueError(
                "a guard may only escalate or refuse; nothing may force an answer (§8.4)"
            )
        return self


def _as_verdict(config: GuardActionConfig, reason: ReasonCode, detail: str) -> GuardVerdict:
    action = Action.ESCALATE if config.action == "escalate" else Action.REFUSE
    return GuardVerdict(action=action, reason_code=reason, detail=detail)


def is_hard_policy(bucket: IntentBucket, intent_label: str | None, config: GuardConfig) -> bool:
    """Whether the turn belongs to the hard-policy risk tier (§8.4 step 2)."""
    if bucket in config.high_risk.buckets:
        return True
    if not intent_label:
        return False
    return intent_label.strip().lower() in config.high_risk.normalised_labels()


def scan(
    query: str,
    passages: Sequence[RetrievedPassage],
    *,
    bucket: IntentBucket,
    intent_label: str | None = None,
    config: GuardConfig | None = None,
) -> TurnSignals:
    """Run the guards on one turn: PII redaction, injection scan, risk-tier lookup."""
    resolved = config or GuardConfig()
    redaction: RedactionResult = redact(query)
    findings = scan_turn(query, passages)
    return TurnSignals(
        pii=redaction.findings,
        injection=findings,
        high_risk=is_hard_policy(bucket, intent_label, resolved),
        redacted_query=redaction.text,
    )


def evaluate_guards(signals: TurnSignals, config: GuardConfig | None = None) -> GuardVerdict | None:
    """Apply the fixed precedence; returns ``None`` when no guard fires."""
    resolved = config or GuardConfig()
    if signals.pii:
        kinds = ",".join(signals.pii_kinds())
        return _as_verdict(
            resolved.pii,
            ReasonCode.PII_DETECTED,
            f"PII detected in the customer turn ({kinds}); redaction is mandatory (INV-05)",
        )
    if signals.injection:
        sources = sorted({finding.source for finding in signals.injection})
        rules = sorted({rule for finding in signals.injection for rule in finding.rules})
        return _as_verdict(
            resolved.injection,
            ReasonCode.INJECTION_SUSPECTED,
            f"instruction-shaped content in {','.join(sources)} (rules: {','.join(rules)})",
        )
    if signals.high_risk:
        return GuardVerdict(
            action=Action.ESCALATE,
            reason_code=ReasonCode.HIGH_RISK_INTENT,
            detail="hard-policy risk tier: a human must handle this turn (§8.4 step 2)",
        )
    return None


def forced_reason_code(verdict: GuardVerdict) -> ReasonCode:
    """The reason code a forced verdict carries (kept as a function for the parity test)."""
    return verdict.reason_code


__all__ = [
    "DEFAULT_GUARDS_PATH",
    "GuardActionConfig",
    "GuardConfig",
    "GuardVerdict",
    "HighRiskConfig",
    "StaleCalibrationConfig",
    "TurnSignals",
    "default_guards_path",
    "evaluate_guards",
    "forced_reason_code",
    "is_hard_policy",
    "load_guard_config",
    "scan",
]
