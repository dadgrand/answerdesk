"""Calibrated escalation: conformal quantile, cost model, PII and injection guards.

Owning task: **t8**. Public surface for the rest of the stack (t9 service, t10 console,
t7/t11 evaluation):

=================================  =====================================================
:class:`ConformalSet`              per-bucket ``q̂`` with the (n+1) finite-sample correction
:class:`CostSet`                   the cost matrix and the bands it implies (no literals)
:class:`ThresholdTable`            conformal ∩ cost ∩ bucket-α, with ``threshold_ref``
:class:`CalibrationArtifact`       sealed, content-addressed calibration state
:class:`DecisionEngine`            guards → staleness → generator → conformity → band
:class:`OverlapScorer`             deterministic answerability baseline
:class:`LLMJudgeScorer`            rubric judge behind the ``ChatProvider`` protocol
``detect`` / ``redact``            PII detection and redaction (phones, e-mail, card, account)
``scan_turn`` / ``evaluate_guards`` injection heuristics and total guard precedence
=================================  =====================================================

Requirements: FR-ESC-01 … FR-ESC-10; derivations in ``docs/escalation.md``.

Every type that crosses this module's boundary comes from :mod:`answerdesk.contracts`.
"""

from __future__ import annotations

from answerdesk.policy.artifact import BucketMetrics, CalibrationArtifact, PooledMetrics
from answerdesk.policy.conformal import (
    ConformalCertificate,
    ConformalQuantile,
    ConformalSet,
    conformal_quantile,
    conformal_rank,
    fit_certificate,
    fit_certificates,
    minimum_calibration_size,
)
from answerdesk.policy.costs import (
    CostProvenance,
    CostSet,
    check_cost_drift,
    cost_set_from_settings,
    load_cost_set,
)
from answerdesk.policy.decision import (
    DecisionEngine,
    GenerationSignalLike,
    PolicyOutcome,
    assert_calibrated_engine,
)
from answerdesk.policy.errors import (
    ArtifactError,
    CalibrationError,
    CostModelError,
    GuardError,
    PolicyError,
    ScorerError,
    StaleCalibrationError,
)
from answerdesk.policy.guards import (
    GuardConfig,
    GuardVerdict,
    TurnSignals,
    evaluate_guards,
    is_hard_policy,
    load_guard_config,
    scan,
)
from answerdesk.policy.injection import (
    detect_policy_injection,
    rule_names,
    scan_passages_policy,
    scan_query_policy,
    scan_turn,
)
from answerdesk.policy.metrics import (
    ReliabilityBin,
    RiskCoveragePoint,
    aurc,
    brier_score,
    clopper_pearson_upper,
    expected_calibration_error,
    reliability_bins,
    risk_at_coverage,
    risk_coverage_curve,
    wilson_interval,
    wilson_upper,
)
from answerdesk.policy.pii import (
    PIIFinding,
    PIIKind,
    RedactionResult,
    contains_pii,
    detect,
    luhn_valid,
    redact,
    redact_messages,
    scan_messages,
)
from answerdesk.policy.scorers import (
    JUDGE_RUBRIC_VERSION,
    LLMJudgeScorer,
    OverlapScorer,
    content_tokens,
)
from answerdesk.policy.thresholds import (
    PolicyThresholds,
    ThresholdTable,
    build_threshold_table,
    derive_thresholds,
)

__all__ = [
    "JUDGE_RUBRIC_VERSION",
    "ArtifactError",
    "BucketMetrics",
    "CalibrationArtifact",
    "CalibrationError",
    "ConformalCertificate",
    "ConformalQuantile",
    "ConformalSet",
    "CostModelError",
    "CostProvenance",
    "CostSet",
    "DecisionEngine",
    "GenerationSignalLike",
    "GuardConfig",
    "GuardError",
    "GuardVerdict",
    "LLMJudgeScorer",
    "OverlapScorer",
    "PIIFinding",
    "PIIKind",
    "PolicyError",
    "PolicyOutcome",
    "PolicyThresholds",
    "PooledMetrics",
    "RedactionResult",
    "ReliabilityBin",
    "RiskCoveragePoint",
    "ScorerError",
    "StaleCalibrationError",
    "ThresholdTable",
    "TurnSignals",
    "assert_calibrated_engine",
    "aurc",
    "brier_score",
    "build_threshold_table",
    "check_cost_drift",
    "clopper_pearson_upper",
    "conformal_quantile",
    "conformal_rank",
    "contains_pii",
    "content_tokens",
    "cost_set_from_settings",
    "derive_thresholds",
    "detect",
    "detect_policy_injection",
    "evaluate_guards",
    "expected_calibration_error",
    "fit_certificate",
    "fit_certificates",
    "is_hard_policy",
    "load_cost_set",
    "load_guard_config",
    "luhn_valid",
    "minimum_calibration_size",
    "redact",
    "redact_messages",
    "reliability_bins",
    "risk_at_coverage",
    "risk_coverage_curve",
    "rule_names",
    "scan",
    "scan_messages",
    "scan_passages_policy",
    "scan_query_policy",
    "scan_turn",
    "wilson_interval",
    "wilson_upper",
]
