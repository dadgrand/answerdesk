"""Typed errors for the policy layer.

The escalation path is safety-critical, so every failure mode is explicit and maps onto a
reason code rather than a bare ``ValueError``: a missing calibration artifact must never
degrade into "answer anyway" (requirements §8.6).

    # AT-ESC-06
"""

from __future__ import annotations

from answerdesk.errors import AnswerdeskError


class PolicyError(AnswerdeskError):
    """Base class for escalation-policy failures."""

    code = "policy_error"


class CalibrationError(PolicyError):
    """The calibration set or artifact is unusable (too small, malformed, stale)."""

    code = "calibration_error"


class CostModelError(PolicyError):
    """The cost set is invalid or carries no provenance (requirements §8.3)."""

    code = "cost_model_error"


class ArtifactError(PolicyError):
    """A calibration artifact is missing, corrupt, or does not belong to this corpus."""

    code = "calibration_artifact_error"


class ScorerError(PolicyError):
    """An answerability scorer produced an unusable value."""

    code = "scorer_error"


class GuardError(PolicyError):
    """A guard configuration is invalid (unknown PII policy, missing lexicon, ...)."""

    code = "guard_error"


class StaleCalibrationError(PolicyError):
    """The artifact is stale (age or corpus version) — warn loudly, never answer silently."""

    code = "stale_calibration"
    http_status = 503


__all__ = [
    "ArtifactError",
    "CalibrationError",
    "CostModelError",
    "GuardError",
    "PolicyError",
    "ScorerError",
    "StaleCalibrationError",
]
