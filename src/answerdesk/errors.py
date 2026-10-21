"""Typed error taxonomy shared by every module.

The HTTP layer (t9) maps these onto status codes and an error envelope; modules raise
them instead of bare ``ValueError`` so that the taxonomy stays closed
(requirements FR-SVC-03, NFR-OPS-01).
"""

from __future__ import annotations

from typing import Any


class AnswerdeskError(Exception):
    """Base class: a failure with a stable machine-readable code."""

    code: str = "answerdesk_error"
    retryable: bool = False
    http_status: int = 500

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_envelope(self, trace_id: str) -> dict[str, Any]:
        """Serialise for an API error response (t9 owns the final envelope shape)."""
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "trace_id": trace_id,
            "retryable": self.retryable,
        }
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class ConfigError(AnswerdeskError):
    """Invalid or incomplete configuration — actionable, raised at startup (FR-OPS-05)."""

    code = "config_error"
    http_status = 500


class ProviderError(AnswerdeskError):
    """A model provider failed after retries/fallback (NFR-OPS-01 failure matrix)."""

    code = "provider_error"
    retryable = True
    http_status = 502


class CorpusMissingError(AnswerdeskError):
    """The corpus or its manifest is missing or unreadable."""

    code = "corpus_missing"
    http_status = 503


class IndexMissingError(AnswerdeskError):
    """The index has not been built for the current corpus version."""

    code = "index_missing"
    http_status = 503


class CalibrationMissingError(AnswerdeskError):
    """No valid calibration artifact: the system must not pretend to be calibrated."""

    code = "calibration_missing"
    http_status = 503


class ContractViolation(AnswerdeskError):
    """A module produced an object that violates ``answerdesk.contracts``."""

    code = "contract_violation"
    http_status = 500


class NotYetImplemented(AnswerdeskError):
    """Skeleton stub: the command exists but its owning task has not implemented it."""

    code = "not_implemented"
    http_status = 501

    def __init__(self, surface: str, owner_task: str) -> None:
        super().__init__(
            f"'{surface}' is part of the frozen CLI/API surface but is implemented by "
            f"task {owner_task}; the skeleton only wires it.",
            detail=f"owning_task={owner_task}",
        )
        self.surface = surface
        self.owner_task = owner_task


__all__ = [
    "AnswerdeskError",
    "CalibrationMissingError",
    "ConfigError",
    "ContractViolation",
    "CorpusMissingError",
    "IndexMissingError",
    "NotYetImplemented",
    "ProviderError",
]
