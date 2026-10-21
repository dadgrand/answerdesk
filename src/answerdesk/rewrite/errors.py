"""Errors for the optional query-rewriter layer.

A rewriter is an optimisation, never a dependency of a correct answer: every failure here is
caught by the backend and degrades to "use the original query" (see :mod:`answerdesk.rewrite.
backend`). The exception types exist so that a caller who *wants* to know can tell "not
configured" from "configured but broken".

    # AT-GEN-13
"""

from __future__ import annotations

from answerdesk.errors import AnswerdeskError


class RewriteError(AnswerdeskError):
    """Base class for query-rewriter failures."""

    code = "rewrite_error"


class RewriterUnavailable(RewriteError):
    """The selected backend cannot run here (missing package, weights, or adapter)."""

    code = "rewriter_unavailable"
    retryable = False

    def __init__(self, message: str, *, missing: list[str] | None = None) -> None:
        super().__init__(message, detail=", ".join(missing or []))
        self.missing = list(missing or [])


class RewriteConfigError(RewriteError):
    """The rewriter configuration is invalid (unknown backend, bad path, bad knob)."""

    code = "rewrite_config_error"


class DatasetError(RewriteError):
    """Training-data construction failed (missing corpus rows, empty split, bad schema)."""

    code = "dataset_error"


class DatasetLeakageError(DatasetError):
    """A document or a clean query appears in more than one split — refuse to train."""

    code = "dataset_leakage"


class RewriteTimeout(RewriteError):
    """A rewrite exceeded its latency budget; the caller falls back to the original query."""

    code = "rewrite_timeout"
    retryable = True


__all__ = [
    "DatasetError",
    "DatasetLeakageError",
    "RewriteConfigError",
    "RewriteError",
    "RewriteTimeout",
    "RewriterUnavailable",
]
