"""Error taxonomy for the model-access layer.

Everything derives from :class:`answerdesk.errors.ProviderError`, so the service
layer (t9) maps the whole family onto one HTTP status without knowing any subclass
(FR-SVC-03). The subclasses exist so that retry, fallback and the failure-mode
matrix (NFR-OPS-01) can distinguish *why* a provider failed.

    # AT-PORT-01
"""

from __future__ import annotations

from answerdesk.errors import ProviderError


class LLMError(ProviderError):
    """Base class for every failure inside ``answerdesk.llm``."""

    code = "provider_error"


class ProviderHTTPError(LLMError):
    """A provider answered with a non-2xx status (or a broken body)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        retry_after_s: float | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message, detail=detail)
        self.status_code = status_code
        self.retry_after_s = retry_after_s

    @property
    def retryable(self) -> bool:  # type: ignore[override]
        """429 and 5xx are transient; 4xx is the caller's fault and is not retried."""
        return self.status_code in {408, 409, 425, 429} or self.status_code >= 500


class ProviderTimeoutError(LLMError):
    """The request exceeded the configured timeout (NFR-OPS-01 failure row 1)."""

    code = "provider_timeout"


class ProviderConnectionError(LLMError):
    """TCP/TLS/DNS failure reaching the provider (unreachable local server, offline)."""

    code = "provider_unreachable"


class ProviderResponseError(LLMError):
    """The provider answered 2xx with a payload we cannot use (malformed JSON, no choices)."""

    code = "provider_bad_response"


class CircuitOpenError(LLMError):
    """The circuit breaker for this provider is open: fail fast, do not queue."""

    code = "provider_circuit_open"
    retryable = True


class AllProvidersFailedError(LLMError):
    """Every provider in the fallback chain failed; carries the per-provider reasons."""

    code = "provider_chain_exhausted"
    retryable = True

    def __init__(self, message: str, *, failures: list[tuple[str, str]], detail: str = "") -> None:
        super().__init__(message, detail=detail or "; ".join(f"{n}: {r}" for n, r in failures))
        self.failures = failures


class PromptError(LLMError):  # pragma: no cover - raised from the generate layer
    """Reserved name kept out of the way: prompt failures live in ``generate``."""

    code = "prompt_error"


__all__ = [
    "AllProvidersFailedError",
    "CircuitOpenError",
    "LLMError",
    "ProviderConnectionError",
    "ProviderHTTPError",
    "ProviderResponseError",
    "ProviderTimeoutError",
]
