"""The service error taxonomy: one closed table, one envelope, one status mapping.

Requirements FR-SVC-03 asks for a *consistent* taxonomy, which in practice means three
properties this module enforces mechanically:

1. **Closed** — every code the service can emit appears in :data:`ERROR_TAXONOMY` with its
   HTTP status, its retryability and a one-line meaning. A test walks the table and asserts
   each row round-trips through the envelope; a code not in the table cannot be produced by
   a :class:`ServiceError` (the constructor refuses).
2. **Lossless transport** — the envelope always carries ``code``, ``message``, ``trace_id``
   and ``retryable``; ``detail`` and ``retry_after_s`` are optional and never contain
   secrets (they are produced by our own modules, never copied from a provider body).
3. **Mapped at the edge** — module errors (:mod:`answerdesk.errors`,
   :mod:`answerdesk.llm.errors`, policy and cache errors) are translated in exactly one
   place, :func:`to_service_error`, so a new module error cannot accidentally become a 500
   with a stack trace.

Status mapping is deliberate:

===============================  ======  ==========================================
code                             status  retryable
===============================  ======  ==========================================
``bad_request`` / ``validation_error``  400 / 422  no
``unauthorized`` / ``forbidden`` / ``admin_disabled``  401 / 403 / 403  no
``not_found`` / ``method_not_allowed``  404 / 405  no
``conflict`` / ``idempotency_conflict``  409 / 409  no
``payload_too_large`` / ``unsupported_media_type``  413 / 415  no
``rate_limited``                 429     yes (``Retry-After``)
``provider_error`` / ``provider_unreachable`` / ``provider_bad_response``  502  yes
``provider_timeout``             504     yes
``provider_circuit_open`` / ``service_unavailable``  503  yes
``corpus_missing`` / ``index_missing`` / ``calibration_missing`` / ``stale_calibration``  503  no (operator action)
``contract_violation`` / ``internal_error`` / ``stream_error``  500  no
===============================  ======  ==========================================

    # AT-SVC-03  # AT-OPS-07
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from answerdesk.errors import AnswerdeskError


@dataclass(frozen=True)
class ErrorSpec:
    """One row of the taxonomy table."""

    status: int
    retryable: bool
    meaning: str


#: The closed error table. Adding a code is a documented API change (docs/api.md §4).
ERROR_TAXONOMY: Mapping[str, ErrorSpec] = {
    "bad_request": ErrorSpec(400, False, "The request is syntactically valid but unusable."),
    "validation_error": ErrorSpec(422, False, "A request field failed schema validation."),
    "unauthorized": ErrorSpec(401, False, "A bearer token is required for this endpoint."),
    "forbidden": ErrorSpec(403, False, "The bearer token is not accepted."),
    "admin_disabled": ErrorSpec(403, False, "Admin endpoints are disabled (no token configured)."),
    "not_found": ErrorSpec(404, False, "The referenced resource does not exist."),
    "method_not_allowed": ErrorSpec(405, False, "The route exists but not for this method."),
    "conflict": ErrorSpec(409, False, "The request conflicts with the current state."),
    "idempotency_conflict": ErrorSpec(
        409, False, "The idempotency key was used with a different request body."
    ),
    "payload_too_large": ErrorSpec(413, False, "The request body exceeds the configured limit."),
    "unsupported_media_type": ErrorSpec(415, False, "The request body is not JSON."),
    "rate_limited": ErrorSpec(429, True, "The client exceeded its request budget."),
    "provider_error": ErrorSpec(502, True, "The model provider failed after retries/fallback."),
    "provider_unreachable": ErrorSpec(502, True, "The model provider could not be reached."),
    "provider_bad_response": ErrorSpec(502, True, "The provider returned an unusable payload."),
    "provider_timeout": ErrorSpec(504, True, "The provider exceeded the configured timeout."),
    "provider_circuit_open": ErrorSpec(503, True, "The provider circuit breaker is open."),
    "provider_chain_exhausted": ErrorSpec(502, True, "Every provider in the fallback chain failed."),
    "corpus_missing": ErrorSpec(503, False, "The corpus (or its manifest) is missing."),
    "index_missing": ErrorSpec(503, False, "The index is missing or built for another corpus."),
    "calibration_missing": ErrorSpec(503, False, "No calibration artifact is available."),
    "stale_calibration": ErrorSpec(503, False, "The calibration artifact is stale; re-calibrate."),
    "service_unavailable": ErrorSpec(503, True, "The service is starting or shutting down."),
    "not_implemented": ErrorSpec(501, False, "The surface exists but is not implemented."),
    "contract_violation": ErrorSpec(500, False, "A module produced a contract violation."),
    "internal_error": ErrorSpec(500, False, "An unexpected failure; details are in the logs."),
    "stream_error": ErrorSpec(500, False, "The SSE stream failed after it started."),
}


class ServiceError(AnswerdeskError):
    """An error with a taxonomy-backed code, status and retryability."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        detail: str | None = None,
        retry_after_s: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        spec = ERROR_TAXONOMY.get(code)
        if spec is None:  # a code outside the table is a programming error, not a 500
            raise ValueError(f"unknown service error code {code!r}")
        super().__init__(message, detail=detail)
        self.code = code
        self.http_status = spec.status
        self.retryable = spec.retryable
        self.retry_after_s = retry_after_s
        self.headers = dict(headers or {})

    def envelope(self, trace_id: str) -> dict[str, Any]:
        payload = super().to_envelope(trace_id)
        if self.retry_after_s is not None:
            payload["retry_after_s"] = round(float(self.retry_after_s), 3)
        return payload

    def response_headers(self, trace_id: str) -> dict[str, str]:
        headers = {"X-Trace-Id": trace_id, **self.headers}
        if self.retry_after_s is not None:
            headers["Retry-After"] = str(max(1, int(round(self.retry_after_s))))
        return headers


# -- concrete errors -------------------------------------------------------------------
class BadRequest(ServiceError):
    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__("bad_request", message, detail=detail)


class ValidationFailed(ServiceError):
    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__("validation_error", message, detail=detail)


class Unauthorized(ServiceError):
    def __init__(self, message: str = "a bearer token is required") -> None:
        super().__init__("unauthorized", message, headers={"WWW-Authenticate": "Bearer"})


class Forbidden(ServiceError):
    def __init__(self, message: str = "the bearer token is not accepted") -> None:
        super().__init__("forbidden", message)


class AdminDisabled(ServiceError):
    def __init__(self) -> None:
        super().__init__(
            "admin_disabled",
            "admin endpoints are disabled: set ANSWERDESK_SERVICE_ADMIN_TOKEN to enable them",
        )


class NotFound(ServiceError):
    def __init__(self, what: str) -> None:
        super().__init__("not_found", f"{what} was not found")


class Conflict(ServiceError):
    def __init__(self, message: str) -> None:
        super().__init__("conflict", message)


class IdempotencyConflict(ServiceError):
    def __init__(self, key: str) -> None:
        super().__init__(
            "idempotency_conflict",
            f"idempotency key {key!r} was already used with a different request body",
        )


class RateLimited(ServiceError):
    def __init__(self, retry_after_s: float, *, limit: int) -> None:
        super().__init__(
            "rate_limited",
            f"rate limit exceeded ({limit} requests/minute); retry in {retry_after_s:.1f}s",
            retry_after_s=retry_after_s,
        )


class PayloadTooLarge(ServiceError):
    def __init__(self, limit_bytes: int) -> None:
        super().__init__("payload_too_large", f"request body exceeds {limit_bytes} bytes")


class UnsupportedMediaType(ServiceError):
    def __init__(self) -> None:
        super().__init__("unsupported_media_type", "Content-Type must be application/json")


class ServiceUnavailable(ServiceError):
    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__("service_unavailable", message, detail=detail)


class StreamFailed(ServiceError):
    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__("stream_error", message, detail=detail)


#: Module error codes that are already taxonomy codes.
_DIRECT_CODES = frozenset(ERROR_TAXONOMY)


def to_service_error(exc: BaseException) -> ServiceError:
    """Translate any module error into the taxonomy (the only mapping point)."""
    if isinstance(exc, ServiceError):
        return exc
    if isinstance(exc, AnswerdeskError):
        code = exc.code if exc.code in _DIRECT_CODES else _code_for_status(exc.http_status)
        return ServiceError(code, exc.message, detail=exc.detail)
    if isinstance(exc, TimeoutError):
        return ServiceError("provider_timeout", "the upstream call timed out")
    return ServiceError("internal_error", "internal error", detail=type(exc).__name__)


def _code_for_status(status: int) -> str:
    if status == 501:
        return "not_implemented"
    if status == 503:
        return "service_unavailable"
    if status == 502:
        return "provider_error"
    if status == 504:
        return "provider_timeout"
    if 400 <= status < 500:
        return "bad_request"
    return "internal_error"


def error_envelope(exc: ServiceError, trace_id: str) -> dict[str, Any]:
    return {"error": exc.envelope(trace_id)}


def constant_time_equals(left: str, right: str) -> bool:
    """Compare a presented token with the configured one without a timing oracle."""
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


__all__ = [
    "ERROR_TAXONOMY",
    "AdminDisabled",
    "BadRequest",
    "Conflict",
    "ErrorSpec",
    "Forbidden",
    "IdempotencyConflict",
    "NotFound",
    "PayloadTooLarge",
    "RateLimited",
    "ServiceError",
    "ServiceUnavailable",
    "StreamFailed",
    "Unauthorized",
    "UnsupportedMediaType",
    "ValidationFailed",
    "constant_time_equals",
    "error_envelope",
    "to_service_error",
]
