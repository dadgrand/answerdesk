"""FastAPI service: SSE chat, sessions, admin ingestion, health and metrics (t9).

Owning task: **t9** (service + observability). Requirements FR-SVC-01 … FR-SVC-08,
NFR-LAT-01/03, NFR-COST-03, NFR-OBS-01/02, NFR-OPS-01; contract in ``docs/api.md``
and ``docs/ops.md``.

The package is *lazy*: importing it never imports FastAPI, opens SQLite or builds an index.
The base install must stay importable without the ``api`` extra (NFR-PORT-03), and a CLI
command that merely lists subcommands must not pay for a web framework. Every public name
below is resolved on first attribute access:

=================================  =====================================================
``create_app``                     FastAPI application factory (lifespan wiring)
``export_openapi`` / ``openapi_schema``  schema export without starting the service
``run_server``                     uvicorn runner + ``--healthcheck`` probe
``ServiceRuntime``                 the wired pipeline (retrieval, generation, policy,
                                   cache, sessions, tracing, metrics)
``TurnRunner``                     one customer turn, in order
``SQLiteSessionStore``             sessions/turns/feedback/tickets/idempotency
``RateLimiter`` / ``RateLimited``  per-client token bucket and its 429 error
``ERROR_TAXONOMY``                 the closed error table (status + retryability)
``ServiceSettings``                ``ANSWERDESK_SERVICE_*`` knobs
=================================  =====================================================

Every type that crosses this module's boundary comes from
:mod:`answerdesk.contracts` (the HTTP wire models are in
:mod:`answerdesk.service.schemas` and never leak into the pipeline).
"""

from __future__ import annotations

from typing import Any

_LAZY: dict[str, tuple[str, str]] = {
    "create_app": ("answerdesk.service.app", "create_app"),
    "export_openapi": ("answerdesk.service.app", "export_openapi"),
    "openapi_schema": ("answerdesk.service.app", "openapi_schema"),
    "run_server": ("answerdesk.service.server", "run_server"),
    "ServiceRuntime": ("answerdesk.service.runtime", "ServiceRuntime"),
    "RuntimeInfo": ("answerdesk.service.runtime", "RuntimeInfo"),
    "CalibrationState": ("answerdesk.service.runtime", "CalibrationState"),
    "SERVICE_VERSION": ("answerdesk.service.runtime", "SERVICE_VERSION"),
    "TurnRequest": ("answerdesk.service.turn", "TurnRequest"),
    "TurnResult": ("answerdesk.service.turn", "TurnResult"),
    "TurnRunner": ("answerdesk.service.turn", "TurnRunner"),
    "SQLiteSessionStore": ("answerdesk.service.sessions", "SQLiteSessionStore"),
    "ServiceSettings": ("answerdesk.service.config", "ServiceSettings"),
    "RateLimiter": ("answerdesk.service.ratelimit", "RateLimiter"),
    "RateDecision": ("answerdesk.service.ratelimit", "RateDecision"),
    "ERROR_TAXONOMY": ("answerdesk.service.errors", "ERROR_TAXONOMY"),
    "ServiceError": ("answerdesk.service.errors", "ServiceError"),
    "to_service_error": ("answerdesk.service.errors", "to_service_error"),
    "SSE_HEADERS": ("answerdesk.service.sse", "SSE_HEADERS"),
    "SSE_MEDIA_TYPE": ("answerdesk.service.sse", "SSE_MEDIA_TYPE"),
    "EVENT_ORDER": ("answerdesk.service.sse", "EVENT_ORDER"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access (keeps FastAPI out of the base install)."""
    try:
        module_name, attribute = _LAZY[name]
    except KeyError as exc:  # pragma: no cover - normal Python behaviour
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    from importlib import import_module

    module = import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(_LAZY)
