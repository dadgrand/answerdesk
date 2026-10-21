"""The FastAPI application: lifespan wiring, middleware, error handlers, health/metrics.

The application object is assembled in one function so that tests and operators can build
it with explicit settings and an injected runtime:

    app = create_app(settings=load_settings(profile="offline"), service_settings=ServiceSettings())

Wiring order (which is observable, so it is fixed here):

1. **request-context middleware** mints/accepts ``X-Trace-Id``, measures the wall clock,
   records the request/latency metrics and echoes the header on every response;
2. **body-limit middleware** rejects an oversized payload with a taxonomy ``413`` before it
   is read;
3. **rate-limit middleware** enforces the per-client token bucket on ``/v1`` and ``/admin``
   (``/healthz`` and ``/metrics`` are exempt — a monitor must never be rate limited);
4. **routes** (:mod:`answerdesk.service.routes`), then the exception handlers.

The lifespan is the only place a :class:`~answerdesk.service.runtime.ServiceRuntime` is
created (or adopted), which is what "lifespan-wired pipeline" means in practice: importing
this module touches no database, no index and no provider.

``/healthz`` returns ``200`` with a per-dependency status; ``?strict=true`` turns a degraded
state into ``503`` for a load balancer that only understands status codes.

    # AT-SVC-07  # AT-OBS-01  # AT-SVC-06  # AT-SVC-03
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from answerdesk.config import Settings, load_settings
from answerdesk.errors import AnswerdeskError
from answerdesk.service.config import ServiceSettings
from answerdesk.service.errors import (
    PayloadTooLarge,
    RateLimited,
    ServiceError,
    to_service_error,
)
from answerdesk.service.ratelimit import RateLimiter
from answerdesk.service.routes import router
from answerdesk.service.runtime import SERVICE_VERSION, ServiceRuntime
from answerdesk.service.schemas import HealthResponse
from answerdesk.service.turn import new_trace_id
from answerdesk.telemetry.metrics import CONTENT_TYPE, Metrics

#: A trace id travels in a header, a JSONL file and metric labels; keep it a safe token.
_TRACE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")

_LOGGER = logging.getLogger("answerdesk.service")


class MetricsRef:
    """A mutable handle so middleware and the lifespan share one metric registry.

    The middleware chain is built before the runtime exists (the lifespan owns runtime
    creation), so the middlewares cannot hold the registry directly. They hold this handle
    and the lifespan points it at the runtime's registry — one registry, no split counters.
    """

    def __init__(self, registry: Metrics | None = None) -> None:
        self.registry: Metrics = registry or Metrics()


class TraceRef:
    """The same indirection for the JSONL writer (used by the HTTP audit line)."""

    def __init__(self, writer: Any = None) -> None:
        self.writer = writer


#: Paths that are never rate-limited or counted as customer requests.
EXEMPT_PATHS = frozenset(
    {"/healthz", "/metrics", "/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}
)

API_DESCRIPTION = """\
Multilingual customer-support RAG+LLM service.

* `POST /v1/chat` streams Server-Sent Events: `retrieval` → `tokens`* → `citations` →
  `decision` → `done`, every event carrying its `trace_id`.
* Sessions and turns are persisted in SQLite; `Idempotency-Key` makes a retry safe.
* `POST /v1/feedback` and `POST /v1/escalations` attach operator actions to a trace.
* `POST /admin/ingest` and `POST /admin/refresh` are bearer-token guarded and disabled
  unless `ANSWERDESK_SERVICE_ADMIN_TOKEN` is configured.
* `GET /healthz` reports dependency status; `GET /metrics` is Prometheus text format.

Full reference with examples: `docs/api.md`. Operations runbook: `docs/ops.md`.
"""

OPENAPI_TAGS = [
    {"name": "chat", "description": "Answer one customer turn (SSE or JSON)."},
    {"name": "sessions", "description": "Conversation persistence and replay."},
    {"name": "feedback", "description": "Customer/operator verdicts attached to a trace."},
    {"name": "escalations", "description": "Human handoff tickets."},
    {"name": "traces", "description": "Redacted JSONL trace lookup."},
    {"name": "admin", "description": "Protected ingestion and index refresh."},
    {"name": "health", "description": "Dependency status and Prometheus metrics."},
]


# -- middleware ------------------------------------------------------------------------
class RequestContextMiddleware:
    """Trace id, wall-clock timing, request/latency metrics and the ``X-Trace-Id`` echo."""

    def __init__(self, app: ASGIApp, *, metrics: MetricsRef, trace: TraceRef | None = None) -> None:
        self.app = app
        self.metrics = metrics
        self.trace = trace

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        provided = headers.get(b"x-trace-id")
        trace_id = provided.decode("latin-1").strip() if provided else new_trace_id()
        # A client-supplied id is echoed into a response header and written to the trace file,
        # so it must be a safe token: a space or a control character would make an invalid
        # header (fuzz finding F-08, docs/robustness.md). Anything else is replaced.
        if not _TRACE_ID_RE.match(trace_id):
            trace_id = new_trace_id()
        state = scope.setdefault("state", {})
        state["trace_id"] = trace_id
        started = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                raw_headers = list(message.get("headers", []))
                if not any(key.lower() == b"x-trace-id" for key, _value in raw_headers):
                    # An idempotent replay answers with the *stored* trace id; only a fresh
                    # response gets the current request's id.
                    raw_headers.append((b"x-trace-id", trace_id.encode("latin-1")))
                    message["headers"] = raw_headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - started
            route = scope.get("route")
            template = getattr(route, "path", None) or scope.get("path", "")
            self.metrics.registry.observe_request(
                method=scope.get("method", "GET"),
                route=template,
                status=status["code"],
                duration_s=duration,
            )
            self._audit(
                trace_id=trace_id,
                method=scope.get("method", "GET"),
                route=template,
                path=scope.get("path", ""),
                status=status["code"],
                duration=duration,
            )

    def _audit(
        self,
        *,
        trace_id: str,
        method: str,
        route: str,
        path: str,
        status: int,
        duration: float,
    ) -> None:
        """One ``kind=http`` JSONL line per request (FR-SVC-08's audit trail).

        The chat turn writes its own, richer line (``kind=turn``); this line is what makes
        "every request has a trace" true for sessions, feedback, tickets and admin calls.
        It never contains a query, a body or a header — only routing and timing facts.
        """
        if self.trace is None or self.trace.writer is None:
            return
        if path in EXEMPT_PATHS:
            return
        with contextlib.suppress(Exception):  # tracing must never break a response
            self.trace.writer.write(
                {
                    "kind": "http",
                    "trace_id": trace_id,
                    "method": method,
                    "route": route,
                    "status": status,
                    "duration_ms": round(duration * 1000.0, 3),
                    "created_at": utc_now(),
                }
            )


class BodyLimitMiddleware:
    """Reject a request whose declared body is larger than the configured cap."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw = headers.get(b"content-length")
        if raw is not None:
            try:
                length = int(raw)
            except ValueError:
                length = 0
            if length > self.max_bytes:
                await _json_error(scope, send, PayloadTooLarge(self.max_bytes))
                return
        await self.app(scope, receive, send)


class RateLimitMiddleware:
    """Per-client token bucket over ``/v1`` and ``/admin`` (FR-SVC-06)."""

    def __init__(self, app: ASGIApp, *, limiter: RateLimiter, metrics: MetricsRef) -> None:
        self.app = app
        self.limiter = limiter
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not (path.startswith("/v1") or path.startswith("/admin")):
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        client = headers.get(b"x-client-id")
        identity = (
            client.decode("latin-1").strip()
            if client
            else (scope.get("client") or ("anonymous", 0))[0]
        )
        decision = self.limiter.check(identity)
        if not decision.allowed:
            self.metrics.registry.rate_limited.inc()
            await _json_error(
                scope,
                send,
                RateLimited(decision.retry_after_s, limit=decision.limit),
            )
            return
        await self.app(scope, receive, send)


async def _json_error(scope: Scope, send: Send, error: ServiceError) -> None:
    trace_id = (scope.get("state") or {}).get("trace_id") or new_trace_id()
    body = json.dumps({"error": error.envelope(trace_id)}, ensure_ascii=False).encode("utf-8")
    headers = [
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"x-trace-id", trace_id.encode("latin-1")),
    ]
    if error.retry_after_s is not None:
        retry_after = str(max(1, round(error.retry_after_s))).encode("ascii")
        headers.append((b"retry-after", retry_after))
    await send({"type": "http.response.start", "status": error.http_status, "headers": headers})
    await send({"type": "http.response.body", "body": body, "more_body": False})


def utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="milliseconds")


# -- application -----------------------------------------------------------------------
def create_app(
    *,
    settings: Settings | None = None,
    service_settings: ServiceSettings | None = None,
    runtime: ServiceRuntime | None = None,
    runtime_factory: Callable[[], ServiceRuntime] | None = None,
) -> FastAPI:
    """Build the application. Nothing is opened until the lifespan starts."""
    core = settings or load_settings()
    service = service_settings or ServiceSettings()
    metrics_ref = MetricsRef(runtime.metrics if runtime is not None else None)
    trace_ref = TraceRef(runtime.tracer if runtime is not None else None)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active = runtime
        if active is None:
            if runtime_factory is not None:
                active = runtime_factory()
            else:
                active = ServiceRuntime.create(core, service, metrics=metrics_ref.registry)
        app.state.runtime = active
        metrics_ref.registry = active.metrics
        trace_ref.writer = active.tracer
        app.state.metrics = active.metrics
        _LOGGER.info(
            "answerdesk service ready",
            extra={
                "profile": active.info.profile,
                "fake": active.info.fake,
                "corpus_version": active.info.corpus_version,
                "index_version": active.info.index_version,
                "calibration": active.info.calibration.threshold_ref,
                "degraded": sorted(active.info.degraded),
            },
        )
        try:
            yield
        finally:
            active.close()

    app = FastAPI(
        title="answerdesk service",
        version=SERVICE_VERSION,
        description=API_DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        docs_url="/docs" if service.docs_enabled else None,
        redoc_url="/redoc" if service.docs_enabled else None,
        openapi_url="/openapi.json" if service.docs_enabled else None,
        lifespan=lifespan,
        contact={
            "name": "answerdesk contributors",
            "url": "https://github.com/answerdesk/answerdesk",
        },
        license_info={"name": "MIT"},
    )

    # The metrics handle exists before the middleware does; the lifespan points it at the
    # runtime's registry so middleware and routes share one registry.
    app.state.metrics_handle = metrics_ref
    app.state.trace_handle = trace_ref
    app.state.metrics = metrics_ref.registry
    app.state.limiter = RateLimiter(
        limit_per_minute=service.rate_limit_per_minute, burst=service.burst
    )

    app.add_middleware(RateLimitMiddleware, limiter=app.state.limiter, metrics=metrics_ref)
    app.add_middleware(BodyLimitMiddleware, max_bytes=service.max_body_bytes)
    app.add_middleware(RequestContextMiddleware, metrics=metrics_ref, trace=trace_ref)
    if service.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=service.cors_origin_list,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Trace-Id", "Retry-After", "X-Idempotent-Replay"],
        )

    app.include_router(router)
    _install_error_handlers(app)
    _install_health_routes(app)
    return app


def _install_health_routes(app: FastAPI) -> None:
    @app.get(
        "/healthz",
        tags=["health"],
        summary="Dependency status (no secrets)",
        response_model=HealthResponse,
    )
    async def healthz(request: Request, strict: bool = False) -> Response:
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is None:  # pragma: no cover - lifespan guarantees it
            raise ServiceError("service_unavailable", "runtime is not initialised")
        payload = runtime.health()
        status_code = 503 if (strict and payload["status"] != "ok") else 200
        return JSONResponse(payload, status_code=status_code)

    @app.get(
        "/metrics",
        tags=["health"],
        summary="Prometheus metrics (text/plain; version=0.0.4)",
        response_class=PlainTextResponse,
    )
    async def metrics(request: Request) -> Response:
        runtime = getattr(request.app.state, "runtime", None)
        registry = runtime.metrics if runtime is not None else request.app.state.metrics
        if runtime is not None and runtime.cache is not None:
            stats = runtime.cache.stats()
            registry.set_cache_entries(stats.entries)
        return PlainTextResponse(registry.render(), media_type=CONTENT_TYPE)

    @app.get("/v1/info", tags=["health"], summary="Non-secret service identity")
    async def info(request: Request) -> dict[str, Any]:
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is None:  # pragma: no cover
            raise ServiceError("service_unavailable", "runtime is not initialised")
        return {
            "service": "answerdesk",
            "version": SERVICE_VERSION,
            "profile": runtime.info.profile,
            "fake": runtime.info.fake,
            "corpus_version": runtime.info.corpus_version,
            "index_version": runtime.info.index_version,
            "chunks": runtime.info.chunk_count,
            "model": runtime.info.model,
            "preset": runtime.info.preset,
            "cache_enabled": runtime.info.cache_enabled,
            "admin_enabled": runtime.service.admin_enabled,
        }


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def handle_service_error(request: Request, exc: ServiceError) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None) or new_trace_id()
        _observe(app, exc.code)
        return JSONResponse(
            {"error": exc.envelope(trace_id)},
            status_code=exc.http_status,
            headers=exc.response_headers(trace_id),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None) or new_trace_id()
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.get('loc', []))}: {error.get('msg', '')}"
            for error in exc.errors()[:5]
        )
        _observe(app, "validation_error")
        return JSONResponse(
            {
                "error": {
                    "code": "validation_error",
                    "message": "request validation failed",
                    "trace_id": trace_id,
                    "retryable": False,
                    "detail": details,
                }
            },
            status_code=422,
            headers={"X-Trace-Id": trace_id},
        )

    @app.exception_handler(AnswerdeskError)
    async def handle_module_error(request: Request, exc: AnswerdeskError) -> JSONResponse:
        """Module errors (index/calibration/provider/…) map through the taxonomy."""
        trace_id = getattr(request.state, "trace_id", None) or new_trace_id()
        error = to_service_error(exc)
        _observe(app, error.code)
        return JSONResponse(
            {"error": error.envelope(trace_id)},
            status_code=error.http_status,
            headers=error.response_headers(trace_id),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None) or new_trace_id()
        code = {404: "not_found", 405: "method_not_allowed", 401: "unauthorized"}.get(
            exc.status_code, "bad_request"
        )
        error = ServiceError(code, str(exc.detail or "request failed"))
        _observe(app, code)
        return JSONResponse(
            {"error": error.envelope(trace_id)},
            status_code=exc.status_code,
            headers={"X-Trace-Id": trace_id},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None) or new_trace_id()
        _LOGGER.exception("unhandled service error", extra={"trace_id": trace_id})
        error = to_service_error(exc)
        _observe(app, error.code)
        return JSONResponse(
            {"error": error.envelope(trace_id)},
            status_code=error.http_status,
            headers={"X-Trace-Id": trace_id},
        )


def _observe(app: FastAPI, code: str) -> None:
    metrics = getattr(app.state, "metrics", None)
    if metrics is not None:
        metrics.observe_error(code)


# -- OpenAPI export --------------------------------------------------------------------
def openapi_schema(
    *,
    settings: Settings | None = None,
    service_settings: ServiceSettings | None = None,
) -> dict[str, Any]:
    """Build the schema without starting the service (no lifespan, no I/O)."""
    app = create_app(settings=settings, service_settings=service_settings)
    return app.openapi()


def app_factory() -> FastAPI:
    """ASGI factory for ``uvicorn answerdesk.service.app:app_factory --factory``."""
    return create_app()


def export_openapi(
    path: str | Path,
    *,
    settings: Settings | None = None,
    service_settings: ServiceSettings | None = None,
) -> Path:
    """Write the OpenAPI document to ``path`` and return it (AT: OpenAPI export)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    schema = openapi_schema(settings=settings, service_settings=service_settings)
    target.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "API_DESCRIPTION",
    "EXEMPT_PATHS",
    "OPENAPI_TAGS",
    "BodyLimitMiddleware",
    "RateLimitMiddleware",
    "RequestContextMiddleware",
    "app_factory",
    "create_app",
    "export_openapi",
    "openapi_schema",
]
