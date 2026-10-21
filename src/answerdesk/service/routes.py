"""HTTP routes: chat (SSE), sessions, feedback, escalation tickets, admin, traces.

Each route is a thin translation layer:

* validate the wire model (:mod:`answerdesk.service.schemas`);
* resolve the runtime and the trace id from the request;
* call the :class:`~answerdesk.service.turn.TurnRunner` (never the pipeline directly);
* serialise a contract object.

Two behaviours are worth reading the code for:

**Idempotency** is checked *before* any work and stored *after* the turn, using the same
JSON the client received — including for streamed turns, where the stored replay is the
non-streaming equivalent (a replay cannot re-stream events that were already sent; see
``docs/api.md`` §3.4).

**Streaming failures** are split by phase. Dependency failures (no index, no calibration,
rate limit) are detected before the response starts and become ordinary HTTP errors with a
status code. A provider failure *inside* the stream becomes an ``error`` event followed by
``done``, because the status line has already been sent. Both paths write the same trace.

    # AT-SVC-01  # AT-SVC-02  # AT-SVC-03  # AT-SVC-04  # AT-SVC-05  # AT-SVC-06
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from answerdesk.contracts import Priority
from answerdesk.service import idempotency
from answerdesk.service.errors import (
    AdminDisabled,
    BadRequest,
    Forbidden,
    NotFound,
    ServiceError,
    Unauthorized,
    constant_time_equals,
    to_service_error,
)
from answerdesk.service.runtime import ServiceRuntime
from answerdesk.service.schemas import (
    AnswerPayload,
    CacheSummary,
    ChatRequest,
    ChatResponse,
    CitationPayload,
    CitationsEvent,
    ClaimPayload,
    DecisionEvent,
    DoneEvent,
    ErrorEnvelope,
    ErrorEvent,
    EscalationSummary,
    EscalationTicketRequest,
    FeedbackRequest,
    FeedbackResponse,
    IngestRequest,
    IngestResponse,
    PassageSummary,
    RetrievalEvent,
    SessionResponse,
    TicketResponse,
    TokenEvent,
    TraceResponse,
    TurnResponse,
)
from answerdesk.service.sse import SSE_HEADERS, SSE_MEDIA_TYPE, sse_event, token_chunks
from answerdesk.service.turn import TurnRequest, TurnResult, TurnRunner, new_trace_id

router = APIRouter()

#: Endpoint names used to scope idempotency keys.
CHAT_ENDPOINT = "chat"
FEEDBACK_ENDPOINT = "feedback"
ESCALATION_ENDPOINT = "escalations"


# -- dependencies ----------------------------------------------------------------------
def get_runtime(request: Request) -> ServiceRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:  # pragma: no cover - lifespan always sets it
        raise ServiceError("service_unavailable", "the service runtime is not initialised")
    return runtime


def get_trace_id(request: Request) -> str:
    trace_id = getattr(request.state, "trace_id", None)
    if not trace_id:  # pragma: no cover - middleware always sets it
        trace_id = new_trace_id()
        request.state.trace_id = trace_id
    return trace_id


def get_runner(runtime: ServiceRuntime = Depends(get_runtime)) -> TurnRunner:
    return TurnRunner(runtime)


async def require_admin(
    runtime: ServiceRuntime = Depends(get_runtime),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    """Bearer-token gate: 401 without a header, 403 when disabled or wrong (FR-SVC-04)."""
    service = runtime.service
    if not service.admin_enabled:
        raise AdminDisabled()
    if not authorization:
        raise Unauthorized("admin endpoints require an Authorization: Bearer <token> header")
    scheme, _, presented = authorization.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        raise Unauthorized("Authorization must use the Bearer scheme")
    if not constant_time_equals(presented.strip(), service.admin_token.get_secret_value()):
        raise Forbidden("the bearer token is not accepted")


def _wants_stream(payload: ChatRequest, request: Request) -> bool:
    if not payload.stream:
        return False
    accept = request.headers.get("accept", "")
    return "application/json" not in accept


# -- chat ------------------------------------------------------------------------------
@router.post(
    "/v1/chat",
    tags=["chat"],
    summary="Answer one customer turn (SSE by default)",
    response_model=None,
    responses={
        200: {
            "description": (
                "text/event-stream with retrieval → tokens* → citations → decision → done, "
                "or a ChatResponse JSON object when stream=false"
            )
        }
    },
)
async def chat(
    payload: ChatRequest,
    request: Request,
    runtime: ServiceRuntime = Depends(get_runtime),
    trace_id: str = Depends(get_trace_id),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> Response:
    runtime.require_answer_path()
    fingerprint = idempotency.request_fingerprint(
        {
            "message": payload.message,
            "session_id": payload.session_id,
            "locale": payload.locale,
            "intent_bucket": payload.intent_bucket.value,
            "intent_label": payload.intent_label,
        }
    )
    stored = idempotency.replay(
        runtime.store,
        key=idempotency_key,
        endpoint=CHAT_ENDPOINT,
        fingerprint=fingerprint,
        ttl_s=runtime.service.idempotency_ttl_s,
    )
    if stored is not None:
        runtime.metrics.idempotent_replays.inc()
        runtime.metrics.observe_request(
            method="POST", route="/v1/chat", status=stored.status, duration_s=0.0
        )
        return JSONResponse(
            stored.payload,
            status_code=stored.status,
            headers={
                "X-Trace-Id": stored.trace_id,
                "X-Idempotent-Replay": "true",
                "Cache-Control": "no-store",
            },
        )

    turn_request = TurnRequest(
        message=payload.message,
        session_id=payload.session_id,
        locale=payload.locale,
        intent_bucket=payload.intent_bucket,
        intent_label=payload.intent_label,
        idempotency_key=idempotency_key,
    )
    runner = TurnRunner(runtime)

    if not _wants_stream(payload, request):
        started = time.perf_counter()
        result = await asyncio.to_thread(runner.run, turn_request, trace_id=trace_id)
        body = _chat_response(result).model_dump(mode="json")
        idempotency.remember(
            runtime.store,
            key=idempotency_key,
            endpoint=CHAT_ENDPOINT,
            fingerprint=fingerprint,
            payload=body,
            trace_id=trace_id,
        )
        runtime.metrics.observe_request(
            method="POST",
            route="/v1/chat",
            status=200,
            duration_s=time.perf_counter() - started,
        )
        return JSONResponse(body, headers={"X-Trace-Id": trace_id, "Cache-Control": "no-store"})

    runtime.require_answer_path()
    generator = _stream_turns(
        runtime=runtime,
        runner=runner,
        turn_request=turn_request,
        trace_id=trace_id,
        fingerprint=fingerprint,
        idempotency_key=idempotency_key,
    )
    return StreamingResponse(
        generator,
        media_type=SSE_MEDIA_TYPE,
        headers={**SSE_HEADERS, "X-Trace-Id": trace_id},
    )


async def _stream_turns(
    *,
    runtime: ServiceRuntime,
    runner: TurnRunner,
    turn_request: TurnRequest,
    trace_id: str,
    fingerprint: str,
    idempotency_key: str | None,
) -> AsyncIterator[str]:
    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(runner.run, turn_request, trace_id=trace_id)
    except BaseException as exc:  # noqa: BLE001 - the stream must always terminate cleanly
        error = to_service_error(exc)
        runtime.metrics.observe_error(error.code)
        payload = error.envelope(trace_id)
        yield sse_event("error", ErrorEvent(trace_id=trace_id, error=ErrorEnvelope(**payload)).model_dump(mode="json"))
        yield sse_event(
            "done",
            {
                "event": "done",
                "trace_id": trace_id,
                "session_id": turn_request.session_id,
                "turn_id": "",
                "error": payload,
                "idempotent_replay": False,
            },
        )
        runtime.metrics.observe_request(
            method="POST",
            route="/v1/chat",
            status=error.http_status,
            duration_s=time.perf_counter() - started,
        )
        return

    yield sse_event("retrieval", _retrieval_event(result).model_dump(mode="json"))
    for index, delta in enumerate(token_chunks(result.answer.text, runtime.service.sse_token_chars)):
        yield sse_event(
            "tokens", TokenEvent(trace_id=trace_id, delta=delta, index=index).model_dump(mode="json")
        )
    yield sse_event("citations", _citations_event(result).model_dump(mode="json"))
    yield sse_event("decision", _decision_event(result).model_dump(mode="json"))
    done = _done_event(result).model_dump(mode="json")
    idempotency.remember(
        runtime.store,
        key=idempotency_key,
        endpoint=CHAT_ENDPOINT,
        fingerprint=fingerprint,
        payload=_chat_response(result).model_dump(mode="json"),
        trace_id=trace_id,
    )
    yield sse_event("done", done)
    runtime.metrics.observe_request(
        method="POST",
        route="/v1/chat",
        status=200,
        duration_s=time.perf_counter() - started,
    )


# -- sessions --------------------------------------------------------------------------
@router.post("/v1/sessions", tags=["sessions"], summary="Create an empty conversation")
async def create_session(
    request: Request,
    locale: str | None = Query(default=None, max_length=8),
    runtime: ServiceRuntime = Depends(get_runtime),
) -> JSONResponse:
    session = runtime.store.create_session(locale=locale)
    runtime.metrics.sessions_created.inc()
    return JSONResponse(
        SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            locale=session.locale,
            turn_count=0,
            turns=[],
        ).model_dump(mode="json"),
        status_code=201,
        headers={"X-Trace-Id": get_trace_id(request)},
    )


@router.get(
    "/v1/sessions/{session_id}",
    tags=["sessions"],
    summary="Read a conversation and its turns",
    response_model=SessionResponse,
)
async def get_session(
    session_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    runtime: ServiceRuntime = Depends(get_runtime),
) -> SessionResponse:
    session = runtime.store.get_session(session_id)
    if session is None:
        raise NotFound(f"session {session_id!r}")
    rows = runtime.store.turns(session_id, limit=limit)
    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        locale=session.locale,
        turn_count=runtime.store.turn_count(session_id),
        turns=[
            TurnResponse(
                turn_id=row.turn_id,
                seq=row.seq,
                message=row.message,
                answer_text=row.answer_text,
                trace_id=row.trace_id,
                action=row.action,
                reason_code=row.reason_code,
                detail=row.detail,
                created_at=row.created_at,
            )
            for row in rows
        ],
    )


# -- feedback --------------------------------------------------------------------------
@router.post(
    "/v1/feedback",
    tags=["feedback"],
    summary="Attach a customer/operator verdict to a trace",
    response_model=FeedbackResponse,
    status_code=201,
)
async def create_feedback(
    payload: FeedbackRequest,
    runtime: ServiceRuntime = Depends(get_runtime),
) -> FeedbackResponse:
    turn = runtime.store.turn_by_trace(payload.trace_id)
    if turn is None:
        raise NotFound(f"trace {payload.trace_id!r}")
    row = runtime.store.add_feedback(
        trace_id=payload.trace_id,
        rating=payload.rating,
        comment=payload.comment,
        session_id=payload.session_id or turn.session_id,
    )
    runtime.metrics.feedback.inc(1.0, payload.rating)
    if payload.rating == "down":
        _record_cache_wrong_hit(runtime, payload.trace_id)
    return FeedbackResponse(
        feedback_id=row.feedback_id,
        trace_id=row.trace_id,
        rating=row.rating,
        created_at=row.created_at,
    )


def _record_cache_wrong_hit(runtime: ServiceRuntime, trace_id: str) -> None:
    """A down-vote on a cache hit is the live signal for the cache error budget (§9.5)."""
    try:
        line = runtime.tracer.read(trace_id)
    except Exception:  # pragma: no cover - tracing must not block feedback
        return
    if not line:
        return
    cache = line.get("cache") or {}
    if cache.get("status") != "hit":
        return
    runtime.metrics.cache_wrong_hits.inc()
    entry_id = cache.get("entry_id")
    if entry_id and runtime.cache is not None:
        try:
            runtime.cache.record_wrong_hit(entry_id)
        except Exception:  # pragma: no cover - eviction race
            pass


# -- escalation tickets ----------------------------------------------------------------
@router.post(
    "/v1/escalations",
    tags=["escalations"],
    summary="Open an escalation ticket against a trace",
    response_model=TicketResponse,
    status_code=201,
)
async def create_escalation(
    payload: EscalationTicketRequest,
    runtime: ServiceRuntime = Depends(get_runtime),
) -> TicketResponse:
    turn = runtime.store.turn_by_trace(payload.trace_id)
    if turn is None:
        raise NotFound(f"trace {payload.trace_id!r}")
    ticket = runtime.store.create_ticket(
        trace_id=payload.trace_id,
        session_id=payload.session_id or turn.session_id,
        reason=payload.reason,
        priority=payload.priority.value,
        note=payload.note,
        status="open",
    )
    return _ticket_response(ticket)


@router.get(
    "/v1/escalations/{ticket_id}",
    tags=["escalations"],
    summary="Read one escalation ticket",
    response_model=TicketResponse,
)
async def get_escalation(ticket_id: str, runtime: ServiceRuntime = Depends(get_runtime)) -> TicketResponse:
    ticket = runtime.store.get_ticket(ticket_id)
    if ticket is None:
        raise NotFound(f"ticket {ticket_id!r}")
    return _ticket_response(ticket)


@router.get(
    "/v1/escalations",
    tags=["escalations"],
    summary="List escalation tickets (newest first)",
)
async def list_escalations(
    status: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=50, ge=1, le=200),
    runtime: ServiceRuntime = Depends(get_runtime),
) -> dict[str, Any]:
    rows = runtime.store.list_tickets(status=status, limit=limit)
    return {"tickets": [_ticket_response(row).model_dump(mode="json") for row in rows]}


# -- traces ----------------------------------------------------------------------------
@router.get(
    "/v1/traces/{trace_id}",
    tags=["traces"],
    summary="Read the redacted JSONL trace line for a turn",
    response_model=TraceResponse,
)
async def get_trace(trace_id: str, runtime: ServiceRuntime = Depends(get_runtime)) -> TraceResponse:
    line = runtime.tracer.read(trace_id)
    if line is None:
        raise NotFound(f"trace {trace_id!r}")
    return TraceResponse(trace_id=trace_id, trace=line)


# -- admin -----------------------------------------------------------------------------
@router.post(
    "/admin/ingest",
    tags=["admin"],
    summary="Ingest a source (or the demo corpus) and rebuild the index",
    response_model=IngestResponse,
    dependencies=[Depends(require_admin)],
)
async def admin_ingest(
    payload: IngestRequest,
    runtime: ServiceRuntime = Depends(get_runtime),
) -> IngestResponse:
    return await _ingest(runtime, payload)


@router.post(
    "/admin/refresh",
    tags=["admin"],
    summary="Rebuild/reload the index and invalidate the answer cache",
    response_model=IngestResponse,
    dependencies=[Depends(require_admin)],
)
async def admin_refresh(
    runtime: ServiceRuntime = Depends(get_runtime),
    force: bool = Query(default=True, description="Rebuild even when the fingerprint matches."),
) -> IngestResponse:
    return await _ingest(runtime, IngestRequest(demo=True, force=force))


async def _ingest(runtime: ServiceRuntime, payload: IngestRequest) -> IngestResponse:
    started = time.perf_counter()
    messages: list[str] = []
    try:
        if payload.source:
            source = Path(payload.source)
            if not source.exists():
                raise BadRequest(f"source does not exist: {payload.source}")
            from answerdesk.ingest.cli import ingest_command
            from answerdesk.ingest.errors import IngestError

            try:
                await asyncio.to_thread(
                    ingest_command,
                    settings=runtime.settings,
                    source=payload.source,
                    demo=False,
                    echo=messages.append,
                )
            except IngestError as exc:
                raise BadRequest(exc.message, detail=exc.detail) from exc
        result = await asyncio.to_thread(runtime.refresh_index, demo=payload.demo, force=payload.force)
    except BadRequest:
        raise
    except Exception as exc:  # noqa: BLE001 - map every operator failure to the taxonomy
        raise to_service_error(exc) from exc
    runtime.metrics.observe_stage("admin_ingest", time.perf_counter() - started)
    return IngestResponse(
        corpus_version=result["corpus_version"],
        documents=int(result.get("documents", 0)),
        chunks=int(result.get("chunks", 0)),
        index_version=result["index_version"],
        reused=bool(result.get("reused", False)),
        elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        manifest_path=str(result.get("manifest_path", "")),
        cache_invalidated=int(result.get("cache_invalidated", 0)),
    )


# -- serialisers -----------------------------------------------------------------------
def _snippet(text: str, limit: int = 400) -> str:
    body = " ".join(text.split())
    return body if len(body) <= limit else body[:limit].rstrip() + "…"


def _retrieval_event(result: TurnResult) -> RetrievalEvent:
    passages: list[PassageSummary] = []
    retrieval = result.retrieval
    if retrieval is not None:
        for passage in retrieval.packed:
            chunk = passage.chunk
            passages.append(
                PassageSummary(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=str(chunk.metadata.get("doc_title", "")),
                    heading_path=list(chunk.heading_path),
                    breadcrumb=chunk.breadcrumb,
                    channel=passage.channel.value,
                    stage=passage.stage,
                    score=round(passage.score, 8),
                    rank=passage.rank,
                    snippet=_snippet(chunk.text),
                )
            )
    channels: dict[str, int] = {}
    for passage in retrieval.candidates if retrieval is not None else []:
        channels[passage.channel.value] = channels.get(passage.channel.value, 0) + 1
    return RetrievalEvent(
        trace_id=result.trace_id,
        query=result.retrieval_query,
        normalized_query=retrieval.normalized_query if retrieval is not None else None,
        language=result.answer.language,
        passages=passages,
        channels=channels,
        fusion=(
            retrieval.fusion.model_dump(mode="json")
            if retrieval is not None and retrieval.fusion is not None
            else None
        ),
        candidates=len(retrieval.candidates) if retrieval is not None else 0,
        dropped_duplicates=retrieval.dropped_duplicates if retrieval is not None else 0,
        cache=CacheSummary(
            status=result.cache_status,
            entry_id=result.cache_entry_id,
            similarity=result.cache_similarity,
            reason=result.cache_reason,
        ),
        timings_ms=result.timings_ms,
    )


def _citations_event(result: TurnResult) -> CitationsEvent:
    return CitationsEvent(
        trace_id=result.trace_id,
        citations=[
            CitationPayload(
                citation_id=citation.citation_id,
                chunk_id=citation.chunk_id,
                doc_id=citation.doc_id,
                doc_title=citation.doc_title,
                heading_path=list(citation.heading_path),
                quote=citation.quote,
                start_char=citation.start_char,
                end_char=citation.end_char,
                doc_hash=citation.doc_hash,
            )
            for citation in result.answer.citations
        ],
        claims=[
            ClaimPayload(
                claim_id=claim.claim_id,
                text=claim.text,
                citation_ids=list(claim.citation_ids),
                support=claim.support.value,
            )
            for claim in result.answer.claims
        ],
    )


def _decision_event(result: TurnResult) -> DecisionEvent:
    decision = result.decision
    escalation = None
    if decision.action.value in ("escalate", "refuse"):
        escalation = EscalationSummary(
            ticket_id=result.ticket_id,
            priority=Priority.NORMAL,
            detail=(
                result.generation.signal.detail
                if result.generation is not None
                else result.cache_reason
            ),
            reason=decision.reason_code,
        )
    return DecisionEvent(
        trace_id=result.trace_id,
        action=decision.action,
        reason_code=decision.reason_code,
        score=decision.score,
        alpha=decision.alpha,
        threshold_ref=decision.threshold_ref,
        escalation=escalation,
    )


def _answer_payload(result: TurnResult) -> AnswerPayload:
    return AnswerPayload(**result.answer.model_dump(mode="json"))


def _done_event(result: TurnResult) -> DoneEvent:
    return DoneEvent(
        trace_id=result.trace_id,
        session_id=result.session_id,
        turn_id=result.turn_id,
        answer=_answer_payload(result),
        usage=result.usage,
        timings_ms=result.timings_ms,
        history_turns=result.history_turns,
        follow_up=result.follow_up,
        ticket_id=result.ticket_id,
        idempotent_replay=False,
    )


def _chat_response(result: TurnResult) -> ChatResponse:
    return ChatResponse(
        trace_id=result.trace_id,
        session_id=result.session_id,
        turn_id=result.turn_id,
        answer=_answer_payload(result),
        usage=result.usage,
        timings_ms=result.timings_ms,
        history_turns=result.history_turns,
        follow_up=result.follow_up,
        ticket_id=result.ticket_id,
        idempotent_replay=False,
        cache=CacheSummary(
            status=result.cache_status,
            entry_id=result.cache_entry_id,
            similarity=result.cache_similarity,
            reason=result.cache_reason,
        ),
    )


def _ticket_response(ticket: Any) -> TicketResponse:
    return TicketResponse(
        ticket_id=ticket.ticket_id,
        trace_id=ticket.trace_id,
        session_id=ticket.session_id,
        status=ticket.status,
        priority=Priority(ticket.priority),
        reason=ticket.reason,
        note=ticket.note,
        created_at=ticket.created_at,
    )


__all__ = [
    "CHAT_ENDPOINT",
    "ESCALATION_ENDPOINT",
    "FEEDBACK_ENDPOINT",
    "get_runner",
    "get_runtime",
    "get_trace_id",
    "require_admin",
    "router",
]
