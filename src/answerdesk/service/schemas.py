"""Wire models: requests, responses and the SSE event payloads.

These models are the *HTTP* contract. They never leak into the pipeline: a route validates
a request, calls :class:`~answerdesk.service.turn.TurnRunner`, and serialises the resulting
:mod:`answerdesk.contracts` objects. Two consequences worth stating:

* the SSE payloads are models too (``RetrievalEvent`` … ``DoneEvent``), so the stream is
  schema-checked in tests and describable in ``docs/api.md`` — an event that drifts from
  the documented shape fails a test rather than a console;
* request models forbid unknown fields. A typo in a client payload is a ``422`` with the
  field name, not a silently ignored parameter.

    # AT-SVC-01  # AT-SVC-03  # AT-SVC-05
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from answerdesk.contracts import (
    Action,
    CacheStatus,
    Citation,
    Claim,
    Decision,
    IntentBucket,
    Priority,
    ReasonCode,
    Usage,
)


class Wire(BaseModel):
    """Strict base for every wire model (unknown fields are a client bug)."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------
class ChatRequest(Wire):
    """One customer turn."""

    message: str = Field(
        min_length=1,
        max_length=8000,
        description="The customer message, in any supported locale (ru, tg, uz, ky, tr, zh, en).",
        examples=["Сколько стоит годовое обслуживание карты «Вектор»?"],
    )
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Conversation to continue. Omit to start a new session.",
        examples=["s-0123456789abcdef"],
    )
    locale: str | None = Field(
        default=None,
        min_length=2,
        max_length=8,
        description="Locale hint for the cache cohort; detected from the text when omitted.",
        examples=["ru"],
    )
    intent_bucket: IntentBucket = Field(
        default=IntentBucket.INFORMATIONAL,
        description="Risk tier that selects the conformal alpha (requirements §8.2).",
    )
    intent_label: str | None = Field(
        default=None,
        max_length=64,
        description="Optional fine-grained intent label used by the high-risk guard.",
        examples=["account"],
    )
    stream: bool = Field(
        default=True,
        description="Stream Server-Sent Events. Set false for a single JSON response.",
    )


class FeedbackRequest(Wire):
    """A thumbs up/down attached to a trace (FR-SVC-05)."""

    trace_id: str = Field(min_length=1, max_length=64, examples=["tr-0123456789abcdef"])
    rating: Literal["up", "down", "neutral"] = Field(description="Customer/operator verdict.")
    comment: str | None = Field(default=None, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)


class EscalationTicketRequest(Wire):
    """An explicit handoff to a human (FR-SVC-05, W2.5)."""

    trace_id: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=3, max_length=500, examples=["Customer asked for a human."])
    priority: Priority = Priority.NORMAL
    note: str | None = Field(default=None, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)


class IngestRequest(Wire):
    """Admin ingestion (FR-SVC-04)."""

    source: str | None = Field(
        default=None,
        max_length=512,
        description="File or directory to ingest. Omit with demo=true for the synthetic corpus.",
    )
    demo: bool = Field(default=False, description="Build/refresh the checked-in demo corpus.")
    force: bool = Field(default=False, description="Rebuild even when the fingerprint matches.")


# --------------------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------------------
class PassageSummary(Wire):
    """A retrieved passage as the console sees it (never the full chunk text)."""

    chunk_id: str
    doc_id: str
    doc_title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    breadcrumb: str = ""
    channel: str
    stage: str
    score: float
    rank: int
    snippet: str


class CacheSummary(Wire):
    status: CacheStatus
    entry_id: str | None = None
    similarity: float | None = None
    reason: str = ""


class RetrievalEvent(Wire):
    """First SSE event: what evidence the turn was built from (FR-SVC-01, FR-UI-04)."""

    event: Literal["retrieval"] = "retrieval"
    trace_id: str
    query: str
    normalized_query: str | None = None
    language: str | None = None
    passages: list[PassageSummary] = Field(default_factory=list)
    channels: dict[str, int] = Field(default_factory=dict)
    fusion: dict[str, Any] | None = None
    candidates: int = 0
    dropped_duplicates: int = 0
    cache: CacheSummary
    timings_ms: dict[str, float] = Field(default_factory=dict)


class TokenEvent(Wire):
    """A delta of the *verified* answer text (see docs/api.md §5.1 for why after verify)."""

    event: Literal["tokens"] = "tokens"
    trace_id: str
    delta: str
    index: int = Field(ge=0)


class CitationPayload(Wire):
    citation_id: int
    chunk_id: str
    doc_id: str
    doc_title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    quote: str
    start_char: int
    end_char: int
    doc_hash: str


class ClaimPayload(Wire):
    claim_id: str
    text: str
    citation_ids: list[int] = Field(default_factory=list)
    support: str


class CitationsEvent(Wire):
    """Verified citations only: this event is never emitted for an unverified draft."""

    event: Literal["citations"] = "citations"
    trace_id: str
    citations: list[CitationPayload] = Field(default_factory=list)
    claims: list[ClaimPayload] = Field(default_factory=list)


class EscalationSummary(Wire):
    ticket_id: str | None = None
    priority: Priority = Priority.NORMAL
    detail: str = ""
    reason: ReasonCode


class DecisionEvent(Wire):
    """The policy decision, with the provenance an operator needs to reproduce it."""

    event: Literal["decision"] = "decision"
    trace_id: str
    action: Action
    reason_code: ReasonCode
    score: float = Field(ge=0.0, le=1.0, description="Calibrated p-hat fed to the decision.")
    alpha: float = Field(gt=0.0, lt=1.0)
    threshold_ref: str
    escalation: EscalationSummary | None = None


class AnswerPayload(Wire):
    """The frozen :class:`answerdesk.contracts.Answer` shape, as emitted on the wire."""

    answer_id: str
    trace_id: str
    language: str
    text: str
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    decision: Decision
    confidence: dict[str, Any]
    cache: dict[str, Any]
    usage: Usage
    provenance: dict[str, Any]


class DoneEvent(Wire):
    """Terminal event: the answer object plus the identifiers a client stores."""

    event: Literal["done"] = "done"
    trace_id: str
    session_id: str
    turn_id: str
    answer: AnswerPayload
    usage: Usage
    timings_ms: dict[str, float] = Field(default_factory=dict)
    history_turns: int = Field(default=0, ge=0)
    follow_up: bool = False
    ticket_id: str | None = None
    idempotent_replay: bool = False


class ErrorEvent(Wire):
    """Mid-stream failure: emitted before ``done`` so a client can distinguish the cases."""

    event: Literal["error"] = "error"
    trace_id: str
    error: ErrorEnvelope


class ErrorEnvelope(Wire):
    code: str
    message: str
    trace_id: str
    retryable: bool
    detail: str | None = None
    retry_after_s: float | None = None


class ChatResponse(Wire):
    """Non-streaming answer (``stream=false`` or an idempotent replay)."""

    trace_id: str
    session_id: str
    turn_id: str
    answer: AnswerPayload
    usage: Usage
    timings_ms: dict[str, float] = Field(default_factory=dict)
    history_turns: int = 0
    follow_up: bool = False
    ticket_id: str | None = None
    idempotent_replay: bool = False
    cache: CacheSummary


class TurnResponse(Wire):
    turn_id: str
    seq: int
    message: str
    answer_text: str
    trace_id: str
    action: str
    reason_code: str
    detail: str = ""
    created_at: str


class SessionResponse(Wire):
    session_id: str
    created_at: str
    updated_at: str
    locale: str | None = None
    turn_count: int = 0
    turns: list[TurnResponse] = Field(default_factory=list)


class FeedbackResponse(Wire):
    feedback_id: str
    trace_id: str
    rating: str
    created_at: str


class TicketResponse(Wire):
    ticket_id: str
    trace_id: str
    session_id: str | None = None
    status: str
    priority: Priority
    reason: str
    note: str = ""
    created_at: str


class IngestResponse(Wire):
    corpus_version: str
    documents: int
    chunks: int
    index_version: str
    reused: bool
    elapsed_ms: float
    manifest_path: str
    cache_invalidated: int = 0


class ComponentHealth(Wire):
    status: Literal["ok", "degraded", "error", "disabled"]
    detail: str = ""
    version: str = ""


class HealthResponse(Wire):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    schema_version: str
    uptime_s: float
    checks: dict[str, ComponentHealth]


class TraceResponse(Wire):
    trace_id: str
    trace: dict[str, Any]


class ErrorResponse(Wire):
    error: ErrorEnvelope


ErrorEvent.model_rebuild()

#: Status codes the API documents per route (used by the OpenAPI metadata).
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "bad_request"},
    401: {"model": ErrorResponse, "description": "unauthorized"},
    403: {"model": ErrorResponse, "description": "forbidden / admin_disabled"},
    404: {"model": ErrorResponse, "description": "not_found"},
    409: {"model": ErrorResponse, "description": "conflict / idempotency_conflict"},
    413: {"model": ErrorResponse, "description": "payload_too_large"},
    422: {"model": ErrorResponse, "description": "validation_error"},
    429: {"model": ErrorResponse, "description": "rate_limited (Retry-After)"},
    500: {"model": ErrorResponse, "description": "internal_error / contract_violation"},
    502: {"model": ErrorResponse, "description": "provider_error family"},
    503: {"model": ErrorResponse, "description": "index/calibration/corpus unavailable"},
    504: {"model": ErrorResponse, "description": "provider_timeout"},
}


__all__ = [
    "ERROR_RESPONSES",
    "AnswerPayload",
    "CacheSummary",
    "ChatRequest",
    "ChatResponse",
    "CitationPayload",
    "CitationsEvent",
    "ClaimPayload",
    "ComponentHealth",
    "DecisionEvent",
    "DoneEvent",
    "ErrorEnvelope",
    "ErrorEvent",
    "ErrorResponse",
    "EscalationSummary",
    "EscalationTicketRequest",
    "FeedbackRequest",
    "FeedbackResponse",
    "HealthResponse",
    "IngestRequest",
    "IngestResponse",
    "PassageSummary",
    "RetrievalEvent",
    "SessionResponse",
    "TicketResponse",
    "TokenEvent",
    "TraceResponse",
    "TurnResponse",
    "Wire",
]
