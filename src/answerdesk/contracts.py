"""Frozen cross-module contracts for answerdesk (schema 1.0.0).

Everything that crosses a module boundary is defined here and nowhere else:

* pydantic models — :class:`Document`, :class:`Chunk`, :class:`RetrievedPassage`,
  :class:`Citation`, :class:`Claim`, :class:`Decision`, :class:`Answer`,
  :class:`EscalationDecision`, :class:`ProviderConfig`, :class:`TraceRecord`,
  :class:`EvalCase` and their supporting value objects;
* ``Protocol`` seams — every pluggable component (embedder, retriever, reranker,
  fuser, chat provider, judge, claim verifier, answerability scorer, cache, loader,
  chunker, trace sink);
* closed enums — :class:`Action`, :class:`ReasonCode` (the closed enum of
  ``docs/requirements.md`` Appendix B), :class:`Support`, :class:`CacheStatus`,
  :class:`Channel`, :class:`IntentBucket`, :class:`EvalCategory`.

Rules (see ``docs/architecture.md`` §3 and ADR-006):

1. Models forbid unknown fields. Adding a field is a contract change: edit this
   module, bump :data:`SCHEMA_VERSION`, and note it in the change log.
2. Models are the *displayed* / *persisted* shape. Internal drafting structures may
   live in their own module; only validated contracts cross the seam.
3. No model performs I/O and no model imports a heavy optional dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

SCHEMA_VERSION: str = "1.0.0"

# --------------------------------------------------------------------------------------
# Enums (closed sets shared by service, console, cache, evaluation and docs)
# --------------------------------------------------------------------------------------


class Action(StrEnum):
    """What the policy decided to do with a turn (requirements §8.4)."""

    ANSWER = "answer"
    CLARIFY = "clarify"
    ESCALATE = "escalate"
    REFUSE = "refuse"


class ReasonCode(StrEnum):
    """Closed reason-code enum — must stay identical to requirements Appendix B.

    A parity test (``tests/unit/test_contracts.py::test_reason_codes_match_requirements``)
    parses the appendix of ``docs/requirements.md`` and fails on drift.
    """

    ANSWERED = "ANSWERED"
    CLARIFY_AMBIGUOUS = "CLARIFY_AMBIGUOUS"
    NO_EVIDENCE = "NO_EVIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    VERIFIER_REJECT = "VERIFIER_REJECT"
    CITATION_FAILED = "CITATION_FAILED"
    HIGH_RISK_INTENT = "HIGH_RISK_INTENT"
    PII_DETECTED = "PII_DETECTED"
    INJECTION_SUSPECTED = "INJECTION_SUSPECTED"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    CORPUS_MISSING = "CORPUS_MISSING"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    STALE_CALIBRATION = "STALE_CALIBRATION"
    OPERATOR_REQUEST = "OPERATOR_REQUEST"
    USER_REQUESTED_HUMAN = "USER_REQUESTED_HUMAN"


class Support(StrEnum):
    """Per-claim verifier verdict (requirements §7.3)."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class CacheStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"
    BYPASSED = "bypassed"


class Channel(StrEnum):
    """Retrieval channel that produced a passage."""

    BM25 = "bm25"
    DENSE = "dense"
    RERANK = "rerank"


class IntentBucket(StrEnum):
    """Risk tier that selects ``alpha`` and cache behaviour (requirements §8.2)."""

    INFORMATIONAL = "informational"
    TRANSACTIONAL = "transactional"
    ACCOUNT_LEGAL = "account_legal"


class EvalCategory(StrEnum):
    """Stratification categories of ``golden/v1`` (requirements §10.1)."""

    DIRECT = "direct"
    NUMERIC_DATE = "numeric_date"
    TYPO_CODE_SWITCH = "typo_code_switch"
    UNANSWERABLE = "unanswerable"
    PII = "pii"
    INJECTION = "injection"


class Split(StrEnum):
    DEV = "dev"
    TEST = "test"


class Priority(StrEnum):
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# --------------------------------------------------------------------------------------
# Base model
# --------------------------------------------------------------------------------------


class Contract(BaseModel):
    """Base class for every contract model.

    ``extra="forbid"`` is deliberate: a field that appears out of nowhere is a
    contract drift that must be reviewed (ADR-006). ``validate_assignment`` keeps a
    mutated contract from becoming invalid after construction.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, validate_default=True)


# --------------------------------------------------------------------------------------
# Ingestion / retrieval
# --------------------------------------------------------------------------------------


class Heading(Contract):
    """One heading in a document's structure, with its position in the raw text."""

    level: int = Field(ge=1, le=6)
    text: str = Field(min_length=1)
    char_offset: int = Field(ge=0)


class Document(Contract):
    """A normalised source document produced by a loader (requirements FR-ING-01)."""

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_format: Literal["pdf", "docx", "html", "md", "txt", "synthetic"]
    text: str
    language: str | None = None
    license: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    byte_size: int = Field(ge=0)
    headings: list[Heading] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def _hex_sha256(cls, value: str) -> str:
        if any(c not in "0123456789abcdef" for c in value):
            raise ValueError("sha256 must be lowercase hex")
        return value


class Chunk(Contract):
    """A retrievable unit with provenance (requirements FR-ING-03/04/05)."""

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    heading_path: list[str] = Field(default_factory=list)
    breadcrumb: str = ""
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    token_count: int = Field(ge=0)
    language: str | None = None
    context_header: str | None = None
    canonical_tokens: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _span_is_ordered(self) -> Chunk:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self

    @property
    def doc_hash(self) -> str:
        """Stable hash of the chunk text, used as the stale-citation guard (CR-6)."""
        import hashlib

        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class RetrievedPassage(Contract):
    """A chunk returned by a retrieval stage, with score and provenance (FR-RET-08)."""

    chunk: Chunk
    channel: Channel
    score: float
    rank: int = Field(ge=1)
    stage: Literal["retrieval", "fusion", "rerank", "pack"] = "retrieval"
    contributions: dict[str, float] = Field(default_factory=dict)


class FusionInfo(Contract):
    """Which fuser ran, with which parameters — part of every retrieval trace."""

    fuser: Literal["rrf", "cc", "none"]
    params: dict[str, float | int | str] = Field(default_factory=dict)


class RetrievalTrace(Contract):
    """Everything the console shows about retrieval (requirements FR-RET-08, FR-UI-04)."""

    query: str
    normalized_query: str | None = None
    language: str | None = None
    candidates: list[RetrievedPassage] = Field(default_factory=list)
    fused: list[RetrievedPassage] = Field(default_factory=list)
    reranked: list[RetrievedPassage] = Field(default_factory=list)
    packed: list[RetrievedPassage] = Field(default_factory=list)
    fusion: FusionInfo | None = None
    dropped_duplicates: int = Field(default=0, ge=0)
    identifier_tokens: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Generation / citations
# --------------------------------------------------------------------------------------


class Citation(Contract):
    """A verified citation: a chunk span plus its verbatim quote (requirements CR-2..CR-6)."""

    citation_id: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    doc_title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    quote: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    doc_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _span_is_ordered(self) -> Citation:
        if self.end_char <= self.start_char:
            raise ValueError("citation end_char must be greater than start_char")
        return self


class Claim(Contract):
    """One atomic assertion in an answer, with the citations that support it (CR-1)."""

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    citation_ids: list[int] = Field(default_factory=list)
    support: Support = Support.SUPPORTED


class Decision(Contract):
    """Policy output for one turn: action, reason code, score and threshold provenance."""

    action: Action
    reason_code: ReasonCode
    score: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(gt=0.0, lt=1.0)
    threshold_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def _reason_matches_action(self) -> Decision:
        if self.action is Action.ANSWER and self.reason_code is not ReasonCode.ANSWERED:
            raise ValueError("an 'answer' decision must carry ReasonCode.ANSWERED")
        if self.action is not Action.ANSWER and self.reason_code is ReasonCode.ANSWERED:
            raise ValueError(f"a '{self.action}' decision cannot carry ReasonCode.ANSWERED")
        return self


class EscalationDecision(Contract):
    """Escalation record attached to a turn and shown to the operator (FR-SVC-05, FR-UI-05)."""

    decision: Decision
    priority: Priority = Priority.NORMAL
    ticket_id: str | None = None
    assigned_to: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def _is_an_escalation(self) -> EscalationDecision:
        if self.decision.action not in (Action.ESCALATE, Action.REFUSE):
            raise ValueError("EscalationDecision requires action=escalate|refuse")
        return self


class Confidence(Contract):
    """Calibrated confidence of an auto-answer (requirements §8.1)."""

    p_correct: float = Field(ge=0.0, le=1.0)
    calibrator_version: str = Field(min_length=1)


class CacheInfo(Contract):
    """Cache status attached to every answer (requirements §9.1)."""

    status: CacheStatus = CacheStatus.MISS
    entry_id: str | None = None
    similarity: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _hit_has_entry(self) -> CacheInfo:
        if self.status is CacheStatus.HIT and not self.entry_id:
            raise ValueError("a cache hit must carry entry_id")
        return self


class Usage(Contract):
    """Token and cost accounting for one model interaction (NFR-COST-01)."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    usd: float = Field(default=0.0, ge=0.0)
    model: str = ""
    prompt_version: str = ""


class Provenance(Contract):
    """Versions that make an answer reproducible and a cache entry invalidatable."""

    corpus_version: str = ""
    index_version: str = ""
    prompt_version: str = ""
    verifier_version: str = ""
    scorer_version: str = ""


class Answer(Contract):
    """The displayed answer object (requirements §7.1). This is the frozen shape."""

    answer_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    language: str = Field(min_length=2)
    text: str
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    decision: Decision
    confidence: Confidence
    cache: CacheInfo = Field(default_factory=CacheInfo)
    usage: Usage = Field(default_factory=Usage)
    provenance: Provenance = Field(default_factory=Provenance)

    @model_validator(mode="after")
    def _citations_resolve(self) -> Answer:
        ids = [c.citation_id for c in self.citations]
        if len(ids) != len(set(ids)):
            raise ValueError("citation_id values must be unique within an answer")
        known = set(ids)
        for claim in self.claims:
            missing = [cid for cid in claim.citation_ids if cid not in known]
            if missing:
                raise ValueError(
                    f"claim {claim.claim_id!r} references unknown citation ids {missing}"
                )
        return self

    @model_validator(mode="after")
    def _displayed_claims_are_cited(self) -> Answer:
        if self.decision.action is Action.ANSWER:
            for claim in self.claims:
                if not claim.citation_ids:
                    raise ValueError(
                        f"claim {claim.claim_id!r} has no citation; an answer may not "
                        "display an uncited claim (requirements CR-1 / INV-03)"
                    )
                if claim.support is Support.UNSUPPORTED:
                    raise ValueError(
                        f"claim {claim.claim_id!r} is unsupported; unsupported claims may "
                        "not be displayed (requirements CR-7 / INV-03)"
                    )
        return self


# --------------------------------------------------------------------------------------
# Provider layer
# --------------------------------------------------------------------------------------


class ProviderConfig(Contract):
    """Resolved provider configuration (NFR-PORT-01). Secrets never serialise in clear."""

    preset: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key: SecretStr | None = None
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    json_mode: bool = False
    headers: dict[str, str] = Field(default_factory=dict)
    requires_api_key: bool = True

    def redacted(self) -> dict[str, Any]:
        """Serialisable view with the key replaced — safe for logs and `config show`."""
        data = self.model_dump(mode="json")
        data["api_key"] = "***" if self.api_key is not None else None
        return data


class Message(Contract):
    role: Role
    content: str


class Completion(Contract):
    """Non-streaming provider response."""

    text: str
    model: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    finish_reason: Literal["stop", "length", "content_filter", "error"] = "stop"
    latency_ms: float = Field(default=0.0, ge=0.0)


class CompletionChunk(Contract):
    """One streamed delta from a provider."""

    delta: str = ""
    done: bool = False
    model: str = ""


class PricingEntry(Contract):
    """Pinned model pricing with provenance (NFR-COST-02)."""

    model: str = Field(min_length=1)
    input_usd_per_1k_tokens: float = Field(ge=0.0)
    output_usd_per_1k_tokens: float = Field(ge=0.0)
    retrieved_at: str = Field(min_length=10)
    source: str = Field(min_length=1)


# --------------------------------------------------------------------------------------
# Observability / evaluation
# --------------------------------------------------------------------------------------


class TraceRecord(Contract):
    """One JSONL trace line per turn (NFR-OBS-01, FR-SVC-08)."""

    trace_id: str = Field(min_length=1)
    session_id: str | None = None
    turn_id: str | None = None
    query: str
    language: str | None = None
    intent_bucket: IntentBucket | None = None
    decision: Decision | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval: RetrievalTrace | None = None
    cache_status: CacheStatus = CacheStatus.MISS
    usage: Usage = Field(default_factory=Usage)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    corpus_version: str = ""
    index_version: str = ""
    prompt_version: str = ""
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvalProvenance(Contract):
    """How a golden case came to exist — required for honest reporting (§10.1)."""

    source: Literal["document", "llm_drafted_human_verified", "perturbation", "public_dataset"] = (
        "document"
    )
    generator: str = "unknown"
    seed: int = 0
    document_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class EvalCase(Contract):
    """One golden-set item: retrieval labels and answer labels kept separate (§10.1)."""

    case_id: str = Field(min_length=1)
    locale: str = Field(min_length=2)
    category: EvalCategory
    question: str = Field(min_length=1)
    answerable: bool
    intent_bucket: IntentBucket = IntentBucket.INFORMATIONAL
    split: Split = Split.TEST
    gold_chunk_ids: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    required_citation_chunk_ids: list[str] = Field(default_factory=list)
    provenance: EvalProvenance = Field(default_factory=EvalProvenance)
    ambiguous: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unanswerable_has_no_gold_answer(self) -> EvalCase:
        if not self.answerable and self.reference_answer is not None:
            raise ValueError("an unanswerable case must not carry a reference answer")
        if self.answerable and not self.gold_chunk_ids:
            raise ValueError("an answerable case must carry at least one gold chunk id")
        return self


class ClaimVerification(Contract):
    """Verifier verdict for one claim (requirements §7.3, F-9)."""

    claim_id: str = Field(min_length=1)
    verdict: Support
    rationale: str = ""
    verifier_version: str = ""


class JudgeScore(Contract):
    """Rubric score from a judge, with the metadata required to trust it (§10.5)."""

    score: float = Field(ge=1.0, le=5.0)
    rubric_version: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    rationale: str = ""
    order_swapped: bool = False


# --------------------------------------------------------------------------------------
# Protocol seams
# --------------------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors (t5 owns index/retrieval, t4 owns corpus text)."""

    name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[RetrievedPassage]: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, passages: Sequence[RetrievedPassage], top_n: int
    ) -> list[RetrievedPassage]: ...


@runtime_checkable
class FusionStrategy(Protocol):
    name: str

    def fuse(
        self, result_sets: Mapping[Channel, Sequence[RetrievedPassage]], k: int
    ) -> list[RetrievedPassage]: ...


@runtime_checkable
class ChatProvider(Protocol):
    """One OpenAI-compatible transport for every provider (NFR-PORT-01, ADR-003)."""

    name: str
    config: ProviderConfig

    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_schema: Mapping[str, Any] | None = None,
    ) -> Completion: ...

    def stream(self, messages: Sequence[Message]) -> Sequence[CompletionChunk]: ...


@runtime_checkable
class ClaimVerifier(Protocol):
    name: str

    def verify(
        self, claims: Sequence[Claim], passages: Mapping[str, Sequence[Chunk]]
    ) -> list[ClaimVerification]: ...


@runtime_checkable
class Judge(Protocol):
    name: str

    def grade(
        self,
        *,
        question: str,
        reference_answer: str | None,
        candidate_answer: str,
        context: Sequence[str],
    ) -> JudgeScore: ...


@runtime_checkable
class AnswerabilityScorer(Protocol):
    """Produces the raw score that conformal calibration turns into a decision (FR-ESC-01)."""

    name: str
    version: str

    def score(self, query: str, passages: Sequence[RetrievedPassage]) -> float: ...


@runtime_checkable
class CacheBackend(Protocol):
    """Minimal semantic-cache seam; guards live in ``answerdesk.cache`` (requirements §9)."""

    name: str

    def get(self, key: str) -> Answer | None: ...

    def put(self, key: str, value: Answer) -> None: ...

    def invalidate(self, corpus_version: str) -> int: ...

    def stats(self) -> Mapping[str, float]: ...


@runtime_checkable
class DocumentLoader(Protocol):
    def load(self, path: str) -> Document: ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class TraceSink(Protocol):
    def write(self, record: TraceRecord) -> None: ...


__all__ = [
    "SCHEMA_VERSION",
    "Action",
    "Answer",
    "AnswerabilityScorer",
    "CacheBackend",
    "CacheInfo",
    "CacheStatus",
    "Channel",
    "ChatProvider",
    "Chunk",
    "Chunker",
    "Citation",
    "Claim",
    "ClaimVerification",
    "ClaimVerifier",
    "Completion",
    "CompletionChunk",
    "Confidence",
    "Contract",
    "Decision",
    "Document",
    "DocumentLoader",
    "Embedder",
    "EscalationDecision",
    "EvalCase",
    "EvalCategory",
    "EvalProvenance",
    "FusionInfo",
    "FusionStrategy",
    "Heading",
    "IntentBucket",
    "Judge",
    "JudgeScore",
    "Message",
    "PricingEntry",
    "Priority",
    "Provenance",
    "ProviderConfig",
    "ReasonCode",
    "Reranker",
    "RetrievalTrace",
    "RetrievedPassage",
    "Retriever",
    "Role",
    "Split",
    "Support",
    "TraceRecord",
    "TraceSink",
    "Usage",
]
