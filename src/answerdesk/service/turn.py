"""One customer turn: guards → cache → retrieval → generation → policy → persistence.

The runner is the only place that knows the *order* of the pipeline, and the order is the
product's safety contract (requirements §8.4, FR-SVC-01/02/08):

1. **redact + identify** — the message is redacted before it is stored or traced, the
   session is resolved (created on first use), and the conversation is replayed into a
   resolved query (:func:`answerdesk.service.signals.replay_history`);
2. **guard pre-flight** — PII / injection / hard-policy guards run *before* any provider
   call, so a blocked turn costs nothing and never sends PII outward (INV-05);
3. **cache** — an exact or guarded-semantic hit short-circuits retrieval and generation,
   but re-runs the guards against the *new* query first (a cached answer is not a licence
   to skip the security check);
4. **retrieve** — ``RetrievalPipeline.retrieve_with_timings`` gives the trace and the
   per-stage milliseconds (NFR-COST-03);
5. **generate** — the grounded pipeline produces a verified, cited answer or an explicit
   signal (t6 owns the verification);
6. **display check + decide** — a cheap service-side re-validation of the rendered text,
   then the calibrated decision engine (guards, conformity, cost bands);
7. **persist + trace + metrics** — the turn is appended to SQLite, one JSONL trace line is
   written with every stage, and the counters are updated.

The result is a :class:`TurnResult`: a frozen :class:`~answerdesk.contracts.Answer`, its
:class:`~answerdesk.contracts.Decision`, the timings, and everything the SSE route and the
console need. Nothing in the route mutates pipeline state.

    # AT-SVC-01  # AT-SVC-02  # AT-SVC-08  # AT-OBS-01  # AT-COST-03  # AT-OPS-07
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from answerdesk.contracts import (
    Action,
    Answer,
    CacheInfo,
    CacheStatus,
    Confidence,
    Decision,
    IntentBucket,
    Priority,
    Provenance,
    ReasonCode,
    RetrievalTrace,
    RetrievedPassage,
    TraceRecord,
    Usage,
)
from answerdesk.generate.citations import validate_render
from answerdesk.generate.language import detect_language
from answerdesk.generate.numeric import numeric_violations
from answerdesk.generate.pipeline import GenerationResult
from answerdesk.policy.decision import PolicyOutcome
from answerdesk.service.signals import (
    HistoryReplay,
    normalise_passages,
    preflight_decision,
    replay_history,
)
from answerdesk.telemetry.redaction import redact_text
from answerdesk.telemetry.tracing import trace_line

#: Priority per risk bucket for auto-created escalation tickets.
_PRIORITY_BY_BUCKET: dict[IntentBucket, Priority] = {
    IntentBucket.INFORMATIONAL: Priority.NORMAL,
    IntentBucket.TRANSACTIONAL: Priority.HIGH,
    IntentBucket.ACCOUNT_LEGAL: Priority.URGENT,
}

#: The trace stages the service emits, in pipeline order.
TRACE_STAGES: tuple[str, ...] = (
    "history",
    "retrieval",
    "rerank",
    "generation",
    "verify",
    "policy",
    "cache",
    "total",
)


def new_trace_id() -> str:
    return f"tr-{uuid.uuid4().hex[:24]}"


@dataclass(frozen=True)
class TurnRequest:
    """A validated request, independent of the HTTP layer."""

    message: str
    session_id: str | None = None
    locale: str | None = None
    intent_bucket: IntentBucket = IntentBucket.INFORMATIONAL
    intent_label: str | None = None
    idempotency_key: str | None = None


@dataclass
class TurnResult:
    """Everything one turn produced (the route serialises this)."""

    trace_id: str
    session_id: str
    turn_id: str
    answer: Answer
    decision: Decision
    usage: Usage
    timings_ms: dict[str, float] = field(default_factory=dict)
    retrieval: RetrievalTrace | None = None
    generation: GenerationResult | None = None
    cache_status: CacheStatus = CacheStatus.MISS
    cache_entry_id: str | None = None
    cache_similarity: float | None = None
    cache_reason: str = ""
    p_hat: float = 0.0
    history_turns: int = 0
    follow_up: bool = False
    retrieval_query: str = ""
    ticket_id: str | None = None
    history: HistoryReplay | None = None
    from_cache: bool = False

    @property
    def answered(self) -> bool:
        return self.decision.action is Action.ANSWER

    def answer_payload(self) -> dict[str, Any]:
        return self.answer.model_dump(mode="json")


class TurnRunner:
    """Run one turn against a :class:`~answerdesk.service.runtime.ServiceRuntime`."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    # -- public -------------------------------------------------------------------
    def run(self, request: TurnRequest, *, trace_id: str | None = None) -> TurnResult:
        runtime = self.runtime
        runtime.require_answer_path()
        assert runtime.engine is not None  # narrowed by require_answer_path
        assert runtime.retriever is not None

        started = time.perf_counter()
        timings: dict[str, float] = {}
        identifiers = trace_id or new_trace_id()
        redacted = redact_text(request.message)
        language = _language(request.locale, request.message)

        session, created = runtime.store.ensure_session(
            request.session_id, locale=request.locale or language
        )
        if created:
            runtime.metrics.sessions_created.inc()

        anchor = time.perf_counter()
        history_rows = runtime.store.history(session.session_id, limit=6)
        replay = replay_history(history_rows, request.message)
        timings["history"] = _ms(anchor)

        cache_status = CacheStatus.MISS
        cache_entry_id: str | None = None
        cache_similarity: float | None = None
        cache_reason = ""
        retrieval: RetrievalTrace | None = None
        generation: GenerationResult | None = None
        usage = Usage()
        p_hat = 0.0
        outcome: PolicyOutcome | None = None
        decision: Decision | None = None
        answer: Answer | None = None

        try:
            # 2. guard pre-flight (no provider call, no retrieval)
            decision = preflight_decision(
                runtime.engine,
                query=request.message,
                bucket=request.intent_bucket,
                intent_label=request.intent_label,
            )
            if decision is not None:
                p_hat = decision.score
                timings["policy"] = _ms(anchor)
            else:
                # 3. cache lookup (guarded, and re-guarded against this query)
                if runtime.cache is not None and runtime.cache.enabled:
                    anchor = time.perf_counter()
                    lookup = runtime.cache.get(
                        query=replay.query,
                        locale=request.locale or language,
                        intent_bucket=request.intent_bucket,
                        intent_label=request.intent_label,
                    )
                    timings["cache"] = _ms(anchor)
                    cache_status = lookup.status
                    cache_entry_id = lookup.entry_id
                    cache_similarity = lookup.similarity
                    cache_reason = lookup.reason
                    runtime.metrics.observe_cache_status(
                        lookup.status.value, kind=_cache_kind(lookup)
                    )
                    if lookup.hit and lookup.answer is not None:
                        regenerated = preflight_decision(
                            runtime.engine,
                            query=request.message,
                            bucket=request.intent_bucket,
                            intent_label=request.intent_label,
                        )
                        if regenerated is None:
                            answer = _from_cache(
                                lookup.answer,
                                identifiers,
                                entry_id=lookup.entry_id,
                                similarity=lookup.similarity,
                            )
                            decision = lookup.answer.decision
                            p_hat = decision.score
                            cache_status = CacheStatus.HIT
                            usage = Usage()
                            runtime.metrics.cache_hit_latency.observe(timings["cache"] / 1000.0)
                        else:
                            cache_status = CacheStatus.MISS
                            cache_reason = "guard demoted a cache hit"
                            decision = regenerated
                            p_hat = decision.score

                if answer is None and decision is None:
                    # 4. retrieval
                    anchor = time.perf_counter()
                    retrieval, stage_timings = runtime.retriever.retrieve_with_timings(
                        replay.query
                    )
                    timings["retrieval"] = _ms(anchor)
                    if "rerank_ms" in stage_timings:
                        timings["rerank"] = float(stage_timings["rerank_ms"])
                    packed = list(retrieval.packed)
                    scored = normalise_passages(packed)
                    p_hat = runtime.scorer.score(replay.query, scored)

                    # 5. generation
                    anchor = time.perf_counter()
                    generation = runtime.answerer.answer(replay.query, packed)
                    timings["generation"] = _ms(anchor)
                    usage = generation.usage

                    # 6a. display check: re-validate what is about to be shown
                    anchor = time.perf_counter()
                    display_problems = _display_problems(generation)
                    timings["verify"] = _ms(anchor)

                    # 6b. policy
                    anchor = time.perf_counter()
                    outcome = runtime.engine.evaluate(
                        p_correct=p_hat,
                        bucket=request.intent_bucket,
                        query=request.message,
                        passages=packed,
                        intent_label=request.intent_label,
                        generation=generation.signal,
                    )
                    decision = outcome.decision
                    timings["policy"] = _ms(anchor)
                    if display_problems and decision.action is Action.ANSWER:
                        decision = _forced(
                            runtime.engine,
                            request.intent_bucket,
                            p_hat,
                            ReasonCode.CITATION_FAILED,
                            "display check failed: " + "; ".join(display_problems[:3]),
                        )

            assert decision is not None  # every path above sets it
            answer = answer or _assemble(
                runtime,
                request=request,
                identifiers=identifiers,
                decision=decision,
                generation=generation,
                usage=usage,
                cache_status=cache_status,
                cache_entry_id=cache_entry_id,
                cache_similarity=cache_similarity,
                p_hat=p_hat,
                language=language,
            )
            if answer.cache.status is not CacheStatus.HIT:
                answer = answer.model_copy(
                    update={
                        "cache": CacheInfo(
                            status=cache_status,
                            entry_id=cache_entry_id if cache_status is CacheStatus.HIT else None,
                            similarity=cache_similarity,
                        )
                    }
                )

            timings["total"] = _ms(started)
            turn = runtime.store.append_turn(
                session_id=session.session_id,
                message=request.message,
                trace_id=identifiers,
                action=decision.action.value,
                reason_code=decision.reason_code.value,
                answer_text=answer.text,
                detail=_detail(generation, outcome, cache_reason),
                idempotency_key=request.idempotency_key,
            )

            ticket_id: str | None = None
            if decision.action in (Action.ESCALATE, Action.REFUSE):
                ticket = runtime.store.create_ticket(
                    trace_id=identifiers,
                    session_id=session.session_id,
                    reason=decision.reason_code.value,
                    priority=_PRIORITY_BY_BUCKET[request.intent_bucket].value,
                    note=_detail(generation, outcome, cache_reason),
                    status="auto",
                )
                ticket_id = ticket.ticket_id

            if (
                answer.decision.action is Action.ANSWER
                and cache_status is not CacheStatus.HIT
                and runtime.cache is not None
                and runtime.cache.enabled
            ):
                _cache_put(runtime, replay, request, language, answer, retrieval)

            result = TurnResult(
                trace_id=identifiers,
                session_id=session.session_id,
                turn_id=turn.turn_id,
                answer=answer,
                decision=decision,
                usage=usage,
                timings_ms=timings,
                retrieval=retrieval,
                generation=generation,
                cache_status=cache_status,
                cache_entry_id=cache_entry_id,
                cache_similarity=cache_similarity,
                cache_reason=cache_reason,
                p_hat=p_hat,
                history_turns=len(history_rows),
                follow_up=replay.follow_up,
                retrieval_query=replay.query,
                ticket_id=ticket_id,
                history=replay,
                from_cache=cache_status is CacheStatus.HIT,
            )
            self._record(
                runtime,
                result,
                redacted,
                language,
                bucket=request.intent_bucket,
                error=None,
            )
            return result
        except Exception as exc:
            timings["total"] = _ms(started)
            self._record_error(
                runtime,
                trace_id=identifiers,
                session_id=session.session_id,
                query=redacted.text,
                language=language,
                timings=timings,
                exc=exc,
                retrieval=retrieval,
                decision=decision,
                p_hat=p_hat,
                usage=usage,
            )
            raise

    # -- telemetry ----------------------------------------------------------------
    def _record(
        self,
        runtime: Any,
        result: TurnResult,
        query: Any,
        language: str,
        *,
        bucket: IntentBucket,
        error: str | None,
    ) -> None:
        record = TraceRecord(
            trace_id=result.trace_id,
            session_id=result.session_id,
            turn_id=result.turn_id,
            query=query.text,
            language=language,
            intent_bucket=bucket,
            decision=result.decision,
            confidence=result.p_hat,
            retrieval=result.retrieval,
            cache_status=result.cache_status,
            usage=result.usage,
            timings_ms=_trace_timings(result.timings_ms),
            corpus_version=runtime.info.corpus_version,
            index_version=runtime.info.index_version,
            prompt_version=runtime.info.prompt_version,
            error=error,
        )
        extra = {
            "service": {
                "version": runtime.info.profile,
                "profile": runtime.info.profile,
                "fake": runtime.info.fake,
                "model": runtime.info.model,
                "preset": runtime.info.preset,
                "corpus_version": runtime.info.corpus_version,
                "index_version": runtime.info.index_version,
                "prompt_version": runtime.info.prompt_version,
                "calibration": {
                    "source": runtime.info.calibration.source,
                    "threshold_ref": runtime.info.calibration.threshold_ref,
                    "stale": runtime.info.calibration.stale,
                    "age_days": runtime.info.calibration.age_days,
                },
                "scorer": {"name": runtime.scorer.name, "version": runtime.scorer.version},
            },
            "retrieval_scores": _scores(result.retrieval),
            "fusion": (
                result.retrieval.fusion.model_dump(mode="json")
                if result.retrieval is not None and result.retrieval.fusion is not None
                else None
            ),
            "cache": {
                "status": result.cache_status.value,
                "entry_id": result.cache_entry_id,
                "similarity": result.cache_similarity,
                "reason": result.cache_reason,
            },
            "escalation": (
                {
                    "ticket_id": result.ticket_id,
                    "reason": result.decision.reason_code.value,
                }
                if result.ticket_id
                else None
            ),
            "history": (
                result.history.as_dict() if result.history is not None else {"turns_used": 0}
            ),
            "generation": (
                {
                    "repaired": result.generation.repaired,
                    "repair_attempts": result.generation.repair_attempts,
                    "dropped_claims": result.generation.dropped_claims,
                    "numeric_violations_pre_repair": (
                        result.generation.numeric_violations_pre_repair
                    ),
                    "verifier_version": result.generation.verifier_version,
                    "language_notice": result.generation.language_notice,
                }
                if result.generation is not None
                else None
            ),
            "provider_redactions": getattr(runtime.provider, "redactions", 0),
            "query_pii": {"redacted": getattr(query, "count", 0), "kinds": list(getattr(query, "kinds", ()))},
        }
        line, _summary = trace_line(record, extra=extra)
        runtime.tracer.write(line)
        runtime.metrics.observe_trace(outcome="ok")
        runtime.metrics.observe_decision(
            result.decision.action.value, result.decision.reason_code.value
        )
        for stage, value in result.timings_ms.items():
            runtime.metrics.observe_stage(stage, value / 1000.0)
        runtime.metrics.observe_usage(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            usd=result.usage.usd,
            model=result.usage.model or runtime.info.model,
        )
        if runtime.cache is not None:
            runtime.metrics.set_cache_entries(runtime.cache.stats().entries)

    def _record_error(
        self,
        runtime: Any,
        *,
        trace_id: str,
        session_id: str,
        query: str,
        language: str,
        timings: dict[str, float],
        exc: BaseException,
        retrieval: RetrievalTrace | None,
        decision: Decision | None,
        p_hat: float,
        usage: Usage,
    ) -> None:
        record = TraceRecord(
            trace_id=trace_id,
            session_id=session_id,
            turn_id=None,
            query=query,
            language=language,
            decision=decision,
            confidence=p_hat,
            retrieval=retrieval,
            cache_status=CacheStatus.MISS,
            usage=usage,
            timings_ms=_trace_timings(timings),
            corpus_version=runtime.info.corpus_version,
            index_version=runtime.info.index_version,
            prompt_version=runtime.info.prompt_version,
            error=f"{type(exc).__name__}: {exc}",
        )
        line, _summary = trace_line(
            record,
            extra={"service": {"profile": runtime.info.profile, "fake": runtime.info.fake}},
        )
        try:
            runtime.tracer.write(line)
            runtime.metrics.observe_trace(outcome="error")
        except Exception:  # pragma: no cover - tracing must never mask the real error
            pass

# -- assembly helpers ------------------------------------------------------------------
def _language(hint: str | None, text: str) -> str:
    if hint and len(hint) >= 2:
        return hint[:8]
    detected = detect_language(text)
    return detected if detected and len(detected) >= 2 else "en"


def _assemble(
    runtime: Any,
    *,
    request: TurnRequest,
    identifiers: str,
    decision: Decision,
    generation: GenerationResult | None,
    usage: Usage,
    cache_status: CacheStatus,
    cache_entry_id: str | None,
    cache_similarity: float | None,
    p_hat: float,
    language: str,
) -> Answer:
    answered = decision.action is Action.ANSWER
    text = generation.text if (answered and generation is not None) else ""
    claims = list(generation.claims) if (answered and generation is not None) else []
    citations = list(generation.citations) if (answered and generation is not None) else []
    return Answer(
        answer_id=f"ans-{uuid.uuid4().hex[:16]}",
        trace_id=identifiers,
        language=generation.language if generation is not None and generation.language else language,
        text=text,
        claims=claims,
        citations=citations,
        decision=decision,
        confidence=Confidence(
            p_correct=round(min(1.0, max(0.0, p_hat)), 10),
            calibrator_version=runtime.info.calibration.threshold_ref or "uncalibrated",
        ),
        cache=CacheInfo(
            status=cache_status,
            entry_id=cache_entry_id if cache_status is CacheStatus.HIT else None,
            similarity=cache_similarity,
        ),
        usage=usage,
        provenance=Provenance(
            corpus_version=runtime.info.corpus_version,
            index_version=runtime.info.index_version,
            prompt_version=generation.prompt_version if generation is not None else runtime.info.prompt_version,
            verifier_version=generation.verifier_version if generation is not None else "",
            scorer_version=runtime.scorer.version,
        ),
    )


def _from_cache(
    cached: Answer,
    trace_id: str,
    *,
    entry_id: str | None,
    similarity: float | None,
) -> Answer:
    """Re-issue a cached answer for the current trace (ids change, content does not)."""
    return cached.model_copy(
        update={
            "answer_id": f"ans-{uuid.uuid4().hex[:16]}",
            "trace_id": trace_id,
            "cache": CacheInfo(
                status=CacheStatus.HIT,
                entry_id=entry_id or cached.cache.entry_id or "unknown",
                similarity=similarity,
            ),
        }
    )


def _forced(
    engine: Any,
    bucket: IntentBucket,
    p_hat: float,
    reason: ReasonCode,
    detail: str,
) -> Decision:
    entry = engine.thresholds.for_bucket(bucket)
    return Decision(
        action=Action.ESCALATE,
        reason_code=reason,
        score=round(min(1.0, max(0.0, p_hat)), 10),
        alpha=entry.alpha_final,
        threshold_ref=engine.threshold_ref(),
    )


def _display_problems(generation: GenerationResult | None) -> list[str]:
    """Cheap, deterministic re-validation of what the client is about to render."""
    if generation is None or generation.signal.action is not Action.ANSWER:
        return []
    problems = list(validate_render(generation.text, generation.claims, generation.citations))
    cited = {citation.chunk_id for citation in generation.citations}
    if not cited:
        problems.append("an answer without a citation cannot be displayed (INV-02)")
    by_id = {citation.citation_id: citation for citation in generation.citations}
    for citation in generation.citations:
        if not citation.quote.strip():
            problems.append(f"citation {citation.citation_id} has an empty quote")
    for claim in generation.claims:
        quoted = [
            by_id[citation_id].quote
            for citation_id in claim.citation_ids
            if citation_id in by_id
        ]
        problems.extend(f"numeric guard: {item}" for item in numeric_violations(claim.text, quoted))
    return problems


def _cache_put(
    runtime: Any,
    replay: HistoryReplay,
    request: TurnRequest,
    language: str,
    answer: Answer,
    retrieval: RetrievalTrace | None,
) -> None:
    fingerprint: dict[str, Any] = {}
    if retrieval is not None:
        fingerprint = {
            "chunk_ids": [passage.chunk.chunk_id for passage in retrieval.packed],
            "scores": [round(passage.score, 6) for passage in retrieval.packed],
        }
    try:
        runtime.cache.put(
            query=replay.query,
            locale=request.locale or language,
            intent_bucket=request.intent_bucket,
            answer=answer,
            retrieval_fingerprint=fingerprint,
        )
    except Exception:  # pragma: no cover - a cache write must never fail a turn
        pass


def _scores(retrieval: RetrievalTrace | None) -> list[dict[str, Any]]:
    if retrieval is None:
        return []
    return [
        {
            "chunk_id": passage.chunk.chunk_id,
            "doc_id": passage.chunk.doc_id,
            "channel": passage.channel.value,
            "score": round(passage.score, 8),
            "rank": passage.rank,
            "stage": passage.stage,
        }
        for passage in retrieval.packed
    ]


def _trace_timings(timings: dict[str, float]) -> dict[str, float]:
    """Map the service's stage names onto the trace's documented stage names."""
    mapped = {
        "analysis": 0.0,
        "retrieval": round(timings.get("retrieval", 0.0), 3),
        "rerank": round(timings.get("rerank", 0.0), 3),
        "generation": round(timings.get("generation", 0.0), 3),
        "verify": round(timings.get("verify", 0.0), 3),
        "policy": round(timings.get("policy", 0.0), 3),
        "cache": round(timings.get("cache", 0.0), 3),
        "total": round(timings.get("total", 0.0), 3),
    }
    return mapped


def _detail(
    generation: GenerationResult | None, outcome: PolicyOutcome | None, cache_reason: str
) -> str:
    if outcome is not None and outcome.notes:
        return outcome.notes
    if generation is not None and generation.signal.detail:
        return generation.signal.detail
    return cache_reason


def _cache_kind(lookup: Any) -> str:
    if lookup.entry_id and lookup.similarity == 1.0:
        return "exact"
    if lookup.entry_id:
        return "semantic"
    return "exact"


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


__all__ = [
    "TRACE_STAGES",
    "TurnRequest",
    "TurnResult",
    "TurnRunner",
    "new_trace_id",
]
