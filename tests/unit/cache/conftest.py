"""Fixtures for the cache tests: chunks, answers, versions and a calibrated cache.

# AT-CAC-01  # AT-CAC-02
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from answerdesk.cache.keys import VersionTuple
from answerdesk.cache.replay import embed_replay_set, generate_replay_set
from answerdesk.cache.semantic_cache import SemanticCache
from answerdesk.cache.thresholds import (
    ThresholdTable,
    build_threshold_table,
)
from answerdesk.contracts import (
    Action,
    Answer,
    CacheInfo,
    Chunk,
    Citation,
    Claim,
    Confidence,
    Decision,
    Embedder,
    Provenance,
    ReasonCode,
    Usage,
)
from answerdesk.testing.doubles import HashingEmbedder

CARD_TEXT = (
    "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц. "
    "Плата списывается первого числа каждого месяца."
)


@pytest.fixture(scope="session")
def versions() -> VersionTuple:
    return VersionTuple(
        corpus_version="corpus-1",
        index_version="index-1",
        prompt_version="grounded-answer-1.0.0",
        model_id="fake-model",
        policy_artifact_hash="c" * 64,
    )


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    return HashingEmbedder(dimension=64)


@pytest.fixture()
def make_chunk() -> Callable[..., Chunk]:
    def _make(
        chunk_id: str = "doc-fee::0",
        *,
        text: str = CARD_TEXT,
        doc_id: str = "doc-fee",
        title: str = "Тарифы по карте «Вектор»",
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            heading_path=[title, "Обслуживание"],
            breadcrumb=f"{title} > Обслуживание",
            start_char=0,
            end_char=len(text),
            token_count=max(1, len(text.split())),
            language="ru",
            metadata={"doc_title": title},
        )

    return _make


@pytest.fixture()
def make_answer() -> Callable[..., Answer]:
    def _make(
        *,
        chunk: Chunk | None = None,
        claim_text: str = "Обслуживание карты «Вектор» стоит 120 рублей в месяц.",
        quote: str | None = None,
        action: Action = Action.ANSWER,
        reason: ReasonCode | None = None,
        corpus_version: str = "corpus-1",
        prompt_version: str = "grounded-answer-1.0.0",
        model: str = "fake-model",
        answer_id: str = "answer-1",
    ) -> Answer:
        resolved_chunk = chunk or make_chunk()  # type: ignore[call-arg]
        resolved_quote = quote or "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц."
        start = resolved_chunk.text.find(resolved_quote)
        citation = Citation(
            citation_id=1,
            chunk_id=resolved_chunk.chunk_id,
            doc_id=resolved_chunk.doc_id,
            doc_title=str(resolved_chunk.metadata.get("doc_title", "")),
            heading_path=list(resolved_chunk.heading_path),
            quote=resolved_quote,
            start_char=max(0, start),
            end_char=max(1, start + len(resolved_quote)),
            doc_hash=resolved_chunk.doc_hash,
        )
        claims = (
            [Claim(claim_id="c1", text=claim_text, citation_ids=[1])]
            if action is Action.ANSWER
            else []
        )
        return Answer(
            answer_id=answer_id,
            trace_id="trace-1",
            language="ru",
            text=f"{claim_text} [1]" if claims else "",
            claims=claims,
            citations=[citation] if claims else [],
            decision=Decision(
                action=action,
                reason_code=reason
                or (ReasonCode.ANSWERED if action is Action.ANSWER else ReasonCode.NO_EVIDENCE),
                score=0.99,
                alpha=0.05,
                threshold_ref="cal-test",
            ),
            confidence=Confidence(p_correct=0.99, calibrator_version="test"),
            cache=CacheInfo(status="miss"),
            usage=Usage(input_tokens=10, output_tokens=5, model=model),
            provenance=Provenance(
                corpus_version=corpus_version,
                index_version="index-1",
                prompt_version=prompt_version,
                verifier_version="lexical-verifier-1.0.0",
                scorer_version="overlap-1.0.0",
            ),
        )

    return _make


@pytest.fixture(scope="session")
def calibrated_table(embedder: Embedder) -> ThresholdTable:
    replay = generate_replay_set(seed=11, pairs_per_cohort=40)
    embedded = embed_replay_set(replay, embedder)
    similarities: dict[str, list[float]] = {}
    wrong: dict[str, list[bool]] = {}
    compatible: dict[str, list[bool]] = {}
    exact: dict[str, list[bool]] = {}
    for item in embedded:
        similarities.setdefault(item.pair.cohort, []).append(item.similarity)
        wrong.setdefault(item.pair.cohort, []).append(item.pair.wrong_if_hit)
        compatible.setdefault(item.pair.cohort, []).append(item.entities_compatible)
        exact.setdefault(item.pair.cohort, []).append(item.exact_equivalent)
    return build_threshold_table(
        similarities_by_cohort=similarities,
        wrong_by_cohort=wrong,
        compatible_by_cohort=compatible,
        exact_by_cohort=exact,
        replay_hash=replay.replay_hash,
        default_threshold=0.99,
        calibrated_at="2026-10-21T00:00:00+00:00",
    )


@pytest.fixture()
def make_cache(
    embedder: Embedder, versions: VersionTuple, make_chunk: Callable[..., Chunk]
) -> Callable[..., SemanticCache]:
    def _make(
        *,
        table: ThresholdTable | None = None,
        chunks: dict[str, Chunk] | None = None,
        enabled: bool = True,
        clock: Callable[[], float] | None = None,
        validate_on_hit: bool = True,
        high_risk_intents: frozenset[str] | None = None,
    ) -> SemanticCache:
        lookup_map = chunks if chunks is not None else {}
        return SemanticCache(
            embedder=embedder,
            versions=versions,
            thresholds=table,
            enabled=enabled,
            default_threshold=0.99,
            chunk_lookup=lambda chunk_id: lookup_map.get(chunk_id),
            clock=clock or (lambda: 1000.0),
            validate_on_hit=validate_on_hit,
            high_risk_intents=high_risk_intents,
        )

    return _make
