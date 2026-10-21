"""The semantic cache: exact path, guards, negative caching, invalidation, counters.

# AT-CAC-01 … AT-CAC-09
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from answerdesk.cache.keys import CacheKey, VersionTuple, extract_entities, normalise_query
from answerdesk.cache.semantic_cache import CacheEntry, SemanticCache
from answerdesk.cache.thresholds import ThresholdTable
from answerdesk.contracts import Action, Answer, CacheStatus, Chunk, IntentBucket, ReasonCode

QUESTION = "Сколько стоит обслуживание карты?"


def _types(value: object) -> type[object]:
    return type(value)


# --------------------------------------------------------------------------------------
# Keys and entities
# --------------------------------------------------------------------------------------


def test_normalisation_makes_the_same_question_one_key(versions: VersionTuple) -> None:
    first = CacheKey.build(
        query="Сколько стоит обслуживание карты?",
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        versions=versions,
    )
    second = CacheKey.build(
        query="  сколько   СТОИТ обслуживание карты  ",
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        versions=versions,
    )
    assert first.normalised_query == second.normalised_query
    assert first.exact_key() == second.exact_key()
    assert normalise_query("«Вектор»?") == "вектор"


def test_entity_extraction_and_guard() -> None:
    left = extract_entities("возврат по заказу ORD-12345 на 4 500 рублей от 05.03.2026")
    right = extract_entities("возврат по заказу ORD-12345 на 4 500 рублей от 05.03.2026")
    other = extract_entities("возврат по заказу ORD-12345 на 5 000 рублей от 05.03.2026")
    assert left.matches(right) is True
    assert left.matches(other) is False
    assert "ORD-12345" in left.identifiers
    assert "4500" in left.numbers
    assert left.describes_only(other) is False


def test_version_tuple_participates_in_the_key(versions: VersionTuple) -> None:
    base = CacheKey.build(
        query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL, versions=versions
    )
    newer_corpus = versions.model_copy(update={"corpus_version": "corpus-2"})
    changed = CacheKey.build(
        query=QUESTION,
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        versions=newer_corpus,
    )
    assert base.exact_key() != changed.exact_key()


def test_a_cache_key_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        CacheKey(normalised_query="", locale="ru")


# --------------------------------------------------------------------------------------
# Exact path (FR-CAC-01)
# --------------------------------------------------------------------------------------


def test_exact_repeat_hits_without_recomputing(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
) -> None:
    """# AT-CAC-01 — second identical call is served from the cache."""
    chunk = make_chunk()
    cache = make_cache(chunks={chunk.chunk_id: chunk})
    answer = make_answer(chunk=chunk)
    cache.put(
        query=QUESTION,
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        answer=answer,
    )
    lookup = cache.get(
        query="сколько стоит обслуживание карты",
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
    )
    assert lookup.status is CacheStatus.HIT
    assert lookup.answer is not None
    assert lookup.answer.answer_id == answer.answer_id
    assert lookup.similarity == pytest.approx(1.0)
    assert cache.stats().exact_hits == 1


def test_missing_entry_is_a_miss(
    make_cache: Callable[..., SemanticCache],
) -> None:
    cache = make_cache()
    lookup = cache.get(query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL)
    assert lookup.status is CacheStatus.MISS
    assert lookup.answer is None


# --------------------------------------------------------------------------------------
# Semantic path and guards (FR-CAC-02, FR-CAC-03)
# --------------------------------------------------------------------------------------


def test_semantic_hit_above_the_calibrated_threshold(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
    calibrated_table: ThresholdTable,
) -> None:
    """# AT-CAC-02 — a close paraphrase hits when the cohort's τ allows it."""
    chunk = make_chunk()
    table = ThresholdTable(
        calibrated_at="2026-10-21T00:00:00+00:00",
        replay_hash="a" * 64,
        epsilon=0.01,
        default_threshold=0.5,
        cohorts=[],
    )
    cache = make_cache(table=table, chunks={chunk.chunk_id: chunk})
    cache.put(
        query=QUESTION,
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        answer=make_answer(chunk=chunk),
    )
    lookup = cache.get(
        query="Сколько стоит обслуживание карты «Вектор»?",
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
    )
    assert lookup.status is CacheStatus.HIT
    assert lookup.similarity is not None and lookup.similarity >= 0.5
    assert cache.stats().semantic_hits == 1

    strict = ThresholdTable(
        calibrated_at="2026-10-21T00:00:00+00:00",
        replay_hash="a" * 64,
        epsilon=0.01,
        default_threshold=0.999,
        cohorts=[],
    )
    strict_cache = make_cache(table=strict, chunks={chunk.chunk_id: chunk})
    strict_cache.put(
        query=QUESTION,
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        answer=make_answer(chunk=chunk),
    )
    miss = strict_cache.get(
        query="Сколько стоит обслуживание карты «Вектор»?",
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
    )
    assert miss.status is CacheStatus.MISS
    assert "threshold" in miss.reason


def test_entity_guard_blocks_a_near_miss(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
) -> None:
    """# AT-CAC-02 — a different amount is a different question, however similar."""
    chunk = make_chunk()
    table = ThresholdTable(
        calibrated_at="2026-10-21T00:00:00+00:00",
        replay_hash="a" * 64,
        epsilon=0.01,
        default_threshold=0.0,
        cohorts=[],
    )
    cache = make_cache(table=table, chunks={chunk.chunk_id: chunk})
    cache.put(
        query="Возврат по заказу ORD-12345 на 4 500 рублей",
        locale="ru",
        intent_bucket=IntentBucket.TRANSACTIONAL,
        answer=make_answer(chunk=chunk),
    )
    lookup = cache.get(
        query="Возврат по заказу ORD-12345 на 5 000 рублей",
        locale="ru",
        intent_bucket=IntentBucket.TRANSACTIONAL,
    )
    assert lookup.status is CacheStatus.MISS
    assert lookup.guard == "G-2"
    assert "entity" in lookup.reason


def _transplant(target: SemanticCache, source: SemanticCache) -> None:
    """Copy entries from one cache into another (to isolate a single guard)."""
    for entry in source.entries():
        target._by_key[entry.key.exact_key()] = entry  # type: ignore[attr-defined]
        target._by_id[entry.entry_id] = entry  # type: ignore[attr-defined]
        target.index.add(entry.entry_id, entry.embedding)


def test_source_doc_guard_detects_deleted_and_edited_chunks(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
) -> None:
    """# AT-CAC-03 — G-3: a citation to a deleted or edited chunk demotes the hit."""
    chunk = make_chunk()
    cache = make_cache(chunks={chunk.chunk_id: chunk})
    answer = make_answer(chunk=chunk)
    cache.put(query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL, answer=answer)

    # The entry is valid while the chunk exists ...
    assert (
        cache.get(query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL).status
        is CacheStatus.HIT
    )

    # ... and becomes a miss as soon as the cited chunk is gone (deleted document).
    deleted = make_cache(chunks={})
    _transplant(deleted, cache)
    missing = deleted.get(query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL)
    assert missing.status is CacheStatus.MISS
    assert missing.guard == "G-3"

    # An edited chunk with the same id is caught by the doc_hash comparison.
    edited_chunk = chunk.model_copy(update={"text": chunk.text + " Условия изменились."})
    edited = make_cache(chunks={chunk.chunk_id: edited_chunk})
    _transplant(edited, cache)
    demoted = edited.get(query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL)
    assert demoted.status is CacheStatus.MISS
    assert demoted.guard == "G-3"


def test_version_guard_rejects_a_different_live_version(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
    embedder: Any,
    versions: VersionTuple,
) -> None:
    """# AT-CAC-02 — G-1: a prompt/model/corpus change invalidates the entry."""
    chunk = make_chunk()
    table = ThresholdTable(
        calibrated_at="2026-10-21T00:00:00+00:00",
        replay_hash="a" * 64,
        epsilon=0.01,
        default_threshold=0.0,
        cohorts=[],
    )
    cache = make_cache(table=table, chunks={chunk.chunk_id: chunk})
    cache.put(
        query=QUESTION,
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        answer=make_answer(chunk=chunk),
    )
    newer = SemanticCache(
        embedder=embedder,
        versions=versions.model_copy(update={"prompt_version": "grounded-answer-1.1.0"}),
        thresholds=table,
        default_threshold=0.0,
        chunk_lookup=lambda chunk_id: chunk,
        clock=lambda: 1000.0,
    )
    # Reuse the stored entry so only the version guard can reject it.
    for entry in cache.entries():
        newer._by_key[entry.key.exact_key()] = entry  # type: ignore[attr-defined]
        newer._by_id[entry.entry_id] = entry  # type: ignore[attr-defined]
        newer.index.add(entry.entry_id, entry.embedding)
    lookup = newer.get(query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL)
    assert lookup.status is CacheStatus.MISS
    assert lookup.guard == "G-1"


def test_structural_revalidation_rejects_a_tampered_quote(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
) -> None:
    """# AT-CAC-03 — G-4: a quote that no longer matches its chunk is not served."""
    chunk = make_chunk()
    cache = make_cache(chunks={chunk.chunk_id: chunk})
    answer = make_answer(chunk=chunk)
    tampered = answer.model_copy(
        update={
            "citations": [
                answer.citations[0].model_copy(
                    update={"quote": "Совершенно другая фраза", "start_char": 0, "end_char": 19}
                )
            ]
        }
    )
    entry = cache.put(
        query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL, answer=tampered
    )
    assert isinstance(entry, CacheEntry)
    lookup = cache.get(query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL)
    assert lookup.status is CacheStatus.MISS
    assert lookup.guard == "G-4"


def test_high_risk_intents_bypass_the_semantic_path(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
) -> None:
    """# AT-CAC-06 — G-5: account/refund/legal/payment never take a semantic hit."""
    chunk = make_chunk()
    table = ThresholdTable(
        calibrated_at="2026-10-21T00:00:00+00:00",
        replay_hash="a" * 64,
        epsilon=0.01,
        default_threshold=0.0,
        cohorts=[],
    )
    cache = make_cache(table=table, chunks={chunk.chunk_id: chunk})
    cache.put(
        query=QUESTION,
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        answer=make_answer(chunk=chunk),
    )
    bypassed = cache.get(
        query="Сколько стоит обслуживание карты «Вектор»?",
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        intent_label="refund",
    )
    assert bypassed.status is CacheStatus.BYPASSED
    assert "G-5" in bypassed.reason

    # Exact match on a high-risk intent is still allowed (FR-CAC-06: exact-match only).
    exact = cache.get(
        query=QUESTION,
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        intent_label="refund",
    )
    assert exact.status is CacheStatus.HIT
    assert cache.stats().semantic_hits == 0


# --------------------------------------------------------------------------------------
# Negative caching, TTL, invalidation, kill switch, counters
# --------------------------------------------------------------------------------------


def test_negative_entries_are_cached_with_the_escalation_ttl(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
) -> None:
    """# AT-CAC-09 — an escalated turn is cached as a non-answer with a short TTL."""
    chunk = make_chunk()
    cache = make_cache(chunks={chunk.chunk_id: chunk})
    escalation = make_answer(chunk=chunk, action=Action.ESCALATE, reason=ReasonCode.NO_EVIDENCE)
    entry = cache.put(
        query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL, answer=escalation
    )
    assert entry.negative is True
    assert entry.ttl_class == "escalation"
    assert entry.ttl_seconds == 600
    lookup = cache.get(query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL)
    assert lookup.status is CacheStatus.HIT
    assert lookup.answer is not None
    assert lookup.answer.text == ""  # never prose
    assert lookup.answer.decision.action is Action.ESCALATE


def test_ttl_expiry_demotes_to_a_miss(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
) -> None:
    chunk = make_chunk()
    now = {"value": 1000.0}
    cache = make_cache(chunks={chunk.chunk_id: chunk}, clock=lambda: now["value"])
    cache.put(
        query=QUESTION,
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        answer=make_answer(chunk=chunk),
    )
    assert cache.get(query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL).hit
    now["value"] += 604800  # the how-to TTL
    lookup = cache.get(query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL)
    assert lookup.status is CacheStatus.MISS
    assert "expired" in lookup.reason
    assert len(cache) == 0


def test_corpus_version_bump_invalidates_everything(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
) -> None:
    """# AT-CAC-05 — the conservative one-line rule: any bump clears the cache."""
    chunk = make_chunk()
    cache = make_cache(chunks={chunk.chunk_id: chunk})
    cache.put(
        query=QUESTION,
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        answer=make_answer(chunk=chunk),
    )
    removed = cache.invalidate(corpus_version="corpus-2")
    assert removed == 1
    assert len(cache) == 0
    assert cache.stats().hits == 0


def test_document_deletion_invalidates_only_affected_entries(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
) -> None:
    first = make_chunk(chunk_id="doc-a::0", doc_id="doc-a")
    second = make_chunk(chunk_id="doc-b::0", doc_id="doc-b", text="Лимит переводов 150 000 рублей.")
    cache = make_cache(chunks={first.chunk_id: first, second.chunk_id: second})
    cache.put(
        query=QUESTION,
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        answer=make_answer(chunk=first),
    )
    cache.put(
        query="Какой лимит переводов?",
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        answer=make_answer(chunk=second, claim_text="Лимит переводов 150 000 рублей."),
    )
    removed = cache.invalidate(document_ids={"doc-a"})
    assert removed == 1
    remaining = cache.entries()
    assert len(remaining) == 1
    assert remaining[0].answer.citations[0].doc_id == "doc-b"


def test_kill_switch_disables_everything(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
) -> None:
    """# AT-CAC-08 — with the cache disabled there are no hits and no writes."""
    chunk = make_chunk()
    cache = make_cache(chunks={chunk.chunk_id: chunk}, enabled=False)
    lookup = cache.get(query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL)
    assert lookup.status is CacheStatus.BYPASSED
    assert "kill switch" in lookup.reason
    from answerdesk.cache.errors import CacheConfigError

    with pytest.raises(CacheConfigError):
        cache.put(
            query=QUESTION,
            locale="ru",
            intent_bucket=IntentBucket.INFORMATIONAL,
            answer=make_answer(chunk=chunk),
        )
    assert cache.stats().bypassed == 1


def test_counters_track_hits_misses_bypass_and_wrong_hits(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
) -> None:
    """# AT-CAC-08 — the counters /metrics exposes."""
    chunk = make_chunk()
    cache = make_cache(chunks={chunk.chunk_id: chunk})
    cache.put(
        query=QUESTION,
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        answer=make_answer(chunk=chunk),
    )
    hit = cache.get(query=QUESTION, locale="ru", intent_bucket=IntentBucket.INFORMATIONAL)
    cache.get(query="совсем другой вопрос", locale="ru", intent_bucket=IntentBucket.INFORMATIONAL)
    cache.get(
        query="Сколько стоит обслуживание карты «Вектор»?",
        locale="ru",
        intent_bucket=IntentBucket.INFORMATIONAL,
        intent_label="legal",
    )
    assert hit.entry_id is not None
    cache.record_wrong_hit(hit.entry_id)
    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.bypassed == 1
    assert stats.wrong_hits == 1
    assert stats.hit_rate == pytest.approx(0.5)
    assert stats.by_intent["informational"] == 3


def test_cache_capacity_evicts_the_oldest(
    make_cache: Callable[..., SemanticCache],
    make_chunk: Callable[..., Chunk],
    make_answer: Callable[..., Answer],
) -> None:
    chunk = make_chunk()
    cache = make_cache(chunks={chunk.chunk_id: chunk})
    cache.max_entries = 2
    for index in range(4):
        cache.put(
            query=f"вопрос номер {index}",
            locale="ru",
            intent_bucket=IntentBucket.INFORMATIONAL,
            answer=make_answer(chunk=chunk, answer_id=f"answer-{index}"),
        )
    assert len(cache) == 2
