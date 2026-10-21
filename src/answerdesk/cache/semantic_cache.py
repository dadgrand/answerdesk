"""The semantic cache: exact + embedding lookup behind five mandatory guards.

Requirements §9.2 fixes the guards and their order; §9.4 fixes invalidation; §9.1 fixes what
an entry *is* (the whole answer artifact, never a bare string). This module implements that
contract and nothing more. Two design choices are worth stating:

* **a hit is re-validated on every read** (G-4). The structural check is string and set work
  (``revalidate.py``), so skipping it "for latency" would be false economy: a wrong hit is
  the worst failure the product can produce (§9.5).
* **the cache can only serve what it was given**. It never re-generates, never patches a
  citation, never merges two answers. If a guard fails, the result is a miss and the caller
  takes the normal (slower) path — which is also what makes the guard logic testable.

``negative`` entries (escalated/refused turns, §9.4) are stored separately by TTL class and
are never returned as prose: a negative hit answers "this is a known escalation", not "here
is an answer".

    # AT-CAC-01  # AT-CAC-02  # AT-CAC-03  # AT-CAC-04
    # AT-CAC-05  # AT-CAC-06  # AT-CAC-08  # AT-CAC-09
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from answerdesk.cache.errors import CacheConfigError
from answerdesk.cache.index import ExactCosineIndex, SimilarityIndex
from answerdesk.cache.keys import CacheKey, EntitySet, VersionTuple, cohort_of
from answerdesk.cache.revalidate import ChunkResolver, RevalidationResult, revalidate_answer
from answerdesk.cache.thresholds import DEFAULT_EPSILON, ThresholdTable
from answerdesk.contracts import (
    Answer,
    CacheStatus,
    Contract,
    Embedder,
    IntentBucket,
)

#: Default TTL classes from requirements Appendix C (§9.4).
DEFAULT_TTL_BY_CLASS: dict[str, int] = {"policy": 3600, "howto": 604800, "escalation": 600}

#: Default semantic-cache guard: these intent labels never use the semantic path (G-5).
DEFAULT_HIGH_RISK_INTENTS: frozenset[str] = frozenset({"account", "refund", "legal", "payment"})


class CacheEntry(Contract):
    """A stored answer artifact with everything needed to invalidate and audit it (§9.1)."""

    entry_id: str = Field(min_length=1)
    key: CacheKey
    answer: Answer
    retrieval_fingerprint: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(ge=0.0)
    ttl_seconds: int = Field(ge=1)
    ttl_class: str = "howto"
    negative: bool = False
    doc_hashes: dict[str, str] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)
    hits: int = Field(default=0, ge=0)
    wrong_hits: int = Field(default=0, ge=0)
    last_hit_at: float | None = None

    def is_expired(self, now: float) -> bool:
        return now - self.created_at >= self.ttl_seconds

    def doc_ids(self) -> set[str]:
        return {citation.doc_id for citation in self.answer.citations}


class CacheLookup(Contract):
    """The result of a cache read: a hit, a miss, a bypass, and *why*."""

    status: CacheStatus
    answer: Answer | None = None
    entry_id: str | None = None
    similarity: float | None = None
    cohort: str = ""
    guard: str = ""
    reason: str = ""

    @property
    def hit(self) -> bool:
        return self.status is CacheStatus.HIT


class CacheStats(Contract):
    """Counters the service exposes on ``/metrics`` (NFR-OBS-01, FR-CAC-08)."""

    entries: int = 0
    hits: int = 0
    misses: int = 0
    bypassed: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    negative_hits: int = 0
    wrong_hits: int = 0
    guard_rejections: int = 0
    invalidations: int = 0
    hit_rate: float = 0.0
    by_intent: dict[str, int] = Field(default_factory=dict)


def _entry_id(key: CacheKey, created_at: float) -> str:
    material = f"{key.exact_key()}:{created_at:.6f}"
    return "ce-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


class SemanticCache:
    """Exact + semantic answer cache with an error budget (requirements §9)."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        versions: VersionTuple,
        thresholds: ThresholdTable | None = None,
        enabled: bool = True,
        epsilon: float = DEFAULT_EPSILON,
        default_threshold: float = 0.93,
        ttl_by_class: Mapping[str, int] | None = None,
        high_risk_intents: frozenset[str] | None = None,
        high_risk_buckets: frozenset[IntentBucket] | None = None,
        chunk_lookup: ChunkResolver | None = None,
        index: SimilarityIndex | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_entries: int = 10_000,
        validate_on_hit: bool = True,
    ) -> None:
        if not 0.0 < epsilon < 1.0:
            raise CacheConfigError(f"epsilon must be in (0, 1), got {epsilon}")
        self.embedder = embedder
        self.versions = versions
        self.thresholds = thresholds
        self.enabled = enabled
        self.epsilon = epsilon
        self.default_threshold = default_threshold
        self.ttl_by_class = dict(ttl_by_class or DEFAULT_TTL_BY_CLASS)
        for name, ttl in self.ttl_by_class.items():
            if ttl <= 0:
                raise CacheConfigError(f"TTL for class {name!r} must be positive, got {ttl}")
        self.high_risk_intents = (
            DEFAULT_HIGH_RISK_INTENTS if high_risk_intents is None else high_risk_intents
        )
        self.high_risk_buckets = (
            frozenset({IntentBucket.ACCOUNT_LEGAL})
            if high_risk_buckets is None
            else high_risk_buckets
        )
        self.chunk_lookup = chunk_lookup
        self.index: SimilarityIndex = index or ExactCosineIndex()
        self.clock = clock
        self.max_entries = max_entries
        self.validate_on_hit = validate_on_hit

        self._by_key: dict[str, CacheEntry] = {}
        self._by_id: dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._bypassed = 0
        self._exact_hits = 0
        self._semantic_hits = 0
        self._negative_hits = 0
        self._wrong_hits = 0
        self._guard_rejections = 0
        self._invalidations = 0
        self._by_intent: dict[str, int] = {}

    # -- guards -------------------------------------------------------------------
    def is_high_risk(self, bucket: IntentBucket, intent_label: str | None) -> bool:
        """G-5: high-risk turns never use the semantic path (exact match only)."""
        if bucket in self.high_risk_buckets:
            return True
        return bool(intent_label and intent_label.strip().lower() in self.high_risk_intents)

    def _guard_versions(self, entry: CacheEntry) -> str | None:
        if not entry.key.versions.matches(self.versions):
            return "G-1 version mismatch (corpus/index/prompt/model/policy)"
        return None

    def _guard_entities(self, entry: CacheEntry, entities: EntitySet) -> str | None:
        if not entry.key.entities.matches(entities):
            return "G-2 entity mismatch (order id / amount / date / plan)"
        return None

    def _guard_source_docs(self, entry: CacheEntry) -> str | None:
        if self.chunk_lookup is None:
            return None
        for citation in entry.answer.citations:
            chunk = self.chunk_lookup(citation.chunk_id)
            if chunk is None:
                return f"G-3 chunk {citation.chunk_id!r} no longer exists"
            expected = entry.doc_hashes.get(citation.chunk_id)
            if expected is not None and chunk.doc_hash != expected:
                return f"G-3 chunk {citation.chunk_id!r} changed (doc_hash mismatch)"
        return None

    def _guard_structure(self, entry: CacheEntry) -> RevalidationResult:
        if not self.validate_on_hit:
            return RevalidationResult(ok=True)
        return revalidate_answer(
            entry.answer,
            chunk_lookup=self.chunk_lookup or (lambda _chunk_id: None),
            corpus_version=self.versions.corpus_version or None,
            prompt_version=self.versions.prompt_version or None,
        )

    def _ttl_for(self, *, ttl_class: str | None, negative: bool, answer: Answer) -> tuple[str, int]:
        if negative:
            chosen = "escalation"
        elif ttl_class is not None:
            chosen = ttl_class
        else:
            chosen = "policy" if answer.decision.action.value in {"escalate", "refuse"} else "howto"
        ttl = self.ttl_by_class.get(chosen)
        if ttl is None:
            raise CacheConfigError(f"unknown TTL class {chosen!r}")
        return chosen, ttl

    def _doc_hashes(self, answer: Answer) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for citation in answer.citations:
            if self.chunk_lookup is not None:
                chunk = self.chunk_lookup(citation.chunk_id)
                if chunk is not None:
                    hashes[citation.chunk_id] = chunk.doc_hash
                    continue
            hashes[citation.chunk_id] = citation.doc_hash
        return hashes

    # -- reads --------------------------------------------------------------------
    def get(
        self,
        *,
        query: str,
        locale: str,
        intent_bucket: IntentBucket,
        intent_label: str | None = None,
        embedding: list[float] | None = None,
        now: float | None = None,
    ) -> CacheLookup:
        """Look up an answer. Never raises for a normal guard failure — it misses instead."""
        resolved_now = self.clock() if now is None else now
        if not self.enabled:
            self._bypassed += 1
            return CacheLookup(status=CacheStatus.BYPASSED, reason="cache disabled (kill switch)")

        key = CacheKey.build(
            query=query, locale=locale, intent_bucket=intent_bucket, versions=self.versions
        )
        self._by_intent[intent_bucket.value] = self._by_intent.get(intent_bucket.value, 0) + 1

        entry = self._by_key.get(key.exact_key())
        if entry is not None:
            return self._serve(
                entry, similarity=1.0, cohort="exact", exact=True, now=resolved_now, key=key
            )

        if self.is_high_risk(intent_bucket, intent_label):
            self._bypassed += 1
            return CacheLookup(
                status=CacheStatus.BYPASSED,
                reason="G-5 high-risk intent: exact match only",
                cohort="exact",
            )

        if self.thresholds is None or not len(self.index):
            self._misses += 1
            return CacheLookup(status=CacheStatus.MISS, reason="no semantic entries")

        cohort = cohort_of(intent_bucket=intent_bucket, locale=locale, query=query)
        threshold = self.thresholds.threshold_for(cohort)
        vector = embedding if embedding is not None else self.embedder.embed_query(query)
        candidates = self.index.search(vector, k=1)
        if not candidates:
            self._misses += 1
            return CacheLookup(status=CacheStatus.MISS, cohort=cohort, reason="no candidates")
        entry_id, similarity = candidates[0]
        if similarity < threshold:
            self._misses += 1
            return CacheLookup(
                status=CacheStatus.MISS,
                similarity=similarity,
                cohort=cohort,
                reason=f"similarity {similarity:.4f} < threshold {threshold:.4f}",
            )
        candidate = self._by_id.get(entry_id)
        if candidate is None:
            self._misses += 1
            return CacheLookup(
                status=CacheStatus.MISS, similarity=similarity, cohort=cohort, reason="stale index"
            )
        return self._serve(
            candidate,
            similarity=similarity,
            cohort=cohort,
            exact=False,
            now=resolved_now,
            key=key,
        )

    def _serve(
        self,
        entry: CacheEntry,
        *,
        similarity: float,
        cohort: str,
        exact: bool,
        now: float,
        key: CacheKey,
    ) -> CacheLookup:
        """Run G-1…G-4 and either serve the entry or demote the read to a miss."""
        if entry.is_expired(now):
            self._evict(entry)
            self._misses += 1
            return CacheLookup(
                status=CacheStatus.MISS,
                cohort=cohort,
                reason=f"entry expired (ttl class {entry.ttl_class})",
            )
        for guard in (
            self._guard_versions(entry),
            self._guard_entities(entry, key.entities),
            self._guard_source_docs(entry),
        ):
            if guard is not None:
                self._guard_rejections += 1
                self._misses += 1
                return CacheLookup(
                    status=CacheStatus.MISS,
                    cohort=cohort,
                    similarity=similarity,
                    entry_id=entry.entry_id,
                    guard=guard.split(" ")[0],
                    reason=guard,
                )
        structural = self._guard_structure(entry)
        if not structural.ok:
            self._guard_rejections += 1
            self._misses += 1
            return CacheLookup(
                status=CacheStatus.MISS,
                cohort=cohort,
                similarity=similarity,
                entry_id=entry.entry_id,
                guard="G-4",
                reason=structural.reason(),
            )

        entry.hits += 1
        entry.last_hit_at = now
        self._hits += 1
        if entry.negative:
            self._negative_hits += 1
        if exact:
            self._exact_hits += 1
        else:
            self._semantic_hits += 1
        return CacheLookup(
            status=CacheStatus.HIT,
            answer=entry.answer,
            entry_id=entry.entry_id,
            similarity=similarity,
            cohort=cohort,
            reason="exact match" if exact else "semantic match within budget",
        )

    # -- writes -------------------------------------------------------------------
    def put(
        self,
        *,
        query: str,
        locale: str,
        intent_bucket: IntentBucket,
        answer: Answer,
        retrieval_fingerprint: Mapping[str, Any] | None = None,
        embedding: list[float] | None = None,
        ttl_class: str | None = None,
        negative: bool | None = None,
        now: float | None = None,
    ) -> CacheEntry:
        """Store the whole artifact. A negative entry stores an escalation, not an answer."""
        resolved_now = self.clock() if now is None else now
        if not self.enabled:
            raise CacheConfigError("cache is disabled; put() must not be called (kill switch)")
        key = CacheKey.build(
            query=query, locale=locale, intent_bucket=intent_bucket, versions=self.versions
        )
        is_negative = (
            bool(negative)
            if negative is not None
            else answer.decision.action.value
            in {
                "escalate",
                "refuse",
            }
        )
        chosen_class, ttl = self._ttl_for(ttl_class=ttl_class, negative=is_negative, answer=answer)
        entry = CacheEntry(
            entry_id=_entry_id(key, resolved_now),
            key=key,
            answer=answer,
            retrieval_fingerprint=dict(retrieval_fingerprint or {}),
            created_at=resolved_now,
            ttl_seconds=ttl,
            ttl_class=chosen_class,
            negative=is_negative,
            doc_hashes=self._doc_hashes(answer),
            embedding=embedding if embedding is not None else self.embedder.embed_query(query),
        )
        self._evict_key(key.exact_key())
        self._by_key[key.exact_key()] = entry
        self._by_id[entry.entry_id] = entry
        self.index.add(entry.entry_id, entry.embedding)
        self._enforce_capacity()
        return entry

    def _enforce_capacity(self) -> None:
        if len(self._by_id) <= self.max_entries:
            return
        oldest = sorted(self._by_id.values(), key=lambda entry: entry.created_at)
        for entry in oldest[: len(self._by_id) - self.max_entries]:
            self._evict(entry)

    def _evict(self, entry: CacheEntry) -> None:
        self._by_key.pop(entry.key.exact_key(), None)
        self._by_id.pop(entry.entry_id, None)
        self.index.remove(entry.entry_id)

    def _evict_key(self, exact_key: str) -> None:
        existing = self._by_key.get(exact_key)
        if existing is not None:
            self._evict(existing)

    # -- invalidation (§9.4) ------------------------------------------------------
    def invalidate(
        self, *, corpus_version: str | None = None, document_ids: set[str] | None = None
    ) -> int:
        """Version-keyed invalidation. A corpus bump clears everything (conservative)."""
        removed = 0
        if corpus_version is not None and corpus_version != self.versions.corpus_version:
            removed = len(self._by_id)
            self._by_key.clear()
            self._by_id.clear()
            self.index = ExactCosineIndex()
            self._invalidations += removed
            return removed
        if document_ids:
            for entry in list(self._by_id.values()):
                if entry.doc_ids() & set(document_ids):
                    self._evict(entry)
                    removed += 1
            self._invalidations += removed
        return removed

    def record_wrong_hit(self, entry_id: str) -> None:
        """Live wrong-hit signal (operator feedback) — feeds the K-1 monitor."""
        entry = self._by_id.get(entry_id)
        if entry is None:
            return
        entry.wrong_hits += 1
        self._wrong_hits += 1

    # -- observability ------------------------------------------------------------
    def stats(self) -> CacheStats:
        lookups = self._hits + self._misses
        return CacheStats(
            entries=len(self._by_id),
            hits=self._hits,
            misses=self._misses,
            bypassed=self._bypassed,
            exact_hits=self._exact_hits,
            semantic_hits=self._semantic_hits,
            negative_hits=self._negative_hits,
            wrong_hits=self._wrong_hits,
            guard_rejections=self._guard_rejections,
            invalidations=self._invalidations,
            hit_rate=(self._hits / lookups) if lookups else 0.0,
            by_intent=dict(self._by_intent),
        )

    def entries(self) -> list[CacheEntry]:
        return list(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)


@dataclass
class CacheProbe:
    """Test helper: a recorded lookup with the query that produced it."""

    query: str
    lookup: CacheLookup
    entities: EntitySet = field(default_factory=EntitySet)


__all__ = [
    "DEFAULT_HIGH_RISK_INTENTS",
    "DEFAULT_TTL_BY_CLASS",
    "CacheEntry",
    "CacheLookup",
    "CacheProbe",
    "CacheStats",
    "SemanticCache",
]
