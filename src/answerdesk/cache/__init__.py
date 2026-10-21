"""Semantic cache with guards and an error budget.

Owning task: **t8**. Public surface for the service (t9) and the console (t10):

=================================  =====================================================
:class:`SemanticCache`             exact + embedding lookup behind guards G-1…G-5
:class:`CacheLookup` / CacheStats   hit/miss/bypass with the reason, and the counters
:class:`CacheKey` / ``VersionTuple``  the §9.1 key: normalised query, locale, entities, versions
:class:`EntitySet`                  the G-2 entity guard (order id / amount / date / plan)
:class:`ThresholdTable`             per-cohort similarity thresholds calibrated to ε
:class:`CacheEvaluation`            K-1/K-2 numbers plus a precision/recall report
``revalidate_answer``               G-4 structural re-check (citations, quotes, INV-04)
``generate_replay_set``             the adversarial near-miss replay set
=================================  =====================================================

Requirements: FR-CAC-01 … FR-CAC-09, NFR-LAT-04; contract in ``docs/cache.md``.

Every type that crosses this module's boundary comes from :mod:`answerdesk.contracts`.
"""

from __future__ import annotations

from answerdesk.cache.errors import (
    CacheArtifactError,
    CacheConfigError,
    CacheError,
    CacheMissError,
)
from answerdesk.cache.index import ExactCosineIndex, SimilarityIndex
from answerdesk.cache.keys import (
    CacheKey,
    EntitySet,
    VersionTuple,
    cohort_of,
    extract_entities,
    locale_family,
    normalise_query,
)
from answerdesk.cache.replay import (
    EmbeddedPair,
    NearMissPair,
    ReplaySet,
    embed_replay_set,
    generate_replay_set,
)
from answerdesk.cache.report import (
    HIT_RATE_FLOOR,
    CacheEvaluation,
    CohortEvaluation,
    evaluate_cache,
    hit_rate_error_curve,
    render_markdown,
    write_report,
)
from answerdesk.cache.revalidate import (
    RevalidationResult,
    chunk_map,
    revalidate_answer,
)
from answerdesk.cache.semantic_cache import (
    DEFAULT_HIGH_RISK_INTENTS,
    DEFAULT_TTL_BY_CLASS,
    CacheEntry,
    CacheLookup,
    CacheStats,
    SemanticCache,
)
from answerdesk.cache.thresholds import (
    DEFAULT_EPSILON,
    UCB_MULTIPLIER,
    CohortThreshold,
    ThresholdTable,
    build_threshold_table,
    calibrate_threshold,
)

__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_HIGH_RISK_INTENTS",
    "DEFAULT_TTL_BY_CLASS",
    "HIT_RATE_FLOOR",
    "UCB_MULTIPLIER",
    "CacheArtifactError",
    "CacheConfigError",
    "CacheEntry",
    "CacheError",
    "CacheEvaluation",
    "CacheKey",
    "CacheLookup",
    "CacheMissError",
    "CacheStats",
    "CohortEvaluation",
    "CohortThreshold",
    "EmbeddedPair",
    "EntitySet",
    "ExactCosineIndex",
    "NearMissPair",
    "ReplaySet",
    "RevalidationResult",
    "SemanticCache",
    "SimilarityIndex",
    "ThresholdTable",
    "VersionTuple",
    "build_threshold_table",
    "calibrate_threshold",
    "chunk_map",
    "cohort_of",
    "embed_replay_set",
    "evaluate_cache",
    "extract_entities",
    "generate_replay_set",
    "hit_rate_error_curve",
    "locale_family",
    "normalise_query",
    "render_markdown",
    "revalidate_answer",
    "write_report",
]
