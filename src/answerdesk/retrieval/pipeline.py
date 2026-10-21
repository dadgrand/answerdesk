"""The retrieval pipeline: retrieve → fuse → dedupe → rerank → diversify → pack.

The stages are fixed, named and individually timed, because everything the operator console
and the ablations need is a function of *which* stage changed a ranking (FR-RET-08)::

    query ──▶ analysis ──▶ sparse probes ─┐
                    │                     ├─▶ fusion ─▶ dedupe ─▶ [rerank] ─▶ [mmr] ─▶ pack
                    └──────▶ dense ───────┘

Design notes that the tests pin down:

* **Two sparse probes.** The expanded query and the identifier must-match query are both
  issued against the sparse index and merged by best rank (:func:`answerdesk.index.
  bm25_index.merge_probes`). Without the second probe, ``ERR-TRF-4012`` is at the mercy of
  the dense channel, which is exactly the documented failure mode (D-07, RT-8).
* **Every stage is optional and every stage is observable.** No reranker and no MMR is a
  valid configuration (the default); the trace then reports empty stages rather than
  silently claiming a stage ran.
* **Timings are returned, not stored in the trace.** ``RetrievalTrace`` is a frozen
  contract with no timing field; callers that need latency get it from
  :meth:`RetrievalPipeline.retrieve_with_timings` (NFR-LAT-02, NFR-COST-03).

The pipeline itself implements :class:`answerdesk.contracts.Retriever`, so a service or a
cache can depend on "a retriever" and be handed a whole hybrid pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from answerdesk.config import Settings
from answerdesk.contracts import (
    Channel,
    Chunk,
    Reranker,
    RetrievalTrace,
    RetrievedPassage,
)
from answerdesk.index.bm25_index import BM25Index, merge_probes
from answerdesk.index.store import IndexBundle
from answerdesk.index.vector_index import ExactVectorIndex
from answerdesk.rerank.mmr import MMRDiversifier, vector_similarity
from answerdesk.retrieval.dedupe import DEFAULT_DEDUPE_COSINE, dedupe_passages
from answerdesk.retrieval.fusion import FusionParams, RRFFusion, SingleChannelFusion
from answerdesk.retrieval.query_analysis import QueryAnalysis, QueryAnalyzer


class PipelineConfig(BaseModel):
    """Stage budgets of the pipeline; mirrors the ``retrieval`` config section."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    top_k: int = Field(default=50, gt=0)
    gate_k: int = Field(default=20, gt=0)
    final_k: int = Field(default=12, gt=0)
    dedupe_cosine: float = Field(default=DEFAULT_DEDUPE_COSINE, gt=0.0, le=1.0)
    rerank_top_in: int = Field(default=50, gt=0)
    rerank_top_out: int = Field(default=12, gt=0)
    mmr_lambda: float | None = Field(default=None, gt=0.0, le=1.0)
    identifier_probe: bool = True
    expand_multi_query: bool = False
    #: Channels to run. Single-channel configurations exist for the ablation arms and for
    #: deployments that deliberately run one retriever; a dense-only pipeline still needs a
    #: BM25 index object for its vocabulary, but never queries it.
    channels: tuple[Literal["bm25", "dense"], ...] = ("bm25", "dense")

    @field_validator("channels")
    @classmethod
    def _channels_non_empty(
        cls, value: tuple[Literal["bm25", "dense"], ...]
    ) -> tuple[Literal["bm25", "dense"], ...]:
        if not value:
            raise ValueError("a pipeline needs at least one channel")
        unknown = sorted(set(value) - {"bm25", "dense"})
        if unknown:
            raise ValueError(f"unknown channels: {unknown}")
        return value


class RetrievalPipeline:
    """Hybrid retriever with a full trace (FR-RET-08) and a ``Retriever`` shape."""

    name = "hybrid"

    def __init__(
        self,
        *,
        bm25: BM25Index,
        dense: ExactVectorIndex | None = None,
        fuser: Any = None,
        analyzer: QueryAnalyzer | None = None,
        reranker: Reranker | None = None,
        diversifier: MMRDiversifier | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.config = config or PipelineConfig()
        if fuser is None:
            fuser = RRFFusion() if dense is not None else SingleChannelFusion()
        self.fuser = fuser
        self.analyzer = analyzer or QueryAnalyzer(bm25.vocabulary)
        self.reranker = reranker
        self.diversifier = diversifier

    # -- construction -------------------------------------------------------------
    @classmethod
    def from_index(
        cls,
        index: IndexBundle,
        *,
        params: FusionParams | None = None,
        reranker: Reranker | None = None,
        mmr_lambda: float | None = None,
        config: PipelineConfig | None = None,
    ) -> RetrievalPipeline:
        """Build from a loaded :class:`~answerdesk.index.store.IndexBundle`."""
        dense: ExactVectorIndex | None = index.dense if index.dense.dimension else None
        fuser = params.build() if params is not None else None
        diversifier = (
            MMRDiversifier(lambda_=mmr_lambda, similarity=vector_similarity(index.dense.vector_for))
            if mmr_lambda is not None
            else None
        )
        return cls(
            bm25=index.bm25,
            dense=dense,
            fuser=fuser,
            reranker=reranker,
            diversifier=diversifier,
            config=config,
        )

    # -- retrieval ----------------------------------------------------------------
    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        """``Retriever`` protocol: the packed top-k for ``query``."""
        trace = self.retrieve(query, final_k=k)
        return trace.packed

    def retrieve(
        self,
        query: str,
        *,
        final_k: int | None = None,
    ) -> RetrievalTrace:
        """Run the pipeline and return the trace (FR-RET-08)."""
        trace, _timings = self.retrieve_with_timings(query, final_k=final_k)
        return trace

    def retrieve_with_timings(
        self,
        query: str,
        *,
        final_k: int | None = None,
    ) -> tuple[RetrievalTrace, dict[str, float]]:
        """Run the pipeline, returning the trace and per-stage wall-clock milliseconds."""
        config = self.config
        limit = final_k if final_k is not None else config.final_k
        timings: dict[str, float] = {}
        started = time.perf_counter()

        analysis = self.analyzer.analyze(query)
        timings["analysis_ms"] = _ms(started)

        started = time.perf_counter()
        sparse = self._sparse_candidates(analysis, config.top_k)
        timings["sparse_ms"] = _ms(started)

        started = time.perf_counter()
        dense = self._dense_candidates(analysis, config.top_k)
        timings["dense_ms"] = _ms(started)

        started = time.perf_counter()
        result_sets: dict[Channel, Sequence[RetrievedPassage]] = {Channel.BM25: sparse}
        if dense:
            result_sets[Channel.DENSE] = dense
        outcome = self.fuser.fuse_detailed(result_sets, config.top_k)
        fused = outcome.passages
        timings["fusion_ms"] = _ms(started)

        started = time.perf_counter()
        deduped, dropped = dedupe_passages(
            fused,
            threshold=config.dedupe_cosine,
            vector_for=self.dense.vector_for if self.dense is not None else None,
        )
        timings["dedupe_ms"] = _ms(started)

        reranked: list[RetrievedPassage] = []
        if self.reranker is not None and deduped:
            started = time.perf_counter()
            window = deduped[: config.rerank_top_in]
            reranked = self.reranker.rerank(query, window, config.rerank_top_out)
            timings["rerank_ms"] = _ms(started)

        packed_source = reranked or deduped
        if self.diversifier is not None and packed_source:
            started = time.perf_counter()
            packed_source = self.diversifier.diversify(packed_source, limit)
            timings["mmr_ms"] = _ms(started)

        packed = _pack(packed_source, limit)
        timings["total_ms"] = round(sum(timings.values()), 3)

        trace = RetrievalTrace(
            query=query,
            normalized_query=analysis.normalized,
            language=analysis.language,
            candidates=[*sparse, *dense],
            fused=fused,
            reranked=reranked,
            packed=packed,
            fusion=outcome.info,
            dropped_duplicates=dropped,
            identifier_tokens=list(analysis.identifier_tokens),
        )
        return trace, timings

    # -- stages -------------------------------------------------------------------
    def _sparse_candidates(self, analysis: QueryAnalysis, top_k: int) -> list[RetrievedPassage]:
        """Expanded-query probe, plus the identifier must-match probe (merged)."""
        if "bm25" not in self.config.channels:
            return []
        primary = self.bm25.search_terms(analysis.lexical_terms, top_k)
        if not (self.config.identifier_probe and analysis.identifier_tokens):
            return primary
        must_match = self.bm25.search_terms(
            analysis.lexical_terms,
            top_k,
            required=[token.lower() for token in analysis.identifier_tokens],
        )
        if not must_match:
            return primary
        return merge_probes(primary, must_match)

    def _dense_candidates(self, analysis: QueryAnalysis, top_k: int) -> list[RetrievedPassage]:
        """Dense probe over the *unexpanded* query, plus optional extra phrasings."""
        if "dense" not in self.config.channels or self.dense is None or self.dense.size == 0:
            return []
        primary = self.dense.search(analysis.dense_text, top_k)
        if not analysis.multi_query:
            return primary
        merged = list(primary)
        seen = {passage.chunk.chunk_id for passage in merged}
        for variant in analysis.multi_query:
            for passage in self.dense.search(variant, max(1, top_k // 2)):
                if passage.chunk.chunk_id in seen:
                    continue
                seen.add(passage.chunk.chunk_id)
                merged.append(passage.model_copy(update={"rank": len(merged) + 1}))
        return merged


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _pack(passages: Sequence[RetrievedPassage], limit: int) -> list[RetrievedPassage]:
    """Final order: truncate to ``limit``, rank 1..n, stage ``pack``."""
    packed: list[RetrievedPassage] = []
    for rank, passage in enumerate(passages[: max(0, limit)], start=1):
        packed.append(
            passage.model_copy(
                update={
                    "rank": rank,
                    "stage": "pack",
                    "contributions": {
                        **passage.contributions,
                        "final_rank": float(rank),
                    },
                }
            )
        )
    return packed


def build_pipeline(
    settings: Settings,
    index: IndexBundle,
    *,
    params: FusionParams | None = None,
    reranker: Reranker | None = None,
    mmr_lambda: float | None = None,
) -> RetrievalPipeline:
    """Resolver from configuration to a ready pipeline (the service path, t9)."""
    config = PipelineConfig(
        top_k=settings.retrieval.top_k,
        gate_k=settings.retrieval.gate_k,
        final_k=settings.generate.context_k_max,
        dedupe_cosine=settings.retrieval.dedupe_cosine,
        rerank_top_in=settings.retrieval.rerank.top_in,
        rerank_top_out=settings.retrieval.rerank.top_out,
        mmr_lambda=mmr_lambda if mmr_lambda is not None else settings.retrieval.mmr_lambda,
    )
    resolved_params = params
    if resolved_params is None:
        resolved_params = FusionParams(
            fuser=settings.retrieval.fuser,
            rrf_k=settings.retrieval.rrf_k,
            alpha=settings.retrieval.cc_alpha if settings.retrieval.fuser == "cc" else None,
        )
        if settings.retrieval.fuser == "cc" and resolved_params.alpha is None:
            resolved_params = resolved_params.model_copy(update={"alpha": 0.75})
    return RetrievalPipeline.from_index(
        index, params=resolved_params, reranker=reranker, config=config
    )


def chunks_of(trace: RetrievalTrace) -> list[Chunk]:
    """Convenience: the packed chunks of a trace, in final order."""
    return [passage.chunk for passage in trace.packed]


def channels_of(trace: RetrievalTrace) -> dict[str, int]:
    """How many candidates each channel contributed — the console's channel breakdown."""
    counts: dict[str, int] = {}
    for passage in trace.candidates:
        counts[passage.channel.value] = counts.get(passage.channel.value, 0) + 1
    return counts


__all__ = [
    "PipelineConfig",
    "RetrievalPipeline",
    "build_pipeline",
    "channels_of",
    "chunks_of",
]
