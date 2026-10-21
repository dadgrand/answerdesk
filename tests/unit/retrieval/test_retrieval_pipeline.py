"""The retrieval pipeline: trace completeness, determinism, optional stages, protocol seam.

# AT-RET-08  # AT-RET-03  # AT-LAT-02
"""

from __future__ import annotations

import pytest

from answerdesk.contracts import Channel, RetrievalTrace, Retriever
from answerdesk.index.bm25_index import BM25Index
from answerdesk.index.store import ChunkStore
from answerdesk.index.vector_index import ExactVectorIndex
from answerdesk.rerank.cross_encoder import LexicalOverlapReranker
from answerdesk.rerank.mmr import MMRDiversifier
from answerdesk.retrieval.fusion import SingleChannelFusion
from answerdesk.retrieval.pipeline import (
    PipelineConfig,
    RetrievalPipeline,
    build_pipeline,
    channels_of,
    chunks_of,
)
from answerdesk.retrieval.query_analysis import QueryAnalyzer
from answerdesk.testing.doubles import HashingEmbedder


def test_pipeline_implements_the_retriever_protocol(pipeline: RetrievalPipeline) -> None:
    """FR-RET-03: a service may depend on "a retriever" and receive the whole pipeline."""
    assert isinstance(pipeline, Retriever)
    passages = pipeline.search("refund policy", 3)
    assert passages
    assert all(passage.rank == index for index, passage in enumerate(passages, start=1))
    assert passages[0].stage == "pack"


def test_trace_contains_every_stage(pipeline: RetrievalPipeline) -> None:
    """FR-RET-08 / §6.5: every stage contributes to the trace object."""
    trace = pipeline.retrieve("refund policy 14 days")
    assert isinstance(trace, RetrievalTrace)
    assert trace.query == "refund policy 14 days"
    assert trace.normalized_query
    assert trace.candidates, "per-channel candidates are missing from the trace"
    assert {passage.channel for passage in trace.candidates} <= {
        Channel.BM25,
        Channel.DENSE,
    }
    assert trace.fused, "the fused ranking is missing from the trace"
    assert trace.packed, "the packed ranking is missing from the trace"
    assert trace.fusion is not None
    assert trace.fusion.fuser in ("rrf", "cc", "none")
    assert all(passage.stage == "fusion" for passage in trace.fused)
    assert all(passage.stage == "pack" for passage in trace.packed)
    assert len(trace.packed) <= pipeline.config.final_k


def test_trace_reports_language_and_identifiers(pipeline: RetrievalPipeline) -> None:
    trace = pipeline.retrieve("Ошибка TS-999 и блокировка карты ORD-2026-10009")
    assert trace.language == "ru"
    assert "TS-999" in trace.identifier_tokens
    assert "ORD-2026-10009" in trace.identifier_tokens


def test_language_is_und_when_the_query_carries_no_evidence(pipeline: RetrievalPipeline) -> None:
    """The detector is allowed to be wrong, so it reports "und" rather than guessing."""
    trace = pipeline.retrieve("Ошибка TS-999 заблокировала карту")
    assert trace.language == "und"


def test_identifier_query_retrieves_the_exact_identifier_chunk(pipeline: RetrievalPipeline) -> None:
    """FR-RET-06 / RT-8: the must-match probe puts the exact-id chunk in the context."""
    trace = pipeline.retrieve("ORD-2026-10009")
    assert "kb-transfer" in {passage.chunk.doc_id for passage in trace.packed}
    blocked = pipeline.retrieve("TS-999")
    assert "kb-card" in {passage.chunk.doc_id for passage in blocked.packed}


def test_chinese_query_retrieves_the_chinese_chunk(pipeline: RetrievalPipeline) -> None:
    trace = pipeline.retrieve("退款订单号")
    assert "kb-zh" in {passage.chunk.doc_id for passage in trace.packed}


def test_two_runs_produce_identical_traces(pipeline: RetrievalPipeline) -> None:
    """AT-RET-01 in pipeline form: same corpus + query -> identical ranking and scores."""
    first = pipeline.retrieve("card replacement fee 300")
    second = pipeline.retrieve("card replacement fee 300")
    assert first.model_dump() == second.model_dump()


def test_timings_are_reported_per_stage(pipeline: RetrievalPipeline) -> None:
    """NFR-LAT-02: retrieval latency is measurable per stage, not as one opaque number."""
    _trace, timings = pipeline.retrieve_with_timings("refund policy")
    for key in ("analysis_ms", "sparse_ms", "dense_ms", "fusion_ms", "dedupe_ms", "total_ms"):
        assert key in timings
        assert timings[key] >= 0.0
    assert timings["total_ms"] >= timings["fusion_ms"]


def test_reranker_changes_the_stage_but_keeps_the_contract(
    bm25_index: BM25Index, dense_index: ExactVectorIndex
) -> None:
    pipeline = RetrievalPipeline(
        bm25=bm25_index,
        dense=dense_index,
        reranker=LexicalOverlapReranker(),
        config=PipelineConfig(top_k=10, gate_k=5, final_k=3, rerank_top_in=10, rerank_top_out=3),
    )
    trace = pipeline.retrieve("refund 14 days")
    assert trace.reranked
    assert all(passage.stage == "rerank" for passage in trace.reranked)
    assert trace.reranked[0].contributions["rerank"] >= 0.0
    assert trace.packed and all(passage.stage == "pack" for passage in trace.packed)


def test_mmr_diversifier_produces_a_pack_stage(
    bm25_index: BM25Index, dense_index: ExactVectorIndex
) -> None:
    pipeline = RetrievalPipeline(
        bm25=bm25_index,
        dense=dense_index,
        diversifier=MMRDiversifier(lambda_=0.7),
        config=PipelineConfig(top_k=10, gate_k=5, final_k=2, mmr_lambda=0.7),
    )
    trace = pipeline.retrieve("refund card policy")
    assert len(trace.packed) <= 2
    assert all(passage.stage == "pack" for passage in trace.packed)
    assert "mmr_relevance" in trace.packed[0].contributions


def test_duplicate_chunks_are_dropped_and_counted(bm25_index: BM25Index) -> None:
    """FR-RET-08: ``dropped_duplicates`` is reported, not silently swallowed."""
    from answerdesk.contracts import RetrievedPassage

    duplicated = bm25_index.chunks[0]
    passages = [
        RetrievedPassage(chunk=duplicated, channel=Channel.BM25, score=5.0, rank=1),
        RetrievedPassage(
            chunk=duplicated.model_copy(update={"chunk_id": "copy::0"}),
            channel=Channel.BM25,
            score=4.0,
            rank=2,
        ),
        RetrievedPassage(chunk=bm25_index.chunks[1], channel=Channel.BM25, score=3.0, rank=3),
    ]
    from answerdesk.retrieval.dedupe import dedupe_passages

    kept, dropped = dedupe_passages(passages, threshold=0.99)
    assert dropped == 1
    assert [passage.chunk.chunk_id for passage in kept] == [
        duplicated.chunk_id,
        bm25_index.chunks[1].chunk_id,
    ]
    assert [passage.rank for passage in kept] == [1, 2]


def test_single_channel_pipeline_reports_no_fusion(bm25_index: BM25Index) -> None:
    pipeline = RetrievalPipeline(
        bm25=bm25_index, dense=None, config=PipelineConfig(top_k=5, gate_k=3, final_k=2)
    )
    trace = pipeline.retrieve("refund policy")
    assert trace.fusion is not None
    assert trace.fusion.fuser == "none"
    assert {passage.channel for passage in trace.candidates} == {Channel.BM25}


def test_empty_query_yields_an_empty_trace(pipeline: RetrievalPipeline) -> None:
    trace = pipeline.retrieve("   ")
    assert trace.packed == []
    assert trace.normalized_query == ""


def test_build_pipeline_reads_the_settings(
    bm25_index: BM25Index, dense_index: ExactVectorIndex, settings
) -> None:
    store = ChunkStore.from_chunks(bm25_index.chunks, corpus_version="corpus-mini")
    from answerdesk.index.store import IndexBundle

    bundle = IndexBundle.build(store, HashingEmbedder(dimension=48))
    built = build_pipeline(settings, bundle)
    assert built.config.top_k == settings.retrieval.top_k
    assert built.config.final_k == settings.generate.context_k_max
    assert built.fuser.name in ("rrf", "cc", "none")
    trace = built.retrieve("тариф")
    assert trace.packed


def test_trace_helpers_expose_channels_and_chunks(pipeline: RetrievalPipeline) -> None:
    trace = pipeline.retrieve("card fee")
    assert set(channels_of(trace)) <= {"bm25", "dense"}
    assert len(chunks_of(trace)) == len(trace.packed)


def test_analyzer_is_bound_to_the_index_vocabulary(pipeline: RetrievalPipeline) -> None:
    assert isinstance(pipeline.analyzer, QueryAnalyzer)
    assert pipeline.analyzer.vocabulary is pipeline.bm25.vocabulary


def test_retrieve_accepts_an_override_of_the_final_size(pipeline: RetrievalPipeline) -> None:
    trace = pipeline.retrieve("refund", final_k=1)
    assert len(trace.packed) == 1


def test_pipeline_config_rejects_zero_budgets() -> None:
    with pytest.raises(ValueError):
        PipelineConfig(top_k=0)


def test_single_channel_configuration_runs_only_that_channel(
    bm25_index: BM25Index, dense_index: ExactVectorIndex
) -> None:
    """The ablation arms need a real one-channel pipeline, not a two-channel one with holes."""
    dense_only = RetrievalPipeline(
        bm25=bm25_index,
        dense=dense_index,
        fuser=SingleChannelFusion(),
        config=PipelineConfig(top_k=5, gate_k=3, final_k=2, channels=("dense",)),
    )
    trace = dense_only.retrieve("card fee")
    assert trace.candidates
    assert {passage.channel for passage in trace.candidates} == {Channel.DENSE}

    sparse_only = RetrievalPipeline(
        bm25=bm25_index,
        dense=dense_index,
        fuser=SingleChannelFusion(),
        config=PipelineConfig(top_k=5, gate_k=3, final_k=2, channels=("bm25",)),
    )
    sparse_trace = sparse_only.retrieve("card fee")
    assert {passage.channel for passage in sparse_trace.candidates} == {Channel.BM25}


def test_pipeline_config_rejects_an_empty_or_unknown_channel_set() -> None:
    with pytest.raises(ValueError, match="at least one channel"):
        PipelineConfig(channels=())
    # The Literal type rejects an unknown channel before the validator even runs, which is
    # the stronger guarantee; either error is acceptable, silence is not.
    with pytest.raises(ValueError, match="colbert"):
        PipelineConfig(channels=("bm25", "colbert"))
