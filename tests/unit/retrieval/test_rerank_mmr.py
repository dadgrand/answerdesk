"""Reranking and MMR: reference computations, determinism, lazy loading, contract shape.

# AT-RET-07
"""

from __future__ import annotations

import importlib.util

import pytest

from answerdesk.contracts import Channel, RetrievedPassage
from answerdesk.rerank.cross_encoder import (
    LazyCrossEncoderReranker,
    LexicalOverlapReranker,
    build_reranker,
)
from answerdesk.rerank.mmr import MMRDiversifier, minmax_relevance, mmr_select
from answerdesk.retrieval.fusion import minmax_normalize


def _passage(chunk_id: str, score: float, text: str, rank: int = 1) -> RetrievedPassage:
    from answerdesk.contracts import Chunk

    return RetrievedPassage(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=f"doc-{chunk_id}",
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=len(text.split()),
        ),
        channel=Channel.BM25,
        score=score,
        rank=rank,
    )


def test_mmr_matches_a_hand_computed_reference() -> None:
    """λ=0.5, relevance min–max of [1.0, 0.5, 0.2] = [1.0, 0.375, 0.0]; sim(p1,p2)=1.

    Step 1: no redundancy  -> 0.5·rel -> p1 (0.500) wins.
    Step 2: p2 = 0.5·0.375 − 0.5·1.0 = −0.3125 ; p3 = 0.0  -> p3 wins.
    Step 3: p2 last.
    """
    passages = [
        _passage("p1", 1.0, "a"),
        _passage("p2", 0.5, "b"),
        _passage("p3", 0.2, "c"),
    ]
    similarities = {("p1", "p2"): 1.0, ("p1", "p3"): 0.0, ("p2", "p3"): 0.0}

    def similarity(left_id: str, right_id: str, _left: str, _right: str) -> float:
        return similarities.get((left_id, right_id), similarities.get((right_id, left_id), 0.0))

    selected = mmr_select(passages, k=3, lambda_=0.5, similarity=similarity)
    assert [passage.chunk.chunk_id for passage in selected] == ["p1", "p3", "p2"]
    assert [passage.rank for passage in selected] == [1, 2, 3]
    assert all(passage.stage == "pack" for passage in selected)
    assert selected[0].contributions["mmr_relevance"] == pytest.approx(1.0)


def test_mmr_with_lambda_one_is_pure_relevance() -> None:
    passages = [_passage("p1", 1.0, "a"), _passage("p2", 0.5, "b"), _passage("p3", 0.2, "c")]
    similarity = lambda *_args: 1.0  # noqa: E731 - every pair is a duplicate
    selected = mmr_select(passages, k=3, lambda_=1.0, similarity=similarity)
    assert [passage.chunk.chunk_id for passage in selected] == ["p1", "p2", "p3"]


def test_mmr_with_low_lambda_prefers_diversity() -> None:
    passages = [_passage("p1", 1.0, "a"), _passage("p2", 0.9, "b"), _passage("p3", 0.1, "c")]
    duplicated = {("p1", "p2"): 1.0}

    def similarity(left_id: str, right_id: str, _left: str, _right: str) -> float:
        return duplicated.get((left_id, right_id), duplicated.get((right_id, left_id), 0.0))

    selected = mmr_select(passages, k=2, lambda_=0.1, similarity=similarity)
    assert [passage.chunk.chunk_id for passage in selected] == ["p1", "p3"]


def test_mmr_default_backend_uses_chunk_text_similarity() -> None:
    passages = [
        _passage("p1", 1.0, "refund policy fourteen days"),
        _passage("p2", 0.9, "refund policy fourteen days"),
        _passage("p3", 0.2, "card replacement fee"),
    ]
    selected = mmr_select(passages, k=2, lambda_=0.3)
    assert [passage.chunk.chunk_id for passage in selected] == ["p1", "p3"]


def test_mmr_bounds_and_degenerate_inputs() -> None:
    assert mmr_select([], k=3) == []
    assert mmr_select([_passage("p1", 1.0, "a")], k=0) == []
    with pytest.raises(ValueError, match="lambda"):
        mmr_select([_passage("p1", 1.0, "a")], k=1, lambda_=1.5)


def test_mmr_diversifier_wraps_the_selection() -> None:
    passages = [_passage("p1", 1.0, "a"), _passage("p2", 0.5, "b")]
    diversifier = MMRDiversifier(lambda_=0.5)
    assert diversifier.name == "mmr"
    assert [p.chunk.chunk_id for p in diversifier.diversify(passages, 1)] == ["p1"]


def test_mmr_duplicated_minmax_matches_fusion_normalisation() -> None:
    """``rerank.mmr`` must not import ``retrieval.fusion`` (that import was a real cycle),
    so the formula is duplicated — and this test keeps the copies identical."""
    for fixture in ([1.0, 2.0, 3.0], [5.0], [7.0, 7.0], [], [-1.0, 0.0, 1.0]):
        assert minmax_relevance(fixture).tolist() == pytest.approx(
            minmax_normalize(fixture).tolist()
        )


def test_lexical_reranker_is_deterministic_and_respects_the_budget() -> None:
    reranker = LexicalOverlapReranker()
    passages = [
        _passage("p1", 1.0, "refund policy fourteen days"),
        _passage("p2", 0.9, "error code TS-999 blocks the card"),
        _passage("p3", 0.8, "card replacement fee 300 RUB"),
    ]
    first = reranker.rerank("error code TS-999", passages, 2)
    second = reranker.rerank("error code TS-999", passages, 2)
    assert [p.chunk.chunk_id for p in first] == [p.chunk.chunk_id for p in second]
    assert len(first) == 2
    assert first[0].chunk.chunk_id == "p2"
    assert first[0].stage == "rerank"
    assert first[0].contributions["fused_score"] == pytest.approx(0.9)
    assert first[0].contributions["rerank"] >= 0.0
    assert [p.rank for p in first] == [1, 2]


def test_lexical_reranker_breaks_ties_by_chunk_id() -> None:
    reranker = LexicalOverlapReranker()
    passages = [_passage("b", 1.0, "no overlap here"), _passage("a", 1.0, "no overlap here")]
    ranked = reranker.rerank("something else entirely", passages, 2)
    assert [p.chunk.chunk_id for p in ranked] == ["a", "b"]


def test_cross_encoder_is_not_imported_by_the_base_install() -> None:
    """FR-RET-07 / NFR-PORT-03: torch and sentence-transformers stay behind the extra."""
    reranker = LazyCrossEncoderReranker()
    assert reranker._model is None
    assert reranker.name.startswith("cross-encoder:")
    if importlib.util.find_spec("sentence_transformers") is not None:  # pragma: no cover
        pytest.skip("sentence-transformers is installed; the missing-extra path cannot run")
    with pytest.raises(Exception, match="local"):
        reranker.rerank("query", [_passage("p1", 1.0, "text")], 1)


def test_build_reranker_returns_the_stand_in_offline(settings) -> None:
    assert settings.retrieval.rerank.enabled is False
    assert isinstance(build_reranker(settings), LexicalOverlapReranker)
    assert build_reranker(settings, offline_stand_in=False) is None


def test_build_reranker_returns_the_cross_encoder_when_enabled(settings) -> None:
    enabled = settings.model_copy(deep=True)
    enabled.retrieval.rerank.enabled = True
    reranker = build_reranker(enabled)
    assert isinstance(reranker, LazyCrossEncoderReranker)
    assert reranker.model_name == settings.retrieval.rerank.model
