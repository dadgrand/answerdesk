"""Fusion maths on synthetic ranks: RRF, convex combination, normalisers, degenerate cases.

Every number here is hand-computed in the docstring of the test that asserts it, because
FR-RET-04's pass condition is literally "equals the hand-computed formula".

    # AT-RET-04
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from answerdesk.contracts import Channel, FusionInfo
from answerdesk.retrieval.fusion import (
    DEFAULT_CC_ALPHA,
    ConvexFusion,
    FusionParams,
    Normalization,
    RRFFusion,
    SingleChannelFusion,
    is_degenerate,
    minmax_normalize,
    normalize,
    zscore_normalize,
)

BM25 = Channel.BM25
DENSE = Channel.DENSE


def _ids(passages) -> list[str]:
    return [passage.chunk.chunk_id for passage in passages]


def test_rrf_matches_the_hand_computed_formula(sized_passages) -> None:
    """RRF(d) = Σ 1/(k + rank); k=60, sets A=[c1,c2,c3] and B=[c3,c1].

    c1 = 1/61 + 1/62 = 0.03252247
    c3 = 1/63 + 1/61 = 0.03226646
    c2 = 1/62        = 0.01612903
    """
    sparse = sized_passages("c", [9.0, 8.0, 7.0], BM25)
    dense = sized_passages("c", [0.9, 0.8], DENSE)
    # Rename so both sets share chunk ids: dense finds c3 first, then c1.
    dense = [
        dense[0].model_copy(update={"chunk": sparse[2].chunk, "rank": 1}),
        dense[1].model_copy(update={"chunk": sparse[0].chunk, "rank": 2}),
    ]
    fused = RRFFusion(60).fuse({BM25: sparse, DENSE: dense}, k=10)
    scores = {passage.chunk.chunk_id: passage.score for passage in fused}
    assert scores["c0"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c2"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["c1"] == pytest.approx(1 / 62)
    assert _ids(fused) == ["c0", "c2", "c1"]
    assert [passage.rank for passage in fused] == [1, 2, 3]


def test_rrf_reports_its_parameters(sized_passages) -> None:
    outcome = RRFFusion(17).fuse_detailed({BM25: sized_passages("c", [3.0, 2.0])}, k=5)
    assert isinstance(outcome.info, FusionInfo)
    assert outcome.info.fuser == "rrf"
    assert outcome.info.params == {"rrf_k": 17}
    assert outcome.degenerate_channels == []


def test_rrf_rank_constant_changes_the_ordering(sized_passages) -> None:
    """FR-RET-04 asks for a *tunable* k, so k must actually matter.

    Sets: A = [x, a, y], B = [b, c, y].
    k=1 : x = 1/2, y = 1/4 + 1/4 = 1/2, b = 1/2  -> three-way tie, broken by chunk id.
    k=60: y = 2/63 = 0.0317 beats x = 1/61 = 0.0164, so y wins outright.
    """
    sparse = sized_passages("c", [5.0, 4.0, 3.0], BM25)
    sparse[0] = sparse[0].model_copy(
        update={"chunk": sparse[0].chunk.model_copy(update={"chunk_id": "x"})}
    )
    sparse[1] = sparse[1].model_copy(
        update={"chunk": sparse[1].chunk.model_copy(update={"chunk_id": "a"})}
    )
    sparse[2] = sparse[2].model_copy(
        update={"chunk": sparse[2].chunk.model_copy(update={"chunk_id": "y"})}
    )
    dense = [
        sparse[0].model_copy(
            update={
                "chunk": sparse[0].chunk.model_copy(update={"chunk_id": "b", "doc_id": "doc-b"}),
                "score": 0.9,
                "rank": 1,
            }
        ),
        sparse[1].model_copy(
            update={
                "chunk": sparse[1].chunk.model_copy(update={"chunk_id": "c", "doc_id": "doc-c"}),
                "score": 0.8,
                "rank": 2,
            }
        ),
        sparse[2].model_copy(update={"chunk": sparse[2].chunk, "score": 0.7, "rank": 3}),
    ]
    small_k = RRFFusion(1).fuse({BM25: sparse, DENSE: dense}, k=5)
    large_k = RRFFusion(60).fuse({BM25: sparse, DENSE: dense}, k=5)
    assert large_k[0].chunk.chunk_id == "y"
    assert small_k[0].chunk.chunk_id != "y"
    assert _ids(small_k) != _ids(large_k)


def test_rrf_breaks_ties_by_chunk_id_not_by_insertion_order(sized_passages) -> None:
    left = sized_passages("b", [1.0])
    right = sized_passages("a", [1.0])
    fused = RRFFusion().fuse({BM25: left, DENSE: right}, k=5)
    assert _ids(fused) == ["a0", "b0"]


def test_convex_combination_matches_the_hand_computed_formula(sized_passages) -> None:
    """CC(d) = Σ w_r·ŝ_r(d), α=0.75 over min–max normalised scores.

    BM25  [10, 5, 1]      -> [1.0, 0.4444, 0.0]     (w = 0.25)
    DENSE [0.2, 0.9] for c1,c0 -> [0.0, 1.0]        (w = 0.75)
    c0 = 0.25·1.0 + 0.75·1.0 = 1.0
    c1 = 0.25·0.4444 + 0.75·0.0 = 0.111111
    c2 = 0.25·0.0 = 0.0
    """
    sparse = sized_passages("c", [10.0, 5.0, 1.0], BM25)
    dense = [
        sized_passages("x", [0.2], DENSE)[0].model_copy(
            update={"chunk": sparse[1].chunk, "rank": 1}
        ),
        sized_passages("x", [0.9], DENSE)[0].model_copy(
            update={"chunk": sparse[0].chunk, "rank": 2}
        ),
    ]
    fused = ConvexFusion(alpha=DEFAULT_CC_ALPHA).fuse({BM25: sparse, DENSE: dense}, k=10)
    scores = {passage.chunk.chunk_id: passage.score for passage in fused}
    assert scores["c0"] == pytest.approx(1.0)
    assert scores["c1"] == pytest.approx(0.25 * (4 / 9))
    assert scores["c2"] == pytest.approx(0.0)
    assert _ids(fused) == ["c0", "c1", "c2"]


def test_convex_combination_records_normalised_weights(sized_passages) -> None:
    outcome = ConvexFusion(alpha=0.6).fuse_detailed(
        {BM25: sized_passages("c", [3.0, 2.0]), DENSE: sized_passages("c", [0.2, 0.1], DENSE)},
        k=5,
    )
    assert outcome.info is not None
    assert outcome.info.fuser == "cc"
    assert outcome.info.params["alpha"] == pytest.approx(0.6)
    assert outcome.info.params["normalization"] == "minmax"
    assert outcome.info.params["w_bm25"] == pytest.approx(0.4)
    assert outcome.info.params["w_dense"] == pytest.approx(0.6)


def test_zscore_normalisation_centres_and_scales() -> None:
    """z = (s − μ)/σ for [1, 2, 3]: μ=2, σ=0.8165 -> [-1.2247, 0.0, 1.2247]."""
    values = zscore_normalize([1.0, 2.0, 3.0])
    assert values.tolist() == pytest.approx([-1.224745, 0.0, 1.224745], abs=1e-6)
    assert normalize([1.0, 2.0, 3.0], Normalization.ZSCORE) is not None


def test_minmax_is_bounded_and_flat_input_maps_to_zero() -> None:
    assert minmax_normalize([4.0, 2.0, 0.0]).tolist() == pytest.approx([1.0, 0.5, 0.0])
    assert minmax_normalize([7.0, 7.0, 7.0]).tolist() == [0.0, 0.0, 0.0]
    assert minmax_normalize([7.0]).tolist() == [1.0]
    assert minmax_normalize([]).size == 0
    assert zscore_normalize([7.0, 7.0]).tolist() == [0.0, 0.0]
    assert zscore_normalize([7.0]).tolist() == [1.0]
    assert is_degenerate([3.0, 3.0]) is True
    assert is_degenerate([3.0, 2.0]) is False
    assert is_degenerate([3.0]) is False


def test_flat_dense_scores_do_not_let_the_dense_channel_dominate(sized_passages) -> None:
    """FR-RET-04's degenerate fixture: flat cosines must contribute nothing, not everything."""
    sparse = sized_passages("c", [10.0, 6.0, 2.0], BM25)
    flat = sized_passages("c", [0.5, 0.5, 0.5], DENSE)
    outcome = ConvexFusion(alpha=0.75).fuse_detailed({BM25: sparse, DENSE: flat}, k=5)
    assert outcome.degenerate_channels == ["dense"]
    assert _ids(outcome.passages) == ["c0", "c1", "c2"]
    for passage in outcome.passages:
        assert passage.contributions["dense"] == pytest.approx(0.0)
    # The sparse channel still decides the order: cosine noise cannot reorder it.
    # BM25 [10, 6, 2] -> min–max [1.0, 0.5, 0.0], weight 0.25 -> [0.25, 0.125, 0.0].
    assert [p.score for p in outcome.passages] == pytest.approx([0.25, 0.125, 0.0], abs=1e-5)


def test_single_candidate_channel_keeps_its_whole_weight(sized_passages) -> None:
    """One candidate is that channel's entire evidence, so it normalises to 1.0, not 0.0."""
    sparse = sized_passages("c", [5.0], BM25)
    dense = sized_passages("c", [0.9, 0.1], DENSE)
    dense[0] = dense[0].model_copy(update={"chunk": sparse[0].chunk})
    fused = ConvexFusion(alpha=0.5).fuse({BM25: sparse, DENSE: dense}, k=5)
    scores = {passage.chunk.chunk_id: passage.score for passage in fused}
    assert scores["c0"] == pytest.approx(0.5 * 1.0 + 0.5 * 1.0)


def test_explicit_weights_override_alpha(sized_passages) -> None:
    fusion = ConvexFusion(alpha=0.9, weights={"bm25": 3.0, "dense": 1.0})
    weights = fusion.weights_for([BM25, DENSE])
    assert weights[BM25] == pytest.approx(0.75)
    assert weights[DENSE] == pytest.approx(0.25)


def test_weights_are_normalised_even_without_a_dense_channel(sized_passages) -> None:
    fusion = ConvexFusion(alpha=0.75)
    assert fusion.weights_for([BM25]) == {BM25: 1.0}


def test_single_channel_fusion_is_a_passthrough(sized_passages) -> None:
    sparse = sized_passages("c", [3.0, 2.0, 1.0], BM25)
    outcome = SingleChannelFusion().fuse_detailed({BM25: sparse}, k=2)
    assert _ids(outcome.passages) == ["c0", "c1"]
    assert outcome.passages[0].stage == "fusion"
    assert outcome.info is not None and outcome.info.fuser == "none"
    assert SingleChannelFusion().fuse({}, 5) == []


def test_empty_result_sets_are_empty_not_errors() -> None:
    assert RRFFusion().fuse({}, 5) == []
    assert ConvexFusion(alpha=0.5).fuse({}, 5) == []


def test_fused_passages_mark_the_dominant_channel(sized_passages) -> None:
    sparse = sized_passages("c", [10.0, 1.0], BM25)
    dense = sized_passages("c", [0.1, 0.9], DENSE)
    fused = ConvexFusion(alpha=0.9).fuse({BM25: sparse, DENSE: dense}, k=5)
    assert fused[0].chunk.chunk_id == "c1"
    assert fused[0].channel is DENSE


def test_fusion_params_validate_and_build() -> None:
    with pytest.raises(ValidationError, match="alpha or explicit"):
        FusionParams(fuser="cc")
    params = FusionParams(fuser="cc", alpha=0.6)
    assert isinstance(params.build(), ConvexFusion)
    assert isinstance(FusionParams(fuser="rrf", rrf_k=17).build(), RRFFusion)
    assert FusionParams(fuser="rrf", rrf_k=17).build().rank_constant == 17  # type: ignore[union-attr]
    assert isinstance(FusionParams(fuser="none").build(), SingleChannelFusion)
    with pytest.raises(ValidationError):
        FusionParams(rrf_k=0)
    with pytest.raises(ValueError, match="alpha"):
        ConvexFusion(alpha=1.0)
    with pytest.raises(ValueError, match="rank_constant"):
        RRFFusion(0)


def test_fusion_output_length_is_capped_and_ranked(sized_passages) -> None:
    sparse = sized_passages("c", [10.0, 9.0, 8.0, 7.0], BM25)
    fused = RRFFusion().fuse({BM25: sparse}, k=2)
    assert len(fused) == 2
    assert [passage.rank for passage in fused] == [1, 2]
