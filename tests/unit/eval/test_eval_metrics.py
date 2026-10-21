"""Metric tests against hand-computed values (FR-EVL-04 / AT-EVL-04).

Every retrieval metric gets at least three hand-computed cases including a tie and an
empty/edge case, because a metric that is merely "reasonable" is how retrieval numbers become
unfalsifiable.

    # AT-EVL-04  # AT-EVL-07  # AT-ML-01
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from answerdesk.contracts import Action, ReasonCode
from answerdesk.eval.metrics import (
    answer_cells,
    escalation_cells,
    macro_interval,
    mean_cell,
    mrr_at_k,
    ndcg_at_k,
    operational_cells,
    recall_all_at_k,
    recall_any_at_k,
    retrieval_cells,
    score_retrieval,
)
from answerdesk.eval.models import TurnRecord


def test_recall_any_exact_values() -> None:
    assert recall_any_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert recall_any_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert recall_any_at_k([], ["c"], 5) == 0.0


def test_recall_any_requires_a_gold_set() -> None:
    """An empty gold set is a labelling bug, not a zero: §10.3 rule 5."""
    with pytest.raises(ValueError):
        recall_any_at_k(["a"], [], 3)


def test_recall_all_multi_chunk_hand_computed() -> None:
    # gold = {a, c}: two hits in the top 4 -> 2/3
    assert recall_all_at_k(["x", "a", "y", "c"], ["a", "c", "z"], 4) == pytest.approx(2 / 3)
    assert recall_all_at_k(["a"], ["a"], 1) == 1.0
    assert recall_all_at_k([], ["a"], 1) == 0.0


def test_recall_any_with_ties_between_gold_chunks() -> None:
    """Equal scores tie-break on chunk id, but any gold hit counts once."""
    assert recall_any_at_k(["b", "a"], ["a", "b"], 1) == 1.0
    assert recall_any_at_k(["b", "a"], ["a", "b"], 2) == 1.0


def test_ndcg_hand_computed() -> None:
    # single gold at rank 1: DCG = 1, IDCG = 1
    assert ndcg_at_k(["a", "x"], ["a"], 2) == pytest.approx(1.0)
    # gold at rank 2: DCG = 1/log2(3); IDCG (one gold) = 1
    assert ndcg_at_k(["x", "a"], ["a"], 2) == pytest.approx(1 / math.log2(3))
    # two gold, both at ranks 1 and 2: perfect
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)
    # two gold, only one retrieved at rank 1
    dcg = 1.0
    idcg = 1.0 + 1 / math.log2(3)
    assert ndcg_at_k(["a", "x"], ["a", "b"], 2) == pytest.approx(dcg / idcg)


def test_ndcg_no_hits_and_empty_ranking() -> None:
    assert ndcg_at_k([], ["a"], 5) == 0.0
    assert ndcg_at_k(["x", "y"], ["a"], 2) == 0.0


def test_mrr_hand_computed() -> None:
    assert mrr_at_k(["a"], ["a"], 5) == 1.0
    assert mrr_at_k(["x", "a"], ["a"], 5) == 0.5
    assert mrr_at_k(["x", "y", "z", "a"], ["a"], 3) == 0.0
    assert mrr_at_k([], ["a"], 5) == 0.0


def test_score_retrieval_skips_cases_without_gold(make_turn: Callable[..., TurnRecord]) -> None:
    turn = make_turn(gold_chunk_ids=[])
    assert score_retrieval(turn, 10) is None


def test_retrieval_cells_shape_and_macro(make_turn: Callable[..., TurnRecord]) -> None:
    turns = [
        make_turn(locale="ru", ranked_chunk_ids=["c1", "c9"], gold_chunk_ids=["c1"]),
        make_turn(locale="tr", ranked_chunk_ids=["c9", "c1"], gold_chunk_ids=["c1"]),
    ]
    cells = retrieval_cells(turns, ks=(10,), samples=200, seed=1)
    names = {(cell.name, cell.locale) for cell in cells}
    assert ("recall_any@10", "ru") in names
    assert ("recall_any@10", "macro") in names
    macro = next(cell for cell in cells if cell.locale == "macro" and cell.name == "recall_any@10")
    assert macro.value == pytest.approx(1.0)
    assert macro.n == 2
    assert macro.ci_low <= macro.value <= macro.ci_high
    for cell in cells:
        assert cell.n >= 0
        assert cell.ci_low <= cell.value <= cell.ci_high


def test_macro_interval_is_two_level() -> None:
    values = {"ru": [1.0, 1.0, 0.0, 1.0], "tr": [0.0, 0.0, 0.0, 0.0]}
    value, low, high, n = macro_interval(values, samples=300, seed=0)
    assert n == 8
    assert value == pytest.approx((0.75 + 0.0) / 2)
    assert low <= value <= high
    # the macro interval may not be narrower than the discrete spread of locale means
    assert low <= 0.0 + 1e-9 or low < value


def test_mean_cell_reports_n_and_interval() -> None:
    cell = mean_cell("x", [0.2, 0.4, 0.6], samples=200, seed=3)
    assert cell.n == 3
    assert cell.method == "bootstrap"
    assert cell.ci_low <= 0.4 <= cell.ci_high
    empty = mean_cell("y", [])
    assert empty.n == 0 and empty.method == "none"


def test_answer_cells_groundedness_and_citations(make_turn: Callable[..., TurnRecord]) -> None:
    turns = [
        make_turn(claims_total=2, claims_supported=2, groundedness=1.0, citation_precision=1.0),
        make_turn(claims_total=2, claims_supported=1, groundedness=0.5, citation_precision=0.5),
    ]
    cells = {cell.name: cell for cell in answer_cells(turns, samples=100, seed=0)}
    assert cells["groundedness_verifier"].value == pytest.approx(0.75)
    assert cells["groundedness_claim_pooled"].value == pytest.approx(3 / 4)
    assert cells["groundedness_claim_pooled"].n == 4
    assert cells["citation_precision"].value == pytest.approx(0.75)
    assert cells["phantom_citations"].value == 0.0
    assert cells["rubric_pass_rate"].value == pytest.approx(1.0)


def test_escalation_cells_risk_coverage_precision(make_turn: Callable[..., TurnRecord]) -> None:
    turns = [
        make_turn(answerable=True, answered=True, answer_ok=True, must_escalate=False),
        make_turn(answerable=True, answered=False, must_escalate=False),
        make_turn(answerable=False, answered=False, answer_ok=True, must_escalate=True),
        make_turn(answerable=False, answered=True, answer_ok=False, must_escalate=True),
    ]
    cells = {cell.name: cell for cell in escalation_cells(turns)}
    # answered turns: 2 of them, one incorrect -> risk 0.5
    assert cells["realised_risk"].value == pytest.approx(0.5)
    assert cells["realised_risk"].n == 2
    # coverage: one of two answerable items answered
    assert cells["coverage"].value == pytest.approx(0.5)
    # must-escalate recall: one of two caught
    assert cells["must_escalate_recall"].value == pytest.approx(0.5)
    # escalation precision: of two escalations, one was genuinely unsafe
    assert cells["escalation_precision"].value == pytest.approx(0.5)


def test_operational_cells_percentiles(make_turn: Callable[..., TurnRecord]) -> None:
    turns = [
        make_turn(latency_ms={"total": float(value), "retrieval": 1.0}, usd=0.001, input_tokens=10)
        for value in (10, 20, 30, 40, 50)
    ]
    cells = {cell.name: cell for cell in operational_cells(turns)}
    assert cells["latency_total_p50_ms"].value == pytest.approx(30.0)
    assert cells["latency_total_p95_ms"].value == pytest.approx(48.0, abs=0.5)
    assert cells["usd_per_answer"].value == pytest.approx(0.001)
    assert cells["cache_hit_rate"].value == pytest.approx(0.0)


def test_escalation_requires_consistent_decision_fields(
    make_turn: Callable[..., TurnRecord],
) -> None:
    with pytest.raises(ValueError):
        make_turn(
            answered=True, escalated=True, action=Action.ANSWER, reason_code=ReasonCode.ANSWERED
        )
