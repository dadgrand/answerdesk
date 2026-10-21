"""Metric definitions and uncertainty helpers, against hand-computed values.

# AT-EVL-04
"""

from __future__ import annotations

import pytest

from answerdesk.retrieval.metrics import (
    evaluate_cases,
    macro_average,
    mcnemar_p_value,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_diff,
    per_case_scores,
    recall_all_at_k,
    recall_any_at_k,
    wilson_interval,
)

GOLD = ["g1", "g2"]


def test_recall_any_and_recall_all_against_hand_computed_values() -> None:
    # top-3 = [x, g1, y]: any-hit yes, half of the gold set found.
    ranked = ["x", "g1", "y", "g2"]
    assert recall_any_at_k(ranked, GOLD, 3) == 1.0
    assert recall_all_at_k(ranked, GOLD, 3) == 0.5
    assert recall_all_at_k(ranked, GOLD, 4) == 1.0
    assert recall_any_at_k(["x", "y"], GOLD, 2) == 0.0
    assert recall_all_at_k(["x", "y"], GOLD, 2) == 0.0


def test_empty_gold_scores_zero_everywhere() -> None:
    """An unanswerable item must never be counted as a hit (§6.1)."""
    assert recall_any_at_k(["a"], [], 5) == 0.0
    assert recall_all_at_k(["a"], [], 5) == 0.0
    assert ndcg_at_k(["a"], [], 5) == 0.0
    assert mrr_at_k(["a"], [], 5) == 0.0


def test_ndcg_matches_the_hand_computed_dcg_over_idcg() -> None:
    # Gold at positions 1 and 3: DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG for two gold = 1/log2(2) + 1/log2(3) = 1.0 + 0.63093 = 1.63093
    ranked = ["g1", "x", "g2"]
    assert ndcg_at_k(ranked, GOLD, 3) == pytest.approx(1.5 / (1.0 + 0.6309297535714574))
    # At k=1 the ideal ranking can only hold min(k, |G_q|) = 1 gold, so IDCG = 1.0 and a
    # gold at rank 1 is a perfect score — this is the §6.1 definition, not an error.
    assert ndcg_at_k(ranked, GOLD, 1) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], GOLD, 2) == 0.0


def test_mrr_is_the_reciprocal_rank_of_the_first_gold() -> None:
    assert mrr_at_k(["x", "g1"], GOLD, 5) == 0.5
    assert mrr_at_k(["g1", "x"], GOLD, 5) == 1.0
    assert mrr_at_k(["x", "y", "g2"], GOLD, 2) == 0.0  # outside the cut-off
    assert mrr_at_k(["x", "y", "g2"], GOLD, 3) == pytest.approx(1 / 3)


def test_k_beyond_the_ranking_uses_the_ranking_that_exists() -> None:
    assert recall_any_at_k(["g1"], GOLD, 100) == 1.0
    assert recall_all_at_k(["g1"], GOLD, 100) == 0.5


def test_evaluate_cases_aggregates_with_the_correct_denominator() -> None:
    rankings = [["g1"], ["x"], ["x", "g2"]]
    gold = [["g1"], ["g1"], ["g2"]]
    metrics = evaluate_cases(rankings, gold, 3)
    assert metrics.n == 3
    assert metrics.k == 3
    assert metrics.recall_any == pytest.approx(2 / 3)
    assert metrics.recall_all == pytest.approx(2 / 3)
    assert metrics.mrr == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert metrics.as_dict()["n"] == 3.0


def test_evaluate_cases_rejects_misaligned_input_and_handles_empty() -> None:
    with pytest.raises(ValueError, match="aligned"):
        evaluate_cases([["g1"]], [], 3)
    empty = evaluate_cases([], [], 3)
    assert empty.n == 0 and empty.recall_any == 0.0


def test_per_case_scores_returns_the_metric_vector() -> None:
    rankings = [["g1"], ["x"]]
    gold = [["g1"], ["g1"]]
    assert per_case_scores(rankings, gold, 3) == [1.0, 0.0]
    assert per_case_scores(rankings, gold, 3, metric="mrr") == [1.0, 0.0]
    with pytest.raises(ValueError, match="unknown metric"):
        per_case_scores(rankings, gold, 3, metric="f1")


def test_macro_average_gives_every_locale_equal_weight() -> None:
    assert macro_average({"ru": 1.0, "zh": 0.0, "ky": 0.5}) == pytest.approx(0.5)
    assert macro_average({}) == 0.0


def test_paired_bootstrap_is_seeded_and_brackets_the_mean() -> None:
    left = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    right = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    mean, low, high = paired_bootstrap_diff(left, right, samples=200, seed=1337)
    assert mean == pytest.approx(sum(a - b for a, b in zip(left, right, strict=True)) / 8)
    assert low <= mean <= high
    # Same seed -> same interval (reproducible evidence, NFR-DET-02).
    assert paired_bootstrap_diff(left, right, samples=200, seed=1337) == (mean, low, high)
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap_diff([1.0], [], samples=10)


def test_paired_bootstrap_on_identical_arms_is_a_zero_interval() -> None:
    values = [1.0, 0.0, 1.0, 1.0]
    mean, low, high = paired_bootstrap_diff(values, values, samples=50, seed=7)
    assert (mean, low, high) == (0.0, 0.0, 0.0)


def test_wilson_interval_is_stable_at_the_extremes() -> None:
    low, high = wilson_interval(0, 20)
    assert low == 0.0
    assert 0.0 < high < 0.2
    low, high = wilson_interval(20, 20)
    assert high == 1.0
    assert 0.8 < low < 1.0
    low, high = wilson_interval(10, 20)
    assert low < 0.5 < high
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_mcnemar_exact_p_value_matches_the_binomial_form() -> None:
    # 0 vs 5 discordant pairs: two-sided exact p = 2 * (1/32) = 0.0625
    assert mcnemar_p_value(0, 5) == pytest.approx(0.0625)
    assert mcnemar_p_value(5, 5) == pytest.approx(1.0)
    assert mcnemar_p_value(0, 0) == 1.0
    # Symmetric in its arguments by construction.
    assert mcnemar_p_value(2, 7) == pytest.approx(mcnemar_p_value(7, 2))
