"""Retrieval metrics, implemented exactly as ``docs/requirements.md`` §6.1 defines them.

The definitions are normative, so they are transcribed literally:

::

    recall_any@k = (1/|Q|) · Σ_q 1[ G_q ∩ top_k(q) ≠ ∅ ]
    recall_all@k = (1/|Q|) · Σ_q |G_q ∩ top_k(q)| / |G_q|
    nDCG_q@k     = DCG_q@k / IDCG_q@k,  DCG_q@k = Σ_{i=1..k} rel_i / log2(i+1)
    MRR@k        = (1/|Q|) · Σ_q 1 / rank_q(first gold)  if found within k else 0

Two recall variants exist because they answer different questions: ``recall_any`` is what
the *user* experiences (one right source made it into the context), ``recall_all`` is what
the *answer* needs when the evidence is spread across chunks. Reporting only one is how
retrieval numbers become unfalsifiable (§6.1).

Edge cases are specified rather than left to chance: no gold at all scores 0 on
every metric (an unanswerable item must not be silently counted as a hit), ties are
resolved by the caller's ranking order (the metrics never re-sort), and ``k`` beyond the
ranking length simply uses the ranking that exists.

Uncertainty helpers are here too — paired bootstrap for "arm A beats arm B" claims and a
Wilson interval for proportions — because a retrieval table without error bars is a
decoration (§6.3).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

#: Channels/arms a metric row can be tagged with.
Row = Mapping[str, float]


def _top(ranked: Sequence[str], k: int) -> list[str]:
    return list(ranked[: max(0, k)])


def recall_any_at_k(ranked: Sequence[str], gold: Sequence[str] | set[str], k: int) -> float:
    """1.0 when at least one gold chunk is in the top-k, else 0.0."""
    if not gold:
        return 0.0
    return 1.0 if set(_top(ranked, k)) & set(gold) else 0.0


def recall_all_at_k(ranked: Sequence[str], gold: Sequence[str] | set[str], k: int) -> float:
    """Fraction of the gold set found in the top-k (1.0 when the gold set is empty)."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return len(set(_top(ranked, k)) & gold_set) / len(gold_set)


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str] | set[str], k: int) -> float:
    """Binary-gain nDCG@k with the ideal ranking built from ``min(k, |G_q|)``."""
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    dcg = 0.0
    for index, chunk_id in enumerate(_top(ranked, k), start=1):
        if chunk_id in gold_set:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(k, len(gold_set))
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def mrr_at_k(ranked: Sequence[str], gold: Sequence[str] | set[str], k: int) -> float:
    """Reciprocal rank of the first gold chunk within k (0.0 when absent)."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    for rank, chunk_id in enumerate(_top(ranked, k), start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True)
class RetrievalMetrics:
    """One metric row: the numbers a release target is compared against (§6.4)."""

    n: int
    k: int
    recall_any: float
    recall_all: float
    ndcg: float
    mrr: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n": float(self.n),
            "k": float(self.k),
            "recall_any": round(self.recall_any, 6),
            "recall_all": round(self.recall_all, 6),
            "ndcg": round(self.ndcg, 6),
            "mrr": round(self.mrr, 6),
        }


def evaluate_cases(
    rankings: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
    k: int,
) -> RetrievalMetrics:
    """Aggregate the four metrics over a list of per-query rankings at one ``k``."""
    if len(rankings) != len(gold):
        raise ValueError("rankings and gold must be aligned")
    n = len(rankings)
    if n == 0:
        return RetrievalMetrics(n=0, k=k, recall_any=0.0, recall_all=0.0, ndcg=0.0, mrr=0.0)
    totals = {"recall_any": 0.0, "recall_all": 0.0, "ndcg": 0.0, "mrr": 0.0}
    for ranked, gold_ids in zip(rankings, gold, strict=True):
        totals["recall_any"] += recall_any_at_k(ranked, gold_ids, k)
        totals["recall_all"] += recall_all_at_k(ranked, gold_ids, k)
        totals["ndcg"] += ndcg_at_k(ranked, gold_ids, k)
        totals["mrr"] += mrr_at_k(ranked, gold_ids, k)
    return RetrievalMetrics(
        n=n,
        k=k,
        recall_any=totals["recall_any"] / n,
        recall_all=totals["recall_all"] / n,
        ndcg=totals["ndcg"] / n,
        mrr=totals["mrr"] / n,
    )


def per_case_scores(
    rankings: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
    k: int,
    metric: str = "recall_any",
) -> list[float]:
    """Per-query metric vector — the input to a paired bootstrap (§6.3 step 5)."""
    functions = {
        "recall_any": recall_any_at_k,
        "recall_all": recall_all_at_k,
        "ndcg": ndcg_at_k,
        "mrr": mrr_at_k,
    }
    try:
        function = functions[metric]
    except KeyError as exc:
        raise ValueError(f"unknown metric {metric!r}; known: {sorted(functions)}") from exc
    return [
        float(function(ranked, gold_ids, k))
        for ranked, gold_ids in zip(rankings, gold, strict=True)
    ]


def macro_average(values: Mapping[str, float]) -> float:
    """Equal-weight mean over locales — the aggregation rule of §6.3 step 4."""
    if not values:
        return 0.0
    return sum(values.values()) / len(values)


def paired_bootstrap_diff(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 1000,
    seed: int = 1337,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean difference ``left − right`` with a percentile CI over paired resamples.

    Returns ``(mean_diff, lower, upper)``. An improvement claim survives only when the
    lower bound is > 0 (§6.3 step 5, RT-7).
    """
    if len(left) != len(right):
        raise ValueError("paired bootstrap needs aligned per-query vectors")
    if not left:
        return 0.0, 0.0, 0.0
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    differences = a - b
    mean_diff = float(differences.mean())
    rng = np.random.default_rng(seed)
    sample_means = np.empty(max(1, samples), dtype=np.float64)
    for index in range(sample_means.size):
        picks = rng.integers(0, differences.size, differences.size)
        sample_means[index] = float(differences[picks].mean())
    lower = float(np.quantile(sample_means, alpha / 2.0))
    upper = float(np.quantile(sample_means, 1.0 - alpha / 2.0))
    return mean_diff, lower, upper


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion — stable at 0 and 1, unlike the normal CI."""
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denominator = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denominator
    margin = (z / denominator) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def mcnemar_p_value(only_left: int, only_right: int) -> float:
    """Exact two-sided McNemar p-value for paired binary outcomes (§6.3 step 5).

    ``only_left`` is the number of queries only arm A got right, ``only_right`` only arm B.
    The exact binomial form is used because the counts here are small (hundreds, not
    thousands) and the chi-square approximation is unreliable at that size.
    """
    total = only_left + only_right
    if total == 0:
        return 1.0
    k = min(only_left, only_right)
    tail = sum(math.comb(total, index) for index in range(0, k + 1)) / (2.0**total)
    return min(1.0, 2.0 * tail)


__all__ = [
    "RetrievalMetrics",
    "evaluate_cases",
    "macro_average",
    "mcnemar_p_value",
    "mrr_at_k",
    "ndcg_at_k",
    "paired_bootstrap_diff",
    "per_case_scores",
    "recall_all_at_k",
    "recall_any_at_k",
    "wilson_interval",
]
