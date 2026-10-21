"""Metric definitions — the numbers §6.1, §7.5, §8.5 and §10.4 say must be reported.

Every function here is a straight transcription of the requirements' formulas; the tests
compare them against hand-computed values including ties and empty sets (AT-EVL-04). Two
conventions matter and are applied consistently:

* **nothing is reported without n and a 95% CI** (FR-EVL-07). A rate gets a Wilson
  interval, a mean gets a paired-able bootstrap interval, a percentile is a distribution
  summary and says so (``method="none"``);
* **per-locale rows first, macro second** (§6.3 step 4). The macro row is the unweighted
  mean of the six locale values, so a well-resourced locale cannot hide a broken one;
  the micro (item-weighted) value is reported alongside as a diagnostic.

Retrieval metrics are defined on the ranked chunk ids the pipeline actually returned at the
k that was fed to the generator (§6.1 last paragraph), not on a friendlier k chosen later.

    # AT-EVL-04  # AT-EVL-07  # AT-ML-01
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from answerdesk.contracts import EvalCategory
from answerdesk.eval.models import MetricCell, TurnRecord
from answerdesk.eval.stats import (
    DEFAULT_BOOTSTRAP,
    Interval,
    bootstrap_interval,
    mean,
    quantile,
    wilson_interval,
)

#: k values the harness reports (§6.2): 20 is the retrieval gate, 10 the top-of-context.
DEFAULT_KS: tuple[int, ...] = (10, 20)
MACRO = "macro"


@dataclass(frozen=True)
class RetrievalScore:
    """Per-query retrieval outcome, kept as floats so ties and empty sets are explicit."""

    case_id: str
    locale: str
    any_hit: float
    all_hit: float
    ndcg: float
    reciprocal_rank: float


# --------------------------------------------------------------------------------------
# §6.1 formulas
# --------------------------------------------------------------------------------------


def recall_any_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """``1[G ∩ top_k ≠ ∅]`` — the primary gate (what the user experiences)."""
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("recall_any is undefined for an empty gold set (§10.3 rule 5)")
    return 1.0 if gold_set & set(ranked[:k]) else 0.0


def recall_all_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """``|G ∩ top_k| / |G|`` — completeness across a multi-chunk evidence set."""
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("recall_all is undefined for an empty gold set (§10.3 rule 5)")
    return len(gold_set & set(ranked[:k])) / len(gold_set)


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k exactly as §6.1 writes it (no gains, no typos)."""
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("nDCG is undefined for an empty gold set")
    dcg = 0.0
    for position, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in gold_set:
            dcg += 1.0 / _log2(position + 1)
    ideal_hits = min(k, len(gold_set))
    idcg = sum(1.0 / _log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """``1 / rank`` of the first gold chunk within k, else 0."""
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("MRR is undefined for an empty gold set")
    for position, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in gold_set:
            return 1.0 / position
    return 0.0


def _log2(value: float) -> float:
    import math

    return math.log2(value)


def score_retrieval(
    turn: TurnRecord, k: int, *, k_ndcg: int | None = None
) -> RetrievalScore | None:
    """Score one turn, or ``None`` when it carries no retrieval label (unanswerable)."""
    if not turn.gold_chunk_ids:
        return None
    return RetrievalScore(
        case_id=turn.case_id,
        locale=turn.locale,
        any_hit=recall_any_at_k(turn.ranked_chunk_ids, turn.gold_chunk_ids, k),
        all_hit=recall_all_at_k(turn.ranked_chunk_ids, turn.gold_chunk_ids, k),
        ndcg=ndcg_at_k(turn.ranked_chunk_ids, turn.gold_chunk_ids, k_ndcg or k),
        reciprocal_rank=mrr_at_k(turn.ranked_chunk_ids, turn.gold_chunk_ids, k),
    )


# --------------------------------------------------------------------------------------
# Cell builders
# --------------------------------------------------------------------------------------


def proportion_cell(
    name: str,
    successes: float,
    n: int,
    *,
    arm: str = "",
    locale: str = "",
    note: str = "",
    category: str = "",
) -> MetricCell:
    interval = wilson_interval(successes, n)
    return MetricCell(
        name=name,
        value=round(interval.value, 6),
        n=n,
        ci_low=round(interval.low, 6),
        ci_high=round(interval.high, 6),
        method="wilson" if n else "none",
        arm=arm,
        locale=locale,
        note=note,
        category=category,
    )


def mean_cell(
    name: str,
    values: Sequence[float],
    *,
    arm: str = "",
    locale: str = "",
    seed: int = 0,
    samples: int = DEFAULT_BOOTSTRAP,
    note: str = "",
    dependency: Callable[[], Interval] | None = None,
) -> MetricCell:
    if not values:
        return MetricCell(
            name=name,
            value=0.0,
            n=0,
            ci_low=0.0,
            ci_high=1.0,
            method="none",
            arm=arm,
            locale=locale,
            note=note or "no items",
        )
    interval = (
        dependency()
        if dependency is not None
        else bootstrap_interval(values, samples=samples, seed=seed)
    )
    return MetricCell(
        name=name,
        value=round(interval.value, 6),
        n=len(values),
        ci_low=round(min(interval.low, interval.value), 6),
        ci_high=round(max(interval.high, interval.value), 6),
        method=interval.method,
        arm=arm,
        locale=locale,
        note=note,
    )


def percentile_cell(
    name: str, values: Sequence[float], q: float, *, arm: str = "", note: str = ""
) -> MetricCell:
    value = quantile(values, q)
    return MetricCell(
        name=name,
        value=round(value, 6),
        n=len(values),
        ci_low=round(min(value, quantile(values, max(0.0, q - 0.05))), 6),
        ci_high=round(max(value, quantile(values, min(1.0, q + 0.05))), 6),
        method="none",
        arm=arm,
        note=note or "distribution summary; the bracket is the p(q-5)..p(q+5) spread",
    )


def _by_locale(turns: Iterable[TurnRecord]) -> dict[str, list[TurnRecord]]:
    grouped: dict[str, list[TurnRecord]] = {}
    for turn in turns:
        grouped.setdefault(turn.locale, []).append(turn)
    return dict(sorted(grouped.items()))


def _macro(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values) if values else 0.0


def macro_interval(
    per_locale_values: Mapping[str, Sequence[float]],
    *,
    samples: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float, int]:
    """Macro mean over locales with a CI from a *two-level* bootstrap.

    Resampling queries inside each locale and recomputing the equal-weight macro mean keeps
    the estimator the report prints inside the interval. Taking the envelope of the locale
    intervals instead would be a different, much wider claim.
    """
    import numpy as np

    locales = sorted(locale for locale, values in per_locale_values.items() if values)
    if not locales:
        return 0.0, 0.0, 1.0, 0
    arrays = {
        locale: np.asarray(list(per_locale_values[locale]), dtype=np.float64) for locale in locales
    }
    value = sum(float(array.mean()) for array in arrays.values()) / len(locales)
    n = sum(int(array.size) for array in arrays.values())
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means = []
        for locale in locales:
            array = arrays[locale]
            pick = generator.integers(0, array.size, size=array.size)
            means.append(float(array[pick].mean()))
        draws[index] = sum(means) / len(means)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(draws, [100 * alpha, 100 * (1 - alpha)])
    return value, float(min(low, value)), float(max(high, value)), n


def _macro_cell(
    name: str,
    per_locale: dict[str, MetricCell],
    per_locale_values: Mapping[str, Sequence[float]],
    *,
    arm: str,
    seed: int,
    samples: int,
) -> MetricCell:
    if not per_locale:
        return MetricCell(
            name=name,
            value=0.0,
            n=0,
            ci_low=0.0,
            ci_high=1.0,
            method="none",
            arm=arm,
            locale=MACRO,
            note="no locale rows",
        )
    value, low, high, n = macro_interval(per_locale_values, samples=samples, seed=seed)
    return MetricCell(
        name=name,
        value=round(value, 6),
        n=n,
        ci_low=round(low, 6),
        ci_high=round(high, 6),
        method="bootstrap",
        arm=arm,
        locale=MACRO,
        note="unweighted mean over the locales present; the interval is a two-level "
        "bootstrap (queries within locale, then the macro mean)",
    )


# --------------------------------------------------------------------------------------
# Layer: retrieval
# --------------------------------------------------------------------------------------


def retrieval_cells(
    turns: Sequence[TurnRecord],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    arm: str = "",
    seed: int = 0,
    samples: int = DEFAULT_BOOTSTRAP,
    locales: Sequence[str] = (),
) -> list[MetricCell]:
    """``recall_any@k``, ``recall_all@k``, ``nDCG@k``, ``MRR@k`` with per-locale + macro rows."""
    cells: list[MetricCell] = []
    if not any(turn.gold_chunk_ids for turn in turns):
        return cells
    for k in ks:
        per_locale_any: dict[str, MetricCell] = {}
        per_locale_all: dict[str, MetricCell] = {}
        per_locale_ndcg: dict[str, MetricCell] = {}
        per_locale_mrr: dict[str, MetricCell] = {}
        values_any: dict[str, list[float]] = {}
        values_all: dict[str, list[float]] = {}
        values_ndcg: dict[str, list[float]] = {}
        values_mrr: dict[str, list[float]] = {}
        for locale, group in _by_locale(turns).items():
            scores = [score for turn in group if (score := score_retrieval(turn, k))]
            if not scores:
                continue
            values_any[locale] = [score.any_hit for score in scores]
            values_all[locale] = [score.all_hit for score in scores]
            values_ndcg[locale] = [score.ndcg for score in scores]
            values_mrr[locale] = [score.reciprocal_rank for score in scores]
            per_locale_any[locale] = mean_cell(
                f"recall_any@{k}",
                values_any[locale],
                arm=arm,
                locale=locale,
                seed=seed,
                samples=samples,
            )
            per_locale_all[locale] = mean_cell(
                f"recall_all@{k}",
                values_all[locale],
                arm=arm,
                locale=locale,
                seed=seed,
                samples=samples,
            )
            per_locale_ndcg[locale] = mean_cell(
                f"nDCG@{k}",
                values_ndcg[locale],
                arm=arm,
                locale=locale,
                seed=seed,
                samples=samples,
            )
            per_locale_mrr[locale] = mean_cell(
                f"MRR@{k}",
                values_mrr[locale],
                arm=arm,
                locale=locale,
                seed=seed,
                samples=samples,
            )
        for name, per_locale, values in (
            (f"recall_any@{k}", per_locale_any, values_any),
            (f"recall_all@{k}", per_locale_all, values_all),
            (f"nDCG@{k}", per_locale_ndcg, values_ndcg),
            (f"MRR@{k}", per_locale_mrr, values_mrr),
        ):
            cells.extend(per_locale.values())
            cells.append(_macro_cell(name, per_locale, values, arm=arm, seed=seed, samples=samples))
    _ = locales  # locale coverage is asserted by the dataset manifest, not here
    return cells


@dataclass(frozen=True)
class PairedScores:
    """Aligned per-case vectors for a paired comparison between two arms (§6.3 step 5)."""

    case_ids: list[str]
    any_hit: list[float]
    all_hit: list[float]


def paired_scores(turns: Sequence[TurnRecord], k: int) -> PairedScores:
    case_ids: list[str] = []
    any_hits: list[float] = []
    all_hits: list[float] = []
    for turn in turns:
        score = score_retrieval(turn, k)
        if score is None:
            continue
        case_ids.append(turn.case_id)
        any_hits.append(score.any_hit)
        all_hits.append(score.all_hit)
    return PairedScores(case_ids=case_ids, any_hit=any_hits, all_hit=all_hits)


# --------------------------------------------------------------------------------------
# Layer: answer quality
# --------------------------------------------------------------------------------------


def answer_cells(
    turns: Sequence[TurnRecord],
    *,
    arm: str = "",
    seed: int = 0,
    samples: int = DEFAULT_BOOTSTRAP,
    judged: bool = False,
) -> list[MetricCell]:
    """Groundedness, citation precision/recall, rubric score and the two structural counts."""
    cells: list[MetricCell] = []
    answered = [turn for turn in turns if turn.answered]
    grounded: list[float] = []
    grounded_judged: list[float] = []
    precision: list[float] = []
    recall: list[float] = []
    rubric: list[float] = []
    supported_claims = 0
    total_claims = 0
    for turn in answered:
        if turn.groundedness is not None:
            grounded.append(turn.groundedness)
        if turn.groundedness_judged is not None:
            grounded_judged.append(turn.groundedness_judged)
        if turn.citation_precision is not None:
            precision.append(turn.citation_precision)
        if turn.citation_recall is not None:
            recall.append(turn.citation_recall)
        if turn.rubric_score is not None:
            rubric.append(turn.rubric_score)
        supported_claims += turn.claims_supported
        total_claims += turn.claims_total

    groundedness_name = "groundedness_judged" if judged else "groundedness_verifier"
    cells.append(
        mean_cell(
            groundedness_name,
            grounded,
            arm=arm,
            seed=seed,
            samples=samples,
            note="mean per-answer claim-support rate" if grounded else "no answered turns",
        )
    )
    cells.append(
        proportion_cell(
            "groundedness_claim_pooled",
            supported_claims,
            total_claims,
            arm=arm,
            note=f"{supported_claims}/{total_claims} displayed claims supported",
        )
    )
    cells.append(
        mean_cell(
            "citation_precision",
            precision,
            arm=arm,
            seed=seed,
            samples=samples,
            note="per-answer mean; a phantom citation is impossible by contract (F-5)",
        )
    )
    cells.append(
        mean_cell(
            "citation_recall",
            recall,
            arm=arm,
            seed=seed,
            samples=samples,
            note="fraction of required evidence chunks actually cited",
        )
    )
    if rubric:
        cells.append(
            mean_cell(
                "rubric_score",
                rubric,
                arm=arm,
                seed=seed,
                samples=samples,
                note="judge rubric, 1-5 (context-anchored, per claim)",
            )
        )
    reference_scores = [
        turn.reference_rubric_score for turn in answered if turn.reference_rubric_score is not None
    ]
    if reference_scores:
        cells.append(
            mean_cell(
                "rubric_reference_overlap",
                reference_scores,
                arm=arm,
                seed=seed,
                samples=samples,
                note="diagnostic: the same rubric against the reference answer; low means the "
                "reference is not entailed by any document",
            )
        )
    passes = [1.0 if turn.rubric_pass else 0.0 for turn in answered if turn.rubric_pass is not None]
    if passes:
        cells.append(
            proportion_cell(
                "rubric_pass_rate",
                sum(passes),
                len(passes),
                arm=arm,
                note="rubric >= 4 (F-1/§8.1 'correct' component)",
            )
        )
    cells.append(
        proportion_cell(
            "phantom_citations",
            sum(turn.phantom_citations for turn in answered),
            len(answered),
            arm=arm,
            note="structural: must be 0 (F-5)",
        )
    )
    cells.append(
        proportion_cell(
            "unsupported_displayed_claims",
            sum(turn.unsupported_displayed_claims for turn in answered),
            len(answered),
            arm=arm,
            note="structural: must be 0 (F-4)",
        )
    )
    language_matches = [
        1.0 if turn.language_match else 0.0 for turn in answered if turn.expected_language
    ]
    if language_matches:
        cells.append(
            proportion_cell(
                "language_agreement",
                sum(language_matches),
                len(language_matches),
                arm=arm,
                note="F-7: answer language equals the detected query language",
            )
        )
    return cells


# --------------------------------------------------------------------------------------
# Layer: escalation
# --------------------------------------------------------------------------------------


def escalation_cells(turns: Sequence[TurnRecord], *, arm: str = "") -> list[MetricCell]:
    """Realised risk, coverage, must-escalate recall and escalation precision (§8.5)."""
    cells: list[MetricCell] = []
    answered = [turn for turn in turns if turn.answered]
    answerable = [turn for turn in turns if turn.answerable]
    must = [turn for turn in turns if turn.must_escalate]
    escalated = [turn for turn in turns if turn.escalated]

    wrong = sum(1 for turn in answered if not turn.answer_ok)
    cells.append(
        proportion_cell(
            "realised_risk",
            wrong,
            len(answered),
            arm=arm,
            note=f"{wrong}/{len(answered)} auto-answered turns were incorrect (E-1)",
        )
    )
    answered_answerable = sum(1 for turn in answerable if turn.answered)
    cells.append(
        proportion_cell(
            "coverage",
            answered_answerable,
            len(answerable),
            arm=arm,
            note="auto-answer coverage on answerable items (E-4)",
        )
    )
    caught = sum(1 for turn in must if turn.escalated)
    cells.append(
        proportion_cell(
            "must_escalate_recall",
            caught,
            len(must),
            arm=arm,
            note=f"{caught}/{len(must)} unsafe or unanswerable turns were escalated (E-2)",
        )
    )
    true_positive = sum(1 for turn in escalated if turn.must_escalate)
    cells.append(
        proportion_cell(
            "escalation_precision",
            true_positive,
            len(escalated),
            arm=arm,
            note="escalated turns that were genuinely unsafe/unanswerable (E-3)",
        )
    )
    for bucket in sorted({turn.intent_bucket.value for turn in turns}):
        group = [turn for turn in turns if turn.intent_bucket.value == bucket]
        group_answered = [turn for turn in group if turn.answered]
        group_answerable = [turn for turn in group if turn.answerable]
        if not group_answered and not group_answerable:
            continue
        cells.append(
            proportion_cell(
                "realised_risk_by_bucket",
                sum(1 for turn in group_answered if not turn.answer_ok),
                len(group_answered),
                arm=arm,
                note=f"intent bucket {bucket} (E-1 per bucket)",
                category=bucket,
            )
        )
        cells.append(
            proportion_cell(
                "coverage_by_bucket",
                sum(1 for turn in group if turn.answerable and turn.answered),
                sum(1 for turn in group if turn.answerable),
                arm=arm,
                note=f"intent bucket {bucket}",
                category=bucket,
            )
        )
    return cells


def category_cells(turns: Sequence[TurnRecord], *, arm: str = "") -> list[MetricCell]:
    """Answer accuracy per golden category — where a failure is attributable (§10.4)."""
    cells: list[MetricCell] = []
    for category in EvalCategory:
        group = [turn for turn in turns if turn.category is category]
        if not group:
            continue
        correct = sum(1 for turn in group if turn.answer_ok)
        cells.append(
            proportion_cell(
                "answer_accuracy_by_category",
                correct,
                len(group),
                arm=arm,
                note=f"category {category.value}",
                category=category.value,
            )
        )
    return cells


# --------------------------------------------------------------------------------------
# Layer: operations
# --------------------------------------------------------------------------------------


def operational_cells(turns: Sequence[TurnRecord], *, arm: str = "") -> list[MetricCell]:
    """p50/p95 latency per stage, USD per answer, token counts and cache hit rate."""
    cells: list[MetricCell] = []
    stages = sorted({stage for turn in turns for stage in turn.latency_ms})
    for stage in stages:
        values = [turn.latency_ms[stage] for turn in turns if stage in turn.latency_ms]
        cells.append(percentile_cell(f"latency_{stage}_p50_ms", values, 0.50, arm=arm))
        cells.append(percentile_cell(f"latency_{stage}_p95_ms", values, 0.95, arm=arm))
    totals = [turn.latency_ms["total"] for turn in turns if "total" in turn.latency_ms]
    cells.append(
        mean_cell(
            "latency_total_mean_ms",
            totals,
            arm=arm,
            note="wall clock measured in-process; not a server-side latency claim (NFR-LAT-01)",
        )
    )
    cells.append(
        mean_cell(
            "usd_per_answer",
            [turn.usd for turn in turns],
            arm=arm,
            note="priced with the pinned pricing table (NFR-COST-02)",
        )
    )
    cells.append(percentile_cell("usd_per_answer_p95", [turn.usd for turn in turns], 0.95, arm=arm))
    cells.append(
        mean_cell("tokens_input_mean", [float(turn.input_tokens) for turn in turns], arm=arm)
    )
    cells.append(
        mean_cell("tokens_output_mean", [float(turn.output_tokens) for turn in turns], arm=arm)
    )
    hits = sum(1 for turn in turns if turn.cache_status.value == "hit")
    cells.append(
        proportion_cell(
            "cache_hit_rate",
            hits,
            len(turns),
            arm=arm,
            note="exact-normalised-query replay; the semantic cache is owned by t8 (§9)",
        )
    )
    return cells


def overall_accuracy(turns: Sequence[TurnRecord]) -> float:
    """Fraction of turns the §8.1 definition of *correct* accepts."""
    if not turns:
        return 0.0
    return sum(1 for turn in turns if turn.answer_ok) / len(turns)


def mean_latency(turns: Sequence[TurnRecord], stage: str = "total") -> float:
    return mean([turn.latency_ms[stage] for turn in turns if stage in turn.latency_ms])


__all__ = [
    "DEFAULT_KS",
    "MACRO",
    "PairedScores",
    "RetrievalScore",
    "answer_cells",
    "category_cells",
    "escalation_cells",
    "macro_interval",
    "mean_cell",
    "mean_latency",
    "mrr_at_k",
    "ndcg_at_k",
    "operational_cells",
    "overall_accuracy",
    "paired_scores",
    "percentile_cell",
    "proportion_cell",
    "recall_all_at_k",
    "recall_any_at_k",
    "retrieval_cells",
    "score_retrieval",
]
