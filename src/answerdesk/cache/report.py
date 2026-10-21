"""Cache evaluation: wrong-hit budget, hit rate, and a precision/recall report.

Requirements §9.5 defines the three numbers that must be read together — K-1 (wrong-hit rate
with its Wilson bound), K-2 (hit-rate floor), K-3/K-4 (structural zeros) — precisely so that
no single one can be quoted alone. This module computes them from the replay set and writes
the report artifact (JSON + markdown + a hit-rate-vs-error plot, §9.6).

* ``wrong_hit_rate`` — wrong hits / hits, with the Wilson 95% upper bound (K-1).
* ``hit_rate`` — hits / pairs (K-2 floor is 0.15 for informational).
* ``precision`` — correct (same-answer) hits / hits.
* ``recall`` — correct hits / pairs whose cached gold answer still applies.

    # AT-CAC-07  # AT-CAC-08
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from answerdesk.cache.replay import EmbeddedPair
from answerdesk.cache.thresholds import DEFAULT_EPSILON, UCB_MULTIPLIER, ThresholdTable
from answerdesk.contracts import Contract
from answerdesk.policy.metrics import wilson_interval, wilson_upper
from answerdesk.policy.plots import plot_hit_rate_vs_error

#: K-2 floor: semantic hit rate for informational intents (requirements §9.5).
HIT_RATE_FLOOR = 0.15


class CohortEvaluation(Contract):
    """One cohort's measured cache behaviour at its calibrated threshold."""

    cohort: str = Field(min_length=1)
    threshold: float = Field(ge=0.0, le=1.0)
    epsilon: float = Field(gt=0.0, lt=1.0)
    n_pairs: int = Field(ge=0)
    hits: int = Field(ge=0)
    correct_hits: int = Field(ge=0)
    wrong_hits: int = Field(ge=0)
    pairs_with_same_answer: int = Field(ge=0)
    wrong_hit_rate: float = Field(ge=0.0, le=1.0)
    wrong_hit_ucb: float = Field(ge=0.0, le=1.0)
    hit_rate: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    feasible: bool = True

    def passes_budget(self) -> bool:
        return (
            self.wrong_hit_rate <= self.epsilon
            and self.wrong_hit_ucb <= UCB_MULTIPLIER * self.epsilon
        )


class CacheEvaluation(Contract):
    """Pooled cache quality plus per-cohort rows (the report artifact)."""

    epsilon: float = DEFAULT_EPSILON
    replay_hash: str = ""
    threshold_table_version: str = "1.0.0"
    n_pairs: int = 0
    hits: int = 0
    correct_hits: int = 0
    wrong_hits: int = 0
    pairs_with_same_answer: int = 0
    wrong_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    wrong_hit_ucb: float = Field(default=0.0, ge=0.0, le=1.0)
    hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    informational_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    cohorts: list[CohortEvaluation] = Field(default_factory=list)

    def passes_k1(self) -> bool:
        return (
            self.wrong_hit_rate <= self.epsilon
            and self.wrong_hit_ucb <= UCB_MULTIPLIER * self.epsilon
        )

    def passes_k2(self) -> bool:
        return self.informational_hit_rate >= HIT_RATE_FLOOR

    def summary_row(self) -> dict[str, object]:
        return {
            "n_pairs": self.n_pairs,
            "hits": self.hits,
            "hit_rate": round(self.hit_rate, 4),
            "wrong_hits": self.wrong_hits,
            "wrong_hit_rate": round(self.wrong_hit_rate, 4),
            "wrong_hit_ucb": round(self.wrong_hit_ucb, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "k1_pass": self.passes_k1(),
            "k2_pass": self.passes_k2(),
        }


def evaluate_cache(
    embedded: Sequence[EmbeddedPair],
    *,
    table: ThresholdTable,
    replay_hash: str,
    epsilon: float = DEFAULT_EPSILON,
) -> CacheEvaluation:
    """Score the threshold table against the replay set (a hit is ``sim >= τ``)."""
    by_cohort: dict[str, list[EmbeddedPair]] = {}
    for item in embedded:
        by_cohort.setdefault(item.pair.cohort, []).append(item)

    rows: list[CohortEvaluation] = []
    total_pairs = total_hits = total_wrong = total_correct = total_same = 0
    informational_hits = informational_pairs = 0

    for cohort in sorted(by_cohort):
        items = by_cohort[cohort]
        threshold = table.threshold_for(cohort)
        hits = wrong = correct = 0
        same_answer = 0
        for item in items:
            if item.pair.same_answer:
                same_answer += 1
            if not item.hits_at(threshold):
                continue
            hits += 1
            if item.pair.wrong_if_hit:
                wrong += 1
            else:
                correct += 1
        rate = wrong / hits if hits else 0.0
        ucb = wilson_upper(wrong, hits) if hits else 0.0
        table_row = table.row_for(cohort)
        row = CohortEvaluation(
            cohort=cohort,
            threshold=threshold,
            epsilon=epsilon,
            n_pairs=len(items),
            hits=hits,
            correct_hits=correct,
            wrong_hits=wrong,
            pairs_with_same_answer=same_answer,
            wrong_hit_rate=round(rate, 10),
            wrong_hit_ucb=round(ucb, 10),
            hit_rate=round(hits / len(items), 10) if items else 0.0,
            precision=round(correct / hits, 10) if hits else 0.0,
            recall=round(correct / same_answer, 10) if same_answer else 0.0,
            feasible=table_row.feasible if table_row is not None else True,
        )
        rows.append(row)
        total_pairs += len(items)
        total_hits += hits
        total_wrong += wrong
        total_correct += correct
        total_same += same_answer
        if cohort.startswith("informational:"):
            informational_hits += hits
            informational_pairs += len(items)

    rate = total_wrong / total_hits if total_hits else 0.0
    _, ucb = wilson_interval(total_wrong, total_hits) if total_hits else (0.0, 0.0)
    return CacheEvaluation(
        epsilon=epsilon,
        replay_hash=replay_hash,
        threshold_table_version=table.table_version,
        n_pairs=total_pairs,
        hits=total_hits,
        correct_hits=total_correct,
        wrong_hits=total_wrong,
        pairs_with_same_answer=total_same,
        wrong_hit_rate=round(rate, 10),
        wrong_hit_ucb=round(ucb, 10),
        hit_rate=round(total_hits / total_pairs, 10) if total_pairs else 0.0,
        precision=round(total_correct / total_hits, 10) if total_hits else 0.0,
        recall=round(total_correct / total_same, 10) if total_same else 0.0,
        informational_hit_rate=(
            round(informational_hits / informational_pairs, 10) if informational_pairs else 0.0
        ),
        cohorts=rows,
    )


def hit_rate_error_curve(
    embedded: Sequence[EmbeddedPair], *, steps: int = 40
) -> list[tuple[float, float]]:
    """Hit rate vs wrong-hit rate over the threshold sweep (vCache-style artifact, §9.6)."""
    if not embedded:
        return []
    similarities = sorted({round(item.similarity, 4) for item in embedded})
    if not similarities:
        return []
    stride = max(1, len(similarities) // steps)
    points: list[tuple[float, float]] = []
    for threshold in similarities[::stride]:
        hits = wrong = 0
        for item in embedded:
            if not item.hits_at(threshold):
                continue
            hits += 1
            if item.pair.wrong_if_hit:
                wrong += 1
        if hits == 0:
            continue
        points.append((wrong / hits, hits / len(embedded)))
    return points


def render_markdown(evaluation: CacheEvaluation, *, table: ThresholdTable) -> str:
    lines = [
        "# Semantic-cache report (synthetic near-miss replay set)",
        "",
        f"* replay hash: `{evaluation.replay_hash[:16]}…`",
        f"* threshold table: {table.table_version} (calibrated {table.calibrated_at})",
        f"* epsilon (wrong-hit budget): {evaluation.epsilon}",
        f"* pairs: {evaluation.n_pairs}; hits: {evaluation.hits}; "
        f"wrong hits: {evaluation.wrong_hits}",
        "",
        "## Release targets",
        "",
        "| metric | value | target | pass |",
        "|---|---|---|---|",
        f"| K-1 wrong-hit rate | {evaluation.wrong_hit_rate:.4f} | ≤ {evaluation.epsilon} | "
        f"{'yes' if evaluation.wrong_hit_rate <= evaluation.epsilon else '**NO**'} |",
        f"| K-1 wrong-hit UCB | {evaluation.wrong_hit_ucb:.4f} | ≤ {2 * evaluation.epsilon:.4f} | "
        f"{'yes' if evaluation.wrong_hit_ucb <= 2 * evaluation.epsilon else '**NO**'} |",
        f"| K-2 informational hit rate | {evaluation.informational_hit_rate:.4f} | ≥ "
        f"{HIT_RATE_FLOOR} | {'yes' if evaluation.passes_k2() else '**NO**'} |",
        f"| precision (correct hits / hits) | {evaluation.precision:.4f} | — | — |",
        f"| recall (correct hits / same-answer pairs) | {evaluation.recall:.4f} | — | — |",
        "",
        "## Per cohort",
        "",
        "| cohort | threshold | pairs | hits | hit rate | wrong hits | precision | recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in evaluation.cohorts:
        lines.append(
            f"| {row.cohort} | {row.threshold:.3f} | {row.n_pairs} | {row.hits} | "
            f"{row.hit_rate:.3f} | {row.wrong_hits} | {row.precision:.3f} | {row.recall:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "1. The replay set is synthetic: paraphrases are token reorderings and near misses",
            "   change one entity. Real traffic is harder; t11 replays the HITL-labelled set.",
            "2. A cohort with no feasible threshold reports `threshold=1.0` (no semantic hits),",
            "   which satisfies K-1 vacuously; K-2 is what prevents that from being a strategy.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    *, evaluation: CacheEvaluation, table: ThresholdTable, out_dir: str | Path
) -> list[Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.append(table.save(target / "cache-thresholds.json"))
    report_json = target / "cache-report.json"
    report_json.write_text(
        json.dumps(
            {
                "evaluation": evaluation.model_dump(mode="json"),
                "threshold_table": table.model_dump(mode="json"),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(report_json)
    markdown = target / "cache-report.md"
    markdown.write_text(render_markdown(evaluation, table=table), encoding="utf-8")
    written.append(markdown)
    return written


__all__ = [
    "HIT_RATE_FLOOR",
    "CacheEvaluation",
    "CohortEvaluation",
    "evaluate_cache",
    "hit_rate_error_curve",
    "plot_hit_rate_vs_error",
    "render_markdown",
    "write_report",
]
