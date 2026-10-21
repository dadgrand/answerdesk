"""Measure what a rewriter is worth: retrieval uplift, latency, and the clean gate.

The harness answers one question honestly: *does rewriting the query change retrieval recall,
and by how much?* It runs the **real** retrieval pipeline (t5/t23) over the frozen corpus index
twice — once with the original query, once with the rewriter's output — on the same cases, so
the comparison is paired, and reports:

* ``recall_any@10`` and ``recall_any@20`` per slice (``noisy_freeform``, held-out
  perturbations, clean) for both arms and the delta;
* a **paired bootstrap** confidence interval on the per-case recall difference (1000 resamples,
  seeded — the same statistic t23 uses for its ablations);
* **latency** overhead per query (p50/p95 of the rewrite call itself);
* the **clean-query non-regression gate**: the clean-slice delta must be ≥ −0.01, because a
  rewriter that helps noisy input while hurting clean input is a net loss.

The retrieval import is local to the functions here, so `tests/unit/rewrite` runs without the
retrieval stack or a built index (they inject a fake searcher instead).

    # AT-GEN-17
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from answerdesk.contracts import Contract
from answerdesk.rewrite.base import QueryRewriter

#: Recall cut-offs reported by the harness (requirements §6.2: 20 is the release gate).
RECALL_KS: tuple[int, ...] = (10, 20)

#: Clean-slice non-regression tolerance (absolute recall drop allowed).
CLEAN_DROP_TOLERANCE = 0.01

#: Default bootstrap resamples / seed (the t23 ablation values, kept identical).
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20261021


class Searcher(Protocol):
    """Minimal retrieval seam: return chunk ids for a query, best first."""

    def search_ids(self, query: str, k: int) -> list[str]: ...


@dataclass
class EvalCase:
    """One evaluation item: the noisy query, its clean reference and the gold chunk ids."""

    case_id: str
    locale: str
    query: str
    gold_chunk_ids: list[str]
    style: str = ""
    category: str = ""
    split: str = "test"


@dataclass
class SliceResult:
    """Per-slice numbers for both arms."""

    name: str
    n: int
    recall: dict[int, float] = field(default_factory=dict)
    recall_rewritten: dict[int, float] = field(default_factory=dict)
    delta: dict[int, float] = field(default_factory=dict)
    ci_low: dict[int, float] = field(default_factory=dict)
    ci_high: dict[int, float] = field(default_factory=dict)
    improved: int = 0
    worsened: int = 0


class LatencySummary(Contract):
    """Rewrite-call latency in milliseconds (retrieval excluded)."""

    p50: float = 0.0
    p95: float = 0.0
    mean: float = 0.0
    note: str = ""


class EvaluationReport(Contract):
    """The persisted result of one rewriter comparison."""

    backend: str
    corpus_version: str = ""
    index_version: str = ""
    n_cases: int = 0
    ks: list[int] = Field(default_factory=lambda: list(RECALL_KS))
    slices: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: LatencySummary = Field(default_factory=LatencySummary)
    clean_gate_passed: bool = True
    clean_gate_delta: float = 0.0
    bootstrap_samples: int = BOOTSTRAP_SAMPLES
    seed: int = BOOTSTRAP_SEED
    notes: str = ""

    def slice_by_name(self, name: str) -> dict[str, Any] | None:
        return next((entry for entry in self.slices if entry.get("name") == name), None)


def _hits(gold: Sequence[str], ranked: Sequence[str], k: int) -> int:
    gold_set = set(gold)
    return 1 if gold_set & set(ranked[:k]) else 0


def paired_bootstrap(
    baseline: Sequence[int],
    candidate: Sequence[int],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Mean paired difference and its 95% percentile interval (bootstrap over cases)."""
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must be paired")
    n = len(baseline)
    if n == 0:
        return 0.0, 0.0, 0.0
    differences = [c - b for b, c in zip(baseline, candidate, strict=True)]
    mean = sum(differences) / n
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        total = 0
        for _ in range(n):
            total += differences[rng.randrange(n)]
        estimates.append(total / n)
    estimates.sort()
    low = estimates[max(0, math.floor(0.025 * samples) - 1)]
    high = estimates[min(samples - 1, math.ceil(0.975 * samples) - 1)]
    return mean, low, high


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _slice(
    name: str,
    cases: Sequence[EvalCase],
    *,
    searcher: Searcher,
    rewriter: QueryRewriter,
    k_max: int,
    samples: int,
    seed: int,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[SliceResult, list[dict[str, Any]], list[float]]:
    baseline_hits: dict[int, list[int]] = {k: [] for k in RECALL_KS}
    rewritten_hits: dict[int, list[int]] = {k: [] for k in RECALL_KS}
    traces: list[dict[str, Any]] = []
    rewrite_latencies: list[float] = []
    # A batch-capable backend (the LoRA one) is used in batch mode so the harness measures
    # quality, not single-query latency; the reported latency is then amortised per item and
    # flagged as such.
    batch_rewriter = getattr(rewriter, "rewrite_many", None)
    batch_results: list[Any] | None = None
    if callable(batch_rewriter):
        batch_started = clock()
        batch_results = batch_rewriter(
            [case.query for case in cases], locales=[case.locale for case in cases]
        )
        batch_ms = (clock() - batch_started) * 1000.0
        per_item = batch_ms / max(1, len(cases))
        rewrite_latencies.extend([per_item] * len(cases))
    for index, case in enumerate(cases):
        if batch_results is not None:
            result = batch_results[index]
        else:
            rewrite_started = clock()
            result = rewriter.rewrite(case.query, locale=case.locale)
            rewrite_latencies.append((clock() - rewrite_started) * 1000.0)
        baseline_ranked = searcher.search_ids(case.query, k_max)
        rewritten_ranked = searcher.search_ids(result.query, k_max)
        entry: dict[str, Any] = {
            "case_id": case.case_id,
            "locale": case.locale,
            "original": case.query,
            "rewritten": result.query,
            "changed": result.changed,
        }
        for k in RECALL_KS:
            base = _hits(case.gold_chunk_ids, baseline_ranked, k)
            cand = _hits(case.gold_chunk_ids, rewritten_ranked, k)
            baseline_hits[k].append(base)
            rewritten_hits[k].append(cand)
            entry[f"hit@{k}"] = base
            entry[f"hit@{k}_rewritten"] = cand
        traces.append(entry)
    result = SliceResult(name=name, n=len(cases))
    for k in RECALL_KS:
        base_scores = baseline_hits[k]
        cand_scores = rewritten_hits[k]
        mean, low, high = paired_bootstrap(base_scores, cand_scores, samples=samples, seed=seed)
        result.recall[k] = round(sum(base_scores) / len(base_scores), 6) if base_scores else 0.0
        result.recall_rewritten[k] = (
            round(sum(cand_scores) / len(cand_scores), 6) if cand_scores else 0.0
        )
        result.delta[k] = round(mean, 6)
        result.ci_low[k] = round(low, 6)
        result.ci_high[k] = round(high, 6)
    result.improved = sum(
        1
        for base, cand in zip(
            baseline_hits[RECALL_KS[-1]], rewritten_hits[RECALL_KS[-1]], strict=True
        )
        if cand > base
    )
    result.worsened = sum(
        1
        for base, cand in zip(
            baseline_hits[RECALL_KS[-1]], rewritten_hits[RECALL_KS[-1]], strict=True
        )
        if cand < base
    )
    return result, traces, rewrite_latencies


def evaluate_rewriter(
    *,
    rewriter: QueryRewriter,
    slices: dict[str, Sequence[EvalCase]],
    searcher: Searcher,
    corpus_version: str = "",
    index_version: str = "",
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    clock: Callable[[], float] = time.perf_counter,
    latency_note: str | None = None,
) -> tuple[EvaluationReport, dict[str, list[dict[str, Any]]]]:
    """Run both arms on every slice; returns the report and the per-case traces."""
    k_max = max(RECALL_KS)
    latencies: list[float] = []
    latency_note = latency_note or ("wall-clock of the rewrite call only (retrieval excluded)")
    batched = callable(getattr(rewriter, "rewrite_many", None))
    if batched:
        latency_note = (
            "batched amortised (batch generation divided by items); not single-request latency"
        )
    report_slices: list[dict[str, Any]] = []
    traces: dict[str, list[dict[str, Any]]] = {}
    for name, cases in slices.items():
        if not cases:
            continue
        result, slice_traces, slice_latencies = _slice(
            name,
            cases,
            searcher=searcher,
            rewriter=rewriter,
            k_max=k_max,
            samples=samples,
            seed=seed,
            clock=clock,
        )
        latencies.extend(slice_latencies)
        traces[name] = slice_traces
        report_slices.append(
            {
                "name": name,
                "n": result.n,
                **{f"recall@{k}": result.recall[k] for k in RECALL_KS},
                **{f"recall@{k}_rewritten": result.recall_rewritten[k] for k in RECALL_KS},
                **{f"delta@{k}": result.delta[k] for k in RECALL_KS},
                **{f"ci_low@{k}": result.ci_low[k] for k in RECALL_KS},
                **{f"ci_high@{k}": result.ci_high[k] for k in RECALL_KS},
                "improved@20": result.improved,
                "worsened@20": result.worsened,
            }
        )

    clean = next((entry for entry in report_slices if entry["name"] == "clean"), None)
    clean_delta = float(clean["delta@20"]) if clean else 0.0
    clean_gate = clean_delta >= -CLEAN_DROP_TOLERANCE
    report = EvaluationReport(
        backend=rewriter.name,
        corpus_version=corpus_version,
        index_version=index_version,
        n_cases=sum(entry["n"] for entry in report_slices),
        slices=report_slices,
        latency_ms=LatencySummary(
            p50=round(_percentile(latencies, 0.5), 3),
            p95=round(_percentile(latencies, 0.95), 3),
            mean=round(statistics.fmean(latencies), 3) if latencies else 0.0,
            note=latency_note,
        ),
        clean_gate_passed=clean_gate,
        clean_gate_delta=round(clean_delta, 6),
        bootstrap_samples=samples,
        seed=seed,
        notes=(
            "paired bootstrap over cases; recall_any@k against gold chunk ids; "
            "the same statistic t23 uses for its noisy-input ablation"
        ),
    )
    return report, dict(traces)


def write_report(
    report: EvaluationReport, out_dir: str | Path, *, traces: dict[str, Any] | None = None
) -> list[Path]:
    """Persist the JSON report, a markdown table and (optionally) per-case traces."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    json_path = target / "rewriter-eval.json"
    payload = report.model_dump(mode="json")
    if traces is not None:
        payload["traces"] = traces
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written.append(json_path)
    lines = [
        "# Query-rewriter retrieval comparison",
        "",
        f"backend: `{report.backend}` · cases: {report.n_cases} · "
        f"corpus: `{report.corpus_version}` · index: `{report.index_version}`",
        "",
        "| slice | n | recall@20 | recall@20 rewritten | Δ@20 | 95% CI | Δ@10 | 95% CI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in report.slices:
        lines.append(
            f"| {entry['name']} | {entry['n']} | {entry['recall@20']:.4f} | "
            f"{entry['recall@20_rewritten']:.4f} | {entry['delta@20']:+.4f} | "
            f"[{entry['ci_low@20']:+.4f}, {entry['ci_high@20']:+.4f}] | "
            f"{entry['delta@10']:+.4f} | "
            f"[{entry['ci_low@10']:+.4f}, {entry['ci_high@10']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            f"clean-slice non-regression gate (Δ@20 ≥ −{CLEAN_DROP_TOLERANCE}): "
            f"**{'passed' if report.clean_gate_passed else 'FAILED'}** "
            f"(Δ = {report.clean_gate_delta:+.4f})",
            "",
            f"latency per query: p50 {report.latency_ms.p50} ms, "
            f"p95 {report.latency_ms.p95} ms ({report.latency_ms.note})",
            "",
            f"bootstrap: {report.bootstrap_samples} resamples, seed {report.seed}",
            "",
        ]
    )
    markdown = target / "rewriter-eval.md"
    markdown.write_text("\n".join(lines), encoding="utf-8")
    written.append(markdown)
    return written


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "CLEAN_DROP_TOLERANCE",
    "RECALL_KS",
    "EvalCase",
    "EvaluationReport",
    "LatencySummary",
    "Searcher",
    "SliceResult",
    "evaluate_rewriter",
    "paired_bootstrap",
    "write_report",
]
