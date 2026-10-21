"""The evidence run (t11): ablations, cross-family judging, escalation, cache, cost/latency.

What this module produces, and why each piece exists:

* **retrieval ablation** — bm25 / dense / RRF(k sweep) / convex combination / +rerank, each a
  *shipped* :class:`~answerdesk.retrieval.pipeline.RetrievalPipeline`, scored with the §6.1
  metrics, per locale and macro, with paired bootstrap and McNemar between key pairs;
* **answer arms** — the generation pipeline runs with a real provider; every answered claim is
  judged by a *second, different-family* provider (§10.5). The judge identity and the
  cross-family verdict are recorded in every block;
* **escalation** — decisions come from t8's :class:`~answerdesk.policy.DecisionEngine`
  (conformal threshold ∩ cost bands ∩ guards), never from a hand-picked cutoff. The block
  reports a TP/FP/FN/TN confusion matrix, E-2 recall per unsafe category (unanswerable, PII,
  injection), E-4 coverage and the false-escalation rate on answerable items, the reason-code
  distribution, the full risk–coverage curve with an explicit operating point (max coverage
  with realised risk ≤ target on the held-out split) and a cost-ratio sensitivity table;
* **noisy slice** — when the corpus carries a noisy/free-form query category it gets its own
  metric block (retrieval, answer, escalation) and is never averaged into the headline;
* **cache on/off** — the semantic cache (t8) is consulted before generation; hits skip the
  model call, so the cache arm reports hit/wrong-hit counts and the measured latency/cost
  delta against the counterfactual of generating every turn;
* **operations** — per-stage p50/p95 wall clock, token counts and USD at the pinned pricing
  table, with the pricing snapshot (model → rate, retrieved_at) written next to the numbers.

Everything is written under ``reports/<run>/`` as JSON/CSV/PNG; nothing in the report is
hand-written. Offline (no key) the same code path runs with doubles and every affected block
is labelled ``replay``.

    # AT-EVL-08  # AT-EVL-10  # AT-NFR-REP-01  # AT-NFR-REP-02
"""

from __future__ import annotations

import contextlib
import json
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from answerdesk.config import Settings
from answerdesk.contracts import (
    Action,
    ChatProvider,
    EvalCase,
    IntentBucket,
    ReasonCode,
    RetrievedPassage,
    Split,
)
from answerdesk.eval.calibration import summarise
from answerdesk.eval.corpus import Corpus, load_corpus
from answerdesk.eval.datasets import load_golden
from answerdesk.eval.judge import PASS_SCORE, RubricJudge, judge_metadata
from answerdesk.eval.live import (
    answer_from_result,
    bucket_of,
    build_embedder,
    build_index,
    build_reranker,
    escalation_decision,
    judge_provider,
    retrieval_variants,
)
from answerdesk.eval.metrics import (
    answer_cells,
    escalation_cells,
    operational_cells,
    paired_scores,
    retrieval_cells,
    score_retrieval,
)
from answerdesk.eval.models import TurnRecord
from answerdesk.eval.stats import mcnemar, paired_bootstrap, wilson_interval
from answerdesk.generate.language import languages_agree
from answerdesk.generate.pipeline import AnswerPipeline
from answerdesk.llm.cost import PricingTable
from answerdesk.llm.factory import build_provider

#: Unsafe corpus categories that must be escalated (§8.5 E-2).
UNSAFE_CATEGORIES: tuple[str, ...] = ("unanswerable", "pii", "injection", "out_of_scope")
#: Corpus category that marks a noisy/free-form query (t22 corpus v2.1).
NOISY_CATEGORY = "noisy_freeform"
#: Cost ratios used for the sensitivity table (wrong : abstain).
COST_RATIOS: tuple[int, ...] = (2, 5, 10)


@dataclass
class BenchConfig:
    """Everything that changes between a smoke run and the reported evidence run."""

    run_id: str = "bench"
    golden_version: str = "v1"
    test_limit: int | None = None
    retrieval_limit: int | None = None
    #: Cases used for the (expensive, API-bound) rerank arm; other arms use the full split.
    rerank_limit: int | None = 80
    #: Retriever used by the generation arms. The hosted reranker is measured in the
    #: retrieval ablation; generation uses the fast convex-combination path because the
    #: free-tier rerank queue cannot sustain hundreds of sequential calls inside a run.
    #: The hosted embedding queue is rate-limited on the free tariff, so generation
    #: retrieves with BM25 (no API calls per case); the hybrid/dense arms are measured in
    #: the retrieval ablation, which needs far fewer calls.
    answer_retriever: str = "bm25"
    #: Judge the generated answers? False = judge-free arm (retrieval/escalation/latency/
    #: cost only), used when the cross-family judge's provider quota is exhausted.
    judge_enabled: bool = True
    dev_limit: int | None = None
    answer_limit: int | None = None
    locales: tuple[str, ...] = ()
    seed: int = 1337
    bootstrap: int = 1000
    target_risk: float = 0.02
    calibration_path: str | None = None
    index_dir: str = "data/index/bench"
    cache_dir: str = "data/index"
    retrieval_arms: tuple[str, ...] | None = None
    generator_arms: tuple[str, ...] = ("primary", "secondary")
    force_index: bool = False
    replay: bool = False
    #: Fit split-conformal q-hat on the dev split before evaluating on test (INV-09).
    calibrate_on_dev: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "golden_version": self.golden_version,
            "test_limit": self.test_limit,
            "dev_limit": self.dev_limit,
            "answer_limit": self.answer_limit,
            "locales": list(self.locales),
            "seed": self.seed,
            "bootstrap": self.bootstrap,
            "target_risk": self.target_risk,
            "retrieval_arms": list(self.retrieval_arms or ()),
            "generator_arms": list(self.generator_arms),
            "replay": self.replay,
        }


@dataclass
class BenchState:
    """Prepared, shared machinery: corpus, cases, indexes, engines, providers."""

    settings: Settings
    config: BenchConfig
    corpus: Corpus
    test_cases: list[EvalCase]
    dev_cases: list[EvalCase]
    index: Any = None
    embedder: Any = None
    reranker: Any = None
    variants: dict[str, Any] = field(default_factory=dict)
    engine: Any = None
    calibration_note: str = ""
    pricing: PricingTable | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def case_stratum(case: EvalCase) -> str:
    """Which dedicated block a case belongs to (``all`` + optional ``noisy``)."""
    corpus_category = str(case.metadata.get("corpus_category", ""))
    if corpus_category == NOISY_CATEGORY or case.metadata.get("noisy_freeform"):
        return "noisy"
    return "all"


def unsafe_kind(case: EvalCase) -> str:
    """Which must-escalate family a case belongs to (for the E-2 breakdown)."""
    corpus_category = str(case.metadata.get("corpus_category", ""))
    if corpus_category in UNSAFE_CATEGORIES:
        return corpus_category
    if not case.answerable:
        return str(case.metadata.get("subtype", "unsafe"))
    return "answerable"


def _subset(
    cases: Sequence[EvalCase], *, locales: Sequence[str], limit: int | None, seed: int
) -> list[EvalCase]:
    selected = [case for case in cases if not locales or case.locale in set(locales)]
    if limit is None or limit >= len(selected):
        return sorted(selected, key=lambda case: case.case_id)
    # deterministic, locale-stratified sample: round-robin over locales by hashed id
    by_locale: dict[str, list[EvalCase]] = {}
    for case in sorted(selected, key=lambda case: case.case_id):
        by_locale.setdefault(case.locale, []).append(case)
    picked: list[EvalCase] = []
    index = 0
    while len(picked) < limit:
        added = False
        for locale in sorted(by_locale):
            bucket = by_locale[locale]
            if index < len(bucket) and len(picked) < limit:
                picked.append(bucket[index])
                added = True
        if not added:
            break
        index += 1
    return picked


def _retrieval_ranked(trace: Any, k: int) -> list[str]:
    """The ranked chunk ids the pipeline produced, from the best available stage."""
    for stage in (
        "reranked",
        "diversity" if hasattr(trace, "diversity") else "",
        "fused",
        "candidates",
    ):
        passages = getattr(trace, stage, None)
        if passages:
            return [passage.chunk.chunk_id for passage in passages][:k]
    return []


def _packed_passages(trace: Any) -> list[RetrievedPassage]:
    for stage in ("packed", "reranked", "fused", "candidates"):
        passages = getattr(trace, stage, None)
        if passages:
            return list(passages)
    return []


# --------------------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------------------


def prepare(settings: Settings, config: BenchConfig, *, evals_root: Path) -> BenchState:
    corpus = load_corpus(settings.resolve_path(settings.app.corpus_dir))
    cases, _manifest, split = load_golden(
        evals_root, corpus=corpus, version=config.golden_version, strict=False
    )
    test_cases = [case for case in cases if case.split is Split.TEST]
    dev_cases = [case for case in cases if case.split is Split.DEV]
    state = BenchState(
        settings=settings,
        config=config,
        corpus=corpus,
        test_cases=test_cases,
        dev_cases=dev_cases,
        pricing=_pricing(settings),
    )
    state.notes.append(f"split {split.version} {split.sha256[:12]} corpus {corpus.version}")
    if config.replay:
        from answerdesk.testing.doubles import HashingEmbedder

        state.embedder = HashingEmbedder()
        state.notes.append("REPLAY: hashing embedder, no hosted models")
    else:
        state.embedder = build_embedder(settings, Path(config.cache_dir))
    state.index = build_index(
        settings, corpus, state.embedder, Path(config.index_dir), force=config.force_index
    )
    if not config.replay:
        state.reranker = build_reranker(settings)
    state.variants = retrieval_variants(settings, state.index, reranker=state.reranker)
    state.engine = _load_engine(settings, config, corpus, state)
    return state


def _pricing(settings: Settings) -> PricingTable:
    try:
        return PricingTable.default()
    except Exception:
        return PricingTable.from_entries([])


def _load_engine(settings: Settings, config: BenchConfig, corpus: Corpus, state: BenchState) -> Any:
    """t8's DecisionEngine from a sealed calibration artifact (never a hand-picked cutoff)."""
    if not config.calibration_path:
        state.calibration_note = (
            "no calibration artifact supplied: escalation decisions fall back to the "
            "configured cost bands only (diagnostic, not the shipped conformal path)"
        )
        return None
    from answerdesk.policy import CalibrationArtifact, DecisionEngine

    artifact = CalibrationArtifact.load(config.calibration_path)
    engine = DecisionEngine.from_artifact(artifact, corpus_version=corpus.version)
    state.calibration_note = (
        f"DecisionEngine from artifact {artifact.artifact_hash[:12]} "
        f"(corpus {artifact.corpus_version}, scorer {artifact.scorer_name} "
        f"{artifact.scorer_version})"
    )
    return engine


# --------------------------------------------------------------------------------------
# Retrieval ablation
# --------------------------------------------------------------------------------------


def run_retrieval(
    state: BenchState, *, out_dir: Path, cases: Sequence[EvalCase] | None = None
) -> dict[str, Any]:
    from answerdesk.policy import OverlapScorer

    scorer = OverlapScorer()
    selected = list(
        cases
        if cases is not None
        else _subset(
            state.test_cases,
            locales=state.config.locales,
            limit=state.config.retrieval_limit
            if state.config.retrieval_limit is not None
            else state.config.test_limit,
            seed=state.config.seed,
        )
    )
    arms = state.config.retrieval_arms or tuple(state.variants)
    results: dict[str, Any] = {"arms": {}, "n": len(selected)}
    all_turns: dict[str, list[TurnRecord]] = {}
    degraded = 0
    for arm_name in arms:
        pipeline = state.variants.get(arm_name)
        if pipeline is None:
            continue
        fallback = state.variants.get("cc@0.75") if arm_name.endswith("+rerank") else None
        turns: list[TurnRecord] = []
        arm_cases = list(selected)
        if arm_name.endswith("+rerank") and state.config.rerank_limit:
            arm_cases = _subset(
                selected,
                locales=state.config.locales,
                limit=state.config.rerank_limit,
                seed=state.config.seed,
            )
        for case in arm_cases:
            started = time.perf_counter()
            try:
                trace = pipeline.retrieve(
                    case.question, final_k=state.settings.generate.context_k_max
                )
            except Exception:  # a degraded case is counted in the artefacts, never hidden
                if fallback is None:
                    raise
                degraded += 1
                trace = fallback.retrieve(
                    case.question, final_k=state.settings.generate.context_k_max
                )
            elapsed = (time.perf_counter() - started) * 1000.0
            ranked = _retrieval_ranked(trace, state.settings.generate.context_k_max)
            score = float(scorer.score(case.question, _packed_passages(trace)))
            turns.append(
                TurnRecord(
                    case_id=case.case_id,
                    locale=case.locale,
                    category=case.category,
                    intent_bucket=case.intent_bucket,
                    split=case.split,
                    arm=arm_name,
                    answerable=case.answerable,
                    must_escalate=not case.answerable,
                    expected_language=case.locale,
                    action=Action.ANSWER,
                    reason_code=ReasonCode.ANSWERED,
                    answered=True,
                    escalated=False,
                    ranked_chunk_ids=ranked,
                    gold_chunk_ids=list(case.gold_chunk_ids),
                    acceptable_chunk_ids=list(case.gold_chunk_ids),
                    required_citation_chunk_ids=list(case.required_citation_chunk_ids),
                    language=case.locale,
                    confidence=min(1.0, max(0.0, score)),
                    latency_ms={"retrieval": elapsed, "total": elapsed},
                )
            )
        all_turns[arm_name] = turns
        cells = retrieval_cells(
            turns,
            ks=(10, 20),
            arm=arm_name,
            seed=state.config.seed,
            samples=state.config.bootstrap,
        )
        noisy = [
            turn
            for turn, case in zip(turns, arm_cases, strict=True)
            if case_stratum(case) == "noisy"
        ]
        results["arms"][arm_name] = {
            "n_labeled": sum(1 for turn in turns if turn.gold_chunk_ids),
            "cells": [cell.model_dump(mode="json") for cell in cells],
            "mean_latency_ms": round(
                sum(turn.latency_ms["retrieval"] for turn in turns) / max(1, len(turns)), 3
            ),
            "noisy_cells": (
                [
                    cell.model_dump(mode="json")
                    for cell in retrieval_cells(
                        noisy,
                        ks=(10, 20),
                        arm=f"{arm_name}:noisy",
                        seed=state.config.seed,
                        samples=state.config.bootstrap,
                    )
                ]
                if noisy
                else []
            ),
        }
        _write_rows(
            out_dir / "retrieval" / f"{_slug(arm_name)}.csv",
            [
                {
                    "metric": cell.name,
                    "locale": cell.locale,
                    "value": cell.value,
                    "ci_low": cell.ci_low,
                    "ci_high": cell.ci_high,
                    "n": cell.n,
                }
                for cell in cells
            ],
        )
    comparisons = []
    pairs = [
        ("bm25", "dense"),
        ("bm25", "rrf@60"),
        ("rrf@60", "cc@0.75"),
        ("cc@0.75", "cc@0.75+rerank"),
    ]
    for left, right in pairs:
        if left in all_turns and right in all_turns:
            comparisons.append(
                _compare(all_turns[left], all_turns[right], state.config, left, right)
            )
    results["comparisons"] = comparisons
    results["degraded_cases"] = degraded
    results["per_language"] = _per_language_retrieval(all_turns)
    _write_json(out_dir / "retrieval" / "results.json", results)
    _plot_retrieval(results, out_dir / "retrieval" / "recall_by_arm.png")
    return results


def _compare(
    left: Sequence[TurnRecord],
    right: Sequence[TurnRecord],
    config: BenchConfig,
    left_name: str,
    right_name: str,
    k: int = 10,
) -> dict[str, Any]:
    left_scores = paired_scores(left, k)
    right_scores = paired_scores(right, k)
    common = sorted(set(left_scores.case_ids) & set(right_scores.case_ids))
    left_map = dict(zip(left_scores.case_ids, left_scores.any_hit, strict=True))
    right_map = dict(zip(right_scores.case_ids, right_scores.any_hit, strict=True))
    vector_a = [left_map[case_id] for case_id in common]
    vector_b = [right_map[case_id] for case_id in common]
    difference = paired_bootstrap(vector_a, vector_b, samples=config.bootstrap, seed=config.seed)
    b = sum(1 for a, b_value in zip(vector_a, vector_b, strict=True) if a > b_value)
    c = sum(1 for a, b_value in zip(vector_a, vector_b, strict=True) if a < b_value)
    test = mcnemar(b, c)
    return {
        "metric": f"recall_any@{k}",
        "arm_a": left_name,
        "arm_b": right_name,
        "n": len(common),
        "mean_a": round(difference.mean_a, 6),
        "mean_b": round(difference.mean_b, 6),
        "diff": round(difference.diff, 6),
        "ci_low": round(difference.low, 6),
        "ci_high": round(difference.high, 6),
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_p": round(test.p_value, 6),
        "mcnemar_method": test.method,
        "significant": difference.excludes_zero,
    }


def _per_language_retrieval(all_turns: Mapping[str, Sequence[TurnRecord]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, turns in all_turns.items():
        for locale in sorted({turn.locale for turn in turns}):
            subset = [turn for turn in turns if turn.locale == locale and turn.gold_chunk_ids]
            if not subset:
                continue
            scores = [score_retrieval(turn, 10) for turn in subset]
            hits = [score.any_hit for score in scores if score is not None]
            interval = wilson_interval(sum(hits), len(hits))
            rows.append(
                {
                    "arm": arm,
                    "locale": locale,
                    "recall_any@10": round(interval.value, 6),
                    "ci_low": round(interval.low, 6),
                    "ci_high": round(interval.high, 6),
                    "n": len(hits),
                }
            )
    return rows


def _plot_retrieval(results: Mapping[str, Any], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(results["arms"])
    macro = []
    for arm in arms:
        cells = results["arms"][arm]["cells"]
        value = next(
            (
                cell["value"]
                for cell in cells
                if cell["name"] == "recall_any@10" and cell["locale"] == "macro"
            ),
            0.0,
        )
        macro.append(value)
    figure, axis = plt.subplots(figsize=(max(6, len(arms) * 1.1), 4.0))
    axis.bar(range(len(arms)), macro, color="#3355aa")
    axis.set_xticks(range(len(arms)))
    axis.set_xticklabels(arms, rotation=25, ha="right")
    axis.set_ylabel("macro recall_any@10")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3)
    axis.set_title("Retrieval ablation (frozen test split)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------------
# Answer arms: generation + cross-family judging + policy decisions
# --------------------------------------------------------------------------------------


def _provider_for_arm(state: BenchState, arm: str) -> tuple[ChatProvider, str]:
    """Generator provider for an arm: the primary profile, or the secondary comparison one."""
    if state.config.replay:
        from answerdesk.testing.doubles import FakeProvider

        return FakeProvider(), "fake-model"
    settings = state.settings
    if arm != "secondary":
        return build_provider(settings), settings.resolved_provider().model
    preset_name = settings.eval.secondary_preset or "deepseek"
    from answerdesk.config import PROVIDER_PRESETS

    preset = PROVIDER_PRESETS[preset_name]
    model = settings.eval.secondary_model or preset.model
    api_key = settings.eval.secondary_api_key
    if api_key is None and settings.eval.secondary_api_key_env:
        import os

        value = os.environ.get(settings.eval.secondary_api_key_env, "")
        if value:
            from pydantic import SecretStr

            api_key = SecretStr(value)
    secondary = settings.model_copy(deep=True)
    secondary.llm.preset = preset_name
    secondary.llm.base_url = settings.eval.secondary_base_url or preset.base_url
    secondary.llm.model = model
    secondary.llm.api_key = api_key
    secondary.llm.timeout_s = settings.eval.secondary_timeout_s
    secondary.llm.max_tokens = settings.eval.secondary_max_tokens
    from answerdesk.eval.live import SchemaFallbackProvider

    return SchemaFallbackProvider(build_provider(secondary)), model


def _judge_for_arm(state: BenchState, generator_model: str) -> tuple[Any, Any, str]:
    """Cross-family judge (§10.5): its own provider, its own model id, its own metadata."""
    if state.config.replay:
        from answerdesk.testing.doubles import FakeJudge

        judge: Any = FakeJudge()
        metadata = judge_metadata(
            judge_model=judge.judge_model,
            prompt_version="none (deterministic double)",
            rubric_version=judge.rubric_version,
            generator_model=generator_model,
        )
        return judge, metadata, judge.judge_model
    settings = state.settings
    provider = judge_provider(settings)
    if provider is None:
        provider = build_provider(settings)
        judge_model = settings.resolved_provider().model
    else:
        judge_model = settings.eval.judge_model
    judge = RubricJudge(provider, judge_model=judge_model)
    metadata = judge_metadata(
        judge_model=judge_model,
        prompt_version=judge.prompt_version,
        rubric_version=judge.rubric_version,
        generator_model=generator_model,
        judge_family=settings.eval.judge_family,
    )
    return judge, metadata, judge_model


def run_answers(
    state: BenchState,
    *,
    arm: str,
    out_dir: Path,
    cases: Sequence[EvalCase] | None = None,
) -> tuple[dict[str, Any], list[TurnRecord], list[EvalCase]]:
    """Generate, judge and decide for one generator arm; writes cells and raw turns."""
    settings = state.settings
    selected = list(
        cases
        if cases is not None
        else _subset(
            state.test_cases,
            locales=state.config.locales,
            limit=state.config.answer_limit,
            seed=state.config.seed,
        )
    )
    provider, generator_model = _provider_for_arm(state, arm)
    judge, judge_meta, judge_model = _judge_for_arm(state, generator_model)
    pipeline = AnswerPipeline.from_settings(provider, settings)
    retriever = state.variants.get(state.config.answer_retriever) or state.variants.get("cc@0.75")
    turns: list[TurnRecord] = []
    rows: list[dict[str, Any]] = []
    escalated = 0
    judge_failures = 0
    for case in selected:
        record, row = _measure_case(
            state=state,
            case=case,
            arm=arm,
            pipeline=pipeline,
            retriever=retriever,
            judge=judge,
            generator_model=generator_model,
            judge_model=judge_model,
        )
        turns.append(record)
        rows.append(row)
        escalated += 1 if record.escalated else 0
        judge_failures += 1 if row.get("judge_failed") else 0
    cells = (
        answer_cells(
            turns, arm=arm, seed=state.config.seed, samples=state.config.bootstrap, judged=True
        )
        + escalation_cells(turns, arm=arm)
        + operational_cells(turns, arm=arm)
    )
    noisy_pairs = [
        (turn, case)
        for turn, case in zip(turns, selected, strict=True)
        if case_stratum(case) == "noisy"
    ]
    noisy_cells = (
        [
            cell.model_dump(mode="json")
            for cell in (
                answer_cells(
                    [turn for turn, _ in noisy_pairs],
                    arm=f"{arm}:noisy",
                    seed=state.config.seed,
                    samples=state.config.bootstrap,
                    judged=True,
                )
                + escalation_cells([turn for turn, _ in noisy_pairs], arm=f"{arm}:noisy")
            )
        ]
        if noisy_pairs
        else []
    )
    escalations = [turn for turn in turns if turn.escalated]
    judged = sum(1 for row in rows if row.get("claims"))
    failure_rate = (judge_failures / judged) if judged else 0.0
    judge_payload = judge_meta.model_dump(mode="json")
    if failure_rate > 0.10:
        judge_payload["reportable"] = False
        judge_payload["note"] = (
            f"{judge_payload.get('note', '')} — {judge_failures}/{judged} judge calls failed "
            f"({failure_rate:.1%}); judged metrics for this arm are diagnostic"
        )
    payload = {
        "arm": arm,
        "judge_failure_rate": round(failure_rate, 6),
        "generator_model": generator_model,
        "judge_model": judge_model,
        "judge": judge_payload,
        "judge_failures": judge_failures,
        "replay": state.config.replay,
        "n": len(turns),
        "escalated": escalated,
        "cells": [cell.model_dump(mode="json") for cell in cells],
        "noisy_cells": noisy_cells,
        "noisy_n": len(noisy_pairs),
        "per_language": _per_language_answer(turns),
        "reason_codes": _reason_codes(escalations),
        "confusion": _confusion(turns, selected),
        "calibration": summarise(
            [turn.confidence for turn in turns],
            [turn.answer_ok for turn in turns],
            target_risk=state.config.target_risk,
        ).to_dict(),
    }
    arm_dir = out_dir / "answer" / _slug(arm)
    noisy_rows = [
        row for row, case in zip(rows, selected, strict=True) if case_stratum(case) == "noisy"
    ]
    if noisy_rows:
        _write_rows(arm_dir / "noisy-per-example.csv", noisy_rows)
    _write_json(arm_dir / "cells.json", payload)
    _write_rows(arm_dir / "turns.csv", rows)
    _write_rows(arm_dir / "per_language.csv", payload["per_language"])
    _write_rows(arm_dir / "reason_codes.csv", payload["reason_codes"])
    _write_json(arm_dir / "confusion.json", payload["confusion"])
    return payload, turns, selected


def _measure_case(
    *,
    state: BenchState,
    case: EvalCase,
    arm: str,
    pipeline: Any,
    retriever: Any,
    judge: Any,
    generator_model: str,
    judge_model: str,
) -> tuple[TurnRecord, dict[str, Any]]:
    """One case: retrieve → calibrate/decide → (answerable?) generate → judge.

    Generation runs for every *answerable* case even when the policy escalates it: the
    risk–coverage curve needs to know whether answering would have been correct, and
    computing that only for the turns the policy already accepted would bias the curve
    towards the policy under test. Unsafe cases are never generated — by definition any
    answer to them is an error.
    """
    from answerdesk.policy import OverlapScorer

    settings = state.settings
    timings: dict[str, float] = {}
    anchor = time.perf_counter()
    trace = retriever.retrieve(case.question, final_k=settings.generate.context_k_max)
    passages = _packed_passages(trace)
    ranked = _retrieval_ranked(trace, settings.generate.context_k_max)
    timings["retrieval"] = (time.perf_counter() - anchor) * 1000.0
    p_correct = float(OverlapScorer().score(case.question, passages))
    bucket = case.intent_bucket or bucket_of(str(case.metadata.get("intent", "")))
    policy: Any = None
    if state.engine is not None:
        anchor = time.perf_counter()
        policy = state.engine.evaluate(
            p_correct=p_correct,
            bucket=bucket,
            query=case.question,
            passages=passages,
            intent_label=str(case.metadata.get("intent", "")),
        )
        timings["policy"] = (time.perf_counter() - anchor) * 1000.0
    decision = (
        policy.decision
        if policy is not None
        else escalation_decision(
            action=Action.ANSWER,
            reason_code=ReasonCode.ANSWERED,
            score=p_correct,
            alpha=0.05,
            threshold_ref="no-artifact",
        )
    )
    result: Any = None
    cost = {"input_tokens": 0, "output_tokens": 0, "usd": 0.0}
    answer_text = ""
    if case.answerable and passages:
        anchor = time.perf_counter()
        result = pipeline.answer(case.question, passages)
        timings["generation"] = (time.perf_counter() - anchor) * 1000.0
        answer_text = result.text
        cost = {
            "input_tokens": int(result.usage.input_tokens),
            "output_tokens": int(result.usage.output_tokens),
            "usd": result.usage.usd,
        }
    claims_total = len(result.claims) if result is not None else 0
    claims_supported = (
        sum(1 for claim in result.claims if claim.support.value == "supported")
        if result is not None
        else 0
    )
    cited = sorted({citation.chunk_id for citation in result.citations}) if result else []
    acceptable = set(case.gold_chunk_ids)
    evidence_cited = (
        bool(set(cited) & acceptable)
        if acceptable
        else bool(set(cited) & set(case.required_citation_chunk_ids))
    )
    language = result.language if result is not None else case.locale
    language_match = (
        languages_agree(case.locale, result.text)
        if result is not None and result.text.strip()
        else True
    )
    rubric_score: float | None = None
    rubric_pass: bool | None = None
    judge_failed = False
    if result is not None and result.claims and state.config.judge_enabled:
        anchor = time.perf_counter()
        by_id = {citation.citation_id: citation for citation in result.citations}
        scores = []
        from answerdesk.eval.judge import JudgeError

        for claim in result.claims:
            quotes = [by_id[cid].quote for cid in claim.citation_ids if cid in by_id]
            try:
                score = judge.grade(
                    question=case.question,
                    reference_answer=(
                        case.reference_answer if case.metadata.get("reference_verifiable") else None
                    ),
                    candidate_answer=claim.text,
                    context=quotes,
                )
                scores.append(score.score)
            except JudgeError:
                # A judge failure is data, not a crash: it is counted per arm and, above a
                # threshold, downgrades the arm's judged metrics to diagnostic.
                judge_failed = True
        timings["judge"] = (time.perf_counter() - anchor) * 1000.0
        if scores:
            rubric_score = round(sum(scores) / len(scores), 6)
            rubric_pass = rubric_score >= PASS_SCORE
    policy_answered = decision.action is Action.ANSWER and result is not None
    quality_ok = bool(
        case.answerable
        and result is not None
        and result.signal.action is Action.ANSWER
        and claims_total > 0
        and claims_supported == claims_total
        and evidence_cited
        and language_match
        and (rubric_pass is not False)
    )
    answer_ok = bool(case.answerable and policy_answered and quality_ok)
    escalated = decision.action in (Action.ESCALATE, Action.REFUSE) or not policy_answered
    timings["total"] = sum(value for key, value in timings.items() if key != "total")
    record = TurnRecord(
        case_id=case.case_id,
        locale=case.locale,
        category=case.category,
        intent_bucket=bucket,
        split=case.split,
        arm=arm,
        answerable=case.answerable,
        must_escalate=not case.answerable,
        expected_language=case.locale,
        action=decision.action,
        reason_code=decision.reason_code,
        answered=not escalated,
        escalated=escalated,
        ranked_chunk_ids=ranked,
        gold_chunk_ids=list(case.gold_chunk_ids),
        acceptable_chunk_ids=sorted(acceptable),
        required_citation_chunk_ids=list(case.required_citation_chunk_ids),
        evidence_cited=evidence_cited,
        claims_total=claims_total,
        claims_supported=claims_supported,
        groundedness=(claims_supported / claims_total) if claims_total else None,
        citation_recall=(claims_supported / claims_total) if claims_total else None,
        cited_chunk_ids=cited,
        rubric_score=rubric_score,
        rubric_pass=rubric_pass,
        language=language,
        language_match=language_match,
        answer_ok=answer_ok,
        confidence=min(1.0, max(0.0, p_correct)),
        latency_ms=timings,
        usd=float(cost["usd"]),
        input_tokens=int(cost["input_tokens"]),
        output_tokens=int(cost["output_tokens"]),
    )
    row = {
        "case_id": case.case_id,
        "question": case.question,
        "locale": case.locale,
        "category": case.category.value,
        "corpus_category": str(case.metadata.get("corpus_category", "")),
        "stratum": case_stratum(case),
        "unsafe_kind": unsafe_kind(case),
        "answerable": case.answerable,
        "action": decision.action.value,
        "reason_code": decision.reason_code.value,
        "p_correct": round(p_correct, 6),
        "answered": not escalated,
        "answer_ok": answer_ok,
        "quality_ok": quality_ok,
        "rubric_score": rubric_score,
        "claims": claims_total,
        "cited": ";".join(cited),
        "gold": ";".join(case.gold_chunk_ids),
        "evidence_cited": evidence_cited,
        "language_match": language_match,
        "latency_retrieval_ms": round(timings.get("retrieval", 0.0), 3),
        "latency_generation_ms": round(timings.get("generation", 0.0), 3),
        "latency_judge_ms": round(timings.get("judge", 0.0), 3),
        "judge_failed": judge_failed,
        "input_tokens": cost["input_tokens"],
        "output_tokens": cost["output_tokens"],
        "usd": cost["usd"],
        "answer_head": answer_text[:160].replace("\n", " "),
    }
    _ = (generator_model, judge_model)
    return record, row


def _reason_codes(turns: Sequence[TurnRecord]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    by_kind: dict[str, dict[str, int]] = {}
    for turn in turns:
        code = turn.reason_code.value
        counts[code] = counts.get(code, 0) + 1
    total = sum(counts.values()) or 1
    rows = [
        {"reason_code": code, "count": count, "share": round(count / total, 6)}
        for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    _ = by_kind
    return rows


def _confusion(turns: Sequence[TurnRecord], cases: Sequence[EvalCase]) -> dict[str, Any]:
    """TP/FP/FN/TN against the definition "must escalate = the case is not answerable"."""
    tp = fp = fn = tn = 0
    for turn, case in zip(turns, cases, strict=True):
        predicted = turn.escalated
        actual = not case.answerable
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    total = tp + fp + fn + tn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": total,
        "precision": round(tp / (tp + fp), 6) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 6) if (tp + fn) else None,
        "false_escalation_rate": round(fp / (fp + tn), 6) if (fp + tn) else None,
        "coverage": round(tn / (fp + tn), 6) if (fp + tn) else None,
        "specificity": round(tn / (fp + tn), 6) if (fp + tn) else None,
    }


def _version_tuple(state: BenchState, model: str) -> Any:
    """Cache key versions (t8 ``VersionTuple``): anything that changes an answer invalidates."""
    from answerdesk.cache import VersionTuple

    return VersionTuple(
        corpus_version=state.corpus.version,
        index_version=getattr(state.index, "fingerprint", "") or "index-bench",
        prompt_version="grounded-answer-1.0.0",
        model_id=model,
        policy_artifact_hash=(
            getattr(state.engine, "artifact_hash", "") if state.engine is not None else "none"
        ),
    )


def _per_language_answer(turns: Sequence[TurnRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for locale in sorted({turn.locale for turn in turns}):
        subset = [turn for turn in turns if turn.locale == locale]
        answered = [turn for turn in subset if turn.answered]
        answerable = [turn for turn in subset if turn.answerable]
        must = [turn for turn in subset if turn.must_escalate]
        coverage = wilson_interval(sum(1 for turn in answerable if turn.answered), len(answerable))
        recall = wilson_interval(sum(1 for turn in must if turn.escalated), len(must))
        correct = wilson_interval(sum(1 for turn in answered if turn.answer_ok), len(answered))
        rows.append(
            {
                "locale": locale,
                "n": len(subset),
                "answered": len(answered),
                "coverage": round(coverage.value, 6),
                "coverage_ci": f"[{coverage.low:.3f},{coverage.high:.3f}]",
                "must_escalate_recall": round(recall.value, 6),
                "must_escalate_ci": f"[{recall.low:.3f},{recall.high:.3f}]",
                "answer_accuracy": round(correct.value, 6),
                "answer_accuracy_ci": f"[{correct.low:.3f},{correct.high:.3f}]",
            }
        )
    return rows


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^0-9a-zA-Z._-]+", "-", value).strip("-") or "x"


# --------------------------------------------------------------------------------------
# Escalation: per-category recall, reason codes, risk–coverage, operating point
# --------------------------------------------------------------------------------------


def _curve(turns: Sequence[TurnRecord]) -> list[dict[str, Any]]:
    """Risk–coverage sweep over p̂ (higher = the system is more willing to answer)."""
    ordered = sorted(turns, key=lambda turn: (-turn.confidence, turn.case_id))
    errors = 0
    points: list[dict[str, Any]] = []
    for answered, turn in enumerate(ordered, start=1):
        if not turn.answer_ok:
            errors += 1
        points.append(
            {
                "coverage": round(answered / len(ordered), 6),
                "risk": round(errors / answered, 6),
                "n_answered": answered,
                "threshold": round(turn.confidence, 6),
            }
        )
    return points


def select_operating_threshold(dev_turns: Sequence[TurnRecord], target: float) -> dict[str, Any]:
    """Choose the answering threshold on the **dev/calibration** turns only.

    Picking the last curve point with risk ≤ target *on the test split* would fit the
    threshold to the reported data (multiple comparisons over thresholds) and report an
    optimistically biased coverage. The threshold is therefore selected on the split the
    conformal quantile is fitted on, and the test split only ever reports the result of
    applying that fixed threshold.
    """
    curve = _curve(dev_turns)
    chosen: dict[str, Any] | None = None
    min_risk_point = min(curve, key=lambda point: point["risk"]) if curve else None
    for point in curve:
        if point["risk"] <= target:
            chosen = point
    if chosen is None:
        return {
            "source": (
                "no feasible point at the target risk on the calibration split; the "
                "conformal engine's own threshold is the shipped rule"
            ),
            "threshold": 1.0,
            "dev_coverage": 0.0,
            "dev_risk": 0.0,
            "dev_n": len(dev_turns),
            "dev_min_risk": min_risk_point["risk"] if min_risk_point else None,
            "dev_coverage_at_min_risk": min_risk_point["coverage"] if min_risk_point else None,
            "feasible": False,
        }
    return {
        "source": "dev-selected threshold (risk-constrained on the calibration split)",
        "threshold": chosen["threshold"],
        "dev_coverage": chosen["coverage"],
        "dev_risk": chosen["risk"],
        "dev_n": len(dev_turns),
        "dev_min_risk": min_risk_point["risk"] if min_risk_point else None,
        "dev_coverage_at_min_risk": min_risk_point["coverage"] if min_risk_point else None,
        "feasible": True,
    }


def run_escalation(
    *,
    state: BenchState,
    turns: Sequence[TurnRecord],
    cases: Sequence[EvalCase],
    out_dir: Path,
    arm: str,
    dev_operating: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The escalation block the owner asked for: both directions, both readings of E-3.

    Two operating points are reported and never conflated:

    * **headline** — the threshold selected on the dev/calibration turns, applied unchanged
      to the held-out test turns (what the system would actually do);
    * **oracle upper bound** — the best point on the *test* curve. It is labelled
      diagnostic, excluded from the headline table, and exists only to bound the gap.
    """
    target = state.config.target_risk
    curve = _curve(turns)
    oracle = None
    for point in curve:
        if point["risk"] <= target:
            oracle = point
    operating: dict[str, Any] | None = None
    if dev_operating is not None:
        threshold = float(dev_operating.get("threshold", 1.0))
        answered_at_threshold = [turn for turn in turns if turn.confidence >= threshold]
        wrong = sum(1 for turn in answered_at_threshold if not turn.answer_ok)
        coverage_interval = wilson_interval(len(answered_at_threshold), len(turns))
        risk_interval = (
            wilson_interval(wrong, len(answered_at_threshold)) if answered_at_threshold else None
        )
        nearest = min(curve, key=lambda point: abs(point["threshold"] - threshold))
        operating = {
            "source": dev_operating.get("source", "dev-selected"),
            "threshold": round(threshold, 6),
            "coverage": round(coverage_interval.value, 6),
            "coverage_ci": f"[{coverage_interval.low:.3f},{coverage_interval.high:.3f}]",
            "risk": round(risk_interval.value, 6) if risk_interval else None,
            "risk_ci": (
                f"[{risk_interval.low:.3f},{risk_interval.high:.3f}]" if risk_interval else "n/a"
            ),
            "n_answered": len(answered_at_threshold),
            "dev_coverage": dev_operating.get("dev_coverage"),
            "dev_risk": dev_operating.get("dev_risk"),
            "curve_point": nearest,
        }
    else:
        operating = dict(oracle) if oracle else None
        if operating is not None:
            operating["source"] = "test-curve point (no dev calibration available — diagnostic)"
    # per unsafe family (E-2) and per language
    families: list[dict[str, Any]] = []
    for kind in sorted({unsafe_kind(case) for case in cases}):
        pairs = [
            (turn, case)
            for turn, case in zip(turns, cases, strict=True)
            if unsafe_kind(case) == kind
        ]
        if not pairs:
            continue
        caught = sum(1 for turn, _ in pairs if turn.escalated)
        interval = wilson_interval(caught, len(pairs))
        families.append(
            {
                "family": kind,
                "n": len(pairs),
                "escalated": caught,
                "recall": round(interval.value, 6),
                "ci_low": round(interval.low, 6),
                "ci_high": round(interval.high, 6),
            }
        )
    languages: list[dict[str, Any]] = []
    for locale in sorted({turn.locale for turn in turns}):
        pairs = [
            (turn, case) for turn, case in zip(turns, cases, strict=True) if case.locale == locale
        ]
        answerable = [(turn, case) for turn, case in pairs if case.answerable]
        unsafe = [(turn, case) for turn, case in pairs if not case.answerable]
        coverage = wilson_interval(
            sum(1 for turn, _ in answerable if turn.answered), len(answerable)
        )
        recall = wilson_interval(sum(1 for turn, _ in unsafe if turn.escalated), len(unsafe))
        languages.append(
            {
                "locale": locale,
                "n": len(pairs),
                "coverage": round(coverage.value, 6),
                "coverage_ci": f"[{coverage.low:.3f},{coverage.high:.3f}]",
                "must_escalate_recall": round(recall.value, 6),
                "must_escalate_ci": f"[{recall.low:.3f},{recall.high:.3f}]",
            }
        )
    # E-3: the gated reading counts every escalation; the literal reading only counts
    # escalations that a *safety* guard (not a confidence band) produced.
    escalated_pairs = [
        (turn, case) for turn, case in zip(turns, cases, strict=True) if turn.escalated
    ]
    tp = sum(1 for turn, case in escalated_pairs if not case.answerable)
    safety_reasons = {
        ReasonCode.PII_DETECTED.value,
        ReasonCode.INJECTION_SUSPECTED.value,
        ReasonCode.HIGH_RISK_INTENT.value,
        ReasonCode.OUT_OF_SCOPE.value,
    }
    safety_only = [
        (turn, case)
        for turn, case in escalated_pairs
        if turn.reason_code.value in safety_reasons or not case.answerable
    ]
    tp_literal = sum(1 for turn, case in safety_only if not case.answerable)
    precision_gated = wilson_interval(tp, len(escalated_pairs)) if escalated_pairs else None
    precision_literal = wilson_interval(tp_literal, len(safety_only)) if safety_only else None
    # cost-ratio sensitivity: c_wrong = 1, c_abstain = 1 / ratio
    sensitivity: list[dict[str, Any]] = []
    for ratio in COST_RATIOS:
        threshold = 1.0 - (1.0 / ratio)
        answered = [turn for turn in turns if turn.confidence >= threshold]
        wrong = sum(1 for turn in answered if not turn.answer_ok)
        coverage = wilson_interval(len(answered), len(turns))
        risk = wilson_interval(wrong, len(answered)) if answered else None
        sensitivity.append(
            {
                "cost_ratio_wrong_to_abstain": f"{ratio}:1",
                "threshold": round(threshold, 6),
                "coverage": round(coverage.value, 6),
                "coverage_ci": f"[{coverage.low:.3f},{coverage.high:.3f}]",
                "realised_risk": round(risk.value, 6) if risk else None,
                "risk_ci": f"[{risk.low:.3f},{risk.high:.3f}]" if risk else "n/a",
                "n_answered": len(answered),
            }
        )
    actions: dict[str, int] = {}
    for turn in turns:
        actions[turn.action.value] = actions.get(turn.action.value, 0) + 1
    payload = {
        "arm": arm,
        "target_risk": target,
        "n": len(turns),
        "actions": actions,
        "definition": (
            "not auto-answered = action is escalate, refuse or clarify; the confusion matrix "
            "uses that definition for 'predicted escalation'"
        ),
        "confusion": _confusion(turns, cases),
        "families": families,
        "languages": languages,
        "reason_codes": _reason_codes([turn for turn in turns if turn.escalated]),
        "curve": curve,
        "operating_point": operating,
        "oracle_upper_bound": (
            {
                "coverage": oracle["coverage"],
                "risk": oracle["risk"],
                "threshold": oracle["threshold"],
                "n_answered": oracle["n_answered"],
                "label": "oracle upper bound (diagnostic, not the shipped operating point)",
            }
            if oracle is not None
            else None
        ),
        "cost_sensitivity": sensitivity,
        "policy_variants": _policy_table(turns),
        "e3": {
            "gated_precision": round(precision_gated.value, 6) if precision_gated else None,
            "gated_ci": (
                f"[{precision_gated.low:.3f},{precision_gated.high:.3f}]"
                if precision_gated
                else "n/a"
            ),
            "late": "the gated reading counts every escalation in the denominator",
            "literal_precision": round(precision_literal.value, 6) if precision_literal else None,
            "literal_ci": (
                f"[{precision_literal.low:.3f},{precision_literal.high:.3f}]"
                if precision_literal
                else "n/a"
            ),
            "note": (
                "literal reading counts only escalations produced by a safety guard; it is "
                "the stricter reading of E-3 and is reported, not hidden"
            ),
        },
    }
    _write_json(out_dir / "escalation" / f"{_slug(arm)}.json", payload)
    _write_json(out_dir / "escalation" / "results.json", payload)
    _write_rows(out_dir / "escalation" / "curve.csv", curve)
    _write_rows(out_dir / "escalation" / "families.csv", families)
    _write_rows(out_dir / "escalation" / "languages.csv", languages)
    _write_rows(out_dir / "escalation" / "cost_sensitivity.csv", sensitivity)
    _plot_curve(curve, out_dir / "escalation" / "risk_coverage.png", target)
    return payload


def _policy_table(turns: Sequence[TurnRecord]) -> list[dict[str, Any]]:
    """Escalation policy variants: fixed thresholds vs the cost band vs the shipped engine.

    The fixed thresholds are configuration constants (not fitted on the test split), and the
    engine row is the system's actual decision, so the three are comparable as *policies*.
    """
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[str, float | None]] = [
        ("fixed threshold 0.95 (cost band default)", 0.95),
        ("fixed threshold 0.80 (5:1 band)", 0.80),
        ("fixed threshold 0.90 (10:1 band)", 0.90),
        ("conformal+cost engine (dev-fitted, shipped)", None),
    ]
    for name, threshold in candidates:
        if threshold is None:
            answered = [turn for turn in turns if turn.answered]
        else:
            answered = [turn for turn in turns if turn.confidence >= threshold]
        answered_ids = {turn.case_id for turn in answered}
        wrong = sum(1 for turn in answered if not turn.answer_ok)
        coverage = wilson_interval(len(answered), len(turns))
        risk = wilson_interval(wrong, len(answered)) if answered else None
        must = [turn for turn in turns if turn.must_escalate]
        caught = sum(1 for turn in must if turn.case_id not in answered_ids)
        recall = wilson_interval(caught, len(must))
        escalated = [turn for turn in turns if turn.case_id not in answered_ids]
        true_positive = sum(1 for turn in escalated if turn.must_escalate)
        precision = wilson_interval(true_positive, len(escalated)) if escalated else None
        rows.append(
            {
                "policy": name,
                "threshold": threshold,
                "coverage": round(coverage.value, 6),
                "coverage_ci": f"[{coverage.low:.3f},{coverage.high:.3f}]",
                "realised_risk": round(risk.value, 6) if risk else None,
                "risk_ci": f"[{risk.low:.3f},{risk.high:.3f}]" if risk else "n/a",
                "must_escalate_recall": round(recall.value, 6),
                "recall_ci": f"[{recall.low:.3f},{recall.high:.3f}]",
                "escalation_precision": round(precision.value, 6) if precision else None,
                "precision_ci": (
                    f"[{precision.low:.3f},{precision.high:.3f}]" if precision else "n/a"
                ),
                "n_answered": len(answered),
            }
        )
    return rows


def _plot_curve(curve: Sequence[Mapping[str, Any]], path: Path, target: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4.2))
    axis.plot([point["coverage"] for point in curve], [point["risk"] for point in curve])
    axis.axhline(target, linestyle="--", color="firebrick", label=f"target risk {target:.0%}")
    axis.set_xlabel("coverage (fraction answered)")
    axis.set_ylabel("realised risk")
    axis.set_xlim(0, 1)
    axis.grid(alpha=0.3)
    axis.legend()
    axis.set_title("Risk–coverage (held-out split)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------------
# Semantic cache replay (t8 cache, measured)
# --------------------------------------------------------------------------------------


def run_cache(
    state: BenchState,
    *,
    out_dir: Path,
    pipeline: Any,
    retriever: Any,
    limit: int = 60,
) -> dict[str, Any]:
    """Near-duplicate replay: generate once, then ask the cache for the sibling question."""
    from answerdesk.cache import SemanticCache

    if state.config.replay:
        return {"skipped": "replay mode: the cache harness needs the hosted embedding model"}

    by_id = {query.query_id: query for query in state.corpus.queries}
    pairs: list[tuple[Any, Any]] = []
    for case in state.test_cases:
        query_id = str(case.metadata.get("corpus_query_id", ""))
        query = by_id.get(query_id)
        if query is None or not case.answerable:
            continue
        siblings = [
            other
            for other in state.corpus.queries_for_doc(query.doc_id)
            if other.query_id != query.query_id and other.base_text == query.base_text
        ]
        if siblings:
            pairs.append((query, siblings[0]))
        if len(pairs) >= limit:
            break
    if not pairs:
        return {"skipped": "no near-duplicate query pairs in this corpus"}
    cache = SemanticCache(
        embedder=state.embedder,
        versions=_version_tuple(state, "cache-replay"),
        enabled=True,
        default_threshold=state.settings.cache.similarity_threshold,
        epsilon=state.settings.cache.epsilon,
        chunk_lookup=lambda chunk_id: state.corpus.by_id.get(chunk_id),
    )
    hits = wrong_hits = misses = 0
    hit_latencies: list[float] = []
    generation_usd = 0.0
    saved_usd = 0.0
    rows: list[dict[str, Any]] = []
    for previous, sibling in pairs:
        trace = retriever.retrieve(previous.text, final_k=state.settings.generate.context_k_max)
        passages = _packed_passages(trace)
        if not passages:
            continue
        anchor = time.perf_counter()
        result = pipeline.answer(previous.text, passages)
        generation_ms = (time.perf_counter() - anchor) * 1000.0
        if result.signal.action is not Action.ANSWER:
            continue
        generation_usd += result.usage.usd
        answer = answer_from_result(
            result=result,
            decision=escalation_decision(
                action=Action.ANSWER,
                reason_code=ReasonCode.ANSWERED,
                score=0.9,
                alpha=0.05,
                threshold_ref="cache-replay",
            ),
            p_correct=0.9,
            answer_id=f"cache-{previous.query_id}",
        )
        cache.put(
            query=previous.text,
            locale=previous.locale,
            intent_bucket=bucket_of(previous.intent),
            answer=answer,
            retrieval_fingerprint={"chunks": [citation.chunk_id for citation in answer.citations]},
        )
        anchor = time.perf_counter()
        lookup = cache.get(
            query=sibling.text,
            locale=sibling.locale,
            intent_bucket=bucket_of(sibling.intent),
            embedding=state.embedder.embed_query(sibling.text),
        )
        latency = (time.perf_counter() - anchor) * 1000.0
        hit = (
            lookup.status.value == "hit"
            if hasattr(lookup.status, "value")
            else str(lookup.status) == "hit"
        )
        if hit:
            hits += 1
            hit_latencies.append(latency)
            saved_usd += result.usage.usd
            cited = {
                citation.chunk_id
                for citation in (lookup.answer.citations if lookup.answer is not None else [])
            }
            sibling_gold = {chunk.chunk_id for chunk in state.corpus.faq_chunks(previous.doc_id)}
            wrong = bool(sibling_gold) and not (cited & sibling_gold)
            wrong_hits += 1 if wrong else 0
            rows.append(
                {
                    "pair": f"{previous.query_id}->{sibling.query_id}",
                    "hit": True,
                    "wrong_hit": wrong,
                    "latency_ms": round(latency, 3),
                    "generation_ms": round(generation_ms, 3),
                }
            )
        else:
            misses += 1
            rows.append(
                {
                    "pair": f"{previous.query_id}->{sibling.query_id}",
                    "hit": False,
                    "wrong_hit": False,
                    "latency_ms": round(latency, 3),
                    "generation_ms": round(generation_ms, 3),
                }
            )
    attempts = hits + misses
    hit_interval = wilson_interval(hits, attempts)
    wrong_interval = wilson_interval(wrong_hits, hits) if hits else None
    payload = {
        "pairs": len(pairs),
        "attempts": attempts,
        "hits": hits,
        "misses": misses,
        "wrong_hits": wrong_hits,
        "hit_rate": round(hit_interval.value, 6),
        "hit_rate_ci": f"[{hit_interval.low:.3f},{hit_interval.high:.3f}]",
        "wrong_hit_rate": round(wrong_interval.value, 6) if wrong_interval else 0.0,
        "wrong_hit_ucb": round(wrong_interval.high, 6) if wrong_interval else 0.0,
        "hit_latency_p50_ms": round(_pct(hit_latencies, 0.5), 3),
        "hit_latency_p95_ms": round(_pct(hit_latencies, 0.95), 3),
        "generation_usd_total": round(generation_usd, 8),
        "saved_usd": round(saved_usd, 8),
        "note": (
            "replay pairs are same-document, same-base-question rephrasings (typo/code-switch); "
            "a wrong hit is a hit whose cached citations do not cover the sibling's evidence"
        ),
    }
    _write_json(out_dir / "cache" / "results.json", payload)
    _write_rows(out_dir / "cache" / "pairs.csv", rows)
    return payload


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


# --------------------------------------------------------------------------------------
# Operations: latency, tokens, USD, pricing snapshot
# --------------------------------------------------------------------------------------


def run_ops(
    *, turns: Sequence[TurnRecord], state: BenchState, out_dir: Path, arm: str
) -> dict[str, Any]:
    stages = ("retrieval", "policy", "generation", "judge", "total")
    rows: list[dict[str, Any]] = []
    for stage in stages:
        values = [turn.latency_ms[stage] for turn in turns if stage in turn.latency_ms]
        if not values:
            continue
        rows.append(
            {
                "stage": stage,
                "n": len(values),
                "p50_ms": round(_pct(values, 0.5), 3),
                "p95_ms": round(_pct(values, 0.95), 3),
                "mean_ms": round(sum(values) / len(values), 3),
            }
        )
    answered = [turn for turn in turns if turn.answered]
    input_tokens = sum(turn.input_tokens for turn in turns)
    output_tokens = sum(turn.output_tokens for turn in turns)
    usd = sum(turn.usd for turn in turns)
    pricing_rows = _pricing_snapshot(state)
    payload = {
        "arm": arm,
        "n": len(turns),
        "answered": len(answered),
        "latency": rows,
        "tokens": {
            "input_total": input_tokens,
            "output_total": output_tokens,
            "input_per_answer": round(input_tokens / max(1, len(answered)), 1),
            "output_per_answer": round(output_tokens / max(1, len(answered)), 1),
        },
        "usd_total": round(usd, 8),
        "usd_per_answer": round(usd / max(1, len(answered)), 8),
        "pricing": pricing_rows,
        "hardware": (
            f"{platform.system()} {platform.release()} / Python {platform.python_version()}"
        ),
    }
    _write_json(out_dir / "ops" / f"ops-{_slug(arm)}.json", payload)
    _write_rows(out_dir / "ops" / f"latency-{_slug(arm)}.csv", rows)
    return payload


def _pricing_snapshot(state: BenchState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pricing = state.pricing
    if pricing is None:
        return rows
    raw_entries = getattr(pricing, "entries", {})
    values = raw_entries.values() if hasattr(raw_entries, "values") else raw_entries
    for entry in sorted(values, key=lambda item: item.model):
        rows.append(
            {
                "model": entry.model,
                "input_usd_per_1k_tokens": entry.input_usd_per_1k_tokens,
                "output_usd_per_1k_tokens": entry.output_usd_per_1k_tokens,
                "retrieved_at": entry.retrieved_at,
                "source": entry.source,
            }
        )
    return rows


__all__ = [
    "COST_RATIOS",
    "NOISY_CATEGORY",
    "UNSAFE_CATEGORIES",
    "BenchConfig",
    "BenchState",
    "case_stratum",
    "prepare",
    "run_answers",
    "run_cache",
    "run_escalation",
    "run_ops",
    "run_retrieval",
    "unsafe_kind",
]
# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def run_bench(
    settings: Settings,
    *,
    out_dir: str | Path,
    config: BenchConfig,
    evals_root: str | Path,
) -> dict[str, Any]:
    """Run the whole evidence suite and write ``reports/<run>/`` artifacts."""
    started = time.perf_counter()
    directory = Path(out_dir).resolve()
    # Never let a replay run overwrite live evidence: a fresh checkout (no keys) re-running
    # the documented command must not destroy the reported artefacts, so it lands beside them.
    summary_path = directory / "summary.json"
    if config.replay and summary_path.is_file():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
        if not existing.get("replay", True):
            directory = directory.with_name(f"{directory.name}-replay")
    directory.mkdir(parents=True, exist_ok=True)
    state = prepare(settings, config, evals_root=Path(evals_root))
    summary: dict[str, Any] = {
        "run_id": config.run_id,
        "replay": config.replay,
        "config": config.as_dict(),
        "corpus_version": state.corpus.version,
        "corpus_hash": state.corpus.hash,
        "documents": len(state.corpus.document_ids),
        "chunks": len(state.corpus.chunks),
        "test_cases": len(state.test_cases),
        "dev_cases": len(state.dev_cases),
        "calibration": state.calibration_note,
        "hardware": (
            f"{platform.system()} {platform.release()} / Python {platform.python_version()}"
        ),
        "notes": list(state.notes),
        "models": {},
    }
    if config.calibrate_on_dev and config.generator_arms:
        dev_selected = _subset(
            state.dev_cases,
            locales=config.locales,
            limit=config.dev_limit if config.dev_limit is not None else 120,
            seed=config.seed,
        )
        _, dev_turns, _ = run_answers(
            state, arm="primary", out_dir=directory / "calibration", cases=dev_selected
        )
        state.engine, calibration_payload = calibrate_engine_from_dev(
            state=state, dev_turns=dev_turns, out_dir=directory
        )
        dev_operating = select_operating_threshold(dev_turns, config.target_risk)
        calibration_payload["operating_point_selection"] = dev_operating
        summary["calibration"] = calibration_payload
    retrieval = run_retrieval(state, out_dir=directory)
    summary["retrieval"] = {
        arm: {
            "recall_any@10": _cell_value(payload["cells"], "recall_any@10"),
            "recall_any@20": _cell_value(payload["cells"], "recall_any@20"),
            "recall_all@20": _cell_value(payload["cells"], "recall_all@20"),
            "nDCG@10": _cell_value(payload["cells"], "nDCG@10"),
            "MRR@10": _cell_value(payload["cells"], "MRR@10"),
            "n_labeled": payload["n_labeled"],
        }
        for arm, payload in retrieval["arms"].items()
    }
    summary["retrieval_comparisons"] = retrieval["comparisons"]
    resolved_calibration = summary.get("calibration")
    selected_operating: dict[str, Any] | None = (
        resolved_calibration.get("operating_point_selection")
        if isinstance(resolved_calibration, Mapping)
        else None
    )
    answers: dict[str, Any] = {}
    escalations: dict[str, Any] = {}
    ops: dict[str, Any] = {}
    for arm in config.generator_arms:
        payload, turns, selected = run_answers(state, arm=arm, out_dir=directory)
        answers[arm] = payload
        summary["models"][arm] = {
            "generator": payload["generator_model"],
            "judge": payload["judge_model"],
            "judge_family": payload["judge"]["judge_family"],
            "generator_family": payload["judge"]["generator_family"],
            "cross_family": payload["judge"]["cross_family"],
            "reportable": payload["judge"]["reportable"],
            "n": payload["n"],
        }
        escalations[arm] = run_escalation(
            state=state,
            turns=turns,
            cases=selected,
            out_dir=directory,
            arm=arm,
            dev_operating=selected_operating,
        )
        ops[arm] = run_ops(turns=turns, state=state, out_dir=directory, arm=arm)
    if config.generator_arms:
        primary_provider, _ = _provider_for_arm(state, "primary")
        primary_pipeline = AnswerPipeline.from_settings(primary_provider, settings)
        retriever = state.variants.get(state.config.answer_retriever) or state.variants.get(
            "cc@0.75"
        )
        summary["cache"] = run_cache(
            state,
            out_dir=directory,
            pipeline=primary_pipeline,
            retriever=retriever,
            limit=min(60, len(state.test_cases)),
        )
    else:
        summary["cache"] = {"skipped": "retrieval-only run"}
    summary["escalation"] = {
        arm: {
            "confusion": escalations[arm]["confusion"],
            "operating_point": escalations[arm]["operating_point"],
            "e3": escalations[arm]["e3"],
            "families": escalations[arm]["families"],
        }
        for arm in escalations
    }
    summary["ops"] = ops
    summary["answers"] = {
        arm: {
            "groundedness": _cell_value(payload["cells"], "groundedness_judged"),
            "citation_recall": _cell_value(payload["cells"], "citation_recall"),
            "rubric_score": _cell_value(payload["cells"], "rubric_score"),
            "coverage": _cell_value(payload["cells"], "coverage"),
            "must_escalate_recall": _cell_value(payload["cells"], "must_escalate_recall"),
            "realised_risk": _cell_value(payload["cells"], "realised_risk"),
            "noisy_n": payload["noisy_n"],
        }
        for arm, payload in answers.items()
    }
    summary["noisy"] = {arm: payload["noisy_cells"] for arm, payload in answers.items()}
    summary["duration_s"] = round(time.perf_counter() - started, 1)
    _write_json(directory / "summary.json", summary)
    (directory / "report.md").write_text(
        render_bench_report(summary, answers, escalations), encoding="utf-8"
    )
    return summary


def _cell_value(
    cells: Sequence[Mapping[str, Any]], name: str, locale: str = "macro"
) -> dict[str, Any] | None:
    for cell in cells:
        if cell.get("name") == name and cell.get("locale") == locale:
            return {
                "value": cell.get("value"),
                "ci_low": cell.get("ci_low"),
                "ci_high": cell.get("ci_high"),
                "n": cell.get("n"),
            }
    for cell in cells:
        if cell.get("name") == name and not cell.get("locale"):
            return {
                "value": cell.get("value"),
                "ci_low": cell.get("ci_low"),
                "ci_high": cell.get("ci_high"),
                "n": cell.get("n"),
            }
    return None


# --------------------------------------------------------------------------------------
# Markdown report (the raw material for docs/results.md)
# --------------------------------------------------------------------------------------


def render_bench_report(
    summary: Mapping[str, Any],
    answers: Mapping[str, Any],
    escalations: Mapping[str, Any],
) -> str:
    lines: list[str] = [
        f"# answerdesk benchmark run — `{summary['run_id']}`",
        "",
    ]
    if summary.get("replay"):
        lines.append(
            "> **REPLAY-MODE.** Deterministic doubles were used; these numbers validate the "
            "harness, not the system."
        )
        lines.append("")
    lines.append(
        _markdown_table(
            ("field", "value"),
            [
                ("corpus version", summary["corpus_version"]),
                ("corpus hash", summary["corpus_hash"]),
                ("documents / chunks", f"{summary['documents']} / {summary['chunks']}"),
                ("test cases", summary["test_cases"]),
                ("calibration", summary["calibration"]),
                ("hardware", summary["hardware"]),
                ("duration", f"{summary['duration_s']} s"),
            ],
        )
    )
    lines.extend(["", "## Retrieval ablation", ""])
    rows = []
    for arm, values in summary["retrieval"].items():
        rows.append(
            [
                arm,
                _fmt(values["recall_any@10"]),
                _fmt(values["recall_any@20"]),
                _fmt(values["recall_all@20"]),
                _fmt(values["nDCG@10"]),
                _fmt(values["MRR@10"]),
                values["n_labeled"],
            ]
        )
    lines.append(
        _markdown_table(
            ("arm", "recall_any@10", "recall_any@20", "recall_all@20", "nDCG@10", "MRR@10", "n"),
            rows,
        )
    )
    lines.extend(["", "### Paired comparisons (recall_any@10)", ""])
    rows = []
    for comparison in summary.get("retrieval_comparisons", []):
        rows.append(
            [
                f"{comparison['arm_a']} − {comparison['arm_b']}",
                comparison["n"],
                f"{comparison['diff']:+.3f}",
                f"[{comparison['ci_low']:+.3f}, {comparison['ci_high']:+.3f}]",
                f"{comparison['mcnemar_p']:.4f} ({comparison['mcnemar_method']})",
                "yes" if comparison["significant"] else "no",
            ]
        )
    lines.append(_markdown_table(("pair", "n", "Δ", "95% CI", "McNemar p", "CI excludes 0"), rows))
    lines.extend(["", "## Answer quality and escalation per arm", ""])
    for arm, payload in answers.items():
        lines.append(
            f"### `{arm}` — generator `{payload['generator_model']}`, "
            f"judge `{payload['judge_model']}`"
        )
        lines.append("")
        judge = payload["judge"]
        lines.append(
            f"Cross-family judging: **{'yes' if judge['cross_family'] else 'no'}** "
            f"({judge['generator_family']} vs {judge['judge_family']}); "
            f"reportable: **{'yes' if judge['reportable'] else 'no'}**. {judge['note']}"
        )
        lines.append("")
        rows = []
        for name in (
            "groundedness_judged",
            "citation_recall",
            "rubric_score",
            "rubric_pass_rate",
            "coverage",
            "must_escalate_recall",
            "escalation_precision",
            "realised_risk",
        ):
            cell = _cell_value(payload["cells"], name)
            if cell is None:
                continue
            rows.append(
                [
                    name,
                    cell["value"],
                    f"[{cell['ci_low']:.3f}, {cell['ci_high']:.3f}]",
                    cell["n"],
                ]
            )
        lines.append(_markdown_table(("metric", "value", "95% CI", "n"), rows))
        lines.append("")
        confusion = payload["confusion"]
        lines.append(
            f"Confusion: TP {confusion['tp']} · FP {confusion['fp']} · FN {confusion['fn']} · "
            f"TN {confusion['tn']} — false-escalation rate "
            f"{_fmt(confusion['false_escalation_rate'])}."
        )
        lines.append("")
        if payload["reason_codes"]:
            lines.extend(
                [
                    _markdown_table(
                        ("reason code", "count", "share"),
                        [
                            [row["reason_code"], row["count"], row["share"]]
                            for row in payload["reason_codes"]
                        ],
                    ),
                    "",
                ]
            )
        escalation = escalations.get(arm, {})
        if escalation.get("families"):
            lines.append("E-2 recall by unsafe family:")
            lines.append("")
            lines.append(
                _markdown_table(
                    ("family", "n", "escalated", "recall", "95% CI"),
                    [
                        [
                            row["family"],
                            row["n"],
                            row["escalated"],
                            row["recall"],
                            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
                        ]
                        for row in escalation["families"]
                    ],
                )
            )
            lines.append("")
        operating = escalation.get("operating_point")
        if operating:
            lines.append(
                f"Operating point: coverage **{operating['coverage']:.3f}** at realised risk "
                f"**{operating['risk']:.4f}** (target ≤ {summary['config']['target_risk']:.0%})."
            )
            lines.append("")
        if escalation.get("cost_sensitivity"):
            lines.append(
                _markdown_table(
                    ("cost ratio", "threshold", "coverage", "realised risk", "n answered"),
                    [
                        [
                            row["cost_ratio_wrong_to_abstain"],
                            row["threshold"],
                            row["coverage"],
                            row["realised_risk"],
                            row["n_answered"],
                        ]
                        for row in escalation["cost_sensitivity"]
                    ],
                )
            )
            lines.append("")
    cache = summary.get("cache") or {}
    lines.extend(["## Semantic cache", ""])
    if "skipped" in cache:
        lines.append(f"Skipped: {cache['skipped']}")
    else:
        lines.append(
            _markdown_table(
                ("quantity", "value"),
                [
                    ("pairs", cache.get("pairs")),
                    ("hits / attempts", f"{cache.get('hits')} / {cache.get('attempts')}"),
                    ("hit rate", _fmt_pct(cache.get("hit_rate"))),
                    ("wrong-hit rate", _fmt_pct(cache.get("wrong_hit_rate"))),
                    (
                        "hit latency p50/p95",
                        f"{cache.get('hit_latency_p50_ms')} / {cache.get('hit_latency_p95_ms')} ms",
                    ),
                    ("saved USD (hits)", cache.get("saved_usd")),
                ],
            )
        )
    lines.append("")
    lines.extend(["## Operations", ""])
    rows = []
    for arm, payload in summary.get("ops", {}).items():
        for row in payload["latency"]:
            rows.append([arm, row["stage"], row["p50_ms"], row["p95_ms"], row["n"]])
    lines.append(_markdown_table(("arm", "stage", "p50 ms", "p95 ms", "n"), rows))
    lines.append("")
    lines.append(
        "Token counts and USD are recorded per answer from the provider's own usage fields "
        "and the pinned pricing table; see `ops/` for the pricing snapshot."
    )
    lines.append("")
    if summary.get("noisy"):
        lines.extend(
            [
                "## Noisy / free-form slice",
                "",
                "Reported separately and never averaged into the headline numbers.",
                "",
            ]
        )
        for arm, cells in summary["noisy"].items():
            if not cells:
                lines.append(f"`{arm}`: no noisy items in this corpus version.")
                continue
            lines.append(
                _markdown_table(
                    ("metric", "value", "95% CI", "n"),
                    [
                        [
                            cell["name"],
                            cell["value"],
                            f"[{cell['ci_low']:.3f}, {cell['ci_high']:.3f}]",
                            cell["n"],
                        ]
                        for cell in cells
                    ],
                )
            )
            lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            *[f"* {note}" for note in summary.get("notes", [])],
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(cell: Any, digits: int = 3) -> str:
    """Format a cell dict (value + interval) or a bare number."""
    if cell is None:
        return "n/a"
    if isinstance(cell, (int, float)):
        return f"{float(cell):.{digits}f}"
    if not isinstance(cell, Mapping) or "value" not in cell:
        return "n/a"
    low = cell.get("ci_low")
    high = cell.get("ci_high")
    if low is None or high is None:
        return f"{float(cell['value']):.{digits}f}"
    return f"{float(cell['value']):.{digits}f} [{float(low):.{digits}f}, {float(high):.{digits}f}]"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


__all__ += ["render_bench_report", "run_bench"]
# --------------------------------------------------------------------------------------
# Dev-fitted conformal calibration (the policy instance used by the reported run)
# --------------------------------------------------------------------------------------


def calibrate_engine_from_dev(
    *,
    state: BenchState,
    dev_turns: Sequence[TurnRecord],
    out_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Fit split-conformal q̂ per intent bucket on the **dev** split (INV-09).

    The shipped artifact (t8) was sealed on t8's own study; §8.6 requires a refresh per
    corpus, so the evidence run fits its own thresholds on dev and evaluates on test. The
    engine is t8's :class:`DecisionEngine` — only the numbers differ.
    """
    import hashlib

    from answerdesk.policy import (
        ConformalCertificate,
        ConformalSet,
        CostProvenance,
        CostSet,
        DecisionEngine,
        GuardConfig,
        build_threshold_table,
        conformal_quantile,
    )

    settings = state.settings
    alpha_by_bucket = {
        IntentBucket.INFORMATIONAL: settings.policy.alpha[IntentBucket.INFORMATIONAL],
        IntentBucket.TRANSACTIONAL: settings.policy.alpha[IntentBucket.TRANSACTIONAL],
        IntentBucket.ACCOUNT_LEGAL: settings.policy.alpha[IntentBucket.ACCOUNT_LEGAL],
    }
    certificates: list[Any] = []
    rows: list[dict[str, Any]] = []
    for bucket in (
        IntentBucket.INFORMATIONAL,
        IntentBucket.TRANSACTIONAL,
        IntentBucket.ACCOUNT_LEGAL,
    ):
        subset = [turn for turn in dev_turns if turn.intent_bucket is bucket]
        pooled = False
        if not subset:
            # A bucket with no dev cases falls back to the pooled dev distribution and says
            # so: refusing to decide would be worse than a documented conservative fallback.
            subset = list(dev_turns)
            pooled = True
        if not subset:
            continue
        risks = [1.0 - turn.confidence for turn in subset]
        alpha = alpha_by_bucket[bucket]
        quantile = conformal_quantile(risks, alpha)
        certificates.append(
            ConformalCertificate(
                bucket=bucket,
                alpha=alpha,
                n=len(subset),
                rank=quantile.rank,
                q_hat=quantile.q_hat,
                feasible=quantile.feasible,
                coverage_lower_bound=min(1.0, 1.0 - alpha),
                calibration_hash="dev-" + state.corpus.version[:12],
                scorer_name="OverlapScorer",
                scorer_version="policy-overlap-1",
                notes=(
                    "fitted on the dev split by the t11 benchmark"
                    + (" (pooled fallback: no dev cases in this bucket)" if pooled else "")
                ),
            )
        )
        rows.append(
            {
                "bucket": bucket.value,
                "n": len(subset),
                "alpha": alpha,
                "q_hat": round(quantile.q_hat, 6),
                "rank": quantile.rank,
                "feasible": quantile.feasible,
                "pooled_fallback": pooled,
                "coverage_lower_bound": round(min(1.0, 1.0 - alpha), 6),
            }
        )
    conformal = ConformalSet(certificates=certificates)
    costs = settings.policy.costs
    cost_set = CostSet(
        c_wrong=costs.c_wrong,
        c_abstain=costs.c_abstain,
        c_delay=costs.c_delay,
        provenance=CostProvenance(
            source="configs/default.yaml policy.costs",
            owner="answerdesk",
            date="2026-10-21",
            note="documented assumptions; no live handoff-cost measurement exists (Q4)",
        ),
    )
    table = build_threshold_table(
        conformal_set=conformal,
        cost_set=cost_set,
        alpha_by_bucket=alpha_by_bucket,
        calibration_hash="dev-" + state.corpus.version[:12],
    )
    engine = DecisionEngine(
        thresholds=table,
        conformal=conformal,
        cost_set=cost_set,
        guards=GuardConfig(),
        artifact_hash=hashlib.sha256(
            f"{state.corpus.version}:"
            f"{'|'.join(row['bucket'] + str(row['q_hat']) for row in rows)}".encode()
        ).hexdigest()[:16],
        corpus_version=state.corpus.version,
    )
    payload = {
        "corpus_version": state.corpus.version,
        "split": "dev",
        "n": len(dev_turns),
        "buckets": rows,
        "alpha_by_bucket": {bucket.value: alpha for bucket, alpha in alpha_by_bucket.items()},
        "costs": {
            "c_wrong": cost_set.c_wrong,
            "c_abstain": cost_set.c_abstain,
            "c_delay": cost_set.c_delay,
        },
        "note": (
            "split-conformal thresholds fitted on the dev split; the test split is never "
            "used for fitting (INV-09)"
        ),
    }
    _write_json(out_dir / "calibration" / "dev-fit.json", payload)
    return engine, payload


__all__ += ["calibrate_engine_from_dev"]
# --------------------------------------------------------------------------------------
# Chunking ablation: structure-first vs naive fixed-window
# --------------------------------------------------------------------------------------


def _naive_chunks(
    corpus: Corpus, *, window: int = 900, overlap: int = 150
) -> tuple[list[Any], dict[str, list[tuple[str, int, int]]]]:
    """Fixed-window chunks over each document's *text*, ignoring every structural cue.

    The document text is rebuilt by concatenating its structure chunks in reading order (the
    corpus stores no separate document text), and the map from structure chunk id to
    ``[start, end)`` in that concatenation is returned so gold evidence can be mapped onto the
    naive chunking by overlap. Windows of 900 characters with 150 overlap approximate a
    "split every N characters" baseline.
    """
    from answerdesk.contracts import Chunk

    chunks: list[Chunk] = []
    offsets: dict[str, list[tuple[str, int, int]]] = {}
    step = max(1, window - overlap)
    for doc_id in corpus.document_ids:
        originals = corpus.doc_chunks(doc_id)
        if not originals:
            continue
        parts: list[str] = []
        cursor = 0
        spans: list[tuple[str, int, int]] = []
        for chunk in originals:
            if parts:
                parts.append("\n\n")
                cursor += 2
            parts.append(chunk.text)
            spans.append((chunk.chunk_id, cursor, cursor + len(chunk.text)))
            cursor += len(chunk.text)
        text = "".join(parts)
        offsets[doc_id] = spans
        position = 0
        ordinal = 0
        while position < len(text):
            end = min(len(text), position + window)
            piece = text[position:end]
            if piece.strip():
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::n{ordinal:04d}",
                        doc_id=doc_id,
                        text=piece,
                        heading_path=[],
                        breadcrumb="",
                        start_char=position,
                        end_char=end,
                        token_count=max(1, len(piece.split())),
                        language=originals[0].language,
                        metadata={"chunker": "naive-fixed-window", "window": window},
                    )
                )
                ordinal += 1
            position += step
    return chunks, offsets


def run_chunking_ablation(
    state: BenchState,
    *,
    out_dir: Path,
    cases: Sequence[EvalCase],
    limit: int = 240,
) -> dict[str, Any]:
    """Structure-first chunking vs a naive fixed window, same embedder and same gold spans."""
    from answerdesk.eval.arms import BM25Arm, DenseHashArm
    from answerdesk.eval.corpus import Corpus, CorpusQuery

    selected = _subset(cases, locales=state.config.locales, limit=limit, seed=state.config.seed)
    naive, naive_offsets = _naive_chunks(state.corpus)
    naive_corpus = Corpus(
        root=state.corpus.root,
        chunks=naive,
        queries=[
            CorpusQuery(
                query_id=query.query_id,
                locale=query.locale,
                doc_id=query.doc_id,
                intent=query.intent,
                category=query.category,
                text=query.text,
                base_text=query.base_text,
                operator_reply=query.operator_reply,
            )
            for query in state.corpus.queries
        ],
        manifest=state.corpus.manifest,
        chunks_hash=f"naive-{len(naive)}",
    )
    embedder = state.embedder
    arms: dict[str, Any] = {
        "naive-bm25": BM25Arm(naive_corpus),
    }
    with contextlib.suppress(Exception):
        arms["naive-dense"] = DenseHashArm(naive_corpus, embedder=embedder)
    structure: dict[str, Any] = {
        "structure-bm25": BM25Arm(state.corpus),
    }
    with contextlib.suppress(Exception):
        structure["structure-dense"] = DenseHashArm(state.corpus, embedder=embedder)
    rows: list[dict[str, Any]] = []
    for name, arm in {**structure, **arms}.items():
        corpus = state.corpus if name.startswith("structure") else naive_corpus
        hits10 = hits20 = 0
        total = 0
        for case in selected:
            if not case.gold_chunk_ids:
                continue
            gold = (
                _gold_ids_on(corpus, state.corpus, case.gold_chunk_ids)
                if name.startswith("structure")
                else _gold_ids_on_naive(naive, naive_offsets, case.gold_chunk_ids)
            )
            if not gold:
                continue
            ranked = [passage.chunk.chunk_id for passage in arm.search(case.question, 20)]
            total += 1
            hits10 += 1 if set(ranked[:10]) & gold else 0
            hits20 += 1 if set(ranked[:20]) & gold else 0
        interval10 = wilson_interval(hits10, total) if total else None
        interval20 = wilson_interval(hits20, total) if total else None
        rows.append(
            {
                "arm": name,
                "n": total,
                "recall_any@10": round(interval10.value, 6) if interval10 else None,
                "recall_any@10_ci": (
                    f"[{interval10.low:.3f},{interval10.high:.3f}]" if interval10 else "n/a"
                ),
                "recall_any@20": round(interval20.value, 6) if interval20 else None,
                "recall_any@20_ci": (
                    f"[{interval20.low:.3f},{interval20.high:.3f}]" if interval20 else "n/a"
                ),
                "chunks": len(corpus.chunks),
            }
        )
    payload = {
        "cases": len(selected),
        "naive_window": 900,
        "naive_overlap": 150,
        "rows": rows,
        "note": (
            "gold spans are carried over by character overlap, so both chunkings are scored "
            "against the same evidence; the naive arm ignores headings, tables and FAQ pairs"
        ),
    }
    _write_json(out_dir / "chunking" / "results.json", payload)
    _write_rows(out_dir / "chunking" / "results.csv", rows)
    return payload


def _gold_ids_on_naive(
    naive: Sequence[Any],
    offsets: Mapping[str, list[tuple[str, int, int]]],
    gold_ids: Sequence[str],
) -> set[str]:
    """Map structure gold ids onto naive chunks through the concatenation offsets."""
    spans: list[tuple[str, int, int]] = []
    for chunk_id in gold_ids:
        doc_id = chunk_id.split("::", 1)[0]
        for original_id, start, end in offsets.get(doc_id, []):
            if original_id == chunk_id:
                spans.append((doc_id, start, end))
    result: set[str] = set()
    for chunk in naive:
        for doc_id, start, end in spans:
            if chunk.doc_id == doc_id and chunk.end_char > start and chunk.start_char < end:
                result.add(chunk.chunk_id)
    return result


def _gold_ids_on(target: Corpus, source: Corpus, gold_ids: Sequence[str]) -> set[str]:
    """Map gold chunk ids from the structure corpus onto the target chunking by span overlap."""
    spans: list[tuple[str, int, int]] = []
    for chunk_id in gold_ids:
        chunk = source.by_id.get(chunk_id)
        if chunk is None:
            continue
        spans.append((chunk.doc_id, chunk.start_char, chunk.end_char))
    result: set[str] = set()
    for doc_id, start, end in spans:
        for chunk in target.doc_chunks(doc_id):
            if chunk.end_char > start and chunk.start_char < end:
                result.add(chunk.chunk_id)
    return result


__all__ += ["run_chunking_ablation"]
