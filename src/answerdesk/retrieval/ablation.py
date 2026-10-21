"""The retrieval ablation runner: ``python -m answerdesk.retrieval.ablation``.

This is the executable answer to "is hybrid retrieval decoration?" (RT-7) — the ablation
the research found nobody publishes. It runs the six arms requirements §6.3 names::

    bm25 · dense · hybrid-rrf · hybrid-convex · hybrid-convex+rerank · hybrid-convex+mmr

and prints one table with ``recall_any@20``, ``recall_any@10``, ``recall_all@20``,
``nDCG@10``, ``MRR@10``, the identifier subset (RT-8), the perturbation delta (RT-9), a
paired-bootstrap interval for the hybrid uplift (RT-7) and p50/p95 latency.

Discipline this runner enforces rather than hopes for:

* fusion parameters (``rrf_k``, ``alpha``, or learnt weights) are fitted on the **dev**
  split only, and :func:`answerdesk.retrieval.fit.fit_fusion_artifact` raises
  :class:`~answerdesk.retrieval.fit.LeakageError` if the two splits touch (INV-09);
* every reported number comes from the **test** split; the dev numbers exist only inside
  the artifact's ``metric_value`` and the search history;
* the artefact header carries the corpus version, the corpus chunk hash, both split hashes
  and the embedder identity, so a table can never be detached from what produced it
  (NFR-REP-02);
* with the ``hashing`` embedder (the default, and the only offline one) the table is a
  **machinery check**, not a quality claim — the runner prints that caveat every time
  (§6.3 step 2).

Candidate lists are computed once per query and replayed through each arm: fusion,
rerank and MMR are pure functions of those lists, so replaying them is mathematically the
same measurement as re-running retrieval while keeping the run to seconds instead of
minutes. Latency is measured separately on a subsample by running the *real* pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from answerdesk.config import Settings, load_settings
from answerdesk.contracts import Channel, RetrievedPassage
from answerdesk.errors import CorpusMissingError
from answerdesk.index.store import (
    ChunkStore,
    EmbedderInfo,
    IndexBundle,
    index_root,
)
from answerdesk.rerank.cross_encoder import LexicalOverlapReranker
from answerdesk.rerank.mmr import DEFAULT_MMR_LAMBDA, MMRDiversifier, vector_similarity
from answerdesk.retrieval.datasets import (
    CLEAN_CATEGORIES,
    NOISY_CATEGORY,
    RetrievalCase,
    SplitBundle,
    build_cases,
    load_query_rows,
    split_cases,
)
from answerdesk.retrieval.fit import (
    CorpusProvenance,
    FusionArtifact,
    FusionObservation,
    SplitProvenance,
    fit_fusion_artifact,
    split_hash,
)
from answerdesk.retrieval.fusion import FusionParams
from answerdesk.retrieval.metrics import (
    evaluate_cases,
    macro_average,
    paired_bootstrap_diff,
    per_case_scores,
)
from answerdesk.retrieval.pipeline import PipelineConfig, RetrievalPipeline
from answerdesk.retrieval.query_analysis import QueryAnalysisConfig, QueryAnalyzer

#: Report format version — bumped when the table's shape changes.
ABLATION_SCHEMA_VERSION = "1.0.0"

#: Evaluation cut-offs, straight from §6.1/§6.2.
RECALL_K = 20
NARROW_K = 10

#: Default caveat that must accompany any table produced with a CI embedder (§6.3).
MACHINERY_CAVEAT = (
    "machinery check, not a quality claim: report evidence only from a real multilingual "
    "embedder (requirements 6.3 step 2)"
)


@dataclass(frozen=True)
class ArmSpec:
    """One ablation arm: which channels run, how they fuse, what optional stages follow."""

    name: str
    fuser: Literal["rrf", "cc", "none"]
    channels: tuple[str, ...] = ("bm25", "dense")
    rerank: bool = False
    mmr: bool = False
    note: str = ""


#: The six arms of §6.3 step 3 / §10.6, in report order.
DEFAULT_ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(name="bm25", fuser="none", channels=("bm25",), note="sparse only"),
    ArmSpec(name="dense", fuser="none", channels=("dense",), note="exact cosine only"),
    ArmSpec(name="hybrid-rrf", fuser="rrf", note="rank fusion, fitted k"),
    ArmSpec(name="hybrid-convex", fuser="cc", note="convex combination, fitted alpha"),
    ArmSpec(name="hybrid-convex+rerank", fuser="cc", rerank=True, note="+ lexical reranker arm"),
    ArmSpec(name="hybrid-convex+mmr", fuser="cc", mmr=True, note="+ MMR diversification"),
)

#: Arm whose per-locale detail is reported (the shipped path, D-03).
PRIMARY_ARM = "hybrid-convex"

#: Shipped arm definitions on disk; the code defaults above must stay identical.
ARMS_FILE = Path("configs/retrieval/ablations.yaml")


def load_arms(path: str | Path) -> tuple[ArmSpec, ...]:
    """Read an ``ablations.yaml`` arm list into :class:`ArmSpec` objects.

    A malformed arms file is a configuration error, not a silent fallback to the defaults:
    an evidence run must never measure something different from what it claims.
    """
    import yaml

    from answerdesk.errors import ConfigError

    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"no arms file at {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source} is not valid YAML: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("arms"), list):
        raise ConfigError(f"{source} must contain an 'arms' list")
    arms: list[ArmSpec] = []
    for entry in payload["arms"]:
        if not isinstance(entry, dict):
            raise ConfigError(f"{source}: every arm must be a mapping")
        name = str(entry.get("name", "")).strip()
        fuser = str(entry.get("fuser", "")).strip()
        if not name:
            raise ConfigError(f"{source}: an arm has no name")
        if fuser not in ("rrf", "cc", "none"):
            raise ConfigError(f"{source}: arm {name!r} has unknown fuser {fuser!r}")
        channels = tuple(str(channel) for channel in entry.get("channels") or ("bm25", "dense"))
        unknown = sorted(set(channels) - {"bm25", "dense"})
        if unknown:
            raise ConfigError(f"{source}: arm {name!r} names unknown channels {unknown}")
        arms.append(
            ArmSpec(
                name=name,
                fuser=fuser,  # type: ignore[arg-type]
                channels=channels,
                rerank=bool(entry.get("rerank", False)),
                mmr=bool(entry.get("mmr", False)),
                note=str(entry.get("note") or "").strip(),
            )
        )
    if not arms:
        raise ConfigError(f"{source}: no arms defined")
    if len({arm.name for arm in arms}) != len(arms):
        raise ConfigError(f"{source}: duplicate arm names")
    return tuple(arms)


class ArmMetrics(BaseModel):
    """All numbers of one arm, at both cut-offs and on the gated subsets."""

    model_config = ConfigDict(extra="forbid")

    name: str
    fuser: str
    rerank: str | None = None
    mmr_lambda: float | None = None
    n: int = 0
    recall_any_at_20: float = 0.0
    recall_any_at_10: float = 0.0
    recall_all_at_20: float = 0.0
    ndcg_at_10: float = 0.0
    mrr_at_10: float = 0.0
    strict_recall_any_at_20: float | None = None
    strict_n: int = 0
    identifier_recall_any_at_20: float | None = None
    identifier_n: int = 0
    perturbation_delta: float | None = None
    per_locale: dict[str, float] = Field(default_factory=dict)
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    note: str = ""


class AblationReport(BaseModel):
    """The committed artifact: parameters, provenance, per-arm numbers."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = ABLATION_SCHEMA_VERSION
    corpus_version: str
    corpus_chunk_hash: str = ""
    embedder: EmbedderInfo
    eval_split: SplitProvenance
    dev_split: SplitProvenance
    split_salt: str
    fitted: FusionArtifact | None = None
    arms: list[ArmMetrics] = Field(default_factory=list)
    hybrid_uplift: float | None = None
    hybrid_uplift_ci: list[float] = Field(default_factory=list)
    hybrid_uplift_strict: float | None = None
    primary_arm: str = PRIMARY_ARM
    noise_ablation: NoiseAblationReport | None = None
    generated_at: str = ""
    timings: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------------------
# candidate replay
# ---------------------------------------------------------------------------------------


@dataclass
class CandidateSet:
    """Per-case candidate lists, computed once and replayed through every arm."""

    sparse: list[RetrievedPassage] = field(default_factory=list)
    dense: list[RetrievedPassage] = field(default_factory=list)


def _candidate_sets(
    pipeline: RetrievalPipeline,
    cases: Sequence[RetrievalCase],
    *,
    top_k: int,
) -> dict[str, CandidateSet]:
    """Run both retrieval channels once per query (the production code path)."""
    cache: dict[str, CandidateSet] = {}
    for case in cases:
        analysis = pipeline.analyzer.analyze(case.query)
        cache[case.case_id] = CandidateSet(
            sparse=pipeline._sparse_candidates(analysis, top_k),
            dense=pipeline._dense_candidates(analysis, top_k),
        )
    return cache


def _ranked_ids(
    arm: ArmSpec,
    *,
    candidates: CandidateSet,
    params: FusionParams,
    config: PipelineConfig,
    reranker: LexicalOverlapReranker,
    query: str,
) -> list[str]:
    """Replay one arm over cached candidates and return the ordered chunk ids."""
    result_sets: dict[Channel, Sequence[RetrievedPassage]] = {}
    if "bm25" in arm.channels and candidates.sparse:
        result_sets[Channel.BM25] = candidates.sparse
    if "dense" in arm.channels and candidates.dense:
        result_sets[Channel.DENSE] = candidates.dense
    if not result_sets:
        return []

    if arm.fuser == "none":
        only = next(iter(result_sets.values()))
        ordered = list(only)[: config.gate_k]
    else:
        ordered = params.build().fuse(result_sets, config.top_k)

    if arm.rerank and ordered:
        ordered = reranker.rerank(query, ordered[: config.rerank_top_in], config.gate_k)
    if arm.mmr and ordered:
        diversifier = MMRDiversifier(lambda_=config.mmr_lambda or DEFAULT_MMR_LAMBDA)
        ordered = diversifier.diversify(ordered, config.gate_k)
    return [passage.chunk.chunk_id for passage in ordered]


# ---------------------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------------------


def _gold_lists(cases: Sequence[RetrievalCase], *, strict: bool) -> list[list[str]]:
    """Gold ids per case: document-level (certain) or strict (heuristic answer chunk)."""
    if strict:
        return [list(case.answer_chunk_ids) for case in cases]
    return [list(case.gold_chunk_ids) for case in cases]


def _strict_metrics(
    cases: Sequence[RetrievalCase], rankings: dict[str, list[str]]
) -> tuple[float | None, int]:
    """``recall_any@20`` against the heuristic single-chunk label, when available."""
    subset = [case for case in cases if case.answer_chunk_ids]
    if not subset:
        return None, 0
    values = [
        float(
            evaluate_cases([rankings[case.case_id]], [case.answer_chunk_ids], RECALL_K).recall_any
        )
        for case in subset
    ]
    return round(sum(values) / len(values), 6), len(subset)


def _subset_metrics(
    cases: Sequence[RetrievalCase],
    rankings: dict[str, list[str]],
    *,
    predicate: Any,
    k: int,
) -> tuple[float, int]:
    subset = [case for case in cases if predicate(case)]
    if not subset:
        return 0.0, 0
    values = [
        float(evaluate_cases([rankings[case.case_id]], [case.gold_chunk_ids], k).recall_any)
        for case in subset
    ]
    return sum(values) / len(values), len(subset)


def _latency(
    pipeline_factory: Any,
    cases: Sequence[RetrievalCase],
    *,
    sample: int,
) -> tuple[float | None, float | None]:
    """p50/p95 of the *real* pipeline on a subsample (NFR-LAT-02 shape)."""
    if sample <= 0 or not cases:
        return None, None
    pipeline: RetrievalPipeline = pipeline_factory()
    timings: list[float] = []
    for case in cases[:sample]:
        _trace, measured = pipeline.retrieve_with_timings(case.query)
        timings.append(float(measured.get("total_ms", 0.0)))
    if not timings:
        return None, None
    ordered = sorted(timings)
    p50 = statistics.median(ordered)
    index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return round(p50, 3), round(ordered[index], 3)


def evaluate_arms(
    *,
    cases: Sequence[RetrievalCase],
    cache: dict[str, CandidateSet],
    params_for: Mapping[str, FusionParams],
    arms: Sequence[ArmSpec],
    config: PipelineConfig,
    latency_cases: Sequence[RetrievalCase] | None = None,
    latency_sample: int = 0,
    pipeline_for: Any = None,
) -> tuple[list[ArmMetrics], dict[str, dict[str, list[str]]]]:
    """Replay every arm and aggregate the metrics the release gates read."""
    reranker = LexicalOverlapReranker()
    rankings_by_arm: dict[str, dict[str, list[str]]] = {}
    results: list[ArmMetrics] = []

    for arm in arms:
        params = params_for[arm.fuser]
        rankings: dict[str, list[str]] = {}
        for case in cases:
            candidates = cache.get(case.case_id) or CandidateSet()
            arm_params = params
            rankings[case.case_id] = _ranked_ids(
                arm,
                candidates=candidates,
                params=arm_params,
                config=config,
                reranker=reranker,
                query=case.query,
            )
        rankings_by_arm[arm.name] = rankings

        gold = [case.gold_chunk_ids for case in cases]
        ordered = [rankings[case.case_id] for case in cases]
        wide = evaluate_cases(ordered, gold, RECALL_K)
        narrow = evaluate_cases(ordered, gold, NARROW_K)

        identifier_recall, identifier_n = _subset_metrics(
            cases, rankings, predicate=lambda case: case.is_identifier, k=RECALL_K
        )
        strict_recall, strict_n = _strict_metrics(cases, rankings)
        perturbed, perturbed_n = _subset_metrics(
            cases, rankings, predicate=lambda case: case.is_perturbed, k=RECALL_K
        )
        clean, clean_n = _subset_metrics(
            cases, rankings, predicate=lambda case: case.category == "direct", k=RECALL_K
        )
        delta = perturbed - clean if perturbed_n and clean_n else None

        per_locale: dict[str, float] = {}
        for locale in sorted({case.locale for case in cases}):
            value, _ = _subset_metrics(
                cases, rankings, predicate=lambda case, loc=locale: case.locale == loc, k=RECALL_K
            )
            per_locale[locale] = round(value, 6)

        p50, p95 = (
            _latency(pipeline_for, latency_cases or (), sample=latency_sample)
            if pipeline_for is not None
            else (None, None)
        )

        results.append(
            ArmMetrics(
                name=arm.name,
                fuser=arm.fuser,
                rerank=reranker.name if arm.rerank else None,
                mmr_lambda=(config.mmr_lambda or DEFAULT_MMR_LAMBDA) if arm.mmr else None,
                n=wide.n,
                recall_any_at_20=round(wide.recall_any, 6),
                recall_any_at_10=round(narrow.recall_any, 6),
                recall_all_at_20=round(wide.recall_all, 6),
                ndcg_at_10=round(narrow.ndcg, 6),
                mrr_at_10=round(narrow.mrr, 6),
                strict_recall_any_at_20=strict_recall,
                strict_n=strict_n,
                identifier_recall_any_at_20=(round(identifier_recall, 6) if identifier_n else None),
                identifier_n=identifier_n,
                perturbation_delta=(round(delta, 6) if delta is not None else None),
                per_locale=per_locale,
                latency_p50_ms=p50,
                latency_p95_ms=p95,
                note=arm.note,
            )
        )
    return results, rankings_by_arm


def arm_params(arm: ArmSpec, fitted: FusionArtifact | None) -> FusionParams:
    """The parameters an arm runs with: fitted where a fuser is used, fixed otherwise."""
    if arm.fuser == "none":
        return FusionParams(fuser="none")
    if fitted is not None:
        if arm.fuser == "rrf":
            return FusionParams(fuser="rrf", rrf_k=fitted.params.rrf_k)
        if fitted.params.fuser == "cc":
            return fitted.params
    return FusionParams(fuser=arm.fuser, rrf_k=60, alpha=0.75)


# ---------------------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------------------


def _logistic_observations(
    cases: Sequence[RetrievalCase],
    cache: dict[str, CandidateSet],
    config: PipelineConfig,
) -> list[FusionObservation]:
    """Training rows for the logistic fit: normalised channel scores + a gold label.

    Built from the **dev** candidates only (INV-09). Features are min–max normalised per
    channel per query, exactly what ``ConvexFusion`` consumes, so the fitted coefficients are
    directly usable as convex weights.
    """
    from answerdesk.retrieval.fusion import minmax_normalize

    observations: list[FusionObservation] = []
    for case in cases:
        candidates = cache.get(case.case_id)
        if candidates is None:
            continue
        sparse = candidates.sparse[: _limit(config)]
        dense = candidates.dense[: _limit(config)]
        sparse_scores = minmax_normalize([passage.score for passage in sparse])
        dense_scores = minmax_normalize([passage.score for passage in dense])
        features: dict[str, dict[str, float]] = {}
        for index, passage in enumerate(sparse):
            features.setdefault(passage.chunk.chunk_id, {})["bm25"] = float(sparse_scores[index])
        for index, passage in enumerate(dense):
            features.setdefault(passage.chunk.chunk_id, {})["dense"] = float(dense_scores[index])
        gold = set(case.gold_chunk_ids)
        for chunk_id, row in features.items():
            observations.append(
                FusionObservation(
                    features={"bm25": row.get("bm25", 0.0), "dense": row.get("dense", 0.0)},
                    label=1 if chunk_id in gold else 0,
                )
            )
    return observations


def _limit(config: PipelineConfig) -> int:
    return config.top_k


class NoiseCell(BaseModel):
    """One (slice, condition, arm) measurement in the noisy-input ablation."""

    model_config = ConfigDict(extra="forbid")

    slice: str
    condition: str
    arm: str
    n: int
    recall_any_at_20: float
    recall_any_at_10: float


class NoiseDelta(BaseModel):
    """``noisy − baseline`` for one (slice, arm), with the paired-bootstrap interval."""

    model_config = ConfigDict(extra="forbid")

    slice: str
    arm: str
    n: int
    delta_at_20: float
    delta_at_10: float
    ci_low: float | None = None
    ci_high: float | None = None
    significant: bool | None = None

    @property
    def improved(self) -> bool:
        return self.delta_at_20 > 0.0


class NoiseAblationReport(BaseModel):
    """Evidence that noisy-input handling helps and clean input does not regress."""

    model_config = ConfigDict(extra="forbid")

    noisy_n: int = 0
    clean_n: int = 0
    cells: list[NoiseCell] = Field(default_factory=list)
    deltas: list[NoiseDelta] = Field(default_factory=list)
    by_style: dict[str, dict[str, float]] = Field(default_factory=dict)
    clean_threshold: float = 0.01
    clean_non_regression_ok: bool = True
    bootstrap_samples: int = 0
    notes: list[str] = Field(default_factory=list)

    def delta(self, slice_name: str, arm: str) -> NoiseDelta | None:
        for delta in self.deltas:
            if delta.slice == slice_name and delta.arm == arm:
                return delta
        return None


#: Conditions compared on both slices: the pre-t23 analyser and the noisy-input analyser.
_NOISE_CONDITIONS: dict[str, QueryAnalysisConfig] = {
    "baseline": QueryAnalysisConfig(
        clause_segmentation=False,
        focus_probe=False,
        fuzzy_edit_distance=False,
        transliteration=False,
        max_expansions=3,
    ),
    "noisy": QueryAnalysisConfig(),
}


def _noise_arms() -> tuple[ArmSpec, ...]:
    return (
        ArmSpec(name="bm25", fuser="none", channels=("bm25",)),
        ArmSpec(name="dense", fuser="none", channels=("dense",)),
        ArmSpec(name="hybrid-rrf", fuser="rrf"),
    )


def run_noise_ablation(
    *,
    index: IndexBundle,
    cases: Sequence[RetrievalCase],
    params: FusionParams,
    config: PipelineConfig,
    bootstrap: int = 0,
    clean_sample: int = 600,
) -> NoiseAblationReport:
    """Measure the noisy-input analyser against the pre-t23 analyser on both slices.

    The same indexes, the same fusion parameters and the same arms are used in both
    conditions — only ``QueryAnalysisConfig`` changes — so a delta is attributable to the
    analysis, not to a different retriever. Candidates are computed once per condition and
    replayed through the arms, exactly like the main table.
    """
    noisy = [case for case in cases if case.category == NOISY_CATEGORY]
    clean = sorted(
        (case for case in cases if case.category in CLEAN_CATEGORIES),
        key=lambda case: case.case_id,
    )[:clean_sample]
    slices = {"noisy": noisy, "clean": clean}
    report = NoiseAblationReport(
        noisy_n=len(noisy),
        clean_n=len(clean),
        bootstrap_samples=bootstrap,
        notes=[
            "both conditions share the same index, fusion parameters and arms; only the "
            "query-analysis config differs (baseline = pre-t23 stages off)",
            "gold labels are document-level (the query's document plus its supporting docs)",
        ],
    )
    if not noisy or not clean:
        report.notes.append("ablation skipped: the corpus has no noisy or no clean slice")
        return report

    reranker = LexicalOverlapReranker()
    rankings: dict[tuple[str, str, str], dict[str, list[str]]] = {}
    for condition, analysis_config in _NOISE_CONDITIONS.items():
        pipeline = RetrievalPipeline(
            bm25=index.bm25,
            dense=index.dense,
            fuser=params.build(),
            analyzer=QueryAnalyzer(index.bm25.vocabulary, config=analysis_config),
            config=config,
        )
        cache = _candidate_sets(pipeline, [*noisy, *clean], top_k=config.top_k)
        for case_slice, case_list in slices.items():
            for arm in _noise_arms():
                rankings[(condition, case_slice, arm.name)] = {
                    case.case_id: _ranked_ids(
                        arm,
                        candidates=cache[case.case_id],
                        params=params,
                        config=config,
                        reranker=reranker,
                        query=case.query,
                    )
                    for case in case_list
                }

    for case_slice, case_list in slices.items():
        gold = [case.gold_chunk_ids for case in case_list]
        for arm in _noise_arms():
            cell_metrics: dict[str, list[float]] = {}
            for condition in _NOISE_CONDITIONS:
                ordered = [
                    rankings[(condition, case_slice, arm.name)][case.case_id] for case in case_list
                ]
                wide = evaluate_cases(ordered, gold, RECALL_K)
                narrow = evaluate_cases(ordered, gold, NARROW_K)
                report.cells.append(
                    NoiseCell(
                        slice=case_slice,
                        condition=condition,
                        arm=arm.name,
                        n=wide.n,
                        recall_any_at_20=round(wide.recall_any, 6),
                        recall_any_at_10=round(narrow.recall_any, 6),
                    )
                )
                cell_metrics[condition] = per_case_scores(ordered, gold, RECALL_K)

            noisy_scores = cell_metrics["noisy"]
            baseline_scores = cell_metrics["baseline"]
            delta20 = sum(noisy_scores) / len(noisy_scores) - sum(baseline_scores) / len(
                baseline_scores
            )
            narrow_delta = _delta_at(rankings, slices, case_slice, arm.name, case_list, NARROW_K)
            ci_low: float | None = None
            ci_high: float | None = None
            significant: bool | None = None
            if bootstrap > 0:
                _mean, ci_low, ci_high = paired_bootstrap_diff(
                    noisy_scores, baseline_scores, samples=bootstrap, seed=1337
                )
                significant = ci_low > 0.0 or ci_high < 0.0
            report.deltas.append(
                NoiseDelta(
                    slice=case_slice,
                    arm=arm.name,
                    n=len(case_list),
                    delta_at_20=round(delta20, 6),
                    delta_at_10=round(narrow_delta, 6),
                    ci_low=(round(ci_low, 6) if ci_low is not None else None),
                    ci_high=(round(ci_high, 6) if ci_high is not None else None),
                    significant=significant,
                )
            )

    # Clean-slice non-regression gate: every arm must stay within the threshold.
    failing = [
        delta.arm
        for delta in report.deltas
        if delta.slice == "clean" and delta.delta_at_20 < -report.clean_threshold
    ]
    report.clean_non_regression_ok = not failing
    if failing:
        report.notes.append(f"clean-slice regression beyond threshold in: {failing}")

    hybrid = next(arm for arm in _noise_arms() if arm.name == "hybrid-rrf")
    for style in sorted({case.style for case in noisy if case.style}):
        style_cases = [case for case in noisy if case.style == style]
        style_gold = [case.gold_chunk_ids for case in style_cases]
        row: dict[str, float] = {"n": float(len(style_cases))}
        for condition in _NOISE_CONDITIONS:
            ordered = [
                rankings[(condition, "noisy", hybrid.name)][case.case_id] for case in style_cases
            ]
            row[condition] = round(evaluate_cases(ordered, style_gold, RECALL_K).recall_any, 6)
        report.by_style[style] = row
    return report


def _delta_at(
    rankings: dict[tuple[str, str, str], dict[str, list[str]]],
    slices: dict[str, list[RetrievalCase]],
    case_slice: str,
    arm_name: str,
    case_list: Sequence[RetrievalCase],
    k: int,
) -> float:
    gold = [case.gold_chunk_ids for case in case_list]
    values = {}
    for condition in _NOISE_CONDITIONS:
        ordered = [rankings[(condition, case_slice, arm_name)][case.case_id] for case in case_list]
        values[condition] = sum(per_case_scores(ordered, gold, k)) / len(case_list)
    return values["noisy"] - values["baseline"]


def _ensure_corpus(corpus_dir: Path, echo: Any) -> None:
    if (corpus_dir / "manifest.json").is_file():
        return
    from answerdesk.ingest.builder import build_demo
    from answerdesk.ingest.cli import synthetic_config_from_env

    echo(f"no corpus at {corpus_dir}; building the demo corpus first")
    build_demo(corpus_dir, config=synthetic_config_from_env())


def run_ablation(
    settings: Settings,
    *,
    corpus_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    arms_path: Path | str | None = None,
    fit: bool = True,
    fit_method: Literal["coordinate_search", "logistic_regression"] = "coordinate_search",
    bootstrap: int = 0,
    latency_sample: int = 20,
    noise_ablation: bool = True,
    noise_clean_sample: int = 600,
    echo: Any = print,
) -> AblationReport:
    """Fit on dev, evaluate every arm on test, write the artifact and return the report.

    ``fit_method`` selects the documented procedure (FR-RET-05): a coordinate search over
    ``rrf_k``/``alpha``, or a logistic regression on per-channel score features that yields
    explicit *learned weights* for the convex combination. Both fit on the dev split only.
    """
    started = time.perf_counter()
    arms = load_arms(arms_path) if arms_path is not None else DEFAULT_ARMS
    corpus = (
        Path(corpus_dir)
        if corpus_dir is not None
        else settings.resolve_path(settings.app.corpus_dir)
    )
    _ensure_corpus(corpus, echo)
    target = Path(out_dir) if out_dir is not None else index_root(settings) / "ablation"

    store = ChunkStore.from_corpus(corpus)
    embedder = _embedder(settings)
    index = IndexBundle.build(store, embedder)
    chunk_hash = store.fingerprint()

    cases = build_cases(store.chunks, load_query_rows(corpus))
    if not cases:
        raise CorpusMissingError(f"no usable queries under {corpus}")
    bundle: SplitBundle = split_cases(cases)
    if not bundle.dev or not bundle.test:  # pragma: no cover - corpus is large enough
        raise CorpusMissingError(
            "the dev/test split produced an empty side; use a larger corpus "
            f"(dev={len(bundle.dev)}, test={len(bundle.test)})"
        )

    config = PipelineConfig(
        top_k=settings.retrieval.top_k,
        gate_k=max(settings.retrieval.gate_k, RECALL_K),
        final_k=settings.generate.context_k_max,
        dedupe_cosine=settings.retrieval.dedupe_cosine,
        rerank_top_in=settings.retrieval.rerank.top_in,
        rerank_top_out=settings.retrieval.gate_k,
        mmr_lambda=settings.retrieval.mmr_lambda,
    )
    base_pipeline = RetrievalPipeline(
        bm25=index.bm25,
        dense=index.dense,
        fuser=FusionParams(fuser="rrf").build(),
        config=config,
    )

    dev_cache = _candidate_sets(base_pipeline, bundle.dev, top_k=config.top_k)
    test_cache = _candidate_sets(base_pipeline, bundle.test, top_k=config.top_k)

    def evaluate_on_dev(params: FusionParams) -> float:
        """Dev-only objective: nDCG@10 of the fused ranking (tie-break: recall_any@20)."""
        reranker = LexicalOverlapReranker()
        arm = ArmSpec(name="fit", fuser=params.fuser)
        ndcg_values: list[float] = []
        for case in bundle.dev:
            ranked = _ranked_ids(
                arm,
                candidates=dev_cache[case.case_id],
                params=params,
                config=config,
                reranker=reranker,
                query=case.query,
            )
            ndcg_values.append(evaluate_cases([ranked], [case.gold_chunk_ids], NARROW_K).ndcg)
        return sum(ndcg_values) / len(ndcg_values)

    observations: list[FusionObservation] | None = None
    if fit and fit_method == "logistic_regression":
        observations = _logistic_observations(bundle.dev, dev_cache, config)
    fitted: FusionArtifact | None = None
    if fit:
        dev_prov = _provenance(bundle.dev, "dev", bundle.salt)
        test_prov = _provenance(bundle.test, "test", bundle.salt)
        fitted = fit_fusion_artifact(
            evaluate=evaluate_on_dev,
            dev_split=dev_prov,
            test_split=test_prov,
            dev_item_ids=[case.case_id for case in bundle.dev],
            test_item_ids=[case.case_id for case in bundle.test],
            corpus=CorpusProvenance(version=store.corpus_version, hash=chunk_hash),
            embedder=EmbedderInfo(name=embedder.name, dimension=embedder.dimension),
            observations=observations,
            method=fit_method,
            metric="nDCG@10",
            notes=[
                "objective: dev nDCG@10; tie-break rule: convex combination wins ties",
                "labels: document-level gold (every chunk of the source document)",
            ],
        )
        assert dev_prov.hash == split_hash([case.case_id for case in bundle.dev])
        fitted.save(target / "fusion.json")
        echo(
            f"fitted fusion: fuser={fitted.params.fuser} "
            f"rrf_k={fitted.params.rrf_k} alpha={fitted.params.alpha} "
            f"dev {fitted.metric}={fitted.metric_value}"
        )

    params_for: dict[str, FusionParams] = {arm.fuser: arm_params(arm, fitted) for arm in arms}

    def pipeline_for_arm(spec: ArmSpec) -> RetrievalPipeline:
        params = params_for[spec.fuser]
        arm_config = config.model_copy(update={"channels": spec.channels})
        return RetrievalPipeline(
            bm25=index.bm25,
            dense=index.dense,
            fuser=params.build(),
            reranker=LexicalOverlapReranker() if spec.rerank else None,
            diversifier=(
                MMRDiversifier(
                    lambda_=config.mmr_lambda or DEFAULT_MMR_LAMBDA,
                    similarity=vector_similarity(index.dense.vector_for),
                )
                if spec.mmr
                else None
            ),
            config=arm_config,
        )

    arms_metrics: list[ArmMetrics] = []
    rankings_by_arm: dict[str, dict[str, list[str]]] = {}
    for spec in arms:
        measured, rankings = evaluate_arms(
            cases=bundle.test,
            cache=test_cache,
            params_for=params_for,
            arms=(spec,),
            config=config,
            latency_cases=bundle.test,
            latency_sample=latency_sample,
            pipeline_for=(lambda spec=spec: pipeline_for_arm(spec)),
        )
        arms_metrics.extend(measured)
        rankings_by_arm.update(rankings)

    noise_report: NoiseAblationReport | None = None
    if noise_ablation:
        noise_report = run_noise_ablation(
            index=index,
            cases=cases,
            params=params_for["rrf"],
            config=config,
            bootstrap=bootstrap,
            clean_sample=noise_clean_sample,
        )
        echo(
            f"noise ablation: noisy n={noise_report.noisy_n} "
            f"clean n={noise_report.clean_n} "
            f"clean_non_regression_ok={noise_report.clean_non_regression_ok}"
        )

    uplift, ci = _hybrid_uplift(bundle.test, rankings_by_arm, bootstrap=bootstrap)
    uplift_strict, _ = _hybrid_uplift(bundle.test, rankings_by_arm, bootstrap=0, strict=True)

    report = AblationReport(
        corpus_version=store.corpus_version,
        corpus_chunk_hash=chunk_hash,
        embedder=EmbedderInfo(name=embedder.name, dimension=embedder.dimension),
        eval_split=_provenance(bundle.test, "test", bundle.salt),
        dev_split=_provenance(bundle.dev, "dev", bundle.salt),
        split_salt=bundle.salt,
        fitted=fitted,
        arms=arms_metrics,
        hybrid_uplift=(round(uplift, 6) if uplift is not None else None),
        hybrid_uplift_ci=[round(value, 6) for value in ci],
        hybrid_uplift_strict=(round(uplift_strict, 6) if uplift_strict is not None else None),
        noise_ablation=noise_report,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        timings={"total_ms": round((time.perf_counter() - started) * 1000.0, 3)},
        notes=[
            MACHINERY_CAVEAT,
            "labels: document-level gold; strict (answer-chunk) labels are used by the "
            "golden-set evaluation, not by this runner",
            "latency measured on a subsample with the real pipeline; metrics replayed from "
            "cached candidates",
        ],
    )
    target.mkdir(parents=True, exist_ok=True)
    (target / "retrieval-ablation.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    table = format_table(report)
    (target / "retrieval-metrics.md").write_text(table + "\n", encoding="utf-8", newline="\n")
    echo(table)
    echo(f"artifacts: {target}")
    return report


def _provenance(cases: Sequence[RetrievalCase], name: str, salt: str) -> SplitProvenance:
    ids = [case.case_id for case in cases]
    return SplitProvenance(
        name=name,  # type: ignore[arg-type]
        hash=split_hash(ids),
        n_items=len(ids),
        n_documents=len({case.gold_document for case in cases}),
        salt=salt,
    )


def _hybrid_uplift(
    cases: Sequence[RetrievalCase],
    rankings_by_arm: dict[str, dict[str, list[str]]],
    *,
    bootstrap: int,
    strict: bool = False,
) -> tuple[float | None, list[float]]:
    """RT-7: hybrid recall_any@20 minus the best single channel, with a paired CI."""
    singles = [name for name in ("bm25", "dense") if name in rankings_by_arm]
    hybrid = next((name for name in rankings_by_arm if name.startswith("hybrid")), None)
    if not singles or hybrid is None:
        return None, []
    eligible = [
        (case, gold_ids)
        for case, gold_ids in zip(cases, _gold_lists(cases, strict=strict), strict=True)
        if gold_ids
    ]
    if not eligible:
        return None, []
    case_ids = [case.case_id for case, _ in eligible]
    gold = [gold_ids for _, gold_ids in eligible]
    hybrid_scores = per_case_scores(
        [rankings_by_arm[hybrid][case_id] for case_id in case_ids], gold, RECALL_K
    )
    best_single_name = max(
        singles,
        key=lambda name: sum(
            per_case_scores(
                [rankings_by_arm[name][case_id] for case_id in case_ids], gold, RECALL_K
            )
        ),
    )
    single_scores = per_case_scores(
        [rankings_by_arm[best_single_name][case_id] for case_id in case_ids], gold, RECALL_K
    )
    uplift = sum(hybrid_scores) / len(hybrid_scores) - sum(single_scores) / len(single_scores)
    if bootstrap <= 0:
        return uplift, []
    _mean, low, high = paired_bootstrap_diff(
        hybrid_scores, single_scores, samples=bootstrap, seed=1337
    )
    return uplift, [low, high]


def _embedder(settings: Settings) -> Any:
    from answerdesk.index.embedders import default_embedder

    return default_embedder(settings)


def format_table(report: AblationReport) -> str:
    """Markdown table + the provenance header every retrieval number must carry."""
    lines = [
        "### Retrieval ablation",
        "",
        f"- corpus: `{report.corpus_version}` (chunk hash `{report.corpus_chunk_hash[:16]}`)",
        f"- embedder: `{report.embedder.name}` (dimension {report.embedder.dimension})",
        f"- test split: {report.eval_split.n_items} queries / "
        f"{report.eval_split.n_documents} documents (hash `{report.eval_split.hash[:16]}`)",
        f"- dev split: {report.dev_split.n_items} queries (hash `{report.dev_split.hash[:16]}`)",
    ]
    if report.fitted is not None:
        params = report.fitted.params
        lines.append(
            f"- fitted on dev: fuser=`{params.fuser}` rrf_k={params.rrf_k} "
            f"alpha={params.alpha} ({report.fitted.metric}={report.fitted.metric_value})"
        )
    if report.embedder.name.startswith("hashing"):
        lines.append(f"- **caveat**: {MACHINERY_CAVEAT}")
    lines.extend(
        [
            "",
            "| arm | recall_any@20 | strict recall@20 | recall_any@10 | recall_all@20 "
            "| nDCG@10 | MRR@10 | id recall@20 | perturb. delta | p50 ms | p95 ms | n |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for arm in report.arms:
        identifier = (
            f"{arm.identifier_recall_any_at_20:.3f} (n={arm.identifier_n})"
            if arm.identifier_recall_any_at_20 is not None
            else "n/a"
        )
        strict = (
            f"{arm.strict_recall_any_at_20:.3f} (n={arm.strict_n})"
            if arm.strict_recall_any_at_20 is not None
            else "n/a"
        )
        delta = f"{arm.perturbation_delta:+.3f}" if arm.perturbation_delta is not None else "n/a"
        p50 = f"{arm.latency_p50_ms:.1f}" if arm.latency_p50_ms is not None else "n/a"
        p95 = f"{arm.latency_p95_ms:.1f}" if arm.latency_p95_ms is not None else "n/a"
        lines.append(
            f"| {arm.name} | {arm.recall_any_at_20:.3f} | {strict} "
            f"| {arm.recall_any_at_10:.3f} "
            f"| {arm.recall_all_at_20:.3f} | {arm.ndcg_at_10:.3f} | {arm.mrr_at_10:.3f} "
            f"| {identifier} | {delta} | {p50} | {p95} | {arm.n} |"
        )
    if report.hybrid_uplift is not None:
        interval = (
            f", 95% CI [{report.hybrid_uplift_ci[0]:+.3f}, {report.hybrid_uplift_ci[1]:+.3f}]"
            if len(report.hybrid_uplift_ci) == 2
            else " (CI not requested: --bootstrap N)"
        )
        lines.append("")
        lines.append(
            f"**RT-7 hybrid uplift** (best hybrid − best single channel, recall_any@20): "
            f"{report.hybrid_uplift:+.3f}{interval}"
        )
        if report.hybrid_uplift_strict is not None:
            lines.append(
                f"**RT-7 on the strict label** (heuristic answer chunk): "
                f"{report.hybrid_uplift_strict:+.3f}"
            )
    primary = next((arm for arm in report.arms if arm.name == report.primary_arm), None)
    if primary is not None and primary.per_locale:
        lines.extend(["", "Per-locale `recall_any@20` (primary arm):", ""])
        header = "| " + " | ".join(sorted(primary.per_locale)) + " | macro |"
        divider = "|" + "---|" * (len(primary.per_locale) + 1)
        values = "| " + " | ".join(
            f"{primary.per_locale[locale]:.3f}" for locale in sorted(primary.per_locale)
        )
        lines.extend([header, divider, f"{values} | {macro_average(primary.per_locale):.3f} |"])
    lines.extend(_format_noise_table(report.noise_ablation))
    for note in report.notes:
        lines.append(f"> {note}")
    return "\n".join(lines)


def _format_noise_table(noise: NoiseAblationReport | None) -> list[str]:
    """Markdown for the noisy-input ablation: deltas first, then the per-style breakdown."""
    if noise is None:
        return []
    lines = [
        "",
        "### Noisy-input ablation (t23)",
        "",
        f"- noisy slice: {noise.noisy_n} queries (`noisy_freeform`), "
        f"clean slice: {noise.clean_n} queries",
        f"- clean non-regression gate (|Δ| ≤ {noise.clean_threshold:.2f}): "
        f"**{'OK' if noise.clean_non_regression_ok else 'FAIL'}**",
        "",
        "| slice | arm | n | baseline recall@20 | noisy recall@20 | Δ@20 | 95% CI | "
        "baseline@10 | noisy@10 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    cells = {(cell.slice, cell.condition, cell.arm): cell for cell in noise.cells}
    for delta in noise.deltas:
        baseline = cells[(delta.slice, "baseline", delta.arm)]
        noisy = cells[(delta.slice, "noisy", delta.arm)]
        interval = (
            f"[{delta.ci_low:+.3f}, {delta.ci_high:+.3f}]"
            if delta.ci_low is not None and delta.ci_high is not None
            else "n/a"
        )
        lines.append(
            f"| {delta.slice} | {delta.arm} | {delta.n} | {baseline.recall_any_at_20:.3f} "
            f"| {noisy.recall_any_at_20:.3f} | {delta.delta_at_20:+.3f} | {interval} "
            f"| {baseline.recall_any_at_10:.3f} | {noisy.recall_any_at_10:.3f} |"
        )
    if noise.by_style:
        lines.extend(
            [
                "",
                "Per-style `recall_any@20` (hybrid arm):",
                "",
                "| style | n | baseline | noisy | Δ |",
                "|---|---|---|---|---|",
            ]
        )
        for style, style_row in sorted(noise.by_style.items()):
            shift = style_row["noisy"] - style_row["baseline"]
            lines.append(
                f"| {style} | {int(style_row['n'])} | {style_row['baseline']:.3f} "
                f"| {style_row['noisy']:.3f} | {shift:+.3f} |"
            )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m answerdesk.retrieval.ablation`` — exit 0 on a completed run."""
    parser = argparse.ArgumentParser(
        prog="python -m answerdesk.retrieval.ablation",
        description="Run the retrieval ablation arms on the demo/dev corpus.",
    )
    parser.add_argument("--corpus", type=Path, default=None, help="corpus directory")
    parser.add_argument("--out", type=Path, default=None, help="artifact directory")
    parser.add_argument(
        "--arms", type=Path, default=None, help="arms file (default: built-in arms)"
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--profile", default=None, help="offline | api | local")
    parser.add_argument(
        "--no-fit", action="store_true", help="use config defaults instead of fitting"
    )
    parser.add_argument(
        "--no-noise-ablation",
        action="store_true",
        help="skip the noisy-input ablation (t23 evidence section)",
    )
    parser.add_argument(
        "--noise-clean-sample",
        type=int,
        default=600,
        help="clean-slice queries used for the noisy ablation non-regression gate",
    )
    parser.add_argument(
        "--method",
        choices=("coordinate_search", "logistic_regression"),
        default="coordinate_search",
        help="dev-split fitting procedure",
    )
    parser.add_argument("--bootstrap", type=int, default=0, help="paired-bootstrap samples (0=off)")
    parser.add_argument("--latency-sample", type=int, default=20, help="queries timed per arm")
    args = parser.parse_args(argv)

    settings = load_settings(config_path=args.config, profile=args.profile)
    report = run_ablation(
        settings,
        corpus_dir=args.corpus,
        out_dir=args.out,
        arms_path=args.arms,
        fit=not args.no_fit,
        fit_method=args.method,
        bootstrap=args.bootstrap,
        latency_sample=args.latency_sample,
        noise_ablation=not args.no_noise_ablation,
        noise_clean_sample=args.noise_clean_sample,
        echo=lambda message: print(message, flush=True),
    )
    if not report.arms:  # pragma: no cover - defensive
        print("no arms were evaluated", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
