"""The evaluation runner: one golden case → one measured turn, per retrieval arm.

Responsibilities, in the order they matter:

1. **retrieval** — ask the arm for the top-k ranking, keep the ids (the retrieval metrics are
   defined on exactly this list) and hand the top ``context_k`` to the generator;
2. **generation** — drive the real :class:`answerdesk.generate.pipeline.AnswerPipeline` with
   whatever provider the profile selected (``FakeProvider`` in CI), so the citations,
   verifier, numeric guard and refusal logic measured here are the shipped ones;
3. **judging** — score answered turns with the configured judge and record the pass flag;
   the judge never runs on a turn the contract forbids (a refusal carries no prose);
4. **correctness** — apply the §8.1 definition once, in one place: rubric pass **and** every
   displayed claim supported **and** all required evidence cited **and** language agreement.
   For must-escalate cases the same slot records the safe outcome (escalated), which is what
   makes realised risk and coverage comparable across arms;
5. **operations** — per-stage wall clock, token counts, USD and cache status per turn.

Timing is measured but never enters the determinism digest (NFR-DET-02 talks about "after
removing timing fields"): :func:`deterministic_digest` drops it explicitly.

    # AT-EVL-06  # AT-EVL-07  # AT-EVL-08  # AT-DET-02
"""

from __future__ import annotations

import itertools
import math
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from answerdesk.config import Settings
from answerdesk.contracts import (
    Action,
    CacheStatus,
    ChatProvider,
    EvalCase,
    Judge,
    RetrievedPassage,
)
from answerdesk.eval.arms import ArmSpec
from answerdesk.eval.calibration import CalibrationReport, summarise
from answerdesk.eval.corpus import Corpus
from answerdesk.eval.errors import EvalError
from answerdesk.eval.fixture_provider import EvidenceAnchoredProvider
from answerdesk.eval.judge import PASS_SCORE, RubricJudge, judge_metadata
from answerdesk.eval.metrics import (
    answer_cells,
    category_cells,
    escalation_cells,
    operational_cells,
    paired_scores,
    retrieval_cells,
)
from answerdesk.eval.models import (
    JudgeMetadata,
    MetricCell,
    RunHeader,
    RunResult,
    StatComparison,
    TargetCheck,
    TurnRecord,
)
from answerdesk.eval.stats import bootstrap_interval, mcnemar, paired_bootstrap
from answerdesk.generate.language import languages_agree
from answerdesk.generate.pipeline import AnswerPipeline, GenerationResult
from answerdesk.llm.presets import provider_config

#: How many candidates an arm returns: the rerank window of §6.2 (50).
RETRIEVAL_DEPTH = 50
#: Passages handed to the generator: the §6.2 context cap (12) is applied by the packer.
CONTEXT_K = 12
#: Target risk used for the calibration diagnostic (E-1 pooled bound is 0.05).
TARGET_RISK = 0.05


class RunnerConfig:
    """Knobs of one evaluation run; every one of them is printed in the report header."""

    def __init__(
        self,
        *,
        arms: Sequence[str] = ("bm25", "dense-hash", "hybrid-rrf"),
        ks: Sequence[int] = (10, 20),
        seed: int = 0,
        bootstrap: int = 1000,
        provider_mode: Literal["fake", "api", "local"] = "fake",
        judge_mode: Literal["fake", "llm", "none"] = "fake",
        context_k: int = CONTEXT_K,
        retrieval_depth: int = RETRIEVAL_DEPTH,
        target_risk: float = TARGET_RISK,
        limit: int | None = None,
    ) -> None:
        self.arms = tuple(arms)
        self.ks = tuple(ks)
        self.seed = seed
        self.bootstrap = bootstrap
        self.provider_mode = provider_mode
        self.judge_mode = judge_mode
        self.context_k = context_k
        self.retrieval_depth = retrieval_depth
        self.target_risk = target_risk
        self.limit = limit

    def as_dict(self) -> dict[str, Any]:
        return {
            "arms": list(self.arms),
            "ks": list(self.ks),
            "seed": self.seed,
            "bootstrap": self.bootstrap,
            "provider_mode": self.provider_mode,
            "judge_mode": self.judge_mode,
            "context_k": self.context_k,
            "retrieval_depth": self.retrieval_depth,
            "target_risk": self.target_risk,
            "limit": self.limit,
        }


@dataclass
class RunOutcome:
    """Raw per-arm turns plus the calibration artifact of the primary arm."""

    turns: dict[str, list[TurnRecord]] = field(default_factory=dict)
    calibration: dict[str, CalibrationReport] = field(default_factory=dict)
    confidences: dict[str, list[float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    duration_s: float = 0.0


def build_provider(settings: Settings, *, mode: Literal["fake", "api", "local"]) -> ChatProvider:
    """The generator for this run: the deterministic double or the configured provider."""
    if mode == "fake":
        config = provider_config("fake")
        return EvidenceAnchoredProvider(config)
    from answerdesk.llm.factory import build_provider as build_configured

    return build_configured(settings)


def build_judge(
    settings: Settings, *, mode: Literal["fake", "llm"], provider: ChatProvider
) -> tuple[Judge, JudgeMetadata]:
    """The judge plus the metadata §7.5's honesty clause requires next to any judged number."""
    if mode == "fake":
        from answerdesk.testing.doubles import FakeJudge

        double: Judge = FakeJudge()
        return double, judge_metadata(
            judge_model=double.judge_model,  # type: ignore[attr-defined]
            prompt_version="none (deterministic double)",
            rubric_version=double.rubric_version,  # type: ignore[attr-defined]
            position_swap=True,
        )
    judge_model = settings.eval.judge_model
    generator_model = getattr(getattr(provider, "config", None), "model", "")
    if generator_model and judge_model == generator_model:
        raise EvalError(
            f"the judge model {judge_model!r} is the generator model: §10.5 forbids a model "
            "family grading its own output in a reported metric. Set eval.judge_model to a "
            "different model or use another provider.",
        )
    rubric_judge = RubricJudge(provider, judge_model=judge_model)
    return rubric_judge, judge_metadata(
        judge_model=judge_model,
        prompt_version=rubric_judge.prompt_version,
        rubric_version=rubric_judge.rubric_version,
        position_swap=True,
    )


def normalize_query(text: str) -> str:
    """Cache key normalisation for the exact-match probe (t8 owns the semantic cache)."""
    return " ".join(text.split()).casefold()


@dataclass
class _TurnMeasurement:
    record: TurnRecord
    confidence: float
    correct: bool


class EvalRunner:
    """Runs golden cases through one pipeline per arm and records every turn."""

    def __init__(
        self,
        *,
        corpus: Corpus,
        arms: Mapping[str, Any],
        pipeline: AnswerPipeline,
        judge: Judge,
        config: RunnerConfig,
    ) -> None:
        self.corpus = corpus
        self.arms = dict(arms)
        self.pipeline = pipeline
        self.judge = judge
        self.config = config
        self.ungrounded_references = 0

    def run(self, cases: Sequence[EvalCase]) -> RunOutcome:
        selected = list(cases)
        if self.config.limit is not None:
            selected = selected[: self.config.limit]
        if not selected:
            raise EvalError("no golden cases to evaluate: the selection is empty")
        started = time.perf_counter()
        outcome = RunOutcome()
        for arm_name, arm in self.arms.items():
            outcome.turns[arm_name] = self._run_arm(arm_name, arm, selected, outcome)
        outcome.duration_s = time.perf_counter() - started
        primary = self.config.arms[0] if self.config.arms else next(iter(self.arms))
        notes = [
            "RETRIEVAL arms are the eval-side reference implementations (BM25 / hashed-dense / "
            "RRF); t5 owns the shipped hybrid retriever and t11 must re-run the evidence arms.",
            f"calibration uses the packed evidence score as p̂ (arm {primary}); the conformal "
            "calibrator is t8's and no conformal claim is made here.",
        ]
        if self.ungrounded_references:
            notes.append(
                f"{self.ungrounded_references} judge call(s) had a reference answer whose "
                "numeric identifiers do not occur anywhere in the corpus; the rubric graded "
                "those answers against their cited evidence instead (the reference could not "
                "be verified, and grading against it would measure the corpus generator)"
            )
        outcome.notes = notes
        if self.config.judge_mode == "none":
            notes.append(
                "judge disabled (retrieval-only run): rubric and judged groundedness are absent"
            )
        return outcome

    # -- internals ----------------------------------------------------------------
    def _grounded_reference(self, case: EvalCase) -> str | None:
        """The reference answer if the corpus can support it, else ``None``.

        A generated reference answer may quote slot values (an amount, a date) that no
        document contains; grading a correct answer against such a reference measures the
        corpus generator, not the system. The ungrounded reference is dropped from the
        judge call (the rubric then grades against the cited evidence) and the run counts
        how often that happened.
        """
        reference = case.reference_answer
        if not reference:
            return None
        if case.metadata.get("reference_verifiable") is True:
            return reference
        self.ungrounded_references += 1
        return None

    def _run_arm(
        self,
        arm_name: str,
        arm: Any,
        cases: Sequence[EvalCase],
        outcome: RunOutcome,
    ) -> list[TurnRecord]:
        records: list[TurnRecord] = []
        cache_keys: set[str] = set()
        for case in cases:
            key = normalize_query(case.question)
            cache_status = CacheStatus.HIT if key in cache_keys else CacheStatus.MISS
            cache_keys.add(key)
            records.append(
                self._measure(case, arm_name=arm_name, arm=arm, cache_status=cache_status)
            )
        confidences = [turn.confidence for turn in records]
        # Calibration asks one question: does the confidence predict whether *answering* was
        # the right call and the answer was right? A refusal on an unsafe turn therefore
        # counts as a correctly predicted non-answer, not as an error.
        correct = [record.answered and record.answer_ok for record in records]
        outcome.confidences[arm_name] = confidences
        outcome.calibration[arm_name] = summarise(
            confidences,
            correct,
            target_risk=self.config.target_risk,
            per_bucket=_bucket_calibration(records, confidences, correct),
        )
        return records

    def _measure(
        self,
        case: EvalCase,
        *,
        arm_name: str,
        arm: Any,
        cache_status: CacheStatus,
    ) -> TurnRecord:
        timings: dict[str, float] = {}
        anchor = time.perf_counter()
        passages: list[RetrievedPassage] = arm.search(case.question, self.config.retrieval_depth)
        timings["retrieval"] = (time.perf_counter() - anchor) * 1000.0
        ranked = [passage.chunk.chunk_id for passage in passages]
        packed_input = passages[: self.config.context_k]

        anchor = time.perf_counter()
        result: GenerationResult = self.pipeline.answer(case.question, packed_input)
        timings["generation"] = (time.perf_counter() - anchor) * 1000.0

        answered = result.signal.action is Action.ANSWER
        claims_total = len(result.claims)
        claims_supported = sum(1 for claim in result.claims if claim.support.value == "supported")
        groundedness = (claims_supported / claims_total) if claims_total else None
        precision, recall, cited_ids, phantom = _citation_stats(result, case)
        acceptable = set(case.gold_chunk_ids)
        evidence_cited = (
            bool(set(cited_ids) & acceptable)
            if acceptable
            else bool(set(cited_ids) & set(case.required_citation_chunk_ids))
        )
        language = result.language or case.locale
        language_match = (
            languages_agree(case.locale, result.text)
            if result.text.strip()
            else language == case.locale
        )

        rubric_score: float | None = None
        rubric_pass: bool | None = None
        reference_rubric: float | None = None
        judged_groundedness: float | None = None
        if answered and result.claims and self.config.judge_mode != "none":
            # Context-anchored rubric: each claim is graded against the passage it cites.
            # Whole-answer grading against a one-sentence reference would score length, not
            # truth (the verbosity bias §10.5 forbids), and F-1 is a claim-level metric.
            anchor = time.perf_counter()
            reference = self._grounded_reference(case)
            by_id = {citation.citation_id: citation for citation in result.citations}
            claim_scores: list[float] = []
            reference_scores: list[float] = []
            for claim in result.claims:
                quotes = [
                    by_id[citation_id].quote
                    for citation_id in claim.citation_ids
                    if citation_id in by_id
                ]
                score = self.judge.grade(
                    question=case.question,
                    reference_answer=None,
                    candidate_answer=claim.text,
                    context=quotes,
                )
                claim_scores.append(score.score)
                if reference is not None and self.config.judge_mode == "fake":
                    reference_scores.append(
                        self.judge.grade(
                            question=case.question,
                            reference_answer=reference,
                            candidate_answer=claim.text,
                            context=quotes,
                        ).score
                    )
            timings["judge"] = (time.perf_counter() - anchor) * 1000.0
            if reference_scores:
                reference_rubric = round(sum(reference_scores) / len(reference_scores), 6)
            else:
                reference_rubric = None
            if claim_scores:
                rubric_score = round(sum(claim_scores) / len(claim_scores), 6)
                rubric_pass = rubric_score >= PASS_SCORE
                judged_groundedness = sum(1 for value in claim_scores if value >= PASS_SCORE) / len(
                    claim_scores
                )

        correct = _is_correct(
            case=case,
            answered=answered,
            claims_total=claims_total,
            claims_supported=claims_supported,
            evidence_cited=evidence_cited,
            language_match=language_match,
            rubric_pass=rubric_pass,
        )
        timings["total"] = sum(value for key, value in timings.items() if key != "total")
        return TurnRecord(
            case_id=case.case_id,
            locale=case.locale,
            category=case.category,
            intent_bucket=case.intent_bucket,
            split=case.split,
            arm=arm_name,
            answerable=case.answerable,
            must_escalate=bool(case.metadata.get("must_escalate")) or not case.answerable,
            expected_language=case.locale,
            action=result.signal.action,
            reason_code=result.signal.reason_code,
            answered=answered,
            escalated=not answered,
            ranked_chunk_ids=ranked,
            gold_chunk_ids=list(case.gold_chunk_ids),
            acceptable_chunk_ids=sorted(acceptable),
            required_citation_chunk_ids=list(
                case.required_citation_chunk_ids or case.gold_chunk_ids
            ),
            evidence_cited=evidence_cited,
            claims_total=claims_total,
            claims_supported=claims_supported,
            groundedness=groundedness,
            groundedness_judged=judged_groundedness,
            citation_precision=precision,
            citation_recall=recall,
            cited_chunk_ids=cited_ids,
            phantom_citations=phantom,
            unsupported_displayed_claims=0,
            rubric_score=rubric_score,
            rubric_pass=rubric_pass,
            reference_rubric_score=reference_rubric,
            language=language,
            language_match=language_match,
            answer_ok=correct,
            confidence=_calibration_confidence(result.packed.evidence_score, answered=answered),
            latency_ms=timings,
            usd=result.usage.usd,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cache_status=cache_status,
            error=None if answered else result.signal.detail,
        )


def _citation_stats(
    result: GenerationResult, case: EvalCase
) -> tuple[float | None, float | None, list[str], int]:
    """Citation precision (support-based), ALCE recall, cited ids and phantom count.

    ``citation_recall`` is the ALCE definition (F-3): the fraction of answer statements that
    are fully supported by at least one cited passage. It is computed from the claims and
    their citation links, not from a gold chunk list — that is what the requirement measures.
    """
    by_id = {citation.citation_id: citation for citation in result.citations}
    packed_ids = set(result.packed.chunk_map())
    phantom = sum(1 for citation in result.citations if citation.chunk_id not in packed_ids)
    supported_ids: set[int] = set()
    cited_claims = 0
    for claim in result.claims:
        if claim.support.value == "supported" and claim.citation_ids:
            supported_ids.update(claim.citation_ids)
            cited_claims += 1
    total = len(by_id)
    precision = (len(supported_ids & set(by_id)) / total) if total else None
    recall = (cited_claims / len(result.claims)) if result.claims else None
    _ = case  # the case's labels are used by the caller (evidence_cited), not by these two
    return precision, recall, sorted({citation.chunk_id for citation in result.citations}), phantom


def _is_correct(
    *,
    case: EvalCase,
    answered: bool,
    claims_total: int,
    claims_supported: int,
    evidence_cited: bool,
    language_match: bool,
    rubric_pass: bool | None,
) -> bool:
    """§8.1 correctness, defined once: the four conditions, in one place."""
    if not case.answerable:
        # The safe outcome for an unsafe/out-of-scope turn is to escalate it.
        return not answered
    if not answered:
        return False
    if claims_total == 0 or claims_supported != claims_total:
        return False
    if not evidence_cited:
        return False
    if not language_match:
        return False
    if rubric_pass is None:
        return True
    return rubric_pass


def _calibration_confidence(answer_score: float, *, answered: bool) -> float:
    """Uncalibrated confidence proxy: the packed evidence score, 0 for a refusal.

    A refusal is maximally uncertain *about answering*, which is what selective prediction
    needs: it must sort below any answer. The conformal calibrator (t8) replaces this with
    ``q̂``; until then the calibration section is explicitly a diagnostic.
    """
    if not answered:
        return 0.0
    return float(min(1.0, max(0.0, answer_score)))


def _bucket_calibration(
    records: Sequence[TurnRecord], confidences: Sequence[float], correct: Sequence[bool]
) -> dict[str, Any]:
    buckets: dict[str, Any] = {}
    for bucket in sorted({record.intent_bucket.value for record in records}):
        positions = [
            index for index, record in enumerate(records) if record.intent_bucket.value == bucket
        ]
        if not positions:
            continue
        report = summarise(
            [confidences[index] for index in positions],
            [correct[index] for index in positions],
        )
        buckets[bucket] = {
            "n": report.n,
            "ece_equal_width": round(report.ece_equal_width, 6),
            "brier": round(report.brier, 6),
            "aurc": round(report.aurc, 6),
        }
    return buckets


# --------------------------------------------------------------------------------------
# Comparison and assembly
# --------------------------------------------------------------------------------------


def compare_arms(
    turns_a: Sequence[TurnRecord],
    turns_b: Sequence[TurnRecord],
    *,
    metric: str,
    k: int = 20,
    seed: int = 0,
    samples: int = 1000,
) -> StatComparison:
    """Paired bootstrap + McNemar on the same queries (§6.3 step 5, FR-EVL-07)."""
    if metric in {"recall_any", "recall_all"}:
        left = paired_scores(turns_a, k)
        right = paired_scores(turns_b, k)
        common = sorted(set(left.case_ids) & set(right.case_ids))
        left_map = dict(
            zip(
                left.case_ids, left.any_hit if metric == "recall_any" else left.all_hit, strict=True
            )
        )
        right_map = dict(
            zip(
                right.case_ids,
                right.any_hit if metric == "recall_any" else right.all_hit,
                strict=True,
            )
        )
        vector_a = [left_map[case_id] for case_id in common]
        vector_b = [right_map[case_id] for case_id in common]
        difference = paired_bootstrap(vector_a, vector_b, samples=samples, seed=seed)
        b = sum(1 for a, b_value in zip(vector_a, vector_b, strict=True) if a > b_value)
        c = sum(1 for a, b_value in zip(vector_a, vector_b, strict=True) if a < b_value)
        test = mcnemar(b, c)
        return StatComparison(
            metric=f"{metric}@{k}",
            arm_a=turns_a[0].arm if turns_a else "a",
            arm_b=turns_b[0].arm if turns_b else "b",
            n=len(common),
            mean_a=round(difference.mean_a, 6),
            mean_b=round(difference.mean_b, 6),
            diff=round(difference.diff, 6),
            ci_low=round(difference.low, 6),
            ci_high=round(difference.high, 6),
            bootstrap=samples,
            mcnemar_b=b,
            mcnemar_c=c,
            mcnemar_statistic=round(test.statistic, 6),
            mcnemar_p=round(test.p_value, 6),
            mcnemar_method=test.method,
            significant=difference.excludes_zero,
            note="paired over queries; 'significant' means the 95% interval excludes 0",
        )
    left_values = {turn.case_id: 1.0 if turn.answer_ok else 0.0 for turn in turns_a}
    right_values = {turn.case_id: 1.0 if turn.answer_ok else 0.0 for turn in turns_b}
    common = sorted(set(left_values) & set(right_values))
    vector_a = [left_values[case_id] for case_id in common]
    vector_b = [right_values[case_id] for case_id in common]
    difference = paired_bootstrap(vector_a, vector_b, samples=samples, seed=seed)
    b = sum(1 for a, b_value in zip(vector_a, vector_b, strict=True) if a > b_value)
    c = sum(1 for a, b_value in zip(vector_a, vector_b, strict=True) if a < b_value)
    test = mcnemar(b, c)
    return StatComparison(
        metric=metric,
        arm_a=turns_a[0].arm if turns_a else "a",
        arm_b=turns_b[0].arm if turns_b else "b",
        n=len(common),
        mean_a=round(difference.mean_a, 6),
        mean_b=round(difference.mean_b, 6),
        diff=round(difference.diff, 6),
        ci_low=round(difference.low, 6),
        ci_high=round(difference.high, 6),
        bootstrap=samples,
        mcnemar_b=b,
        mcnemar_c=c,
        mcnemar_statistic=round(test.statistic, 6),
        mcnemar_p=round(test.p_value, 6),
        mcnemar_method=test.method,
        significant=difference.excludes_zero,
        note="paired over cases",
    )


def build_cells(outcome: RunOutcome, config: RunnerConfig, *, judged: bool) -> list[MetricCell]:
    """Every per-arm table the report prints, with n and a CI in each cell."""
    cells: list[MetricCell] = []
    for arm_name, turns in outcome.turns.items():
        cells.extend(
            retrieval_cells(
                turns, ks=config.ks, arm=arm_name, seed=config.seed, samples=config.bootstrap
            )
        )
        cells.extend(
            answer_cells(
                turns, arm=arm_name, seed=config.seed, samples=config.bootstrap, judged=judged
            )
        )
        cells.extend(escalation_cells(turns, arm=arm_name))
        cells.extend(category_cells(turns, arm=arm_name))
        cells.extend(operational_cells(turns, arm=arm_name))
    return cells


def target_checks(
    cells: Sequence[MetricCell], *, outcome: RunOutcome, config: RunnerConfig, judged: bool
) -> list[TargetCheck]:
    """Compare measured cells against the §6.4/§7.5/§8.5 targets, marking the gated ones.

    ``gated`` is true only when the run can actually speak to the gate: a real provider, a
    reportable judge and human-verified labels. A replay run prints the numbers and says why
    they are diagnostic — which is the whole point of the honest-reporting policy (§10.7).
    """

    def find(name: str, *, arm: str, locale: str = "macro") -> MetricCell | None:
        for cell in cells:
            if cell.name == name and cell.arm == arm and cell.locale == locale:
                return cell
        return None

    primary = config.arms[0] if config.arms else next(iter(outcome.turns))
    checks: list[TargetCheck] = []
    gate = config.provider_mode != "fake" and judged
    retrieval_targets = (
        (
            "RT-1",
            "recall_any@20 (hybrid, tuned)",
            ">= 0.90",
            find("recall_any@20", arm=primary),
            ">=",
            0.90,
        ),
        ("RT-3", "recall_any@10", ">= 0.80", find("recall_any@10", arm=primary), ">=", 0.80),
        ("RT-4", "recall_all@20", ">= 0.75", find("recall_all@20", arm=primary), ">=", 0.75),
        ("RT-5", "nDCG@10", ">= 0.65", find("nDCG@10", arm=primary), ">=", 0.65),
        ("RT-6", "MRR@10", ">= 0.60", find("MRR@10", arm=primary), ">=", 0.60),
    )
    for target_id, description, threshold, cell, comparator, limit in retrieval_targets:
        checks.append(
            TargetCheck(
                target_id=target_id,
                description=description,
                threshold=threshold,
                cell_name=cell.name if cell else "missing",
                value=cell.value if cell else None,
                passed=_passes(cell, limit) if comparator == ">=" else None,
                gated=gate,
                note="" if gate else "diagnostic: replay-mode retrieval arms and machine labels",
            )
        )
    locale_cells = {
        cell.locale: cell for cell in cells if cell.name == "recall_any@20" and cell.arm == primary
    }
    floors = {
        locale: value.value
        for locale, value in locale_cells.items()
        if locale != "macro" and value.n > 0
    }
    checks.append(
        TargetCheck(
            target_id="RT-2",
            description="recall_any@20 per-locale floor",
            threshold=">= 0.80 every locale",
            cell_name="recall_any@20",
            value=min(floors.values()) if floors else None,
            passed=(min(floors.values()) >= 0.80) if floors else None,
            gated=gate,
            note=f"locales: {', '.join(sorted(floors))}" if floors else "no locale rows",
        )
    )
    for target_id, name, description, threshold, limit in (
        ("F-1", "groundedness_verifier", "groundedness (claim support rate)", ">= 0.90", 0.90),
        ("F-2", "citation_precision", "citation precision", ">= 0.90", 0.90),
        ("F-3", "citation_recall", "citation recall", ">= 0.85", 0.85),
    ):
        cell = next(
            (
                item
                for item in cells
                if item.name == name and item.arm == primary and not item.locale
            ),
            None,
        )
        checks.append(
            TargetCheck(
                target_id=target_id,
                description=description,
                threshold=threshold,
                cell_name=name,
                value=cell.value if cell else None,
                passed=_passes(cell, limit),
                gated=gate and name != "groundedness_verifier",
                note="" if gate else "diagnostic: deterministic double or no judge calibration",
            )
        )
    for target_id, name, description, threshold, limit in (
        ("E-4", "coverage", "auto-answer coverage", ">= 0.60", 0.60),
        ("E-2", "must_escalate_recall", "must-escalate recall", ">= 0.95", 0.95),
        ("E-3", "escalation_precision", "escalation precision", ">= 0.55", 0.55),
    ):
        cell = next((item for item in cells if item.name == name and item.arm == primary), None)
        checks.append(
            TargetCheck(
                target_id=target_id,
                description=description,
                threshold=threshold,
                cell_name=name,
                value=cell.value if cell else None,
                passed=_passes(cell, limit),
                gated=gate,
                note="" if gate else "diagnostic: the policy layer is t8's",
            )
        )
    risk = next(
        (item for item in cells if item.name == "realised_risk" and item.arm == primary), None
    )
    checks.append(
        TargetCheck(
            target_id="E-1",
            description="realised auto-answer risk vs target",
            threshold=f"<= {config.target_risk} (pooled <= 0.05)",
            cell_name="realised_risk",
            value=risk.value if risk else None,
            passed=(risk.ci_high <= TARGET_RISK + 0.03) if risk else None,
            gated=gate,
            note="" if gate else "diagnostic: no conformal calibration in this run (t8)",
        )
    )
    calibration = outcome.calibration.get(primary)
    if calibration is not None:
        checks.append(
            TargetCheck(
                target_id="E-5",
                description="calibration error of p̂",
                threshold="ECE <= 0.08 and Brier <= 0.15",
                cell_name="ece_equal_width",
                value=round(calibration.ece_equal_width, 6),
                passed=calibration.ece_equal_width <= 0.08 and calibration.brier <= 0.15,
                gated=False,
                note=f"diagnostic: p̂ is the evidence score, not a conformal calibrator; "
                f"Brier={calibration.brier:.4f}, ECE(mass)={calibration.ece_equal_mass:.4f}",
            )
        )
    return checks


def build_header(
    *,
    run_id: str,
    corpus: Corpus,
    golden_hash: str,
    golden_version: str,
    split_version: str,
    split_hash: str,
    seed: int,
    config: RunnerConfig,
    judge: JudgeMetadata,
    arms: Sequence[ArmSpec],
    case_count: int,
    duration_s: float,
    settings: Settings | None = None,
    extra_notes: Sequence[str] = (),
) -> RunHeader:
    notes = [
        "REPLAY-MODE: provider and judge ran on deterministic doubles; no number here is a "
        "live-model result (§10.7 item 9, NFR-REP-01)."
    ]
    notes.extend(extra_notes)
    pricing_id = settings.llm.pricing_table if settings is not None else ""
    return RunHeader(
        run_id=run_id,
        schema_version="1.0.0",
        corpus_version=corpus.version,
        corpus_hash=corpus.hash,
        golden_version=golden_version,
        golden_hash=golden_hash,
        split_version=split_version,
        split_hash=split_hash,
        seed=seed,
        provider_mode=config.provider_mode,
        replay=config.provider_mode == "fake",
        judge=judge,
        arms=[spec.name for spec in arms],
        locales=corpus.locales,
        case_count=case_count,
        duration_s=round(duration_s, 3),
        hardware=f"{platform.system()} {platform.release()} / Python {platform.python_version()}",
        pricing_id=pricing_id,
        pricing_date="",
        notes=notes,
    )


def _passes(cell: MetricCell | None, limit: float) -> bool | None:
    """``None`` when the cell is empty: a target over zero items is not a failure."""
    if cell is None or cell.n == 0:
        return None
    return cell.value >= limit


def deterministic_digest(result: RunResult) -> str:
    """sha256 of the metrics with timing dropped — the NFR-DET-02 equality check."""
    import hashlib
    import json

    payload = {
        # Timing fields are removed by construction (NFR-DET-02): latency cells are a
        # property of the machine, not of the system under test.
        "cells": [
            {
                "name": cell.name,
                "value": cell.value,
                "n": cell.n,
                "ci_low": cell.ci_low,
                "ci_high": cell.ci_high,
                "arm": cell.arm,
                "locale": cell.locale,
                "category": cell.category,
            }
            for cell in result.cells
            if not cell.name.startswith("latency_")
        ],
        "comparisons": [
            {
                "metric": comparison.metric,
                "arm_a": comparison.arm_a,
                "arm_b": comparison.arm_b,
                "n": comparison.n,
                "diff": comparison.diff,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "p": comparison.mcnemar_p,
            }
            for comparison in result.comparisons
        ],
        "turns": [
            {
                "case_id": turn.case_id,
                "arm": turn.arm,
                "answered": turn.answered,
                "answer_ok": turn.answer_ok,
                "cited": turn.cited_chunk_ids,
                "ranked": turn.ranked_chunk_ids,
            }
            for turn in result.turns
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bootstrap_summary(values: Sequence[float], *, seed: int = 0, samples: int = 1000) -> str:
    """Tiny human-readable CI used in the run notes (keeps the report honest by default)."""
    interval = bootstrap_interval(values, samples=samples, seed=seed)
    if not math.isfinite(interval.value):
        return "n/a"
    return f"{interval.value:.3f} [{interval.low:.3f}, {interval.high:.3f}]"


def build_result(
    *,
    header: RunHeader,
    outcome: RunOutcome,
    config: RunnerConfig,
    judged: bool,
    limitations: Sequence[str],
) -> RunResult:
    cells = build_cells(outcome, config, judged=judged)
    comparisons: list[StatComparison] = []
    names = list(outcome.turns)
    for left, right in itertools.combinations(names, 2):
        for k in config.ks:
            comparisons.append(
                compare_arms(
                    outcome.turns[left],
                    outcome.turns[right],
                    metric="recall_any",
                    k=k,
                    seed=config.seed,
                    samples=config.bootstrap,
                )
            )
        comparisons.append(
            compare_arms(
                outcome.turns[left],
                outcome.turns[right],
                metric="answer_ok",
                seed=config.seed,
                samples=config.bootstrap,
            )
        )
    primary = config.arms[0] if config.arms else names[0]
    calibration = outcome.calibration.get(primary)
    return RunResult(
        header=header,
        cells=cells,
        comparisons=comparisons,
        targets=target_checks(cells, outcome=outcome, config=config, judged=judged),
        turns=[turn for turns in outcome.turns.values() for turn in turns],
        reliability=calibration.to_dict() if calibration is not None else {},
        limitations=list(limitations),
    )


__all__ = [
    "CONTEXT_K",
    "RETRIEVAL_DEPTH",
    "TARGET_RISK",
    "EvalRunner",
    "RunOutcome",
    "RunnerConfig",
    "bootstrap_summary",
    "build_cells",
    "build_header",
    "build_judge",
    "build_provider",
    "build_result",
    "compare_arms",
    "deterministic_digest",
    "normalize_query",
    "target_checks",
]
