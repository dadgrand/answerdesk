"""Benchmark-harness tests: family enforcement, strata, confusion, curve, cost sensitivity.

These cover the parts of the t11 harness that are pure functions and therefore testable
without a network: the cross-family judge rule (§10.5), the noisy/free-form stratum, the
TP/FP/FN/TN matrix, the risk–coverage sweep and the cost-ratio translation. The live run
itself is an artefact, not a unit test.

    # AT-EVL-05  # AT-EVL-10  # AT-ESC-07
"""

from __future__ import annotations

from answerdesk.contracts import Action, EvalCategory, ReasonCode, Split
from answerdesk.eval.bench import (
    COST_RATIOS,
    BenchConfig,
    _confusion,
    _curve,
    _subset,
    case_stratum,
    unsafe_kind,
)
from answerdesk.eval.judge import judge_metadata, model_family
from answerdesk.eval.models import TurnRecord


def _turn(
    case_id: str, *, answered: bool, ok: bool, confidence: float, bucket: str = "informational"
) -> TurnRecord:
    return TurnRecord(
        case_id=case_id,
        locale="ru",
        category=EvalCategory.DIRECT,
        arm="primary",
        answerable=True,
        expected_language="ru",
        action=Action.ANSWER if answered else Action.ESCALATE,
        reason_code=ReasonCode.ANSWERED if answered else ReasonCode.NO_EVIDENCE,
        answered=answered,
        escalated=not answered,
        confidence=confidence,
        answer_ok=ok,
        language="ru",
    )


def test_model_family_mapping() -> None:
    assert model_family("qwen3.6-35b-a3b") == "qwen"
    assert model_family("Qwen3-Embedding-0.6B") == "qwen"
    assert model_family("deepseek-flash") == "deepseek"
    assert model_family("gpt-4o-mini") == "openai"
    assert model_family("claude-3-5-sonnet") == "anthropic"
    assert model_family("llama-3.1-8b") == "meta"
    assert model_family("some-new-model") == "some"


def test_cross_family_judge_is_required_for_reportability() -> None:
    cross = judge_metadata(
        judge_model="deepseek-flash",
        prompt_version="judge-rubric-1",
        rubric_version="rubric-v1",
        generator_model="qwen3.6-35b-a3b-noreason",
    )
    assert cross.cross_family is True
    assert cross.reportable is True
    assert cross.generator_family == "qwen" and cross.judge_family == "deepseek"
    same = judge_metadata(
        judge_model="qwen3.6-35b-a3b-noreason",
        prompt_version="judge-rubric-1",
        rubric_version="rubric-v1",
        generator_model="qwen3.6-35b-a3b-noreason",
    )
    assert same.cross_family is False
    assert same.reportable is False
    assert "same-family" in same.note
    unknown = judge_metadata(
        judge_model="deepseek-flash",
        prompt_version="p",
        rubric_version="r",
    )
    assert unknown.reportable is False
    assert "generator model not recorded" in unknown.note


def test_fixture_case_tags_stratum_and_unsafe_kind() -> None:
    from answerdesk.contracts import EvalCase

    noisy = EvalCase(
        case_id="n1",
        locale="ru",
        category=EvalCategory.TYPO_CODE_SWITCH,
        question="q",
        answerable=True,
        split=Split.TEST,
        gold_chunk_ids=["c1"],
        reference_answer="a",
        metadata={"corpus_category": "noisy_freeform"},
    )
    assert case_stratum(noisy) == "noisy"
    clean = noisy.model_copy(update={"metadata": {"corpus_category": "direct"}})
    assert case_stratum(clean) == "all"
    unanswerable = clean.model_copy(
        update={"answerable": False, "metadata": {"corpus_category": "unanswerable"}}
    )
    assert unsafe_kind(unanswerable) == "unanswerable"
    assert unsafe_kind(clean) == "answerable"


def test_confusion_matrix_math() -> None:
    from answerdesk.contracts import EvalCase

    def case(case_id: str, answerable: bool) -> EvalCase:
        return EvalCase(
            case_id=case_id,
            locale="ru",
            category=EvalCategory.DIRECT,
            question="q",
            answerable=answerable,
            split=Split.TEST,
            gold_chunk_ids=["c1"] if answerable else [],
            reference_answer="a" if answerable else None,
        )

    cases = [case("a", True), case("b", True), case("c", False), case("d", False)]
    turns = [
        _turn("a", answered=True, ok=True, confidence=0.9),  # TN
        _turn("b", answered=False, ok=False, confidence=0.2),  # FP
        _turn("c", answered=False, ok=False, confidence=0.1),  # TP
        _turn("d", answered=True, ok=False, confidence=0.8),  # FN
    ]
    matrix = _confusion(turns, cases)
    assert (matrix["tp"], matrix["fp"], matrix["fn"], matrix["tn"]) == (1, 1, 1, 1)
    assert matrix["precision"] == 0.5
    assert matrix["recall"] == 0.5
    assert matrix["false_escalation_rate"] == 0.5
    assert matrix["coverage"] == 0.5


def test_risk_coverage_curve_is_monotone_in_coverage() -> None:
    turns = [
        _turn(f"c{index}", answered=True, ok=index % 2 == 0, confidence=1.0 - index * 0.1)
        for index in range(10)
    ]
    curve = _curve(turns)
    assert len(curve) == 10
    assert [point["coverage"] for point in curve] == sorted(point["coverage"] for point in curve)
    assert curve[-1]["coverage"] == 1.0
    # the first answered turn was correct (alternating pattern), so risk starts at 0
    assert curve[0]["risk"] == 0.0
    assert 0.0 <= curve[-1]["risk"] <= 1.0


def test_cost_sensitivity_ratios_are_declared() -> None:
    assert COST_RATIOS == (2, 5, 10)
    thresholds = [1.0 - (1.0 / ratio) for ratio in COST_RATIOS]
    assert thresholds == [0.5, 0.8, 0.9]
    assert thresholds == sorted(thresholds)


def test_subset_is_deterministic_and_locale_stratified() -> None:
    from answerdesk.contracts import EvalCase

    cases = [
        EvalCase(
            case_id=f"{locale}-{index}",
            locale=locale,
            category=EvalCategory.DIRECT,
            question="q",
            answerable=True,
            split=Split.TEST,
            gold_chunk_ids=["c1"],
            reference_answer="a",
        )
        for locale in ("ru", "tr")
        for index in range(5)
    ]
    first = _subset(cases, locales=(), limit=6, seed=1)
    second = _subset(cases, locales=(), limit=6, seed=1)
    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert len(first) == 6
    assert {case.locale for case in first} == {"ru", "tr"}
    assert _subset(cases, locales=("tr",), limit=None, seed=1)[0].locale == "tr"


def test_bench_config_records_the_evidence_settings() -> None:
    config = BenchConfig(run_id="bench", answer_limit=400, retrieval_limit=540, target_risk=0.02)
    payload = config.as_dict()
    assert payload["answer_limit"] == 400
    assert payload["target_risk"] == 0.02
    assert payload["replay"] is False
