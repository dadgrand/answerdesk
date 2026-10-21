"""Runner tests: offline end-to-end, determinism, judgment, loud failures.

# AT-EVL-06  # AT-EVL-07  # AT-EVL-08  # AT-DET-02
"""

from __future__ import annotations

import pytest

from answerdesk.config import Settings
from answerdesk.contracts import EvalCase, Split
from answerdesk.eval.arms import build_arms
from answerdesk.eval.corpus import Corpus
from answerdesk.eval.errors import EvalError
from answerdesk.eval.fixture_provider import EvidenceAnchoredProvider
from answerdesk.eval.judge import judge_metadata
from answerdesk.eval.models import RunHeader
from answerdesk.eval.runner import (
    EvalRunner,
    RunnerConfig,
    build_header,
    build_judge,
    build_provider,
    build_result,
    compare_arms,
    deterministic_digest,
)
from answerdesk.generate.pipeline import AnswerPipeline
from answerdesk.llm.presets import provider_config
from answerdesk.testing.doubles import FakeJudge


def _runner(
    corpus: Corpus,
    cases: list[EvalCase],
    *,
    arms: tuple[str, ...] = ("bm25", "hybrid-rrf"),
    judge_mode: str = "fake",
) -> EvalRunner:
    config = RunnerConfig(arms=arms, ks=(10,), seed=5, bootstrap=120, judge_mode=judge_mode)  # type: ignore[arg-type]
    provider = EvidenceAnchoredProvider(provider_config("fake"))
    pipeline = AnswerPipeline.from_settings(provider, Settings())
    return EvalRunner(
        corpus=corpus,
        arms=build_arms(corpus, arms),
        pipeline=pipeline,
        judge=FakeJudge(),
        config=config,
    )


def test_runner_produces_one_turn_per_case_and_arm(
    tiny_corpus: Corpus, tiny_cases: list[EvalCase]
) -> None:
    selected = [case for case in tiny_cases if case.split is Split.TEST]
    outcome = _runner(tiny_corpus, selected).run(selected)
    assert set(outcome.turns) == {"bm25", "hybrid-rrf"}
    for turns in outcome.turns.values():
        assert len(turns) == len(selected)
        assert {turn.case_id for turn in turns} == {case.case_id for case in selected}
    first = outcome.turns["bm25"][0]
    assert first.answered or first.escalated
    assert "total" in first.latency_ms
    assert first.confidence >= 0.0


def test_runner_is_deterministic_apart_from_timing(
    tiny_corpus: Corpus, tiny_cases: list[EvalCase]
) -> None:
    selected = [case for case in tiny_cases if case.split is Split.TEST]
    first = _runner(tiny_corpus, selected).run(selected)
    second = _runner(tiny_corpus, selected).run(selected)
    judge = judge_metadata(
        judge_model="fake-judge", prompt_version="none", rubric_version="fake-rubric-1"
    )
    header = build_header(
        run_id="determinism",
        corpus=tiny_corpus,
        golden_hash="0" * 64,
        golden_version="v1",
        split_version="v1",
        split_hash="1" * 64,
        seed=5,
        config=RunnerConfig(arms=("bm25", "hybrid-rrf"), seed=5, bootstrap=120),
        judge=judge,
        arms=[],
        case_count=len(selected),
        duration_s=first.duration_s,
    )
    left = build_result(
        header=header,
        outcome=first,
        config=RunnerConfig(arms=("bm25", "hybrid-rrf"), seed=5, bootstrap=120),
        judged=False,
        limitations=[],
    )
    right = build_result(
        header=header,
        outcome=second,
        config=RunnerConfig(arms=("bm25", "hybrid-rrf"), seed=5, bootstrap=120),
        judged=False,
        limitations=[],
    )
    assert deterministic_digest(left) == deterministic_digest(right)


def test_runner_escalates_unanswerable_cases(
    tiny_corpus: Corpus, tiny_cases: list[EvalCase]
) -> None:
    """The absent-identifier family must not be answered: that is the whole point of it."""
    unanswerable = [case for case in tiny_cases if case.split is Split.TEST and not case.answerable]
    assert unanswerable
    outcome = _runner(tiny_corpus, unanswerable).run(unanswerable)
    turns = outcome.turns["bm25"]
    assert all(turn.must_escalate for turn in turns)
    assert any(turn.escalated for turn in turns)


def test_runner_records_retrieval_and_answer_cells(
    tiny_corpus: Corpus, tiny_cases: list[EvalCase]
) -> None:
    selected = [case for case in tiny_cases if case.split is Split.TEST]
    runner = _runner(tiny_corpus, selected)
    outcome = runner.run(selected)
    config = runner.config
    header = build_header(
        run_id="cells",
        corpus=tiny_corpus,
        golden_hash="0" * 64,
        golden_version="v1",
        split_version="v1",
        split_hash="1" * 64,
        seed=5,
        config=config,
        judge=judge_metadata(judge_model="fake-judge", prompt_version="none", rubric_version="r"),
        arms=[],
        case_count=len(selected),
        duration_s=0.0,
    )
    result = build_result(
        header=header, outcome=outcome, config=config, judged=False, limitations=[]
    )
    names = {cell.name for cell in result.cells}
    assert {"recall_any@10", "citation_recall", "realised_risk", "coverage"} <= names
    assert result.comparisons, "a multi-arm run must produce paired comparisons"
    assert result.reliability["n"] == len(selected)


def test_compare_arms_is_paired_and_reports_mcnemar() -> None:
    from answerdesk.contracts import Action, ReasonCode
    from answerdesk.eval.models import TurnRecord

    def turn(case_id: str, arm: str, ranked: list[str], ok: bool) -> TurnRecord:
        return TurnRecord(
            case_id=case_id,
            locale="ru",
            category="direct",
            arm=arm,
            answerable=True,
            expected_language="ru",
            language="ru",
            action=Action.ANSWER,
            reason_code=ReasonCode.ANSWERED,
            answered=True,
            escalated=False,
            ranked_chunk_ids=ranked,
            gold_chunk_ids=["c1"],
            answer_ok=ok,
        )

    left = [turn(f"c{i}", "a", ["c1", "c2"], True) for i in range(6)]
    right = [turn(f"c{i}", "b", ["c2", "c1"], i < 2) for i in range(6)]
    comparison = compare_arms(left, right, metric="recall_any", k=1, samples=200, seed=1)
    assert comparison.n == 6
    assert comparison.mean_a == 1.0
    assert comparison.mean_b == 0.0
    assert comparison.diff == pytest.approx(1.0)
    assert comparison.ci_low > 0.0
    assert comparison.mcnemar_method == "exact_binomial"
    assert comparison.significant


def test_build_provider_fake_is_a_fake_provider() -> None:
    from answerdesk.testing.doubles import FakeProvider

    provider = build_provider(Settings(), mode="fake")
    assert isinstance(provider, FakeProvider)
    assert provider.name == "fake-evidence"


def test_build_judge_refuses_a_generator_judging_itself() -> None:
    settings = Settings()
    settings.eval.judge_model = "fake-model"  # equal to the provider's model
    with pytest.raises(EvalError) as excinfo:
        build_judge(settings, mode="llm", provider=EvidenceAnchoredProvider())
    assert "own output" in str(excinfo.value)


def test_build_judge_metadata_is_honest_about_reportability() -> None:
    judge, metadata = build_judge(Settings(), mode="fake", provider=EvidenceAnchoredProvider())
    assert judge.name == "fake-judge"
    assert metadata.reportable is False
    assert metadata.rubric_version
    assert metadata.prompt_version


def test_empty_selection_is_an_error(tiny_corpus: Corpus) -> None:
    with pytest.raises(EvalError):
        _runner(tiny_corpus, []).run([])


def test_header_carries_every_reproducibility_field(tiny_corpus: Corpus) -> None:
    header = build_header(
        run_id="h",
        corpus=tiny_corpus,
        golden_hash="a" * 64,
        golden_version="v1",
        split_version="v1",
        split_hash="b" * 64,
        seed=3,
        config=RunnerConfig(),
        judge=judge_metadata(judge_model="m", prompt_version="p", rubric_version="r"),
        arms=[],
        case_count=0,
        duration_s=0.0,
    )
    assert isinstance(header, RunHeader)
    assert header.corpus_hash == tiny_corpus.hash
    assert header.seed == 3
    assert any("REPLAY" in note for note in header.notes)
