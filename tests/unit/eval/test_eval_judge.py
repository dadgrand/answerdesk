"""Judge tests: rubric parsing, position swap, metadata, agreement (AT-EVL-05).

# AT-EVL-05
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from answerdesk.contracts import JudgeScore, Message
from answerdesk.eval.judge import (
    RUBRIC_VERSION,
    AbsolutePairwiseAdapter,
    JudgeError,
    RubricJudge,
    compare_pair,
    human_agreement,
    judge_metadata,
    position_swap_audit,
)
from answerdesk.testing.doubles import FakeJudge, FakeProvider


class BiasedJudge:
    """Prefers whichever candidate it is shown *first* — exactly what position swap removes.

    The adapter grades candidates one after another, so "first slot" is the odd-numbered
    call. A judge like this turns a real leaderboard upside down; after averaging both orders
    its preference must cancel out.
    """

    name = "biased"

    def __init__(self) -> None:
        self.calls = 0

    def grade(
        self,
        *,
        question: str,
        reference_answer: str | None,
        candidate_answer: str,
        context: Sequence[str],
    ) -> JudgeScore:
        self.calls += 1
        first_slot = self.calls % 2 == 1
        return JudgeScore(
            score=5.0 if first_slot else 2.0,
            rubric_version="rubric-v1",
            judge_model="biased",
            rationale="first-slot bias",
        )


def test_fake_judge_is_deterministic() -> None:
    judge = FakeJudge()
    kwargs = {
        "question": "how long does a refund take?",
        "reference_answer": "refunds take fourteen days",
        "candidate_answer": "refunds take fourteen days",
        "context": ["refunds take fourteen days"],
    }
    assert judge.grade(**kwargs).score == judge.grade(**kwargs).score
    assert judge.grade(**kwargs).score == pytest.approx(5.0)


def test_position_swap_averages_a_biased_judge() -> None:
    comparison = compare_pair(
        BiasedJudge(),
        question="q",
        answer_a="candidate a",
        answer_b="candidate b",
        context=["ctx"],
    )
    assert comparison.ab.winner == "a"
    assert comparison.ba.winner == "b"  # after the swap the bias points the other way
    assert not comparison.order_consistent
    assert comparison.score_a == pytest.approx(comparison.score_b)  # bias averaged out
    assert len(comparison.rows) == 2
    assert {row.order for row in comparison.rows} == {"ab", "ba"}


def test_position_swap_audit_fails_a_biased_judge() -> None:
    items = [
        {"question": "q1", "answer_a": "x", "answer_b": "y", "context": ["c"]},
        {"question": "q2", "answer_a": "z", "answer_b": "w", "context": ["c"]},
    ]
    audit = position_swap_audit(BiasedJudge(), items)
    assert audit.n == 2
    assert audit.invariance == 0.0
    assert not audit.passed


def test_position_swap_audit_passes_an_order_invariant_judge() -> None:
    judge = FakeJudge()
    items = [
        {
            "question": "q",
            "answer_a": "refunds take fourteen days",
            "answer_b": "blocked cards stay blocked",
            "context": ["refunds take fourteen days"],
        }
    ]
    audit = position_swap_audit(AbsolutePairwiseAdapter(judge), items)
    assert audit.invariance == 1.0
    assert audit.passed


def test_rubric_judge_parses_json_and_records_versions() -> None:
    provider = FakeProvider(responses={"grade": json.dumps({"score": 4, "rationale": "grounded"})})
    judge = RubricJudge(provider, judge_model="judge-x")
    score = judge.grade(
        question="grade this",
        reference_answer=None,
        candidate_answer="answer",
        context=["context"],
    )
    assert score.score == 4.0
    assert score.rubric_version == RUBRIC_VERSION
    assert score.judge_model == "judge-x"
    assert score.rationale == "grounded"
    assert judge.calls == 1


def test_rubric_judge_wraps_scores_into_range() -> None:
    provider = FakeProvider(responses={"grade": json.dumps({"score": 9})})
    judge = RubricJudge(provider, judge_model="judge-x")
    score = judge.grade(
        question="grade", reference_answer=None, candidate_answer="candidate", context=[]
    )
    assert score.score == 5.0


def test_rubric_judge_fails_loudly_on_garbage() -> None:
    provider = FakeProvider(responses={"grade": "I think it is fine."})
    judge = RubricJudge(provider, judge_model="judge-x")
    with pytest.raises(JudgeError):
        judge.grade(question="grade", reference_answer=None, candidate_answer="x", context=[])


def test_judge_metadata_is_not_reportable_without_a_generator_model() -> None:
    metadata = judge_metadata(
        judge_model="fake-judge", prompt_version="none", rubric_version="fake-rubric-1"
    )
    assert metadata.reportable is False
    assert "generator model not recorded" in metadata.note
    assert metadata.position_swap is True


def test_judge_metadata_reportability_follows_the_family_rule() -> None:
    """§10.5: reportable means cross-family; κ is reported separately (F-10)."""
    cross = judge_metadata(
        judge_model="deepseek-flash",
        prompt_version="p1",
        rubric_version="r1",
        generator_model="qwen3.6-35b-a3b-noreason",
        agreement={"ru": (0.72, 150), "tr": (0.65, 120)},
    )
    assert cross.reportable is True
    assert cross.human_calibrated is True
    assert cross.agreement_n == {"ru": 150, "tr": 120}
    same_family = judge_metadata(
        judge_model="qwen3.6-35b-a3b-noreason",
        prompt_version="p1",
        rubric_version="r1",
        generator_model="qwen3.6-35b-a3b-noreason",
        agreement={"ru": (0.72, 150)},
    )
    assert same_family.reportable is False
    assert "same-family" in same_family.note
    uncalibrated = judge_metadata(
        judge_model="deepseek-flash",
        prompt_version="p1",
        rubric_version="r1",
        generator_model="qwen3.6-35b-a3b-noreason",
        agreement={"ru": (0.4, 150)},
    )
    assert uncalibrated.reportable is True  # cross-family, but κ is below the reporting floor
    assert uncalibrated.human_calibrated is False
    assert "κ" in uncalibrated.note


def test_human_agreement_is_per_locale_kappa() -> None:
    judged = {"a": True, "b": True, "c": False, "d": True}
    human = [
        {"case_id": "a", "locale": "ru", "human_pass": True},
        {"case_id": "b", "locale": "ru", "human_pass": True},
        {"case_id": "c", "locale": "ru", "human_pass": False},
        {"case_id": "d", "locale": "ru", "human_pass": False},
        {"case_id": "missing", "locale": "ru", "human_pass": True},
    ]
    agreement = human_agreement(human, judged)
    assert set(agreement) == {"ru"}  # a single language gets no pooled row
    kappa, n = agreement["ru"]
    assert n == 4
    assert -1.0 <= kappa <= 1.0


def test_human_agreement_pools_when_several_locales() -> None:
    judged = {"a": True, "b": False, "c": True, "d": False}
    human = [
        {"case_id": "a", "locale": "ru", "human_pass": True},
        {"case_id": "b", "locale": "ru", "human_pass": False},
        {"case_id": "c", "locale": "tr", "human_pass": True},
        {"case_id": "d", "locale": "tr", "human_pass": True},
    ]
    agreement = human_agreement(human, judged)
    assert set(agreement) == {"all", "ru", "tr"}
    assert agreement["all"][1] == 4


def test_fake_provider_is_a_chat_provider_with_messages() -> None:
    provider = FakeProvider()
    completion = provider.complete([Message(role="user", content="hi")])
    assert "hi" in completion.text
