"""The repair state machine and the numeric guard, end to end.

# AT-GEN-07  # AT-GEN-12
"""

from __future__ import annotations

from collections.abc import Callable

from support import (
    CARD_FEE_CLAIM,
    CARD_FEE_QUOTE,
    FEE_CHARGE_QUOTE,
    ScriptedProvider,
    draft_json,
)

from answerdesk.contracts import Action, ReasonCode, RetrievedPassage
from answerdesk.generate.citations import markers_in
from answerdesk.generate.pipeline import AnswerPipeline

QUESTION = "Сколько стоит обслуживание карты?"
PARAPHRASED_QUOTE = "Обслуживание карты стоит 120 рублей ежемесячно."
WRONG_NUMBER_CLAIM = "Комиссия за перевод составляет 999 рублей."


def test_one_repair_pass_keeps_supported_claims(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    """# AT-GEN-07 — an unsupported claim triggers exactly one repair, then it is dropped."""
    bad_draft = draft_json(
        [
            (CARD_FEE_CLAIM, "p1", CARD_FEE_QUOTE),
            ("Обслуживание карты стоит 120 рублей ежемесячно.", "p1", PARAPHRASED_QUOTE),
        ]
    )
    repaired_draft = draft_json([(CARD_FEE_CLAIM, "p1", CARD_FEE_QUOTE)])
    provider = ScriptedProvider([bad_draft, repaired_draft])
    pipeline = pipeline_factory(provider)
    result = pipeline.answer(QUESTION, passages)

    assert provider.call_count == 2, "exactly one repair pass (FR-GEN-07)"
    assert result.repaired is True
    assert result.repair_attempts == 1
    assert result.signal.action is Action.ANSWER
    assert [claim.text for claim in result.claims] == [CARD_FEE_CLAIM]
    assert all(claim.support.value == "supported" for claim in result.claims)


def test_repair_that_does_not_help_drops_the_claim(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    """# AT-GEN-12 — a claim quoting a number absent from its evidence is never displayed."""
    bad = draft_json([(WRONG_NUMBER_CLAIM, "p1", FEE_CHARGE_QUOTE)])
    provider = ScriptedProvider([bad, bad])
    pipeline = pipeline_factory(provider)
    result = pipeline.answer(QUESTION, passages)

    assert provider.call_count == 2
    assert result.signal.action is Action.ESCALATE
    assert result.signal.reason_code is ReasonCode.CITATION_FAILED
    assert result.text == ""
    assert result.claims == []
    assert result.numeric_violations_pre_repair == 1
    assert result.dropped_claims == 1


def test_repair_budget_zero_means_no_second_call(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    bad = draft_json([(WRONG_NUMBER_CLAIM, "p1", FEE_CHARGE_QUOTE)])
    provider = ScriptedProvider([bad])
    pipeline = pipeline_factory(provider, repair_passes=0)
    result = pipeline.answer(QUESTION, passages)

    assert provider.call_count == 1
    assert result.repaired is False
    assert result.signal.reason_code is ReasonCode.CITATION_FAILED


def test_repair_refusal_keeps_the_previously_supported_claims(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    """A repair that declares total insufficiency must not erase good claims."""
    bad_draft = draft_json(
        [
            (CARD_FEE_CLAIM, "p1", CARD_FEE_QUOTE),
            ("Выдуманное утверждение без опоры.", "p1", "Выдуманное утверждение без опоры."),
        ]
    )
    refusal = draft_json([], insufficient=True, why="не могу подтвердить")
    provider = ScriptedProvider([bad_draft, refusal])
    pipeline = pipeline_factory(provider)
    result = pipeline.answer(QUESTION, passages)

    assert provider.call_count == 2
    assert result.signal.action is Action.ANSWER
    assert [claim.text for claim in result.claims] == [CARD_FEE_CLAIM]
    assert result.dropped_claims == 1


def test_verifier_rejection_without_numeric_or_quote_problem(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    """# AT-GEN-07 — reason code distinguishes a verifier rejection from a citation failure."""
    unsupported = "Данная информация подтверждена независимым аудитом."
    draft = draft_json([(unsupported, "p1", CARD_FEE_QUOTE)])
    provider = ScriptedProvider([draft, draft])
    pipeline = pipeline_factory(provider)
    result = pipeline.answer(QUESTION, passages)

    assert result.signal.reason_code is ReasonCode.VERIFIER_REJECT
    assert result.text == ""
    assert result.numeric_violations_pre_repair == 0


def test_one_bad_claim_does_not_destroy_a_good_answer(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    """The false-rejection budget matters: a good claim must survive a bad neighbour."""
    draft = draft_json(
        [
            (CARD_FEE_CLAIM, "p1", CARD_FEE_QUOTE),
            ("Плата списывается первого числа каждого месяца.", "p1", FEE_CHARGE_QUOTE),
            (WRONG_NUMBER_CLAIM, "p1", FEE_CHARGE_QUOTE),
        ]
    )
    provider = ScriptedProvider([draft, draft])
    pipeline = pipeline_factory(provider)
    result = pipeline.answer(QUESTION, passages)

    assert result.signal.action is Action.ANSWER
    assert len(result.claims) == 2
    assert result.dropped_claims == 1
    assert result.numeric_violations_pre_repair == 1
    assert markers_in(result.text) == [1, 2]
