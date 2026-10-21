"""The answer pipeline: grounded answers, explicit refusals, and their contract shape.

# AT-GEN-01  # AT-GEN-02  # AT-GEN-08  # AT-GEN-10
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from pydantic import ValidationError
from support import (
    CARD_FEE_CLAIM,
    CARD_FEE_QUOTE,
    FEE_CHARGE_QUOTE,
    ScriptedProvider,
    draft_json,
)

from answerdesk.contracts import Action, Citation, Claim, ReasonCode, RetrievedPassage, Support
from answerdesk.generate.citations import markers_in
from answerdesk.generate.pipeline import AnswerPipeline, GenerationResult, GenerationSignal

QUESTION = "Сколько стоит обслуживание карты?"


# --------------------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------------------


def test_answer_is_a_structured_cited_object(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
    answer_draft: str,
) -> None:
    """# AT-GEN-01 — ordered claims, each with a citation; markers derived from structure."""
    pipeline = pipeline_factory(ScriptedProvider([answer_draft]))
    result = pipeline.answer(QUESTION, passages)

    assert result.signal.action is Action.ANSWER
    assert result.signal.reason_code is ReasonCode.ANSWERED
    assert result.language == "ru"
    assert result.text.startswith(CARD_FEE_CLAIM)
    assert markers_in(result.text) == [1]
    assert [claim.claim_id for claim in result.claims] == ["c1"]
    assert result.claims[0].support is Support.SUPPORTED
    assert result.claims[0].citation_ids == [1]
    assert result.citations[0].quote == CARD_FEE_QUOTE
    assert result.prompt_version == "grounded-answer-1.0.0"
    assert result.verifier_version == "lexical-verifier-1.0.0"
    assert result.repaired is False
    assert result.dropped_claims == 0


def test_pipeline_records_usage_and_evidence_provenance(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
    answer_draft: str,
) -> None:
    """# AT-GEN-01 — NFR-COST-01: the answer carries its own cost/provenance record."""
    pipeline = pipeline_factory(ScriptedProvider([answer_draft]))
    result = pipeline.answer(QUESTION, passages)

    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.usage.model == "fake-model"
    assert result.usage.prompt_version == "grounded-answer-1.0.0"
    assert result.usage.usd == pytest.approx(0.0)  # the CI double is priced at zero
    assert result.model == "fake-model"
    assert result.packed.labels == ["p1", "p2", "p3"]
    assert result.signal.evidence_score == pytest.approx(0.94)
    assert result.packed.token_count <= result.packed.budget


def test_claims_are_numbered_in_order_and_pruned_to_used_citations(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    draft = draft_json(
        [
            (CARD_FEE_CLAIM, "p1", CARD_FEE_QUOTE),
            ("Плата списывается первого числа каждого месяца.", "p1", FEE_CHARGE_QUOTE),
        ]
    )
    pipeline = pipeline_factory(ScriptedProvider([draft]))
    result = pipeline.answer(QUESTION, passages)

    assert [claim.citation_ids for claim in result.claims] == [[1], [2]]
    assert [citation.citation_id for citation in result.citations] == [1, 2]
    assert markers_in(result.text) == [1, 2]
    assert result.claims[0].text == CARD_FEE_CLAIM


def test_result_round_trips_through_json(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
    answer_draft: str,
) -> None:
    """# AT-GEN-01 — the displayed shape is a contract object, not an ad-hoc dict."""
    pipeline = pipeline_factory(ScriptedProvider([answer_draft]))
    result = pipeline.answer(QUESTION, passages)
    payload = json.loads(result.to_json())
    restored = GenerationResult.model_validate(payload)
    assert restored.to_json() == result.to_json()
    assert restored.to_answer_parts()[0] == result.text


def test_answer_result_may_not_carry_an_uncited_claim() -> None:
    """# AT-GEN-01 — INV-03 enforced by the contract, not by convention."""
    with pytest.raises(ValidationError, match="without a citation"):
        GenerationResult(
            question=QUESTION,
            language="ru",
            text="uncited",
            claims=[Claim(claim_id="c1", text="uncited", citation_ids=[])],
            signal=GenerationSignal(action=Action.ANSWER, reason_code=ReasonCode.ANSWERED),
        )


def test_answer_result_may_not_display_an_unsupported_claim() -> None:
    with pytest.raises(ValidationError, match="unsupported"):
        GenerationResult(
            question=QUESTION,
            language="ru",
            text="x [1]",
            claims=[
                Claim(
                    claim_id="c1",
                    text="x",
                    citation_ids=[1],
                    support=Support.UNSUPPORTED,
                )
            ],
            citations=[
                Citation(
                    citation_id=1,
                    chunk_id="c",
                    doc_id="d",
                    quote="x",
                    start_char=0,
                    end_char=1,
                    doc_hash="0" * 64,
                )
            ],
            signal=GenerationSignal(action=Action.ANSWER, reason_code=ReasonCode.ANSWERED),
        )


def test_escalation_result_may_not_carry_prose() -> None:
    """# AT-GEN-10 — CR-8: a refusal is a signal, never a "best guess"."""
    with pytest.raises(ValidationError, match="no prose"):
        GenerationResult(
            question=QUESTION,
            language="ru",
            text="best guess",
            claims=[Claim(claim_id="c1", text="best guess", citation_ids=[1])],
            signal=GenerationSignal(action=Action.ESCALATE, reason_code=ReasonCode.NO_EVIDENCE),
        )


# --------------------------------------------------------------------------------------
# No evidence, no answer
# --------------------------------------------------------------------------------------


def test_empty_evidence_escalates_without_calling_the_model(
    pipeline_factory: Callable[..., AnswerPipeline],
) -> None:
    """# AT-GEN-10 — FR-GEN-10: a structured signal instead of prose, at zero model cost."""
    provider = ScriptedProvider([])
    pipeline = pipeline_factory(provider)
    result = pipeline.answer(QUESTION, [])

    assert result.signal.action is Action.ESCALATE
    assert result.signal.reason_code is ReasonCode.NO_EVIDENCE
    assert result.text == ""
    assert result.claims == []
    assert result.citations == []
    assert provider.call_count == 0
    assert result.packed.is_empty
    assert result.usage.input_tokens == 0


def test_model_declared_insufficiency_escalates(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    """# AT-GEN-10 — the refusal channel of the schema is honoured (CR-8)."""
    refusal = draft_json([], insufficient=True, why="В документах нет цены обслуживания.")
    provider = ScriptedProvider([refusal])
    pipeline = pipeline_factory(provider)
    result = pipeline.answer(QUESTION, passages)

    assert result.signal.action is Action.ESCALATE
    assert result.signal.reason_code is ReasonCode.NO_EVIDENCE
    assert "нет цены" in result.signal.detail
    assert result.text == ""
    assert result.usage.input_tokens > 0  # the call happened and was accounted


def test_twenty_no_evidence_queries_produce_twenty_structured_signals(
    pipeline_factory: Callable[..., AnswerPipeline],
) -> None:
    """# AT-GEN-10 — the released fixture set: 0 prose answers out of 20."""
    provider = ScriptedProvider([])
    pipeline = pipeline_factory(provider)
    questions = [f"Вопрос без доказательств номер {index}?" for index in range(20)]
    results = [pipeline.answer(question, []) for question in questions]

    assert len(results) == 20
    assert all(result.text == "" for result in results)
    assert all(result.claims == [] for result in results)
    assert all(result.signal.action is Action.ESCALATE for result in results)
    assert {result.signal.reason_code for result in results} == {ReasonCode.NO_EVIDENCE}
    assert provider.call_count == 0


def test_malformed_output_is_reasked_once_then_escalates(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    """# AT-GEN-10 — NFR-OPS-01 row 'malformed model JSON' has a defined behaviour."""
    provider = ScriptedProvider(["this is not json", "still not json"])
    pipeline = pipeline_factory(provider)
    result = pipeline.answer(QUESTION, passages)

    assert provider.call_count == 2  # one strict re-ask, never a third attempt
    assert result.signal.action is Action.ESCALATE
    assert result.signal.reason_code is ReasonCode.PROVIDER_ERROR
    assert "strict re-ask" in result.signal.detail
    assert result.text == ""


def test_malformed_then_valid_output_recovers(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
    answer_draft: str,
) -> None:
    provider = ScriptedProvider(["```json\n{broken\n```", answer_draft])
    pipeline = pipeline_factory(provider)
    result = pipeline.answer(QUESTION, passages)

    assert result.signal.is_answer is True
    assert provider.call_count == 2
    assert markers_in(result.text) == [1]


def test_fenced_json_is_accepted(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
    answer_draft: str,
) -> None:
    """Models wrap JSON in fences; the parser tolerates it, the validator does not relax."""
    provider = ScriptedProvider([f"Here is the answer:\n```json\n{answer_draft}\n```\nDone."])
    pipeline = pipeline_factory(provider)
    result = pipeline.answer(QUESTION, passages)
    assert result.signal.is_answer is True


def test_completion_schema_is_sent_to_the_provider(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
    answer_draft: str,
) -> None:
    """# AT-GEN-01 — the model is asked for structure (json_schema), never for prose markers."""
    provider = ScriptedProvider([answer_draft])
    pipeline = pipeline_factory(provider)
    pipeline.answer(QUESTION, passages)

    schema = provider.schemas[0]
    assert schema is not None
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["insufficient_evidence", "why", "claims"]
    system_prompt = provider.calls[0][0].content
    assert "JSON object" in system_prompt
    assert "<passage" in provider.calls[0][1].content
