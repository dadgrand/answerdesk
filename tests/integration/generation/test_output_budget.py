"""Output budget: the cap, the claim limit, the compact schema and the extractive instruction.

The budget is a *contract*, not a request: a model that ignores the instruction still cannot
produce more than ``max_claims`` claims, and both the cap and the claim limit are deterministic
functions a test can assert without a model.

    # AT-LAT-02  # AT-GEN-05
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
from support import CARD_FEE_QUOTE, draft_json

from answerdesk.contracts import Completion, Message, RetrievedPassage, Usage
from answerdesk.generate.budget import (
    EXTRACTIVE_INSTRUCTION,
    AnswerBudget,
    budget_schema,
    cap_claims,
    compact_schema,
)
from answerdesk.generate.pipeline import AnswerPipeline
from answerdesk.generate.schema import load_answer_schema

pytestmark = pytest.mark.integration

CARD_CLAIM = "Обслуживание карты «Вектор» стоит 120 рублей в месяц."


class CountingProvider:
    """Returns a fixed draft and records what it was asked for."""

    name = "counting"

    def __init__(self, draft: str | None = None) -> None:
        from answerdesk.testing.doubles import default_fake_provider_config

        self.config = default_fake_provider_config()
        self.draft = draft or draft_json([(CARD_CLAIM, "p1", CARD_FEE_QUOTE)])
        self.calls: list[list[Message]] = []
        self.schemas: list[dict[str, Any] | None] = []

    def complete(self, messages: Sequence[Message], *, response_schema: Any = None) -> Completion:
        self.calls.append(list(messages))
        self.schemas.append(dict(response_schema) if response_schema is not None else None)
        return Completion(
            text=self.draft,
            model="counting",
            input_tokens=10,
            output_tokens=max(1, len(self.draft) // 4),
        )

    def stream(self, messages: Sequence[Message], *, response_schema: Any = None):
        from answerdesk.contracts import CompletionChunk

        self.calls.append(list(messages))
        self.schemas.append(dict(response_schema) if response_schema is not None else None)
        return [
            CompletionChunk(delta=self.draft, done=False, model="counting"),
            CompletionChunk(delta="", done=True, model="counting"),
        ]


def test_compact_schema_drops_documentation_and_makes_why_optional() -> None:
    full = load_answer_schema()
    compacted, dropped = compact_schema(full)
    assert dropped == ["why"]
    assert "why" not in compacted["required"]
    assert "why" in compacted["properties"], "the field stays allowed, just not required"
    assert "description" not in str(compacted), "documentation tokens are the point"
    # Same shape: closed objects, every remaining property still required.
    assert compacted["additionalProperties"] is False
    assert set(compacted["required"]) == {"insufficient_evidence", "claims"}
    assert len(str(compacted)) < len(str(full))


def test_budget_schema_names_what_the_model_was_constrained_by() -> None:
    full = load_answer_schema()
    baseline = budget_schema(full, budget=AnswerBudget.baseline())
    compact = budget_schema(full, budget=AnswerBudget.budget())
    assert baseline.compact is False and baseline.required_dropped == []
    assert compact.compact is True and compact.required_dropped == ["why"]
    assert baseline.digest != compact.digest
    assert baseline.digest.startswith("sha256:")


def test_extractive_instruction_is_only_in_the_budget_arm() -> None:
    assert AnswerBudget.baseline().instruction() == ""
    instruction = AnswerBudget.budget().instruction()
    assert instruction
    assert EXTRACTIVE_INSTRUCTION.split("{")[0] in instruction
    assert "2" in instruction, "the claim limit is stated to the model as well as enforced"


def test_cap_claims_truncates_a_parsed_payload() -> None:
    payload = {"claims": [{"text": f"c{index}"} for index in range(5)], "why": ""}
    capped = cap_claims(payload, max_claims=2)
    assert len(capped["claims"]) == 2
    assert (
        cap_claims(payload, max_claims=9) is payload
        or len(cap_claims(payload, max_claims=9)["claims"]) == 5
    )


def test_pipeline_sends_the_compact_schema_and_the_instruction(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    provider = CountingProvider()
    pipeline = pipeline_factory(provider, repair_passes=0)
    pipeline.answer("Сколько стоит обслуживание карты?", passages, budget=AnswerBudget.budget())

    schema = provider.schemas[0]
    assert schema is not None and schema.get("additionalProperties") is False
    assert "why" not in (schema.get("required") or [])
    user_message = provider.calls[0][1].content
    assert "at most 2 claims" in user_message
    assert "Copy the answering sentence" in user_message


def test_pipeline_enforces_the_claim_limit_even_when_the_model_ignores_it(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    """The model returns five claims; the budget arm keeps two and drops the rest."""
    claims = [(CARD_CLAIM, "p1", CARD_FEE_QUOTE)] * 5
    provider = CountingProvider(draft_json(claims))
    pipeline = pipeline_factory(provider, repair_passes=0)
    result = pipeline.answer(
        "Сколько стоит обслуживание карты?", passages, budget=AnswerBudget.budget()
    )
    assert len(result.claims) == 2, [claim.text for claim in result.claims]


def test_baseline_arm_is_unchanged(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    """No budget → the shipped behaviour: full schema, no instruction, six claims allowed."""
    claims = [(CARD_CLAIM, "p1", CARD_FEE_QUOTE)] * 5
    provider = CountingProvider(draft_json(claims))
    pipeline = pipeline_factory(provider, repair_passes=0)
    result = pipeline.answer("Сколько стоит обслуживание карты?", passages)
    assert len(result.claims) == 5
    schema = provider.schemas[0]
    assert schema is not None and "why" in (schema.get("required") or [])
    assert "Copy the answering sentence" not in provider.calls[0][1].content
    assert result.signal.action.value == "answer"


def test_budget_cap_shrinks_the_answer_allowance_used_by_the_packer(
    pipeline_factory: Callable[..., AnswerPipeline],
) -> None:
    pipeline = pipeline_factory(CountingProvider())
    baseline_cap = pipeline.output_cap(AnswerBudget.baseline())
    budget_cap = pipeline.output_cap(AnswerBudget.budget())
    assert budget_cap < baseline_cap
    assert budget_cap == min(pipeline.max_answer_tokens, 160)


def test_usage_is_reported_for_both_arms(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    for arm in (AnswerBudget.baseline(), AnswerBudget.budget()):
        provider = CountingProvider()
        pipeline = pipeline_factory(provider, repair_passes=0)
        result = pipeline.answer("Сколько стоит обслуживание карты?", passages, budget=arm)
        assert isinstance(result.usage, Usage)
        assert result.usage.output_tokens > 0
