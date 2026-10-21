"""Streaming vs final payload: the citation set must not change after verification.

CR-12 says a stream is never allowed to reveal a citation that is later retracted, and
FR-GEN-03 says numbering must be identical in streamed and final payloads. The model never
emits markers at all (it emits JSON), so the renderer is the only source of markers; this
test pins that down in both directions.

    # AT-GEN-03
"""

from __future__ import annotations

from collections.abc import Callable

from support import ScriptedProvider, draft_json

from answerdesk.contracts import RetrievedPassage
from answerdesk.generate.citations import CitationBuilder, markers_in, render_answer_text
from answerdesk.generate.draft import parse_answer_draft
from answerdesk.generate.pipeline import AnswerPipeline
from answerdesk.generate.prompts import PromptRegistry

QUESTION = "Сколько стоит обслуживание карты?"


def test_stream_contains_no_citation_markers(
    pipeline_factory: Callable[..., AnswerPipeline],
    registry: PromptRegistry,
    passages: list[RetrievedPassage],
    answer_draft: str,
) -> None:
    """# AT-GEN-03 — citations come from structure, so the stream carries none."""
    pipeline = pipeline_factory(ScriptedProvider([answer_draft]))
    result = pipeline.answer(QUESTION, passages)
    assert markers_in(result.text) == [1]

    prompt = registry.get("grounded-answer")
    messages = pipeline.build_messages(
        prompt, question=QUESTION, language="ru", packed=result.packed
    )
    streamed = ScriptedProvider([answer_draft]).stream(messages)
    chunks = list(streamed)
    text = "".join(chunk.delta for chunk in chunks)

    assert markers_in(text) == [], "the model must never emit citation markers"
    assert [chunk.done for chunk in chunks].count(True) == 1


def test_streamed_and_final_citations_are_identical(
    pipeline_factory: Callable[..., AnswerPipeline],
    registry: PromptRegistry,
    passages: list[RetrievedPassage],
    answer_draft: str,
) -> None:
    """# AT-GEN-03 — same transcript, same citation payload, streamed or not."""
    pipeline = pipeline_factory(ScriptedProvider([answer_draft]))
    result = pipeline.answer(QUESTION, passages)

    prompt = registry.get("grounded-answer")
    messages = pipeline.build_messages(
        prompt, question=QUESTION, language="ru", packed=result.packed
    )
    text = "".join(chunk.delta for chunk in ScriptedProvider([answer_draft]).stream(messages))
    streamed_build = CitationBuilder().build(parse_answer_draft(text), result.packed)

    assert [c.model_dump() for c in streamed_build.citations] == [
        c.model_dump() for c in result.citations
    ]
    assert [claim.citation_ids for claim in streamed_build.claims] == [
        claim.citation_ids for claim in result.claims
    ]


def test_numbering_is_stable_when_claims_are_rendered_twice(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    """# AT-GEN-03 — rendering is a pure function of the structure."""
    draft = draft_json(
        [
            (
                "Плата списывается первого числа каждого месяца.",
                "p1",
                "Плата списывается первого числа каждого месяца.",
            ),
            (
                "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц.",
                "p1",
                "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц.",
            ),
        ]
    )
    pipeline = pipeline_factory(ScriptedProvider([draft]))
    result = pipeline.answer(QUESTION, passages)

    first_render = render_answer_text(result.claims)
    second_render = render_answer_text(list(result.claims))
    assert first_render == second_render
    assert markers_in(first_render) == [1, 2]
    assert [citation.citation_id for citation in result.citations] == [1, 2]
