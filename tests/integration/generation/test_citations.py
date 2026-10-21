"""Citation contract: phantom labels, non-verbatim quotes, spans, numbering.

These are the deterministic checks of requirements §7.2 — no model, no judge, no
tolerance. A citation either resolves to a real span of a packed chunk or it is rejected.

    # AT-GEN-02  # AT-GEN-03
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from support import (
    CARD_FEE_QUOTE,
    ScriptedProvider,
    draft_json,
)

from answerdesk.contracts import Channel, Chunk, RetrievedPassage
from answerdesk.generate.citations import (
    CitationBuilder,
    find_quote_span,
    markers_in,
    normalise_whitespace,
    render_answer_text,
    validate_render,
)
from answerdesk.generate.draft import AnswerDraft
from answerdesk.generate.packer import ContextPacker
from answerdesk.generate.pipeline import AnswerPipeline

PACKER = ContextPacker()


def packed_for(passage: RetrievedPassage):
    return PACKER.pack([passage])


def draft(*claims: tuple[str, str, str]) -> AnswerDraft:
    return AnswerDraft.model_validate_json(draft_json(claims))


# --------------------------------------------------------------------------------------
# Whitespace-normalised substring matching (CR-3) and spans (CR-4)
# --------------------------------------------------------------------------------------


def test_quote_span_maps_back_to_raw_offsets() -> None:
    """# AT-GEN-02 — the span indexes the stored chunk text, and contains the quote."""
    chunk_text = "Обслуживание карты\n   стоит 120 рублей."
    quote = "Обслуживание карты стоит 120"
    span = find_quote_span(chunk_text, quote)
    assert span is not None
    start, end = span
    assert chunk_text[start:end].split() == quote.split()
    assert start == 0
    assert end <= len(chunk_text)


def test_quote_must_be_verbatim_not_paraphrased() -> None:
    """# AT-GEN-02 — CR-3 is case-sensitive and whitespace-normalised only."""
    assert find_quote_span("Refunds are processed within 14 days.", "refunds are processed") is None
    assert (
        find_quote_span("Refunds are processed within 14 days.", "Refunds   are processed")
        is not None
    )
    assert find_quote_span("Refunds are processed within 14 days.", "") is None
    assert find_quote_span("Refunds are 14 days.", "14 days and more") is None


def test_normalise_whitespace_collapses_runs() -> None:
    assert normalise_whitespace("  a\n\n b\t c  ") == "a b c"


# --------------------------------------------------------------------------------------
# Citation building
# --------------------------------------------------------------------------------------


def test_citation_resolves_to_the_packed_chunk(
    make_passage: Callable[..., RetrievedPassage],
) -> None:
    """# AT-GEN-02 — chunk id, doc id, title, heading, quote, span and doc hash."""
    packed = packed_for(make_passage(score=0.9, rank=1))
    built = CitationBuilder().build(draft((CARD_FEE_QUOTE, "p1", CARD_FEE_QUOTE)), packed)

    assert built.issues == []
    assert len(built.citations) == 1
    citation = built.citations[0]
    chunk = packed.passages[0].chunk
    assert citation.citation_id == 1
    assert citation.chunk_id == chunk.chunk_id
    assert citation.doc_id == chunk.doc_id
    assert citation.doc_title == "Тарифы по карте «Вектор»"
    assert citation.heading_path == chunk.heading_path
    assert citation.quote == CARD_FEE_QUOTE
    assert chunk.text[citation.start_char : citation.end_char] == CARD_FEE_QUOTE
    assert citation.doc_hash == chunk.doc_hash


def test_phantom_passage_label_is_rejected(
    make_passage: Callable[..., RetrievedPassage],
) -> None:
    """# AT-GEN-02 — CR-2: a citation must resolve to a passage in the packed context."""
    packed = packed_for(make_passage(score=0.9, rank=1))
    built = CitationBuilder().build(draft((CARD_FEE_QUOTE, "p7", CARD_FEE_QUOTE)), packed)

    assert built.citations == []
    assert built.claims[0].citation_ids == []
    assert built.phantom_labels == ["p7"]
    assert [issue.code for issue in built.issues] == ["phantom_passage"]


def test_quote_from_the_wrong_passage_is_rejected(
    make_passage: Callable[..., RetrievedPassage],
    make_chunk: Callable[..., Chunk],
) -> None:
    """# AT-GEN-02 — a real quote attached to the wrong chunk is still a phantom support."""
    other = make_chunk(
        chunk_id="chunk-other",
        text="Суточный лимит переводов — 150 000 рублей.",
        doc_id="doc-other",
        title="Лимиты",
    )
    packed = PACKER.pack(
        [
            make_passage(score=0.9, rank=1),
            RetrievedPassage(chunk=other, channel=Channel.BM25, score=0.6, rank=2),
        ]
    )
    built = CitationBuilder().build(draft((CARD_FEE_QUOTE, "p2", CARD_FEE_QUOTE)), packed)
    assert built.citations == []
    assert [issue.code for issue in built.issues] == ["quote_not_found"]


def test_numbering_is_by_first_appearance(
    make_chunk: Callable[..., Chunk],
    make_passage: Callable[..., RetrievedPassage],
) -> None:
    """# AT-GEN-03 — citation ids follow the order the claims introduce them."""
    first = make_chunk(chunk_id="chunk-a", text="Alpha sentence one.", doc_id="doc-a", title="A")
    second = make_chunk(chunk_id="chunk-b", text="Beta sentence two.", doc_id="doc-b", title="B")
    packed = PACKER.pack(
        [
            make_passage(first, score=0.9, rank=1),
            make_passage(second, score=0.8, rank=2),
        ]
    )
    # Deliberately cite p2 first, then p1.
    built = CitationBuilder().build(
        draft(
            ("Beta sentence two.", "p2", "Beta sentence two."),
            ("Alpha sentence one.", "p1", "Alpha sentence one."),
        ),
        packed,
    )
    assert [c.citation_id for c in built.citations] == [1, 2]
    assert [c.chunk_id for c in built.citations] == ["chunk-b", "chunk-a"]
    assert [claim.citation_ids for claim in built.claims] == [[1], [2]]

    text = render_answer_text(built.claims)
    assert markers_in(text) == [1, 2]
    assert validate_render(text, built.claims, built.citations) == []


def test_repeated_support_link_reuses_one_citation(
    make_passage: Callable[..., RetrievedPassage],
) -> None:
    """# AT-GEN-03 — the same chunk+quote cited twice keeps a stable single number."""
    packed = packed_for(make_passage(score=0.9, rank=1))
    built = CitationBuilder().build(
        draft(
            (CARD_FEE_QUOTE, "p1", CARD_FEE_QUOTE),
            (
                "Плата списывается первого числа каждого месяца.",
                "p1",
                "Плата списывается первого числа каждого месяца.",
            ),
        ),
        packed,
    )
    assert [c.citation_id for c in built.citations] == [1, 2]
    # Both claims share the first citation only when the quote is identical; here the
    # quotes differ, so each claim gets its own citation.
    assert [claim.citation_ids for claim in built.claims] == [[1], [2]]

    repeated = CitationBuilder().build(
        draft(
            (CARD_FEE_QUOTE, "p1", CARD_FEE_QUOTE),
            ("Ещё одна формулировка того же факта.", "p1", CARD_FEE_QUOTE),
        ),
        packed,
    )
    assert len(repeated.citations) == 1
    assert [claim.citation_ids for claim in repeated.claims] == [[1], [1]]


def test_multiple_citations_on_one_claim_keep_order(
    make_chunk: Callable[..., Chunk],
    make_passage: Callable[..., RetrievedPassage],
) -> None:
    first = make_chunk(chunk_id="chunk-a", text="Alpha sentence one.", doc_id="doc-a", title="A")
    second = make_chunk(chunk_id="chunk-b", text="Beta sentence two.", doc_id="doc-b", title="B")
    packed = PACKER.pack(
        [make_passage(first, score=0.9, rank=1), make_passage(second, score=0.8, rank=2)]
    )
    built = CitationBuilder().build(
        draft(
            (
                "Alpha sentence one and Beta sentence two.",
                "p1",
                "Alpha sentence one.",
            ),
            (
                "Beta sentence two.",
                "p2",
                "Beta sentence two.",
            ),
        ),
        packed,
    )
    built_claim = built.claims[0]
    assert built_claim.citation_ids == [1]

    both = CitationBuilder().build(
        AnswerDraft.model_validate(
            {
                "insufficient_evidence": False,
                "why": "",
                "claims": [
                    {
                        "text": "Alpha and beta.",
                        "citations": [
                            {"passage": "p2", "quote": "Beta sentence two."},
                            {"passage": "p1", "quote": "Alpha sentence one."},
                        ],
                    }
                ],
            }
        ),
        packed,
    )
    assert both.claims[0].citation_ids == [1, 2]
    assert [c.chunk_id for c in both.citations] == ["chunk-b", "chunk-a"]


def test_render_validation_catches_a_wrong_marker(
    make_passage: Callable[..., RetrievedPassage],
) -> None:
    """# AT-GEN-02 / CR-5 — markers and structure must stay bijective."""
    packed = packed_for(make_passage(score=0.9, rank=1))
    built = CitationBuilder().build(draft((CARD_FEE_QUOTE, "p1", CARD_FEE_QUOTE)), packed)
    tampered = f"{CARD_FEE_QUOTE} [2]"
    problems = validate_render(tampered, built.claims, built.citations)
    assert problems and "do not match" in problems[0]
    assert validate_render(f"{CARD_FEE_QUOTE} [1]", built.claims, built.citations) == []


# --------------------------------------------------------------------------------------
# End to end: unsupported citations never reach the answer
# --------------------------------------------------------------------------------------


def test_phantom_citation_never_reaches_the_answer(
    pipeline_factory: Callable[..., AnswerPipeline],
    make_passage: Callable[..., RetrievedPassage],
) -> None:
    """# AT-GEN-02 — INV-02: no displayed citation that does not resolve."""
    provider = ScriptedProvider(
        [
            draft_json([(CARD_FEE_QUOTE, "p9", CARD_FEE_QUOTE)]),
            draft_json([(CARD_FEE_QUOTE, "p9", CARD_FEE_QUOTE)]),
        ]
    )
    pipeline = pipeline_factory(provider)
    result = pipeline.answer("Сколько стоит обслуживание карты?", [make_passage()])

    assert result.signal.is_answer is False
    assert result.citations == []
    assert result.text == ""
    assert result.signal.reason_code.value == "CITATION_FAILED"


def test_answer_citations_are_substrings_of_packed_chunks(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
    answer_draft: str,
) -> None:
    """# AT-GEN-02 — every displayed citation resolves and quotes verbatim."""
    provider = ScriptedProvider([answer_draft])
    pipeline = pipeline_factory(provider)
    result = pipeline.answer("Сколько стоит обслуживание карты?", passages)

    assert result.signal.is_answer is True
    packed_chunks = {item.chunk_id: item.chunk for item in result.packed.passages}
    for citation in result.citations:
        chunk = packed_chunks[citation.chunk_id]
        assert citation.quote in normalise_whitespace(chunk.text)
        assert chunk.text[citation.start_char : citation.end_char] == citation.quote
        assert citation.doc_hash == chunk.doc_hash


@pytest.mark.parametrize(
    ("claim_text", "quote"),
    [
        (CARD_FEE_QUOTE, CARD_FEE_QUOTE),
        (
            "Плата списывается первого числа каждого месяца.",
            "Плата списывается первого числа каждого месяца.",
        ),
    ],
)
def test_rendered_markers_match_claim_structure(
    claim_text: str,
    quote: str,
    pipeline_factory: Callable[..., AnswerPipeline],
    make_passage: Callable[..., RetrievedPassage],
) -> None:
    """# AT-GEN-03 — FR-GEN-01: markers come from structure, never parsed out of prose."""
    pipeline = pipeline_factory(ScriptedProvider([draft_json([(claim_text, "p1", quote)])]))
    result = pipeline.answer("Сколько стоит обслуживание карты?", [make_passage()])
    assert result.signal.is_answer is True
    assert markers_in(result.text) == [citation.citation_id for citation in result.citations]
    assert validate_render(result.text, result.claims, result.citations) == []
