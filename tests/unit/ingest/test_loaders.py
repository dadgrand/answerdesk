"""Loader tests: structure, encodings, and the failure modes (FR-ING-01, FR-ING-04).

# AT-ING-01  # AT-ING-04
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from answerdesk.contracts import Document
from answerdesk.ingest.errors import EmptyDocumentError, UnsupportedFormatError
from answerdesk.ingest.loaders import decode_bytes, load_document, load_markdown_text
from answerdesk.ingest.structure import Block, BlockKind, blocks_from_metadata


def _blocks(document: Document) -> list[Block]:
    blocks = blocks_from_metadata(document.metadata)
    assert blocks is not None
    return blocks


def _heading_texts(document: Document) -> list[str]:
    return [heading.text for heading in document.headings]


def _kinds(document: Document) -> list[str]:
    return [block.kind.value for block in _blocks(document)]


def test_markdown_preserves_heading_hierarchy_and_tables(markdown_text: str) -> None:
    """AT-ING-01: headings keep their level; a table becomes one atomic block."""
    document = load_markdown_text(
        markdown_text, source_path="docs/policy.md", doc_id="md-fixture", license="CC0-1.0"
    )
    assert _heading_texts(document) == [
        "Политика возвратов",
        "Комиссии",
        "Порядок действий",
        "Вопросы",
    ]
    assert [heading.level for heading in document.headings] == [1, 2, 2, 2]
    kinds = _kinds(document)
    assert kinds.count("table") == 1
    assert kinds.count("faq_question") == 2
    assert kinds.count("faq_answer") == 2
    assert kinds.count("list_item") == 2
    table = next(block for block in _blocks(document) if block.kind is BlockKind.TABLE)
    assert "| Комиссия | 2,5 EUR | 14 дней |" in table.text


def test_markdown_front_matter_supplies_provenance(markdown_text: str) -> None:
    """AT-ING-01/04: front matter is the per-file license and language declaration."""
    document = load_markdown_text(markdown_text, source_path="docs/policy.md", doc_id=None)
    assert document.doc_id == "md-fixture"
    assert document.language == "ru"
    assert document.license == "CC0-1.0"
    assert document.title.startswith("Refund policy")


def test_document_spans_are_exact(markdown_text: str) -> None:
    """AT-ING-01: ``document.text`` is exactly the block list joined by blank lines."""
    document = load_markdown_text(markdown_text, source_path="docs/x.md", doc_id="spans")
    assert "\n\n".join(block.text for block in _blocks(document)) == document.text
    for block in _blocks(document):
        assert document.text[block.start_char : block.end_char] == block.text


def test_html_loader_recovers_structure(html_text: str, write_bytes: Callable[..., Path]) -> None:
    """AT-ING-01: h1/h2, table, list, definition-list FAQ pair and quote survive HTML."""
    path = write_bytes("page.html", html_text.encode("utf-8"))
    document = load_document(path)
    assert _heading_texts(document) == ["Возвраты и комиссии", "Комиссии"]
    kinds = set(_kinds(document))
    assert {"table", "list_item", "faq_question", "faq_answer", "quote"} <= kinds
    assert document.language == "ru"
    assert document.source_format == "html"
    assert document.title == "Возвраты и комиссии"


def test_docx_loader_recovers_structure(
    docx_bytes: bytes, write_bytes: Callable[..., Path]
) -> None:
    """AT-ING-01: Word heading styles become heading blocks with their level."""
    path = write_bytes("policy.docx", docx_bytes)
    document = load_document(path)
    assert document.source_format == "docx"
    assert _heading_texts(document) == ["Tariffs and fees", "Cash withdrawal"]
    assert [heading.level for heading in document.headings] == [1, 2]
    kinds = _kinds(document)
    assert kinds.count("table") == 1
    assert kinds.count("list_item") == 2
    assert "The basic plan costs 50 RUB per month." in document.text


def test_pdf_loader_recovers_headings_from_type_size(
    make_pdf: Callable[[list[tuple[float, str]]], bytes],
    write_bytes: Callable[..., Path],
) -> None:
    """AT-ING-01: a PDF has no markup, so headings come from font size and numbering."""
    pdf = make_pdf(
        [
            (18.0, "Tariffs and fees"),
            (9.0, "The basic plan costs 50 RUB per month."),
            (9.0, "The daily limit is 25 000 RUB."),
            (12.0, "2. Cash withdrawal"),
            (9.0, "Withdrawal is free in our own ATMs."),
        ]
    )
    document = load_document(write_bytes("policy.pdf", pdf))
    assert document.source_format == "pdf"
    assert "Tariffs and fees" in document.text
    headings = _heading_texts(document)
    assert "Tariffs and fees" in headings
    assert any("Cash withdrawal" in heading for heading in headings)
    assert document.language is None


def test_txt_loader_handles_encodings(write_bytes: Callable[..., Path]) -> None:
    """AT-ING-01: CP1251, UTF-16, BOM and CRLF all decode to the same text."""
    body = "Возврат обрабатывается 14 дней.\n\nКомиссия 2,5 EUR."
    variants = {
        "cp1251.txt": body.encode("cp1251"),
        "utf16.txt": body.encode("utf-16"),
        "bom.txt": b"\xef\xbb\xbf" + body.encode("utf-8"),
        "crlf.txt": body.replace("\n", "\r\n").encode("utf-8"),
    }
    texts = set()
    for name, data in variants.items():
        document = load_document(write_bytes(name, data))
        texts.add(document.text)
        assert document.source_format == "txt"
    assert texts == {body}


def test_decode_bytes_falls_back_to_cp1251() -> None:
    """AT-ING-01: a legacy Russian export must not raise a Unicode error."""
    assert decode_bytes("Привет".encode("cp1251")) == "Привет"
    assert decode_bytes(b"\xef\xbb\xbfhello") == "hello"


def test_unknown_suffix_is_refused(write_bytes: Callable[..., Path]) -> None:
    """AT-ING-01: an unsupported format fails loudly; it does not become an empty doc."""
    with pytest.raises(UnsupportedFormatError):
        load_document(write_bytes("data.rtf", b"{\\rtf1}"))


def test_empty_document_is_refused(write_bytes: Callable[..., Path]) -> None:
    """FR-ING-03: an empty file is an error, never a zero-chunk document."""
    with pytest.raises(EmptyDocumentError):
        load_document(write_bytes("empty.md", b"   \n\n  \n"))
    with pytest.raises(EmptyDocumentError):
        load_document(write_bytes("empty.txt", b"\n\n\n"))


def test_pdf_without_text_layer_is_refused(
    make_pdf: Callable[[list[tuple[float, str]]], bytes], write_bytes: Callable[..., Path]
) -> None:
    """A scanned PDF has no text; the loader says so instead of inventing content."""
    with pytest.raises(EmptyDocumentError):
        load_document(write_bytes("scanned.pdf", make_pdf([])))


def test_table_rows_are_one_line_each(markdown_text: str) -> None:
    """AT-ING-02 precondition: one table row is one rendered line, so it can be split."""
    document = load_markdown_text(markdown_text, source_path="docs/p.md", doc_id="rows")
    table = next(block for block in _blocks(document) if block.kind is BlockKind.TABLE)
    for line in table.text.splitlines():
        assert line.startswith("| ") and line.endswith(" |")


def test_heading_share_on_fixture_set(
    html_text: str, markdown_text: str, write_bytes: Callable[..., Path]
) -> None:
    """AT-ING-01 pass condition: 100% of fixture headings survive (target >= 95%)."""
    documents = [
        load_markdown_text(markdown_text, source_path="a.md", doc_id="a"),
        load_document(write_bytes("b.html", html_text.encode("utf-8"))),
    ]
    detected = sum(len(document.headings) for document in documents)
    assert detected == 6, f"detected {detected} of 6 fixture headings"


def test_block_kinds_are_closed_set(markdown_text: str) -> None:
    """The block model is a closed enum: a loader cannot invent a kind."""
    document = load_markdown_text(markdown_text, source_path="a.md", doc_id="a")
    kinds = {block.kind for block in _blocks(document)}
    assert kinds <= set(BlockKind)


def test_broken_front_matter_keeps_the_document() -> None:
    """A title with an unquoted colon must not cost us the document or its provenance."""
    text = "---\ntitle: Refunds: rules\nlicense: CC0-1.0\n---\n\n# Refunds\n\nBody text."
    document = load_markdown_text(text, source_path="docs/r.md", doc_id="r1")
    assert document.text.startswith("Refunds")
    assert "Body text." in document.text
