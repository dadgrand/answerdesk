"""Chunker tests: budget, overlap math, atomic units, heading paths (FR-ING-02/03/04/05).

# AT-ING-02  # AT-ING-03  # AT-ING-04  # AT-ING-05
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from answerdesk.contracts import Chunk, Document, Heading
from answerdesk.ingest.chunker import ChunkerConfig, StructureChunker
from answerdesk.ingest.errors import EmptyDocumentError
from answerdesk.ingest.loaders import load_markdown_text
from answerdesk.ingest.tokenizer import estimate_tokens

BUDGET = 120


def _config(**overrides: object) -> ChunkerConfig:
    base: dict[str, object] = {"token_budget": BUDGET, "overlap_tokens": 24}
    base.update(overrides)
    return ChunkerConfig.model_validate(base)


def _document(text: str, *, doc_id: str = "doc") -> Document:
    return load_markdown_text(text, source_path=f"{doc_id}.md", doc_id=doc_id, license="CC0-1.0")


def _big_table(rows: int) -> str:
    header = "| Показатель | Значение | Срок |\n| --- | --- | --- |\n"
    body = "\n".join(
        f"| Строка {index} | {index * 7} RUB | {index} дней |" for index in range(rows)
    )
    return f"# Тарифы\n\nВводный абзац про тарифы и лимиты.\n\n## Таблица\n\n{header}{body}\n"


def test_every_chunk_respects_the_budget() -> None:
    """AT-ING-03: 100% of chunks are within budget on a document that forces splitting."""
    paragraphs = "\n\n".join(
        f"Абзац номер {index}. " + "Возврат обрабатывается в течение 14 дней. " * 6
        for index in range(12)
    )
    document = _document(f"# Политика\n\n{paragraphs}\n")
    chunks = StructureChunker(_config()).chunk(document)
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.token_count <= BUDGET
        assert chunk.text.strip()
        assert not chunk.text.isspace()


def test_overlap_is_whole_units_within_the_documented_bound() -> None:
    """AT-ING-03: consecutive chunks of one section share text, bounded by overlap_tokens."""
    paragraphs = "\n\n".join(f"Пункт {index}: " + "детали процедуры " * 8 for index in range(10))
    document = _document(f"# Раздел\n\n{paragraphs}\n")
    config = _config(overlap_tokens=40)
    chunks = StructureChunker(config).chunk(document)
    assert len(chunks) >= 3
    overlaps = 0
    for previous, current in itertools.pairwise(chunks):
        shared = _shared_prefix(previous.text, current.text)
        if shared:
            overlaps += 1
            assert estimate_tokens(shared) <= config.overlap_tokens
            assert len(shared) < len(previous.text)  # the chunks still differ
    assert overlaps >= 1, "overlap_tokens > 0 must produce overlapping chunks"


def test_zero_overlap_disables_the_shared_prefix() -> None:
    """AT-ING-03: ``overlap_tokens=0`` is honoured, not silently ignored."""
    paragraphs = "\n\n".join(f"Пункт {index}: " + "детали процедуры " * 8 for index in range(10))
    document = _document(f"# Раздел\n\n{paragraphs}\n")
    chunks = StructureChunker(_config(overlap_tokens=0)).chunk(document)
    for previous, current in itertools.pairwise(chunks):
        assert _shared_prefix(previous.text, current.text) == ""


def test_overlap_never_spans_a_section_boundary() -> None:
    """AT-ING-02/03: a heading is a hard boundary — the next chunk starts fresh."""
    text = (
        "# Раздел А\n\n"
        + "Первый раздел говорит о лимитах. " * 20
        + "\n\n# Раздел Б\n\nВторой раздел говорит о комиссиях.\n"
    )
    chunks = StructureChunker(_config()).chunk(_document(text))
    second = next(chunk for chunk in chunks if chunk.heading_path == ["Раздел Б"])
    assert "Первый раздел" not in second.text


def test_heading_path_and_breadcrumb_on_every_chunk() -> None:
    """AT-ING-04: structured formats give every chunk a heading path and breadcrumb."""
    text = (
        "# Продукт\n\nВведение в продукт.\n\n"
        "## Биллинг\n\nОплата раз в месяц.\n\n"
        "### Возвраты\n\nВозврат за 14 дней.\n"
    )
    chunks = StructureChunker(_config()).chunk(_document(text))
    assert chunks
    for chunk in chunks:
        assert chunk.heading_path, f"chunk {chunk.chunk_id} lost its heading path"
        assert chunk.breadcrumb == " > ".join(chunk.heading_path)
        assert chunk.doc_id == "doc"
        assert chunk.chunk_id.startswith("doc::c")


def test_table_is_atomic_when_it_fits() -> None:
    """AT-ING-02: a table that fits the budget is never split."""
    document = _document(_big_table(3))
    chunks = StructureChunker(_config(token_budget=512, overlap_tokens=0)).chunk(document)
    table_chunks = [chunk for chunk in chunks if "| Строка 0 |" in chunk.text]
    assert len(table_chunks) == 1
    assert "| Строка 2 |" in table_chunks[0].text


def test_oversized_table_splits_at_rows_and_repeats_the_header() -> None:
    """AT-ING-02: an oversized table splits at row boundaries with the header repeated."""
    document = _document(_big_table(60))
    chunks = StructureChunker(_config(overlap_tokens=0)).chunk(document)
    table_chunks = [chunk for chunk in chunks if chunk.metadata["table_header_repeated"]]
    assert len(table_chunks) > 1
    for chunk in table_chunks:
        lines = chunk.text.splitlines()
        assert lines[0].startswith("| Показатель |"), "header must be repeated on every fragment"
        assert lines[1].startswith("| ---")
        for row in lines[2:]:
            assert row.startswith("| ") and row.endswith(" |"), "a row was split mid-cell"
        assert chunk.token_count <= BUDGET


def test_numbered_procedure_is_not_split_when_it_fits() -> None:
    """AT-ING-02: a procedure that fits stays in one chunk, steps intact."""
    text = "# Порядок\n\n1. Проверьте реквизиты.\n2. Подтвердите операцию.\n3. Сохраните номер.\n"
    chunks = StructureChunker(_config(token_budget=512, overlap_tokens=0)).chunk(_document(text))
    procedure = [chunk for chunk in chunks if "1. Проверьте" in chunk.text]
    assert len(procedure) == 1
    assert "3. Сохраните" in procedure[0].text


def test_faq_pair_is_never_separated() -> None:
    """AT-ING-02: Q and A of one FAQ pair always share a chunk."""
    text = "# Вопросы\n\nQ: Сколько идёт возврат?\nA: Возврат идёт 14 дней.\n"
    chunks = StructureChunker(_config()).chunk(_document(text))
    pair_chunks = [chunk for chunk in chunks if "Сколько идёт возврат" in chunk.text]
    assert len(pair_chunks) == 1
    assert "Возврат идёт 14 дней" in pair_chunks[0].text


def test_huge_paragraph_is_hard_split_without_loss() -> None:
    """AT-ING-03: a paragraph larger than the budget becomes several bounded chunks."""
    sentence = "Возврат обрабатывается в течение четырнадцати календарных дней. "
    document = _document(f"# Один абзац\n\n{sentence * 60}\n")
    chunks = StructureChunker(_config(overlap_tokens=0)).chunk(document)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= BUDGET
    words = " ".join(chunk.text for chunk in chunks).split()
    assert words.count("Возврат") == 60


def test_no_chunk_is_empty_and_ids_are_unique() -> None:
    """AT-ING-03: never an empty chunk; ``chunk_id`` is unique and ordered."""
    document = _document(_big_table(20))
    chunks = StructureChunker(_config()).chunk(document)
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert all(chunk.text.strip() for chunk in chunks)
    assert ids == sorted(ids)


def test_span_invariant_holds_for_unsplit_chunks() -> None:
    """AT-ING-04: a contiguous chunk's text is exactly ``document.text[start:end]``."""
    document = _document("# Раздел\n\nПервый абзац с деталями.\n\nВторой абзац с деталями.\n")
    chunks = StructureChunker(_config(token_budget=512, overlap_tokens=0)).chunk(document)
    for chunk in chunks:
        assert chunk.metadata["span_exact"] is True
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text


def test_chunking_is_deterministic() -> None:
    """AT-ING-03/06: the same document and config produce identical chunks."""
    document = _document(_big_table(30))
    first = StructureChunker(_config()).chunk(document)
    second = StructureChunker(_config()).chunk(document)
    assert [chunk.model_dump() for chunk in first] == [chunk.model_dump() for chunk in second]


def test_canonical_tokens_are_present_without_touching_the_text() -> None:
    """AT-ING-05: dates/money/ids are canonicalised into the sidecar, text is untouched."""
    text = "# Комиссии\n\n5 марта 2026 комиссия составила 12,5 €, заказ ORD-2026-000123.\n"
    document = _document(text)
    chunks = StructureChunker(_config()).chunk(document)
    joined_tokens = {token for chunk in chunks for token in chunk.canonical_tokens}
    assert "2026-03-05" in joined_tokens
    assert "12.50 EUR" in joined_tokens
    assert "ORD-2026-000123" in joined_tokens
    assert "5 марта 2026" in chunks[0].text
    assert "12,5 €" in chunks[0].text


def test_context_header_is_off_by_default_and_separate_when_on() -> None:
    """AT-ING-04: the contextual header is opt-in, bounded and stored separately."""
    text = "# Раздел\n\n" + "Абзац про возвраты и комиссии. " * 30
    document = _document(text)
    plain = StructureChunker(_config()).chunk(document)
    assert all(chunk.context_header is None for chunk in plain)

    config = _config(context_header=True)
    enriched = StructureChunker(config).chunk(document)
    for chunk in enriched:
        assert chunk.context_header is not None
        assert estimate_tokens(chunk.context_header) <= config.context_header_max_tokens
        assert chunk.context_header not in chunk.text


def test_invalid_config_is_rejected() -> None:
    """AT-ING-03: an overlap larger than the budget is a configuration error."""
    with pytest.raises(ValueError):
        ChunkerConfig(token_budget=100, overlap_tokens=100)
    with pytest.raises(ValueError):
        ChunkerConfig(token_budget=10)


def test_empty_document_is_refused() -> None:
    """AT-ING-03: no blocks -> a typed error, not an empty chunk list."""
    document = Document(
        doc_id="empty",
        title="Empty",
        source_path="empty.md",
        source_format="md",
        text="   ",
        license="CC0-1.0",
        sha256="0" * 64,
        byte_size=3,
        headings=[Heading(level=1, text="Empty", char_offset=0)],
    )
    with pytest.raises(EmptyDocumentError):
        StructureChunker(_config()).chunk(document)


def test_demo_style_document_keeps_every_invariant(tmp_path: Path) -> None:
    """AT-ING-03/04: a realistic article satisfies budget, heading and span invariants."""
    body = []
    for section in range(4):
        body.append(f"## Раздел {section}\n")
        body.append(" ".join(f"Предложение {section}-{index} о лимитах." for index in range(12)))
        body.append("")
    document = _document("# Статья\n\n" + "\n".join(body))
    chunks: list[Chunk] = StructureChunker(_config()).chunk(document)
    assert chunks
    for chunk in chunks:
        assert chunk.token_count <= BUDGET
        assert chunk.heading_path
        assert chunk.breadcrumb
        assert chunk.language == "ru"
        assert chunk.canonical_tokens is not None


def _shared_prefix(previous: str, current: str) -> str:
    """Longest suffix of ``previous`` that is also a prefix of ``current``."""
    limit = min(len(previous), len(current))
    for size in range(limit, 0, -1):
        if previous.endswith(current[:size]):
            return current[:size]
    return ""
