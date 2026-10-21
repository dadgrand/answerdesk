"""The block model: the structure a loader recovers and the chunker consumes.

A :class:`~answerdesk.contracts.Document` is a flat text plus a heading list, which is
not enough to honour FR-ING-02 (never split a table, a procedure or a FAQ pair). The
loaders therefore record a *block list* — the recovered structure — in
``Document.metadata["blocks"]`` as plain JSON objects. That keeps the frozen contract
untouched (a field cannot be added without a schema bump, ADR-006) while still giving
the chunker exact structure and exact character spans.

Invariants (asserted by the chunker and by ``tests/unit/ingest/test_structure.py``):

1. ``Document.text`` is exactly the block texts joined by ``"\\n\\n"``.
2. ``block.start_char``/``block.end_char`` index into ``document.text`` and satisfy
   ``document.text[start:end] == block.text``.
3. Blocks are ordered by ``start_char`` and do not overlap.
4. A heading block is a section boundary: no chunk may span it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from answerdesk.contracts import Document, Heading

#: Maximum heading depth (matches :class:`answerdesk.contracts.Heading`).
MAX_HEADING_LEVEL = 6

#: Separator between blocks; also the separator inside a chunk that spans blocks.
BLOCK_SEPARATOR = "\n\n"


class IngestModel(BaseModel):
    """Base for ingestion-internal models: strict, mutable-by-validation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BlockKind(StrEnum):
    """Recovered structural kind of one block."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FAQ_QUESTION = "faq_question"
    FAQ_ANSWER = "faq_answer"
    CODE = "code"
    QUOTE = "quote"


#: Kinds that must never be separated from their immediate predecessor (FR-ING-02).
PAIR_KINDS: tuple[BlockKind, ...] = (BlockKind.FAQ_QUESTION,)


class Block(IngestModel):
    """One structurally classified piece of a document."""

    kind: BlockKind
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    level: int | None = Field(default=None, ge=1, le=MAX_HEADING_LEVEL)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Block:
        if self.end_char <= self.start_char:
            raise ValueError("block end_char must be greater than start_char")
        if self.kind is BlockKind.HEADING and self.level is None:
            raise ValueError("a heading block needs a level")
        if self.kind is not BlockKind.HEADING and self.level is not None:
            raise ValueError("only heading blocks carry a level")
        return self

    @property
    def is_heading(self) -> bool:
        return self.kind is BlockKind.HEADING

    @property
    def token_length(self) -> int:
        from answerdesk.ingest.tokenizer import estimate_tokens

        return estimate_tokens(self.text)


class BlockBuilder:
    """Accumulates blocks and the matching document text with exact offsets.

    The builder is the only place that writes ``Document.text``, so invariant 1 and 2
    above are true by construction rather than by hope.
    """

    def __init__(self) -> None:
        self._blocks: list[Block] = []
        self._parts: list[str] = []
        self._offset = 0

    def add(
        self,
        kind: BlockKind,
        text: str,
        *,
        level: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Block | None:
        """Append one block. Blank text is dropped (never an empty chunk, FR-ING-03)."""
        cleaned = text.strip()
        if not cleaned:
            return None
        start = self._offset + (len(BLOCK_SEPARATOR) if self._parts else 0)
        end = start + len(cleaned)
        block = Block(
            kind=kind,
            text=cleaned,
            start_char=start,
            end_char=end,
            level=level,
            meta=dict(meta or {}),
        )
        self._blocks.append(block)
        self._parts.append(cleaned)
        self._offset = end
        return block

    def heading(self, level: int, text: str) -> Block | None:
        return self.add(BlockKind.HEADING, text, level=max(1, min(MAX_HEADING_LEVEL, level)))

    @property
    def blocks(self) -> list[Block]:
        return list(self._blocks)

    def result(self) -> tuple[str, list[Block]]:
        """``(document_text, blocks)`` — the two halves of invariant 1."""
        return BLOCK_SEPARATOR.join(self._parts), list(self._blocks)


def heading_path_of(blocks: Sequence[Block], position: int) -> list[str]:
    """Heading path (outermost first) that applies *at* block ``position``.

    Walks the prefix only, so the result is independent of later headings; a heading
    block at ``position`` closes its own level (a chunk never starts inside a heading).
    """
    stack: list[tuple[int, str]] = []
    for block in blocks[:position]:
        if not block.is_heading:
            continue
        level = block.level or 1
        stack = [(depth, text) for depth, text in stack if depth < level]
        stack.append((level, block.text))
    return [text for _, text in stack]


def breadcrumb_of(heading_path: Sequence[str]) -> str:
    """``["Product", "Billing", "Refunds"] -> "Product > Billing > Refunds"``."""
    return " > ".join(part.strip() for part in heading_path if part.strip())


def headings_from_blocks(blocks: Sequence[Block]) -> list[Heading]:
    """Contract ``Heading`` records for the document record (FR-ING-01)."""
    return [
        Heading(level=block.level or 1, text=block.text, char_offset=block.start_char)
        for block in blocks
        if block.is_heading
    ]


def blocks_from_metadata(metadata: dict[str, Any]) -> list[Block] | None:
    """Read the block list a loader stored; ``None`` when the document has none."""
    raw = metadata.get("blocks")
    if not isinstance(raw, list) or not raw:
        return None
    return [Block.model_validate(item) for item in raw]


def blocks_as_metadata(blocks: Sequence[Block]) -> list[dict[str, Any]]:
    """JSON-serialisable form, for ``Document.metadata["blocks"]``."""
    return [block.model_dump(mode="json") for block in blocks]


def blocks_from_document(document: Document) -> list[Block]:
    """Blocks of ``document`` — from metadata when present, otherwise reconstructed.

    The fallback exists because documents also arrive from places that do not run a
    loader (a golden-set fixture built by hand, a cache replay). It partitions the
    text at the heading offsets the contract carries and splits the remaining text at
    blank lines, so a chunker invariant holds for every document, not just ours.
    """
    stored = blocks_from_metadata(document.metadata)
    if stored is not None:
        return stored
    return _blocks_from_text_and_headings(document)


def _blocks_from_text_and_headings(document: Document) -> list[Block]:
    text = document.text
    headings = sorted(document.headings, key=lambda item: item.char_offset)
    builder = BlockBuilder()
    cursor = 0
    for heading in headings:
        offset = max(cursor, min(heading.char_offset, len(text)))
        _add_paragraph_blocks(builder, text[cursor:offset])
        builder.add(BlockKind.HEADING, heading.text, level=heading.level)
        cursor = offset
    _add_paragraph_blocks(builder, text[cursor:])
    _, blocks = builder.result()
    return blocks


def _add_paragraph_blocks(builder: BlockBuilder, chunk: str) -> None:
    """Split a text span at blank lines and append the non-empty pieces as paragraphs."""
    for piece in re.split(r"\n\s*\n", chunk):
        builder.add(BlockKind.PARAGRAPH, piece)


def render_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a table as a pipe table (one row per line, header repeated by the splitter).

    A pipe table is used instead of the ``Chunk.metadata`` structure because the text
    itself must be self-contained: a retrieved chunk is what a citation quotes
    verbatim (requirements CR-2), so the row values have to be in the chunk text.
    """
    width = max([len(header), *(len(row) for row in rows)]) if rows else len(header)
    if width == 0:
        return ""
    cells_header = [_cell(header[i] if i < len(header) else "") for i in range(width)]
    lines = ["| " + " | ".join(cells_header) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    for row in rows:
        cells = [_cell(row[i] if i < len(row) else "") for i in range(width)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _cell(value: str) -> str:
    """Escape a pipe and collapse newlines so one row stays one line (FR-ING-02)."""
    return " ".join(value.replace("|", "/").split())


def parse_table_rows(text: str) -> list[str]:
    """Row lines of a rendered table, without the header/separator pair.

    The chunker needs this to split an oversized table at row boundaries while
    repeating the header, which FR-ING-02 explicitly allows.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or not lines[1].strip().startswith("| ---"):
        return lines
    return lines[2:]


def table_header_lines(text: str) -> list[str]:
    """The header + separator lines of a rendered table (repeated on every fragment)."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 2 and lines[1].strip().startswith("| ---"):
        return lines[:2]
    return lines[:1]


__all__ = [
    "BLOCK_SEPARATOR",
    "MAX_HEADING_LEVEL",
    "PAIR_KINDS",
    "Block",
    "BlockBuilder",
    "BlockKind",
    "IngestModel",
    "blocks_as_metadata",
    "blocks_from_document",
    "blocks_from_metadata",
    "breadcrumb_of",
    "heading_path_of",
    "headings_from_blocks",
    "parse_table_rows",
    "render_table",
    "table_header_lines",
]
