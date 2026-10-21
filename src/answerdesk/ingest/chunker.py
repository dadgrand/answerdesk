"""Structure-first chunking with a hard token budget (FR-ING-02/03/04/05).

The chunker consumes the block list a loader recovered and produces
:class:`~answerdesk.contracts.Chunk` records. Three properties matter more than
elegance:

1. **Never split an atomic unit.** A table, a numbered procedure (a run of list items)
   and a FAQ question/answer pair stay together (FR-ING-02). A unit is only split when
   it alone exceeds the budget, and then it is split at its natural boundary — a table
   at a row, a procedure at a step — with the table header repeated on every fragment.
2. **Every chunk carries its provenance.** ``doc_id``, ``chunk_id``, ``heading_path``,
   ``breadcrumb``, character span, token count and canonical tokens (FR-ING-04/05).
3. **The budget is a hard cap.** No chunk exceeds ``token_budget``; nothing empty or
   whitespace-only is emitted (FR-ING-03).

Overlap math (documented because "some overlap" is not a specification)::

    overlap_budget = min(overlap_tokens, floor(previous_chunk_tokens / 2))
    overlap        = the longest suffix of the previous chunk's *whole units* whose
                     token sum is <= overlap_budget, kept only if
                     overlap_tokens + next_unit_tokens <= token_budget

Whole units, never partial text: half a quoted table row in the overlap would be a
citation hazard, and half a step of a procedure is worse than no overlap. Because the
suffix is bounded by ``previous_chunk_tokens / 2``, two consecutive chunks always
differ; because it is dropped when it would overflow, the budget holds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import model_validator

from answerdesk.contracts import Chunk, Document
from answerdesk.ingest.canonical import canonicalize
from answerdesk.ingest.errors import EmptyDocumentError
from answerdesk.ingest.loaders import split_sentences
from answerdesk.ingest.structure import (
    Block,
    BlockKind,
    IngestModel,
    blocks_from_document,
    breadcrumb_of,
    heading_path_of,
    parse_table_rows,
    table_header_lines,
)
from answerdesk.ingest.tokenizer import estimate_tokens, truncate_to_tokens

#: Kinds that always form their own unit even when neighbours share the kind.
_ATOMIC_KINDS = frozenset({BlockKind.TABLE})

_WORD_RE = re.compile(r"\S+")
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?…]+[\"'»”’)\]]*\s+|[。！？；]")


class ChunkerConfig(IngestModel):
    """Chunking knobs. Defaults are the shipped contract (FR-ING-03)."""

    token_budget: int = 512
    overlap_tokens: int = 64
    context_header: bool = False
    context_header_min_tokens: int = 50
    context_header_max_tokens: int = 100
    chunk_id_prefix: str = "c"
    repeat_table_header: bool = True

    @model_validator(mode="after")
    def _bounds(self) -> ChunkerConfig:
        if self.token_budget < 64:
            raise ValueError("token_budget must be >= 64 (smaller chunks stop being useful)")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0")
        if self.overlap_tokens >= self.token_budget:
            raise ValueError("overlap_tokens must be smaller than token_budget")
        if not 0 < self.context_header_min_tokens <= self.context_header_max_tokens:
            raise ValueError("context header bounds must satisfy 0 < min <= max")
        return self


@dataclass
class _Unit:
    """One atomic chunking unit: blocks that may never be separated."""

    block_indices: list[int] = field(default_factory=list)
    start: int = 0
    end: int = 0
    tokens: int = 0
    section: int = 0
    kind: BlockKind = BlockKind.PARAGRAPH

    @property
    def first_block(self) -> int:
        return self.block_indices[0]


@dataclass
class _Fragment:
    """An oversized unit cut down to the budget (table row group, step, sentence...)."""

    text: str
    start: int
    end: int
    tokens: int
    kind: BlockKind
    first_block: int
    section: int
    table_header_repeated: bool = False


class StructureChunker:
    """The shipped :class:`answerdesk.contracts.Chunker` implementation."""

    name = "structure-first-v1"

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    # -- public API -----------------------------------------------------------------
    def chunk(self, document: Document) -> list[Chunk]:
        blocks = blocks_from_document(document)
        if not blocks:
            raise EmptyDocumentError(f"document {document.doc_id!r} has no blocks to chunk")
        units = _build_units(blocks, document)
        if not units:
            raise EmptyDocumentError(f"document {document.doc_id!r} produced no chunkable units")

        chunks: list[Chunk] = []
        pending: list[_Unit] = []
        pending_tokens = 0
        pending_section = 0
        pending_prefix_start: int | None = None
        prev_units: list[_Unit] = []
        prev_tokens = 0
        prev_section = -1
        prev_text = ""
        prev_end = 0

        def emit(units_to_emit: list[_Unit], tokens: int, section: int, prefix: int | None) -> None:
            nonlocal prev_units, prev_tokens, prev_section, prev_text, prev_end
            chunk = self._make_chunk(document, blocks, units_to_emit, len(chunks), prefix)
            chunks.append(chunk)
            prev_units, prev_tokens, prev_section = list(units_to_emit), tokens, section
            prev_text, prev_end = chunk.text, chunk.end_char

        def flush() -> None:
            nonlocal pending, pending_tokens, pending_prefix_start
            if pending:
                emit(pending, pending_tokens, pending_section, pending_prefix_start)
            pending, pending_tokens, pending_prefix_start = [], 0, None

        def seed_overlap(section: int) -> tuple[list[_Unit], int, int | None]:
            """Overlap seed: whole units first, then a sentence-level textual tail."""
            if not prev_units or prev_section != section or self.config.overlap_tokens <= 0:
                return [], 0, None
            budget = min(self.config.overlap_tokens, prev_tokens // 2)
            if budget <= 0:
                return [], 0, None
            seed: list[_Unit] = []
            total = 0
            for candidate in reversed(prev_units):
                if total + candidate.tokens > budget:
                    break
                seed.insert(0, candidate)
                total += candidate.tokens
            if seed:
                return seed, total, None
            tail = _overlap_tail(prev_text, budget)
            if not tail:
                return [], 0, None
            return [], estimate_tokens(tail), prev_end - len(tail)

        for unit in units:
            if pending and unit.section != pending_section:
                flush()
            if unit.tokens > self.config.token_budget:
                flush()
                for fragment in self._split_oversized(document, blocks, unit):
                    fragment_chunk = self._make_fragment_chunk(
                        document, blocks, fragment, len(chunks)
                    )
                    chunks.append(fragment_chunk)
                prev_units, prev_tokens, prev_section = [], 0, -1
                prev_text, prev_end = "", 0
                continue
            if pending and pending_tokens + unit.tokens > self.config.token_budget:
                flush()
            if not pending:
                seed, seed_tokens, prefix_start = seed_overlap(unit.section)
                if (seed or prefix_start is not None) and seed_tokens + unit.tokens <= (
                    self.config.token_budget
                ):
                    pending, pending_tokens = seed, seed_tokens
                    pending_prefix_start = prefix_start
                else:
                    pending, pending_tokens, pending_prefix_start = [], 0, None
            pending.append(unit)
            pending_tokens += unit.tokens
            pending_section = unit.section
        flush()
        return chunks

    def chunk_all(self, documents: list[Document]) -> list[Chunk]:
        """Chunk many documents in the given order (the corpus builder's entry point)."""
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk(document))
        return chunks

    # -- chunk construction ---------------------------------------------------------
    def _make_chunk(
        self,
        document: Document,
        blocks: list[Block],
        units: list[_Unit],
        ordinal: int,
        prefix_start: int | None = None,
    ) -> Chunk:
        start = units[0].start if prefix_start is None else min(prefix_start, units[0].start)
        end = units[-1].end
        raw = document.text[start:end]
        text = raw.strip()
        if not text:
            raise EmptyDocumentError(f"chunker produced an empty chunk in {document.doc_id!r}")
        leading = len(raw) - len(raw.lstrip())
        heading_path = heading_path_of(blocks, units[0].first_block)
        kinds: list[str] = []
        for unit in units:
            kind = unit.kind.value
            if kind not in kinds:
                kinds.append(kind)
        return self._build_chunk(
            document,
            text=text,
            start_char=start + leading,
            end_char=start + leading + len(text),
            ordinal=ordinal,
            heading_path=heading_path,
            kinds=kinds,
            first_block=units[0].first_block,
            section=units[0].section,
            table_header_repeated=False,
            split=False,
        )

    def _make_fragment_chunk(
        self, document: Document, blocks: list[Block], fragment: _Fragment, ordinal: int
    ) -> Chunk:
        start = min(fragment.start, fragment.end - 1)
        end = max(fragment.end, start + 1)
        return self._build_chunk(
            document,
            text=fragment.text.strip(),
            start_char=start,
            end_char=end,
            ordinal=ordinal,
            heading_path=heading_path_of(blocks, fragment.first_block),
            kinds=[fragment.kind.value],
            first_block=fragment.first_block,
            section=fragment.section,
            table_header_repeated=fragment.table_header_repeated,
            split=True,
        )

    def _build_chunk(
        self,
        document: Document,
        *,
        text: str,
        start_char: int,
        end_char: int,
        ordinal: int,
        heading_path: list[str],
        kinds: list[str],
        first_block: int,
        section: int,
        table_header_repeated: bool,
        split: bool,
    ) -> Chunk:
        if not text:
            raise EmptyDocumentError(f"chunker produced an empty chunk in {document.doc_id!r}")
        exact_span = document.text[start_char:end_char] == text
        return Chunk(
            chunk_id=f"{document.doc_id}::{self.config.chunk_id_prefix}{ordinal:04d}",
            doc_id=document.doc_id,
            text=text,
            heading_path=list(heading_path),
            breadcrumb=breadcrumb_of(heading_path),
            start_char=start_char,
            end_char=end_char,
            token_count=estimate_tokens(text),
            language=document.language,
            context_header=(
                self._context_header(document, heading_path, first_block, text)
                if self.config.context_header
                else None
            ),
            canonical_tokens=canonicalize(text),
            metadata={
                "ordinal": ordinal,
                "block_kinds": kinds,
                "section_index": section,
                "split": split,
                "span_exact": exact_span,
                "table_header_repeated": table_header_repeated,
                "doc_title": document.title,
                "license": document.license,
                "source_format": document.source_format,
                "chunker": self.name,
            },
        )

    # -- oversized units -------------------------------------------------------------
    def _split_oversized(
        self, document: Document, blocks: list[Block], unit: _Unit
    ) -> list[_Fragment]:
        first_block = blocks[unit.first_block]
        if first_block.kind is BlockKind.TABLE:
            return self._split_table(unit, first_block)
        if first_block.kind is BlockKind.LIST_ITEM and len(unit.block_indices) > 1:
            return self._split_list(unit, blocks)
        return self._split_text(unit, blocks)

    def _split_table(self, unit: _Unit, block: Block) -> list[_Fragment]:
        """Split at row boundaries; repeat the header on every fragment (FR-ING-02)."""
        header_lines = table_header_lines(block.text)
        header = "\n".join(header_lines)
        header_tokens = estimate_tokens(header) if header else 0
        rows = parse_table_rows(block.text)
        if not rows:
            return [
                _Fragment(
                    text=block.text,
                    start=block.start_char,
                    end=block.end_char,
                    tokens=estimate_tokens(block.text),
                    kind=BlockKind.TABLE,
                    first_block=unit.first_block,
                    section=unit.section,
                )
            ]
        fragments: list[_Fragment] = []
        current: list[str] = []

        def push(rows_to_push: list[str], text_override: str | None = None) -> None:
            body = "\n".join(rows_to_push)
            text = text_override if text_override is not None else "\n".join([header, body])
            fragments.append(
                _Fragment(
                    text=text,
                    start=block.start_char,
                    end=block.end_char,
                    tokens=estimate_tokens(text),
                    kind=BlockKind.TABLE,
                    first_block=unit.first_block,
                    section=unit.section,
                    table_header_repeated=self.config.repeat_table_header and bool(header),
                )
            )

        for row in rows:
            candidate = "\n".join([header, *current, row])
            if estimate_tokens(candidate) <= self.config.token_budget:
                current.append(row)
                continue
            if current:
                push(list(current))
                current = []
            if estimate_tokens("\n".join([header, row])) <= self.config.token_budget:
                current.append(row)
                continue
            # Even one row is over budget: hard-split the row, header repeated.
            for piece in _hard_split(row, max(1, self.config.token_budget - header_tokens)):
                push([piece])
        if current:
            push(list(current))
        return fragments

    def _split_list(self, unit: _Unit, blocks: list[Block]) -> list[_Fragment]:
        """Split a procedure at step boundaries; a single oversized step is hard-split."""
        fragments: list[_Fragment] = []
        current: list[str] = []
        current_start: int | None = None
        current_end = 0

        def push() -> None:
            nonlocal current, current_start, current_end
            if not current:
                return
            text = "\n".join(current)
            fragments.append(
                _Fragment(
                    text=text,
                    start=current_start if current_start is not None else unit.start,
                    end=current_end or unit.end,
                    tokens=estimate_tokens(text),
                    kind=BlockKind.LIST_ITEM,
                    first_block=unit.first_block,
                    section=unit.section,
                )
            )
            current, current_start = [], None

        for index in unit.block_indices:
            block = blocks[index]
            if estimate_tokens(block.text) > self.config.token_budget:
                push()
                for piece in _hard_split(block.text, self.config.token_budget):
                    fragments.append(
                        _Fragment(
                            text=piece,
                            start=block.start_char,
                            end=block.end_char,
                            tokens=estimate_tokens(piece),
                            kind=BlockKind.LIST_ITEM,
                            first_block=unit.first_block,
                            section=unit.section,
                        )
                    )
                continue
            candidate_text = "\n".join([*current, block.text])
            if current and estimate_tokens(candidate_text) > self.config.token_budget:
                push()
            current_start = block.start_char if current_start is None else current_start
            current.append(block.text)
            current_end = block.end_char
        push()
        return fragments

    def _split_text(self, unit: _Unit, blocks: list[Block]) -> list[_Fragment]:
        """Split prose at sentence boundaries, then at word boundaries."""
        fragments: list[_Fragment] = []
        for index in unit.block_indices:
            block = blocks[index]
            pieces = _hard_split(block.text, self.config.token_budget)
            if not pieces:
                continue
            if len(pieces) == 1:
                fragments.append(
                    _Fragment(
                        text=pieces[0],
                        start=block.start_char,
                        end=block.end_char,
                        tokens=estimate_tokens(pieces[0]),
                        kind=block.kind,
                        first_block=unit.first_block,
                        section=unit.section,
                    )
                )
                continue
            cursor = 0
            for piece in pieces:
                offset = block.text.find(piece, cursor)
                offset = cursor if offset == -1 else offset
                start = block.start_char + offset
                fragments.append(
                    _Fragment(
                        text=piece,
                        start=start,
                        end=start + len(piece),
                        tokens=estimate_tokens(piece),
                        kind=block.kind,
                        first_block=unit.first_block,
                        section=unit.section,
                    )
                )
                cursor = offset + len(piece)
        return fragments

    # -- contextual header (FR-ING-04, off by default) ------------------------------
    def _context_header(
        self,
        document: Document,
        heading_path: list[str],
        first_block: int,
        chunk_text: str,
    ) -> str:
        parts = [f"Document: {document.title}"]
        if heading_path:
            parts.append(f"Section: {breadcrumb_of(heading_path)}")
        if document.language:
            parts.append(f"Language: {document.language}")
        parts.append(f"Context: {_intro(chunk_text)}")
        header = "\n".join(parts)
        if estimate_tokens(header) < self.config.context_header_min_tokens:
            header = f"{header}\nExcerpt: {chunk_text}"
        return truncate_to_tokens(header, self.config.context_header_max_tokens)


# --------------------------------------------------------------------------------------
# unit construction
# --------------------------------------------------------------------------------------


def _build_units(blocks: list[Block], document: Document) -> list[_Unit]:
    """Group blocks into atomic units, tracking section numbers and exact spans."""
    units: list[_Unit] = []
    section = 0
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if block.is_heading:
            section += 1
            index += 1
            continue
        if block.kind is BlockKind.LIST_ITEM:
            indices = [index]
            cursor = index + 1
            while cursor < len(blocks) and blocks[cursor].kind is BlockKind.LIST_ITEM:
                indices.append(cursor)
                cursor += 1
            units.append(_unit(document, indices, blocks, section, BlockKind.LIST_ITEM))
            index = cursor
            continue
        if (
            block.kind is BlockKind.FAQ_QUESTION
            and index + 1 < len(blocks)
            and blocks[index + 1].kind is BlockKind.FAQ_ANSWER
        ):
            units.append(
                _unit(document, [index, index + 1], blocks, section, BlockKind.FAQ_QUESTION)
            )
            index += 2
            continue
        units.append(_unit(document, [index], blocks, section, block.kind))
        index += 1
    return [unit for unit in units if unit.end > unit.start]


def _unit(
    document: Document, indices: list[int], blocks: list[Block], section: int, kind: BlockKind
) -> _Unit:
    start = blocks[indices[0]].start_char
    end = blocks[indices[-1]].end_char
    return _Unit(
        block_indices=list(indices),
        start=start,
        end=end,
        tokens=estimate_tokens(document.text[start:end]),
        section=section,
        kind=kind,
    )


def _intro(text: str) -> str:
    """First sentences of the chunk, used as context (never as new content)."""
    sentences = split_sentences(text) or [text]
    return " ".join(sentences[:2])[:400]


# --------------------------------------------------------------------------------------
# hard split
# --------------------------------------------------------------------------------------


def _overlap_tail(text: str, budget: int) -> str:
    """Longest *raw suffix* of ``text`` within ``budget``, cut at a sentence boundary.

    Used when the previous chunk's whole units are each larger than the overlap budget
    (a paragraph of 90 tokens with ``overlap_tokens=48``): capping overlap at unit
    granularity there would silently mean "no overlap at all". A sentence-grained tail
    keeps the chunk a verbatim slice of the document, which is what the citation
    verifier relies on (requirements CR-2).
    """
    if budget <= 0 or not text.strip():
        return ""
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        suffix = text[match.end() :]
        if not suffix.strip() or len(suffix) >= len(text):
            continue
        if estimate_tokens(suffix) <= budget:
            return suffix
    kept: list[str] = []
    for word in reversed(text.split()):
        candidate = [word, *kept]
        if estimate_tokens(" ".join(candidate)) > budget:
            break
        kept = candidate
    tail = " ".join(kept)
    return tail if tail and len(tail) < len(text) else ""


def _hard_split(text: str, budget: int) -> list[str]:
    """Cut ``text`` into pieces of at most ``budget`` estimated tokens.

    Boundaries in order of preference: sentence, then word. No piece is empty, and the
    concatenation of the pieces reproduces the text's word content, so every chunk
    stays a faithful quote source for the citation verifier.
    """
    if budget <= 0 or not text.strip():
        return []
    atoms = _atoms(text, budget)
    pieces: list[str] = []
    current: list[str] = []
    for atom in atoms:
        candidate = " ".join([*current, atom])
        if current and estimate_tokens(candidate) > budget:
            pieces.append(" ".join(current))
            current = [atom]
        else:
            current.append(atom)
    if current:
        pieces.append(" ".join(current))
    return [piece for piece in pieces if piece.strip()]


def _atoms(text: str, budget: int) -> list[str]:
    atoms: list[str] = []
    for sentence in split_sentences(text) or [text]:
        if estimate_tokens(sentence) <= budget:
            atoms.append(sentence)
            continue
        for word in _WORD_RE.findall(sentence):
            if estimate_tokens(word) <= budget:
                atoms.append(word)
            else:  # a single "word" longer than the budget (a hash, a long URL)
                atoms.extend(_char_pieces(word, budget))
    return [atom for atom in atoms if atom.strip()]


def _char_pieces(word: str, budget: int) -> list[str]:
    size = max(1, budget * 3)
    return [word[index : index + size] for index in range(0, len(word), size)]


__all__ = ["ChunkerConfig", "StructureChunker"]
