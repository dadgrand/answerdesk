"""Loaders for PDF, DOCX, HTML, Markdown and plain text (FR-ING-01).

Each loader recovers *structure*, not just text: headings (with level), paragraphs,
list items, tables, code, quotes and FAQ question/answer pairs. The recovered blocks
are stored in ``Document.metadata["blocks"]`` (see :mod:`answerdesk.ingest.structure`)
and :class:`answerdesk.contracts.Document` stays the frozen, format-neutral record.

Design rules
------------
* **Lazy imports.** ``pypdf``, ``python-docx`` and ``bs4`` are imported inside their
  loader. Importing :mod:`answerdesk.ingest` (or the CLI) never pays for them, which is
  what NFR-PORT-03 asks for.
* **Language-agnostic cleaning.** Normalisation touches only encoding artefacts and
  whitespace: CRLF/CR -> LF, NUL removal, trailing-space stripping, runs of 3+ blank
  lines -> 2. Nothing is transliterated, stemmed or lower-cased, so identifiers and
  dates survive byte-for-byte (FR-ING-05, INV-04).
* **Explicit failure.** An empty file, a scanned PDF with no text layer, an encrypted
  PDF or an unknown suffix raises a typed :mod:`answerdesk.ingest.errors` exception —
  never a silent empty document.
* **No network, no model.** Every parser is local and deterministic.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path

from answerdesk.contracts import Document
from answerdesk.ingest.errors import EmptyDocumentError, IngestError, UnsupportedFormatError
from answerdesk.ingest.structure import (
    Block,
    BlockBuilder,
    BlockKind,
    blocks_as_metadata,
    headings_from_blocks,
    render_table,
)

#: Suffix -> ``Document.source_format`` (the five formats of FR-ING-01).
SUPPORTED_FORMATS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
}

#: Default license when neither the caller nor the file declares one. Deliberately not
#: an open license: an undeclared file is *unknown*, and the manifest must say so.
DEFAULT_LICENSE = "unknown"

#: FAQ markers per locale (lower-case, matched at the start of a line/paragraph).
QUESTION_MARKERS: tuple[str, ...] = (
    "q:",
    "вопрос:",
    "в:",
    "savol:",
    "суроо:",
    "soru:",
    "问题:",
    "سوال:",
)
ANSWER_MARKERS: tuple[str, ...] = (
    "a:",
    "ответ:",
    "о:",
    "javob:",
    "жооп:",
    "cevap:",
    "答案:",
    "回答:",
    "جواب:",
)

_HEADING_MD_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*#*\s*$")
_LIST_MD_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>[-*+]|\d{1,3}[.)])\s+(?P<text>.+)$")
_SETEXT_H1_RE = re.compile(r"^=+\s*$")
_SETEXT_H2_RE = re.compile(r"^-{2,}\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*(?:/|of|из)\s*\d{1,4}\s*$", re.IGNORECASE)
#: A boundary is terminal punctuation (+ optional quote/bracket) followed by space, or
#: a CJK terminator (which needs no space). ``12.5`` never splits because the dot is
#: not followed by whitespace — money must survive the splitter (FR-ING-05).
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?…]+[\"'»”’)\]]*\s+|[。！？；]")


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------


def load_document(
    path: str | Path,
    *,
    license: str | None = None,
    language: str | None = None,
    doc_id: str | None = None,
    title: str | None = None,
) -> Document:
    """Load ``path`` into a :class:`~answerdesk.contracts.Document` (FR-ING-01).

    ``license``/``language``/``doc_id``/``title`` override what the file declares; a
    Markdown/HTML front matter block can declare the same fields.
    """
    source = Path(path)
    if not source.is_file():
        raise UnsupportedFormatError(f"no such file: {source}")
    fmt = SUPPORTED_FORMATS.get(source.suffix.lower())
    if fmt is None:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise UnsupportedFormatError(
            f"unsupported source format {source.suffix!r} for {source.name}; supported: {supported}"
        )
    data = source.read_bytes()
    source_path = source.as_posix()
    loader = _LOADERS[fmt]
    return loader(
        data,
        source_path=source_path,
        license=license,
        language=language,
        doc_id=doc_id,
        title=title,
    )


def load_markdown_text(
    text: str,
    *,
    source_path: str,
    doc_id: str | None,
    license: str | None = None,
    language: str | None = None,
    title: str | None = None,
) -> Document:
    """Parse Markdown that is already in memory (the synthetic corpus builder)."""
    return _load_markdown(
        text.encode("utf-8"),
        source_path=source_path,
        license=license,
        language=language,
        doc_id=doc_id,
        title=title,
    )


def load_directory(
    directory: str | Path,
    *,
    license: str | None = None,
    recursive: bool = True,
) -> list[Document]:
    """Load every supported file under ``directory`` in a deterministic order."""
    root = Path(directory)
    if not root.is_dir():
        raise UnsupportedFormatError(f"no such directory: {root}")
    pattern = "**/*" if recursive else "*"
    paths = sorted(
        (p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS),
        key=lambda p: p.as_posix(),
    )
    return [load_document(path, license=license) for path in paths]


def decode_bytes(data: bytes) -> str:
    """Decode a text file the way a human would: BOM first, then UTF-8, then CP1251.

    CP1251 is in the chain because a Russian-language support export still arrives as
    a legacy-encoded text file often enough that failing on it would be a bug report.
    """
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for encoding in ("utf-8", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def normalise_text(text: str) -> str:
    """Encoding- and whitespace-level cleaning only (never linguistic normalisation)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "").replace("\ufeff", "")
    text = text.replace("\u00a0", " ").replace("\t", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    joined = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", joined).strip("\n")


def guess_language(text: str) -> str | None:
    """Cheap script/marker heuristic; ``None`` when the script is not one of the six.

    It is a *hint* recorded with the chunk, never a gate: the caller's explicit
    ``language`` always wins (NFR-ML-02 — no translation, answer in the source
    language).
    """
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    lowered = text.lower()
    if re.search(r"[ҳҷқӣ]|(?:аст|мебошад)", lowered):
        return "tg"
    if re.search(r"[өң]|(?:салым|керек)", lowered):
        return "ky"
    if re.search(r"o['ʻ’]|g['ʻ’]", lowered):
        return "uz"
    if re.search(r"[ışğ]|(?:ve|için|ile)\b", lowered):
        return "tr"
    if re.search(r"[\u0400-\u04ff]", text):
        return "ru"
    return None


def split_sentences(text: str) -> list[str]:
    """Sentence split that is script-aware enough for the six demo locales.

    CJK text is split on ``。！？`` and on line breaks; other scripts on terminal
    punctuation followed by whitespace. Over-long "sentences" (a table row, an
    unpunctuated paragraph) come back as one piece and the chunker hard-splits those
    at word boundaries.
    """
    if not text.strip():
        return []
    pieces: list[str] = []
    for line in text.split("\n"):
        start = 0
        for match in _SENTENCE_BOUNDARY_RE.finditer(line):
            pieces.append(line[start : match.end()].strip())
            start = match.end()
        tail = line[start:].strip()
        if tail:
            pieces.append(tail)
    return [piece for piece in pieces if piece]


# --------------------------------------------------------------------------------------
# Markdown / plain text
# --------------------------------------------------------------------------------------


def _load_markdown(
    data: bytes,
    *,
    source_path: str,
    license: str | None,
    language: str | None,
    doc_id: str | None,
    title: str | None,
) -> Document:
    text = normalise_text(decode_bytes(data))
    front_matter, body = _split_front_matter(text)
    builder = _build_markdown_blocks(body)
    return _finalise(
        builder,
        source_path=source_path,
        source_format="md",
        raw_text=body,
        front_matter=front_matter,
        doc_id=doc_id,
        title=title,
        license=license,
        language=language,
    )


def _load_txt(
    data: bytes,
    *,
    source_path: str,
    license: str | None,
    language: str | None,
    doc_id: str | None,
    title: str | None,
) -> Document:
    text = normalise_text(decode_bytes(data))
    front_matter, body = _split_front_matter(text)
    builder = BlockBuilder()
    for paragraph in re.split(r"\n\s*\n", body):
        for line in paragraph.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            _add_text_block(builder, stripped)
    return _finalise(
        builder,
        source_path=source_path,
        source_format="txt",
        raw_text=body,
        front_matter=front_matter,
        doc_id=doc_id,
        title=title,
        license=license,
        language=language,
    )


def _build_markdown_blocks(body: str) -> BlockBuilder:
    """Recover Markdown structure: headings, tables, lists, quotes, fences, FAQ pairs."""
    builder = BlockBuilder()
    lines = body.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        if _FENCE_RE.match(line):
            fence = stripped[:3]
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith(fence):
                code_lines.append(lines[index])
                index += 1
            index += 1  # closing fence
            builder.add(BlockKind.CODE, "\n".join(code_lines))
            continue

        heading = _HEADING_MD_RE.match(stripped)
        if heading:
            builder.heading(len(heading["hashes"]), heading["text"])
            index += 1
            continue

        if index + 1 < len(lines) and stripped and not stripped.startswith("|"):
            underline = lines[index + 1].strip()
            if _SETEXT_H1_RE.match(underline):
                builder.heading(1, stripped)
                index += 2
                continue
            if _SETEXT_H2_RE.match(underline) and not _LIST_MD_RE.match(stripped):
                builder.heading(2, stripped)
                index += 2
                continue

        if stripped.startswith("|") and stripped.count("|") >= 2:
            table_lines: list[str] = []
            while index < len(lines):
                candidate = lines[index].strip()
                if not candidate.startswith("|"):
                    break
                table_lines.append(candidate)
                index += 1
            _add_pipe_table(builder, table_lines)
            continue

        quote_lines: list[str] = []
        while index < len(lines) and lines[index].strip().startswith(">"):
            quote_lines.append(lines[index].strip().lstrip(">").strip())
            index += 1
        if quote_lines:
            builder.add(BlockKind.QUOTE, " ".join(part for part in quote_lines if part))
            continue

        list_match = _LIST_MD_RE.match(line)
        if list_match:
            item_lines = [list_match["text"].strip()]
            marker = list_match["marker"]
            index += 1
            while index < len(lines):
                candidate = lines[index]
                if not candidate.strip():
                    break
                if _LIST_MD_RE.match(candidate) or _HEADING_MD_RE.match(candidate.strip()):
                    break
                if candidate.startswith((" ", "\t")) or not candidate.strip():
                    item_lines.append(candidate.strip())
                    index += 1
                    continue
                break
            builder.add(
                BlockKind.LIST_ITEM,
                f"{marker} {' '.join(part for part in item_lines if part)}",
                meta={"marker": marker},
            )
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                break
            if (
                _HEADING_MD_RE.match(candidate)
                or _LIST_MD_RE.match(lines[index])
                or candidate.startswith(("|", ">", "```", "~~~"))
                or _FENCE_RE.match(candidate)
                or is_faq_marker(candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        _add_text_block(builder, " ".join(paragraph_lines))
    return builder


def _add_pipe_table(builder: BlockBuilder, table_lines: Sequence[str]) -> None:
    rows = [_split_pipe_row(line) for line in table_lines]
    if len(rows) >= 2 and _TABLE_SEPARATOR_RE.match(table_lines[1].strip()):
        header, body_rows = rows[0], rows[2:]
    else:
        header, body_rows = rows[0], rows[1:]
    builder.add(
        BlockKind.TABLE,
        render_table(header, body_rows),
        meta={"column_count": len(header), "row_count": len(body_rows)},
    )


def _split_pipe_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def is_faq_marker(text: str) -> bool:
    """True when a line/paragraph starts with a locale FAQ marker (``Q:``/``A:``/…)."""
    lowered = text.lower()
    return any(lowered.startswith(marker) for marker in (*QUESTION_MARKERS, *ANSWER_MARKERS))


def _add_text_block(builder: BlockBuilder, text: str) -> None:
    """Classify a text block as FAQ question/answer pair, or a plain paragraph."""
    lowered = text.lower()
    if any(lowered.startswith(marker) for marker in QUESTION_MARKERS):
        builder.add(BlockKind.FAQ_QUESTION, text)
    elif any(lowered.startswith(marker) for marker in ANSWER_MARKERS):
        builder.add(BlockKind.FAQ_ANSWER, text)
    else:
        builder.add(BlockKind.PARAGRAPH, text)


def _split_front_matter(text: str) -> tuple[dict[str, object], str]:
    """Split a leading ``---`` YAML block; returns ``(mapping, body)``.

    Front matter is how a locale corpus file declares ``title``, ``language``,
    ``license``, ``doc_id`` and ``intent`` — the provenance the manifest records
    (FR-ING-08).
    """
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    header = text[4:end]
    body = text[end + 4 :].lstrip("\n")
    try:
        import yaml

        parsed = yaml.safe_load(header)
    except Exception:
        return _front_matter_fallback(header), body
    if not isinstance(parsed, dict):
        return _front_matter_fallback(header), body
    cleaned: dict[str, object] = {}
    for key, value in parsed.items():
        if isinstance(value, str | int | float | bool) and not isinstance(value, bool):
            cleaned[str(key)] = value
        else:
            cleaned[str(key)] = str(value)
    return cleaned, body


def _front_matter_fallback(header: str) -> dict[str, object]:
    """Line-based ``key: value`` parse for a header YAML cannot read.

    A title with an unquoted colon is the common case: refusing the whole document
    because of it would drop a perfectly good article, and silently dropping the
    provenance would put an unlicensed, unlabelled document into the corpus.
    """
    parsed: dict[str, object] = {}
    for line in header.splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip("\"'")
        if not key:
            continue
        if value.isdigit():
            parsed[key] = int(value)
        elif value:
            parsed[key] = value
    return parsed


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------


def _load_html(
    data: bytes,
    *,
    source_path: str,
    license: str | None,
    language: str | None,
    doc_id: str | None,
    title: str | None,
) -> Document:
    from bs4 import BeautifulSoup

    text = decode_bytes(data)
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "svg", "head"]):
        tag.decompose()
    builder = BlockBuilder()
    root = soup.body or soup
    _walk_html(root, builder)
    page_title = title
    if page_title is None and soup.title is not None and soup.title.string:
        page_title = soup.title.string.strip()
    language = language or _html_lang(soup)
    return _finalise(
        builder,
        source_path=source_path,
        source_format="html",
        raw_text=root.get_text(" ", strip=True),
        front_matter={},
        doc_id=doc_id,
        title=page_title,
        license=license,
        language=language,
    )


def _html_lang(soup: object) -> str | None:
    html_tag = getattr(soup, "html", None)
    if html_tag is None:
        return None
    value = html_tag.get("lang")
    if isinstance(value, str) and value.strip():
        return value.strip().lower().split("-")[0]
    return None


_HTML_BLOCK_TAGS = {"div", "section", "article", "main", "body", "header", "footer", "aside"}


def _walk_html(node: object, builder: BlockBuilder) -> None:
    from bs4 import NavigableString, Tag

    if isinstance(node, NavigableString):
        text = str(node).strip()
        if text:
            _add_text_block(builder, re.sub(r"\s+", " ", text))
        return
    if not isinstance(node, Tag):
        return
    name = node.name.lower()
    if re.fullmatch(r"h[1-6]", name):
        builder.heading(int(name[1]), _tag_text(node))
        return
    if name == "table":
        _add_html_table(node, builder)
        return
    if name == "pre":
        builder.add(BlockKind.CODE, node.get_text("\n").strip())
        return
    if name == "blockquote":
        pieces = [_tag_text(child) for child in node.find_all(["p", "div"], recursive=False)]
        quoted = " ".join(piece for piece in pieces if piece) or _tag_text(node)
        builder.add(BlockKind.QUOTE, quoted)
        return
    if name == "dl":
        for child in node.children:
            if not isinstance(child, Tag):
                continue
            if child.name == "dt":
                builder.add(BlockKind.FAQ_QUESTION, _tag_text(child))
            elif child.name == "dd":
                builder.add(BlockKind.FAQ_ANSWER, _tag_text(child))
        return
    if name in ("ul", "ol"):
        for position, item in enumerate(node.find_all("li", recursive=False), start=1):
            text = _tag_text(item)
            if not text:
                continue
            marker = f"{position}." if name == "ol" else "-"
            builder.add(BlockKind.LIST_ITEM, f"{marker} {text}", meta={"marker": marker})
        return
    if name == "p":
        _add_text_block(builder, _tag_text(node))
        return
    if name == "li":
        builder.add(BlockKind.LIST_ITEM, f"- {_tag_text(node)}", meta={"marker": "-"})
        return
    if name in ("td", "th"):
        _add_text_block(builder, _tag_text(node))
        return
    if name in _HTML_BLOCK_TAGS or name in ("html",):
        for child in node.children:
            _walk_html(child, builder)
        return
    text = _tag_text(node)
    if text:
        _add_text_block(builder, text)


def _tag_text(node: object) -> str:
    text = getattr(node, "get_text", None)
    raw = text(" ", strip=True) if callable(text) else str(node)
    return re.sub(r"\s+", " ", raw).strip()


def _add_html_table(node: object, builder: BlockBuilder) -> None:
    rows: list[list[str]] = []
    get_rows = getattr(node, "find_all", None)
    if not callable(get_rows):
        return
    for tr in get_rows("tr"):
        cells = tr.find_all(["th", "td"])
        if cells:
            rows.append([_tag_text(cell) for cell in cells])
    if not rows:
        return
    header = rows[0]
    body = rows[1:]
    builder.add(
        BlockKind.TABLE,
        render_table(header, body),
        meta={"column_count": len(header), "row_count": len(body)},
    )


# --------------------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------------------


def _load_docx(
    data: bytes,
    *,
    source_path: str,
    license: str | None,
    language: str | None,
    doc_id: str | None,
    title: str | None,
) -> Document:
    import io

    from docx import Document as DocxDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = DocxDocument(io.BytesIO(data))
    except Exception as exc:
        raise IngestError(f"cannot read DOCX {source_path}: {exc}") from exc
    builder = BlockBuilder()
    body = document.element.body
    list_counter = 0
    for child in body.iterchildren():
        tag = str(child.tag).rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = re.sub(r"\s+", " ", paragraph.text).strip()
            if not text:
                continue
            kind, level = _docx_kind(paragraph.style.name if paragraph.style else "")
            if kind is BlockKind.HEADING:
                list_counter = 0
                builder.heading(level or 1, text)
            elif kind is BlockKind.LIST_ITEM:
                # Word stores numbering in the list definition, not in the run text:
                # re-create it so a procedure keeps its steps when it is quoted.
                style_name = paragraph.style.name if paragraph.style else ""
                ordered = "Number" in style_name or "Номер" in style_name
                list_counter = list_counter + 1 if ordered else 0
                marker = f"{list_counter}." if ordered else "-"
                builder.add(BlockKind.LIST_ITEM, f"{marker} {text}", meta={"marker": style_name})
            else:
                list_counter = 0
                _add_text_block(builder, text)
        elif tag == "tbl":
            table = Table(child, document)
            rows = [
                [re.sub(r"\s+", " ", cell.text).strip() for cell in row.cells] for row in table.rows
            ]
            if not rows:
                continue
            builder.add(
                BlockKind.TABLE,
                render_table(rows[0], rows[1:]),
                meta={"column_count": len(rows[0]), "row_count": len(rows) - 1},
            )
    core_title = title
    if core_title is None:
        try:
            core_title = (document.core_properties.title or "").strip() or None
        except Exception:
            core_title = None
    return _finalise(
        builder,
        source_path=source_path,
        source_format="docx",
        raw_text="\n".join(p.text for p in document.paragraphs),
        front_matter={},
        doc_id=doc_id,
        title=core_title,
        license=license,
        language=language,
    )


def _docx_kind(style_name: str) -> tuple[BlockKind, int | None]:
    name = style_name or ""
    heading = re.match(r"^Heading\s*(\d)$", name)
    if heading:
        return BlockKind.HEADING, int(heading[1])
    if name in ("Title", "Заголовок"):
        return BlockKind.HEADING, 1
    if name in ("Subtitle", "Подзаголовок"):
        return BlockKind.HEADING, 2
    if name.startswith(("List", "Список")):
        return BlockKind.LIST_ITEM, None
    return BlockKind.PARAGRAPH, None


# --------------------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------------------

#: A PDF line whose font is this much larger than the body font is a heading.
_PDF_HEADING_SCALE = 1.15
_PDF_NUMBERED_HEADING_RE = re.compile(r"^(?P<number>\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+\S")


def _load_pdf(
    data: bytes,
    *,
    source_path: str,
    license: str | None,
    language: str | None,
    doc_id: str | None,
    title: str | None,
) -> Document:
    import io

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise IngestError(f"cannot read encrypted PDF {source_path}")
        pages = list(reader.pages)
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"cannot read PDF {source_path}: {exc}") from exc
    if not pages:
        raise EmptyDocumentError(f"PDF {source_path} has no pages")
    lines: list[_PdfLine] = []
    for page in pages:
        lines.extend(_pdf_page_lines(page))
    lines = _drop_pdf_page_furniture(lines, page_count=len(pages))
    builder = BlockBuilder()
    _pdf_lines_to_blocks(lines, builder)
    title = title or _pdf_outline_title(reader) or (lines[0].text if lines else None)
    return _finalise(
        builder,
        source_path=source_path,
        source_format="pdf",
        raw_text="\n".join(line.text for line in lines),
        front_matter={},
        doc_id=doc_id,
        title=title,
        license=license,
        language=language,
    )


class _PdfLine:
    """One extracted PDF line with the largest font size seen on it."""

    __slots__ = ("size", "text")

    def __init__(self, text: str, size: float) -> None:
        self.text = text
        self.size = size

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_PdfLine({self.text!r}, {self.size})"


def _pdf_page_lines(page: object) -> list[_PdfLine]:
    """Extract lines and their font sizes from one page (pypdf visitor API).

    ``extract_text`` alone loses the only structural signal a PDF reliably has: type
    size. The visitor receives every text fragment with its font size, so headings can
    be recovered by size ranking instead of guessed from wording.
    """
    fragments: list[tuple[str, float]] = []

    def visitor(
        text: str, cm: object, tm: object, font: object, font_size: float | None
    ) -> None:  # pragma: no cover - signature required by pypdf
        if text:
            fragments.append((text, float(font_size or 0.0)))

    try:
        text = page.extract_text(visitor_text=visitor)  # type: ignore[attr-defined]
    except TypeError:  # pragma: no cover - older pypdf without visitor support
        text = page.extract_text()  # type: ignore[attr-defined]
        fragments = [(text, 0.0)]
    if not fragments:
        return _lines_from_plain_text(text or "")
    lines: list[_PdfLine] = []
    buffer: list[str] = []
    size = 0.0
    for fragment, fragment_size in fragments:
        parts = fragment.split("\n")
        for position, part in enumerate(parts):
            if position > 0:
                if buffer:
                    lines.append(_PdfLine("".join(buffer).strip(), size))
                buffer, size = [], 0.0
            if part.strip():
                buffer.append(part)
                size = max(size, fragment_size)
    if buffer:
        lines.append(_PdfLine("".join(buffer).strip(), size))
    return [line for line in lines if line.text]


def _lines_from_plain_text(text: str) -> list[_PdfLine]:
    return [_PdfLine(line.strip(), 0.0) for line in text.split("\n") if line.strip()]


def _drop_pdf_page_furniture(lines: Sequence[_PdfLine], *, page_count: int) -> list[_PdfLine]:
    """Remove obvious page numbers/footers; never removes body text."""
    kept: list[_PdfLine] = []
    for line in lines:
        stripped = line.text.strip()
        if len(stripped) < 2:
            continue
        if _PAGE_NUMBER_RE.match(stripped) and page_count >= 2:
            continue
        kept.append(line)
    return kept


def _pdf_lines_to_blocks(lines: Sequence[_PdfLine], builder: BlockBuilder) -> None:
    body_size = _body_font_size(lines)
    paragraph: list[str] = []
    for line in lines:
        text = line.text.strip()
        level = _pdf_heading_level(text, line.size, body_size)
        if level is not None:
            _flush_paragraph(builder, paragraph)
            builder.heading(level, text)
            continue
        paragraph.append(text)
    _flush_paragraph(builder, paragraph)


def _flush_paragraph(builder: BlockBuilder, paragraph: list[str]) -> None:
    if not paragraph:
        return
    # A PDF line break is a layout break, not a sentence break: join with spaces.
    _add_text_block(builder, " ".join(paragraph))
    paragraph.clear()


def _body_font_size(lines: Sequence[_PdfLine]) -> float:
    sizes = sorted(line.size for line in lines if line.size > 0)
    if not sizes:
        return 0.0
    return sizes[len(sizes) // 2]


def _pdf_heading_level(text: str, size: float, body_size: float) -> int | None:
    numbered = _PDF_NUMBERED_HEADING_RE.match(text)
    if numbered and len(text) <= 90:
        return min(6, numbered["number"].count(".") + 1)
    if body_size > 0 and size >= body_size * _PDF_HEADING_SCALE and len(text) <= 90:
        return 2 if size < body_size * 1.6 else 1
    letters = [char for char in text if char.isalpha()]
    if len(text) <= 60 and len(letters) >= 3 and all(char.isupper() for char in letters):
        return 2
    return None


def _pdf_outline_title(reader: object) -> str | None:
    """First bookmark title, when the PDF has an outline."""
    try:
        outline = reader.outline  # type: ignore[attr-defined]
    except Exception:
        return None
    for item in outline or []:
        title = getattr(item, "title", None)
        if isinstance(title, str) and title.strip():
            return title.strip()
    return None


# --------------------------------------------------------------------------------------
# finalisation
# --------------------------------------------------------------------------------------

_LOADERS = {
    "md": _load_markdown,
    "txt": _load_txt,
    "html": _load_html,
    "docx": _load_docx,
    "pdf": _load_pdf,
}


def _finalise(
    builder: BlockBuilder,
    *,
    source_path: str,
    source_format: str,
    raw_text: str,
    front_matter: dict[str, object],
    doc_id: str | None,
    title: str | None,
    license: str | None,
    language: str | None,
) -> Document:
    text, blocks = builder.result()
    if not text.strip() or not blocks:
        raise EmptyDocumentError(
            f"{source_path} produced no extractable text ({source_format})",
            detail="a scanned or image-only document needs OCR, which v1.0 does not do",
        )
    resolved_title = (
        title
        or _front_matter_str(front_matter, "title")
        or _first_heading(blocks)
        or Path(source_path).stem
    )
    resolved_license = license or _front_matter_str(front_matter, "license") or DEFAULT_LICENSE
    resolved_language = language or _front_matter_str(front_matter, "language")
    if resolved_language is None:
        resolved_language = guess_language(text)
    resolved_doc_id = (
        doc_id or _front_matter_str(front_matter, "doc_id") or _derive_doc_id(source_path)
    )
    return Document(
        doc_id=resolved_doc_id,
        title=resolved_title,
        source_path=source_path,
        source_format=source_format,  # type: ignore[arg-type]
        text=text,
        language=resolved_language,
        license=resolved_license,
        sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        byte_size=len(raw_text.encode("utf-8")),
        headings=headings_from_blocks(blocks),
        metadata={
            "blocks": blocks_as_metadata(blocks),
            "front_matter": front_matter,
            "loader": f"{source_format}-loader-v1",
            "block_count": len(blocks),
        },
    )


def _front_matter_str(front_matter: dict[str, object], key: str) -> str | None:
    value = front_matter.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int | float):
        return str(value)
    return None


def _first_heading(blocks: Iterable[Block]) -> str | None:
    for block in blocks:
        if block.is_heading:
            return block.text
    return None


def _derive_doc_id(source_path: str) -> str:
    """Stable ``doc_id`` from the path: readable slug + path hash (no collisions)."""
    stem = Path(source_path).stem
    slug = re.sub(r"[^0-9a-z]+", "-", stem.lower()).strip("-") or "doc"
    digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


__all__ = [
    "ANSWER_MARKERS",
    "DEFAULT_LICENSE",
    "QUESTION_MARKERS",
    "SUPPORTED_FORMATS",
    "decode_bytes",
    "guess_language",
    "load_directory",
    "load_document",
    "load_markdown_text",
    "normalise_text",
    "split_sentences",
]
