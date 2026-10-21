"""Read-only view of the ingested corpus that the harness labels against.

The evaluation layer never re-chunks anything: it consumes ``data/corpus`` exactly as the
ingestion pipeline produced it (``chunks.jsonl`` + ``manifest.json`` + the query pool in
``queries/queries.jsonl``). That keeps "the corpus the retriever sees" and "the corpus the
labels refer to" the same artifact, which is the only way a gold chunk id can be trusted
(FR-EVL-02, §10.2).

Tokenisation here is deliberately simple and language-agnostic: lower-case, Unicode word
characters, length >= 3 after normalisation. The corpus is Cyrillic, Latin and CJK; a
stemmer per locale would be another thing to validate, and the labels do not need it.

    # AT-EVL-02  # AT-DET-02
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from answerdesk.contracts import Chunk
from answerdesk.eval.errors import CorpusUnavailableError, EvalDataError

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
#: Identifier anchors: ``ORD-2026-10029``, ``ERR-KYC-3982``, long digit runs, dates.
_ANCHOR = re.compile(
    r"\b[A-Z][A-Z0-9]{1,}-\d{2,}(?:-\d{2,})?\b|\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"
    r"|\b\d{4,}\b|\b\d[\d\s.,]{2,}\d\b",
    re.UNICODE,
)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def words(text: str) -> list[str]:
    """Lower-cased word tokens (letters only) of length >= 3; CJK yields single chars."""
    lowered = text.lower()
    tokens = [token for token in _WORD.findall(lowered) if len(token) >= 3]
    tokens.extend(_CJK.findall(text))
    return tokens


def content_words(text: str) -> set[str]:
    return set(words(text))


def anchor_tokens(text: str) -> list[str]:
    """Exact identifiers/numbers, normalised so ``1 200`` and ``1200`` collide."""
    anchors: list[str] = []
    for match in _ANCHOR.findall(text):
        normalised = re.sub(r"[\s.,]", "", match).upper()
        if len(normalised) >= 3 and normalised not in anchors:
            anchors.append(normalised)
    return anchors


def stable_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CorpusQuery:
    """One item of the corpus query pool — a candidate golden case (provenance, no label)."""

    query_id: str
    locale: str
    doc_id: str
    intent: str
    category: str
    text: str
    base_text: str
    operator_reply: str
    perturbations: tuple[str, ...] = ()
    answerable: bool = True
    supporting_doc_ids: tuple[str, ...] = ()
    archetype: str = ""

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> CorpusQuery:
        return cls(
            query_id=str(payload["query_id"]),
            locale=str(payload["locale"]),
            doc_id=str(payload["doc_id"]),
            intent=str(payload.get("intent", "")),
            category=str(payload.get("category", "direct")),
            text=str(payload["text"]),
            base_text=str(payload.get("base_text") or payload["text"]),
            operator_reply=str(payload.get("operator_reply", "")),
            perturbations=tuple(payload.get("perturbations") or ()),
            answerable=bool(payload.get("answerable", True)),
            supporting_doc_ids=tuple(payload.get("supporting_doc_ids") or ()),
            archetype=str(payload.get("archetype", "")),
        )


class Corpus:
    """The ingested corpus, indexed for labelling and for the offline retrieval arms."""

    def __init__(
        self,
        *,
        root: Path,
        chunks: list[Chunk],
        queries: list[CorpusQuery],
        manifest: dict[str, Any],
        chunks_hash: str,
    ) -> None:
        self.root = root
        self.chunks = chunks
        self.queries = queries
        self.manifest = manifest
        self.chunks_hash = chunks_hash
        self.by_id: dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in chunks}
        self.by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            self.by_doc[chunk.doc_id].append(chunk)
        self.document_ids: list[str] = sorted(self.by_doc)
        self.locales: list[str] = sorted({chunk.language for chunk in chunks if chunk.language})
        self._anchor_index: dict[str, list[str]] | None = None
        self._chunk_tokens: dict[str, set[str]] | None = None

    # -- identity -----------------------------------------------------------------
    @property
    def version(self) -> str:
        return str(self.manifest.get("corpus_version", "unknown"))

    @property
    def hash(self) -> str:
        """Content hash of the chunk set — the value every report prints."""
        return self.chunks_hash

    # -- queries ------------------------------------------------------------------
    def query_pool(self, *, locale: str | None = None) -> list[CorpusQuery]:
        if locale is None:
            return list(self.queries)
        return [query for query in self.queries if query.locale == locale]

    def queries_for_doc(self, doc_id: str) -> list[CorpusQuery]:
        return [query for query in self.queries if query.doc_id == doc_id]

    # -- chunks -------------------------------------------------------------------
    def doc_chunks(self, doc_id: str) -> list[Chunk]:
        """Chunks of one document in reading order (``ordinal`` when present)."""
        chunks = list(self.by_doc.get(doc_id, []))
        chunks.sort(key=lambda chunk: (int(chunk.metadata.get("ordinal", 0)), chunk.chunk_id))
        return chunks

    def faq_chunks(self, doc_id: str) -> list[Chunk]:
        """Chunks holding the FAQ section — the structural evidence of a generated query.

        The synthetic generator builds every query from an intent FAQ pair, and the chunker
        keeps FAQ pairs atomic, so the FAQ chunk is the exact evidence passage for a
        direct question. This is a *structural* label, not a lexical guess.
        """
        return [
            chunk
            for chunk in self.doc_chunks(doc_id)
            if "faq_question" in (chunk.metadata.get("block_kinds") or [])
            or ("Q:" in chunk.text and "A:" in chunk.text)
        ]

    def anchor_index(self) -> dict[str, list[str]]:
        """``anchor -> chunk ids`` for exact-identifier matching (detects absent ids)."""
        if self._anchor_index is None:
            index: dict[str, list[str]] = defaultdict(list)
            for chunk in self.chunks:
                for anchor in anchor_tokens(chunk.text):
                    index[anchor].append(chunk.chunk_id)
            self._anchor_index = dict(index)
        return self._anchor_index

    def contains_anchor(self, anchor: str) -> bool:
        """Whether any chunk carries exactly this identifier/number."""
        needle = re.sub(r"[\s.,]", "", anchor).upper()
        return needle in self.anchor_index()

    def chunk_tokens(self) -> dict[str, set[str]]:
        """Content tokens per chunk: words plus canonical/identifier tokens."""
        if self._chunk_tokens is None:
            tokens: dict[str, set[str]] = {}
            for chunk in self.chunks:
                bag = content_words(chunk.text) | content_words(chunk.breadcrumb)
                bag |= {token.lower() for token in chunk.canonical_tokens}
                tokens[chunk.chunk_id] = bag
            self._chunk_tokens = tokens
        return self._chunk_tokens


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalDataError(f"cannot read {path}: {exc}") from exc
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise EvalDataError(f"{path}:{number} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise EvalDataError(f"{path}:{number} is not a JSON object")
        rows.append(payload)
    return rows


def load_corpus(corpus_dir: str | Path = "data/corpus") -> Corpus:
    """Load ``chunks.jsonl``, ``manifest.json`` and the query pool.

    A missing corpus is an actionable error, not an empty run: an evaluation over zero
    chunks would report perfect metrics on nothing.
    """
    root = Path(corpus_dir)
    chunks_path = root / "chunks.jsonl"
    manifest_path = root / "manifest.json"
    queries_path = root / "queries" / "queries.jsonl"
    missing = [path for path in (chunks_path, manifest_path) if not path.is_file()]
    if missing:
        raise CorpusUnavailableError(
            f"corpus artifacts are missing: {', '.join(str(path) for path in missing)}. "
            "Run `answerdesk ingest --demo` (or point --corpus at an ingested corpus) first.",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalDataError(f"cannot read {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise EvalDataError(f"{manifest_path} must contain a JSON object")

    chunks: list[Chunk] = []
    for row in _read_jsonl(chunks_path):
        try:
            chunks.append(Chunk.model_validate(row))
        except Exception as exc:  # pydantic ValidationError
            raise EvalDataError(f"chunk row does not match the contract: {exc}") from exc
    if not chunks:
        raise EvalDataError(f"{chunks_path} holds no chunks")
    duplicate_ids = len(chunks) - len({chunk.chunk_id for chunk in chunks})
    if duplicate_ids:
        raise EvalDataError(f"{chunks_path} has {duplicate_ids} duplicate chunk ids")

    queries: list[CorpusQuery] = []
    if queries_path.is_file():
        queries = [CorpusQuery.from_json(row) for row in _read_jsonl(queries_path)]

    return Corpus(
        root=root,
        chunks=chunks,
        queries=queries,
        manifest=manifest,
        chunks_hash=file_sha256(chunks_path),
    )


__all__ = [
    "Corpus",
    "CorpusQuery",
    "anchor_tokens",
    "content_words",
    "file_sha256",
    "load_corpus",
    "stable_hash",
    "words",
]
