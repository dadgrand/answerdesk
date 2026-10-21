"""Citations: from "the model said so" to a span you can click.

This module implements the citation contract of requirements §7.2 in code:

* **CR-2 / no phantom citations** — a citation's passage label must exist in the packed
  context; anything else is rejected, never resolved "somehow".
* **CR-3 / verbatim quote** — the quote is a whitespace-normalised substring of the
  chunk's stored text, case-sensitive. A model that paraphrases a quote has not cited.
* **CR-4 / span** — the character offsets are derived from that very substring, so the
  console can highlight exactly the words the claim rests on (FR-UI-03).
* **CR-5 / numbering** — citation ids are assigned by *first appearance* across the
  ordered claims, and the rendered markers are derived from the structure, never parsed
  back out of prose (FR-GEN-01, FR-GEN-03).
* **CR-6 / doc hash** — the chunk's own content hash is recorded, so a cache hit can be
  re-validated and a stale citation cannot be displayed (consumed by t8).

    # AT-GEN-01  # AT-GEN-02  # AT-GEN-03
"""

from __future__ import annotations

import re

from pydantic import Field

from answerdesk.contracts import Chunk, Citation, Claim, Contract, Support
from answerdesk.generate.draft import AnswerDraft
from answerdesk.generate.packer import PackedContext, PackedPassage

_MARKER = re.compile(r"\[(\d+)\]")


def normalise_whitespace(text: str) -> str:
    """Collapse every whitespace run to one space and trim (CR-3's normalisation)."""
    return " ".join(text.split())


def _normalised_index(text: str) -> tuple[str, list[int]]:
    """Whitespace-normalised text plus a map from normalised to original offsets."""
    chars: list[str] = []
    positions: list[int] = []
    pending_space = False
    for index, char in enumerate(text):
        if char.isspace():
            pending_space = True
            continue
        if pending_space and chars:
            chars.append(" ")
            positions.append(index)
        pending_space = False
        chars.append(char)
        positions.append(index)
    return "".join(chars), positions


def find_quote_span(chunk_text: str, quote: str) -> tuple[int, int] | None:
    """Offsets of ``quote`` inside ``chunk_text``, or ``None`` when it is not a substring."""
    needle = normalise_whitespace(quote)
    if not needle:
        return None
    haystack, positions = _normalised_index(chunk_text)
    found = haystack.find(needle)
    if found < 0:
        return None
    start = positions[found]
    end = positions[found + len(needle) - 1] + 1
    return start, end


class CitationIssue(Contract):
    """A rejected citation draft, kept for the repair prompt and the trace."""

    claim_id: str = ""
    claim_index: int = Field(default=0, ge=0)
    code: str
    passage_label: str = ""
    quote: str = ""
    detail: str = ""


class CitationBuild(Contract):
    """Claims and citations produced from one draft, plus the rejected parts."""

    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    issues: list[CitationIssue] = Field(default_factory=list)
    phantom_labels: list[str] = Field(default_factory=list)

    def citation_map(self) -> dict[int, Citation]:
        return {citation.citation_id: citation for citation in self.citations}

    def uncited_claim_indexes(self) -> list[int]:
        return [index for index, claim in enumerate(self.claims) if not claim.citation_ids]


class CitationBuilder:
    """Turns a validated :class:`AnswerDraft` into contract ``Claim``/``Citation`` objects."""

    def build(self, draft: AnswerDraft, packed: PackedContext) -> CitationBuild:
        claims: list[Claim] = []
        citations: list[Citation] = []
        issues: list[CitationIssue] = []
        phantom_labels: list[str] = []
        # (chunk_id, normalised quote) -> citation_id, so a repeated support link is one
        # citation (and its number stays stable).
        assigned: dict[tuple[str, str], int] = {}

        for claim_index, claim_draft in enumerate(draft.claims, start=1):
            claim_id = f"c{claim_index}"
            citation_ids: list[int] = []
            for citation_draft in claim_draft.citations:
                packed_passage = packed.by_label(citation_draft.passage)
                if packed_passage is None:
                    if citation_draft.passage not in phantom_labels:
                        phantom_labels.append(citation_draft.passage)
                    issues.append(
                        CitationIssue(
                            claim_id=claim_id,
                            claim_index=claim_index,
                            code="phantom_passage",
                            passage_label=citation_draft.passage,
                            quote=citation_draft.quote,
                            detail="passage label is not part of the packed context",
                        )
                    )
                    continue
                citation = self._citation_for(packed_passage, citation_draft.quote)
                if citation is None:
                    issues.append(
                        CitationIssue(
                            claim_id=claim_id,
                            claim_index=claim_index,
                            code="quote_not_found",
                            passage_label=citation_draft.passage,
                            quote=citation_draft.quote,
                            detail="quote is not a whitespace-normalised substring of the chunk",
                        )
                    )
                    continue
                key = (citation.chunk_id, citation.quote)
                existing = assigned.get(key)
                if existing is not None:
                    if existing not in citation_ids:
                        citation_ids.append(existing)
                    continue
                citation_id = len(citations) + 1
                assigned[key] = citation_id
                citations.append(citation.model_copy(update={"citation_id": citation_id}))
                citation_ids.append(citation_id)

            claims.append(
                Claim(
                    claim_id=claim_id,
                    text=claim_draft.text.strip(),
                    citation_ids=citation_ids,
                    support=Support.SUPPORTED if citation_ids else Support.UNSUPPORTED,
                )
            )
        return CitationBuild(
            claims=claims,
            citations=citations,
            issues=issues,
            phantom_labels=phantom_labels,
        )

    @staticmethod
    def _citation_for(packed_passage: PackedPassage, quote: str) -> Citation | None:
        chunk = packed_passage.chunk
        span = find_quote_span(chunk.text, quote)
        if span is None:
            return None
        start, end = span
        return Citation(
            citation_id=1,  # renumbered by the caller in first-appearance order
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            doc_title=packed_passage.doc_title,
            heading_path=list(chunk.heading_path),
            quote=normalise_whitespace(quote),
            start_char=start,
            end_char=end,
            doc_hash=chunk.doc_hash,
        )


def cited_chunks(
    claim: Claim, citations: dict[int, Citation], chunks: dict[str, Chunk]
) -> list[Chunk]:
    """The chunks a claim actually rests on (used by the verifier and the numeric guard)."""
    result: list[Chunk] = []
    for citation_id in claim.citation_ids:
        citation = citations.get(citation_id)
        if citation is None:
            continue
        chunk = chunks.get(citation.chunk_id)
        if chunk is not None:
            result.append(chunk)
    return result


def render_answer_text(claims: list[Claim], *, notice: str | None = None) -> str:
    """Render claim text with markers derived from structure (CR-5, FR-GEN-01).

    Markers are appended per claim, in ``claim.citation_ids`` order, and never parsed out
    of model prose — the model is asked for JSON precisely so this step is total.
    """
    parts: list[str] = []
    for claim in claims:
        text = normalise_whitespace(claim.text)
        if claim.citation_ids:
            markers = "".join(f"[{citation_id}]" for citation_id in claim.citation_ids)
            parts.append(f"{text} {markers}".strip())
        else:
            parts.append(text)
    body = " ".join(part for part in parts if part)
    if notice:
        body = f"{body}\n\n{notice}" if body else notice
    return body


def markers_in(text: str) -> list[int]:
    """Citation markers in rendered text, in order of appearance."""
    return [int(match.group(1)) for match in _MARKER.finditer(text)]


def validate_render(text: str, claims: list[Claim], citations: list[Citation]) -> list[str]:
    """Structural CR-5 / CR-1 checks; returns a list of problems (empty means valid)."""
    problems: list[str] = []
    expected = [
        citation_id for claim in claims for citation_id in claim.citation_ids if claim.citation_ids
    ]
    observed = markers_in(text)
    if observed != expected:
        problems.append(f"markers {observed} do not match claim citation ids {expected}")
    known = {citation.citation_id for citation in citations}
    for citation_id in observed:
        if citation_id not in known:
            problems.append(f"marker [{citation_id}] has no citation object")
    for claim in claims:
        if not claim.citation_ids:
            problems.append(f"claim {claim.claim_id!r} is displayed without a citation")
    ordered_unique: list[int] = []
    for citation_id in expected:
        if citation_id not in ordered_unique:
            ordered_unique.append(citation_id)
    if ordered_unique != sorted(ordered_unique):
        problems.append(f"citation numbering is not by first appearance: {ordered_unique}")
    return problems


__all__ = [
    "CitationBuild",
    "CitationBuilder",
    "CitationIssue",
    "cited_chunks",
    "find_quote_span",
    "markers_in",
    "normalise_whitespace",
    "render_answer_text",
    "validate_render",
]
