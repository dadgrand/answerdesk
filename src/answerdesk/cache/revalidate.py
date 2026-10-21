"""G-4: cheap structural re-validation of a cache hit (requirements §9.2).

A hit is never served straight from storage. Before display, the cached *artifact* is
re-checked with the same deterministic rules the generation layer applies:

* every citation resolves to a chunk that still exists (source-doc guard input);
* every quote is still a substring of that chunk's stored text (CR-3);
* the numeric guard passes again — every number in a displayed claim is present, after
  canonicalisation, in a chunk the claim cites (INV-04 / CR-9);
* the rendered markers still match the claim structure (CR-5).

No model is involved, no network, no tokeniser: string and set operations only, which is why
the requirements can demand it "never be skipped for latency" (§9.2).

    # AT-CAC-03
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from pydantic import Field

from answerdesk.contracts import Answer, Chunk, Contract
from answerdesk.generate.citations import find_quote_span, markers_in, validate_render
from answerdesk.generate.numeric import numeric_violations

#: Resolver used by the cache: chunk id → current chunk (``None`` when it no longer exists).
ChunkResolver = Callable[[str], Chunk | None]


class RevalidationResult(Contract):
    """Outcome of the structural re-check, with the reason a hit was demoted."""

    ok: bool
    problems: list[str] = Field(default_factory=list)

    def reason(self) -> str:
        return "; ".join(self.problems)


def revalidate_answer(
    answer: Answer,
    *,
    chunk_lookup: ChunkResolver,
    corpus_version: str | None = None,
    prompt_version: str | None = None,
    model_id: str | None = None,
) -> RevalidationResult:
    """Re-check a cached :class:`Answer`; ``ok=False`` means "treat as a miss"."""
    problems: list[str] = []

    if corpus_version is not None and answer.provenance.corpus_version != corpus_version:
        problems.append(
            f"corpus version {answer.provenance.corpus_version!r} != live {corpus_version!r}"
        )
    if prompt_version is not None and answer.provenance.prompt_version != prompt_version:
        problems.append(
            f"prompt version {answer.provenance.prompt_version!r} != live {prompt_version!r}"
        )
    if model_id is not None and answer.usage.model != model_id:
        problems.append(f"model {answer.usage.model!r} != live {model_id!r}")

    chunks: dict[str, Chunk] = {}
    for citation in answer.citations:
        chunk = chunk_lookup(citation.chunk_id)
        if chunk is None:
            problems.append(f"citation {citation.citation_id} resolves to a missing chunk")
            continue
        chunks[citation.chunk_id] = chunk
        if chunk.doc_hash != citation.doc_hash:
            problems.append(f"citation {citation.citation_id} doc_hash changed (stale citation)")
        if find_quote_span(chunk.text, citation.quote) is None:
            problems.append(f"citation {citation.citation_id} quote is no longer a substring")

    if not problems:
        for claim in answer.claims:
            cited = [
                chunks[citation.chunk_id]
                for citation in answer.citations
                if citation.citation_id in claim.citation_ids and citation.chunk_id in chunks
            ]
            missing = numeric_violations(claim.text, [chunk.text for chunk in cited])
            if missing:
                problems.append(
                    f"claim {claim.claim_id!r} quotes numbers absent from its evidence: {missing}"
                )
        problems.extend(validate_render(answer.text, list(answer.claims), list(answer.citations)))
        known_markers = set(markers_in(answer.text))
        known_citations = {citation.citation_id for citation in answer.citations}
        if not known_markers <= known_citations:
            problems.append("rendered markers reference citations that do not exist")

    return RevalidationResult(ok=not problems, problems=problems)


def chunk_map(chunks: Mapping[str, Chunk]) -> ChunkResolver:
    """Adapter: a static mapping as a resolver (used by tests and the replay harness)."""

    def _lookup(chunk_id: str) -> Chunk | None:
        return chunks.get(chunk_id)

    return _lookup


__all__ = ["ChunkResolver", "RevalidationResult", "chunk_map", "revalidate_answer"]
