"""Claim verification: is this claim actually supported by the passage it cites?

The verifier is the gate between "the model wrote a sentence" and "the sentence is
displayed" (requirements §7.3, FR-GEN-07, CR-7). Three independent checks run in a
fixed order, and the *deterministic* ones decide first:

1. **Citation integrity** — every citation of the claim must still resolve to a packed
   chunk and its quote must still be a substring (round 2 of CR-2/CR-3, after citation
   building).
2. **Numeric guard** (CR-9 / INV-04) — every number or date in the claim must appear,
   after canonicalisation, in a chunk the claim cites. This check never asks a model.
3. **Lexical support** — content-token coverage of the claim against its cited chunks.
   A claim that reuses little of its evidence is not supported by it, however confident
   it sounds.

Verdicts: ``supported`` / ``partial`` / ``unsupported``. The pipeline repairs exactly
once, then drops what is still unsupported — an unsupported claim is never displayed
(INV-03).

The default lexical verifier is deliberately boring and deterministic so that offline CI
can prove the *state machine*. A judge- or NLI-based verifier is a drop-in replacement
behind :class:`answerdesk.contracts.ClaimVerifier`; the released F-9 numbers (precision,
recall, false-rejection) are measured on a labelled claim set by t11.

    # AT-GEN-07  # AT-GEN-12
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import Field

from answerdesk.contracts import Chunk, Citation, Claim, ClaimVerification, Contract, Support
from answerdesk.generate.citations import find_quote_span
from answerdesk.generate.numeric import numeric_violations

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

#: Support and partial-support thresholds on content-token coverage.
DEFAULT_SUPPORT_THRESHOLD = 0.60
DEFAULT_PARTIAL_THRESHOLD = 0.35

#: Minimal multilingual stopword list — only used to decide *what is content*, never a verdict.
STOPWORDS: frozenset[str] = frozenset(
    {
        # English
        "the",
        "and",
        "for",
        "are",
        "was",
        "were",
        "with",
        "that",
        "this",
        "from",
        "you",
        "your",
        "our",
        "can",
        "will",
        "not",
        "but",
        "has",
        "have",
        "how",
        "what",
        "when",
        "which",
        "there",
        "their",
        "its",
        "into",
        "than",
        "then",
        "they",
        "does",
        "did",
        # Russian
        "или",
        "для",
        "как",
        "что",
        "это",
        "все",
        "вас",
        "ваш",
        "ваша",
        "ваши",
        "если",
        "при",
        "без",
        "так",
        "они",
        "она",
        "оно",
        "его",
        "её",
        "же",
        "ли",
        "бы",
        "мы",
        "не",
        "на",
        "по",
        "из",
        "от",
        "до",
        "за",
        "то",
        # Turkish
        "ve",
        "ile",
        "bir",
        "bu",
        "için",
        "olarak",
        "daha",
        "çok",
        "gibi",
        "mi",
        "mı",
        # Uzbek / Tajik / Kyrgyz common particles
        "ва",
        "ба",
        "аз",
        "дар",
        "учун",
        "билан",
        "жана",
        "менен",
        "жок",
        "бар",
    }
)


def content_tokens(text: str) -> set[str]:
    """Lowercased content words: letters only, length >= 3, not a stopword."""
    return {
        token.lower()
        for token in _WORD.findall(text)
        if len(token) >= 3 and token.lower() not in STOPWORDS
    }


def coverage(claim_text: str, evidence_texts: Sequence[str]) -> float:
    """Fraction of the claim's content tokens that appear in the cited evidence."""
    claim_tokens = content_tokens(claim_text)
    if not claim_tokens:
        return 1.0
    evidence: set[str] = set()
    for text in evidence_texts:
        evidence |= content_tokens(text)
    return len(claim_tokens & evidence) / len(claim_tokens)


class ClaimCheck(Contract):
    """Detailed per-claim verdict — the console's diagnostics drawer reads this."""

    claim_id: str
    verdict: Support
    coverage: float = Field(ge=0.0, le=1.0)
    quote_ok: bool = True
    numeric_ok: bool = True
    missing_numbers: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    rationale: str = ""

    @property
    def needs_repair(self) -> bool:
        return self.verdict is Support.UNSUPPORTED


class LexicalVerifier:
    """Deterministic claim verifier (default ``generate.verifier: lexical``)."""

    name = "lexical-verifier"
    version = "lexical-verifier-1.0.0"

    def __init__(
        self,
        *,
        support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
        partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
    ) -> None:
        if not 0.0 <= partial_threshold <= support_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= partial <= support <= 1")
        self.support_threshold = support_threshold
        self.partial_threshold = partial_threshold

    # -- detailed API (used by the pipeline) --------------------------------------
    def check(
        self,
        claims: Sequence[Claim],
        citations: Mapping[int, Citation],
        chunks: Mapping[str, Chunk],
    ) -> list[ClaimCheck]:
        checks: list[ClaimCheck] = []
        for claim in claims:
            claim_citations = [
                citations[citation_id]
                for citation_id in claim.citation_ids
                if citation_id in citations
            ]
            cited_chunks = [
                chunks[citation.chunk_id]
                for citation in claim_citations
                if citation.chunk_id in chunks
            ]
            quote_ok = bool(claim_citations) and all(
                citation.chunk_id in chunks
                and find_quote_span(chunks[citation.chunk_id].text, citation.quote) is not None
                for citation in claim_citations
            )
            evidence_texts = [chunk.text for chunk in cited_chunks]
            missing_numbers = numeric_violations(claim.text, evidence_texts)
            numeric_ok = not missing_numbers
            ratio = coverage(claim.text, evidence_texts)
            if not claim_citations:
                verdict = Support.UNSUPPORTED
                rationale = "claim has no resolvable citation"
            elif not quote_ok:
                verdict = Support.UNSUPPORTED
                rationale = "a cited quote is no longer a substring of its chunk"
            elif not numeric_ok:
                verdict = Support.UNSUPPORTED
                rationale = f"numbers absent from cited evidence: {missing_numbers}"
            elif ratio >= self.support_threshold:
                verdict = Support.SUPPORTED
                rationale = f"coverage {ratio:.3f} >= {self.support_threshold:.2f}"
            elif ratio >= self.partial_threshold:
                verdict = Support.PARTIAL
                rationale = f"coverage {ratio:.3f} in partial band"
            else:
                verdict = Support.UNSUPPORTED
                rationale = f"coverage {ratio:.3f} < {self.partial_threshold:.2f}"
            checks.append(
                ClaimCheck(
                    claim_id=claim.claim_id,
                    verdict=verdict,
                    coverage=round(ratio, 6),
                    quote_ok=quote_ok,
                    numeric_ok=numeric_ok,
                    missing_numbers=missing_numbers,
                    cited_chunk_ids=[chunk.chunk_id for chunk in cited_chunks],
                    rationale=rationale,
                )
            )
        return checks

    # -- contract Protocol (claims + the chunks each claim rests on) ---------------
    def verify(
        self, claims: Sequence[Claim], passages: Mapping[str, Sequence[Chunk]]
    ) -> list[ClaimVerification]:
        """``ClaimVerifier`` implementation.

        ``passages`` maps **claim id** to the chunks the caller considers evidence for that
        claim. Keys are claim ids (not chunk ids) because the protocol's mapping is
        per-claim; a missing key means "no evidence supplied" and therefore ``unsupported``.
        """
        verdicts: list[ClaimVerification] = []
        for claim in claims:
            chunks = passages.get(claim.claim_id, ())
            evidence_texts = [chunk.text for chunk in chunks]
            missing_numbers = numeric_violations(claim.text, evidence_texts)
            ratio = coverage(claim.text, evidence_texts)
            if missing_numbers or not chunks:
                verdict = Support.UNSUPPORTED
                rationale = (
                    f"numbers absent from evidence: {missing_numbers}"
                    if missing_numbers
                    else "no evidence supplied for this claim"
                )
            elif ratio >= self.support_threshold:
                verdict = Support.SUPPORTED
                rationale = f"coverage {ratio:.3f}"
            elif ratio >= self.partial_threshold:
                verdict = Support.PARTIAL
                rationale = f"coverage {ratio:.3f}"
            else:
                verdict = Support.UNSUPPORTED
                rationale = f"coverage {ratio:.3f}"
            verdicts.append(
                ClaimVerification(
                    claim_id=claim.claim_id,
                    verdict=verdict,
                    rationale=rationale,
                    verifier_version=self.version,
                )
            )
        return verdicts


__all__ = [
    "DEFAULT_PARTIAL_THRESHOLD",
    "DEFAULT_SUPPORT_THRESHOLD",
    "STOPWORDS",
    "ClaimCheck",
    "LexicalVerifier",
    "content_tokens",
    "coverage",
]
