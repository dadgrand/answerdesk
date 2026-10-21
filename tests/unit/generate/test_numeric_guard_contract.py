"""Unit layer for R-04: the verifier consults the numeric guard, and in the right order.

`tests/integration/generation/test_verifier_numeric.py` covers the end-to-end verdicts; this
module isolates `LexicalVerifier.check` so the *decision order* is pinned without any pipeline,
provider or packer in the way:

``citations → quote → numbers → coverage``

A regression that moves the numeric branch below the coverage branch (or removes it) changes the
verdict here even when every claim is otherwise well formed.

    # AT-GEN-12  # CR-9
"""

from __future__ import annotations

import pytest

from answerdesk.contracts import Chunk, Citation, Claim, Support
from answerdesk.generate.verifier import LexicalVerifier

EVIDENCE = "Обслуживание дебетовой карты стоит 500 рублей в месяц."


def _check(claim_text: str, *, evidence: str = EVIDENCE, quote: str | None = None):
    chunk = Chunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        text=evidence,
        heading_path=["Тарифы"],
        breadcrumb="Тарифы",
        start_char=0,
        end_char=len(evidence),
        token_count=len(evidence.split()),
        language="ru",
    )
    citation = Citation(
        citation_id=1,
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        doc_title="Тарифы",
        heading_path=list(chunk.heading_path),
        quote=quote if quote is not None else evidence,
        start_char=0,
        end_char=len(evidence),
        doc_hash=chunk.doc_hash,
    )
    verifier = LexicalVerifier()
    [check] = verifier.check(
        [Claim(claim_id="c1", text=claim_text, citation_ids=[1])],
        {1: citation},
        {chunk.chunk_id: chunk},
    )
    return check


def test_numeric_violation_is_reported_on_the_check() -> None:
    check = _check("Обслуживание дебетовой карты стоит 999 рублей в месяц.")
    assert check.numeric_ok is False
    assert check.missing_numbers == ["999"]
    assert check.verdict is Support.UNSUPPORTED
    assert check.rationale.startswith("numbers absent from cited evidence")
    assert check.citation_ids if hasattr(check, "citation_ids") else True
    assert check.cited_chunk_ids == ["chunk-1"]


def test_numeric_defect_outranks_a_partial_coverage_score() -> None:
    """Even at partial coverage the reason must be numeric — the guard runs before coverage."""
    claim = "Стоимость обслуживания 999 рублей, списание ежемесячно."
    check = _check(claim)
    assert check.numeric_ok is False
    assert check.verdict is Support.UNSUPPORTED
    assert "999" in check.rationale
    assert "coverage" not in check.rationale


def test_matching_numbers_keep_the_coverage_verdict() -> None:
    check = _check("Обслуживание дебетовой карты стоит 500 рублей в месяц.")
    assert check.numeric_ok is True
    assert check.verdict is Support.SUPPORTED
    assert check.rationale.startswith("coverage")


def test_a_broken_quote_still_wins_over_the_numbers() -> None:
    """Citation integrity is checked first; the guard must not mask a phantom quote."""
    check = _check(
        "Обслуживание дебетовой карты стоит 999 рублей в месяц.",
        quote="Такой фразы в документе нет.",
    )
    assert check.quote_ok is False
    assert check.verdict is Support.UNSUPPORTED
    assert "substring" in check.rationale


def test_claim_without_a_resolvable_citation_is_unsupported() -> None:
    chunk = Chunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        text=EVIDENCE,
        heading_path=["Тарифы"],
        breadcrumb="Тарифы",
        start_char=0,
        end_char=len(EVIDENCE),
        token_count=len(EVIDENCE.split()),
        language="ru",
    )
    [check] = LexicalVerifier().check(
        [Claim(claim_id="c1", text="Обслуживание стоит 500 рублей.", citation_ids=[])],
        {},
        {chunk.chunk_id: chunk},
    )
    assert check.verdict is Support.UNSUPPORTED
    assert "no resolvable citation" in check.rationale
    # With no cited evidence there is nothing to compare against, so every claim number is
    # "absent" by definition; the verdict still names the real defect (the citation).
    assert check.numeric_ok is False


@pytest.mark.parametrize(
    ("claim", "missing"),
    [
        ("Комиссия 500 рублей.", []),
        ("Комиссия 501 рублей.", ["501"]),
        ("Комиссия 500,50 рублей.", ["500.5"]),
        ("Лимит 500 000 рублей.", ["500000"]),
    ],
    ids=["match", "integer", "decimal", "thousands"],
)
def test_missing_numbers_are_exact(claim: str, missing: list[str]) -> None:
    check = _check(claim, evidence="Комиссия 500 рублей. Лимит 150 000 рублей.")
    assert check.missing_numbers == missing
