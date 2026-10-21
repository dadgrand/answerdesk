"""R-04: the numeric guard where it is *enforced* (the verifier), not just where it is defined.

The t6 numeric tests (`test_numeric_guard.py`) cover canonicalisation and `numeric_violations()`.
They cannot fail if the verifier stops *consulting* that function: the review's mutation M6
(`elif not numeric_ok:` → `elif False:`) left all 41 generation tests green while an independent
probe showed a claim with **coverage 1.000** and a wrong number (`999` where the evidence says
`500`) being returned as `supported`. These tests close that gap at the two levels where the
verdict is actually produced:

* the verifier (`LexicalVerifier.check`) — wrong number at full lexical coverage is
  `unsupported` *with a numeric rationale*, and the matching claim is still `supported`;
* the pipeline (`AnswerPipeline.answer`) — such a claim is dropped (or repaired away), so a wrong
  number can never reach the answer text.

The mismatch matrix spans integer, decimal, thousands-separator, currency and date forms in ru,
uz and zh answer text, because canonicalisation differs per script and per separator convention.

    # AT-GEN-12  # AT-GEN-02  # INV-04  # CR-9
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from support import CARD_FEE_QUOTE, ScriptedProvider, as_chat_provider, draft_json

from answerdesk.contracts import Chunk, Citation, Claim, RetrievedPassage, Support
from answerdesk.generate.pipeline import AnswerPipeline
from answerdesk.generate.verifier import LexicalVerifier, coverage

#: (locale, form, evidence sentence, claim with a *wrong* number, claim with the right one)
#: The claim keeps every word of the evidence, so lexical coverage is exactly 1.0 and the
#: number is the only defect — the R-04 condition.
MISMATCHES: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "ru",
        "integer",
        "Комиссия за перевод составляет 500 рублей.",
        "Комиссия за перевод составляет 999 рублей.",
        "Комиссия за перевод составляет 500 рублей.",
    ),
    (
        "ru",
        "decimal",
        "Ставка по вкладу составляет 12,5% годовых.",
        "Ставка по вкладу составляет 12,7% годовых.",
        "Ставка по вкладу составляет 12,5% годовых.",
    ),
    (
        "ru",
        "thousands_separator",
        "Суточный лимит переводов составляет 150 000 рублей.",
        "Суточный лимит переводов составляет 500 000 рублей.",
        "Суточный лимит переводов составляет 150 000 рублей.",
    ),
    (
        "ru",
        "currency",
        "Обслуживание карты стоит 120 рублей в месяц.",
        "Обслуживание карты стоит 115 рублей в месяц.",
        "Обслуживание карты стоит 120 рублей в месяц.",
    ),
    (
        "ru",
        "date",
        "Плата списывается 5 марта 2026 года.",
        "Плата списывается 6 марта 2026 года.",
        "Плата списывается 5 марта 2026 года.",
    ),
    (
        "uz",
        "currency",
        "Karta xizmati bir oyda 500 so'm turadi.",
        "Karta xizmati bir oyda 999 so'm turadi.",
        "Karta xizmati bir oyda 500 so'm turadi.",
    ),
    (
        "uz",
        "decimal",
        "Omonat foiz stavkasi 12,5% ni tashkil qiladi.",
        "Omonat foiz stavkasi 14,5% ni tashkil qiladi.",
        "Omonat foiz stavkasi 12,5% ni tashkil qiladi.",
    ),
    (
        "uz",
        "date",
        "To'lov 05.03.2026 kuni yechib olinadi.",
        "To'lov 06.03.2026 kuni yechib olinadi.",
        "To'lov 05.03.2026 kuni yechib olinadi.",
    ),
    (
        "zh",
        "currency",
        "转账手续费为500元。",
        "转账手续费为999元。",
        "转账手续费为500元。",
    ),
    (
        "zh",
        "decimal",
        "存款年利率为12.5%。",
        "存款年利率为12.7%。",
        "存款年利率为12.5%。",
    ),
    (
        "zh",
        "thousands_separator",
        "每日转账限额为150,000元。",
        "每日转账限额为500,000元。",
        "每日转账限额为150,000元。",
    ),
    (
        "zh",
        "date",
        "费用将于2026年3月5日扣除。",
        "费用将于2026年3月6日扣除。",
        "费用将于2026年3月5日扣除。",
    ),
)


def _verify(claim_text: str, evidence_text: str, *, locale: str = "ru"):
    """Run the shipped verifier on one claim citing one chunk and return its ``ClaimCheck``."""
    chunk = Chunk(
        chunk_id=f"chunk-{locale}-numeric",
        doc_id=f"doc-{locale}-numeric",
        text=evidence_text,
        heading_path=["Числа"],
        breadcrumb="Числа",
        start_char=0,
        end_char=len(evidence_text),
        token_count=len(evidence_text.split()),
        language=locale,
    )
    citation = Citation(
        citation_id=1,
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        doc_title="Числа",
        heading_path=list(chunk.heading_path),
        quote=evidence_text,
        start_char=0,
        end_char=len(evidence_text),
        doc_hash=chunk.doc_hash,
    )
    claim = Claim(claim_id="c1", text=claim_text, citation_ids=[1])
    verifier = LexicalVerifier()
    [check] = verifier.check([claim], {1: citation}, {chunk.chunk_id: chunk})
    return check, coverage(claim_text, [evidence_text])


def test_verifier_rejects_a_wrong_number_at_full_coverage() -> None:
    """The R-04 probe, verbatim: 999 against evidence that says 500, coverage 1.000."""
    evidence = "Обслуживание дебетовой карты стоит 500 рублей в месяц."
    claim = "Обслуживание дебетовой карты стоит 999 рублей в месяц."
    check, ratio = _verify(claim, evidence)

    assert ratio == pytest.approx(1.0), "the probe must have perfect lexical coverage"
    assert check.coverage == pytest.approx(1.0)
    assert check.verdict is Support.UNSUPPORTED, check.rationale
    assert check.numeric_ok is False
    assert check.missing_numbers == ["999"]
    assert "number" in check.rationale.lower(), check.rationale
    assert check.needs_repair is True


def test_verifier_accepts_the_same_claim_when_the_number_matches() -> None:
    """No behaviour change for a claim whose number is in its evidence."""
    evidence = "Обслуживание дебетовой карты стоит 500 рублей в месяц."
    check, ratio = _verify("Обслуживание дебетовой карты стоит 500 рублей в месяц.", evidence)
    assert ratio == pytest.approx(1.0)
    assert check.numeric_ok is True
    assert check.missing_numbers == []
    assert check.verdict is Support.SUPPORTED, check.rationale


@pytest.mark.parametrize(
    ("locale", "form", "evidence", "wrong_claim", "right_claim"),
    MISMATCHES,
    ids=[f"{locale}-{form}" for locale, form, *_ in MISMATCHES],
)
def test_numeric_mismatch_is_rejected_across_forms_and_locales(
    locale: str, form: str, evidence: str, wrong_claim: str, right_claim: str
) -> None:
    """Integer / decimal / thousands / currency / date mismatches in ru, uz and zh."""
    wrong, wrong_ratio = _verify(wrong_claim, evidence, locale=locale)
    assert wrong_ratio == pytest.approx(1.0), (
        f"{locale}/{form}: the words are identical, so only the number can be the defect"
    )
    assert wrong.verdict is Support.UNSUPPORTED, f"{locale}/{form}: {wrong.rationale}"
    assert wrong.numeric_ok is False, f"{locale}/{form} was not caught by the numeric guard"
    assert wrong.missing_numbers, f"{locale}/{form}: no missing token was reported"
    assert "number" in wrong.rationale.lower(), wrong.rationale

    right, right_ratio = _verify(right_claim, evidence, locale=locale)
    assert right_ratio == pytest.approx(1.0)
    assert right.numeric_ok is True, f"{locale}/{form}: a matching number was rejected"
    assert right.verdict is Support.SUPPORTED, f"{locale}/{form}: {right.rationale}"


def test_numeric_check_runs_before_coverage_is_consulted() -> None:
    """A numeric defect outranks a *high* coverage score, not just a perfect one.

    The review's mutation disabled the numeric branch and let coverage decide; here the claim
    reuses almost every content word but quotes a different fee, which must still be rejected.
    """
    evidence = "Обслуживание дебетовой карты «Вектор» стоит 500 рублей в месяц."
    claim = "Обслуживание дебетовой карты «Вектор» стоит 999 рублей в месяц."
    check, ratio = _verify(claim, evidence)
    assert ratio == pytest.approx(1.0)
    assert check.numeric_ok is False
    assert check.verdict is Support.UNSUPPORTED
    assert "999" in check.rationale


def test_wrong_number_is_dropped_before_display(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    """End to end: the pipeline drops the claim, so the wrong number never reaches the text."""
    wrong_claim = "Обслуживание карты «Вектор» стоит 999 рублей в месяц."
    provider = ScriptedProvider([draft_json([(wrong_claim, "p1", CARD_FEE_QUOTE)])])
    pipeline = pipeline_factory(as_chat_provider(provider), repair_passes=0)

    result = pipeline.answer("Сколько стоит обслуживание карты?", passages)

    assert result.dropped_claims == 1, result.signal.detail
    assert result.numeric_violations_pre_repair == 1
    assert result.claims == []
    assert "999" not in result.text
    assert result.signal.action.value != "answer", result.signal


def test_wrong_number_triggers_the_repair_pass(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    """With a repair pass available, the guard drives a second draft instead of dropping."""
    wrong_claim = "Обслуживание карты «Вектор» стоит 999 рублей в месяц."
    right_claim = "Обслуживание карты «Вектор» стоит 120 рублей в месяц."
    provider = ScriptedProvider(
        [
            draft_json([(wrong_claim, "p1", CARD_FEE_QUOTE)]),
            draft_json([(right_claim, "p1", CARD_FEE_QUOTE)]),
        ]
    )
    pipeline = pipeline_factory(as_chat_provider(provider), repair_passes=1)

    result = pipeline.answer("Сколько стоит обслуживание карты?", passages)

    assert result.repaired is True
    assert result.numeric_violations_pre_repair == 1
    assert "999" not in result.text
    assert "120" in result.text
    assert [claim.text for claim in result.claims] == [right_claim]


def test_pipeline_does_not_repair_when_every_number_matches(
    pipeline_factory: Callable[..., AnswerPipeline],
    passages: list[RetrievedPassage],
) -> None:
    """The matching case stays a single-pass answer — the guard must not add a rejection."""
    right_claim = "Обслуживание карты «Вектор» стоит 120 рублей в месяц."
    provider = ScriptedProvider([draft_json([(right_claim, "p1", CARD_FEE_QUOTE)])])
    pipeline = pipeline_factory(as_chat_provider(provider), repair_passes=1)

    result = pipeline.answer("Сколько стоит обслуживание карты?", passages)

    assert result.repaired is False
    assert result.numeric_violations_pre_repair == 0
    assert provider.call_count == 1, "a supported claim must not trigger a second model call"
    assert "120" in result.text


def test_numeric_guard_is_deterministic_across_runs() -> None:
    """Same inputs, same verdict: the guard is arithmetic, never a model."""
    evidence = "Суточный лимит переводов составляет 150 000 рублей."
    claim = "Суточный лимит переводов составляет 500 000 рублей."
    verdicts = {_verify(claim, evidence)[0].verdict for _ in range(3)}
    assert verdicts == {Support.UNSUPPORTED}
