"""The deterministic normaliser (control arm and fallback backend).

# AT-GEN-14
"""

from __future__ import annotations

import pytest

from answerdesk.rewrite.deterministic import (
    DeterministicRewriter,
    transliterate_to_cyrillic,
)


def test_fillers_are_removed_from_both_edges() -> None:
    rewriter = DeterministicRewriter()
    cases = {
        "Здравствуйте, подскажите пожалуйста сколько стоит карта": "сколько стоит карта",
        "сколько стоит карта, спасибо": "сколько стоит карта, спасибо",
        "подскажите где отделение": "где отделение",
        "Please how do i block the card": "block the card",
    }
    for source, expected in cases.items():
        # locale="tr" keeps the Latin text Latin: transliteration is for Cyrillic locales only.
        assert rewriter.rewrite(source, locale="tr").query == expected


def test_transliteration_only_applies_to_wholly_latin_queries() -> None:
    rewriter = DeterministicRewriter()
    assert rewriter.rewrite("zaym kak oformit").query == "займ как оформить"
    assert rewriter.rewrite("dengi kak poluchit").query == "деньги как получить"
    # A known non-Cyrillic locale disables transliteration entirely.
    assert rewriter.rewrite("block the card", locale="tr").query == "block the card"
    # Mixed input keeps its identifiers untouched.
    mixed = rewriter.rewrite("zaym ORD-12345 kak oformit", locale="ru")
    assert "ORD-12345" in mixed.query
    assert "займ" in mixed.query


def test_transliteration_table_covers_the_corpus_sounds() -> None:
    assert transliterate_to_cyrillic("shch") == "щ"
    assert transliterate_to_cyrillic("zh") == "ж"
    assert transliterate_to_cyrillic("yu") == "ю"
    assert transliterate_to_cyrillic("perevod") == "перевод"
    assert transliterate_to_cyrillic("123") == "123"


def test_curated_repairs_fix_the_known_misspellings() -> None:
    rewriter = DeterministicRewriter()
    assert rewriter.rewrite("денги не дошли").query == "деньги не дошли"
    assert rewriter.rewrite("первод таджикистан").query == "перевод таджикистан"
    assert rewriter.rewrite("займ как оформит").query == "займ как оформить"


def test_repairs_do_not_touch_clean_text() -> None:
    rewriter = DeterministicRewriter()
    clean = "Сколько стоит обслуживание карты «Вектор»?"
    assert rewriter.rewrite(clean).query == clean


def test_output_is_idempotent() -> None:
    rewriter = DeterministicRewriter()
    once = rewriter.rewrite("Здравствуйте! zaym kak oformit, денги ne doshli").query
    twice = rewriter.rewrite(once).query
    assert once == twice


def test_notes_explain_what_happened() -> None:
    rewriter = DeterministicRewriter()
    result = rewriter.rewrite("подскажите zaym kak oformit")
    assert result.changed is True
    assert "filler-removed" in result.notes
    assert "transliterated" in result.notes
    assert result.backend == "deterministic"


def test_knobs_disable_each_stage() -> None:
    no_translit = DeterministicRewriter(transliterate=False)
    assert no_translit.rewrite("zaym").query == "zaym"
    no_fillers = DeterministicRewriter(drop_fillers=False)
    assert no_fillers.rewrite("подскажите займ").query == "подскажите займ"
    no_repair = DeterministicRewriter(repair=False)
    assert no_repair.rewrite("денги").query == "денги"


@pytest.mark.parametrize("query", ["", "   ", "?!.", "\u00a0"])
def test_degenerate_input_is_safe(query: str) -> None:
    result = DeterministicRewriter().rewrite(query)
    assert result.query.strip() or result.query == " "
    assert result.available is True


def test_unicode_is_normalised() -> None:
    rewriter = DeterministicRewriter()
    # Whitespace is collapsed; sentence punctuation is preserved (a clean query must stay
    # byte-identical so the clean arm of the measurement is a true baseline).
    assert rewriter.rewrite("Сколько  стоит\u00a0карта?").query == "Сколько стоит карта?"
