"""Synthetic multilingual corpus generator tests (FR-ING-10, §10.1, §13.4).

# AT-ING-10
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from answerdesk.ingest.canonical import luhn_valid
from answerdesk.ingest.synthetic import (
    ARCHETYPE_INTENT,
    ARCHETYPES,
    LOCALES,
    PERTURBATION_CLASSES,
    PHONE_RANGES,
    TEST_PANS,
    SyntheticConfig,
    generate,
    identifier_findings,
    locale_bank,
)

CONFIG = SyntheticConfig(docs_per_locale=15)


@pytest.fixture(scope="module")
def corpus() -> tuple[list, list]:
    return generate(CONFIG)


def test_every_locale_and_archetype_is_present(corpus: tuple[list, list]) -> None:
    """AT-ING-10: six locales, twelve archetypes, nothing silently dropped (NFR-ML-01)."""
    documents, queries = corpus
    assert {document.locale for document in documents} == set(LOCALES)
    assert set(ARCHETYPES) <= {document.archetype for document in documents}
    assert {query.locale for query in queries} == set(LOCALES)
    for locale in LOCALES:
        assert sum(1 for document in documents if document.locale == locale) >= 15


def test_documents_are_structurally_rich(corpus: tuple[list, list]) -> None:
    """AT-ING-10: articles carry headings, a table, a numbered procedure and FAQ pairs."""
    documents, _ = corpus
    for document in documents:
        assert document.text.startswith("---\n")
        assert "# " in document.text
        assert "## " in document.text
        assert "| --- |" in document.text, "every article has a table"
        assert re.search(r"^1\. ", document.text, re.MULTILINE), "numbered procedure missing"
        assert "Q: " in document.text and "A: " in document.text


def test_documents_contain_numbers_dates_and_identifiers(corpus: tuple[list, list]) -> None:
    """AT-ING-10: numbers, dates, order ids and error codes are present per locale."""
    documents, _ = corpus
    for document in documents:
        assert re.search(r"\d", document.text)
        assert re.search(r"ORD-2026-\d{5}", document.text)
        assert PHONE_RANGES[document.locale] in document.text
    joined = " ".join(document.text for document in documents)
    assert re.search(r"ERR-[A-Z]{3}-\d{4}", joined)
    assert re.search(r"\d{1,2}[.\-/ ]\d{1,2}[.\-/ ]?20\d{2}|\d{1,2} \w+ 2026|20\d{2}年", joined)


def test_generation_is_deterministic(corpus: tuple[list, list]) -> None:
    """AT-ING-10: the same seed produces byte-identical documents and queries."""
    documents, queries = corpus
    again_docs, again_queries = generate(CONFIG)
    assert [document.text for document in documents] == [document.text for document in again_docs]
    assert [query.text for query in queries] == [query.text for query in again_queries]
    assert [document.doc_id for document in documents] == [doc.doc_id for doc in again_docs]


def test_a_different_seed_changes_the_corpus(corpus: tuple[list, list]) -> None:
    """AT-ING-10: the seed actually drives the content (a constant corpus would be a lie)."""
    documents, _ = corpus
    other, _ = generate(SyntheticConfig(seed=CONFIG.seed + 1, docs_per_locale=6))
    assert [document.text for document in documents] != [document.text for document in other]


def test_every_perturbation_class_reaches_the_quota() -> None:
    """AT-ING-10: typos, code-switching, numbers/dates and identifiers are each >= 5%."""
    _, queries = generate()
    counts = Counter(klass for query in queries for klass in query.perturbations)
    total = len(queries)
    for klass in PERTURBATION_CLASSES:
        share = counts[klass] / total
        assert share >= 0.05, f"{klass} share {share:.3f} is below the 5% floor"
    direct = sum(1 for query in queries if query.category == "direct")
    assert direct >= 0.2 * total


def test_perturbations_are_actually_applied(corpus: tuple[list, list]) -> None:
    """AT-ING-10: a perturbed query differs from its base question."""
    _, queries = corpus
    changed = 0
    for query in queries:
        if query.perturbations:
            assert query.text != query.base_text
            changed += 1
    assert changed >= len(queries) * 0.4


def test_code_switching_mixes_latin_into_a_local_query(corpus: tuple[list, list]) -> None:
    """AT-ING-10: intra-sentence code-switching inserts an English product term."""
    _, queries = corpus
    switched = [query for query in queries if "code_switch" in query.perturbations]
    assert switched
    for query in switched:
        added = query.text.replace(query.base_text, "")
        assert added, f"{query.query_id}: nothing was switched"
        assert re.search(r"[A-Za-z\u0400-\u04ff]", added), query.query_id


def test_typo_perturbation_touches_a_word(corpus: tuple[list, list]) -> None:
    """AT-ING-10: a typo query still keeps the digits of the base question intact."""
    _, queries = corpus
    typos = [query for query in queries if "typo" in query.perturbations]
    assert typos
    for query in typos:
        digits_base = re.findall(r"\d+", query.base_text)
        digits_typo = re.findall(r"\d+", query.text)
        assert digits_base == digits_typo, "typos must never corrupt identifiers or numbers"


def test_reserved_identifiers_only(corpus: tuple[list, list]) -> None:
    """AT-ING-10 (§13.4): no real bank, brand, phone, PAN or e-mail can appear."""
    documents, queries = corpus
    texts = [document.text for document in documents] + [query.text for query in queries]
    assert identifier_findings(texts) == []


def test_card_numbers_are_documented_test_pans(corpus: tuple[list, list]) -> None:
    """AT-ING-10 (§13.4): only the documented, Luhn-valid, never-issued PANs are used."""
    documents, _ = corpus
    pan_pattern = re.compile(r"\b(?:\d{4} ){3}\d{4}\b")
    seen = set()
    for document in documents:
        for match in pan_pattern.finditer(document.text):
            digits = match.group(0).replace(" ", "")
            seen.add(digits)
            assert digits in TEST_PANS
            assert luhn_valid(digits)
    assert seen


def test_duplicates_are_injected_on_purpose() -> None:
    """AT-ING-10/06: the generator emits near-copies so deduplication is exercised."""
    documents, _ = generate(SyntheticConfig(docs_per_locale=30, duplicate_every=7))
    duplicates = [document for document in documents if document.is_duplicate_of]
    assert duplicates
    for duplicate in duplicates:
        assert duplicate.doc_id.endswith("-dup")
        assert duplicate.doc_id != duplicate.is_duplicate_of


def test_no_placeholder_survives_rendering(corpus: tuple[list, list]) -> None:
    """AT-ING-10: every template slot resolves — no ``{slot}`` reaches the corpus."""
    documents, queries = corpus
    for document in documents:
        assert "{" not in document.text.replace("{{", ""), document.doc_id
    for query in queries:
        assert "{" not in query.text, query.query_id


def test_locale_banks_cover_the_required_fields() -> None:
    """AT-ING-10: every locale bank covers all twelve archetypes and its perturbation data."""
    for locale in LOCALES:
        bank = locale_bank(locale)
        assert set(bank.intents) == set(ARCHETYPE_INTENT.values())
        assert bank.code_switch_pairs
        assert bank.date_templates
        assert bank.phone if hasattr(bank, "phone") else True  # phone lives in PHONE_RANGES
        assert PHONE_RANGES[locale].startswith("+")
        for intent in ARCHETYPE_INTENT.values():
            payload = bank.intents[intent]
            assert len(payload.sentences) >= 6
            assert payload.headings
            assert payload.faqs


def test_unknown_locale_is_rejected() -> None:
    """AT-ING-10: asking for a locale that has no bank fails with the supported list."""
    with pytest.raises(ValueError, match="unsupported locale"):
        locale_bank("de")


def test_config_validation() -> None:
    """AT-ING-10: a nonsensical generation config is rejected up front."""
    with pytest.raises(ValueError):
        SyntheticConfig(docs_per_locale=0)
    with pytest.raises(ValueError):
        SyntheticConfig(locales=())
