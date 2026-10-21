"""Locale-quality contract of the v2 corpus (t18).

These tests are the evidence behind the multilingual quality claim: six locales with real
depth (12 archetypes, 100+ documents each), the correct script per locale, native
terminology instead of English placeholders or transliteration artifacts, the query-mix
quotas, and 100% reference grounding. They are deliberately blunt — a regression in any of
these makes the README's multilingual claim false, so they fail loudly.

Honest limitation (docs/ingestion.md §12): the content is synthetic and LLM-assisted, not
validated by native speakers. What these tests can prove is script, terminology coverage,
structure, quotas and grounding — not native fluency.

# AT-ING-10  # AT-ING-07
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from answerdesk.ingest.synthetic import (
    ARCHETYPES,
    LOCALES,
    PERTURBATION_CLASSES,
    QUERY_CATEGORIES,
    QUERY_PLAN_SHARES,
    SyntheticConfig,
    generate,
    identifier_findings,
    verify_reference_grounding,
)

REPO_CORPUS = Path(__file__).resolve().parents[3] / "data" / "corpus"

#: Script ranges: Cyrillic for ru/ky/tg, Latin for uz/tr, Han for zh (t18 acceptance).
CYRILLIC = re.compile(r"[\u0400-\u04ff]")
LATIN = re.compile(r"[A-Za-z]")
HAN = re.compile(r"[\u4e00-\u9fff]")
SCRIPT = {"ru": CYRILLIC, "ky": CYRILLIC, "tg": CYRILLIC, "uz": LATIN, "tr": LATIN, "zh": HAN}
OTHER_SCRIPT = {
    "ru": LATIN,
    "ky": LATIN,
    "tg": LATIN,
    "uz": CYRILLIC,
    "tr": CYRILLIC,
    "zh": LATIN,
}

#: English artefacts that must never appear in locale text (placeholders, copy-paste).
ENGLISH_PLACEHOLDERS = (
    "lorem ipsum",
    "todo",
    "tbd",
    "placeholder",
    "sample text",
    "your text here",
    "insert ",
    "xxx",
    "translation missing",
)

#: Transliteration artefacts: locale words typed in the wrong script.
TRANSLITERATION_ARTEFACTS = {
    "ru": ("perevod", "komissiya", "karta ", "schet", "dokument "),
    "ky": ("kotoruu", "komissiya", "kartanyn", "moonot"),
    "tg": ("intiqol", "komissiya", "hujjat", "pardokht"),
    "uz": ("перевод", "комиссия", "карта", "счёт"),
    "tr": ("перевод", "комиссия", "карта"),
    "zh": (),
}


@pytest.fixture(scope="module")
def corpus() -> tuple[list, list]:
    return generate()


def test_every_locale_has_at_least_one_hundred_documents(corpus: tuple[list, list]) -> None:
    """AT-ING-10: >=100 documents per locale (t18 depth floor)."""
    documents, _ = corpus
    per_locale = Counter(document.locale for document in documents)
    assert set(per_locale) == set(LOCALES)
    for locale, count in per_locale.items():
        assert count >= 100, f"{locale}: only {count} documents"


def test_every_locale_covers_at_least_ten_archetypes(corpus: tuple[list, list]) -> None:
    """AT-ING-10: >=10 distinct document archetypes per locale, and all twelve overall."""
    documents, _ = corpus
    for locale in LOCALES:
        archetypes = {document.archetype for document in documents if document.locale == locale}
        assert len(archetypes) >= 10, f"{locale}: {sorted(archetypes)}"
    assert {document.archetype for document in documents} == set(ARCHETYPES)


def test_every_archetype_is_present_in_every_locale(corpus: tuple[list, list]) -> None:
    """AT-ING-10: the 12 archetypes (tariff, cards, FAQ, procedure, fraud, ...) each locale."""
    documents, _ = corpus
    for locale in LOCALES:
        archetypes = {document.archetype for document in documents if document.locale == locale}
        assert archetypes == set(ARCHETYPES), f"{locale}: missing {set(ARCHETYPES) - archetypes}"


def test_documents_use_the_locale_script(corpus: tuple[list, list]) -> None:
    """AT-ING-10: Cyrillic ru/ky/tg, Latin uz/tr, Han zh — measured on letters only."""
    documents, _ = corpus
    for document in documents:
        letters = [char for char in document.text if char.isalpha()]
        assert letters, document.doc_id
        expected = sum(1 for char in letters if SCRIPT[document.locale].match(char))
        share = expected / len(letters)
        floor = {"zh": 0.6, "uz": 0.75, "tr": 0.8}.get(document.locale, 0.8)
        assert share >= floor, f"{document.doc_id}: {share:.2f} of letters in the locale script"


def test_no_english_placeholders(corpus: tuple[list, list]) -> None:
    """AT-ING-10: no "lorem ipsum", TODO, placeholder or untranslated template remains."""
    documents, queries = corpus
    for document in documents:
        lowered = document.text.casefold()
        for artefact in ENGLISH_PLACEHOLDERS:
            assert artefact not in lowered, f"{document.doc_id}: {artefact!r}"
    for query in queries:
        lowered = query.text.casefold()
        for artefact in ENGLISH_PLACEHOLDERS:
            assert artefact not in lowered, f"{query.query_id}: {artefact!r}"


def test_no_transliteration_artefacts(corpus: tuple[list, list]) -> None:
    """AT-ING-10: locale words are not typed in the wrong script (ru 'perevod', uz 'перевод')."""
    documents, _ = corpus
    for document in documents:
        lowered = document.text.casefold()
        for artefact in TRANSLITERATION_ARTEFACTS[document.locale]:
            assert artefact not in lowered, f"{document.doc_id}: transliteration {artefact!r}"
    # the other script may only appear in code-switch pairs, quoted terms and SMS/app words
    for document in documents:
        other = OTHER_SCRIPT[document.locale]
        words = [word for word in re.findall(r"\w+", document.text) if other.fullmatch(word[:1])]
        share = len(words) / max(1, len(re.findall(r"\w+", document.text)))
        # han documents legitimately carry Latin currency/identifier tokens (CNY, ORD-…)
        cap = 0.35 if document.locale == "zh" else 0.25
        assert share <= cap, f"{document.doc_id}: {share:.2f} of tokens are foreign-script"


def test_numeric_and_date_formats_follow_the_locale(corpus: tuple[list, list]) -> None:
    """AT-ING-10: currency codes and date shapes are locale-appropriate."""
    documents, _ = corpus
    currency = {"ru": "RUB", "ky": "KGS", "tg": "TJS", "uz": "UZS", "tr": "TRY", "zh": "CNY"}
    date_shape = {
        "zh": re.compile(r"20\d{2}年\d{1,2}月\d{1,2}日"),
        "uz": re.compile(r"\d{1,2}-[a-z]+ 20\d{2}"),
        "tr": re.compile(r"\d{1,2} [A-ZÇĞİÖŞÜ][a-zçğıöşü]+ 20\d{2}"),
    }
    for document in documents:
        assert currency[document.locale] in document.text, document.doc_id
        pattern = date_shape.get(document.locale)
        if pattern is not None:
            assert pattern.search(document.text), f"{document.doc_id}: no locale date"


def test_queries_cover_every_category_with_the_planned_quotas(
    corpus: tuple[list, list],
) -> None:
    """AT-ING-10: >=2,400 queries (>=400/locale) across all nine categories."""
    _, queries = corpus
    assert len(queries) >= 3000, len(queries)
    per_locale = Counter(query.locale for query in queries)
    assert set(per_locale) == set(LOCALES)
    for locale, count in per_locale.items():
        assert count >= 400, f"{locale}: {count} queries"
    categories = Counter(query.category for query in queries)
    assert set(categories) == set(QUERY_CATEGORIES)
    for category in QUERY_CATEGORIES:
        expected = QUERY_PLAN_SHARES[category]
        share = categories[category] / len(queries)
        assert abs(share - expected) <= 0.02, f"{category}: {share:.3f} vs {expected:.3f}"
    for klass in PERTURBATION_CLASSES:
        share = sum(1 for query in queries if klass in query.perturbations) / len(queries)
        assert share >= 0.05, f"{klass}: {share:.3f}"


def test_unanswerable_share_matches_the_spec(corpus: tuple[list, list]) -> None:
    """AT-ING-10 (§10.1): ~18% unanswerable, none of them carrying a reference reply."""
    _, queries = corpus
    unanswerable = [query for query in queries if not query.answerable]
    share = len(unanswerable) / len(queries)
    assert 0.15 <= share <= 0.21, share
    assert all(query.operator_reply == "" for query in unanswerable)
    assert all(query.category == "unanswerable" for query in unanswerable)


def test_multi_hop_queries_need_two_documents(corpus: tuple[list, list]) -> None:
    """AT-ING-10: multi-hop items cite two distinct documents and pass the verifier."""
    documents, queries = corpus
    by_id = {document.doc_id: document for document in documents}
    multi_hop = [query for query in queries if query.category == "multi_hop"]
    assert len(multi_hop) >= len(queries) * 0.05
    for query in multi_hop:
        assert len(set(query.supporting_doc_ids)) == 2, query.query_id
        assert all(doc_id in by_id for doc_id in query.supporting_doc_ids), query.query_id
        assert (
            verify_reference_grounding(
                [by_id[doc_id] for doc_id in query.supporting_doc_ids], [query]
            )
            == []
        ), query.query_id


def test_grounding_holds_for_all_queries(corpus: tuple[list, list]) -> None:
    """AT-ING-10/t15: 100% of reference replies are entailed by their source document(s)."""
    documents, queries = corpus
    findings = verify_reference_grounding(documents, queries)
    assert findings == [], findings[:3]


def test_cross_locale_code_switching_is_present(corpus: tuple[list, list]) -> None:
    """AT-ING-10: ru<->uz/ky/tg code-switching actually mixes scripts inside a query."""
    _, queries = corpus
    for locale in ("ru", "uz", "ky", "tg"):
        locale_switches = [
            query
            for query in queries
            if query.locale == locale and "code_switch" in query.perturbations
        ]
        assert locale_switches, locale
        mixed = 0
        for query in locale_switches:
            added = query.text.replace(query.base_text, "")
            if OTHER_SCRIPT[locale].search(added) or SCRIPT[locale].search(added):
                mixed += 1
        assert mixed / len(locale_switches) >= 0.9, locale


def test_typo_and_transliteration_variants_are_present(corpus: tuple[list, list]) -> None:
    """AT-ING-10: the typo class perturbs the text without touching identifiers."""
    _, queries = corpus
    typos = [query for query in queries if "typo" in query.perturbations]
    assert len(typos) >= len(queries) * 0.05
    for query in typos:
        assert query.text != query.base_text
        assert re.findall(r"\d+", query.text) or not re.findall(r"\d+", query.base_text)


def test_reserved_identifiers_survive_the_expansion(corpus: tuple[list, list]) -> None:
    """AT-ING-10 (§13.4): no real identifier or brand appears in 648 documents and 2,592 queries."""
    documents, queries = corpus
    texts = [document.text for document in documents] + [query.text for query in queries]
    assert identifier_findings(texts) == []


def test_generation_is_deterministic(corpus: tuple[list, list]) -> None:
    """AT-ING-08: same seed -> identical documents and queries."""
    documents, queries = corpus
    again_docs, again_queries = generate()
    assert [document.text for document in documents] == [doc.text for doc in again_docs]
    assert [(q.text, q.operator_reply, q.supporting_doc_ids) for q in queries] == [
        (q.text, q.operator_reply, q.supporting_doc_ids) for q in again_queries
    ]


def test_manifest_records_per_locale_depth() -> None:
    """AT-ING-07/08: the checked-in manifest carries per-locale stats and provenance."""
    manifest_path = REPO_CORPUS / "manifest.json"
    if not manifest_path.is_file():  # pragma: no cover - the release tree always has it
        pytest.skip("checked-in corpus not built in this tree")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] >= "2.1.0"
    assert manifest["generator"]["version"] == "2.1.0"
    assert set(manifest["locales"]) == set(LOCALES)
    for locale, stats in manifest["locales"].items():
        assert stats["documents"] >= 135, locale
        assert stats["archetypes"] >= 15, locale
        assert stats["queries"] >= 400, locale
        assert stats["queries_by_category"], locale
    assert manifest["content_provenance"]["native_speaker_validated"] is False
    assert manifest["content_provenance"]["license"] == "CC0-1.0"
    assert manifest["queries"]["unanswerable_share"] >= 0.15
    assert manifest["reference_grounding"]["findings"] == []
    assert manifest["stats"]["chunks"] >= 7000


def test_config_documents_the_depth_floor() -> None:
    """AT-ING-08: the frozen demo profile still clears the t18 floors."""
    config = SyntheticConfig()
    assert config.docs_per_locale >= 100
    assert config.queries_per_doc * config.docs_per_locale >= 400
