"""Noisy free-form queries: the owner's reported cases and the malformed-input contract (t22).

Real support traffic is not clean FAQ text. This suite locks the ``noisy_freeform`` slice:
telegraphic style, run-on sentences without punctuation, dropped prepositions,
typo/transliteration mixes — each with an intent label and either a grounded reference
reply or an explicit unanswerable flag. The owner's exact Russian examples and their
translations must be reproducible byte for byte, because t23's robustness ablation is
measured on this slice.

# AT-ING-10  # AT-ING-11
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from answerdesk.ingest.noisy import NOISY_STYLES, OWNER_EXAMPLES, PREPOSITIONS, examples_for
from answerdesk.ingest.synthetic import (
    LOCALES,
    SyntheticConfig,
    generate,
    verify_reference_grounding,
)

REPO_CORPUS = Path(__file__).resolve().parents[3] / "data" / "corpus"

#: Sentence punctuation that a telegraphic or run-on message must not contain.
PUNCTUATION = re.compile(r"[.,!?;:…。！？；：]")


@pytest.fixture(scope="module")
def corpus() -> tuple[list, list]:
    return generate()


@pytest.fixture(scope="module")
def noisy(corpus: tuple[list, list]) -> list:
    _, queries = corpus
    return [query for query in queries if query.category == "noisy_freeform"]


def test_noisy_slice_meets_the_size_floor(noisy: list) -> None:
    """AT-ING-11: >=300 noisy items, >=40 per locale, all six locales represented."""
    assert len(noisy) >= 300, len(noisy)
    per_locale = Counter(query.locale for query in noisy)
    assert set(per_locale) == set(LOCALES)
    for locale, count in per_locale.items():
        assert count >= 40, f"{locale}: {count}"


def test_every_noisy_style_is_present(noisy: list) -> None:
    """AT-ING-11: telegraphic, run-on, dropped preposition, typo mix and transliteration."""
    styles = Counter(query.style for query in noisy)
    for style in NOISY_STYLES:
        assert styles.get(style, 0) > 0, f"style {style!r} missing"


def test_owner_russian_examples_are_verbatim(noisy: list) -> None:
    """AT-ING-11: the three reported Russian messages appear exactly, once each."""
    russian = [query.text for query in noisy if query.locale == "ru"]
    for text in (
        "как займ оформит",
        "деньги как получит",
        "денги первод таджикистан сделал денги нет брату не дошли",
    ):
        assert text in russian, f"owner example missing verbatim: {text!r}"
        assert russian.count(text) == 1, f"owner example duplicated: {text!r}"


def test_owner_examples_have_translations_in_every_locale(noisy: list) -> None:
    """AT-ING-11: uz/tg/ky/zh/tr carry a natural noisy translation of each case."""
    for locale in LOCALES:
        expected = examples_for(locale)
        assert expected, locale
        texts = [query.text for query in noisy if query.locale == locale]
        for example in expected:
            assert example.text in texts, f"{locale}: {example.text!r}"


def test_transliteration_variants_are_present(noisy: list) -> None:
    """AT-ING-11: Latin-keyboard transliterations of the reported cases are in the slice."""
    transliterated = {query.text for query in noisy if query.style == "translit"}
    assert {"zaym kak oformit", "dengi kak poluchit"} <= transliterated
    assert any("perevod" in text and "tadzhikistan" in text for text in transliterated)


def test_noisy_items_carry_an_intent_label(noisy: list) -> None:
    """AT-ING-11: every noisy item names the archetype its answer lives in."""
    for query in noisy:
        assert query.archetype, query.query_id
        assert query.intent, query.query_id
        assert query.archetype == query.intent, query.query_id


def test_noisy_items_are_grounded_or_explicitly_unanswerable(
    corpus: tuple[list, list], noisy: list
) -> None:
    """AT-ING-11: no noisy item is both unanswered and unlabelled."""
    documents, _ = corpus
    by_id = {document.doc_id: document for document in documents}
    for query in noisy:
        if not query.answerable:
            assert query.operator_reply == "", query.query_id
            continue
        assert query.operator_reply, query.query_id
        sources = [by_id[doc_id] for doc_id in query.supporting_doc_ids]
        assert verify_reference_grounding(sources, [query]) == [], query.query_id


def test_telegraphic_items_have_no_sentence_punctuation(noisy: list) -> None:
    """AT-ING-11: telegraphic style drops punctuation and filler words."""
    items = [query for query in noisy if query.style == "telegraphic" and query.locale != "ru"]
    assert items
    for query in items:
        assert not PUNCTUATION.search(query.text), query.text
        assert query.text == query.text.lower() or query.locale == "zh", query.text


def test_run_on_items_are_unpunctuated_and_long(noisy: list) -> None:
    """AT-ING-11: run-on messages glue clauses without punctuation."""
    items = [query for query in noisy if query.style == "run_on" and query.locale != "ru"]
    assert items
    for query in items:
        assert not PUNCTUATION.search(query.text), query.text
        assert len(query.text.split()) >= 4, query.text


def test_dropped_preposition_items_lose_their_prepositions(noisy: list) -> None:
    """AT-ING-11: the dropped-preposition style really removes short prepositions."""
    checked = 0
    for query in noisy:
        if query.style != "dropped_preposition":
            continue
        words = {word.casefold() for word in re.findall(r"\w+", query.text)}
        removable = {word.casefold() for word in PREPOSITIONS.get(query.locale, ())}
        assert not (words & removable), f"{query.query_id}: preposition survived: {query.text}"
        checked += 1
    assert checked >= 10


def test_typo_and_transliteration_mix_change_the_surface_form(noisy: list) -> None:
    """AT-ING-11: typo/transliteration items differ from their canonical question."""
    items = [query for query in noisy if query.style == "typo_mix"]
    assert items
    for query in items:
        assert query.text != query.base_text, query.query_id


def test_noisy_slice_is_deterministic(noisy: list) -> None:
    """AT-ING-08/11: the same seed reproduces the noisy slice byte for byte."""
    _, again = generate()
    again_noisy = [query for query in again if query.category == "noisy_freeform"]
    assert [(q.text, q.operator_reply, q.doc_id, q.style) for q in noisy] == [
        (q.text, q.operator_reply, q.doc_id, q.style) for q in again_noisy
    ]


def test_manifest_records_the_noisy_slice() -> None:
    """AT-ING-07/11: the checked-in manifest counts the noisy slice and its styles."""
    manifest_path = REPO_CORPUS / "manifest.json"
    if not manifest_path.is_file():  # pragma: no cover - the release tree always has it
        pytest.skip("checked-in corpus not built in this tree")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    categories = manifest["queries"]["categories"]
    assert categories["noisy_freeform"] >= 300, categories
    styles = manifest["queries"]["noisy_styles"]
    for style in NOISY_STYLES:
        assert styles.get(style, 0) > 0, styles
    assert manifest["schema_version"] >= "2.1.0"
    assert manifest["generator"]["version"] == "2.1.0"
    assert manifest["locales"]["ru"]["archetypes"] >= 15
    assert manifest["reference_grounding"]["findings"] == []


def test_owner_examples_are_present_in_the_checked_in_artifact() -> None:
    """AT-ING-11: the committed queries file contains the reported cases verbatim."""
    queries_path = REPO_CORPUS / "queries" / "queries.jsonl"
    if not queries_path.is_file():  # pragma: no cover
        pytest.skip("checked-in corpus not built in this tree")
    records = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines()]
    russian = {record["text"] for record in records if record["locale"] == "ru"}
    for text in (
        "как займ оформит",
        "деньги как получит",
        "денги первод таджикистан сделал денги нет брату не дошли",
    ):
        assert text in russian, text
    noisy = [record for record in records if record["category"] == "noisy_freeform"]
    assert len(noisy) >= 300
    assert all(record["archetype"] and record["style"] for record in noisy)


def test_config_defaults_keep_the_slice_above_the_floor() -> None:
    """AT-ING-11: the frozen profile still clears the noisy floor."""
    config = SyntheticConfig()
    total_queries = config.docs_per_locale * config.queries_per_doc * len(config.locales)
    assert total_queries >= 3_000
    assert OWNER_EXAMPLES, "the owner's reported cases must stay in the repository"
