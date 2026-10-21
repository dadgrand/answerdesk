"""Near-duplicate removal tests: determinism, threshold, exact duplicates (FR-ING-06).

# AT-ING-06
"""

from __future__ import annotations

import pytest

from answerdesk.ingest.dedup import (
    DedupConfig,
    NearDuplicateDetector,
    jaccard,
    minhash_signature,
    normalise_for_shingles,
    permutations,
    shingles_of,
    text_hash,
)

BASE = (
    "Возврат обрабатывается в течение 14 дней после одобрения заявки. "
    "Комиссия за перевод составляет 2,5 EUR. Лимит на снятие наличных — 25 000 RUB в сутки. "
    "Если карта утеряна, заблокируйте её в приложении и закажите перевыпуск."
)
#: A page-sized article: the realistic unit of deduplication (a 4-line snippet is
#: proportionally too short for a 5-gram Jaccard threshold to be meaningful).
LONG = " ".join(
    [
        BASE,
        "Справка о движении средств готовится три рабочих дня и доступна в приложении.",
        "Для международного перевода нужны реквизиты получателя и подтверждение суммы.",
        "Проверка личности занимает до шести часов, в редких случаях — до трёх дней.",
        "Смена тарифа возможна не чаще одного раза в тридцать дней и подтверждается SMS.",
        "Готовый документ можно забрать в отделении в рабочие дни до восемнадцати часов.",
    ]
)
#: A near-copy of :data:`LONG`: one number changed (the realistic duplicate: a re-export).
NEAR = LONG.replace("14 дней", "15 дней")
OTHER = (
    "Проверка личности занимает до 6 часов. Подготовьте фото паспорта и селфи. "
    "Адрес подтверждается документом не старше трёх месяцев. "
    "При отказе подайте заявку повторно."
)


def test_identical_texts_are_removed() -> None:
    """AT-ING-06: an exact duplicate is dropped and the survivor is the id-sorted first."""
    detector = NearDuplicateDetector()
    result = detector.dedupe([("b", BASE), ("a", BASE)])
    assert result.kept == ["a"]
    assert [pair.dropped_id for pair in result.dropped] == ["b"]
    assert result.dropped[0].exact_text_match is True
    assert result.dropped[0].kept_id == "a"


def test_near_duplicate_above_threshold_is_removed() -> None:
    """AT-ING-06: a lightly edited copy (one number) is still a duplicate at 0.80."""
    detector = NearDuplicateDetector(DedupConfig(threshold=0.80))
    result = detector.dedupe([("doc-a", LONG), ("doc-b", NEAR), ("doc-c", OTHER)])
    dropped = {pair.dropped_id for pair in result.dropped}
    assert dropped == {"doc-b"}
    assert result.kept == ["doc-a", "doc-c"]
    assert result.dropped[0].similarity >= 0.80


def test_distinct_texts_survive() -> None:
    """AT-ING-06: unrelated documents are kept — dedup must not eat the corpus."""
    detector = NearDuplicateDetector()
    result = detector.dedupe([("a", BASE), ("b", OTHER)])
    assert result.kept == ["a", "b"]
    assert result.dropped == []


def test_threshold_is_the_deciding_parameter() -> None:
    """AT-ING-06: raising the threshold keeps a pair the default would remove."""
    similarity = NearDuplicateDetector().similarity(LONG, NEAR)
    assert 0.5 < similarity < 0.95
    strict = NearDuplicateDetector(DedupConfig(threshold=min(0.99, similarity + 0.01)))
    assert strict.dedupe([("a", LONG), ("b", NEAR)]).kept == ["a", "b"]
    loose = NearDuplicateDetector(DedupConfig(threshold=max(0.5, similarity - 0.01)))
    assert loose.dedupe([("a", LONG), ("b", NEAR)]).kept == ["a"]


def test_survivors_do_not_depend_on_input_order() -> None:
    """AT-ING-06: the survivor *set* is order-independent (kept ids keep input order)."""
    items = [("doc-b", NEAR), ("doc-a", LONG), ("doc-c", OTHER), ("doc-d", LONG)]
    first = NearDuplicateDetector().dedupe(items)
    second = NearDuplicateDetector().dedupe(list(reversed(items)))
    assert set(first.kept) == set(second.kept) == {"doc-a", "doc-c"}
    assert {pair.dropped_id for pair in first.dropped} == {"doc-b", "doc-d"}
    assert {pair.dropped_id for pair in second.dropped} == {"doc-b", "doc-d"}


def test_two_runs_are_byte_identical() -> None:
    """AT-ING-06: the whole result object is deterministic (it is recorded in the manifest)."""
    items = [("doc-a", BASE), ("doc-b", NEAR), ("doc-c", OTHER)]
    first = NearDuplicateDetector().dedupe(items)
    second = NearDuplicateDetector().dedupe(items)
    assert first.model_dump() == second.model_dump()


def test_empty_and_short_inputs_are_safe() -> None:
    """AT-ING-06: no documents, empty text and one-word documents do not crash."""
    detector = NearDuplicateDetector()
    assert detector.dedupe([]).kept == []
    assert detector.dedupe([("a", "")]).kept == ["a"]
    assert detector.dedupe([("a", "возврат"), ("b", "возврат")]).kept == ["a"]


def test_signature_and_jaccard_helpers() -> None:
    """AT-ING-06: the MinHash estimate tracks the exact Jaccard within estimator error."""
    detector = NearDuplicateDetector(DedupConfig(num_perm=256))
    left = detector.shingles(BASE)
    right = detector.shingles(NEAR)
    exact = jaccard(left, right)
    perms = permutations(DedupConfig(num_perm=256))
    estimate = (
        sum(
            a == b
            for a, b in zip(
                minhash_signature(left, perms), minhash_signature(right, perms), strict=True
            )
        )
        / 256
    )
    assert abs(estimate - exact) < 0.12, f"estimate {estimate} vs exact {exact}"


def test_shingles_are_case_and_space_insensitive() -> None:
    """AT-ING-06: normalisation is NFKC + casefold + whitespace collapse, nothing else."""
    assert normalise_for_shingles("  Возврат   Дней ") == "возврат дней"
    assert shingles_of("a b c d e") == shingles_of("A  B  C  D  E")
    assert text_hash("A  B") == text_hash("a b")


def test_config_validation() -> None:
    """AT-ING-06: an incoherent dedup configuration is rejected, not silently defaulted."""
    with pytest.raises(ValueError):
        DedupConfig(threshold=1.5)
    with pytest.raises(ValueError):
        DedupConfig(num_perm=30, bands=16)
