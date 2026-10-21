"""Dataset tests: reproducibility, composition, loud validation (AT-EVL-01/02/03).

# AT-EVL-01  # AT-EVL-02  # AT-EVL-03  # AT-DET-02  # AT-ML-01
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fixtures import tiny_config, write_tiny_corpus

from answerdesk.contracts import EvalCase, EvalCategory, Split
from answerdesk.eval.corpus import Corpus, load_corpus
from answerdesk.eval.datasets import (
    allocate,
    faq_question_index,
    generate_golden,
    golden_hash,
    label_evidence,
    load_golden,
    question_stem,
    reference_is_verifiable,
    validate_cases,
    write_golden,
)
from answerdesk.eval.errors import (
    CorpusUnavailableError,
    EvalDataError,
    MissingLabelError,
    SplitLeakageError,
    UnknownChunkError,
)
from answerdesk.eval.splits import build_split, build_stratified_split, read_split, split_hash


def test_generation_is_reproducible_and_seeded(tmp_path: Path) -> None:
    corpus = load_corpus(write_tiny_corpus(tmp_path / "a"))
    first = generate_golden(corpus, tiny_config(), strict=False)
    second = generate_golden(corpus, tiny_config(), strict=False)
    assert golden_hash(first.cases) == golden_hash(second.cases)
    assert first.split.sha256 == second.split.sha256
    other = generate_golden(corpus, tiny_config(seed=99), strict=False)
    assert golden_hash(first.cases) != golden_hash(other.cases)


def test_label_classes_are_separate(tiny_cases: list[EvalCase]) -> None:
    answerable = [case for case in tiny_cases if case.answerable]
    unanswerable = [case for case in tiny_cases if not case.answerable]
    assert answerable and unanswerable
    for case in answerable:
        assert case.gold_chunk_ids, "retrieval labels are mandatory for an answerable case"
        assert case.reference_answer, "answer labels are mandatory for an answerable case"
        assert case.required_citation_chunk_ids
    for case in unanswerable:
        assert case.reference_answer is None
        assert case.metadata.get("must_escalate") is True


def test_categories_and_locales_cover_the_requirement(tiny_cases: list[EvalCase]) -> None:
    categories = {case.category for case in tiny_cases}
    assert {
        EvalCategory.DIRECT,
        EvalCategory.NUMERIC_DATE,
        EvalCategory.UNANSWERABLE,
        EvalCategory.PII,
        EvalCategory.INJECTION,
    } <= categories
    assert {case.locale for case in tiny_cases} == {"ru", "tg", "uz", "ky", "tr", "zh"}
    for case in tiny_cases:
        assert case.split in {Split.DEV, Split.TEST}


def test_every_split_honours_document_disjointness(tiny_golden: object) -> None:
    golden = tiny_golden
    dev_docs = set(golden.split.dev_doc_ids)  # type: ignore[attr-defined]
    test_docs = set(golden.split.test_doc_ids)  # type: ignore[attr-defined]
    assert not dev_docs & test_docs
    for case in golden.cases:  # type: ignore[attr-defined]
        document = case.provenance.document_ids[0]
        if case.split is Split.TEST:
            assert document in test_docs
        else:
            assert document in dev_docs


def test_balanced_split_feeds_the_quotas(tiny_corpus: Corpus) -> None:
    manifest = build_stratified_split(
        tiny_corpus,
        seed=424242,
        corpus_version=tiny_corpus.version,
        test_ratio=0.5,
        quotas={"direct": 3, "numeric_date": 3, "typo_code_switch": 1},
    )
    assert manifest.test_doc_ids and manifest.dev_doc_ids
    assert not set(manifest.test_doc_ids) & set(manifest.dev_doc_ids)
    # every numeric-needing document is in the test half because the need is 3 and the
    # fixture has one numeric query per document
    numeric_docs = {
        doc
        for doc in manifest.test_doc_ids
        if any(query.category == "numeric_date" for query in tiny_corpus.queries_for_doc(doc))
    }
    assert len(numeric_docs) >= 3


def test_split_manifest_hash_round_trip(tmp_path: Path, tiny_corpus: Corpus) -> None:
    manifest = build_split(
        tiny_corpus.document_ids, seed=7, corpus_version=tiny_corpus.version, test_ratio=0.4
    )
    assert manifest.sha256 == split_hash(manifest)
    from answerdesk.eval.splits import write_split

    write_split(manifest, tmp_path / "splits" / "v1.json")
    assert read_split(tmp_path / "splits" / "v1.json").sha256 == manifest.sha256


def test_edited_split_manifest_is_rejected(tmp_path: Path, tiny_corpus: Corpus) -> None:
    manifest = build_split(
        tiny_corpus.document_ids, seed=7, corpus_version=tiny_corpus.version, test_ratio=0.4
    )
    from answerdesk.eval.splits import write_split

    path = tmp_path / "splits" / "v1.json"
    write_split(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["test_doc_ids"] = payload["dev_doc_ids"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvalDataError):
        read_split(path)


def test_strict_generation_fails_loudly_on_a_tiny_corpus(tiny_corpus: Corpus) -> None:
    """§10.1 composition is a hard requirement; a corpus that cannot meet it must fail."""
    with pytest.raises(EvalDataError) as excinfo:
        generate_golden(tiny_corpus, tiny_config(), strict=True)
    assert "composition" in str(excinfo.value)


def test_allocate_uses_largest_remainder() -> None:
    counts = allocate(60, {"a": 0.35, "b": 0.20, "c": 0.15, "d": 0.18, "e": 0.06, "f": 0.06})
    assert sum(counts.values()) == 60
    assert counts["a"] == 21
    assert counts["b"] == 12
    assert counts["c"] == 9


def test_question_stem_masks_digits() -> None:
    assert question_stem("Сколько стоит тариф 2026?") == question_stem("Сколько стоит тариф 1999?")


def test_equivalence_class_widens_the_gold_set(tiny_corpus: Corpus) -> None:
    """Same question template in several documents => several acceptable evidence chunks."""
    index = faq_question_index(tiny_corpus)
    label = label_evidence(
        tiny_corpus,
        doc_id="doc-ru-00",
        question="Как заблокировать карту? #0",
        equivalence_index=index,
    )
    assert len(label.gold) > len(label.primary)
    assert set(label.primary) <= set(label.gold)
    assert label.rule == "structural-faq-equivalence"


def test_reference_verifiability_flags_slot_mismatch(tiny_corpus: Corpus) -> None:
    assert reference_is_verifiable(tiny_corpus, ["doc-ru-00::c0000"], "Статус: ORD-2026-10001.")
    assert not reference_is_verifiable(tiny_corpus, ["doc-ru-00::c0000"], "Статус: ORD-2026-99999.")
    assert not reference_is_verifiable(tiny_corpus, ["doc-ru-00::c0000"], None)


def test_write_and_load_golden_round_trip(tmp_path: Path, tiny_corpus: Corpus) -> None:
    golden = generate_golden(tiny_corpus, tiny_config(), strict=False)
    manifest = write_golden(golden, tiny_corpus, tmp_path / "evals")
    cases, loaded_manifest, split = load_golden(
        tmp_path / "evals", corpus=tiny_corpus, strict=False
    )
    assert loaded_manifest.version == manifest.version
    assert loaded_manifest.split_hash == golden.split.sha256
    assert split.sha256 == golden.split.sha256
    assert len(cases) == len(
        [case for case in golden.cases if case.split in {Split.TEST, Split.DEV}]
    )
    for relative in ("test.jsonl", "dev.jsonl", "manifest.json"):
        assert (tmp_path / "evals" / "golden" / "v1" / relative).is_file()
    for name in ("near_miss", "claims", "injection", "pii", "perturbation"):
        assert (tmp_path / "evals" / "golden" / "v1" / "adversarial" / f"{name}.jsonl").is_file()


def test_missing_labels_fail_loudly(tiny_golden: object, tiny_corpus: Corpus) -> None:
    cases = list(tiny_golden.cases)  # type: ignore[attr-defined]
    index = next(i for i, case in enumerate(cases) if case.answerable)
    cases[index] = cases[index].model_copy(update={"reference_answer": None})
    with pytest.raises(MissingLabelError) as excinfo:
        validate_cases(
            cases,
            corpus=tiny_corpus,
            split=tiny_golden.split,  # type: ignore[attr-defined]
        )
    assert cases[index].case_id in excinfo.value.case_ids


def test_unknown_gold_chunk_fails_loudly(tiny_golden: object, tiny_corpus: Corpus) -> None:
    cases = list(tiny_golden.cases)  # type: ignore[attr-defined]
    index = next(i for i, case in enumerate(cases) if case.answerable)
    cases[index] = cases[index].model_copy(
        update={"gold_chunk_ids": ["not-a-chunk"], "required_citation_chunk_ids": ["not-a-chunk"]}
    )
    with pytest.raises(UnknownChunkError):
        validate_cases(cases, corpus=tiny_corpus, split=tiny_golden.split)  # type: ignore[attr-defined]


def test_split_leakage_fails_loudly(tiny_golden: object, tiny_corpus: Corpus) -> None:
    cases = list(tiny_golden.cases)  # type: ignore[attr-defined]
    dev_doc = tiny_golden.split.dev_doc_ids[0]  # type: ignore[attr-defined]
    index = next(i for i, case in enumerate(cases) if case.split is Split.TEST)
    leaked = cases[index].model_copy(
        update={
            "provenance": cases[index].provenance.model_copy(update={"document_ids": [dev_doc]})
        }
    )
    cases[index] = leaked
    with pytest.raises(SplitLeakageError) as excinfo:
        validate_cases(cases, corpus=tiny_corpus, split=tiny_golden.split)  # type: ignore[attr-defined]
    assert dev_doc in excinfo.value.document_ids


def test_missing_corpus_artifacts_are_actionable(tmp_path: Path) -> None:
    with pytest.raises(CorpusUnavailableError) as excinfo:
        load_corpus(tmp_path / "nothing")
    assert "ingest" in str(excinfo.value)


def test_duplicate_case_ids_fail(tiny_golden: object, tiny_corpus: Corpus) -> None:
    cases = list(tiny_golden.cases)  # type: ignore[attr-defined]
    cases.append(cases[0])
    with pytest.raises(EvalDataError):
        validate_cases(cases, corpus=tiny_corpus, split=tiny_golden.split)  # type: ignore[attr-defined]


def test_load_golden_rejects_a_changed_corpus(tmp_path: Path, tiny_corpus: Corpus) -> None:
    golden = generate_golden(tiny_corpus, tiny_config(), strict=False)
    write_golden(golden, tiny_corpus, tmp_path / "evals")
    # Touch the chunk file: the content hash no longer matches the frozen manifest.
    chunks = tmp_path / "corpus" / "chunks.jsonl"
    chunks.write_text(chunks.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rebuilt = load_corpus(tmp_path / "corpus")
    with pytest.raises(EvalDataError) as excinfo:
        load_golden(tmp_path / "evals", corpus=rebuilt)
    assert "corpus" in str(excinfo.value).lower()


def test_load_golden_without_artifacts_is_actionable(tmp_path: Path, tiny_corpus: Corpus) -> None:
    with pytest.raises(EvalDataError) as excinfo:
        load_golden(tmp_path / "evals", corpus=tiny_corpus)
    assert "regenerate" in str(excinfo.value)
