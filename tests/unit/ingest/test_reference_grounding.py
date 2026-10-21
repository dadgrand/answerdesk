"""Reference grounding: every seed's reference reply must be entailed by its own document.

This is the t15 regression suite. The defect it locks down: ``operator_reply`` used to be
drawn from a *different* RNG than the document, so a reference answer could quote an
amount, a date or an order id that appeared nowhere in the document it cited — making
answer-correctness labels unverifiable and every "answer quality" number downstream
optimistic.

The tests below prove the fix end to end (question -> document -> reference reply) over
all 720 seeds of the checked-in corpus profile, prove that the verifier actually fails on
drift, and prove the corpus builder refuses to write a corpus that fails the gate.

# AT-ING-10  # AT-ING-08
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from answerdesk.ingest.builder import build_demo
from answerdesk.ingest.canonical import canonicalize
from answerdesk.ingest.synthetic import (
    DEMO_SEED,
    PERTURBATION_CLASSES,
    SyntheticConfig,
    generate,
    grounding_report,
    identifier_findings,
    reference_claims,
    verify_reference_grounding,
)

#: Articles per locale of the checked-in demo corpus -> 810 documents / 3,240 queries.
DOCS_PER_LOCALE = SyntheticConfig().docs_per_locale
QUERIES_PER_DOC = SyntheticConfig().queries_per_doc
LOCALES = len(SyntheticConfig().locales)

#: Identifiers as they appear in generated text.
_IDENTIFIER_RE = re.compile(r"ORD-\d{4}-\d{5,6}|ERR-[A-Z]{3}-\d{4}|\b(?:\d{4} ){3}\d{4}\b")
#: Dates as they appear in generated text (locale forms are covered by the generators).
_DATE_RE = re.compile(
    r"20\d{2}年\d{1,2}月\d{1,2}日"
    r"|\b\d{1,2}[./-]\d{1,2}[./-](?:20\d{2}|\d{2})\b"
    r"|\b\d{1,2}[- ](?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|"
    r"декабр|yanvar|fevral|mart|aprel|may|iyun|iyul|avgust|sentabr|oktabr|noyabr|dekabr|"
    r"Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)"
    r"(?: \d{4})?",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def corpus() -> tuple[list, list]:
    """The full demo generation: 810 documents and 3,240 queries (one slot draw each)."""
    return generate()


@pytest.fixture(scope="module")
def texts(corpus: tuple[list, list]) -> dict[str, str]:
    documents, _ = corpus
    return {document.doc_id: document.text for document in documents}


def test_every_seed_reference_is_grounded_in_its_own_document(corpus: tuple[list, list]) -> None:
    """AT-ING-10: the verifier passes 100% over all 3,240 seeds — no exceptions."""
    documents, queries = corpus
    assert len(queries) == DOCS_PER_LOCALE * LOCALES * QUERIES_PER_DOC >= 2_400
    findings = verify_reference_grounding(documents, queries)
    assert findings == [], f"{len(findings)} ungrounded reference claims, first: {findings[:3]}"
    report = grounding_report(documents, queries)
    assert report["checked"] == len(queries)
    assert report["findings"] == []


SINGLE_DOC = {
    "direct",
    "numeric_date",
    "identifier",
    "code_switch",
    "typo",
    "noisy_freeform",
}


def test_reference_reply_is_a_verbatim_document_sentence(
    corpus: tuple[list, list], texts: dict[str, str]
) -> None:
    """AT-ING-10: single-document replies are literally printed in their document."""
    _, queries = corpus
    checked = 0
    for query in queries:
        if query.category not in SINGLE_DOC:
            continue
        document_text = texts[query.doc_id]
        assert query.operator_reply, query.query_id
        assert query.operator_reply in document_text, (
            f"{query.query_id}: reference reply is not a sentence of its own document"
        )
        assert f"A: {query.operator_reply}" in document_text, query.query_id
        checked += 1
    assert checked >= len(queries) * 0.5


class _Doc:
    """Minimal ``DocumentLike`` stand-in for two-document checks."""

    def __init__(self, doc_id: str, text: str) -> None:
        self.doc_id = doc_id
        self.text = text


def test_multi_hop_reply_spans_its_two_supporting_documents(
    corpus: tuple[list, list], texts: dict[str, str]
) -> None:
    """AT-ING-10: a multi-hop answer needs both documents, one grounded sentence each."""
    _, queries = corpus
    multi_hop = [query for query in queries if len(query.supporting_doc_ids) > 1]
    assert multi_hop
    for query in multi_hop:
        docs = [_Doc(doc_id, texts[doc_id]) for doc_id in query.supporting_doc_ids]
        assert verify_reference_grounding(docs, [query]) == []
        reply = query.operator_reply
        for doc in docs:
            windows = [reply[start : start + 12] for start in range(0, max(1, len(reply) - 12), 4)]
            assert any(window in doc.text for window in windows), (
                f"{query.query_id}: {doc.doc_id} contributes no grounded sentence"
            )


def test_unanswerable_queries_carry_no_reference_reply(corpus: tuple[list, list]) -> None:
    """AT-ING-10: unanswerable items are marked and have no reply to ground."""
    _, queries = corpus
    unanswerable = [query for query in queries if not query.answerable]
    share = len(unanswerable) / len(queries)
    assert 0.15 <= share <= 0.21, share
    assert all(query.operator_reply == "" for query in unanswerable)
    assert all(query.category == "unanswerable" for query in unanswerable)


def test_question_of_every_seed_belongs_to_the_same_document(
    corpus: tuple[list, list], texts: dict[str, str]
) -> None:
    """AT-ING-10: the unperturbed question is the document's own FAQ question."""
    _, queries = corpus
    for query in queries:
        if query.category in {"multi_hop", "pii", "injection", "unanswerable"}:
            continue
        assert query.base_text in texts[query.doc_id], query.query_id
        assert f"Q: {query.base_text}" in texts[query.doc_id], query.query_id


def test_verifier_fails_on_a_drifted_reference(corpus: tuple[list, list]) -> None:
    """AT-ING-10: the gate is real — a different amount in the reply is caught."""
    documents, queries = corpus
    target = next(query for query in queries if re.search(r"\d", query.operator_reply))
    drifted_reply = re.sub(
        r"\d+", lambda match: str(int(match.group(0)) + 1), target.operator_reply, count=1
    )
    drifted = replace(target, operator_reply=drifted_reply)
    assert drifted.operator_reply != target.operator_reply, "fixture must change the reply"
    findings = verify_reference_grounding(documents, [drifted])
    assert findings, "a reference reply that disagrees with its document must not pass"
    assert all(finding.query_id == drifted.query_id for finding in findings)
    assert {finding.claim_kind for finding in findings} & {"canonical", "token", "sentence"}


def test_verifier_fails_on_a_missing_document(corpus: tuple[list, list]) -> None:
    """AT-ING-10: a query citing a document that is not in the corpus is a finding."""
    documents, queries = corpus
    victim = queries[0]
    remaining = [document for document in documents if document.doc_id != victim.doc_id]
    findings = verify_reference_grounding(remaining, [victim])
    assert len(findings) == 1
    assert findings[0].claim_kind == "document"
    assert "not in the corpus" in findings[0].reason


def test_reference_claims_are_parsed_into_three_kinds(corpus: tuple[list, list]) -> None:
    """AT-ING-05/10: a reply yields sentence, canonical-value and token claims."""
    _, queries = corpus
    reply = next(
        q.operator_reply
        for q in queries
        if re.search(r"\d[\d ]*[.,]?\d*\s*(?:RUB|TJS|UZS|KGS|TRY|CNY)", q.operator_reply)
    )
    kinds = {claim.kind for claim in reference_claims(reply)}
    assert "sentence" in kinds
    assert "token" in kinds
    assert "canonical" in kinds, "digits in a reply must produce a canonical claim"


def test_canonical_claims_of_every_reply_exist_in_the_document(
    corpus: tuple[list, list], texts: dict[str, str]
) -> None:
    """AT-ING-10: dates/money/ids of a reply come from the same draw as the document."""
    _, queries = corpus
    for query in queries:
        if query.category not in SINGLE_DOC:
            continue
        document_canonical = set(canonicalize(texts[query.doc_id]))
        for value in canonicalize(query.operator_reply):
            assert value in document_canonical or value in texts[query.doc_id], (
                f"{query.query_id}: canonical value {value!r} is not in its document"
            )


def test_injected_identifiers_are_the_document_own_identifiers(
    corpus: tuple[list, list], texts: dict[str, str]
) -> None:
    """AT-ING-10: the identifier perturbation only quotes identifiers of the document."""
    _, queries = corpus
    checked = 0
    for query in queries:
        if query.category != "identifier":
            continue
        assert query.text != query.base_text or _IDENTIFIER_RE.search(query.base_text), (
            f"{query.query_id}: nothing was injected"
        )
        for value in _IDENTIFIER_RE.findall(query.text):
            assert value in texts[query.doc_id], (
                f"{query.query_id}: identifier {value!r} does not occur in the document"
            )
        checked += 1
    assert checked >= len(queries) * 0.05


def test_injected_dates_are_the_document_own_dates(
    corpus: tuple[list, list], texts: dict[str, str]
) -> None:
    """AT-ING-10: the date perturbation only quotes a date the document states."""
    _, queries = corpus
    checked = 0
    for query in queries:
        if query.category != "numeric_date":
            continue
        assert query.text != query.base_text, f"{query.query_id}: nothing was injected"
        for value in _DATE_RE.findall(query.text):
            assert value in texts[query.doc_id], (
                f"{query.query_id}: date {value!r} does not occur in the document"
            )
        checked += 1
    assert checked >= len(queries) * 0.05


def test_every_document_carries_its_reference_block(
    corpus: tuple[list, list], texts: dict[str, str]
) -> None:
    """AT-ING-10: the as-of date, order id and error code are printed in every article.

    Without this block a perturbed question could legitimately mention a value the
    article does not state; with it, every injected identifier has a home in the text.
    """
    documents, _ = corpus
    for document in documents:
        body = texts[document.doc_id]
        assert _IDENTIFIER_RE.search(body), document.doc_id
        assert re.search(r"ERR-[A-Z]{3}-\d{4}", body), document.doc_id
        assert _DATE_RE.search(body), document.doc_id


def test_perturbation_quota_survives_the_alignment(corpus: tuple[list, list]) -> None:
    """AT-ING-10: every perturbation class is still ≥ 5% of the query set."""
    _, queries = corpus
    for klass in PERTURBATION_CLASSES:
        share = sum(1 for query in queries if klass in query.perturbations) / len(queries)
        assert share >= 0.05, f"{klass} share dropped to {share:.3f}"


def test_generation_is_deterministic(corpus: tuple[list, list]) -> None:
    """AT-ING-08: the same seed reproduces documents, questions and reference replies."""
    documents, queries = corpus
    again_documents, again_queries = generate()
    assert [document.text for document in documents] == [doc.text for doc in again_documents]
    assert [(q.text, q.base_text, q.operator_reply) for q in queries] == [
        (q.text, q.base_text, q.operator_reply) for q in again_queries
    ]


def test_reserved_identifiers_still_hold(corpus: tuple[list, list]) -> None:
    """AT-ING-10 (§13.4): the aligned corpus still contains no real identifiers."""
    documents, queries = corpus
    texts = [document.text for document in documents] + [query.text for query in queries]
    assert identifier_findings(texts) == []


def test_build_records_the_grounding_gate_and_keeps_chunk_ids(tmp_path: Path) -> None:
    """AT-ING-07/08/10: the manifest records the gate; chunk ids keep ``::cNNNN``."""
    result = build_demo(tmp_path / "corpus", config=SyntheticConfig(docs_per_locale=2))
    manifest = result.manifest
    assert manifest.reference_grounding["findings"] == []
    assert manifest.reference_grounding["checked"] == manifest.queries["count"] > 0
    assert manifest.queries["reference_grounding"] == "verified"
    assert manifest.generator is not None
    assert manifest.generator.version == "2.1.0"

    chunk_ids = [
        json.loads(line)["chunk_id"]
        for line in Path(result.chunks_path).read_text(encoding="utf-8").splitlines()
    ]
    assert chunk_ids, "the build must produce chunks"
    pattern = re.compile(r"^syn-[a-z]{2}-\d{4}::c\d{4}$")
    assert all(pattern.match(chunk_id) for chunk_id in chunk_ids), chunk_ids[:5]

    queries = [
        json.loads(line) for line in Path(result.queries_path or "").read_text("utf-8").splitlines()
    ]
    document_texts = {
        record.doc_id: (Path(result.corpus_dir) / record.path).read_text(encoding="utf-8")
        for record in manifest.sources
    }
    for query in queries:
        if not query["answerable"] or len(query["supporting_doc_ids"]) > 1:
            continue
        assert query["operator_reply"] in document_texts[query["doc_id"]], query["query_id"]


def test_seed_is_the_frozen_demo_seed() -> None:
    """AT-ING-08: the checked-in corpus profile is the frozen seed (no silent drift)."""
    assert SyntheticConfig().seed == DEMO_SEED
    assert DOCS_PER_LOCALE >= 100


def test_checked_in_corpus_files_are_grounded() -> None:
    """AT-ING-10: the committed artifact itself passes — file level, not just in memory."""
    corpus_dir = Path(__file__).resolve().parents[3] / "data" / "corpus"
    queries_path = corpus_dir / "queries" / "queries.jsonl"
    if not queries_path.is_file():  # pragma: no cover - the release tree always has it
        pytest.skip("checked-in corpus not built in this tree")

    from answerdesk.contracts import Document

    records = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) >= 2_400
    documents = [
        Document(
            doc_id=record["doc_id"],
            title=record["doc_id"],
            source_path=record["doc_id"],
            source_format="md",
            text=(corpus_dir / "docs" / record["locale"] / f"{record['doc_id']}.md").read_text(
                encoding="utf-8"
            ),
            license="CC0-1.0",
            sha256="0" * 64,
            byte_size=1,
        )
        for record in records
    ]
    unique: dict[str, Document] = {document.doc_id: document for document in documents}
    from answerdesk.ingest.synthetic import GeneratedQuery

    queries = [
        GeneratedQuery(
            query_id=record["query_id"],
            locale=record["locale"],
            intent=record["intent"],
            doc_id=record["doc_id"],
            text=record["text"],
            base_text=record["base_text"],
            category=record["category"],
            perturbations=tuple(record["perturbations"]),
            operator_reply=record["operator_reply"],
            archetype=record.get("archetype", ""),
            supporting_doc_ids=tuple(record.get("supporting_doc_ids", ())),
            answerable=record.get("answerable", True),
        )
        for record in records
    ]
    assert verify_reference_grounding(list(unique.values()), queries) == []
