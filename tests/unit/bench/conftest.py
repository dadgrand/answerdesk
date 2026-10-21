"""Fixtures for the ``answerdesk-bench`` tests.

Two kinds of data are available here:

* the **committed release** (``data/bench/answerdesk-bench/v0.1``) as parsed items/passages,
  used by the schema, count, licence, dedup and leakage tests;
* a **miniature build input** written to ``tmp_path`` — a tiny Belebele-like cache, a tiny
  Mr. TyDi pool and a tiny synthetic corpus — used by the determinism and negative-verify
  tests. It exists so those tests never need the 1.6 GB download and never touch the
  network (``tests/conftest.py`` blocks sockets globally).

The scripts package is imported directly from ``scripts/`` because it is build tooling, not
part of the installed ``answerdesk`` package.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

BENCH_DIR = REPO_ROOT / "data" / "bench" / "answerdesk-bench" / "v0.1"

from bench.checks import read_bench  # noqa: E402
from bench.config import (  # noqa: E402
    BELEBELE_LANGS,
    DOMAIN_QUOTAS,
    LICENSE_FILES,
)
from bench.build import SamplingLimits, assemble  # noqa: E402
from bench.templates import PHONE_RANGES, RESERVED_EMAIL  # noqa: E402

MINI_LIMITS = SamplingLimits(belebele_test=2, belebele_dev=1, domain_quotas={key: 1 for key in DOMAIN_QUOTAS})


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture(scope="session")
def bench_dir() -> Path:
    return BENCH_DIR


@pytest.fixture(scope="session")
def manifest(bench_dir: Path) -> dict[str, Any]:
    return json.loads((bench_dir / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def data_files(bench_dir: Path, manifest: dict[str, Any]) -> dict[str, bytes]:
    return {row["path"]: (bench_dir / row["path"]).read_bytes() for row in manifest["files"]}


@pytest.fixture(scope="session")
def bench(data_files: dict[str, bytes]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    return read_bench(data_files)


@pytest.fixture(scope="session")
def items(bench: tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    return bench[0]


@pytest.fixture(scope="session")
def passages(bench: tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, Any]]:
    return bench[1]


# --------------------------------------------------------------------------- mini build


def _write_belebele(cache: Path, language: str, variant: str) -> dict[str, Any]:
    rows = []
    for index in range(20):
        passage = index // 2
        rows.append(
            {
                "question_number": index + 1,
                "flores_passage": f"Passage number {passage} in {variant} with enough words to look like text.",
                "question": f"Question number {index + 1} for {language}?",
                "mc_answer1": f"alpha {index}",
                "mc_answer2": f"beta {index}",
                "mc_answer3": f"gamma {index}",
                "mc_answer4": f"delta {index}",
                "correct_answer_num": str((index % 4) + 1),
                "link": f"https://example.org/{variant}/{passage}",
                "dialect": variant,
                "ds": "2023-05-20",
            }
        )
    payload = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")
    relative = f"raw/belebele/{variant}.jsonl"
    (cache / relative).parent.mkdir(parents=True, exist_ok=True)
    (cache / relative).write_bytes(payload)
    return {
        "cache_path": relative,
        "url": f"https://example.invalid/{variant}.jsonl",
        "published_path": f"data/{variant}.jsonl",
        "language": language,
        "sha256": _sha(payload),
        "bytes": len(payload),
    }


def _write_mrtydi(cache: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    passages = [
        {
            "docid": f"d{index}",
            "title": f"Title {index}",
            "text": f"Passage {index} text with several words for retrieval {index}.",
        }
        for index in range(4)
    ]
    pool_path = cache / "derived/mrtydi-ru/passages.jsonl"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in passages) + "\n", encoding="utf-8"
    )
    files = []
    for split, indexes in (("dev", [0, 1]), ("test", [2, 3])):
        rows = [
            {
                "query_id": str(index),
                "query": f"native question {split} {index}",
                "positive_passages": [{"docid": f"d{index}"}],
            }
            for index in indexes
        ]
        payload = gzip.compress(("\n".join(json.dumps(row) for row in rows) + "\n").encode("utf-8"))
        relative = f"raw/mrtydi/{split}.ru.jsonl.gz"
        (cache / relative).parent.mkdir(parents=True, exist_ok=True)
        (cache / relative).write_bytes(payload)
        files.append(
            {
                "cache_path": relative,
                "url": f"https://example.invalid/{split}.jsonl.gz",
                "published_path": f"mrtydi-v1.1-russian/{split}.jsonl.gz",
                "language": split,
                "sha256": _sha(payload),
                "bytes": len(payload),
            }
        )
    corpus_payload = b"not a real archive"
    corpus_relative = "raw/mrtydi/corpus.ru.jsonl.gz"
    (cache / corpus_relative).write_bytes(corpus_payload)
    corpus = {
        "cache_path": corpus_relative,
        "url": "https://example.invalid/corpus.jsonl.gz",
        "published_path": "mrtydi-v1.1-russian/corpus.jsonl.gz",
        "language": "ru",
        "sha256": _sha(corpus_payload),
        "bytes": len(corpus_payload),
    }
    return files, corpus


def _write_corpus(corpus: Path) -> dict[str, Any]:
    (corpus / "queries").mkdir(parents=True, exist_ok=True)
    documents: dict[str, dict[str, str]] = {}
    queries: list[dict[str, Any]] = []
    for position, language in enumerate(("ru", "uz", "tg", "ky", "zh", "tr")):
        phone = PHONE_RANGES[language]
        for doc_index in range(6):
            doc_id = f"syn-{language}-{doc_index:04d}"
            answer_one = f"First answer for {language} article {doc_index}."
            answer_two = f"Second answer for {language} article {doc_index}."
            text = (
                f"# Article {doc_id}\n\n"
                f"## FAQ\n\nQ: First question {language} {doc_index}?\nA: {answer_one}\n\n"
                f"Q: Second question {language} {doc_index}?\nA: {answer_two}\n\n"
                f"## Contacts\n\nPhone: {phone}.\nMail: {RESERVED_EMAIL}.\n"
                f"Order number: ORD-2026-{10000 + position * 10 + doc_index}.\n"
            )
            path = corpus / "docs" / language / f"{doc_id}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            documents[doc_id] = text
            if doc_index < 2:
                queries.append(
                    {
                        "query_id": f"{doc_id}-q{doc_index}",
                        "locale": language,
                        "doc_id": doc_id,
                        "supporting_doc_ids": [doc_id],
                        "category": "direct" if doc_index == 0 else "numeric_date",
                        "answerable": True,
                        "text": f"Question about article {doc_id}?",
                        "operator_reply": answer_one,
                        "intents": "faq",
                    }
                )
        for variant in range(4):
            queries.append(
                {
                    "query_id": f"{language}-unanswerable-{variant}",
                    "locale": language,
                    "doc_id": f"syn-{language}-0000",
                    "supporting_doc_ids": [f"syn-{language}-0000"],
                    "category": "unanswerable",
                    "answerable": False,
                    "text": f"Completely out of domain question {language} {variant}?",
                    "operator_reply": "",
                }
            )
    (corpus / "queries" / "queries.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in queries) + "\n", encoding="utf-8"
    )
    manifest = {
        "generator": {"name": "answerdesk-synthetic", "version": "2.0.0"},
        "corpus_version": "corpus-test",
        "license": "CC0-1.0",
    }
    (corpus / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


@pytest.fixture(scope="session")
def mini_build(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A tiny but complete build input: cache, corpus, golden split, lock and limits."""
    root = tmp_path_factory.mktemp("mini-bench")
    cache = root / "cache"
    corpus = root / "corpus"
    golden = root / "golden"
    corpus_manifest = _write_corpus(corpus)
    belebele_files = [_write_belebele(cache, language, variant) for language, variant in BELEBELE_LANGS.items()]
    mrtydi_files, mrtydi_corpus = _write_mrtydi(cache)
    license_entries = []
    for cache_relative, _release_relative in LICENSE_FILES:
        payload = f"licence text for {cache_relative}\n".encode()
        (cache / cache_relative).parent.mkdir(parents=True, exist_ok=True)
        (cache / cache_relative).write_bytes(payload)
        license_entries.append(
            {
                "cache_path": cache_relative,
                "url": f"https://example.invalid/{Path(cache_relative).name}",
                "published_path": "",
                "language": "",
                "sha256": _sha(payload),
                "bytes": len(payload),
            }
        )

    (golden / "v1").mkdir(parents=True, exist_ok=True)
    used_doc = "syn-ru-0005"
    (golden / "v1" / "test.jsonl").write_text(
        json.dumps({"case_id": "v1-ru-test-001", "question": "golden question", "metadata": {"doc_id": used_doc}}) + "\n",
        encoding="utf-8",
    )
    splits = {"dev_doc_ids": [], "test_doc_ids": [used_doc]}
    (golden / "splits.json").write_text(json.dumps(splits), encoding="utf-8")

    queries_payload = (corpus / "queries" / "queries.jsonl").read_bytes()
    corpus_manifest_payload = (corpus / "manifest.json").read_bytes()
    lock = {
        "benchmark": "answerdesk-bench",
        "version": "0.1.0",
        "sources": [
            {
                "id": "belebele",
                "dataset": "facebook/belebele",
                "version": "fixture-sha",
                "license": "CC-BY-SA-4.0",
                "files": belebele_files,
            },
            {
                "id": "mrtydi",
                "dataset": "castorini/mr-tydi",
                "version": "fixture-sha",
                "license": "Apache-2.0",
                "files": mrtydi_files,
            },
            {
                "id": "mrtydi-corpus",
                "dataset": "castorini/mr-tydi-corpus",
                "version": "fixture-sha",
                "license": "Apache-2.0",
                "files": [mrtydi_corpus],
            },
            {
                "id": "license-texts",
                "dataset": "license-texts",
                "version": "fixture",
                "license": "n/a",
                "files": license_entries,
            },
        ],
        "local_sources": [
            {
                "id": "answerdesk-synthetic",
                "dataset": "answerdesk-synthetic",
                "version": f"answerdesk-synthetic-v{corpus_manifest['generator']['version']}",
                "license": "CC0-1.0",
                "files": [
                    {
                        "repo_path": "data/corpus/queries/queries.jsonl",
                        "sha256": _sha(queries_payload),
                        "bytes": len(queries_payload),
                    },
                    {
                        "repo_path": "data/corpus/manifest.json",
                        "sha256": _sha(corpus_manifest_payload),
                        "bytes": len(corpus_manifest_payload),
                    },
                ],
            }
        ],
    }
    lock_path = root / "sources.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    yield {
        "root": root,
        "cache": cache,
        "corpus": corpus,
        "golden": golden / "v1",
        "splits": golden / "splits.json",
        "lock": lock_path,
        "limits": MINI_LIMITS,
        "used_doc": used_doc,
    }


@pytest.fixture(scope="session")
def mini_outputs(mini_build: dict[str, Any]) -> dict[str, Any]:
    """Build the mini release twice into separate directories."""
    outputs = {}
    for name in ("first", "second"):
        out_dir = mini_build["root"] / name
        files, manifest = assemble(
            cache_dir=mini_build["cache"],
            corpus_dir=mini_build["corpus"],
            golden_dir=mini_build["golden"],
            splits_file=mini_build["splits"],
            lock_path=mini_build["lock"],
            out_dir=out_dir,
            lax_cache=True,
            write=True,
            enforce_checks=False,
            limits=mini_build["limits"],
        )
        outputs[name] = {"out_dir": out_dir, "files": files, "manifest": manifest}
    return outputs
