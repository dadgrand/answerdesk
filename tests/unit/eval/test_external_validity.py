"""External-validity arm: offline tests (no network, no live providers).

The fixtures build a miniature answerdesk-bench so every reader, metric, calibration helper
and the runner can be exercised without the committed data or an API key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerdesk.eval.external import datasets as ext_datasets
from answerdesk.eval.external.bench import Bench
from answerdesk.eval.external.calibration import (
    build_template,
    score_sheets,
    write_protocol,
)
from answerdesk.eval.external.datasets import (
    ExternalDataError,
    cache_root,
    ensure,
    load_curated,
    verify_upstream,
)
from answerdesk.eval.external.qa import build_prompt, delta, parse_letter, summarise
from answerdesk.eval.external.retrieval import (
    DenseRanker,
    HybridRanker,
    LexicalRanker,
    compare,
    evaluate,
    evaluate_scores,
    score_queries,
)
from answerdesk.eval.external.runner import ExternalConfig, run_external

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _qa_row(
    item_id: str, language: str, answer_index: int, split: str = "test"
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "language": language,
        "question": f"Секретный вопрос {item_id}?",
        "options": ["a", "b", "c", "d"],
        "answer_index": answer_index,
        "gold_evidence_ids": [f"p-{item_id}"],
        "split": split,
        "license": "CC-BY-SA-4.0",
        "suite": "belebele_qa",
    }


def _retrieval_row(
    item_id: str, language: str, passage: str, split: str = "test"
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "language": language,
        "query": f"запрос {item_id}",
        "qrels": [passage],
        "split": split,
        "license": "CC-BY-SA-4.0",
        "suite": "belebele_retrieval",
    }


@pytest.fixture()
def mini_bench(tmp_path: Path) -> Bench:
    root = tmp_path / "data" / "bench" / "answerdesk-bench" / "v0.1"
    _write_jsonl(
        root / "belebele" / "qa" / "ru.jsonl",
        [_qa_row("q1", "ru", 0), _qa_row("q2", "ru", 2), _qa_row("q3", "ru", 3, split="dev")],
    )
    _write_jsonl(
        root / "belebele" / "retrieval" / "ru" / "queries.jsonl",
        [_retrieval_row("r1", "ru", "p-1"), _retrieval_row("r2", "ru", "p-2")],
    )
    _write_jsonl(
        root / "belebele" / "retrieval" / "ru" / "corpus.jsonl",
        [
            {
                "passage_id": "p-1",
                "language": "ru",
                "text": "альфа браво",
                "license": "CC-BY-SA-4.0",
            },
            {
                "passage_id": "p-2",
                "language": "ru",
                "text": "гамма дельта",
                "license": "CC-BY-SA-4.0",
            },
            {"passage_id": "p-3", "language": "ru", "text": "эпсилон", "license": "CC-BY-SA-4.0"},
        ],
    )
    _write_jsonl(
        root / "mrtydi" / "ru" / "queries.jsonl",
        [
            {
                "item_id": "m1",
                "language": "ru",
                "query": "альфа",
                "qrels": {"p-1": 1},
                "split": "test",
                "license": "Apache-2.0",
                "suite": "mrtydi_retrieval",
            }
        ],
    )
    _write_jsonl(
        root / "mrtydi" / "ru" / "corpus.jsonl",
        [
            {"passage_id": "p-1", "language": "ru", "text": "альфа браво", "license": "Apache-2.0"},
            {"passage_id": "p-2", "language": "ru", "text": "гамма", "license": "Apache-2.0"},
        ],
    )
    _write_jsonl(
        root / "domain" / "ru.jsonl",
        [
            {
                "item_id": "d1",
                "language": "ru",
                "question": "вопрос",
                "split": "test",
                "suite": "domain",
                "license": "CC0-1.0",
            }
        ],
    )
    manifest = {
        "benchmark": "answerdesk-bench",
        "version": "0.1.0",
        "schema_version": "1.0.0",
        "release_date": "2026-10-21",
        "self_sha256": "0" * 64,
        "license_map": {"belebele/": "CC-BY-SA-4.0", "mrtydi/": "Apache-2.0", "domain/": "CC0-1.0"},
        "languages": ["ru"],
        "suites": {
            "belebele_qa": {
                "items": 2,
                "languages": ["ru"],
                "license": "CC-BY-SA-4.0",
                "task": "multiple_choice",
                "notes": "",
                "paths": ["belebele/qa/ru.jsonl"],
            },
            "belebele_retrieval": {
                "items": 2,
                "languages": ["ru"],
                "license": "CC-BY-SA-4.0",
                "task": "retrieval",
                "notes": "",
                "paths": ["belebele/retrieval/ru/queries.jsonl"],
            },
            "mrtydi_retrieval": {
                "items": 1,
                "languages": ["ru"],
                "license": "Apache-2.0",
                "task": "retrieval",
                "notes": "",
                "paths": ["mrtydi/ru/queries.jsonl"],
            },
            "domain": {
                "items": 1,
                "languages": ["ru"],
                "license": "CC0-1.0",
                "task": "qa",
                "notes": "",
                "paths": ["domain/ru.jsonl"],
            },
        },
        "counts": {
            "by_suite": {
                "belebele_qa": 2,
                "belebele_retrieval": 2,
                "mrtydi_retrieval": 1,
                "domain": 1,
            },
            "by_language": {"ru": 6},
        },
        "files": [
            {"path": "belebele/qa/ru.jsonl", "sha256": "", "bytes": 1, "items": 2},
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return Bench(load_curated(tmp_path, verify=False))


# --------------------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------------------


def test_registry_marks_licence_and_redistribution() -> None:
    belebele = ext_datasets.REGISTRY["belebele"]
    assert belebele.licence == "CC-BY-SA-4.0"
    assert belebele.redistribute is False
    assert ext_datasets.REGISTRY["mrtydi"].redistribute is True


def test_curated_load_verifies_hashes(tmp_path: Path) -> None:
    root = tmp_path / "data" / "bench" / "answerdesk-bench" / "v0.1"
    root.mkdir(parents=True)
    payload = b'{"a": 1}\n'
    (root / "data.jsonl").write_bytes(payload)
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "self_sha256": "f" * 64,
                "files": [
                    {"path": "data.jsonl", "sha256": digest, "bytes": len(payload), "items": 1}
                ],
            }
        ),
        encoding="utf-8",
    )
    record = load_curated(tmp_path, verify=True)
    assert record.manifest_sha256 == "f" * 64
    (root / "data.jsonl").write_bytes(b"tampered")
    with pytest.raises(ExternalDataError):
        load_curated(tmp_path, verify=True)


def test_upstream_missing_cache_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANSWERDESK_EVAL_CACHE", str(tmp_path / "cache"))
    with pytest.raises(ExternalDataError) as excinfo:
        ensure("miracl")
    assert "not cached" in excinfo.value.message
    assert "--fetch" in str(excinfo.value.detail)


def test_verify_upstream_reports_missing_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANSWERDESK_EVAL_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "cache" / "mrtydi"
    root.mkdir(parents=True)
    payload = verify_upstream("mrtydi")
    assert payload["verified"] is False


def test_cache_root_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANSWERDESK_EVAL_CACHE", str(tmp_path / "elsewhere"))
    assert cache_root() == tmp_path / "elsewhere"


# --------------------------------------------------------------------------------------
# bench reader
# --------------------------------------------------------------------------------------


def test_bench_readers(mini_bench: Bench) -> None:
    assert [item.item_id for item in mini_bench.qa("ru")] == ["q1", "q2"]
    assert [item.item_id for item in mini_bench.qa("ru", split=None)] == ["q1", "q2", "q3"]
    assert mini_bench.qa("ru")[0].gold_passage_ids == ("p-q1",)
    assert mini_bench.retrieval_queries("ru")[0].qrels["p-1"] == 1
    assert [passage.passage_id for passage in mini_bench.passages("ru")] == ["p-1", "p-2", "p-3"]
    assert mini_bench.mrtydi_queries()[0].qrels == {"p-1": 1}
    assert mini_bench.domain("ru")[0]["item_id"] == "d1"
    counts = mini_bench.suite_counts()
    assert counts["belebele_qa"]["items"] == 2


def test_bench_missing_file_is_external_error(mini_bench: Bench) -> None:
    with pytest.raises(ExternalDataError):
        mini_bench.qa("zz")


# --------------------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------------------


def test_retrieval_metrics_and_ci(mini_bench: Bench) -> None:
    queries = mini_bench.retrieval_queries("ru")
    lexical = LexicalRanker(mini_bench.passages("ru"))
    report = evaluate(lexical, queries, samples=200, seed=1)
    assert report.n == 2
    assert 0.0 <= report.ndcg10 <= 1.0
    assert report.ndcg10_ci[0] <= report.ndcg10_ci[1]


def test_hybrid_batched_matches_per_query(mini_bench: Bench) -> None:
    from answerdesk.testing.doubles import HashingEmbedder

    queries = mini_bench.retrieval_queries("ru")
    passages = mini_bench.passages("ru")
    lexical = LexicalRanker(passages)
    dense = DenseRanker(passages, HashingEmbedder())
    hybrid = HybridRanker(passages, lexical, dense, alpha=0.5)
    per_query = evaluate(hybrid, queries, samples=100, seed=2)
    matrix = score_queries(hybrid, queries)
    batched = evaluate_scores(
        arm="hybrid",
        scores=matrix,
        queries=queries,
        passage_ids=[passage.passage_id for passage in passages],
        samples=100,
        seed=2,
    )
    assert per_query.ndcg10 == pytest.approx(batched.ndcg10)
    assert per_query.recall10 == pytest.approx(batched.recall10)


def test_compare_flags_zero_crossing(mini_bench: Bench) -> None:
    queries = mini_bench.retrieval_queries("ru")
    passages = mini_bench.passages("ru")
    lexical = LexicalRanker(passages)
    base = evaluate(lexical, queries, samples=100, seed=3)
    same = evaluate(lexical, queries, samples=100, seed=3)
    payload = compare(base, same, samples=100, seed=3)
    assert payload["ndcg@10"]["ci_crosses_zero"] is True
    assert payload["ndcg@10"]["delta"] == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# qa
# --------------------------------------------------------------------------------------


def test_prompt_conditions_differ_only_by_context(mini_bench: Bench) -> None:
    item = mini_bench.qa("ru")[0]
    passage = mini_bench.passages("ru")[0]
    closed = build_prompt(item, "closed", [])
    rag = build_prompt(item, "rag", [passage])
    assert "Контекст" not in closed[1]["content"]
    assert "Контекст" in rag[1]["content"]
    assert passage.text in rag[1]["content"]
    assert closed[1]["content"].endswith(
        "Ответь одной буквой варианта. / Answer with one option letter."
    )


def test_parse_letter() -> None:
    assert parse_letter("B") == "B"
    assert parse_letter("Ответ: c") == "C"
    assert parse_letter("не знаю") is None


def test_delta_detects_rag_gain() -> None:
    rows = []
    for index in range(20):
        rows.append({"item_id": f"i{index}", "condition": "closed", "correct": index < 5})
        rows.append({"item_id": f"i{index}", "condition": "rag", "correct": index < 15})
    payload = delta(rows, samples=300, seed=0)
    assert payload["n"] == 20
    assert payload["delta"] == pytest.approx(0.5)
    assert payload["ci_low"] > 0.0
    assert payload["ci_crosses_zero"] is False


def test_summarise_uses_wilson() -> None:
    rows = [
        {"condition": "closed", "correct": True},
        {"condition": "closed", "correct": False},
        {"condition": "rag", "correct": True},
    ]
    closed = summarise(rows, "closed")
    assert closed.n == 2 and closed.correct == 1
    assert closed.ci[0] < closed.accuracy < closed.ci[1]


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------


def test_template_and_protocol(tmp_path: Path) -> None:
    rows = [{"item_id": f"q{index:03d}", "question": "вопрос"} for index in range(200)]
    template = build_template(language="ru", items=rows, out_dir=tmp_path, limit=150)
    assert template.items == 150
    header = template.path.read_text(encoding="utf-8").splitlines()[0]
    assert "human_label_a" in header and "adjudicated_label" in header
    protocol = write_protocol(tmp_path, languages=["ru"], limit=150)
    assert "diagnostic" in protocol.read_text(encoding="utf-8")


def test_score_sheets_kappa(tmp_path: Path) -> None:
    import csv

    path = tmp_path / "calibration-ru.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id", "language", "judge_label", "human_label_a", "adjudicated_label"],
        )
        writer.writeheader()
        for index in range(10):
            label = "supported" if index % 2 == 0 else "unsupported"
            writer.writerow(
                {
                    "item_id": f"q{index}",
                    "language": "ru",
                    "judge_label": label,
                    "human_label_a": label,
                    "adjudicated_label": label,
                }
            )
    payload = score_sheets([path])
    assert payload["per_language"]["ru"]["kappa"] == pytest.approx(1.0)
    assert payload["status"] == "calibrated"


def test_score_sheets_pending_without_labels(tmp_path: Path) -> None:
    import csv

    path = tmp_path / "calibration-ru.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "language", "judge_label"])
        writer.writeheader()
        writer.writerow({"item_id": "q1", "language": "ru", "judge_label": ""})
    payload = score_sheets([path])
    assert payload["status"] == "pending-human-labels"
    assert payload["overall_kappa"] is None


# --------------------------------------------------------------------------------------
# runner (offline, deterministic doubles)
# --------------------------------------------------------------------------------------


def test_runner_offline_end_to_end(mini_bench: Bench, tmp_path: Path) -> None:
    config = ExternalConfig(
        run_id="external-test",
        report_dir=tmp_path / "external-test",
        suites=("belebele_qa", "mrtydi_retrieval", "belebele_retrieval", "domain"),
        languages=("ru",),
        limit=2,
        samples=100,
        replay=True,
        calibrate_items=5,
    )
    summary = run_external(
        config,
        bench=mini_bench,
        provider=_FakeAnswerer("A"),
        embedder=_FakeEmbedder(),
    )
    assert summary["belebele_qa"]["language_results"]["ru"]["n"] == 2
    assert summary["mrtydi_retrieval"]["n"] == 1
    assert summary["judge_gate"]["status"] == "diagnostic-only"
    assert summary["calibration"]["status"] == "pending-human-labels"
    assert (config.report_dir / "summary.json").is_file()
    assert (config.report_dir / "report.md").is_file()
    assert (config.report_dir / "calibration" / "PROTOCOL.md").is_file()
    text = (config.report_dir / "summary.json").read_text(encoding="utf-8")
    assert "Секретный вопрос" not in text  # dataset text never leaves the adapter


class _FakeAnswerer:
    def __init__(self, letter: str) -> None:
        self.letter = letter

    def complete(self, messages, *, response_schema=None):
        from answerdesk.contracts import Completion

        return Completion(text=self.letter, model="fake", finish_reason="stop")

    def stream(self, messages):
        return iter(())


class _FakeEmbedder:
    def embed_documents(self, texts):
        return [[float(len(text) % 7), 1.0, 0.5] for text in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.5]
