"""The ``answerdesk ingest`` CLI surface end to end (FR-ING-07, FR-OPS-01).

# AT-ING-07  # AT-ING-08
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from answerdesk.cli import app

runner = CliRunner()


def _combined(result: object) -> str:
    return (getattr(result, "output", "") or "") + (getattr(result, "stderr", "") or "")


@pytest.fixture()
def small_corpus_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the CLI at a temp corpus dir and a small (fast) synthetic build."""
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setenv("ANSWERDESK_APP__CORPUS_DIR", str(corpus_dir))
    monkeypatch.setenv("ANSWERDESK_INGEST_DOCS_PER_LOCALE", "3")
    return corpus_dir


def test_ingest_demo_builds_a_corpus(small_corpus_env: Path) -> None:
    """AT-ING-07/08: ``ingest --demo`` writes documents, a manifest and the chunk file."""
    result = runner.invoke(app, ["ingest", "--demo"])
    assert result.exit_code == 0, _combined(result)
    manifest = json.loads((small_corpus_env / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["documents"] >= 6
    assert manifest["stats"]["chunks"] >= 1
    assert manifest["license"] == "CC0-1.0"
    assert (small_corpus_env / "chunks.jsonl").is_file()
    assert (small_corpus_env / "queries" / "queries.jsonl").is_file()
    text = _combined(result)
    assert "corpus_version" in text
    assert "chunks:" in text


def test_ingest_demo_is_idempotent(small_corpus_env: Path) -> None:
    """AT-ING-08: running the command twice leaves the manifest byte-identical."""
    assert runner.invoke(app, ["ingest", "--demo"]).exit_code == 0
    first = (small_corpus_env / "manifest.json").read_bytes()
    assert runner.invoke(app, ["ingest", "--demo"]).exit_code == 0
    assert (small_corpus_env / "manifest.json").read_bytes() == first


def test_ingest_without_a_source_explains_itself(small_corpus_env: Path) -> None:
    """FR-OPS-01: an empty invocation exits 2 with an actionable message, never a no-op."""
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 2
    assert "--demo" in _combined(result)


def test_ingest_directory_of_documents(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """AT-ING-01/08: a directory of mixed formats is ingested through the same command."""
    source = tmp_path / "kb"
    source.mkdir()
    (source / "a.md").write_text("# Возвраты\n\nВозврат за 14 дней.\n", encoding="utf-8")
    (source / "b.txt").write_text("Комиссия 2,5 EUR.\n\nЛимит 25 000 RUB.\n", encoding="utf-8")
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setenv("ANSWERDESK_APP__CORPUS_DIR", str(corpus_dir))
    result = runner.invoke(app, ["ingest", str(source)])
    assert result.exit_code == 0, _combined(result)
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["documents"] == 2
    assert {source["source_format"] for source in manifest["sources"]} == {"md", "txt"}
    assert all(source["license"] == "unknown" for source in manifest["sources"])


def test_ingest_reports_a_bad_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """FR-OPS-01: a missing source exits 2 with the reason on stderr."""
    monkeypatch.setenv("ANSWERDESK_APP__CORPUS_DIR", str(tmp_path / "corpus"))
    result = runner.invoke(app, ["ingest", str(tmp_path / "does-not-exist")])
    assert result.exit_code == 2
    assert "no such directory" in _combined(result)


def test_ingest_rejects_scale_and_demo_together(small_corpus_env: Path, tmp_path: Path) -> None:
    """A contradictory invocation is refused instead of guessing."""
    result = runner.invoke(app, ["ingest", "--demo", str(tmp_path)])
    assert result.exit_code == 2
    assert "choose one" in _combined(result)
