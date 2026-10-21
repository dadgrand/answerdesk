"""CLI surface tests (FR-OPS-01): every documented subcommand exists and fails loudly.

# AT-OPS-01  # AT-OPS-05
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from answerdesk import PACKAGE_VERSION
from answerdesk.cli import app
from answerdesk.contracts import SCHEMA_VERSION

runner = CliRunner()

DOCUMENTED_COMMANDS = (
    "version",
    "config",
    "ingest",
    "index",
    "ask",
    "calibrate",
    "serve",
    "eval",
    "demo",
)


def _combined(result: object) -> str:
    """stdout + stderr, whichever click/typer version puts them where."""
    output = getattr(result, "output", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    return output + stderr


def test_help_lists_every_documented_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    text = _combined(result)
    for command in DOCUMENTED_COMMANDS:
        assert command in text, f"{command} missing from --help"


def test_version_prints_package_and_schema_versions() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert PACKAGE_VERSION in result.output
    assert SCHEMA_VERSION in result.output


@pytest.mark.parametrize(
    "command", ["ingest", "index", "ask", "calibrate", "serve", "eval", "demo", "config"]
)
def test_every_subcommand_has_help(command: str) -> None:
    """FR-OPS-01: --help works for every subcommand and exits 0."""
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, _combined(result)


def test_config_show_redacts_and_reports_the_profile() -> None:
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0, _combined(result)
    payload = json.loads(result.output)
    assert payload["profile"] == "offline"
    assert payload["llm"]["preset"] == "fake"
    assert payload["llm"]["api_key"] is None
    assert payload["retrieval"]["rrf_k"] == 60


def test_config_check_resolves_the_offline_provider() -> None:
    result = runner.invoke(app, ["config", "check", "--profile", "offline"])
    assert result.exit_code == 0, _combined(result)
    payload = json.loads(result.output)
    assert payload["preset"] == "fake"
    assert payload["requires_api_key"] is False


def test_config_check_fails_actionably_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANSWERDESK_LLM__API_KEY", raising=False)
    result = runner.invoke(app, ["config", "check", "--profile", "api"])
    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in _combined(result)


def test_config_check_never_prints_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-cli-0123456789abcdef"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    result = runner.invoke(app, ["config", "check", "--profile", "api"])
    assert result.exit_code == 0, _combined(result)
    assert secret not in _combined(result)
    assert json.loads(result.output)["api_key"] == "***"


def test_missing_config_file_exits_two(tmp_path: Path) -> None:
    result = runner.invoke(app, ["config", "show", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2
    assert "not found" in _combined(result)


@pytest.mark.parametrize(
    ("argv", "owner"),
    [
        (["demo", "--smoke"], "t14"),
    ],
)
def test_stubs_name_their_owning_task_and_exit_two(argv: list[str], owner: str) -> None:
    """A stub must never pretend to work: exit 2 and say which task implements it."""
    result = runner.invoke(app, argv)
    assert result.exit_code == 2, _combined(result)
    text = _combined(result)
    assert owner in text
    assert "not implemented" in text.lower() or "implemented by" in text.lower()


def test_serve_delegates_to_the_service_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-SVC-01/07: `serve` is implemented and delegates to answerdesk.service."""
    import answerdesk.service.server as server_module

    captured: dict[str, object] = {}

    def fake_run_server(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(server_module, "run_server", fake_run_server)
    result = runner.invoke(app, ["serve", "--fake", "--port", "8799", "--healthcheck"])
    assert result.exit_code == 0, _combined(result)
    assert captured["fake"] is True
    assert captured["port"] == 8799
    assert captured["healthcheck"] is True
    assert captured["export_openapi_path"] is None


def test_serve_exports_the_openapi_schema_without_a_server(tmp_path: Path) -> None:
    """The OpenAPI document is reproducible from one command (docs/api.md §7)."""
    target = tmp_path / "openapi.json"
    result = runner.invoke(app, ["serve", "--export-openapi", str(target)])
    assert result.exit_code == 0, _combined(result)
    assert target.is_file()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["info"]["title"] == "answerdesk service"
    assert "/v1/chat" in payload["paths"]
    assert "/healthz" in payload["paths"]


def test_index_builds_and_then_reuses_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR-OPS-02: ``index`` builds both indexes, then reports ``unchanged`` for the same corpus.

    Uses a one-document-per-locale corpus so the command stays a fast unit test; the
    corpus build is the same code path as ``answerdesk ingest --demo``.
    """
    monkeypatch.setenv("ANSWERDESK_APP__CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setenv("ANSWERDESK_APP__INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("ANSWERDESK_INGEST_DOCS_PER_LOCALE", "1")

    first = runner.invoke(app, ["index", "--demo"])
    assert first.exit_code == 0, _combined(first)
    text = _combined(first)
    assert "index: built" in text
    assert "corpus_version: corpus-" in text
    assert "index_version: index-" in text

    second = runner.invoke(app, ["index", "--demo"])
    assert second.exit_code == 0, _combined(second)
    assert "index: unchanged (reused)" in _combined(second)
    assert _index_version(second) == _index_version(first)


def _index_version(result: object) -> str:
    for line in _combined(result).splitlines():
        if line.startswith("index_version:"):
            return line.split(":", 1)[1].strip()
    return ""


def test_index_reports_a_missing_corpus_actionably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing corpus without ``--demo`` exits 2 and names the command that builds it."""
    monkeypatch.setenv("ANSWERDESK_APP__CORPUS_DIR", str(tmp_path / "no-corpus"))
    monkeypatch.setenv("ANSWERDESK_APP__INDEX_DIR", str(tmp_path / "index"))
    result = runner.invoke(app, ["index"])
    assert result.exit_code == 2, _combined(result)
    text = _combined(result)
    assert "index error" in text.lower()
    assert "ingest" in text.lower()


def test_eval_help_documents_the_harness_flags() -> None:
    """FR-EVL-08: the eval command is implemented, and its flags are the documented ones."""
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0, _combined(result)
    text = _combined(result)
    for flag in ("--synthetic", "--provider", "--report", "--all", "--retrieval-only"):
        assert flag in text, f"{flag} missing from `eval --help`"


def test_eval_delegates_and_reports_a_missing_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing corpus is an actionable exit 2, never an empty run (§10.2)."""
    monkeypatch.setenv("ANSWERDESK_APP__CORPUS_DIR", str(tmp_path / "no-corpus"))
    result = runner.invoke(
        app,
        ["eval", "--synthetic", "--provider", "fake", "--report", str(tmp_path / "run")],
    )
    assert result.exit_code == 2, _combined(result)
    text = _combined(result)
    assert "eval error" in text.lower()
    assert "ingest" in text.lower()


def test_unknown_command_is_a_usage_error() -> None:
    result = runner.invoke(app, ["frobnicate"])
    assert result.exit_code == 2
