"""Malformed configuration and CLI input.

A bad config is an operator error, not a crash: the CLI exits 2 with a message that names the
file and the problem, and never prints a traceback. The service must not start with a config
it cannot understand.

    # AT-OPS-05  # NFR-OPS-01
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from answerdesk.cli import app

runner = CliRunner()

BAD_CONFIGS: tuple[tuple[str, str], ...] = (
    ("not_yaml", "::: not yaml :::"),
    ("list_top_level", "- a\n- b\n"),
    ("wrong_type", "llm: 5\n"),
    ("unknown_key", "nonsense: true\n"),
    ("bad_profile", "profile: nonsense\n"),
    ("bad_nested_type", "llm:\n  temperature: not-a-number\n"),
    ("huge_value", "app:\n  title: '" + "x" * 200_000 + "'\n"),
)


@pytest.mark.parametrize(("label", "body"), BAD_CONFIGS, ids=[case[0] for case in BAD_CONFIGS])
def test_bad_config_exits_two_without_a_traceback(tmp_path: Path, label: str, body: str) -> None:
    path = tmp_path / f"{label}.yaml"
    path.write_text(body, encoding="utf-8")
    result = runner.invoke(app, ["config", "show", "--config", str(path)])
    assert result.exit_code == 2, (label, result.exit_code, result.output[:300])
    output = result.output
    assert "Traceback" not in output, output[:400]
    assert "config error" in output.lower(), output[:300]


def test_missing_and_directory_paths_are_config_errors(tmp_path: Path) -> None:
    for path in (tmp_path / "missing.yaml", tmp_path / "subdir"):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "subdir":
            path.mkdir(exist_ok=True)
        result = runner.invoke(app, ["config", "show", "--config", str(path)])
        assert result.exit_code == 2, result.output[:200]
        assert "not found" in result.output.lower(), result.output[:200]


def test_garbage_environment_variables_exit_two(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANSWERDESK_LLM__TEMPERATURE", "very hot")
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 2, result.output[:300]
    assert "Traceback" not in result.output


def test_unknown_profile_is_rejected() -> None:
    result = runner.invoke(app, ["config", "show", "--profile", "nonsense"])
    assert result.exit_code == 2, result.output[:300]
    assert "Traceback" not in result.output


def test_serve_export_openapi_refuses_a_bad_config(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("llm: 5\n", encoding="utf-8")
    out = tmp_path / "openapi.json"
    result = runner.invoke(app, ["serve", "--config", str(path), "--export-openapi", str(out)])
    assert result.exit_code == 2, result.output[:300]
    assert not out.exists()
    assert "Traceback" not in result.output


def test_an_empty_config_file_is_valid_and_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["config", "show", "--config", str(path)])
    assert result.exit_code == 0, result.output[:300]
    assert '"profile"' in result.output


def test_version_works_even_with_a_broken_config(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(":::", encoding="utf-8")
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "answerdesk" in result.output.lower()
