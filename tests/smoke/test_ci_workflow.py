"""The CI workflow is part of the contract, so it is tested like code.

NFR-DET-01 (offline, no key), NFR-DET-03 (fast gate + full gate), NFR-SEC-03 (no secrets
in the repo) and NFR-PORT-02 (Windows and Linux parity) are only true if the workflow
actually says so.

    # AT-DET-01  # AT-DET-03  # AT-SEC-03
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def workflow_text() -> str:
    assert WORKFLOW.is_file(), ".github/workflows/ci.yml is missing"
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow(workflow_text: str) -> dict[str, Any]:
    parsed = yaml.safe_load(workflow_text)
    assert isinstance(parsed, dict)
    return parsed


def test_workflow_triggers_on_push_and_pull_request(workflow: dict[str, Any]) -> None:
    # `on:` is parsed as the boolean True by YAML 1.1 loaders; accept either spelling.
    triggers = workflow.get("on", workflow.get(True))
    assert triggers is not None
    assert "push" in triggers
    assert "pull_request" in triggers


def test_matrix_covers_windows_and_linux(workflow: dict[str, Any]) -> None:
    """NFR-PORT-02: the local-first target is Windows, so CI must run it."""
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]
    assert "ubuntu-latest" in matrix["os"]
    assert "windows-latest" in matrix["os"]
    assert "3.12" in [str(version) for version in matrix["python-version"]]


def test_gate_runs_tests_lint_and_types_offline(workflow_text: str) -> None:
    assert "uv sync --extra dev" in workflow_text
    # the service, property and load suites need the api extra; without it they would
    # importorskip and the gate would test nothing (t34)
    assert "--extra api" in workflow_text
    assert "ANSWERDESK_REQUIRE_API" in workflow_text
    assert "uv run pytest -q" in workflow_text
    assert "ruff check ." in workflow_text
    assert "ruff format --check ." in workflow_text
    assert "mypy" in workflow_text
    assert "from answerdesk.contracts import Answer" in workflow_text


def test_no_api_key_or_secret_is_required(workflow_text: str) -> None:
    """NFR-DET-01 / NFR-SEC-03: CI must never need a key, and must never carry one."""
    assert "secrets." not in workflow_text
    assert "OPENAI_API_KEY" not in workflow_text
    assert "DEEPSEEK_API_KEY" not in workflow_text
    assert "ANSWERDESK_PROFILE" in workflow_text


def test_fast_and_full_gates_exist(workflow: dict[str, Any]) -> None:
    """NFR-DET-03: a ≤3 min gate for every push, a ≤10 min gate for the whole suite."""
    jobs = workflow["jobs"]
    assert "fast" in jobs
    assert "test" in jobs
    fast_runs = " ".join(
        str(step.get("run", "")) for step in jobs["fast"].get("steps", []) if isinstance(step, dict)
    )
    assert "-m smoke" in fast_runs


def test_workflow_is_pinned_and_minimal_permissions(workflow: dict[str, Any]) -> None:
    assert workflow.get("permissions", {}).get("contents") == "read"
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            uses = step.get("uses")
            if uses:
                assert "@" in uses, f"unpinned action: {uses}"


def test_gitignore_protects_generated_state() -> None:
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("logs/", "data/index/", ".coverage", ".ruff_cache/"):
        assert pattern in text, f".gitignore lacks {pattern}"
