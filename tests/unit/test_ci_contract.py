"""The CI contract itself: workflow shape and the no-silent-skip guard.

The pipeline can only be trusted if two things stay true: the workflow installs the extras
the tests need, and a missing extra fails the run instead of skipping it. This module pins
both by reading the workflow and by exercising the conftest helper with the import blinded.

    # AT-OPS-01  # NFR-OPS-01
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CONFTESTS = {
    "service": ROOT / "tests" / "integration" / "service" / "conftest.py",
    "property": ROOT / "tests" / "property" / "conftest.py",
    "load": ROOT / "tests" / "load" / "conftest.py",
}


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps(job: dict) -> list[dict]:
    return list(job["steps"])


def _run_text(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in _steps(job))


def _load_conftest(name: str, path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import a conftest by path; its directory may import sibling helpers."""
    module_name = f"_ci_contract_{name}_conftest"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    monkeypatch.syspath_prepend(str(path.parent))
    spec.loader.exec_module(module)
    return module


def test_workflow_lints_the_whole_repository() -> None:
    workflow = _workflow()
    fast = workflow["jobs"]["fast"]
    text = _run_text(fast)
    assert "ruff check ." in text, text
    assert "ruff format --check ." in text, text


def test_workflow_installs_the_extras_the_tests_need() -> None:
    workflow = _workflow()
    test_job = workflow["jobs"]["test"]
    text = _run_text(test_job)
    assert "--extra dev" in text and "--extra api" in text, text
    assert "--frozen" in text, text
    # the guard that turns a missing extra into a failure, not a skip
    assert test_job["env"]["ANSWERDESK_REQUIRE_API"] == "1"


def test_workflow_has_no_secrets_and_caches_on_the_lock() -> None:
    raw = WORKFLOW.read_text(encoding="utf-8")
    for marker in ("secrets.", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "password:"):
        assert marker not in raw, marker
    workflow = _workflow()
    for job_name in ("fast", "test"):
        uv_steps = [
            step
            for step in _steps(workflow["jobs"][job_name])
            if str(step.get("uses", "")).startswith("astral-sh/setup-uv")
        ]
        assert uv_steps, job_name
        with_block = uv_steps[0]["with"]
        assert with_block["enable-cache"] is True
        assert with_block["cache-dependency-glob"] == "uv.lock"


def test_workflow_matrix_and_budget() -> None:
    workflow = _workflow()
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]
    versions = set(matrix["python-version"])
    versions.update(entry["python-version"] for entry in matrix.get("include", []))
    assert {"3.11", "3.12"} <= versions, versions
    assert workflow["jobs"]["test"]["runs-on"].startswith("${{ matrix.os }}")
    assert workflow["jobs"]["test"]["timeout-minutes"] <= 20
    assert workflow["jobs"]["fast"]["timeout-minutes"] <= 10
    # no container / service / network-dependent step (comments may mention them)
    raw = WORKFLOW.read_text(encoding="utf-8").lower()
    assert "uses: docker" not in raw
    assert "\n    services:" not in raw
    assert "--with" not in raw


@pytest.mark.parametrize("suite", sorted(CONFTESTS))
def test_api_suites_keep_a_local_importorskip(suite: str) -> None:
    """Locally the extra is optional, so the directories still skip instead of erroring."""
    source = CONFTESTS[suite].read_text(encoding="utf-8")
    assert "pytest.importorskip(" in source, suite
    assert '"fastapi"' in source, suite


def _load_root_conftest(name: str) -> ModuleType:
    module_name = f"_ci_contract_{name}_conftest"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "tests" / "conftest.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _blind_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _import(name: str, *args, **kwargs):
        if name.startswith(("fastapi", "starlette", "sse_starlette")):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


def test_configure_guard_is_off_without_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_root_conftest("off")
    monkeypatch.delenv("ANSWERDESK_REQUIRE_API", raising=False)
    _blind_fastapi(monkeypatch)
    module.pytest_configure(None)  # must not raise


def test_configure_guard_aborts_ci_without_the_api_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_root_conftest("abort")
    monkeypatch.setenv("ANSWERDESK_REQUIRE_API", "1")
    _blind_fastapi(monkeypatch)
    with pytest.raises(pytest.UsageError, match="api extra is missing"):
        module.pytest_configure(None)


def test_configure_guard_passes_when_the_extra_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi", reason="this test asserts the pass-through case")
    module = _load_root_conftest("present")
    monkeypatch.setenv("ANSWERDESK_REQUIRE_API", "1")
    module.pytest_configure(None)  # must not raise
