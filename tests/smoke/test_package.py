"""Packaging and documentation smoke tests (NFR-PORT-02, NFR-PORT-03, NFR-DOC-01).

The base install must stay importable without torch, fastapi or qdrant: module tasks add
those behind extras, and this test is what keeps that promise honest.

    # AT-PORT-02  # AT-PORT-03  # AT-DOC-01  # AT-QUAL-01
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest

import answerdesk
from answerdesk.contracts import SCHEMA_VERSION, Answer

REPO_ROOT = Path(__file__).resolve().parents[2]

SUBPACKAGES = (
    "answerdesk.cache",
    "answerdesk.eval",
    "answerdesk.generate",
    "answerdesk.index",
    "answerdesk.ingest",
    "answerdesk.llm",
    "answerdesk.policy",
    "answerdesk.rerank",
    "answerdesk.retrieval",
    "answerdesk.service",
    "answerdesk.telemetry",
    "answerdesk.testing",
)

HEAVY_MODULES = ("torch", "sentence_transformers", "transformers", "fastapi", "qdrant_client")


pytestmark = pytest.mark.smoke


def test_package_imports_and_exposes_versions() -> None:
    assert isinstance(answerdesk.__version__, str)
    assert SCHEMA_VERSION == "1.0.0"
    assert Answer.__name__ == "Answer"


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_every_module_boundary_is_importable_and_documented(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__doc__, f"{module_name} has no docstring"
    assert "Owning task" in module.__doc__, f"{module_name} does not name its owning task"


def test_base_install_does_not_import_heavy_dependencies() -> None:
    """NFR-PORT-03: torch/fastapi/qdrant only ever arrive through extras."""
    loaded = [name for name in HEAVY_MODULES if name in sys.modules]
    assert loaded == [], f"heavy modules imported by the base install: {loaded}"


def test_no_globally_installed_optional_dependency_leaks_into_the_base_install() -> None:
    from answerdesk.config import Settings

    settings = Settings()
    assert settings.retrieval.rerank.enabled is False
    assert settings.embedding.backend == "hashing"


def test_pyproject_extras_and_tooling_are_configured() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    extras = project["optional-dependencies"]
    for extra in ("dev", "local", "qdrant", "api"):
        assert extra in extras, f"missing extra: {extra}"
    assert any("torch" in dep for dep in extras["local"])
    assert any("sentence-transformers" in dep for dep in extras["local"])
    assert any("qdrant-client" in dep for dep in extras["qdrant"])
    assert any("fastapi" in dep for dep in extras["api"])

    base = " ".join(project["dependencies"]).lower()
    for heavy in ("torch", "sentence-transformers", "fastapi", "qdrant"):
        assert heavy not in base, f"{heavy} must not be a base dependency"

    assert project["requires-python"] == ">=3.11,<3.14"
    assert project["scripts"]["answerdesk"] == "answerdesk.cli:main"

    ruff = pyproject["tool"]["ruff"]
    assert ruff["line-length"] == 100
    assert "E" in ruff["lint"]["select"]
    assert pyproject["tool"]["mypy"]["disallow_untyped_defs"] is True
    assert pyproject["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_documentation_set_exists_and_names_the_contracts() -> None:
    for name in ("architecture.md", "design-decisions.md"):
        path = REPO_ROOT / "docs" / name
        assert path.is_file(), f"docs/{name} is missing"
        assert path.stat().st_size > 2000, f"docs/{name} looks like a stub"

    architecture = (REPO_ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    for required in ("answerdesk.contracts", "Protocol", "token budget", "Provider"):
        assert required.lower() in architecture.lower(), f"architecture.md lacks {required!r}"

    decisions = (REPO_ROOT / "docs" / "design-decisions.md").read_text(encoding="utf-8")
    for adr in ("ADR-001", "ADR-002", "ADR-003", "ADR-004"):
        assert adr in decisions, f"design-decisions.md lacks {adr}"


def test_license_and_env_example_are_present_and_safe() -> None:
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License")
    assert "Permission is hereby granted" in license_text

    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ANSWERDESK_PROFILE" in env_example
    for line in env_example.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        _, _, value = stripped.partition("=")
        assert len(value.strip()) < 40, f".env.example must not carry a real secret: {stripped}"


def test_gitignore_covers_local_state() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".venv", ".env", "__pycache__", ".pytest_cache", "reports/"):
        assert pattern in gitignore, f".gitignore lacks {pattern}"
