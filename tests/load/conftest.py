"""Fixtures for the load-verification tests.

The harness lives in ``scripts/load.py`` (it must be runnable as ``python scripts/load.py``),
so the tests import it by path under a unique module name. Everything here is offline: the
app is driven through ``httpx.ASGITransport`` and the provider is the in-process fake.

    # AT-OPS-07  # NFR-LAT-01
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip(
    "fastapi",
    reason=(
        "the load harness drives the HTTP app: uv sync --extra api --extra dev "
        "(CI installs dev only and skips this directory)"
    ),
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "load.py"


@pytest.fixture(scope="session")
def load_harness() -> ModuleType:
    """Import ``scripts/load.py`` once per session under a private module name."""
    name = "answerdesk_load_harness"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader, SCRIPT
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
