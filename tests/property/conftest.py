"""Fixtures for the robustness/fuzz suite.

The API stack is an extra (``uv sync --extra api --extra dev``); without it the whole
directory is skipped, exactly like ``tests/integration/service``. Every fixture is
in-memory and deterministic: the provider is a fake, the store is SQLite in ``tmp_path`` and
the trace sink is a file in ``tmp_path``.

    # AT-OPS-07
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "fastapi",
    reason=(
        "the robustness suite drives the HTTP service: uv sync --extra api --extra dev "
        "(CI installs dev only and skips this directory)"
    ),
)

from answerdesk.config import Settings, load_settings
from answerdesk.generate.demo import demo_responses
from answerdesk.llm.presets import provider_config
from answerdesk.service.app import create_app
from answerdesk.service.config import ServiceSettings
from answerdesk.service.runtime import ServiceRuntime
from answerdesk.testing.doubles import FakeProvider

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _leave_the_base_import_graph_clean() -> Iterator[None]:
    """Restore ``sys.modules`` so the base-install import check (smoke) stays valid."""
    yield
    for name in list(sys.modules):
        if name.split(".")[0] in {"fastapi", "starlette", "sse_starlette"}:
            sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def core_settings() -> Settings:
    return load_settings(profile="offline")


@pytest.fixture()
def make_service_settings(tmp_path: Path) -> Callable[..., ServiceSettings]:
    """Build an isolated service configuration; every call gets its own SQLite/trace dir."""

    counter = {"n": 0}

    def _make(**overrides: Any) -> ServiceSettings:
        counter["n"] += 1
        base = dict(
            db_path=str(tmp_path / f"service-{counter['n']}.db"),
            trace_dir=str(tmp_path / f"traces-{counter['n']}"),
            admin_token="fuzz-admin-token",
            rate_limit_per_minute=6000,
            rate_limit_burst=6000,
            idempotency_ttl_s=3600,
            sse_token_chars=12,
            calibration_max_age_days=365,
        )
        base.update(overrides)
        return ServiceSettings(**base)

    return _make


@pytest.fixture()
def make_app(
    core_settings: Settings,
    make_service_settings: Callable[..., ServiceSettings],
) -> Iterator[Callable[..., tuple[Any, ServiceRuntime, Any]]]:
    """Build an app (and its runtime) around an optional provider/retriever override."""
    runtimes: list[ServiceRuntime] = []

    def _make(
        *,
        provider: Any = None,
        retriever: Any = None,
        service: ServiceSettings | None = None,
    ) -> tuple[Any, ServiceRuntime, Any]:
        settings = service or make_service_settings()
        inner = provider or FakeProvider(provider_config("fake"), responses=demo_responses())
        runtime = ServiceRuntime.create(
            core_settings, settings, fake=True, provider=inner, retriever=retriever
        )
        runtimes.append(runtime)
        app = create_app(settings=core_settings, service_settings=settings, runtime=runtime)
        return app, runtime, inner

    yield _make
    for runtime in reversed(runtimes):
        runtime.close()
