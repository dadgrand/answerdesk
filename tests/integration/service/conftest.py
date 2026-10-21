"""Shared fixtures for the service integration tests (offline, deterministic, no network).

The tests drive the *real* app through ``httpx.AsyncClient`` + ``ASGITransport`` with the
lifespan context manager, so middleware, exception handlers, the SQLite store, the JSONL
tracer and the metrics registry are all exercised — only the model provider is a double
(:class:`answerdesk.testing.doubles.FakeProvider`).

``pytest-asyncio`` is not a dependency (§13.3 allowlist), so every test is a plain sync
function that calls :func:`support.run` on an async scenario.

    # AT-SVC-01  # AT-SVC-02  # AT-SVC-07  # AT-DET-01
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "fastapi",
    reason=(
        "the service tests need the `api` extra: uv sync --extra api --extra dev "
        "(CI installs it and fails without it)"
    ),
)
from service_support import ADMIN_TOKEN, lifespan_client, run

from answerdesk.config import Settings, load_settings
from answerdesk.generate.demo import demo_responses
from answerdesk.llm.presets import provider_config
from answerdesk.service.app import create_app
from answerdesk.service.config import ServiceSettings
from answerdesk.service.runtime import ServiceRuntime
from answerdesk.testing.doubles import FakeProvider

pytestmark = pytest.mark.integration

#: Modules the ``api`` extra installs. The service suite is their only consumer.
_SERVICE_ONLY_MODULES = ("fastapi", "starlette", "sse_starlette")

#: The api extra is optional locally; CI installs it, and tests/conftest.py aborts the
#: session when ANSWERDESK_REQUIRE_API=1 and the extra is missing (t34).


@pytest.fixture(autouse=True)
def _leave_the_base_import_graph_clean() -> Iterator[None]:
    """Restore ``sys.modules`` after each service test.

    ``tests/smoke/test_package.py::test_base_install_does_not_import_heavy_dependencies``
    asserts that the *base* install never imports FastAPI. That assertion is about the
    package graph, but it observes a process-global: this directory is the one place that
    legitimately imports the ``api`` extra, so it leaves ``sys.modules`` as it found it.
    The service modules themselves are imported once per session and keep their references,
    so this only removes the module cache entries, not any live object.
    """
    yield
    import sys

    for name in list(sys.modules):
        if name.split(".")[0] in _SERVICE_ONLY_MODULES:
            sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ambient proxy variables must not leak into a hermetic test (INV-08)."""
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        if os.environ.get(name):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def core_settings() -> Settings:
    return load_settings(profile="offline")


@pytest.fixture()
def service_settings(tmp_path: Path) -> ServiceSettings:
    return ServiceSettings(
        db_path=str(tmp_path / "service.db"),
        trace_dir=str(tmp_path / "traces"),
        admin_token=ADMIN_TOKEN,
        rate_limit_per_minute=600,
        rate_limit_burst=600,
        idempotency_ttl_s=3600,
        sse_token_chars=12,
        calibration_max_age_days=365,
    )


@pytest.fixture()
def fake_provider() -> FakeProvider:
    """The provider the runtime is built with, so a test can count model calls."""
    return FakeProvider(provider_config("fake"), responses=demo_responses())


@pytest.fixture()
def runtime(
    core_settings: Settings,
    service_settings: ServiceSettings,
    fake_provider: FakeProvider,
) -> Iterator[ServiceRuntime]:
    built = ServiceRuntime.create(
        core_settings, service_settings, fake=True, provider=fake_provider
    )
    try:
        yield built
    finally:
        built.close()


@pytest.fixture()
def app(
    core_settings: Settings,
    service_settings: ServiceSettings,
    runtime: ServiceRuntime,
) -> Any:
    return create_app(settings=core_settings, service_settings=service_settings, runtime=runtime)


@pytest.fixture()
def make_client(app: Any) -> Callable[[Callable[..., Any]], Any]:
    """Run an async scenario ``async def scenario(client, runtime)`` against a live app."""

    def _run(scenario: Callable[[Any, Any], Any]) -> Any:
        async def _wrapper() -> Any:
            async with lifespan_client(app) as client:
                return await scenario(client, app.state.runtime)

        return run(_wrapper())

    return _run


__all__ = ["app", "core_settings", "fake_provider", "make_client", "runtime", "service_settings"]
