"""Helpers shared by the service integration tests (no fixtures, no pytest magic).

Kept separate from ``conftest.py`` so test modules import one obvious place for the SSE
decoder, the fixture constants and the failing providers — the same layout the generation
integration tests use.

    # AT-SVC-01  # AT-SVC-03  # AT-OPS-07
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from answerdesk.contracts import ChatProvider, Completion
from answerdesk.errors import ProviderError
from answerdesk.llm.errors import ProviderTimeoutError
from answerdesk.service.runtime import ServiceRuntime
from answerdesk.testing.doubles import FakeProvider

#: The token the admin tests present (never a real secret; test-only).
ADMIN_TOKEN = "test-admin-token-0123456789"

#: A question whose terms appear verbatim in the demo corpus — the offline answer path.
ANSWERABLE_QUESTION = "обслуживание карты"

#: A question that cannot be answered from the demo corpus.
UNANSWERABLE_QUESTION = "Какие часы работы у офиса в Антарктиде?"


def run(coro: Any) -> Any:
    """Run one async scenario to completion (no pytest-asyncio dependency).

    ``tests/conftest.py`` installs a session-wide guard that fails any ``socket.connect``
    (INV-08). ``asyncio.run`` needs a loop, and on Windows the event loop builds its
    self-pipe through a loopback ``socketpair`` — a loopback connect that is not network
    access. The guard therefore gets a loopback-only exception for the duration of the
    loop; every other destination still raises the INV-08 error.
    """
    import _socket
    import socket

    original = _socket.socket.connect
    patched_guard = socket.socket.connect

    def _loopback_allowed(self: Any, address: Any, *args: Any, **kwargs: Any) -> Any:
        host = address[0] if isinstance(address, tuple) and address else address
        if host in ("127.0.0.1", "::1", "localhost", ""):
            return original(self, address, *args, **kwargs)
        return patched_guard(self, address, *args, **kwargs)

    socket.socket.connect = _loopback_allowed  # type: ignore[method-assign]
    try:
        return asyncio.run(coro)
    finally:
        socket.socket.connect = patched_guard  # type: ignore[method-assign]


@asynccontextmanager
async def lifespan_client(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    """An AsyncClient whose requests run inside the app's lifespan (runtime is built)."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://service.test") as client:
            yield client


@dataclass
class SSEFrame:
    event: str
    data: dict[str, Any]
    raw: str


def parse_sse(body: str) -> list[SSEFrame]:
    """Parse an SSE body into frames (the test's independent decoder)."""
    frames: list[SSEFrame] = []
    for block in body.split("\n\n"):
        if not block.strip() or block.lstrip().startswith(":"):
            continue
        event = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: ") :])
        if not event:
            continue
        payload = json.loads("\n".join(data_lines)) if data_lines else {}
        frames.append(SSEFrame(event=event, data=payload, raw=block))
    return frames


def event_names(frames: list[SSEFrame]) -> list[str]:
    return [frame.event for frame in frames]


def first(frames: list[SSEFrame], event: str) -> SSEFrame | None:
    return next((frame for frame in frames if frame.event == event), None)


def events_text(frames: list[SSEFrame]) -> str:
    return "".join(frame.data.get("delta", "") for frame in frames if frame.event == "tokens")


class TimeoutProvider(FakeProvider):
    """A provider that always times out (NFR-OPS-01 failure row 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "timeout-provider"

    def complete(self, messages: Any, *, response_schema: Any = None) -> Completion:
        raise ProviderTimeoutError("the provider timed out after 60.0s")


class ServerErrorProvider(FakeProvider):
    """A provider that always fails with a 5xx (failure row 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "server-error-provider"

    def complete(self, messages: Any, *, response_schema: Any = None) -> Completion:
        raise ProviderError("provider returned 500 Internal Server Error")


class GarbageProvider(FakeProvider):
    """A provider that returns prose where JSON was required (failure row 3)."""

    def __init__(self) -> None:
        super().__init__(responses={})
        self.name = "garbage-provider"

    def complete(self, messages: Any, *, response_schema: Any = None) -> Completion:
        return Completion(
            text="Sure! Here is a friendly answer without any JSON.",
            model=self.config.model,
            input_tokens=12,
            output_tokens=9,
        )


def runtime_with_provider(
    provider: ChatProvider,
    core: Any,
    service: Any,
) -> ServiceRuntime:
    return ServiceRuntime.create(core, service, fake=True, provider=provider)


__all__ = [
    "ADMIN_TOKEN",
    "ANSWERABLE_QUESTION",
    "UNANSWERABLE_QUESTION",
    "GarbageProvider",
    "SSEFrame",
    "ServerErrorProvider",
    "TimeoutProvider",
    "event_names",
    "events_text",
    "first",
    "lifespan_client",
    "parse_sse",
    "run",
    "runtime_with_provider",
]
