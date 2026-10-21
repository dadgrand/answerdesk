"""Streaming degradation, provider SSE parsing and client cancellation.

The HTTP status line is sent before the first token, so a failure *inside* the stream cannot
change it. The contract (docs/api.md §5.4) is an ``error`` frame followed by ``done``, and a
client that disconnects must not leave a dead coroutine, a stuck idempotency lock or an
unhandled exception behind.

    # AT-OPS-07  # AT-SVC-06
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fuzz_support import (
    ANSWERABLE,
    RaisingProvider,
    assert_clean_stream,
    parse_sse,
    run,
    scenario,
)

pytestmark = pytest.mark.integration


def _provider_config():
    from answerdesk.contracts import ProviderConfig

    return ProviderConfig(
        preset="ollama",
        base_url="http://provider.test/v1",
        model="test-model",
        timeout_s=5.0,
        max_retries=0,
    )


def _sse_response(body: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"content-type": "text/event-stream"},
        content=body.encode("utf-8"),
    )


def test_truncated_provider_sse_is_an_empty_completion_not_a_hang() -> None:
    """A body that stops mid-frame must terminate quickly and never invent text.

    The LLM layer treats an unterminated frame as "the stream ended" (limitation L-3 in
    docs/robustness.md); the *service* is what makes that safe: an empty completion reaches
    the generation pipeline, which escalates PROVIDER_ERROR instead of displaying anything
    (see test_degenerate_provider_output_never_becomes_an_answer).
    """
    from answerdesk.llm.providers import HttpChatProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return _sse_response('data: {"choices":[{"delta":{"content":"hel')

    provider = HttpChatProvider(_provider_config(), transport=httpx.MockTransport(handler))
    chunks = list(provider.stream([_user_message("hi")]))
    assert "".join(chunk.delta for chunk in chunks) == ""
    assert chunks[-1].done is True


def test_provider_json_instead_of_sse_is_ignored_safely() -> None:
    from answerdesk.llm.providers import HttpChatProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    provider = HttpChatProvider(_provider_config(), transport=httpx.MockTransport(handler))
    chunks = list(provider.stream([_user_message("hi")]))
    assert "".join(chunk.delta for chunk in chunks) == ""
    assert chunks[-1].done is True


def test_unknown_sse_events_and_comments_are_ignored() -> None:
    from answerdesk.llm.providers import HttpChatProvider

    body = (
        ": keep-alive\n\n"
        'event: unknown\ndata: {"nonsense": true}\n\n'
        'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _sse_response(body)

    provider = HttpChatProvider(_provider_config(), transport=httpx.MockTransport(handler))
    chunks = list(provider.stream([_user_message("hi")]))
    text = "".join(chunk.delta for chunk in chunks if chunk.delta)
    assert text == "hello world"
    assert chunks[-1].done is True


def test_provider_http_500_on_the_stream_is_retryable() -> None:
    from answerdesk.errors import ProviderError
    from answerdesk.llm.providers import HttpChatProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    provider = HttpChatProvider(_provider_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as excinfo:
        list(provider.stream([_user_message("hi")]))
    assert getattr(excinfo.value, "retryable", False) is True


def test_mid_stream_failure_keeps_the_frame_contract(make_app) -> None:
    app, _runtime, _provider = make_app(provider=RaisingProvider(TimeoutError("slow provider")))

    async def _scenario(client, _runtime):
        response = await client.post("/v1/chat", json={"message": ANSWERABLE})
        frames = parse_sse(response.text)
        assert_clean_stream(frames)
        assert frames[0].event == "error"  # no retrieval frame was ever sent
        assert frames[0].data["error"]["code"] == "provider_timeout"

    scenario(app, _scenario)


def test_every_frame_carries_the_same_trace_id(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        response = await client.post("/v1/chat", json={"message": ANSWERABLE})
        frames = parse_sse(response.text)
        trace_ids = {frame.data.get("trace_id") for frame in frames}
        assert len(trace_ids) == 1
        assert response.headers["X-Trace-Id"] in trace_ids

    scenario(app, _scenario)


# -- cancellation ------------------------------------------------------------------------
def _scope(body: bytes) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat",
        "raw_path": b"/v1/chat",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"fuzz.test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 51234),
        "server": ("fuzz.test", 80),
    }


async def _drive_then_disconnect(app, body: bytes) -> dict:
    """Send the request, deliver the body, then answer ``http.disconnect``."""
    state = {"started": False, "chunks": 0, "messages": 0}
    delivered = {"body": False}

    async def receive() -> dict:
        if not delivered["body"]:
            delivered["body"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            state["started"] = True
        elif message["type"] == "http.response.body":
            state["chunks"] += 1
        state["messages"] += 1

    await app(_scope(body), receive, send)
    return state


def test_client_disconnect_does_not_raise_and_the_service_stays_up(make_app) -> None:
    app, _runtime, _provider = make_app()
    body = json.dumps({"message": ANSWERABLE}).encode()

    async def _scenario():
        # One lifespan: the app closes its runtime on shutdown, so the follow-up request
        # has to happen before the context manager exits.
        async with app.router.lifespan_context(app):
            state = await asyncio.wait_for(_drive_then_disconnect(app, body), timeout=30.0)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://fuzz.test"
            ) as client:
                response = await client.post(
                    "/v1/chat", json={"message": ANSWERABLE, "stream": False}
                )
                assert response.status_code == 200
                health = await client.get("/healthz")
                assert health.status_code == 200
        assert state["started"] is True

    run(_scenario())


def test_cancelled_stream_releases_the_idempotency_lock(make_app) -> None:
    app, _runtime, _provider = make_app()
    key = {"Idempotency-Key": "cancel-me"}

    async def _scenario(client, _runtime):
        async def reader() -> None:
            async with client.stream(
                "POST", "/v1/chat", json={"message": ANSWERABLE}, headers=key
            ) as response:
                assert response.status_code == 200
                async for _chunk in response.aiter_text():
                    break  # stop reading; the response is closed here

        await asyncio.wait_for(reader(), timeout=30.0)
        # The guard must be free again: a retry with the same key completes.
        response = await client.post(
            "/v1/chat", json={"message": ANSWERABLE, "stream": False}, headers=key
        )
        assert response.status_code == 200
        assert response.headers.get("X-Idempotent-Replay") in {None, "true"}

    scenario(app, _scenario, raise_app_exceptions=False)


def test_provider_timeout_is_fast(make_app) -> None:
    """A 60-second model timeout must not become a 60-second request in tests."""
    import time

    app, _runtime, _provider = make_app(provider=RaisingProvider(TimeoutError("timed out")))

    async def _scenario(client, _runtime):
        started = time.perf_counter()
        response = await client.post("/v1/chat", json={"message": ANSWERABLE, "stream": False})
        elapsed = time.perf_counter() - started
        assert response.status_code == 504
        assert elapsed < 5.0, elapsed

    scenario(app, _scenario, raise_app_exceptions=False)


def _user_message(content: str):
    from answerdesk.contracts import Message

    return Message(role="user", content=content)
