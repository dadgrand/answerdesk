"""A mock OpenAI-compatible HTTP server for provider tests.

Real sockets are forbidden in the suite (``tests/conftest.py`` patches
``socket.connect``, INV-08 / NFR-DET-01), so "a local mock HTTP server" is implemented at
the httpx transport layer: :class:`MockOpenAI` is an HTTP handler that receives real
``httpx.Request`` objects (URL, headers, JSON body) and returns real ``httpx.Response``
objects (status, headers, SSE byte streams). Everything the provider does over the wire
is therefore exercised — request construction, status handling, retry headers, SSE
parsing — with zero network access.

A genuinely socket-based loopback test exists too, but only runs when
``ANSWERDESK_ALLOW_NETWORK=1`` (``tests/integration/generation/test_local_http_server.py``).

    # AT-PORT-01
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx


def completion_payload(
    text: str,
    *,
    model: str = "local-model",
    input_tokens: int = 12,
    output_tokens: int = 34,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """The smallest realistic non-streaming chat-completions response."""
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
    }


def sse_body(deltas: Sequence[str], *, model: str = "local-model") -> str:
    """Encode deltas as an SSE body, exactly as a chat-completions server streams them."""
    lines: list[str] = []
    for delta in deltas:
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
        }
        lines.append(f"data: {json.dumps(payload, ensure_ascii=False)}")
    final = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    lines.append(f"data: {json.dumps(final)}")
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


@dataclass
class Reply:
    """One scripted server reply (or transport-level failure)."""

    status: int = 200
    payload: dict[str, Any] | None = None
    text: str | None = None
    sse: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    raise_exc: BaseException | None = None


class MockOpenAI:
    """Scripted OpenAI-compatible server. Pops one queued reply per request."""

    def __init__(self, replies: Sequence[Reply] | None = None) -> None:
        self.replies: list[Reply] = list(replies or [])
        self.requests: list[httpx.Request] = []
        self.default_text = "ok"

    # -- scripting ----------------------------------------------------------------
    def queue(self, reply: Reply) -> MockOpenAI:
        self.replies.append(reply)
        return self

    def queue_json(
        self,
        text: str,
        *,
        status: int = 200,
        model: str = "local-model",
        input_tokens: int = 12,
        output_tokens: int = 34,
        headers: Mapping[str, str] | None = None,
    ) -> MockOpenAI:
        return self.queue(
            Reply(
                status=status,
                payload=completion_payload(
                    text, model=model, input_tokens=input_tokens, output_tokens=output_tokens
                ),
                headers=dict(headers or {}),
            )
        )

    def queue_status(
        self, status: int, *, body: str = "error", headers: Mapping[str, str] | None = None
    ) -> MockOpenAI:
        return self.queue(Reply(status=status, text=body, headers=dict(headers or {})))

    def queue_sse(self, deltas: Sequence[str], *, model: str = "local-model") -> MockOpenAI:
        return self.queue(
            Reply(
                status=200,
                sse=sse_body(deltas, model=model),
                headers={"content-type": "text/event-stream"},
            )
        )

    def queue_timeout(self) -> MockOpenAI:
        return self.queue(Reply(raise_exc=httpx.ReadTimeout("scripted timeout")))

    def queue_transport_error(self) -> MockOpenAI:
        return self.queue(Reply(raise_exc=httpx.ConnectError("scripted connection failure")))

    # -- httpx plumbing -----------------------------------------------------------
    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if self.replies else Reply(text=self.default_text)
        if reply.raise_exc is not None:
            raise reply.raise_exc
        headers = {"content-type": "application/json", **dict(reply.headers)}
        if reply.sse is not None:
            headers["content-type"] = "text/event-stream"
            return httpx.Response(reply.status, content=reply.sse.encode("utf-8"), headers=headers)
        if reply.payload is not None:
            return httpx.Response(reply.status, json=reply.payload, headers=headers)
        return httpx.Response(reply.status, text=reply.text or "", headers=headers)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    # -- assertions helpers -------------------------------------------------------
    @property
    def call_count(self) -> int:
        return len(self.requests)

    def last_body(self) -> dict[str, Any]:
        body = self.requests[-1].content
        return json.loads(body)

    def last_headers(self) -> httpx.Headers:
        return self.requests[-1].headers

    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]


__all__ = ["MockOpenAI", "Reply", "completion_payload", "sse_body"]
