"""A genuinely socket-based loopback server test (opt-in).

Every other provider test runs on ``httpx.MockTransport`` because the default suite blocks
sockets (INV-08 / NFR-DET-01). This test exists to prove that the *real* transport works
over a real TCP connection to a local OpenAI-compatible server with **no API key** — the
"point it at a local model" promise of NFR-PORT-01. It only runs with the documented
escape hatch:

    ANSWERDESK_ALLOW_NETWORK=1 uv run pytest -q \
        tests/integration/generation/test_local_http_server.py

It binds ``127.0.0.1`` only; nothing leaves the machine.

    # AT-PORT-01
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from answerdesk.contracts import Message, Role
from answerdesk.llm.presets import provider_config
from answerdesk.llm.providers import HttpChatProvider

pytestmark = pytest.mark.skipif(
    os.environ.get("ANSWERDESK_ALLOW_NETWORK") != "1",
    reason="loopback socket test: set ANSWERDESK_ALLOW_NETWORK=1 to run it",
)

SEEN: list[dict[str, Any]] = []


class LoopbackHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible chat-completions server."""

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        SEEN.append(
            {
                "path": self.path,
                "body": body,
                "authorization": self.headers.get("authorization"),
            }
        )
        payload = {
            "model": body.get("model", "local-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args: object) -> None:  # keep pytest output clean
        return


def test_real_loopback_server_is_reachable_without_a_key() -> None:
    """# AT-PORT-01 — real TCP, real JSON, zero keys, one client code path."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), LoopbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider: HttpChatProvider | None = None
    try:
        port = server.server_address[1]
        config = provider_config(
            "vllm", base_url=f"http://127.0.0.1:{port}/v1", timeout_s=5.0, max_retries=0
        )
        assert config.api_key is None
        provider = HttpChatProvider(config)
        completion = provider.complete([Message(role=Role.USER, content="ping")])

        assert completion.text == "pong"
        assert completion.input_tokens == 3
        assert SEEN[-1]["path"] == "/v1/chat/completions"
        assert SEEN[-1]["authorization"] is None
        assert SEEN[-1]["body"]["model"] == config.model
    finally:
        if provider is not None:
            provider.close()
        server.shutdown()
        server.server_close()
