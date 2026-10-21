"""Proxy tolerance for the hosted embedder: an unusable ambient proxy must not crash it.

The defect this file pins down: ``httpx.Client(trust_env=True)`` — the right default, because
a corporate proxy must keep working — raises ``ValueError: Unknown scheme for proxy URL`` at
*construction* when the ambient proxy is ``socks4://…`` (this machine) or ``socks5://…``
without the optional ``socksio`` package. That killed live retrieval before the first request.
The policy under test: ignore the unusable ambient configuration with a warning, keep a valid
one, and never let either path raise a bare ``ValueError``.

    # AT-RET-02  # AT-RET-07  # AT-NFR-PORT-01
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable

import httpx
import pytest

from answerdesk.config import load_settings
from answerdesk.index.embedders import ApiEmbedder, default_embedder
from answerdesk.index.http import PROXY_ENV_VARS, build_http_client

#: The value this machine exposes: a scheme httpx has no handler for.
UNUSABLE_PROXY = "socks4://127.0.0.1:10808"
Handler = Callable[[httpx.Request], httpx.Response]


def _break_ambient_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the ambient proxy unusable *regardless of the shell* (t50, R-16).

    Setting the proxy variables alone is not enough: a developer who followed the README's
    `NO_PROXY='*'` workaround has that variable exported, httpx then bypasses the proxy, and
    the fallback under test never triggers. Clearing it makes the scenario depend on what the
    test sets, not on the environment that ran it.
    """
    for name in PROXY_ENV_VARS:
        monkeypatch.setenv(name, UNUSABLE_PROXY)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)


def _clear_ambient_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _three_dim_vectors(request: httpx.Request) -> httpx.Response:
    import json

    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "data": [
                {"index": index, "embedding": [0.5, 0.25, 0.125]}
                for index, _text in enumerate(body["input"])
            ]
        },
    )


def test_httpx_alone_still_raises_on_this_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reproduction: without the factory, construction itself fails."""
    _break_ambient_proxy(monkeypatch)
    with pytest.raises(ValueError, match="Unknown scheme"):
        httpx.Client(timeout=5.0)


def test_build_http_client_falls_back_when_the_ambient_proxy_is_unusable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _break_ambient_proxy(monkeypatch)
    calls: list[dict[str, object]] = []
    real_client = httpx.Client

    def spy(*args: object, **kwargs: object) -> httpx.Client:
        calls.append(dict(kwargs))
        return real_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "Client", spy)
    with caplog.at_level(logging.WARNING, logger="answerdesk.index.http"):
        client = build_http_client(timeout=5.0)

    assert len(calls) == 2, calls
    assert calls[0].get("trust_env", True) is True  # ambient proxy was respected first
    assert calls[1]["trust_env"] is False  # then deliberately ignored
    assert client.trust_env is False
    assert "unusable ambient proxy" in caplog.text
    client.close()


def test_build_http_client_keeps_an_explicit_valid_proxy(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A corporate proxy must keep working: no fallback, no warning."""
    _clear_ambient_proxy(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    calls: list[dict[str, object]] = []
    real_client = httpx.Client

    def spy(*args: object, **kwargs: object) -> httpx.Client:
        calls.append(dict(kwargs))
        return real_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "Client", spy)
    with caplog.at_level(logging.WARNING, logger="answerdesk.index.http"):
        client = build_http_client(timeout=5.0)

    assert len(calls) == 1
    assert calls[0].get("trust_env", True) is True
    assert client.trust_env is True
    assert "unusable ambient proxy" not in caplog.text
    client.close()


def test_embedder_runs_under_a_broken_ambient_proxy_with_mocktransport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter executes end-to-end (no socket) while the ambient proxy is unusable."""
    _break_ambient_proxy(monkeypatch)
    embedder = ApiEmbedder(
        base_url="https://api.example.test/v1",
        model="bge-m3",
        batch_size=2,
        transport=httpx.MockTransport(_three_dim_vectors),
        sleep=lambda _seconds: None,
        rng=random.Random(0),
    )
    vectors = embedder.embed_documents(["a", "bb", "ccc"])
    assert len(vectors) == 3
    assert embedder.request_count == 2  # 2 + 1 inputs
    assert embedder.dimension == 3
    embedder.close()


def test_default_embedder_reaches_the_network_layer_under_a_broken_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration-style: config -> ApiEmbedder -> request, under the bad ambient env.

    The offline gate forbids sockets, so the observable proof that the client was built is
    the *guard's* RuntimeError instead of httpx's construction-time ``ValueError``.
    """
    _break_ambient_proxy(monkeypatch)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-test-not-a-real-key")
    settings = load_settings(
        profile="api",
        overrides={
            "embedding": {
                "backend": "api",
                "api_base_url": "https://api.neuraldeep.ru/v1",
                "api_model": "bge-m3",
            }
        },
    )
    embedder = default_embedder(settings)
    assert isinstance(embedder, ApiEmbedder)

    with pytest.raises(RuntimeError) as error:
        embedder.embed_query("Сколько стоит обслуживание карты?")
    message = str(error.value)
    assert "outbound network access is forbidden" in message
    assert "Unknown scheme" not in message
    assert not isinstance(error.value, ValueError)


def test_explicit_client_injection_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller-supplied client (tests, custom transports) is used verbatim."""
    _break_ambient_proxy(monkeypatch)
    injected = httpx.Client(transport=httpx.MockTransport(_three_dim_vectors), trust_env=False)
    embedder = ApiEmbedder(
        base_url="https://api.example.test/v1",
        model="bge-m3",
        batch_size=4,
        client=injected,
        sleep=lambda _seconds: None,
        rng=random.Random(0),
    )
    assert embedder.embed_query("x") == [0.5, 0.25, 0.125]
    assert embedder.request_count == 1
