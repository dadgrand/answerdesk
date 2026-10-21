"""Proxy tolerance for the hosted reranker — the same policy as the embedder's.

``ApiReranker`` builds its client through :func:`answerdesk.index.http.build_http_client`
too, so an unusable ambient proxy (``socks4://…``, or ``socks5://`` without ``socksio``)
is ignored with a warning instead of raising ``ValueError`` while constructing the client.

    # AT-RET-07
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable

import httpx
import pytest

from answerdesk.contracts import Channel, RetrievedPassage
from answerdesk.index.http import PROXY_ENV_VARS
from answerdesk.rerank.api import ApiReranker

UNUSABLE_PROXY = "socks4://127.0.0.1:10808"


def _break_ambient_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_ENV_VARS:
        monkeypatch.setenv(name, UNUSABLE_PROXY)


def _passage(chunk_id: str) -> RetrievedPassage:
    from answerdesk.contracts import Chunk

    text = f"text for {chunk_id}"
    return RetrievedPassage(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=f"doc-{chunk_id}",
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=3,
        ),
        channel=Channel.BM25,
        score=1.0,
        rank=1,
    )


def _scoring_handler(request: httpx.Request) -> httpx.Response:
    """Score by the *global* passage number in the text, so batching stays invisible."""
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "results": [
                {"index": index, "relevance_score": 1.0 - int(document.rsplit("p", 1)[1]) / 10.0}
                for index, document in enumerate(body["documents"])
            ]
        },
    )


Handler = Callable[[httpx.Request], httpx.Response]


def test_reranker_runs_under_a_broken_ambient_proxy_with_mocktransport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rerank call executes end-to-end (no socket) while the ambient proxy is unusable."""
    _break_ambient_proxy(monkeypatch)
    reranker = ApiReranker(
        base_url="https://api.example.test/v1",
        model="bge-reranker",
        batch_size=2,
        transport=httpx.MockTransport(_scoring_handler),
        sleep=lambda _seconds: None,
        rng=random.Random(0),
    )
    ranked = reranker.rerank("query", [_passage(f"p{index}") for index in range(3)], 2)

    assert reranker.request_count == 2  # 2 + 1 documents
    assert [passage.chunk.chunk_id for passage in ranked] == ["p0", "p1"]
    assert [passage.rank for passage in ranked] == [1, 2]
    assert all(passage.stage == "rerank" for passage in ranked)
    reranker.close()


def test_reranker_client_is_built_lazily_through_the_proxy_tolerant_factory(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    _break_ambient_proxy(monkeypatch)
    calls: list[dict[str, object]] = []
    real_client = httpx.Client

    def spy(*args: object, **kwargs: object) -> httpx.Client:
        calls.append(dict(kwargs))
        return real_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "Client", spy)
    reranker = ApiReranker(base_url="https://api.example.test/v1", model="bge-reranker")
    assert reranker.request_count == 0  # nothing is built at construction time
    with caplog.at_level(logging.WARNING, logger="answerdesk.index.http"):
        # The laziness *is* the contract: nothing is built until the first call.
        client = reranker._client_or_create()
    assert calls[0].get("trust_env", True) is True
    assert calls[-1]["trust_env"] is False
    assert client.trust_env is False
    assert "unusable ambient proxy" in caplog.text
    reranker.close()
