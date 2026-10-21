"""Hosted embedder: batching, retries, dimension pinning, error mapping, key hygiene.

Every test drives :class:`answerdesk.index.embedders.ApiEmbedder` through
``httpx.MockTransport``, so the suite stays network-free (INV-08) while still exercising
the real request/response path, the retry loop and the JSON shape handling.

    # AT-RET-02  # AT-RET-05  # AT-NFR-SEC-02
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable

import httpx
import pytest

from answerdesk.errors import ConfigError
from answerdesk.index.embedders import ApiEmbedder

Handler = Callable[[httpx.Request], httpx.Response]
BASE_URL = "https://api.example.test/v1"


def _embedder(handler: Handler, **overrides: object) -> ApiEmbedder:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    params: dict[str, object] = {
        "base_url": BASE_URL,
        "model": "bge-m3",
        "batch_size": 2,
        "max_retries": 2,
        "client": client,
        "sleep": lambda _seconds: None,
        "rng": random.Random(0),
    }
    params.update(overrides)
    return ApiEmbedder(**params)  # type: ignore[arg-type]


def _vectors(*vectors: list[float]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "bge-m3",
            "data": [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)],
        },
    )


def _recording(handler: Handler, sink: list[httpx.Request]) -> Handler:
    def _inner(request: httpx.Request) -> httpx.Response:
        sink.append(request)
        return handler(request)

    return _inner


def test_batches_requests_and_preserves_input_order() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return _vectors(*[[float(len(text)), 1.0] for text in body["input"]])

    embedder = _embedder(_recording(handler, requests))
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    vectors = embedder.embed_documents(texts)

    assert embedder.request_count == 3  # ceil(5 / 2)
    assert [json.loads(request.content)["input"] for request in requests] == [
        ["a", "bb"],
        ["ccc", "dddd"],
        ["eeeee"],
    ]
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert embedder.dimension == 2


def test_server_side_order_is_corrected_by_the_index_field() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        rows = [
            {"index": index, "embedding": [float(len(text)), 0.0]}
            for index, text in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(rows))})

    embedder = _embedder(handler)
    vectors = embedder.embed_documents(["a", "bb", "ccc", "dddd"])
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0, 4.0]


def test_retries_429_then_succeeds() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return _vectors([1.0, 0.0])

    embedder = _embedder(handler, sleep=sleeps.append)
    assert embedder.embed_documents(["x"]) == [[1.0, 0.0]]
    assert calls["count"] == 2
    assert len(sleeps) == 1
    assert sleeps[0] >= 2.0  # Retry-After is honoured as a floor


def test_retries_5xx_until_the_budget_is_spent_then_raises() -> None:
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream busy")

    embedder = _embedder(handler, sleep=sleeps.append, max_retries=2)
    with pytest.raises(ConfigError, match="failed after 3 attempt"):
        embedder.embed_documents(["x"])
    assert embedder.request_count == 3  # 1 attempt + 2 retries
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0]  # exponential backoff, not a fixed sleep


def test_transport_errors_are_retried_then_mapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    embedder = _embedder(handler, max_retries=1)
    with pytest.raises(ConfigError, match="ConnectError"):
        embedder.embed_documents(["x"])
    assert embedder.request_count == 2


def test_401_is_actionable_and_never_contains_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-live-0123456789abcdef"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid api key")

    embedder = _embedder(handler, api_key=_secret(secret), max_retries=3)
    with pytest.raises(ConfigError) as error:
        embedder.embed_documents(["x"])
    message = str(error.value)
    assert "api_key_env" in message
    assert secret not in message
    assert embedder.request_count == 1  # 4xx is never retried


def test_api_key_is_sent_as_a_bearer_token() -> None:
    requests: list[httpx.Request] = []
    secret = "sk-live-0123456789abcdef"

    def handler(request: httpx.Request) -> httpx.Response:
        return _vectors([1.0, 0.0])

    embedder = _embedder(_recording(handler, requests), api_key=_secret(secret))
    embedder.embed_documents(["x"])
    assert requests[0].headers["Authorization"] == f"Bearer {secret}"
    assert secret not in repr(embedder)


def test_no_authorization_header_without_a_key() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return _vectors([1.0, 0.0])

    embedder = _embedder(_recording(handler, requests))
    embedder.embed_documents(["x"])
    assert "Authorization" not in requests[0].headers


def test_dimension_is_discovered_from_the_first_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return _vectors(*[[1.0, 0.0, 0.5] for _ in body["input"]])

    embedder = _embedder(handler, batch_size=3)
    assert embedder.dimension == 0
    embedder.embed_documents(["a", "bb", "ccc"])
    assert embedder.dimension == 3


def test_declared_dimension_mismatch_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _vectors([1.0, 0.0, 0.5])

    embedder = _embedder(handler, declared_dimension=4)
    with pytest.raises(ConfigError, match="api_dimension"):
        embedder.embed_documents(["x"])


def test_dimension_changing_mid_run_is_refused() -> None:
    widths = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        widths["count"] += 1
        width = 3 if widths["count"] == 1 else 5
        return _vectors([1.0] * width)

    embedder = _embedder(handler, batch_size=1)
    embedder.embed_documents(["a"])
    assert embedder.dimension == 3
    with pytest.raises(ConfigError, match="changed dimensionality"):
        embedder.embed_documents(["b"])


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(200, text="<html>not json</html>"), "non-JSON"),
        (httpx.Response(200, json={"model": "bge-m3"}), "no 'data' array"),
        (httpx.Response(200, json={"data": []}), "returned 0 vectors for 1 inputs"),
        (
            httpx.Response(200, json={"data": [{"index": 0, "embedding": []}]}),
            "empty vector",
        ),
        (
            httpx.Response(200, json={"data": [{"index": 0, "embedding": ["a", "b"]}]}),
            "non-numeric vector",
        ),
        (httpx.Response(200, json={"data": [{"index": 0}]}), "without 'embedding'"),
        (httpx.Response(200, json=["not", "an", "object"]), "not an object"),
    ],
)
def test_malformed_payloads_map_to_config_error(response: httpx.Response, match: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return response

    embedder = _embedder(handler, batch_size=1)
    with pytest.raises(ConfigError, match=match):
        embedder.embed_documents(["x"])


def test_empty_input_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request expected")

    embedder = _embedder(handler)
    assert embedder.embed_documents([]) == []
    assert embedder.request_count == 0


def test_blank_input_is_refused_before_the_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request expected")

    embedder = _embedder(handler)
    with pytest.raises(ConfigError, match="empty inputs"):
        embedder.embed_documents(["ok", "   "])
    assert embedder.request_count == 0


def test_embed_query_uses_the_document_path() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return _vectors([0.5, 0.5])

    embedder = _embedder(_recording(handler, requests))
    assert embedder.embed_query("карта") == [0.5, 0.5]
    assert json.loads(requests[0].content)["input"] == ["карта"]


def test_constructor_validates_its_arguments() -> None:
    with pytest.raises(ConfigError, match="api_base_url"):
        ApiEmbedder(base_url="", model="bge-m3")
    with pytest.raises(ConfigError, match="model name"):
        ApiEmbedder(base_url=BASE_URL, model="")
    with pytest.raises(ConfigError, match="batch_size"):
        ApiEmbedder(base_url=BASE_URL, model="bge-m3", batch_size=0)


def test_name_is_part_of_the_identity() -> None:
    embedder = ApiEmbedder(base_url=BASE_URL, model="bge-m3")
    assert embedder.name == "api:bge-m3"
    assert ApiEmbedder(base_url=BASE_URL, model="m", name="custom").name == "custom"


def _secret(value: str):
    from pydantic import SecretStr

    return SecretStr(value)
