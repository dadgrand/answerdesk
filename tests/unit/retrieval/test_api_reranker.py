"""Hosted reranker: Cohere-style payload, index mapping, batching, retries, selection.

Driven entirely through ``httpx.MockTransport`` — no network in CI (INV-08).

    # AT-RET-07  # AT-RET-08
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from answerdesk.contracts import Channel, RetrievedPassage
from answerdesk.errors import ConfigError
from answerdesk.rerank.api import ApiReranker, api_reranker_from_settings
from answerdesk.rerank.cross_encoder import (
    LazyCrossEncoderReranker,
    LexicalOverlapReranker,
    build_reranker,
)

Handler = Callable[[httpx.Request], httpx.Response]
BASE_URL = "https://api.example.test/v1"


def _passage(chunk_id: str, score: float = 1.0, text: str | None = None) -> RetrievedPassage:
    from answerdesk.contracts import Chunk

    body = text or f"text for {chunk_id}"
    return RetrievedPassage(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=f"doc-{chunk_id}",
            text=body,
            start_char=0,
            end_char=len(body),
            token_count=len(body.split()),
        ),
        channel=Channel.BM25,
        score=score,
        rank=1,
    )


def _reranker(handler: Handler, **overrides: object) -> ApiReranker:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    params: dict[str, object] = {
        "base_url": BASE_URL,
        "model": "bge-reranker",
        "batch_size": 16,
        "max_retries": 2,
        "client": client,
        "sleep": lambda _seconds: None,
        "rng": random.Random(0),
    }
    params.update(overrides)
    return ApiReranker(**params)  # type: ignore[arg-type]


def _results(*pairs: tuple[int, float]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"results": [{"index": index, "relevance_score": score} for index, score in pairs]},
    )


def _echo_scores(request: httpx.Request) -> httpx.Response:
    """Score every document by its position, best first (a deterministic fake server)."""
    body = json.loads(request.content)
    count = len(body["documents"])
    return _results(*[(index, 1.0 - index / (count + 1)) for index in range(count)])


def test_posts_a_cohere_style_payload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _results((0, 0.2), (1, 0.9))

    reranker = _reranker(handler)
    passages = [_passage("a"), _passage("b")]
    ranked = reranker.rerank("query text", passages, 2)

    body = json.loads(requests[0].content)
    assert requests[0].url.path.endswith("/v1/rerank")
    assert body["model"] == "bge-reranker"
    assert body["query"] == "query text"
    assert body["documents"] == ["text for a", "text for b"]
    assert body["top_n"] == 2
    assert [passage.chunk.chunk_id for passage in ranked] == ["b", "a"]
    assert [passage.rank for passage in ranked] == [1, 2]
    assert all(passage.stage == "rerank" for passage in ranked)
    assert ranked[0].score == pytest.approx(0.9)
    assert ranked[0].contributions["rerank"] == pytest.approx(0.9)
    assert ranked[0].contributions["fused_score"] == pytest.approx(1.0)


def test_results_are_mapped_by_index_not_by_position() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _results((2, 0.7), (0, 0.3))  # server answers out of order, partially

    ranked = _reranker(handler).rerank("q", [_passage("a"), _passage("b"), _passage("c")], 3)
    assert [(passage.chunk.chunk_id, round(passage.score, 3)) for passage in ranked] == [
        ("c", 0.7),
        ("a", 0.3),
    ]


def test_shuffled_result_order_does_not_change_the_output() -> None:
    payloads = [
        _results((0, 0.1), (1, 0.5), (2, 0.9)),
        _results((2, 0.9), (0, 0.1), (1, 0.5)),
        _results((1, 0.5), (2, 0.9), (0, 0.1)),
    ]
    outputs = []
    for payload in payloads:
        reranker = _reranker(lambda request, payload=payload: payload)
        ranked = reranker.rerank("q", [_passage("a"), _passage("b"), _passage("c")], 3)
        outputs.append([passage.chunk.chunk_id for passage in ranked])
    assert outputs[0] == outputs[1] == outputs[2] == ["c", "b", "a"]


def test_top_n_truncates_after_merging() -> None:
    reranker = _reranker(_echo_scores)
    ranked = reranker.rerank("q", [_passage(f"p{index}") for index in range(6)], 3)
    assert [passage.chunk.chunk_id for passage in ranked] == ["p0", "p1", "p2"]
    assert [passage.rank for passage in ranked] == [1, 2, 3]


def test_batching_merges_global_indices() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Score by the *global* passage number embedded in the text, not by batch position."""
        requests.append(request)
        body = json.loads(request.content)
        rows = [
            (index, 1.0 - int(document.rsplit("p", 1)[1]) / 10.0)
            for index, document in enumerate(body["documents"])
        ]
        return _results(*rows)

    reranker = _reranker(handler, batch_size=2)
    passages = [_passage(f"p{index}") for index in range(5)]
    ranked = reranker.rerank("q", passages, 5)

    assert reranker.request_count == 3
    assert [len(json.loads(request.content)["documents"]) for request in requests] == [2, 2, 1]
    # Batching is invisible: the merged ranking is the same one an unbatched call gives.
    assert [passage.chunk.chunk_id for passage in ranked] == [
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
    ]
    assert [passage.score for passage in ranked] == sorted(
        (passage.score for passage in ranked), reverse=True
    )


def test_retries_5xx_then_succeeds() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(500, text="boom")
        return _results((0, 0.5))

    ranked = _reranker(handler, sleep=sleeps.append).rerank("q", [_passage("a")], 1)
    assert calls["count"] == 2
    assert len(sleeps) == 1
    assert [passage.chunk.chunk_id for passage in ranked] == ["a"]


def test_retry_budget_exhaustion_maps_to_config_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    reranker = _reranker(handler, max_retries=1)
    with pytest.raises(ConfigError, match="failed after 2 attempt"):
        reranker.rerank("q", [_passage("a")], 1)
    assert reranker.request_count == 2


def test_401_is_actionable_and_hides_the_key() -> None:
    secret = "sk-live-0123456789abcdef"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    reranker = _reranker(handler, api_key=SecretStr(secret), max_retries=3)
    with pytest.raises(ConfigError) as error:
        reranker.rerank("q", [_passage("a")], 1)
    message = str(error.value)
    assert "api_key_env" in message
    assert secret not in message
    assert secret not in repr(reranker)


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(200, text="not json"), "non-JSON"),
        (httpx.Response(200, json={"results": "nope"}), "non-list 'results'"),
        (httpx.Response(200, json={"results": [{"index": 5, "relevance_score": 1.0}]}), "outside"),
        (
            httpx.Response(200, json={"results": [{"index": 0, "relevance_score": "high"}]}),
            "non-numeric",
        ),
        (
            httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 0, "relevance_score": 0.5},
                        {"index": 0, "relevance_score": 0.4},
                    ]
                },
            ),
            "duplicate index 0",
        ),
        (httpx.Response(200, json={"results": [{"index": 0}]}), "non-numeric"),
        (httpx.Response(200, json={"results": ["row"]}), "non-object result row"),
    ],
)
def test_malformed_payloads_map_to_config_error(response: httpx.Response, match: str) -> None:
    reranker = _reranker(lambda request: response)
    with pytest.raises(ConfigError, match=match):
        reranker.rerank("q", [_passage("a")], 1)


def test_empty_results_are_refused_rather_than_silently_unranked() -> None:
    """A rerank stage that did not run must fail loudly, not return the fused order."""
    reranker = _reranker(lambda request: httpx.Response(200, json={"results": []}))
    with pytest.raises(ConfigError, match="no scored documents"):
        reranker.rerank("q", [_passage("a")], 1)


def test_no_documents_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request expected")

    reranker = _reranker(handler)
    assert reranker.rerank("q", [], 5) == []
    assert reranker.rerank("q", [_passage("a")], 0) == []
    assert reranker.request_count == 0


def test_constructor_validates_its_arguments() -> None:
    with pytest.raises(ConfigError, match="api_base_url"):
        ApiReranker(base_url="", model="bge-reranker")
    with pytest.raises(ConfigError, match="model name"):
        ApiReranker(base_url=BASE_URL, model="")
    with pytest.raises(ConfigError, match="batch_size"):
        ApiReranker(base_url=BASE_URL, model="bge-reranker", batch_size=0)


def test_name_reflects_the_model() -> None:
    assert ApiReranker(base_url=BASE_URL, model="bge-reranker").name == ("api-rerank:bge-reranker")


# -- selection through build_reranker ---------------------------------------------------


def _settings(enabled: bool = True, backend: str = "api", **embedding_overrides: object):
    from answerdesk.config import load_settings

    return load_settings(
        profile="api",
        overrides={
            "embedding": {"backend": "hashing", **embedding_overrides},
            "retrieval": {
                "rerank": {
                    "enabled": enabled,
                    "backend": backend,
                    "api_base_url": BASE_URL,
                    "api_model": "bge-reranker",
                    "api_max_retries": 1,
                }
            },
        },
    )


def test_build_reranker_selects_the_api_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-test-key")
    reranker = build_reranker(_settings())
    assert isinstance(reranker, ApiReranker)
    assert reranker.model == "bge-reranker"


def test_build_reranker_selects_the_lexical_and_cross_encoder_backends() -> None:
    assert isinstance(build_reranker(_settings(backend="lexical")), LexicalOverlapReranker)
    assert isinstance(build_reranker(_settings(backend="cross_encoder")), LazyCrossEncoderReranker)


def test_disabled_rerank_still_returns_the_offline_stand_in() -> None:
    settings = _settings(enabled=False)
    assert isinstance(build_reranker(settings), LexicalOverlapReranker)
    assert build_reranker(settings, offline_stand_in=False) is None


def test_api_backend_without_a_base_url_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANSWERDESK_RETRIEVAL__RERANK__API_BASE_URL", "")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-test-key")
    settings = _settings()
    settings.retrieval.rerank.api_base_url = None
    with pytest.raises(ConfigError, match="api_base_url"):
        build_reranker(settings)


def test_api_reranker_from_settings_requires_a_key_for_a_remote_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    settings = _settings()
    with pytest.raises(ConfigError, match="OPENAI_COMPATIBLE_API_KEY"):
        api_reranker_from_settings(settings)


def test_api_reranker_falls_back_to_the_model_name() -> None:
    """``api_model`` is optional: ``model`` is the fallback, so one key configures both."""
    settings = _settings()
    settings.retrieval.rerank.api_model = None
    settings.retrieval.rerank.api_base_url = "http://127.0.0.1:8080/v1"
    assert api_reranker_from_settings(settings).model == settings.retrieval.rerank.model
