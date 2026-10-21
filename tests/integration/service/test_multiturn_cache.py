"""Regression: a second distinct turn must not fail when the cache is non-empty.

The service used to pass the raw cosine similarity of a cache *miss* into
``CacheInfo(similarity=...)``, whose frozen contract is 0..1. The hashing embedder
can produce a negative cosine for an unrelated query, so the second distinct turn
in a process raised ``pydantic.ValidationError`` and the SSE stream degraded to an
``error`` event (a non-streaming call would have been a 500).

    # AT-SVC-01
"""

from __future__ import annotations

import json

import httpx
from service_support import ANSWERABLE_QUESTION, UNANSWERABLE_QUESTION, parse_sse

CHAT = "/v1/chat"


def test_second_distinct_turn_after_a_cache_put_is_not_an_error(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> list[httpx.Response]:
        first = await client.post(
            CHAT,
            json={"message": ANSWERABLE_QUESTION},
            headers={"Accept": "text/event-stream"},
        )
        assert first.status_code == 200, first.text
        # The first turn must have populated the semantic cache for this to be the bug.
        second = await client.post(
            CHAT,
            json={"message": UNANSWERABLE_QUESTION},
            headers={"Accept": "text/event-stream"},
        )
        assert second.status_code == 200, second.text
        return [first, second]

    first, second = make_client(scenario)
    for response in (first, second):
        frames = parse_sse(response.text)
        assert frames, response.text
        assert frames[-1].event == "done", [frame.event for frame in frames]
        assert not any(frame.event == "error" for frame in frames), response.text


def test_second_distinct_turn_on_the_json_path_returns_200(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> httpx.Response:
        first = await client.post(
            CHAT,
            json={"message": ANSWERABLE_QUESTION, "stream": False},
            headers={"Accept": "application/json"},
        )
        assert first.status_code == 200, first.text
        second = await client.post(
            CHAT,
            json={"message": UNANSWERABLE_QUESTION, "stream": False},
            headers={"Accept": "application/json"},
        )
        return second

    second = make_client(scenario)
    assert second.status_code == 200, second.text
    payload = json.loads(second.text)
    assert payload["cache"]["similarity"] is None or 0.0 <= payload["cache"]["similarity"] <= 1.0
