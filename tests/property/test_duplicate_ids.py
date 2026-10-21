"""Duplicate identifiers, concurrent retries and replay consistency.

An ``Idempotency-Key`` is a promise: N concurrent retries must produce one turn, one stored
response and N identical bodies. Before the fix (F-03) the second retry hit the UNIQUE index
on ``turns.idempotency_key`` and the client saw a 500 raised by its own retry.

    # AT-SVC-05  # AT-OPS-07
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fuzz_support import (
    ANSWERABLE,
    PoisonedRetriever,
    assert_taxonomy,
    scenario,
)

pytestmark = pytest.mark.integration


def test_concurrent_retries_of_one_key_run_the_turn_once(make_app) -> None:
    app, _runtime, provider = make_app()
    headers = {"Idempotency-Key": "retry-me"}
    payload = {"message": ANSWERABLE, "stream": False}

    async def _scenario(client, _runtime):
        responses = await asyncio.gather(
            *[client.post("/v1/chat", json=payload, headers=headers) for _ in range(4)]
        )
        assert [response.status_code for response in responses] == [200, 200, 200, 200]
        bodies = {json.dumps(response.json(), sort_keys=True) for response in responses}
        assert len(bodies) == 1, "a replay must return the same JSON document"
        assert len(provider.calls) == 1, f"the turn ran {len(provider.calls)} times"
        replays = [r.headers.get("X-Idempotent-Replay") for r in responses]
        assert replays.count("true") == 3
        assert len({response.headers["X-Trace-Id"] for response in responses}) == 1

    scenario(app, _scenario)


def test_same_key_with_a_different_body_is_a_conflict(make_app) -> None:
    app, _runtime, _provider = make_app()
    headers = {"Idempotency-Key": "conflict-me"}

    async def _scenario(client, _runtime):
        first = await client.post("/v1/chat", json={"message": ANSWERABLE}, headers=headers)
        assert first.status_code == 200
        second = await client.post("/v1/chat", json={"message": ANSWERABLE + "?"}, headers=headers)
        error = assert_taxonomy(second, expected={409})
        assert error["code"] == "idempotency_conflict"

    scenario(app, _scenario)


def test_distinct_keys_in_the_same_session_both_run(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        session = await client.post("/v1/sessions")
        session_id = session.json()["session_id"]
        responses = await asyncio.gather(
            client.post(
                "/v1/chat",
                json={"message": "обслуживание карты", "session_id": session_id, "stream": False},
                headers={"Idempotency-Key": "key-a"},
            ),
            client.post(
                "/v1/chat",
                json={"message": "тарифы", "session_id": session_id, "stream": False},
                headers={"Idempotency-Key": "key-b"},
            ),
        )
        assert all(response.status_code == 200 for response in responses), [
            response.text[:120] for response in responses
        ]

    scenario(app, _scenario)


def test_the_same_trace_id_on_two_requests_keeps_both_traces(make_app) -> None:
    app, runtime, _provider = make_app()
    trace_id = "tr-0123456789abcdef01234567"

    async def _scenario(client, _runtime):
        for message in ("обслуживание карты", "тарифы"):
            response = await client.post(
                "/v1/chat",
                json={"message": message, "stream": False},
                headers={"X-Trace-Id": trace_id},
            )
            assert response.status_code == 200
            assert response.headers["X-Trace-Id"] == trace_id
        trace = await client.get(f"/v1/traces/{trace_id}")
        assert trace.status_code == 200

    scenario(app, _scenario)
    assert runtime.tracer.read(trace_id, kind="turn") is not None


def test_duplicate_feedback_is_recorded_without_a_crash(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        turn = await client.post("/v1/chat", json={"message": ANSWERABLE, "stream": False})
        trace_id = turn.json()["trace_id"]
        for rating in ("up", "up", "down"):
            response = await client.post(
                "/v1/feedback", json={"trace_id": trace_id, "rating": rating}
            )
            assert response.status_code in {200, 201}, response.text[:200]
            assert_taxonomy(response)
        health = await client.get("/healthz")
        assert health.status_code == 200

    scenario(app, _scenario)


def test_duplicate_escalation_tickets_do_not_crash(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        turn = await client.post("/v1/chat", json={"message": "тарифы", "stream": False})
        trace_id = turn.json()["trace_id"]
        first = await client.post(
            "/v1/escalations", json={"trace_id": trace_id, "reason": "customer asked for a human"}
        )
        assert_taxonomy(first)
        second = await client.post(
            "/v1/escalations", json={"trace_id": trace_id, "reason": "customer asked again"}
        )
        assert_taxonomy(second, allow_500=False)
        assert second.status_code < 500

    scenario(app, _scenario)


def test_duplicate_chunk_ids_in_retrieval_do_not_duplicate_citations(make_app) -> None:
    _app, runtime, _provider = make_app()
    duplicated = PoisonedRetriever(runtime.retriever, duplicate=True)
    app2, _runtime2, _provider2 = make_app(retriever=duplicated)

    async def _scenario(client, _runtime):
        response = await client.post("/v1/chat", json={"message": ANSWERABLE, "stream": False})
        assert response.status_code == 200
        citations = response.json()["answer"]["citations"]
        ids = [citation["citation_id"] for citation in citations]
        assert len(ids) == len(set(ids)), ids
        assert len(citations) <= 2

    scenario(app2, _scenario)
