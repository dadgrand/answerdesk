"""Degenerate sizes: 1 MB bodies, the exact byte limit, huge headers and huge queries.

Three different boundaries are exercised on purpose: the HTTP body limit (413 before the
body is read), the schema limit (422), and the pipeline itself (a million-character query
still returns a decision in seconds, because retrieval is bounded by the query terms it can
use).

    # AT-OPS-07  # NFR-OPS-01
"""

from __future__ import annotations

import json
import time

import pytest
from fuzz_support import ANSWERABLE, PoisonedRetriever, assert_taxonomy, parse_sse, scenario

pytestmark = pytest.mark.integration

BODY_LIMIT = 262_144


def test_one_megabyte_body_is_413_before_parsing(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        body = b'{"message": "' + b"x" * (1024 * 1024) + b'"}'
        response = await client.post(
            "/v1/chat", content=body, headers={"Content-Type": "application/json"}
        )
        error = assert_taxonomy(response, expected={413})
        assert error["code"] == "payload_too_large"

    scenario(app, _scenario)


def test_body_limit_boundary(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        base = len(b'{"message": ""}')
        at_limit = b'{"message": "' + b"x" * (BODY_LIMIT - base) + b'"}'
        over_limit = b'{"message": "' + b"x" * (BODY_LIMIT - base + 1) + b'"}'
        assert len(at_limit) == BODY_LIMIT
        response = await client.post(
            "/v1/chat", content=at_limit, headers={"Content-Type": "application/json"}
        )
        # accepted by the body limit, rejected by the schema (message max_length=8000)
        assert response.status_code == 422, response.text[:200]
        response = await client.post(
            "/v1/chat", content=over_limit, headers={"Content-Type": "application/json"}
        )
        assert_taxonomy(response, expected={413})

    scenario(app, _scenario)


def test_huge_idempotency_key_is_rejected(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        response = await client.post(
            "/v1/chat",
            json={"message": ANSWERABLE},
            headers={"Idempotency-Key": "k" * 5000},
        )
        error = assert_taxonomy(response, expected={400})
        assert error["code"] == "bad_request"
        # a normal-length key is accepted
        response = await client.post(
            "/v1/chat",
            json={"message": ANSWERABLE},
            headers={"Idempotency-Key": "k" * 200},
        )
        assert response.status_code == 200

    scenario(app, _scenario)


def test_huge_trace_id_header_is_replaced(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        response = await client.post(
            "/v1/chat",
            json={"message": ANSWERABLE},
            headers={"X-Trace-Id": "x" * 100_000, "Idempotency-Key": "huge-trace"},
        )
        assert response.status_code == 200
        assert response.headers["X-Trace-Id"].startswith("tr-")

    scenario(app, _scenario)


def test_long_session_id_is_a_validation_error(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        response = await client.post("/v1/chat", json={"message": "hi", "session_id": "s" * 500})
        error = assert_taxonomy(response, expected={422})
        assert error["code"] == "validation_error"

    scenario(app, _scenario)


def test_maximum_message_length_is_accepted(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        response = await client.post("/v1/chat", json={"message": "я" * 8000, "stream": True})
        assert response.status_code == 200
        frames = parse_sse(response.text)
        assert frames[-1].event == "done"

    scenario(app, _scenario)


def test_million_character_query_reaches_a_decision(make_app) -> None:
    """The schema caps the HTTP message at 8 000; the pipeline itself must still be bounded.

    A one-million-character message (≈200k tokens) is fed to the runner directly — the same
    path a future bulk endpoint would use — and must return a decision (escalation here,
    because the corpus cannot support it) instead of hanging or exhausting memory.
    """
    _app, runtime, _provider = make_app()

    from answerdesk.service.turn import TurnRequest, TurnRunner

    message = "слово " * 166_667
    assert len(message) >= 1_000_000
    started = time.perf_counter()
    result = TurnRunner(runtime).run(TurnRequest(message=message))
    elapsed = time.perf_counter() - started
    assert result.decision.action in {"answer", "escalate", "clarify", "refuse"}
    assert elapsed < 30.0, f"a 1 MB query took {elapsed:.1f}s"


def test_megabyte_of_retrieved_evidence_does_not_break_the_turn(make_app) -> None:
    """A 1 MB *chunk* is a corpus defect; the service must still answer or escalate."""
    _app, runtime, _provider = make_app()
    # build the app first, then rebuild with the poisoned retriever (same settings)
    poisoned = PoisonedRetriever(runtime.retriever, text="правило " * 150_000)
    app2, _runtime2, _provider2 = make_app(retriever=poisoned)

    async def _scenario(client, _runtime):
        response = await client.post("/v1/chat", json={"message": ANSWERABLE, "stream": False})
        assert response.status_code == 200, response.text[:200]
        body = response.json()
        assert body["answer"]["decision"]["action"] in {"answer", "escalate", "clarify", "refuse"}
        trace = await client.get(f"/v1/traces/{body['trace_id']}")
        assert trace.status_code == 200
        json.dumps(trace.json())

    scenario(app2, _scenario)
