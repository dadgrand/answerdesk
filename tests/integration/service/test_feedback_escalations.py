"""FR-SVC-05: feedback and escalation tickets persist and link to their trace.

    # AT-SVC-05  # AT-CAC-08
"""

from __future__ import annotations

import httpx

from support import ANSWERABLE_QUESTION, parse_sse

CHAT = "/v1/chat"


def test_feedback_round_trip_and_trace_linkage(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        answer = await client.post(
            CHAT, json={"message": ANSWERABLE_QUESTION, "stream": False}
        )
        trace_id = answer.json()["trace_id"]
        created = await client.post(
            "/v1/feedback",
            json={"trace_id": trace_id, "rating": "up", "comment": "Точно ответили"},
        )
        rows = runtime.store.feedback_for(trace_id)
        return {"created": created, "rows": rows, "trace_id": trace_id}

    outcome = make_client(scenario)
    created = outcome["created"]
    assert created.status_code == 201, created.text
    payload = created.json()
    assert payload["trace_id"] == outcome["trace_id"]
    assert payload["rating"] == "up"
    assert payload["feedback_id"]
    assert outcome["rows"], "the feedback must be persisted against the trace"
    assert outcome["rows"][0].comment == "Точно ответили"


def test_feedback_for_an_unknown_trace_is_404(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> httpx.Response:
        return await client.post("/v1/feedback", json={"trace_id": "tr-nope", "rating": "down"})

    response = make_client(scenario)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_escalation_ticket_round_trip(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        answer = await client.post(
            CHAT,
            json={"message": "Мой номер карты 4111 1111 1111 1111, проверьте", "stream": False},
        )
        trace_id = answer.json()["trace_id"]
        created = await client.post(
            "/v1/escalations",
            json={
                "trace_id": trace_id,
                "reason": "Клиент просит человека",
                "priority": "high",
                "note": "Проверить вручную",
            },
        )
        ticket = created.json()
        fetched = await client.get(f"/v1/escalations/{ticket['ticket_id']}")
        listed = await client.get("/v1/escalations", params={"limit": 5})
        return {
            "created": created,
            "fetched": fetched,
            "listed": listed.json(),
            "trace": runtime.tracer.read(trace_id),
        }

    outcome = make_client(scenario)
    assert outcome["created"].status_code == 201, outcome["created"].text
    ticket = outcome["created"].json()
    assert ticket["status"] == "open"
    assert ticket["priority"] == "high"
    assert ticket["trace_id"] == outcome["trace"]["trace_id"]
    assert outcome["fetched"].json()["ticket_id"] == ticket["ticket_id"]
    assert any(row["ticket_id"] == ticket["ticket_id"] for row in outcome["listed"]["tickets"])


def test_escalation_for_an_unknown_trace_is_404(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> httpx.Response:
        return await client.post(
            "/v1/escalations", json={"trace_id": "tr-missing", "reason": "no such trace"}
        )

    response = make_client(scenario)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_auto_escalation_creates_a_ticket(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        frames = parse_sse(
            (
                await client.post(
                    CHAT, json={"message": "Какие часы работы у офиса в Антарктиде?"}
                )
            ).text
        )
        done = frames[-1].data
        return {"done": done, "trace": runtime.tracer.read(done["trace_id"])}

    outcome = make_client(scenario)
    done = outcome["done"]
    assert done["ticket_id"], "an escalated turn must open an operator ticket"
    assert outcome["trace"]["escalation"]["ticket_id"] == done["ticket_id"]
    assert outcome["trace"]["escalation"]["reason"] == done["answer"]["decision"]["reason_code"]


def test_down_vote_on_a_cache_hit_counts_as_a_wrong_hit(make_client, runtime) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        first = await client.post(
            CHAT, json={"message": ANSWERABLE_QUESTION, "stream": False}
        )
        second = await client.post(
            CHAT, json={"message": ANSWERABLE_QUESTION, "stream": False}
        )
        payload = second.json()
        response = await client.post(
            "/v1/feedback", json={"trace_id": payload["trace_id"], "rating": "down"}
        )
        return {"status": response.status_code, "cache": payload["cache"]["status"]}

    outcome = make_client(scenario)
    assert outcome["status"] == 201
    assert outcome["cache"] == "hit"
    assert runtime.metrics.cache_wrong_hits.value() >= 1.0
