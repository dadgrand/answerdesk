"""FR-SVC-08 + NFR-COST-03: every request writes a redacted JSONL trace line.

Two kinds of line share the file: ``kind=turn`` (the rich answer trace) and ``kind=http``
(the routing/timing audit line the middleware writes for non-chat endpoints). The tests
above all read the file itself, not an in-memory double, so the persisted shape is what is
asserted.

    # AT-SVC-08  # AT-COST-03  # AT-OBS-01  # AT-SEC-02
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from service_support import ANSWERABLE_QUESTION, parse_sse

from answerdesk.contracts import Message, Role
from answerdesk.service.providers import RedactingProvider
from answerdesk.testing.doubles import FakeProvider

CHAT = "/v1/chat"

REQUIRED_TURN_FIELDS = (
    "trace_id",
    "session_id",
    "turn_id",
    "query",
    "language",
    "intent_bucket",
    "decision",
    "confidence",
    "p_hat",
    "retrieval",
    "cache_status",
    "usage",
    "timings_ms",
    "stage_latency_ms",
    "corpus_version",
    "index_version",
    "prompt_version",
    "retrieval_scores",
    "fusion",
    "cache",
    "history",
    "kind",
    "schema_version",
)


def _lines(directory: Path) -> list[dict]:
    path = directory / "traces.jsonl"
    assert path.is_file(), f"no trace file at {path}"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_every_request_writes_one_line_and_answer_turns_write_the_rich_one(
    service_settings, make_client, runtime
) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> list[str]:
        await client.post(CHAT, json={"message": ANSWERABLE_QUESTION, "stream": False})
        await client.post("/v1/chat", json={"message": "обслуживание карты"})
        created = await client.post("/v1/sessions")
        await client.get(f"/v1/sessions/{created.json()['session_id']}")
        await client.get("/healthz")
        await client.get("/metrics")
        return []

    make_client(scenario)
    lines = _lines(Path(service_settings.trace_dir))
    kinds = [line["kind"] for line in lines]
    assert kinds.count("turn") == 2
    http_routes = {line["route"] for line in lines if line["kind"] == "http"}
    assert "/v1/sessions" in http_routes
    assert "/v1/sessions/{session_id}" in http_routes
    # health/metrics scrapes are exempt: they are infrastructure, not product traffic.
    assert not any(line.get("route") in ("/healthz", "/metrics") for line in lines)


def test_turn_line_carries_every_documented_field(service_settings, make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> str:
        response = await client.post(
            CHAT, json={"message": ANSWERABLE_QUESTION, "session_id": "s-trace", "stream": False}
        )
        assert response.status_code == 200, response.text
        return response.json()["trace_id"]

    trace_id = make_client(scenario)
    line = next(
        row
        for row in _lines(Path(service_settings.trace_dir))
        if row["kind"] == "turn" and row["trace_id"] == trace_id
    )
    for field in REQUIRED_TURN_FIELDS:
        assert field in line, field

    assert line["query"] == ANSWERABLE_QUESTION
    assert line["decision"]["action"] == "answer"
    assert line["decision"]["reason_code"] == "ANSWERED"
    assert 0.0 <= line["decision"]["alpha"] < 1.0
    assert line["decision"]["threshold_ref"].startswith("cal-")
    assert line["p_hat"] == line["confidence"]
    assert line["timings_ms"]["retrieval"] >= 0
    assert line["timings_ms"]["generation"] > 0
    assert "verify" in line["timings_ms"] and "total" in line["timings_ms"]
    assert line["retrieval"]["packed"], "the packed order must be traceable"
    assert line["retrieval_scores"], "per-passage scores must be traceable"
    assert line["retrieval_scores"][0]["chunk_id"]
    assert line["fusion"] and line["fusion"]["fuser"] in ("rrf", "cc", "none")
    assert line["usage"]["model"]
    assert line["usage"]["input_tokens"] >= 0
    assert line["corpus_version"].startswith("corpus-demo-")
    assert line["index_version"].startswith("index-")
    assert line["prompt_version"]
    assert line["service"]["scorer"]["name"] == "overlap-baseline"
    assert line["cache"]["status"] in ("hit", "miss", "bypassed")


def test_escalated_turn_records_the_reason_and_the_ticket(service_settings, make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        response = await client.post(
            CHAT,
            json={"message": "Мой номер карты 4111 1111 1111 1111, проверьте", "stream": False},
        )
        payload = response.json()
        frames = parse_sse((await client.post(CHAT, json={"message": ANSWERABLE_QUESTION})).text)
        return {"payload": payload, "answer_frame": frames[-1].data}

    outcome = make_client(scenario)
    text = str(outcome["payload"])
    payload = outcome["payload"]
    line = next(
        row
        for row in _lines(Path(service_settings.trace_dir))
        if row.get("trace_id") == payload["trace_id"] and row["kind"] == "turn"
    )
    assert line["decision"]["action"] == "escalate"
    assert line["decision"]["reason_code"] == "PII_DETECTED"
    assert line["escalation"]["reason"] == "PII_DETECTED"
    assert line["escalation"]["ticket_id"].startswith("tk-")
    assert "extraction" not in text


def test_traces_are_pii_redacted(service_settings, make_client, runtime) -> None:
    pii = "4111 1111 1111 1111"
    email = "customer@example.com"

    async def scenario(client: httpx.AsyncClient, runtime) -> None:
        await client.post(
            CHAT,
            json={
                "message": f"проверьте карту {pii} и напишите на {email}",
                "stream": False,
            },
        )

    make_client(scenario)
    path = Path(service_settings.trace_dir) / "traces.jsonl"
    raw = path.read_text(encoding="utf-8")
    assert pii not in raw
    assert email not in raw
    turn = next(row for row in _lines(Path(service_settings.trace_dir)) if row["kind"] == "turn")
    assert turn["query_pii"]["redacted"] >= 2
    assert "card" in turn["query_pii"]["kinds"] or "email" in turn["query_pii"]["kinds"]
    assert turn["pii"]["redacted"] == 0  # the stored line was already clean


def test_redacting_provider_strips_pii_before_the_inner_provider() -> None:
    """INV-05 at the provider seam: the double never sees the customer's card number."""
    inner = FakeProvider()
    provider = RedactingProvider(inner)
    probe = Message(role=Role.USER, content="card 4111 1111 1111 1111 email a@b.co")
    provider.complete([probe], response_schema={"type": "object"})
    delivered = inner.calls[-1][0].content
    assert "4111 1111 1111 1111" not in delivered
    assert "a@b.co" not in delivered
    assert provider.redactions >= 1


def test_trace_endpoint_returns_the_persisted_line(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        chat = await client.post(CHAT, json={"message": ANSWERABLE_QUESTION, "stream": False})
        trace_id = chat.json()["trace_id"]
        response = await client.get(f"/v1/traces/{trace_id}")
        missing = await client.get("/v1/traces/tr-nope")
        return {"response": response, "missing": missing}

    outcome = make_client(scenario)
    assert outcome["response"].status_code == 200
    body = outcome["response"].json()
    assert body["trace_id"] == body["trace"]["trace_id"]
    assert body["trace"]["kind"] == "turn"
    assert outcome["missing"].status_code == 404
