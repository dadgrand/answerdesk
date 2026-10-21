"""FR-SVC-01: the chat endpoint streams the documented SSE sequence, offline.

Every event must carry ``trace_id``; citations may only appear after verification; the
concatenated token deltas must equal the verified answer text. The JSON path
(``stream=false``) must carry the same identifiers so a console can treat both the same.

    # AT-SVC-01  # AT-UI-08  # AT-GEN-09
"""

from __future__ import annotations

import json

import httpx

from answerdesk.contracts import Action
from support import (
    ANSWERABLE_QUESTION,
    UNANSWERABLE_QUESTION,
    events_text,
    parse_sse,
)

CHAT = "/v1/chat"
ORDER = ["retrieval", "citations", "decision", "done"]


def test_stream_emits_the_contract_event_order_and_trace_ids(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        response = await client.post(
            CHAT, json={"message": ANSWERABLE_QUESTION}, headers={"Accept": "text/event-stream"}
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-trace-id"]
        frames = parse_sse(response.text)
        return {
            "frames": frames,
            "trace_id": response.headers["x-trace-id"],
            "body": response.text,
        }

    outcome = make_client(scenario)
    frames = outcome["frames"]
    assert frames, outcome["body"]
    assert frames[-1].event == "done"
    assert frames[0].event == "retrieval"
    names = [frame.event for frame in frames]
    # retrieval → tokens* → citations → decision → done
    assert names[0] == "retrieval"
    assert names[-1] == "done"
    tail = [name for name in names if name != "tokens"]
    assert tail == ORDER or tail == ["retrieval", "citations", "decision", "done"], names
    for frame in frames:
        assert frame.data["trace_id"] == outcome["trace_id"], frame.event


def test_answer_stream_carries_verified_citations_and_tokens(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        response = await client.post(CHAT, json={"message": ANSWERABLE_QUESTION})
        frames = parse_sse(response.text)
        return {"frames": frames, "body": response.text}

    outcome = make_client(scenario)
    frames = outcome["frames"]
    done = next(frame for frame in frames if frame.event == "done")
    answer = done.data["answer"]
    assert answer["decision"]["action"] == Action.ANSWER.value, outcome["body"]

    citations = next(frame for frame in frames if frame.event == "citations").data
    assert citations["citations"], "a verified answer must stream at least one citation"
    assert citations["claims"]
    for claim in citations["claims"]:
        assert claim["citation_ids"], "a displayed claim must stay cited (INV-03)"
    cited_ids = {citation["citation_id"] for citation in citations["citations"]}
    assert all(
        citation_id in cited_ids
        for claim in citations["claims"]
        for citation_id in claim["citation_ids"]
    )

    retrieval = next(frame for frame in frames if frame.event == "retrieval").data
    passage_ids = {passage["chunk_id"] for passage in retrieval["passages"]}
    for citation in citations["citations"]:
        assert citation["chunk_id"] in passage_ids, "citations must resolve to packed passages"

    assert events_text(frames) == answer["text"], "token deltas must reproduce the answer text"
    assert answer["citations"]


def test_escalation_stream_has_no_tokens_and_a_reason(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> list:
        response = await client.post(
            CHAT, json={"message": "Мой номер карты 4111 1111 1111 1111, проверьте баланс"}
        )
        assert response.status_code == 200, response.text
        return parse_sse(response.text)

    frames = make_client(scenario)
    names = [frame.event for frame in frames]
    assert "tokens" not in names, "an escalated turn must not stream prose (CR-8)"
    decision = next(frame for frame in frames if frame.event == "decision").data
    assert decision["action"] in ("escalate", "refuse")
    assert decision["reason_code"] == "PII_DETECTED"
    assert decision["alpha"] > 0
    assert decision["threshold_ref"].startswith("cal-")
    assert decision["escalation"]["reason"] == "PII_DETECTED"
    done = frames[-1].data
    assert done["answer"]["text"] == ""
    assert done["ticket_id"]


def test_unanswerable_question_escalates_with_a_reason_code(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> list:
        response = await client.post(CHAT, json={"message": UNANSWERABLE_QUESTION})
        return parse_sse(response.text)

    frames = make_client(scenario)
    decision = next(frame for frame in frames if frame.event == "decision").data
    assert decision["action"] in ("escalate", "refuse", "clarify")
    assert decision["reason_code"] not in ("ANSWERED",)


def test_json_mode_returns_the_same_identifiers(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> tuple[dict, dict]:
        body = {"message": ANSWERABLE_QUESTION, "session_id": "s-json-mode-test"}
        streamed = parse_sse((await client.post(CHAT, json=body)).text)
        plain = await client.post(CHAT, json={**body, "stream": False})
        assert plain.status_code == 200, plain.text
        assert plain.headers["content-type"].startswith("application/json")
        return plain.json(), streamed[-1].data

    plain, done = make_client(scenario)
    assert plain["trace_id"]
    assert plain["session_id"] == done["session_id"]
    assert plain["answer"]["decision"]["action"] == done["answer"]["decision"]["action"]
    assert plain["cache"]["status"] in ("hit", "miss", "bypassed")


def test_accept_json_forces_the_single_response(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> httpx.Response:
        return await client.post(
            CHAT,
            json={"message": ANSWERABLE_QUESTION, "stream": True},
            headers={"Accept": "application/json"},
        )

    response = make_client(scenario)
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["answer"]


def test_client_supplied_trace_id_is_echoed_and_used(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> tuple[httpx.Response, dict]:
        response = await client.post(
            CHAT,
            json={"message": ANSWERABLE_QUESTION, "stream": False},
            headers={"X-Trace-Id": "tr-client-supplied-0001"},
        )
        return response, response.json()

    response, payload = make_client(scenario)
    assert response.headers["x-trace-id"] == "tr-client-supplied-0001"
    assert payload["trace_id"] == "tr-client-supplied-0001"


def test_citations_event_is_json_serialisable_per_locale(make_client) -> None:
    """The framing must survive non-ASCII text without breaking the frame (Cyrillic/CJK)."""

    async def scenario(client: httpx.AsyncClient, runtime) -> list:
        response = await client.post(
            CHAT, json={"message": "обслуживание карты", "locale": "ru"}
        )
        return parse_sse(response.text)

    frames = make_client(scenario)
    for frame in frames:
        json.dumps(frame.data, ensure_ascii=False)
    retrieval = next(frame for frame in frames if frame.event == "retrieval").data
    assert retrieval["language"].startswith("ru")
