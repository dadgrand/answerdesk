"""FR-SVC-02: sessions and turns persist, survive a restart, and replay into the pipeline.

# AT-SVC-02  # AT-SVC-05  # AT-SEC-02
"""

from __future__ import annotations

import httpx
from service_support import ANSWERABLE_QUESTION, parse_sse

from answerdesk.service.app import create_app
from answerdesk.service.runtime import ServiceRuntime

CHAT = "/v1/chat"


def test_session_is_created_and_lists_its_turn(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> tuple[dict, dict]:
        created = await client.post("/v1/sessions")
        assert created.status_code == 201, created.text
        session = created.json()
        assert session["turn_count"] == 0
        response = await client.post(
            CHAT,
            json={"message": ANSWERABLE_QUESTION, "session_id": session["session_id"]},
        )
        frames = parse_sse(response.text)
        done = frames[-1].data
        loaded = await client.get(f"/v1/sessions/{session['session_id']}")
        assert loaded.status_code == 200, loaded.text
        return session, {"done": done, "session": loaded.json()}

    session, outcome = make_client(scenario)
    loaded = outcome["session"]
    assert loaded["session_id"] == session["session_id"]
    assert loaded["turn_count"] == 1
    assert loaded["turns"][0]["trace_id"] == outcome["done"]["trace_id"]
    assert loaded["turns"][0]["action"] == outcome["done"]["answer"]["decision"]["action"]
    assert loaded["turns"][0]["reason_code"]


def test_unknown_session_is_404(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> httpx.Response:
        return await client.get("/v1/sessions/s-does-not-exist")

    response = make_client(scenario)
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "not_found"
    assert error["trace_id"]
    assert error["retryable"] is False


def test_turns_survive_a_process_restart(core_settings, service_settings, tmp_path) -> None:
    """The SQLite store is the memory: a second app on the same file must see the turn."""
    first = ServiceRuntime.create(core_settings, service_settings, fake=True)
    app_one = create_app(settings=core_settings, service_settings=service_settings, runtime=first)
    try:
        from service_support import lifespan_client, run

        async def write() -> str:
            async with lifespan_client(app_one) as client:
                response = await client.post(
                    CHAT, json={"message": ANSWERABLE_QUESTION, "session_id": "s-restart"}
                )
                assert response.status_code == 200, response.text
                return response.headers["x-trace-id"]

        trace_id = run(write())
    finally:
        first.close()

    second = ServiceRuntime.create(core_settings, service_settings, fake=True)
    app_two = create_app(settings=core_settings, service_settings=service_settings, runtime=second)
    try:
        from service_support import lifespan_client, run

        async def read() -> dict:
            async with lifespan_client(app_two) as client:
                response = await client.get("/v1/sessions/s-restart")
                assert response.status_code == 200, response.text
                return response.json()

        session = run(read())
    finally:
        second.close()

    assert session["turn_count"] == 1
    assert session["turns"][0]["trace_id"] == trace_id
    assert session["turns"][0]["message"]


def test_history_is_replayed_into_the_pipeline(make_client) -> None:
    """A short follow-up resolves against the previous turn (FR-SVC-02)."""

    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        first = await client.post(
            CHAT,
            json={"message": ANSWERABLE_QUESTION, "session_id": "s-follow-up", "stream": False},
        )
        assert first.status_code == 200, first.text
        second = await client.post(
            CHAT,
            json={"message": "а для второй карты?", "session_id": "s-follow-up"},
        )
        frames = parse_sse(second.text)
        return {
            "done": frames[-1].data,
            "retrieval": next(frame.data for frame in frames if frame.event == "retrieval"),
            "frames": [frame.event for frame in frames],
            "trace": runtime.tracer.read(frames[-1].data["trace_id"], kind="turn"),
        }

    outcome = make_client(scenario)
    done = outcome["done"]
    assert done["history_turns"] == 1
    assert done["follow_up"] is True
    assert done["session_id"] == "s-follow-up"
    assert ANSWERABLE_QUESTION in outcome["retrieval"]["query"]
    assert outcome["trace"]["history"]["follow_up"] is True
    assert outcome["trace"]["history"]["turns_used"] == 1


def test_each_turn_costs_one_provider_call(make_client, fake_provider) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> int:
        for message in (ANSWERABLE_QUESTION, "а для второй карты?"):
            response = await client.post(
                CHAT, json={"message": message, "session_id": "s-calls", "stream": False}
            )
            assert response.status_code == 200, response.text
        return len(fake_provider.calls)

    assert make_client(scenario) == 2


def test_stored_messages_are_pii_redacted(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        response = await client.post(
            CHAT,
            json={
                "message": "Мой номер карты 4111 1111 1111 1111, проверьте",
                "session_id": "s-pii",
                "stream": False,
            },
        )
        assert response.status_code == 200, response.text
        loaded = await client.get("/v1/sessions/s-pii")
        return {
            "session": loaded.json(),
            "trace": runtime.tracer.read(response.json()["trace_id"], kind="turn"),
        }

    outcome = make_client(scenario)
    stored = outcome["session"]["turns"][0]["message"]
    assert "4111 1111 1111 1111" not in stored
    assert "[CARD]" in stored
    trace_text = str(outcome["trace"])
    assert "4111 1111 1111 1111" not in trace_text
    assert outcome["trace"]["query_pii"]["redacted"] >= 1
