"""NFR-OPS-01: the failure-mode matrix — every row has a defined behaviour and a test.

=============================  =========================================  ==============
row                            behaviour                                  HTTP
=============================  =========================================  ==============
provider timeout               ``provider_timeout`` envelope / SSE error   504
provider 5xx                   ``provider_error`` envelope / SSE error     502
malformed model JSON           generator escalates ``PROVIDER_ERROR``      200
empty retrieval                decision ``NO_EVIDENCE``, no prose          200
index missing                  ``index_missing``                            503
calibration missing            ``calibration_missing``                      503
corpus version mismatch        ``STALE_CALIBRATION`` escalation, degraded   200 / health
=============================  =========================================  ==============

The table is repeated in ``docs/ops.md`` §4 with the operator action for each row.

    # AT-OPS-07
"""

from __future__ import annotations

from pathlib import Path

import httpx
from service_support import (
    ANSWERABLE_QUESTION,
    GarbageProvider,
    ServerErrorProvider,
    TimeoutProvider,
    lifespan_client,
    parse_sse,
    run,
    runtime_with_provider,
)

from answerdesk.config import load_settings
from answerdesk.contracts import RetrievalTrace
from answerdesk.policy import CalibrationArtifact, DecisionEngine
from answerdesk.service.app import create_app
from answerdesk.service.config import ServiceSettings
from answerdesk.service.runtime import ServiceRuntime

CHAT = "/v1/chat"


def _client_for(runtime: ServiceRuntime, core, service) -> tuple[object, ServiceRuntime]:
    return create_app(settings=core, service_settings=service, runtime=runtime), runtime


def test_provider_timeout_is_a_504(service_settings) -> None:
    core = load_settings(profile="offline")
    runtime = runtime_with_provider(TimeoutProvider(), core, service_settings)
    app = create_app(settings=core, service_settings=service_settings, runtime=runtime)

    async def scenario() -> tuple[httpx.Response, dict]:
        async with lifespan_client(app) as client:
            response = await client.post(
                CHAT, json={"message": "обслуживание карты", "stream": False}
            )
            return response, runtime.tracer.read(response.headers["x-trace-id"], kind="turn")

    try:
        response, trace = run(scenario())
    finally:
        runtime.close()
    assert response.status_code == 504
    error = response.json()["error"]
    assert error["code"] == "provider_timeout"
    assert error["retryable"] is True
    assert trace["error"] and "ProviderTimeoutError" in trace["error"]


def test_provider_5xx_is_a_502(service_settings) -> None:
    core = load_settings(profile="offline")
    runtime = runtime_with_provider(ServerErrorProvider(), core, service_settings)
    app = create_app(settings=core, service_settings=service_settings, runtime=runtime)

    async def scenario() -> httpx.Response:
        async with lifespan_client(app) as client:
            return await client.post(CHAT, json={"message": "обслуживание карты", "stream": False})

    try:
        response = run(scenario())
    finally:
        runtime.close()
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "provider_error"


def test_provider_failure_mid_stream_emits_error_then_done(service_settings) -> None:
    core = load_settings(profile="offline")
    runtime = runtime_with_provider(TimeoutProvider(), core, service_settings)
    app = create_app(settings=core, service_settings=service_settings, runtime=runtime)

    async def scenario() -> httpx.Response:
        async with lifespan_client(app) as client:
            return await client.post(CHAT, json={"message": "обслуживание карты"})

    try:
        response = run(scenario())
    finally:
        runtime.close()
    # The status line is already sent when the failure happens; the stream is well-formed.
    assert response.status_code == 200
    frames = parse_sse(response.text)
    names = [frame.event for frame in frames]
    assert names == ["error", "done"], names
    assert frames[0].data["error"]["code"] == "provider_timeout"
    assert frames[0].data["trace_id"] == frames[1].data["trace_id"]


def test_malformed_model_json_escalates_instead_of_answering(service_settings) -> None:
    core = load_settings(profile="offline")
    runtime = runtime_with_provider(GarbageProvider(), core, service_settings)
    app = create_app(settings=core, service_settings=service_settings, runtime=runtime)

    async def scenario() -> tuple[httpx.Response, dict]:
        async with lifespan_client(app) as client:
            response = await client.post(
                CHAT, json={"message": ANSWERABLE_QUESTION, "stream": False}
            )
            return response, runtime.tracer.read(response.json()["trace_id"], kind="turn")

    try:
        response, trace = run(scenario())
    finally:
        runtime.close()
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]["decision"]["action"] == "escalate"
    assert payload["answer"]["decision"]["reason_code"] == "PROVIDER_ERROR"
    assert payload["answer"]["text"] == ""
    assert trace["decision"]["reason_code"] == "PROVIDER_ERROR"


class EmptyRetriever:
    """A retriever that finds nothing — the empty-retrieval failure row."""

    name = "empty"

    def search(self, query: str, k: int) -> list:
        return []

    def retrieve(self, query: str, *, final_k: int | None = None) -> RetrievalTrace:
        return RetrievalTrace(query=query)

    def retrieve_with_timings(
        self, query: str, *, final_k: int | None = None
    ) -> tuple[RetrievalTrace, dict[str, float]]:
        return RetrievalTrace(query=query), {"total_ms": 0.0}


def test_empty_retrieval_escalates_with_no_evidence(service_settings) -> None:
    core = load_settings(profile="offline")
    runtime = ServiceRuntime.create(
        core,
        service_settings,
        fake=True,
        retriever=EmptyRetriever(),  # type: ignore[arg-type]
    )
    app = create_app(settings=core, service_settings=service_settings, runtime=runtime)

    async def scenario() -> dict:
        async with lifespan_client(app) as client:
            response = await client.post(
                CHAT, json={"message": ANSWERABLE_QUESTION, "stream": False}
            )
            return response.json()

    try:
        payload = run(scenario())
    finally:
        runtime.close()
    assert payload["answer"]["decision"]["action"] == "escalate"
    assert payload["answer"]["decision"]["reason_code"] in ("NO_EVIDENCE", "LOW_CONFIDENCE")
    assert payload["answer"]["text"] == ""


def test_index_missing_is_a_503(tmp_path: Path) -> None:
    settings = load_settings(
        profile="offline",
        overrides={
            "app": {"index_dir": str(tmp_path / "absent"), "corpus_dir": str(tmp_path / "corpus")}
        },
    )
    service = ServiceSettings(
        db_path=str(tmp_path / "db.sqlite"),
        trace_dir=str(tmp_path / "traces"),
        calibration_artifact=str(tmp_path / "absent.json"),
    )
    runtime = ServiceRuntime.create(settings, service, fake=False)
    app = create_app(settings=settings, service_settings=service, runtime=runtime)

    async def scenario() -> httpx.Response:
        async with lifespan_client(app) as client:
            return await client.post(CHAT, json={"message": "hi", "stream": False})

    try:
        response = run(scenario())
    finally:
        runtime.close()
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "index_missing"


def test_calibration_missing_is_a_503(core_settings, service_settings, tmp_path: Path) -> None:
    service = ServiceSettings(
        db_path=service_settings.db_path,
        trace_dir=service_settings.trace_dir,
        admin_token=service_settings.admin_token,
        calibration_artifact=str(tmp_path / "absent.json"),
    )
    runtime = ServiceRuntime.create(core_settings, service, fake=True)
    app = create_app(settings=core_settings, service_settings=service, runtime=runtime)

    async def scenario() -> httpx.Response:
        async with lifespan_client(app) as client:
            return await client.post(CHAT, json={"message": "hi", "stream": False})

    try:
        response = run(scenario())
    finally:
        runtime.close()
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "calibration_missing"


def test_corpus_version_mismatch_escalates_and_degrades_health(
    core_settings, service_settings
) -> None:
    artifact_path = core_settings.resolve_path("reports/policy-smoke/calibration-artifact.json")
    artifact = CalibrationArtifact.load(artifact_path)
    engine = DecisionEngine.from_artifact(artifact, corpus_version="corpus-from-another-run")
    assert engine.stale is True

    runtime = ServiceRuntime.create(core_settings, service_settings, fake=True, engine=engine)
    app = create_app(settings=core_settings, service_settings=service_settings, runtime=runtime)

    async def scenario() -> dict:
        async with lifespan_client(app) as client:
            health = (await client.get("/healthz")).json()
            response = await client.post(
                CHAT, json={"message": ANSWERABLE_QUESTION, "stream": False}
            )
            return {"health": health, "payload": response.json()}

    try:
        outcome = run(scenario())
    finally:
        runtime.close()

    assert outcome["health"]["status"] == "degraded"
    assert outcome["health"]["checks"]["calibration"]["status"] == "degraded"
    assert "stale" in outcome["health"]["checks"]["calibration"]["detail"]
    decision = outcome["payload"]["answer"]["decision"]
    assert decision["action"] == "escalate"
    assert decision["reason_code"] == "STALE_CALIBRATION"
    assert outcome["payload"]["answer"]["text"] == ""


def test_failure_rows_are_documented_in_ops() -> None:
    """The matrix in docs/ops.md must name every row this module tests."""
    ops = Path(__file__).resolve().parents[3] / "docs" / "ops.md"
    assert ops.is_file(), "docs/ops.md is missing"
    text = ops.read_text(encoding="utf-8").lower()
    for row in (
        "provider timeout",
        "provider 5xx",
        "malformed model json",
        "empty retrieval",
        "index missing",
        "calibration missing",
        "corpus version mismatch",
    ):
        assert row in text, row
