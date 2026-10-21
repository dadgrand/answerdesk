"""FR-SVC-07 + NFR-OBS-01: dependency health without secrets, and Prometheus metrics.

# AT-SVC-07  # AT-OBS-01  # AT-COST-03  # AT-CAC-08
"""

from __future__ import annotations

from pathlib import Path

import httpx
from service_support import ANSWERABLE_QUESTION, lifespan_client, run

from answerdesk.config import load_settings
from answerdesk.service.app import create_app
from answerdesk.service.config import ServiceSettings
from answerdesk.service.runtime import ServiceRuntime
from answerdesk.telemetry.metrics import parse_exposition

PROMETHEUS_FAMILIES = (
    "answerdesk_http_requests_total",
    "answerdesk_http_request_duration_seconds",
    "answerdesk_http_errors_total",
    "answerdesk_decisions_total",
    "answerdesk_escalations_total",
    "answerdesk_refusals_total",
    "answerdesk_cache_hits_total",
    "answerdesk_cache_misses_total",
    "answerdesk_cache_bypass_total",
    "answerdesk_cache_wrong_hits_total",
    "answerdesk_model_tokens_total",
    "answerdesk_model_cost_usd_total",
    "answerdesk_traces_written_total",
    "answerdesk_stage_duration_seconds",
)


def test_health_reports_every_dependency_without_secrets(make_client, service_settings) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        response = await client.get("/healthz")
        assert response.status_code == 200
        return response.json()

    payload = make_client(scenario)
    assert payload["status"] == "ok"
    assert payload["service"] == "answerdesk"
    assert payload["version"]
    assert payload["schema_version"] == "1.0.0"
    assert payload["uptime_s"] >= 0
    for component in (
        "service",
        "database",
        "index",
        "corpus",
        "provider",
        "calibration",
        "cache",
        "admin",
        "tracing",
    ):
        assert component in payload["checks"], component
    assert payload["checks"]["index"]["status"] == "ok"
    assert payload["checks"]["calibration"]["status"] == "ok"
    assert payload["checks"]["admin"]["status"] == "ok"

    text = str(payload)
    assert service_settings.admin_token.get_secret_value() not in text
    assert "api_key=absent" in payload["checks"]["provider"]["detail"]
    assert "test-admin-token" not in text


def test_health_strict_mode_returns_503_when_degraded(tmp_path: Path, core_settings) -> None:
    settings = load_settings(
        profile="offline",
        overrides={
            "app": {
                "corpus_dir": str(tmp_path / "no-corpus"),
                "index_dir": str(tmp_path / "no-index"),
            }
        },
    )
    service = ServiceSettings(
        db_path=str(tmp_path / "db.sqlite"),
        trace_dir=str(tmp_path / "traces"),
        calibration_artifact=str(tmp_path / "missing-artifact.json"),
    )
    runtime = ServiceRuntime.create(settings, service, fake=False)
    app = create_app(settings=settings, service_settings=service, runtime=runtime)

    async def scenario() -> dict:
        async with lifespan_client(app) as client:
            relaxed = await client.get("/healthz")
            strict = await client.get("/healthz", params={"strict": "true"})
            chat = await client.post("/v1/chat", json={"message": "hi", "stream": False})
            return {"relaxed": relaxed, "strict": strict, "chat": chat}

    try:
        outcome = run(scenario())
    finally:
        runtime.close()

    assert outcome["relaxed"].status_code == 200
    payload = outcome["relaxed"].json()
    assert payload["status"] == "degraded"
    assert payload["checks"]["index"]["status"] == "error"
    assert payload["checks"]["calibration"]["status"] == "error"
    assert outcome["strict"].status_code == 503
    assert outcome["chat"].status_code == 503
    assert outcome["chat"].json()["error"]["code"] == "index_missing"


def test_calibration_missing_is_a_503_even_with_an_index(
    core_settings, service_settings, tmp_path: Path
) -> None:
    service = ServiceSettings(
        db_path=service_settings.db_path,
        trace_dir=service_settings.trace_dir,
        admin_token=service_settings.admin_token,
        calibration_artifact=str(tmp_path / "absent.json"),
    )
    runtime = ServiceRuntime.create(core_settings, service, fake=True)
    app = create_app(settings=core_settings, service_settings=service, runtime=runtime)

    async def scenario() -> tuple[httpx.Response, dict]:
        async with lifespan_client(app) as client:
            chat = await client.post("/v1/chat", json={"message": "hi", "stream": False})
            health = await client.get("/healthz")
            return chat, health.json()

    try:
        chat, health = run(scenario())
    finally:
        runtime.close()
    assert chat.status_code == 503
    assert chat.json()["error"]["code"] == "calibration_missing"
    assert health["status"] == "degraded"
    assert health["checks"]["index"]["status"] == "ok"


def test_metrics_expose_the_documented_families(make_client, runtime) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> str:
        await client.post("/v1/chat", json={"message": ANSWERABLE_QUESTION, "stream": False})
        await client.post("/v1/chat", json={"message": "мой номер 4111 1111 1111 1111"})
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        return response.text

    text = make_client(scenario)
    samples = parse_exposition(text)
    for family in PROMETHEUS_FAMILIES:
        # Every family is always declared (HELP/TYPE) even when it has no sample yet.
        assert f"# TYPE {family} " in text, family

    requests = samples["answerdesk_http_requests_total"]
    chat_requests = [value for labels, value in requests if labels.get("route") == "/v1/chat"]
    assert chat_requests and sum(chat_requests) >= 2

    decisions = samples["answerdesk_decisions_total"]
    actions = {labels.get("action") for labels, _value in decisions}
    assert {"answer", "escalate"} & actions

    latency = samples["answerdesk_http_request_duration_seconds_bucket"]
    assert latency
    assert any(labels.get("le") == "+Inf" for labels, _value in latency)

    # The escalation path must be counted with its reason code.
    escalations = samples["answerdesk_escalations_total"]
    assert any(labels.get("reason") for labels, _value in escalations)

    assert samples["answerdesk_traces_written_total"]
    assert samples["answerdesk_model_tokens_total"]
    assert runtime.metrics.snapshot()["cache_misses"] >= 1


def test_cache_kill_switch_is_visible_in_health_and_metrics(
    core_settings, service_settings
) -> None:
    settings = core_settings.model_copy(deep=True)
    settings.cache.enabled = False
    runtime = ServiceRuntime.create(settings, service_settings, fake=True)
    app = create_app(settings=settings, service_settings=service_settings, runtime=runtime)

    async def scenario() -> dict:
        async with lifespan_client(app) as client:
            first = await client.post(
                "/v1/chat", json={"message": ANSWERABLE_QUESTION, "stream": False}
            )
            second = await client.post(
                "/v1/chat", json={"message": ANSWERABLE_QUESTION, "stream": False}
            )
            health = (await client.get("/healthz")).json()
            return {
                "first": first.json(),
                "second": second.json(),
                "health": health,
                "metrics": (await client.get("/metrics")).text,
            }

    try:
        outcome = run(scenario())
        snapshot = runtime.metrics.snapshot()
    finally:
        runtime.close()

    assert outcome["first"]["cache"]["status"] == "bypassed"
    assert outcome["second"]["cache"]["status"] == "bypassed"
    assert outcome["health"]["checks"]["cache"]["status"] == "disabled"
    assert snapshot["cache_hits"] == {}


def test_metrics_endpoint_does_not_require_the_answer_path(tmp_path: Path, core_settings) -> None:
    settings = load_settings(
        profile="offline",
        overrides={"app": {"index_dir": str(tmp_path / "absent-index")}},
    )
    service = ServiceSettings(
        db_path=str(tmp_path / "db.sqlite"),
        trace_dir=str(tmp_path / "traces"),
        calibration_artifact=str(tmp_path / "absent.json"),
    )
    runtime = ServiceRuntime.create(settings, service, fake=False)
    app = create_app(settings=settings, service_settings=service, runtime=runtime)

    async def scenario() -> tuple[int, int]:
        async with lifespan_client(app) as client:
            metrics = await client.get("/metrics")
            health = await client.get("/healthz")
            return metrics.status_code, health.status_code

    try:
        metrics_status, health_status = run(scenario())
    finally:
        runtime.close()
    assert metrics_status == 200
    assert health_status == 200
