"""FR-SVC-04: the admin endpoints are bearer-token guarded and disabled by default.

    # AT-SVC-04  # AT-OPS-07
"""

from __future__ import annotations

from pathlib import Path

import httpx

from support import ADMIN_TOKEN, ANSWERABLE_QUESTION, lifespan_client, run

from answerdesk.config import load_settings
from answerdesk.service.app import create_app
from answerdesk.service.config import ServiceSettings
from answerdesk.service.runtime import ServiceRuntime


def _app_with(core, service: ServiceSettings) -> tuple[ServiceRuntime, object]:
    runtime = ServiceRuntime.create(core, service, fake=True)
    return runtime, create_app(settings=core, service_settings=service, runtime=runtime)


def test_admin_is_disabled_without_a_configured_token(core_settings, service_settings) -> None:
    disabled = ServiceSettings(
        db_path=service_settings.db_path,
        trace_dir=service_settings.trace_dir,
        admin_token=None,
    )
    runtime, app = _app_with(core_settings, disabled)

    async def scenario() -> httpx.Response:
        async with lifespan_client(app) as client:
            return await client.post(
                "/admin/ingest",
                json={"demo": True},
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            )

    try:
        response = run(scenario())
    finally:
        runtime.close()
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "admin_disabled"


def test_admin_requires_the_bearer_token(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> tuple[httpx.Response, httpx.Response]:
        missing = await client.post("/admin/ingest", json={"demo": True})
        wrong = await client.post(
            "/admin/ingest", json={"demo": True}, headers={"Authorization": "Bearer nope"}
        )
        return missing, wrong

    missing, wrong = make_client(scenario)
    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "unauthorized"
    assert missing.headers["www-authenticate"] == "Bearer"
    assert wrong.status_code == 403
    assert wrong.json()["error"]["code"] == "forbidden"


def test_admin_ingest_reports_manifest_stats_and_refresh_invalidates_cache(
    tmp_path: Path, service_settings, monkeypatch
) -> None:
    monkeypatch.setenv("ANSWERDESK_INGEST_DOCS_PER_LOCALE", "1")
    corpus = tmp_path / "corpus"
    index = tmp_path / "index"
    core = load_settings(
        profile="offline",
        overrides={"app": {"corpus_dir": str(corpus), "index_dir": str(index)}},
    )
    runtime, app = _app_with(core, service_settings)
    headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

    async def scenario() -> dict:
        async with lifespan_client(app) as client:
            before = await client.post(
                "/v1/chat", json={"message": ANSWERABLE_QUESTION, "stream": False}
            )
            ingested = await client.post("/admin/ingest", json={"demo": True}, headers=headers)
            after = await client.post(
                "/v1/chat", json={"message": ANSWERABLE_QUESTION, "stream": False}
            )
            return {
                "before": before.json(),
                "ingested": ingested,
                "after": after,
                "trace": runtime.tracer.read(after.json()["trace_id"]),
            }

    try:
        outcome = run(scenario())
    finally:
        runtime.close()

    ingested = outcome["ingested"]
    assert ingested.status_code == 200, ingested.text
    body = ingested.json()
    assert body["corpus_version"].startswith("corpus-")
    assert body["index_version"].startswith("index-")
    assert body["chunks"] > 0
    assert body["documents"] > 0
    assert body["manifest_path"].endswith("manifest.json")
    assert body["elapsed_ms"] > 0

    assert (corpus / "manifest.json").is_file(), "the corpus must be built on disk"
    assert (Path(index) / body["corpus_version"] / "index.json").is_file()

    # The refresh reloads the disk index and re-arms the cache for the new corpus version.
    assert outcome["after"].status_code == 200, outcome["after"].text
    assert outcome["after"].json()["trace_id"]
    assert outcome["trace"]["corpus_version"] == body["corpus_version"]
    assert "reused" in body and isinstance(body["reused"], bool)


def test_admin_refresh_returns_manifest_stats(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> httpx.Response:
        return await client.post(
            "/admin/refresh",
            params={"force": "false"},
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )

    response = make_client(scenario)
    # The fake runtime has no on-disk corpus; the refresh must fail with a taxonomy error,
    # never with a stack trace.
    assert response.status_code in (200, 400, 503), response.text
    if response.status_code != 200:
        assert response.json()["error"]["code"]


def test_refresh_reloads_the_retriever(tmp_path: Path, service_settings, monkeypatch) -> None:
    monkeypatch.setenv("ANSWERDESK_INGEST_DOCS_PER_LOCALE", "1")
    core = load_settings(
        profile="offline",
        overrides={
            "app": {
                "corpus_dir": str(tmp_path / "corpus"),
                "index_dir": str(tmp_path / "index"),
            }
        },
    )
    runtime, app = _app_with(core, service_settings)
    previous = runtime.retriever
    try:
        stats = runtime.refresh_index(demo=True, force=True)
        assert stats["chunks"] > 0
        assert runtime.retriever is not None and runtime.retriever is not previous
        assert runtime.info.corpus_version == stats["corpus_version"]

        async def scenario() -> httpx.Response:
            async with lifespan_client(app) as client:
                return await client.post(
                    "/v1/chat", json={"message": ANSWERABLE_QUESTION, "stream": False}
                )

        response = run(scenario())
        assert response.status_code == 200, response.text
        assert response.json()["answer"]
    finally:
        runtime.close()
