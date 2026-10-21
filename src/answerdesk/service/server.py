"""Running the service: uvicorn wiring, the ``--healthcheck`` probe and OpenAPI export.

``answerdesk serve`` is the documented way to start the HTTP surface (§17.2 CMD-M). It has
to satisfy three constraints that pull in different directions:

* **offline** — ``--fake`` must start with no key, no network and no corpus build, then
  answer ``/healthz`` (that is exactly how CI proves the wiring);
* **honest** — a failure to bind the port or to become healthy must exit non-zero with a
  message an operator can act on, never "started" followed by a stack trace;
* **no surprise processes** — this module never daemonises. ``--healthcheck`` starts the
  server in a background thread, probes it, shuts it down and exits.

``--export-openapi`` writes the schema and exits without starting a server, which is what
keeps the API document reproducible in CI (``docs/api.md`` §7).

    # AT-SVC-07  # AT-OPS-07  # AT-OPS-01
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from answerdesk.config import Settings
from answerdesk.service.app import create_app, export_openapi
from answerdesk.service.config import ServiceSettings
from answerdesk.service.runtime import ServiceRuntime

#: How long ``--healthcheck`` waits for the service to answer ``/healthz``.
HEALTHCHECK_TIMEOUT_S = 30.0

#: Poll interval for the health probe.
HEALTHCHECK_INTERVAL_S = 0.1


def build_app(
    *,
    settings: Settings,
    service_settings: ServiceSettings,
    fake: bool,
) -> Any:
    """Build the app with a runtime factory so the pipeline is created at startup."""
    if not fake:
        return create_app(settings=settings, service_settings=service_settings)
    return create_app(
        settings=settings,
        service_settings=service_settings,
        runtime_factory=lambda: ServiceRuntime.create(settings, service_settings, fake=True),
    )


def run_server(
    *,
    settings: Settings,
    service_settings: ServiceSettings | None = None,
    fake: bool = False,
    host: str | None = None,
    port: int | None = None,
    healthcheck: bool = False,
    export_openapi_path: str | Path | None = None,
    log_level: str = "warning",
) -> int:
    """Start the service (or export the schema / run the health probe) and return an exit code."""
    service = service_settings or ServiceSettings()
    resolved_host = host or service.host
    resolved_port = port or service.port

    if export_openapi_path is not None:
        target = export_openapi(export_openapi_path, settings=settings, service_settings=service)
        print(json.dumps({"openapi": str(target), "service": "answerdesk"}, ensure_ascii=False))
        return 0

    import uvicorn

    app = build_app(settings=settings, service_settings=service, fake=fake)
    config = uvicorn.Config(
        app,
        host=resolved_host,
        port=resolved_port,
        log_level=log_level,
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    url = f"http://{resolved_host}:{resolved_port}"
    if not healthcheck:
        print(json.dumps({"url": url, "docs": f"{url}/docs", "health": f"{url}/healthz"}))
        server.run()
        return 0
    return _probe(server, url=url)


def _probe(server: Any, *, url: str) -> int:
    """Run the server in a thread, wait for ``/healthz``, then shut it down."""
    thread = threading.Thread(target=server.run, name="answerdesk-healthcheck", daemon=True)
    thread.start()
    deadline = time.monotonic() + HEALTHCHECK_TIMEOUT_S
    payload: dict[str, Any] | None = None
    error: str | None = None
    try:
        import httpx

        # trust_env=False: the probe talks to 127.0.0.1, so an ambient proxy is never
        # applicable — and on Windows `httpx` reads the system proxy, where an unusable
        # `socks4://` entry made Client() raise before the first request (exit 1 with a
        # traceback instead of a clean healthcheck).
        with httpx.Client(timeout=2.0, trust_env=False) as client:
            while time.monotonic() < deadline:
                if not thread.is_alive():
                    error = "the server thread exited before answering /healthz"
                    break
                try:
                    response = client.get(f"{url}/healthz")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    time.sleep(HEALTHCHECK_INTERVAL_S)
                    continue
                if response.status_code == 200:
                    payload = response.json()
                    break
                error = f"/healthz returned {response.status_code}"
                time.sleep(HEALTHCHECK_INTERVAL_S)
            else:
                error = f"timed out after {HEALTHCHECK_TIMEOUT_S:.0f}s waiting for /healthz"
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)

    if payload is None:
        print(
            json.dumps(
                {"url": url, "healthy": False, "error": error or "no health payload"},
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps({"url": url, "healthy": True, "health": payload}, ensure_ascii=False))
    return 0


__all__ = [
    "HEALTHCHECK_INTERVAL_S",
    "HEALTHCHECK_TIMEOUT_S",
    "build_app",
    "run_server",
]
