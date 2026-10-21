"""Service configuration: the HTTP-only knobs, separate from the frozen core settings.

``answerdesk.config.Settings`` is frozen and owns the model/retrieval/policy surface. The
service has its own small set of operational knobs (bind address, SQLite path, trace
directory, admin token, rate limits) and they live here so the frozen config does not grow
a section that only one task needs.

Everything is an environment variable with the ``ANSWERDESK_SERVICE_`` prefix, so an
operator can run the reference stack without editing files:

===============================  ==================================================
``ANSWERDESK_SERVICE_HOST``      bind address (default ``127.0.0.1``; local-first)
``ANSWERDESK_SERVICE_PORT``      bind port (default ``8765``)
``ANSWERDESK_SERVICE_DB``        SQLite file (default ``data/service/answerdesk.db``)
``ANSWERDESK_SERVICE_TRACE_DIR`` JSONL trace directory (default ``logs/traces``)
``ANSWERDESK_SERVICE_ADMIN_TOKEN`` bearer token; admin endpoints are **disabled**
                                 unless this is set
``ANSWERDESK_SERVICE_RATE_LIMIT`` requests per minute per client (default 60)
``ANSWERDESK_SERVICE_RATE_BURST`` burst allowance (default = limit)
``ANSWERDESK_SERVICE_MAX_BODY_BYTES`` request body cap (default 262144)
``ANSWERDESK_SERVICE_IDEMPOTENCY_TTL_S`` replay window (default 86400)
``ANSWERDESK_SERVICE_TRACE_MAX_BYTES`` trace rotation threshold (default 32 MiB)
``ANSWERDESK_SERVICE_CALIBRATION_ARTIFACT`` explicit calibration artifact path
``ANSWERDESK_SERVICE_CALIBRATION_MAX_AGE_DAYS`` staleness ceiling (default 365)
``ANSWERDESK_SERVICE_CACHE_THRESHOLDS`` calibrated cache thresholds (enables semantic hits)
``ANSWERDESK_SERVICE_SSE_TOKEN_CHARS`` token-event chunk size (default 24)
``ANSWERDESK_SERVICE_DOCS`` expose ``/docs`` and ``/redoc`` (default true)
``ANSWERDESK_SERVICE_CORS_ORIGINS`` comma-separated console origins
===============================  ==================================================

The admin token is a :class:`pydantic.SecretStr`; it never appears in a repr, a log line,
``/healthz`` or an OpenAPI schema. ``/healthz`` reports *whether* admin is enabled, never
the value (INV-05, §13.5).

    # AT-SVC-04  # AT-SVC-07  # AT-OPS-07
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from answerdesk.config import Settings

#: Default calibration artifact shipped with the repository (t8's synthetic study).
DEFAULT_CALIBRATION_ARTIFACT = "reports/policy-smoke/calibration-artifact.json"

#: Default cache-threshold table shipped with the repository (t8's replay study).
DEFAULT_CACHE_THRESHOLDS = "reports/cache-smoke/cache-thresholds.json"


class ServiceSettings(BaseSettings):
    """HTTP service knobs (``ANSWERDESK_SERVICE_*``)."""

    model_config = SettingsConfigDict(
        env_prefix="ANSWERDESK_SERVICE_",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    db_path: str = "data/service/answerdesk.db"
    trace_dir: str = "logs/traces"
    admin_token: SecretStr | None = None
    rate_limit_per_minute: int = Field(default=60, ge=1)
    rate_limit_burst: int | None = Field(default=None, ge=1)
    max_body_bytes: int = Field(default=262_144, ge=1024)
    idempotency_ttl_s: int = Field(default=86_400, ge=0)
    trace_max_bytes: int = Field(default=32 * 1024 * 1024, ge=0)
    calibration_artifact: str | None = DEFAULT_CALIBRATION_ARTIFACT
    calibration_max_age_days: int = Field(default=365, ge=1)
    cache_thresholds: str | None = DEFAULT_CACHE_THRESHOLDS
    sse_token_chars: int = Field(default=24, ge=1, le=512)
    docs_enabled: bool = True
    cors_origins: str = "http://localhost:5173"

    # -- derived -------------------------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def admin_enabled(self) -> bool:
        return self.admin_token is not None and bool(self.admin_token.get_secret_value())

    @property
    def burst(self) -> int:
        return self.rate_limit_burst or self.rate_limit_per_minute

    def resolve(self, core: Settings, value: str | None) -> Path | None:
        """Resolve a configured path against the repository root (like core settings)."""
        if not value:
            return None
        return core.resolve_path(value)

    def database_path(self, core: Settings) -> Path:
        return core.resolve_path(self.db_path)

    def traces_path(self, core: Settings) -> Path:
        return core.resolve_path(self.trace_dir)


def load_service_settings(**overrides: object) -> ServiceSettings:
    """Load the service settings from the environment, with explicit overrides."""
    return ServiceSettings(**overrides)  # type: ignore[arg-type]


__all__ = [
    "DEFAULT_CACHE_THRESHOLDS",
    "DEFAULT_CALIBRATION_ARTIFACT",
    "ServiceSettings",
    "load_service_settings",
]
