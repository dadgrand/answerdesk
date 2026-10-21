"""The runtime: one object that wires every frozen seam for one process.

``ServiceRuntime.create(...)`` is the only place where the service composes the stack, and
it is deliberately boring and explicit:

    settings ──▶ embedder ──▶ index ──▶ RetrievalPipeline ─┐
                           (or an in-memory demo index)   │
    profile ──▶ provider ──▶ RedactingProvider ──▶ AnswerPipeline
    policy artifact ──▶ DecisionEngine (guards, conformal, cost bands)
    reports/cache thresholds ──▶ SemanticCache
    ServiceSettings ──▶ SQLiteSessionStore, JsonlTraceWriter, Metrics

Failure handling is part of the contract (NFR-OPS-01): a missing index or calibration
artifact does **not** stop the process. The runtime records the reason in
:attr:`ServiceRuntime.degraded`, ``/healthz`` reports it, and the answer path raises a
taxonomy error (``503 index_missing`` / ``503 calibration_missing``) instead of pretending.
That is what lets an operator see *why* a fresh deployment cannot answer.

Offline mode (``--fake``, profile ``offline``) is fully in-memory and deterministic: the
demo corpus becomes the index, the demo provider answers, and the sealed calibration
artifact is still loaded from disk (so the decision path is the real one). No socket, no
API key, no corpus build — which is what INV-08 asks of a CI path.

    # AT-SVC-07  # AT-OPS-07  # AT-PORT-01  # AT-LAT-03
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from answerdesk import PACKAGE_VERSION
from answerdesk.cache import (
    SemanticCache,
    chunk_map,
)
from answerdesk.cache import (
    ThresholdTable as CacheThresholdTable,
)
from answerdesk.cache.keys import VersionTuple
from answerdesk.config import Settings, load_settings
from answerdesk.contracts import (
    SCHEMA_VERSION,
    ChatProvider,
    Chunk,
    Usage,
)
from answerdesk.errors import (
    CalibrationMissingError,
    CorpusMissingError,
    IndexMissingError,
)
from answerdesk.generate import ANSWER_PROMPT_ID, AnswerPipeline, PromptRegistry
from answerdesk.generate.demo import demo_corpus, demo_responses
from answerdesk.index.embedders import default_embedder
from answerdesk.index.store import ChunkStore, IndexBundle, load_indexes
from answerdesk.llm.presets import provider_config
from answerdesk.policy import (
    CalibrationArtifact,
    DecisionEngine,
    GuardConfig,
    load_guard_config,
)
from answerdesk.policy.scorers import OverlapScorer
from answerdesk.policy.security_agent import SecurityValidator
from answerdesk.retrieval.pipeline import RetrievalPipeline, build_pipeline
from answerdesk.service.config import (
    DEFAULT_CACHE_THRESHOLDS,
    DEFAULT_CALIBRATION_ARTIFACT,
    ServiceSettings,
)
from answerdesk.service.providers import RedactingProvider, wrap_provider
from answerdesk.service.sessions import SQLiteSessionStore
from answerdesk.telemetry.metrics import Metrics
from answerdesk.telemetry.tracing import JsonlTraceWriter, TraceWriter

#: Service version reported by ``/healthz`` and the OpenAPI document.
SERVICE_VERSION = PACKAGE_VERSION

#: Corpus version used by the in-memory demo index (content-addressed, stable).
DEMO_CORPUS_PREFIX = "corpus-demo-"


@dataclass(frozen=True)
class CalibrationState:
    """Where the decision thresholds came from and whether they are still valid."""

    source: str
    threshold_ref: str = ""
    artifact_hash: str = ""
    artifact_corpus_version: str = ""
    age_days: float = 0.0
    stale: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.threshold_ref)


@dataclass
class RuntimeInfo:
    """Immutable-ish facts the health payload and traces report."""

    profile: str = "offline"
    fake: bool = False
    corpus_version: str = ""
    index_version: str = ""
    chunk_count: int = 0
    embedder: str = ""
    model: str = ""
    preset: str = ""
    prompt_version: str = ""
    calibration: CalibrationState = field(
        default_factory=lambda: CalibrationState(source="unavailable")
    )
    cache_enabled: bool = False
    cache_thresholds: str = ""
    degraded: dict[str, str] = field(default_factory=dict)


class ServiceRuntime:
    """Everything one service process needs, created once at startup."""

    def __init__(
        self,
        *,
        settings: Settings,
        service: ServiceSettings,
        embedder: Any,
        index: IndexBundle | None,
        retriever: RetrievalPipeline | None,
        provider: RedactingProvider,
        answerer: AnswerPipeline,
        security: SecurityValidator | None,
        engine: DecisionEngine | None,
        guards: GuardConfig,
        scorer: OverlapScorer,
        cache: SemanticCache | None,
        store: SQLiteSessionStore,
        tracer: TraceWriter,
        metrics: Metrics,
        info: RuntimeInfo,
    ) -> None:
        self.settings = settings
        self.service = service
        self.embedder = embedder
        self.index = index
        self.retriever = retriever
        self.provider = provider
        self.answerer = answerer
        self.security = security
        self.engine = engine
        self.guards = guards
        self.scorer = scorer
        self.cache = cache
        self.store = store
        self.tracer = tracer
        self.metrics = metrics
        self.info = info
        self.started_at = time.monotonic()
        self.started_wall = time.time()
        self._chunk_lookup = (
            chunk_map({chunk.chunk_id: chunk for chunk in index.store.chunks})
            if index is not None
            else None
        )

    # -- construction --------------------------------------------------------------
    @classmethod
    def create(
        cls,
        core: Settings | None = None,
        service: ServiceSettings | None = None,
        *,
        fake: bool | None = None,
        provider: ChatProvider | None = None,
        index: IndexBundle | None = None,
        retriever: RetrievalPipeline | None = None,
        engine: DecisionEngine | None = None,
        store: SQLiteSessionStore | None = None,
        tracer: TraceWriter | None = None,
        metrics: Metrics | None = None,
        embedder: Any = None,
        guards: GuardConfig | None = None,
    ) -> ServiceRuntime:
        """Build a ready runtime; every collaborator can be injected for tests."""
        settings = core if core is not None else load_settings()
        svc = service if service is not None else ServiceSettings()
        is_fake = settings.profile == "offline" if fake is None else fake
        degraded: dict[str, str] = {}

        resolved_embedder = embedder or default_embedder(settings)

        resolved_index = index
        if resolved_index is None:
            if is_fake:
                resolved_index = _demo_index(resolved_embedder)
            else:
                try:
                    resolved_index = load_indexes(settings, embedder=resolved_embedder)
                except (CorpusMissingError, IndexMissingError) as exc:
                    degraded["index"] = exc.message

        resolved_retriever = retriever
        if resolved_retriever is None and resolved_index is not None:
            resolved_retriever = build_pipeline(settings, resolved_index)

        resolved_provider_inner = provider
        if resolved_provider_inner is None:
            if is_fake:
                from answerdesk.testing.doubles import FakeProvider

                resolved_provider_inner = FakeProvider(
                    provider_config("fake"), responses=demo_responses()
                )
            else:
                from answerdesk.llm.factory import build_provider

                resolved_provider_inner = build_provider(settings)
        wrapped = wrap_provider(resolved_provider_inner)
        answerer = AnswerPipeline.from_settings(wrapped, settings)
        security = None
        if settings.security.enabled:
            from answerdesk.policy.security_agent import SecurityValidator

            security = SecurityValidator(
                wrapped,
                fail_closed=settings.security.fail_closed,
                max_chars=settings.security.max_chars,
            )

        resolved_guards = guards or load_guard_config()
        calibration, resolved_engine = _load_engine(settings, svc, resolved_guards, engine=engine)
        if resolved_engine is None and calibration.error:
            degraded["calibration"] = calibration.error

        prompt_version = _prompt_version(answerer)
        versions = VersionTuple(
            corpus_version=resolved_index.store.corpus_version if resolved_index else "",
            index_version=resolved_index.manifest.index_version if resolved_index else "",
            prompt_version=prompt_version,
            model_id=wrapped.config.model,
            policy_artifact_hash=calibration.artifact_hash,
        )
        cache, cache_label = _build_cache(
            settings, svc, resolved_embedder, versions, resolved_index
        )
        if cache is None:
            degraded["cache"] = cache_label

        resolved_store = store or SQLiteSessionStore(svc.database_path(settings))
        resolved_tracer = tracer or JsonlTraceWriter(
            svc.traces_path(settings), max_bytes=svc.trace_max_bytes
        )
        resolved_metrics = metrics or Metrics()

        info = RuntimeInfo(
            profile=settings.profile,
            fake=is_fake,
            corpus_version=versions.corpus_version,
            index_version=versions.index_version,
            chunk_count=len(resolved_index.store) if resolved_index is not None else 0,
            embedder=getattr(resolved_embedder, "name", "unknown"),
            model=wrapped.config.model,
            preset=wrapped.config.preset,
            prompt_version=prompt_version,
            calibration=calibration,
            cache_enabled=bool(cache is not None and cache.enabled),
            cache_thresholds=cache_label,
            degraded=degraded,
        )
        return cls(
            settings=settings,
            service=svc,
            embedder=resolved_embedder,
            index=resolved_index,
            retriever=resolved_retriever,
            provider=wrapped,
            answerer=answerer,
            security=security,
            engine=resolved_engine,
            guards=resolved_guards,
            scorer=OverlapScorer(),
            cache=cache,
            store=resolved_store,
            tracer=resolved_tracer,
            metrics=resolved_metrics,
            info=info,
        )

    # -- capabilities --------------------------------------------------------------
    def require_answer_path(self) -> None:
        """Raise the taxonomy error for a missing dependency (never a silent fallback)."""
        if self.retriever is None:
            raise IndexMissingError(
                "no index is loaded, so no evidence can be retrieved; "
                "run `answerdesk index --demo` (or configure a corpus) first",
                detail=self.info.degraded.get("index"),
            )
        if self.engine is None:
            raise CalibrationMissingError(
                "no calibration artifact is loaded, so the service cannot claim to be "
                "calibrated; run `answerdesk calibrate --synthetic` or point "
                "ANSWERDESK_SERVICE_CALIBRATION_ARTIFACT at a sealed artifact",
                detail=self.info.calibration.error,
            )

    @property
    def chunk_lookup(self) -> Any:
        return self._chunk_lookup

    def usage_from_answer(self, usage: Usage) -> None:
        """Mirror a turn's usage into the metric registry (one place for the cost math)."""
        self.metrics.observe_usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            usd=usage.usd,
            model=usage.model or self.info.model,
        )

    # -- health --------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """The ``/healthz`` payload: dependency status, never a secret (FR-SVC-07)."""
        checks: dict[str, dict[str, Any]] = {}
        checks["service"] = {
            "status": "ok",
            "version": SERVICE_VERSION,
            "detail": self.info.profile,
        }

        db_state = "ok" if self.store is not None else "error"
        db_detail = ""
        try:
            counts = self.store.stats()
            db_detail = f"{counts['sessions']} sessions, {counts['turns']} turns"
        except Exception as exc:  # pragma: no cover - sqlite failure
            db_state = "error"
            db_detail = type(exc).__name__
        checks["database"] = {"status": db_state, "detail": db_detail}

        if self.index is not None:
            checks["index"] = {
                "status": "ok",
                "version": self.info.index_version,
                "detail": f"{self.info.chunk_count} chunks ({self.info.embedder})",
            }
            checks["corpus"] = {
                "status": "ok",
                "version": self.info.corpus_version,
                "detail": "demo corpus (in-memory)" if self.info.fake else "loaded from disk",
            }
        else:
            reason = self.info.degraded.get("index", "not loaded")
            checks["index"] = {"status": "error", "detail": reason}
            checks["corpus"] = {"status": "error", "detail": reason}

        api_key_configured = self.provider.config.api_key is not None
        checks["provider"] = {
            "status": "ok",
            "version": self.provider.config.model,
            "detail": (
                f"preset={self.provider.config.preset} base_url={self.provider.config.base_url} "
                f"api_key={'configured' if api_key_configured else 'absent'}"
            ),
        }

        calibration = self.info.calibration
        if calibration.error:
            checks["calibration"] = {"status": "error", "detail": calibration.error}
        elif calibration.stale:
            checks["calibration"] = {
                "status": "degraded",
                "version": calibration.threshold_ref,
                "detail": (
                    f"stale: age {calibration.age_days:.1f}d, "
                    f"artifact corpus {calibration.artifact_corpus_version!r}"
                ),
            }
        else:
            checks["calibration"] = {
                "status": "ok",
                "version": calibration.threshold_ref,
                "detail": (
                    f"{calibration.source}; age {calibration.age_days:.1f}d; "
                    f"artifact corpus {calibration.artifact_corpus_version!r}"
                ),
            }

        if self.cache is not None and self.cache.enabled:
            stats = self.cache.stats()
            checks["cache"] = {
                "status": "ok",
                "version": self.info.cache_thresholds,
                "detail": (
                    f"{stats.entries} entries, {stats.hits} hits, "
                    f"{stats.misses} misses, {stats.guard_rejections} guard rejections"
                ),
            }
        else:
            checks["cache"] = {
                "status": "disabled",
                "detail": self.info.degraded.get("cache", "cache disabled by configuration"),
            }

        checks["admin"] = {
            "status": "ok" if self.service.admin_enabled else "disabled",
            "detail": "bearer token configured" if self.service.admin_enabled else "no token",
        }
        checks["tracing"] = {
            "status": "ok",
            "detail": repo_relative(getattr(self.tracer, "path", None)),
        }

        degraded = any(check["status"] in ("degraded", "error") for check in checks.values())
        return {
            "status": "degraded" if degraded else "ok",
            "service": "answerdesk",
            "version": SERVICE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "uptime_s": round(time.monotonic() - self.started_at, 3),
            "checks": checks,
        }

    # -- lifecycle -----------------------------------------------------------------
    def refresh_index(self, *, demo: bool = False, force: bool = False) -> dict[str, Any]:
        """Rebuild/reload the index from disk and re-wire the retriever + cache.

        This is what ``POST /admin/refresh`` runs. It is deliberately synchronous and
        explicit: the old index keeps serving until the new one loads, then the runtime
        swaps the references, invalidates the cache for the new corpus version and reports
        the manifest stats. A failure leaves the previous index in place.
        """
        from answerdesk.index.store import build_indexes, load_indexes

        build = build_indexes(self.settings, demo=demo, force=force)
        index = load_indexes(self.settings, embedder=self.embedder)
        retriever = build_pipeline(self.settings, index)
        prompt_version = self.info.prompt_version
        versions = VersionTuple(
            corpus_version=index.store.corpus_version,
            index_version=index.manifest.index_version,
            prompt_version=prompt_version,
            model_id=self.provider.config.model,
            policy_artifact_hash=self.info.calibration.artifact_hash,
        )
        invalidated = 0
        if self.cache is not None:
            try:
                invalidated = self.cache.invalidate(corpus_version=index.store.corpus_version)
            except Exception:  # pragma: no cover - cache invalidation is best effort
                invalidated = 0
        self.index = index
        self.retriever = retriever
        self._chunk_lookup = chunk_map({chunk.chunk_id: chunk for chunk in index.store.chunks})
        self.info.corpus_version = versions.corpus_version
        self.info.index_version = versions.index_version
        self.info.chunk_count = len(index.store)
        self.info.degraded.pop("index", None)
        self.cache, cache_label = _build_cache(
            self.settings, self.service, self.embedder, versions, index
        )
        self.info.cache_enabled = bool(self.cache is not None and self.cache.enabled)
        self.info.cache_thresholds = cache_label
        corpus_dir = self.settings.resolve_path(self.settings.app.corpus_dir)
        documents = 0
        manifest_path = corpus_dir / "manifest.json"
        try:
            from answerdesk.ingest.builder import load_manifest

            manifest = load_manifest(corpus_dir)
            documents = int(manifest.stats.documents)
        except Exception:  # pragma: no cover - a manifest is optional for serving
            documents = 0
        return {
            "corpus_version": versions.corpus_version,
            "index_version": versions.index_version,
            "chunks": len(index.store),
            "documents": documents,
            "reused": build.reused,
            "manifest_path": str(manifest_path),
            "cache_invalidated": invalidated,
            "index_dir": str(build.index_dir),
        }

    def close(self) -> None:
        try:
            self.tracer.close()
        finally:
            with contextlib.suppress(Exception):  # already closed
                self.store.close()


# -- helpers ---------------------------------------------------------------------------
def _demo_index(embedder: Any) -> IndexBundle:
    chunks: Sequence[Chunk] = demo_corpus()
    digest = hashlib.sha256("|".join(chunk.chunk_id for chunk in chunks).encode("utf-8"))
    store = ChunkStore.from_chunks(
        chunks, corpus_version=DEMO_CORPUS_PREFIX + digest.hexdigest()[:12]
    )
    return IndexBundle.build(store, embedder)


#: Repository root: health details report paths relative to it, so an operator reading a
#: health payload (or a pasted one in a ticket) does not learn the host layout (R-27).
REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_relative(path: object) -> str:
    """A path as the repository sees it: relative when inside, basename when outside."""
    if path is None:
        return "not configured"
    try:
        resolved = Path(str(path)).resolve()
    except (OSError, ValueError):  # pragma: no cover - defensive
        return str(path)
    for root in (REPO_ROOT,):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return resolved.name


def _prompt_version(answerer: AnswerPipeline) -> str:
    try:
        registry = getattr(answerer, "registry", None) or PromptRegistry.load()
        return registry.get(ANSWER_PROMPT_ID).prompt_version
    except Exception:  # pragma: no cover - registry is shipped with the package
        return ""


def _load_engine(
    settings: Settings,
    svc: ServiceSettings,
    guards: GuardConfig,
    *,
    engine: DecisionEngine | None,
) -> tuple[CalibrationState, DecisionEngine | None]:
    if engine is not None:
        return (
            CalibrationState(
                source="injected",
                threshold_ref=engine.threshold_ref(),
                artifact_hash=engine.artifact_hash,
                artifact_corpus_version=engine.corpus_version,
                stale=engine.stale,
            ),
            engine,
        )
    configured = svc.calibration_artifact or settings.policy.calibration_artifact
    if configured:
        path = svc.resolve(settings, configured)
        if path is None or not Path(path).is_file():
            return (
                CalibrationState(
                    source=str(path or configured),
                    error=f"calibration artifact not found: {path or configured}",
                ),
                None,
            )
    else:
        path = settings.resolve_path(DEFAULT_CALIBRATION_ARTIFACT)
        if not path.is_file():
            return CalibrationState(
                source="unavailable", error="no calibration artifact found"
            ), None
    try:
        artifact = CalibrationArtifact.load(path)
    except Exception as exc:
        return CalibrationState(
            source=repo_relative(path), error=f"cannot load calibration artifact: {exc}"
        ), None
    try:
        resolved = DecisionEngine.from_artifact(
            artifact,
            guards=guards,
            corpus_version=None,
            max_age_days=svc.calibration_max_age_days,
        )
    except Exception as exc:
        return CalibrationState(
            source=repo_relative(path), error=f"invalid calibration artifact: {exc}"
        ), None
    return (
        CalibrationState(
            source=repo_relative(path),
            threshold_ref=resolved.threshold_ref(),
            artifact_hash=artifact.artifact_hash,
            artifact_corpus_version=artifact.corpus_version,
            age_days=round(artifact.age_days(), 3),
            stale=resolved.stale,
        ),
        resolved,
    )


def _build_cache(
    settings: Settings,
    svc: ServiceSettings,
    embedder: Any,
    versions: VersionTuple,
    index: IndexBundle | None,
) -> tuple[SemanticCache | None, str]:
    if not settings.cache.enabled:
        return None, "disabled by ANSWERDESK_CACHE__ENABLED"
    configured = svc.cache_thresholds or DEFAULT_CACHE_THRESHOLDS
    path = svc.resolve(settings, configured) if configured else None
    thresholds: CacheThresholdTable | None = None
    label = "exact-only (no calibrated thresholds)"
    if path is not None and Path(path).is_file():
        try:
            thresholds = CacheThresholdTable.load(path)
            label = repo_relative(path)
        except Exception as exc:  # pragma: no cover - corrupt artifact
            label = f"exact-only (thresholds unreadable: {type(exc).__name__})"
    lookup = (
        chunk_map({chunk.chunk_id: chunk for chunk in index.store.chunks})
        if index is not None
        else None
    )
    try:
        cache = SemanticCache(
            embedder=embedder,
            versions=versions,
            thresholds=thresholds,
            enabled=True,
            epsilon=settings.cache.epsilon,
            default_threshold=settings.cache.similarity_threshold,
            ttl_by_class=settings.cache.ttl_class,
            high_risk_intents=frozenset(settings.cache.high_risk_intents),
            chunk_lookup=lookup,
        )
    except Exception as exc:  # pragma: no cover - configuration error
        return None, f"cache unavailable: {exc}"
    return cache, label


__all__ = [
    "DEMO_CORPUS_PREFIX",
    "SERVICE_VERSION",
    "CalibrationState",
    "RuntimeInfo",
    "ServiceRuntime",
]
