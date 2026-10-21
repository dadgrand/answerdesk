"""Live components for the evidence run (t11): real embedder, reranker, judge, cache.

Everything here is a thin adapter over the shipped stack — the benchmark must measure the
system the repository ships, not a parallel implementation:

* :class:`CachedApiEmbedder` wraps the hosted ``bge-m3`` embedder (t16) with a disk cache and
  request pacing. The cache is derived data under ``data/index`` (git-ignored); the model id
  and dimension are forwarded unchanged, so the index fingerprint still identifies
  ``api:bge-m3/1024``.
* :func:`build_index` builds or reuses an :class:`~answerdesk.index.store.IndexBundle` through
  the shipped builder, keyed by the chunk set + embedder identity.
* :func:`retrieval_variants` wires the shipped :class:`RetrievalPipeline` into the ablation
  arms (bm25 / dense / RRF k / convex-combination / +reranker).
* :func:`judge_provider` builds a *second* provider for the judge, so §10.5's cross-family
  rule is enforced structurally rather than by convention.

No network call happens at import time and every helper is lazy: the offline gate imports
this module without a key.

    # AT-EVL-05  # AT-EVL-10  # AT-NFR-PORT-02
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from answerdesk.config import PROVIDER_PRESETS, Settings
from answerdesk.contracts import (
    Action,
    Answer,
    CacheInfo,
    ChatProvider,
    Chunk,
    Claim,
    Citation,
    Confidence,
    Decision,
    IntentBucket,
    ProviderConfig,
    ReasonCode,
    Usage,
)
from answerdesk.eval.corpus import Corpus
from answerdesk.index.embedders import api_embedder_from_settings
from answerdesk.index.store import ChunkStore, IndexBundle
from answerdesk.rerank import api_reranker_from_settings
from answerdesk.retrieval.pipeline import build_pipeline
from answerdesk.retrieval.fusion import FusionParams

#: Embedding cache file name (inside the git-ignored index directory).
CACHE_FILE = "bench-embeddings.npz"


def _text_key(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


class EmbeddingCache:
    """Content-addressed vector cache: one ``.npz`` under ``data/index``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._vectors: dict[str, np.ndarray] = {}
        self.dirty = False
        if self.path.is_file():
            try:
                with np.load(self.path, allow_pickle=False) as payload:
                    self._vectors = {key: payload[key] for key in payload.files}
            except (OSError, ValueError):  # pragma: no cover - a corrupt cache is regenerated
                self._vectors = {}

    def __len__(self) -> int:
        return len(self._vectors)

    def get(self, text: str) -> list[float] | None:
        vector = self._vectors.get(_text_key(text))
        return None if vector is None else [float(value) for value in vector]

    def put(self, text: str, vector: Sequence[float]) -> None:
        self._vectors[_text_key(text)] = np.asarray(list(vector), dtype=np.float32)
        self.dirty = True

    def flush(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **self._vectors)
        self.dirty = False


class CachedApiEmbedder:
    """Paced, cached wrapper around the hosted embedder (implements ``Embedder``)."""

    def __init__(
        self,
        inner: Any,
        cache: EmbeddingCache,
        *,
        batch_size: int = 32,
        min_interval_s: float = 0.0,
        max_retries: int = 5,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.batch_size = max(1, batch_size)
        self.min_interval_s = max(0.0, min_interval_s)
        self.max_retries = max(1, max_retries)
        self._last_call = 0.0
        self.calls = 0
        self.cache_hits = 0

    @property
    def name(self) -> str:
        return str(self.inner.name)

    @property
    def dimension(self) -> int:
        return int(self.inner.dimension)

    def _pace(self) -> None:
        if self.min_interval_s <= 0:
            return
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        for attempt in range(self.max_retries):
            try:
                self._pace()
                self.calls += 1
                return self.inner.embed_documents(list(texts))
            except Exception as exc:  # noqa: BLE001 - provider errors are retried, then surfaced
                status = getattr(exc, "status_code", None)
                transient = status in (429, 500, 502, 503, 504) or "429" in str(exc)
                if attempt == self.max_retries - 1 or not transient:
                    raise
                time.sleep(min(30.0, 1.5 * (2**attempt)))
        raise RuntimeError("unreachable")  # pragma: no cover

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = [self.cache.get(text) for text in texts]
        missing = [index for index, vector in enumerate(vectors) if vector is None]
        self.cache_hits += len(texts) - len(missing)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            payload = [texts[index] for index in batch]
            embedded = self._embed(payload)
            for index, text, vector in zip(batch, payload, embedded, strict=True):
                vectors[index] = list(vector)
                self.cache.put(text, vector)
        self.cache.flush()
        return [vector or [] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        cached = self.cache.get(text)
        if cached is not None:
            self.cache_hits += 1
            return cached
        vector = list(self._embed([text])[0])
        self.cache.put(text, vector)
        self.cache.flush()
        return vector


def build_embedder(settings: Settings, cache_dir: str | Path) -> CachedApiEmbedder:
    """Hosted embedder with pacing + cache; requires ``embedding.backend = api``."""
    if settings.embedding.backend != "api":
        raise ValueError(
            "the evidence run needs the hosted embedder: "
            "set ANSWERDESK_EMBEDDING__BACKEND=api (scripts/live-env.ps1 neuraldeep does this)"
        )
    inner = api_embedder_from_settings(settings)
    return CachedApiEmbedder(
        inner,
        EmbeddingCache(Path(cache_dir) / CACHE_FILE),
        batch_size=settings.embedding.batch_size,
        min_interval_s=settings.eval.min_call_interval_s,
    )


def build_index(
    settings: Settings,
    corpus: Corpus,
    embedder: CachedApiEmbedder,
    index_dir: str | Path,
    *,
    force: bool = False,
) -> IndexBundle:
    """Build (or reuse) the index bundle for this corpus + embedder identity."""
    directory = Path(index_dir)
    skip = directory / ".ok"
    if not force and skip.is_file():
        try:
            return IndexBundle.load(directory, embedder)
        except Exception:  # noqa: BLE001 - a stale/incompatible directory is rebuilt
            pass
    store = ChunkStore.from_chunks(list(corpus.chunks), corpus_version=corpus.version)
    started = time.perf_counter()
    bundle = IndexBundle.build(store, embedder)
    bundle.save(directory)
    directory.mkdir(parents=True, exist_ok=True)
    skip.write_text(
        json.dumps(
            {
                "corpus_version": corpus.version,
                "embedder": embedder.name,
                "dimension": embedder.dimension,
                "chunks": len(corpus.chunks),
                "build_s": round(time.perf_counter() - started, 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return bundle


def build_reranker(settings: Settings) -> Any:
    if settings.retrieval.rerank.backend != "api":
        raise ValueError("the evidence run needs the hosted reranker (rerank.backend=api)")
    return api_reranker_from_settings(settings)


def retrieval_variants(
    settings: Settings,
    index: IndexBundle,
    *,
    reranker: Any | None = None,
    alphas: Sequence[float] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9),
) -> dict[str, Any]:
    """The ablation arms, each a shipped ``RetrievalPipeline``.

    ``bm25``/``dense`` are single-channel (the pipeline runs one channel only);
    ``rrf@k`` sweeps the fusion constant; ``cc@alpha`` is convex combination with
    min-max normalisation; ``cc@0.75+rerank`` adds the hosted cross-encoder.
    """
    from answerdesk.retrieval.pipeline import PipelineConfig, RetrievalPipeline

    base = PipelineConfig(
        top_k=settings.retrieval.top_k,
        gate_k=settings.retrieval.gate_k,
        final_k=settings.generate.context_k_max,
        dedupe_cosine=settings.retrieval.dedupe_cosine,
    )
    arms: dict[str, Any] = {}
    arms["bm25"] = RetrievalPipeline.from_index(
        index,
        params=FusionParams(fuser="none"),
        config=base.model_copy(update={"channels": ("bm25",)}),
    )
    arms["dense"] = RetrievalPipeline.from_index(
        index,
        params=FusionParams(fuser="none"),
        config=base.model_copy(update={"channels": ("dense",)}),
    )
    for k in (10, 60, 100):
        arms[f"rrf@{k}"] = RetrievalPipeline.from_index(
            index, params=FusionParams(fuser="rrf", rrf_k=k), config=base
        )
    for alpha in alphas:
        arms[f"cc@{alpha}"] = RetrievalPipeline.from_index(
            index,
            params=FusionParams(fuser="cc", alpha=alpha, normalization="minmax"),
            config=base,
        )
    if reranker is not None:
        arms["cc@0.75+rerank"] = RetrievalPipeline.from_index(
            index,
            params=FusionParams(fuser="cc", alpha=0.75, normalization="minmax"),
            reranker=reranker,
            config=base,
        )
    return arms


def judge_provider(settings: Settings) -> ChatProvider | None:
    """The judge's own provider (``None`` = judge on the generator provider)."""
    if settings.eval.judge_preset is None:
        return None
    from answerdesk.llm.factory import build_provider

    preset_name = settings.eval.judge_preset
    if preset_name not in PROVIDER_PRESETS:
        raise ValueError(f"unknown judge preset {preset_name!r}")
    preset = PROVIDER_PRESETS[preset_name]
    api_key = settings.eval.judge_api_key
    if api_key is None and settings.eval.judge_api_key_env:
        import os

        value = os.environ.get(settings.eval.judge_api_key_env, "")
        if value:
            from pydantic import SecretStr

            api_key = SecretStr(value)
    judge_settings = settings.model_copy(deep=True)
    judge_settings.llm.preset = preset_name
    judge_settings.llm.base_url = settings.eval.judge_base_url or preset.base_url
    judge_settings.llm.model = settings.eval.judge_model
    judge_settings.llm.api_key = api_key
    judge_settings.llm.timeout_s = settings.eval.judge_timeout_s
    judge_settings.llm.max_tokens = settings.eval.judge_max_tokens
    judge_settings.llm.temperature = settings.eval.judge_temperature
    judge_settings.llm.json_mode = False
    return build_provider(judge_settings)


def answer_from_result(
    *,
    result: Any,
    decision: Decision,
    p_correct: float,
    answer_id: str,
) -> Answer:
    """Promote a ``GenerationResult`` to the contract ``Answer`` (the cache's value shape)."""
    claims = [
        Claim(
            claim_id=claim.claim_id,
            text=claim.text,
            citation_ids=list(claim.citation_ids),
            support=claim.support,
        )
        for claim in result.claims
    ]
    citations = [
        Citation(
            citation_id=citation.citation_id,
            chunk_id=citation.chunk_id,
            doc_id=citation.doc_id,
            doc_title=citation.doc_title,
            heading_path=list(citation.heading_path),
            quote=citation.quote,
            start_char=citation.start_char,
            end_char=citation.end_char,
            doc_hash=citation.doc_hash,
        )
        for citation in result.citations
    ]
    return Answer(
        answer_id=answer_id,
        trace_id=f"trace-{answer_id}",
        language=result.language or "en",
        text=result.text,
        claims=claims,
        citations=citations,
        decision=decision,
        confidence=Confidence(p_correct=p_correct, calibrator_version="bench-cache"),
        cache=CacheInfo(status="miss"),
        usage=Usage(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            usd=result.usage.usd,
            model=result.usage.model,
            prompt_version=result.usage.prompt_version,
        ),
    )


def escalation_decision(
    *, action: Action, reason_code: ReasonCode, score: float, alpha: float, threshold_ref: str
) -> Decision:
    return Decision(
        action=action,
        reason_code=reason_code,
        score=min(1.0, max(0.0, score)),
        alpha=alpha,
        threshold_ref=threshold_ref,
    )


def bucket_of(intent: str) -> IntentBucket:
    high_risk = {"account", "refund", "legal", "payment", "fraud", "closure", "security"}
    transactional = {"transfers", "fees", "tariff_schedule", "close_account", "fraud_policy"}
    if intent in high_risk:
        return IntentBucket.ACCOUNT_LEGAL
    if intent in transactional:
        return IntentBucket.TRANSACTIONAL
    return IntentBucket.INFORMATIONAL


__all__ = [
    "CACHE_FILE",
    "CachedApiEmbedder",
    "EmbeddingCache",
    "answer_from_result",
    "bucket_of",
    "build_embedder",
    "build_index",
    "build_reranker",
    "escalation_decision",
    "judge_provider",
    "retrieval_variants",
]
