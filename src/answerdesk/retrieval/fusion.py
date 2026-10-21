"""Fusion of sparse and dense result sets: RRF and convex combination (FR-RET-04).

Two fusers, both implementing :class:`answerdesk.contracts.FusionStrategy`.

**Reciprocal rank fusion** (Cormack et al., 2009) — the untuned cold-start default::

    RRF(d) = Σ_{r ∈ R} 1 / (k_rrf + rank_r(d)),      k_rrf = 60 by default

It uses ranks only, so it needs no score calibration and no per-corpus tuning. That is
also its weakness: it throws away score magnitude, and Bruch et al. (arXiv:2210.11934)
show it is parameter-sensitive and loses to a tuned convex combination both in and out of
domain. Hence: RRF ships as the default, convex combination is the tuned path (D-03).

**Convex combination** — the tuned path::

    CC(d) = Σ_{r ∈ R} w_r · ŝ_r(d),        Σ_r w_r = 1,  w_r ≥ 0

with one weight per channel (default ``w_dense = α = 0.75``, ``w_bm25 = 1 − α``, the
Weaviate convention) and per-channel normalisation ``ŝ_r`` to a common scale. Two
normalisers are provided:

* ``minmax``: ``ŝ = (s − min) / (max − min)`` — bounded to [0, 1], so one channel cannot
  swamp another; but it is **query-relative**: the same chunk gets a different normalised
  score depending on the rest of that query's candidate list, and a single outlier
  compresses everything else. This is the documented pitfall, not an accident.
* ``zscore``: ``ŝ = (s − μ) / σ`` — keeps relative spacing under outliers, but is
  unbounded, so a channel with a long score tail can exceed its weight.

**Degenerate distributions.** A channel whose scores are all equal carries no ranking
information. Min–max would map it to a constant; mapping it to 1.0 would let a useless
channel dominate every fused ranking, so a flat multi-item distribution is mapped to
**0.0** and reported in :attr:`FusionOutcome.degenerate_channels`. A single-item channel
maps to 1.0 (all of that channel's evidence is that one chunk). This asymmetry is the
answer to the "flat cosines" fixture in FR-RET-04.

Fusion is order-deterministic: ties break on ``chunk_id``, never on dict insertion order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from answerdesk.contracts import Channel, FusionInfo, RetrievedPassage

#: Default RRF rank constant (Elasticsearch and Weaviate both default to 60).
DEFAULT_RRF_K = 60

#: Default weight of the dense channel in a two-channel convex combination.
DEFAULT_CC_ALPHA = 0.75

#: Below this, a score vector is treated as flat (float32 rounding noise).
_FLAT_EPS = 1e-12


class Normalization(StrEnum):
    """Per-channel score normalisers available to convex fusion."""

    MINMAX = "minmax"
    ZSCORE = "zscore"


def minmax_normalize(scores: Sequence[float]) -> np.ndarray:
    """``(s − min) / (max − min)`` in [0, 1]; flat multi-item input maps to zeros.

    Returning zeros (rather than ones or a constant) is what stops a channel with no
    discriminative power from contributing a constant offset to every candidate.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == 1:
        return np.ones(1, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if high - low <= _FLAT_EPS:
        return np.zeros(values.size, dtype=np.float64)
    return (values - low) / (high - low)


def zscore_normalize(scores: Sequence[float]) -> np.ndarray:
    """``(s − μ) / σ``; a zero-variance or single-item input maps to zeros/one."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == 1:
        return np.ones(1, dtype=np.float64)
    sigma = float(values.std())
    if sigma <= _FLAT_EPS:
        return np.zeros(values.size, dtype=np.float64)
    return (values - float(values.mean())) / sigma


def normalize(scores: Sequence[float], method: Normalization) -> np.ndarray:
    """Dispatch on the configured normaliser."""
    if method is Normalization.ZSCORE:
        return zscore_normalize(scores)
    return minmax_normalize(scores)


def is_degenerate(scores: Sequence[float]) -> bool:
    """True when a channel's scores carry no ranking information at all."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size <= 1:
        return False
    return float(values.max() - values.min()) <= _FLAT_EPS


@dataclass
class FusionOutcome:
    """Fused ranking plus the diagnostics the trace and the ablations need."""

    passages: list[RetrievedPassage] = field(default_factory=list)
    degenerate_channels: list[str] = field(default_factory=list)
    info: FusionInfo | None = None


def _ordered_keys(result_sets: Mapping[Channel, Sequence[RetrievedPassage]]) -> list[Channel]:
    """Channel order is fixed (BM25 before DENSE) so summation order never varies."""
    return [channel for channel in (Channel.BM25, Channel.DENSE) if channel in result_sets]


def _candidates(
    result_sets: Mapping[Channel, Sequence[RetrievedPassage]],
) -> dict[str, dict[Channel, tuple[int, float]]]:
    """``chunk_id -> {channel: (rank, score)}`` — the fusion input, deduplicated."""
    table: dict[str, dict[Channel, tuple[int, float]]] = {}
    for channel in _ordered_keys(result_sets):
        for passage in result_sets[channel]:
            entry = table.setdefault(passage.chunk.chunk_id, {})
            previous = entry.get(channel)
            if previous is None or passage.rank < previous[0]:
                entry[channel] = (passage.rank, passage.score)
    return table


def _chunk_lookup(
    result_sets: Mapping[Channel, Sequence[RetrievedPassage]],
) -> dict[str, RetrievedPassage]:
    lookup: dict[str, RetrievedPassage] = {}
    for channel in _ordered_keys(result_sets):
        for passage in result_sets[channel]:
            lookup.setdefault(passage.chunk.chunk_id, passage)
    return lookup


def _materialise(
    scores: dict[str, float],
    contributions: dict[str, dict[str, float]],
    lookup: Mapping[str, RetrievedPassage],
    k: int,
) -> list[RetrievedPassage]:
    """Sort by ``(-score, chunk_id)``, truncate to ``k``, rebuild rank 1..n.

    A fused passage reports the channel that contributed most to its score, so a trace
    still says *where* a candidate came from after fusion (FR-RET-08).
    """
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: max(0, k)]
    passages: list[RetrievedPassage] = []
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        source = lookup[chunk_id]
        parts = contributions.get(chunk_id, {})
        best_channel = source.channel
        if parts:
            best_name = max(parts.items(), key=lambda item: (item[1], item[0]))[0]
            best_channel = Channel(best_name)
        passages.append(
            RetrievedPassage(
                chunk=source.chunk,
                channel=best_channel,
                score=float(score),
                rank=rank,
                stage="fusion",
                contributions={name: float(value) for name, value in sorted(parts.items())},
            )
        )
    return passages


class RRFFusion:
    """Reciprocal rank fusion with a tunable rank constant (FR-RET-04)."""

    name = "rrf"

    def __init__(self, rank_constant: int = DEFAULT_RRF_K) -> None:
        if rank_constant < 1:
            raise ValueError("RRF rank_constant must be >= 1")
        self.rank_constant = rank_constant

    def fuse(
        self, result_sets: Mapping[Channel, Sequence[RetrievedPassage]], k: int
    ) -> list[RetrievedPassage]:
        """``FusionStrategy``: fused ranking of length ``min(k, candidates)``."""
        return self.fuse_detailed(result_sets, k).passages

    def fuse_detailed(
        self, result_sets: Mapping[Channel, Sequence[RetrievedPassage]], k: int
    ) -> FusionOutcome:
        table = _candidates(result_sets)
        scores: dict[str, float] = {}
        contributions: dict[str, dict[str, float]] = {}
        for chunk_id, per_channel in table.items():
            total = 0.0
            parts: dict[str, float] = {}
            for channel, (rank, _score) in per_channel.items():
                value = 1.0 / (self.rank_constant + rank)
                parts[channel.value] = value
                total += value
            scores[chunk_id] = total
            contributions[chunk_id] = parts
        info = FusionInfo(fuser="rrf", params={"rrf_k": self.rank_constant})
        return FusionOutcome(
            passages=_materialise(scores, contributions, _chunk_lookup(result_sets), k),
            info=info,
        )


class ConvexFusion:
    """Weighted convex combination of per-channel normalised scores (FR-RET-04)."""

    name = "cc"

    def __init__(
        self,
        *,
        alpha: float = DEFAULT_CC_ALPHA,
        normalization: Normalization | str = Normalization.MINMAX,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.normalization = Normalization(normalization)
        self.weights = dict(weights or {})

    def weights_for(self, channels: Sequence[Channel]) -> dict[Channel, float]:
        """Per-channel weights, normalised to sum to 1 over the channels present.

        Default: ``dense = alpha``, ``bm25 = 1 − alpha``. An explicit ``weights`` mapping
        (a *learnt* artifact, see :mod:`answerdesk.retrieval.fit`) overrides the default;
        missing channels simply do not participate.
        """
        raw: dict[Channel, float] = {}
        for channel in channels:
            configured = self.weights.get(channel.value)
            if configured is not None:
                raw[channel] = float(configured)
            elif len(channels) == 1:
                raw[channel] = 1.0
            else:
                raw[channel] = self.alpha if channel is Channel.DENSE else 1.0 - self.alpha
        total = sum(raw.values())
        if total <= 0.0:
            equal = 1.0 / len(raw) if raw else 0.0
            return {channel: equal for channel in raw}
        return {channel: value / total for channel, value in raw.items()}

    def fuse(
        self, result_sets: Mapping[Channel, Sequence[RetrievedPassage]], k: int
    ) -> list[RetrievedPassage]:
        """``FusionStrategy``: fused ranking of length ``min(k, candidates)``."""
        return self.fuse_detailed(result_sets, k).passages

    def fuse_detailed(
        self, result_sets: Mapping[Channel, Sequence[RetrievedPassage]], k: int
    ) -> FusionOutcome:
        channels = _ordered_keys(result_sets)
        if not channels:
            return FusionOutcome(info=FusionInfo(fuser="cc", params={"alpha": self.alpha}))
        weights = self.weights_for(channels)
        normalised: dict[Channel, dict[str, float]] = {}
        degenerate: list[str] = []
        for channel in channels:
            passages = result_sets[channel]
            values = normalize([passage.score for passage in passages], self.normalization)
            if is_degenerate([passage.score for passage in passages]):
                degenerate.append(channel.value)
            normalised[channel] = {
                passage.chunk.chunk_id: float(values[index])
                for index, passage in enumerate(passages)
            }

        scores: dict[str, float] = {}
        contributions: dict[str, dict[str, float]] = {}
        for chunk_id in {key for mapping in normalised.values() for key in mapping}:
            total = 0.0
            parts: dict[str, float] = {}
            for channel in channels:
                value = normalised[channel].get(chunk_id)
                if value is None:
                    continue
                weight = weights[channel]
                weighted = weight * value
                parts[channel.value] = weighted
                total += weighted
            scores[chunk_id] = total
            contributions[chunk_id] = parts

        info = FusionInfo(
            fuser="cc",
            params={
                "alpha": round(self.alpha, 6),
                "normalization": self.normalization.value,
                **{f"w_{channel.value}": round(weight, 6) for channel, weight in weights.items()},
            },
        )
        return FusionOutcome(
            passages=_materialise(scores, contributions, _chunk_lookup(result_sets), k),
            degenerate_channels=degenerate,
            info=info,
        )


class SingleChannelFusion:
    """Passthrough for a one-channel configuration (``bm25`` / ``dense`` ablation arms)."""

    name = "none"

    def fuse(
        self, result_sets: Mapping[Channel, Sequence[RetrievedPassage]], k: int
    ) -> list[RetrievedPassage]:
        return self.fuse_detailed(result_sets, k).passages

    def fuse_detailed(
        self, result_sets: Mapping[Channel, Sequence[RetrievedPassage]], k: int
    ) -> FusionOutcome:
        channels = _ordered_keys(result_sets)
        if not channels:
            return FusionOutcome(info=FusionInfo(fuser="none", params={}))
        channel = channels[0]
        passages = list(result_sets[channel])[: max(0, k)]
        rebuilt = [
            passage.model_copy(
                update={
                    "rank": rank,
                    "stage": "fusion",
                    "contributions": {channel.value: float(passage.score)},
                }
            )
            for rank, passage in enumerate(passages, start=1)
        ]
        return FusionOutcome(
            passages=rebuilt, info=FusionInfo(fuser="none", params={"channel": channel.value})
        )


class FusionParams(BaseModel):
    """The tunable surface of fusion, and the shape of the persisted artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fuser: Literal["rrf", "cc", "none"] = "rrf"
    rrf_k: int = Field(default=DEFAULT_RRF_K, ge=1, le=1000)
    alpha: float | None = Field(default=None, gt=0.0, lt=1.0)
    normalization: Normalization = Normalization.MINMAX
    weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _cc_needs_alpha_or_weights(self) -> FusionParams:
        if self.fuser == "cc" and self.alpha is None and not self.weights:
            raise ValueError("convex fusion needs either alpha or explicit per-channel weights")
        return self

    def build(self) -> RRFFusion | ConvexFusion | SingleChannelFusion:
        """Instantiate the strategy this parameter set describes."""
        if self.fuser == "rrf":
            return RRFFusion(self.rrf_k)
        if self.fuser == "cc":
            return ConvexFusion(
                alpha=self.alpha if self.alpha is not None else DEFAULT_CC_ALPHA,
                normalization=self.normalization,
                weights=self.weights,
            )
        return SingleChannelFusion()


__all__ = [
    "DEFAULT_CC_ALPHA",
    "DEFAULT_RRF_K",
    "ConvexFusion",
    "FusionOutcome",
    "FusionParams",
    "Normalization",
    "RRFFusion",
    "SingleChannelFusion",
    "is_degenerate",
    "minmax_normalize",
    "normalize",
    "zscore_normalize",
]
