"""Deterministic near-duplicate removal with MinHash + LSH (FR-ING-06).

Why MinHash and not an embedding: deduplication must be *reproducible offline, before
any model exists* — it runs on the ingest path, in CI, with no network and no weights.
MinHash over word shingles is exact in its inputs and deterministic in its randomness,
which is what "same input -> same survivors" requires.

Signature
---------
For a document ``d`` with shingle set ``S(d)``::

    sig_i(d) = min_{s in S(d)} ((a_i * h(s) + b_i) mod p),  p = 2^61 - 1
    J_hat(d1, d2) = (1/k) * sum_i [sig_i(d1) == sig_i(d2)]

``h`` is BLAKE2b (never Python's randomised ``hash()``), ``(a_i, b_i)`` come from a
seeded Mersenne Twister, so the signature of a text is identical in every process, on
every OS, forever. ``k = num_perm``; the estimator has standard error ``sqrt(J(1-J)/k)``
(0.062 at J = 0.5 with k = 64).

Decision rule
-------------
Items are processed in ``id`` order, so the survivor set does not depend on iteration
order. An item is dropped when its **exact** Jaccard similarity to an already-kept item
is ``>= threshold`` (default 0.80) — candidates come from LSH banding, but the decision
uses the exact shingle-set Jaccard, so the threshold means what it says rather than
"roughly, on average". The kept item is the earlier one in ``id`` order: choose the
survivor by content, never by file-walk order.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from pydantic import model_validator

from answerdesk.ingest.structure import IngestModel

_PRIME = (1 << 61) - 1
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: Default Jaccard threshold. Two support articles that share 80% of their 5-word
#: shingles are the same article: they answer the same question with the same wording.
DEFAULT_THRESHOLD = 0.80
DEFAULT_NUM_PERM = 64
DEFAULT_SHINGLE_SIZE = 5
DEFAULT_SEED = 1337
DEFAULT_BANDS = 16


class DedupConfig(IngestModel):
    """Deduplication knobs, recorded in the manifest so a run is reproducible."""

    threshold: float = DEFAULT_THRESHOLD
    num_perm: int = DEFAULT_NUM_PERM
    shingle_size: int = DEFAULT_SHINGLE_SIZE
    seed: int = DEFAULT_SEED
    bands: int = DEFAULT_BANDS
    #: Bound on the shingled set used for the signature: keeps a 600-document build
    #: fast without changing determinism (the shingles are sorted before truncation).
    max_shingles: int = 512

    @model_validator(mode="after")
    def _bounds(self) -> DedupConfig:
        if not 0.5 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0.5, 1.0]")
        if self.num_perm < 8:
            raise ValueError("num_perm must be >= 8")
        if self.num_perm % self.bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        if self.shingle_size < 1:
            raise ValueError("shingle_size must be >= 1")
        if self.max_shingles < 1:
            raise ValueError("max_shingles must be >= 1")
        return self


class DuplicatePair(IngestModel):
    """One removal decision, with the evidence that justified it."""

    dropped_id: str
    kept_id: str
    similarity: float
    exact_text_match: bool = False


class DedupResult(IngestModel):
    """Survivors plus every removal, in input order."""

    kept: list[str]
    dropped: list[DuplicatePair]
    threshold: float
    num_perm: int
    shingle_size: int
    seed: int
    bands: int

    @property
    def removed_count(self) -> int:
        return len(self.dropped)


def normalise_for_shingles(text: str) -> str:
    """Case-fold, NFKC, collapse whitespace — nothing else (no stemming, no stopwords)."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_TOKEN_RE.findall(text))


def text_hash(text: str) -> str:
    """Stable content hash of the normalised text (exact-duplicate fast path)."""
    return hashlib.sha256(normalise_for_shingles(text).encode("utf-8")).hexdigest()


def shingles_of(text: str, *, shingle_size: int = DEFAULT_SHINGLE_SIZE) -> frozenset[int]:
    """Hashed word n-grams; a text shorter than ``n`` yields one shingle of its words."""
    tokens = normalise_for_shingles(text).split()
    if not tokens:
        return frozenset()
    n = max(1, shingle_size)
    if len(tokens) < n:
        return frozenset({_hash_gram(" ".join(tokens))})
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return frozenset(_hash_gram(gram) for gram in grams)


def _hash_gram(gram: str) -> int:
    return int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    """Exact Jaccard similarity of two shingle sets."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    return intersection / len(left | right)


def permutations(config: DedupConfig) -> list[tuple[int, int]]:
    """``num_perm`` seeded (a, b) pairs — the MinHash family for this configuration."""
    import random

    rng = random.Random(config.seed)
    return [
        (rng.randrange(1, _PRIME - 1), rng.randrange(0, _PRIME - 1)) for _ in range(config.num_perm)
    ]


def minhash_signature(shingles: frozenset[int], perms: list[tuple[int, int]]) -> tuple[int, ...]:
    """MinHash signature of a shingle set; deterministic for fixed ``perms``."""
    if not shingles:
        return tuple([_PRIME] * len(perms))
    return tuple(min(((a * shingle + b) % _PRIME) for shingle in shingles) for a, b in perms)


class NearDuplicateDetector:
    """Banded-LSH candidate search + exact Jaccard confirmation."""

    def __init__(self, config: DedupConfig | None = None) -> None:
        self.config = config or DedupConfig()
        self._perms = permutations(self.config)
        self.rows_per_band = self.config.num_perm // self.config.bands

    def signature(self, text: str) -> tuple[int, ...]:
        return minhash_signature(self.shingles(text), self._perms)

    def shingles(self, text: str) -> frozenset[int]:
        shingles = shingles_of(text, shingle_size=self.config.shingle_size)
        limit = self.config.max_shingles
        if limit <= 0 or len(shingles) <= limit:
            return shingles
        return frozenset(sorted(shingles)[:limit])

    def similarity(self, left: str, right: str) -> float:
        """Exact Jaccard of two texts (used by tests and by the confirmation step)."""
        return jaccard(self.shingles(left), self.shingles(right))

    def dedupe(self, items: list[tuple[str, str]]) -> DedupResult:
        """Remove near-duplicates from ``(id, text)`` pairs; survivors keep input order."""
        ordered = sorted(items, key=lambda item: item[0])
        kept: dict[str, str] = {}
        kept_shingles: dict[str, frozenset[int]] = {}
        kept_hashes: dict[str, str] = {}
        buckets: dict[tuple[int, bytes], list[str]] = {}
        dropped: list[DuplicatePair] = []

        for item_id, text in ordered:
            shingles = self.shingles(text)
            digest = text_hash(text)
            match: tuple[str, float, bool] | None = None
            # 1. exact-duplicate fast path
            for candidate_id, candidate_hash in kept_hashes.items():
                if candidate_hash == digest:
                    match = (candidate_id, 1.0, True)
                    break
            # 2. LSH candidates confirmed by exact Jaccard
            if match is None:
                candidates: set[str] = set()
                for band, band_hash in self._band_keys(self.signature(text)):
                    candidates.update(buckets.get((band, band_hash), ()))
                for candidate_id in sorted(candidates):
                    similarity = jaccard(shingles, kept_shingles[candidate_id])
                    if similarity >= self.config.threshold:
                        match = (candidate_id, similarity, False)
                        break
            if match is not None:
                dropped.append(
                    DuplicatePair(
                        dropped_id=item_id,
                        kept_id=match[0],
                        similarity=round(match[1], 6),
                        exact_text_match=match[2],
                    )
                )
                continue
            kept[item_id] = text
            kept_shingles[item_id] = shingles
            kept_hashes[item_id] = digest
            for key in self._band_keys(self.signature(text)):
                buckets.setdefault(key, []).append(item_id)

        return DedupResult(
            kept=[item_id for item_id, _ in items if item_id in kept],
            dropped=sorted(dropped, key=lambda pair: pair.dropped_id),
            threshold=self.config.threshold,
            num_perm=self.config.num_perm,
            shingle_size=self.config.shingle_size,
            seed=self.config.seed,
            bands=self.config.bands,
        )

    def _band_keys(self, signature: tuple[int, ...]) -> list[tuple[int, bytes]]:
        keys: list[tuple[int, bytes]] = []
        for band in range(self.config.bands):
            start = band * self.rows_per_band
            block = signature[start : start + self.rows_per_band]
            digest = hashlib.blake2b(
                ",".join(str(value) for value in block).encode("utf-8"), digest_size=8
            ).digest()
            keys.append((band, digest))
        return keys


__all__ = [
    "DEFAULT_BANDS",
    "DEFAULT_NUM_PERM",
    "DEFAULT_SEED",
    "DEFAULT_SHINGLE_SIZE",
    "DEFAULT_THRESHOLD",
    "DedupConfig",
    "DedupResult",
    "DuplicatePair",
    "NearDuplicateDetector",
    "jaccard",
    "minhash_signature",
    "normalise_for_shingles",
    "permutations",
    "shingles_of",
    "text_hash",
]
