"""The near-miss replay set: how a wrong hit is measured, and why the set is adversarial.

Requirements §9.5 is explicit that the set must be built adversarially — random query pairs
are too easy and would make the wrong-hit budget vacuous. A pair is a *near miss* when the
two questions are embedding-similar but differ in a detail that changes the answer: a
different plan, amount, date, order state or locale.

The synthetic generator produces three families per cohort:

* **paraphrase** — same tokens in a different order/punctuation: high similarity, same gold
  answer (these are the hits the cache is *for*, and they define K-2's hit-rate floor);
* **near-miss** — one entity changed (plan suffix, amount, date): high similarity, different
  gold answer (a hit here is a wrong hit);
* **distractor** — related but different intent: lower similarity, different answer.

The similarity a pair receives is whatever the configured embedder says, not a label: the
calibration in ``thresholds.py`` only sees (similarity, wrong?) and has to find the threshold
that keeps the budget. That is the honest direction of the test.

    # AT-CAC-07
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence

from pydantic import Field

from answerdesk.cache.keys import cohort_of, extract_entities, normalise_query
from answerdesk.contracts import Contract, Embedder, IntentBucket

#: The replay-set generator's version, recorded in the threshold table's provenance.
REPLAY_VERSION = "near-miss-replay-1.0.0"

LOCALES: tuple[str, ...] = ("ru", "tg", "uz", "ky", "tr", "zh")

_BASE_BY_BUCKET: dict[IntentBucket, str] = {
    IntentBucket.INFORMATIONAL: "Сколько стоит обслуживание дебетовой карты «Вектор»?",
    IntentBucket.TRANSACTIONAL: "Где мой возврат по платежу 4 500 рублей от 05.03.2026?",
    IntentBucket.ACCOUNT_LEGAL: "Как закрыть счёт и отозвать согласие на обработку данных?",
}

_PARAPHRASE_SUFFIXES: tuple[str, ...] = (
    "Сколько стоит обслуживание дебетовой карты «Вектор»",
    "Обслуживание дебетовой карты «Вектор» стоит сколько",
    "Сколько стоит обслуживание дебетовой карты «Вектор» ?",
)

#: Near misses change exactly one answer-relevant detail.
_NEAR_MISS_VARIANTS: tuple[str, ...] = (
    "Сколько стоит обслуживание дебетовой карты «Вектор Плюс»?",
    "Сколько стоит обслуживание кредитной карты «Вектор»?",
    "Сколько стоит обслуживание дебетовой карты «Вектор» в 2027 году?",
    "Сколько стоит обслуживание дебетовой карты «Вектор» для юридических лиц?",
)

_DISTRACTORS: tuple[str, ...] = (
    "Как заблокировать карту «Вектор»?",
    "Какие документы нужны для оформления карты?",
    "Какой кэшбэк начисляется по карте «Вектор»?",
)


class NearMissPair(Contract):
    """One replay pair: the cached question, the new question and the truth about them."""

    pair_id: str = Field(min_length=1)
    cohort: str = Field(min_length=1)
    family: str = Field(pattern=r"^(paraphrase|near_miss|distractor)$")
    locale: str = Field(min_length=2)
    intent_bucket: IntentBucket
    query_prev: str = Field(min_length=1)
    query_new: str = Field(min_length=1)
    same_answer: bool

    @property
    def wrong_if_hit(self) -> bool:
        return not self.same_answer


class EmbeddedPair(Contract):
    """A pair plus the two facts the cache actually acts on.

    * ``similarity`` — what the embedder says (the semantic path's input);
    * ``entities_compatible`` — whether the G-2 entity guard would let the hit through;
    * ``exact_equivalent`` — whether normalisation makes the two queries the *same* key, in
      which case the exact cache serves them regardless of any threshold (FR-CAC-01).

    A hit is therefore `exact_equivalent or (similarity >= tau and entities_compatible)`;
    calibating tau on raw similarity alone would ignore the guard that makes the semantic
    path safe in the first place.
    """

    pair: NearMissPair
    similarity: float = Field(ge=0.0, le=1.0)
    entities_compatible: bool = True
    exact_equivalent: bool = False

    def hits_at(self, threshold: float) -> bool:
        if self.exact_equivalent:
            return True
        return self.similarity >= threshold and self.entities_compatible

    @property
    def wrong_if_served(self) -> bool:
        return self.pair.wrong_if_hit


class ReplaySet(Contract):
    """The whole replay set with its content hash (the threshold table's provenance)."""

    version: str = REPLAY_VERSION
    seed: int = 0
    pairs: list[NearMissPair] = Field(default_factory=list)
    replay_hash: str = ""

    def by_cohort(self) -> dict[str, list[NearMissPair]]:
        grouped: dict[str, list[NearMissPair]] = {}
        for pair in self.pairs:
            grouped.setdefault(pair.cohort, []).append(pair)
        return grouped

    def summary(self) -> dict[str, object]:
        families: dict[str, int] = {}
        for pair in self.pairs:
            families[pair.family] = families.get(pair.family, 0) + 1
        return {
            "version": self.version,
            "seed": self.seed,
            "n_pairs": len(self.pairs),
            "replay_hash": self.replay_hash,
            "families": families,
            "cohorts": sorted(self.by_cohort()),
            "locales": sorted({pair.locale for pair in self.pairs}),
        }


def _hash_pairs(pairs: Sequence[NearMissPair]) -> str:
    payload = json.dumps(
        [pair.model_dump(mode="json") for pair in pairs], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_replay_set(
    *,
    seed: int = 20261020,
    pairs_per_cohort: int = 120,
    buckets: Sequence[IntentBucket] = tuple(IntentBucket),
    locales: Sequence[str] = LOCALES,
) -> ReplaySet:
    """Deterministically build the near-miss replay set per cohort."""
    rng = random.Random(seed)
    pairs: list[NearMissPair] = []
    index = 0
    for bucket in buckets:
        base = _BASE_BY_BUCKET[bucket]
        for locale in locales:
            cohort = cohort_of(intent_bucket=bucket, locale=locale, query=base)
            for slot in range(pairs_per_cohort):
                family = ("paraphrase", "near_miss", "distractor")[slot % 3]
                if family == "paraphrase":
                    query_new = base
                    if slot % 9 == 0:
                        query_new = _PARAPHRASE_SUFFIXES[rng.randrange(len(_PARAPHRASE_SUFFIXES))]
                elif family == "near_miss":
                    query_new = _NEAR_MISS_VARIANTS[rng.randrange(len(_NEAR_MISS_VARIANTS))]
                else:
                    query_new = _DISTRACTORS[rng.randrange(len(_DISTRACTORS))]
                pairs.append(
                    NearMissPair(
                        pair_id=f"nm-{index:04d}",
                        cohort=cohort,
                        family=family,
                        locale=locale,
                        intent_bucket=bucket,
                        query_prev=base,
                        query_new=query_new,
                        same_answer=family == "paraphrase",
                    )
                )
                index += 1
    return ReplaySet(seed=seed, pairs=pairs, replay_hash=_hash_pairs(pairs))


def embed_replay_set(replay: ReplaySet, embedder: Embedder) -> list[EmbeddedPair]:
    """Measure the similarity of every pair with the configured embedder."""
    embedded: list[EmbeddedPair] = []
    for pair in replay.pairs:
        left = embedder.embed_query(pair.query_prev)
        right = embedder.embed_query(pair.query_new)
        similarity = sum(a * b for a, b in zip(left, right, strict=True))
        entities_compatible = extract_entities(pair.query_prev).matches(
            extract_entities(pair.query_new)
        )
        embedded.append(
            EmbeddedPair(
                pair=pair,
                similarity=max(0.0, min(1.0, similarity)),
                entities_compatible=entities_compatible,
                exact_equivalent=normalise_query(pair.query_prev)
                == normalise_query(pair.query_new),
            )
        )
    return embedded


def similarities_by_cohort(embedded: Sequence[EmbeddedPair]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for item in embedded:
        grouped.setdefault(item.pair.cohort, []).append(item.similarity)
    return grouped


def wrong_by_cohort(embedded: Sequence[EmbeddedPair]) -> dict[str, list[bool]]:
    grouped: dict[str, list[bool]] = {}
    for item in embedded:
        grouped.setdefault(item.pair.cohort, []).append(item.pair.wrong_if_hit)
    return grouped


def compatibility_by_cohort(embedded: Sequence[EmbeddedPair]) -> dict[str, list[bool]]:
    grouped: dict[str, list[bool]] = {}
    for item in embedded:
        grouped.setdefault(item.pair.cohort, []).append(item.entities_compatible)
    return grouped


def exact_by_cohort(embedded: Sequence[EmbeddedPair]) -> dict[str, list[bool]]:
    grouped: dict[str, list[bool]] = {}
    for item in embedded:
        grouped.setdefault(item.pair.cohort, []).append(item.exact_equivalent)
    return grouped


__all__ = [
    "LOCALES",
    "REPLAY_VERSION",
    "EmbeddedPair",
    "NearMissPair",
    "ReplaySet",
    "compatibility_by_cohort",
    "embed_replay_set",
    "exact_by_cohort",
    "generate_replay_set",
    "similarities_by_cohort",
    "wrong_by_cohort",
]
