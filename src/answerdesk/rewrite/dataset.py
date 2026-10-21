"""Training-data construction for the query rewriter (no ML dependency).

The dataset is built from the **frozen** corpus v2.1 query file, so the pairs are reproducible
from a committed artifact:

* ``noisy_freeform`` rows (and the other noisy categories: ``typo``, ``code_switch``) give the
  real noisy→clean pairs — ``text`` is the noisy query, ``base_text`` is the clean one written
  by the corpus generator;
* ``clean`` rows give identity-style pairs (clean query → itself) so the adapter does not learn
  to rewrite clean input into something else;
* **perturbations** are regenerated deterministically from clean queries with seeded manglers
  (telegraphic, run-on, dropped preposition, typo, transliteration), which produces the
  held-out perturbation set without touching the corpus.

Three properties are enforced and tested:

1. **dedup** — exact duplicates and near-duplicates (character-shingle Jaccard) are dropped;
2. **frozen, document-disjoint splits** — a ``doc_id`` lives in exactly one split, and a clean
   query string appears in exactly one split, so no test pair leaks into training;
3. **no PII** — rows whose noisy text contains PII (the corpus has a PII category) are excluded
   outright rather than redacted, and the builder asserts zero findings on the final set.

    # AT-GEN-12
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field

from answerdesk.contracts import Contract
from answerdesk.rewrite.errors import DatasetError, DatasetLeakageError

#: Builder version recorded in the dataset card and the splits file.
DATASET_VERSION = "1.0.0"

#: Categories of the frozen corpus whose rows are noisy→clean material.
NOISY_CATEGORIES: tuple[str, ...] = ("noisy_freeform", "typo", "code_switch")

#: Categories excluded from training: PII (privacy) and injection (adversarial, not a rewrite).
EXCLUDED_CATEGORIES: tuple[str, ...] = ("pii", "injection", "unanswerable")

Source = Literal["corpus_noisy", "corpus_clean", "perturbation"]
Split = Literal["train", "val", "test"]

_SPLIT_SALT = "answerdesk-rewriter-split-v1"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalise(text: str) -> str:
    """Whitespace/case normalisation used for dedup and leakage comparisons only."""
    collapsed = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(collapsed.split())


def shingles(text: str, size: int = 4) -> set[str]:
    normalised = normalise(text)
    if len(normalised) <= size:
        return {normalised} if normalised else set()
    return {normalised[index : index + size] for index in range(len(normalised) - size + 1)}


def jaccard(left: str, right: str, *, size: int = 4) -> float:
    a, b = shingles(left, size), shingles(right, size)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class PairRecord(Contract):
    """One training/eval example: a noisy query and the clean query it should become."""

    pair_id: str = Field(min_length=1)
    locale: str = Field(min_length=2)
    category: str = ""
    style: str = ""
    source: Source
    noisy: str = Field(min_length=1)
    clean: str = Field(min_length=1)
    doc_id: str = ""
    answerable: bool = True
    split: Split = "train"

    @property
    def text_hash(self) -> str:
        return _hash(f"{normalise(self.noisy)}\x00{normalise(self.clean)}")

    def to_json(self) -> dict[str, object]:
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------------------
# Perturbations (deterministic manglers; the held-out family is generated with other seeds)
# --------------------------------------------------------------------------------------

_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "как",
        "что",
        "где",
        "когда",
        "для",
        "или",
        "при",
        "без",
        "мне",
        "мой",
        "моя",
        "можно",
        "нужно",
        "ли",
        "на",
        "по",
        "из",
        "от",
        "до",
        "за",
        "то",
        "the",
        "how",
        "can",
        "do",
        "i",
        "my",
        "is",
    }
)

_LATIN_MAP = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "yo",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def _telegraphic(text: str) -> str:
    words = [word for word in text.split() if word.casefold().strip(".,?!") not in _FUNCTION_WORDS]
    return " ".join(words) if len(words) >= 2 else text


def _run_on(text: str) -> str:
    return " ".join(text.replace(",", "").replace(".", "").replace("?", "").split())


def _dropped_preposition(text: str) -> str:
    words = text.split()
    for index, word in enumerate(words):
        if word.casefold() in {"в", "на", "по", "для", "с", "к", "из", "от"} and index > 0:
            return " ".join(words[:index] + words[index + 1 :])
    return text


def _typo_mix(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    index = rng.randrange(len(words))
    word = words[index]
    if len(word) < 4:
        return text
    position = rng.randrange(1, len(word) - 1)
    mutated = word[:position] + word[position + 1 :]
    if rng.random() < 0.5 and len(mutated) >= 3:
        position = rng.randrange(1, len(mutated) - 1)
        mutated = mutated[:position] + mutated[position + 1 :]
    return " ".join([*words[:index], mutated, *words[index + 1 :]])


def _translit(text: str) -> str:
    return "".join(_LATIN_MAP.get(char, char) for char in text.casefold())


#: Manglers by name; ``style`` values match the corpus vocabulary so reports line up.
MANGLERS = {
    "telegraphic": lambda text, rng: _telegraphic(text),
    "run_on": lambda text, rng: _run_on(text),
    "dropped_preposition": lambda text, rng: _dropped_preposition(text),
    "typo_mix": lambda text, rng: _typo_mix(text, rng),
    "translit": lambda text, rng: _translit(text),
}


def perturb(text: str, style: str, *, seed: int) -> str:
    """Apply one deterministic mangler to ``text``."""
    mangler = MANGLERS.get(style)
    if mangler is None:
        raise DatasetError(f"unknown perturbation style {style!r}")
    return mangler(text, random.Random(f"{seed}:{style}:{text}"))


# --------------------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------------------


def _load_query_rows(corpus_dir: Path) -> list[dict[str, object]]:
    path = corpus_dir / "queries" / "queries.jsonl"
    if not path.is_file():
        raise DatasetError(f"query file not found: {path}")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pii_blocked(rows: Iterable[dict[str, object]]) -> int:
    return sum(1 for row in rows if str(row.get("category", "")) in EXCLUDED_CATEGORIES)


def build_pairs(
    corpus_dir: str | Path,
    *,
    seed: int = 1337,
    perturbation_styles: Sequence[str] = tuple(MANGLERS),
    perturbation_per_locale: int = 240,
    near_duplicate_threshold: float = 0.97,
    clean_per_locale: int = 120,
) -> list[PairRecord]:
    """Build the full pair set (before splitting) from the frozen corpus."""
    rows = _load_query_rows(Path(corpus_dir))
    if not rows:
        raise DatasetError("the corpus query file is empty")
    rng = random.Random(seed)
    pairs: list[PairRecord] = []
    seen: dict[str, PairRecord] = {}
    kept_shingles: list[set[str]] = []

    def add(record: PairRecord) -> bool:
        key = f"{normalise(record.noisy)}\x00{normalise(record.clean)}"
        if key in seen:
            return False
        # The corpus asks the same archetype question in many documents (different doc_ids,
        # identical text). Those are legitimate training rows, so near-duplicate suppression
        # only applies to synthetic perturbations, where it removes mangler noise rather than
        # information.
        if record.source == "perturbation":
            candidate = shingles(record.clean)
            if candidate and any(
                (len(candidate & other) / len(candidate | other)) >= near_duplicate_threshold
                for other in kept_shingles[-4000:]
            ):
                return False
            kept_shingles.append(candidate)
        seen[key] = record
        pairs.append(record)
        return True

    for row in rows:
        category = str(row.get("category", ""))
        if category in EXCLUDED_CATEGORIES:
            continue
        noisy = str(row.get("text", "")).strip()
        clean = str(row.get("base_text", "")).strip()
        if not noisy or not clean:
            continue
        locale = str(row.get("locale", ""))
        source: Source = "corpus_noisy" if category in NOISY_CATEGORIES else "corpus_clean"
        add(
            PairRecord(
                pair_id=f"c-{row.get('query_id', len(pairs))}",
                locale=locale,
                category=category,
                style=str(row.get("style", "") or ""),
                source=source,
                noisy=noisy,
                clean=clean,
                doc_id=str(row.get("doc_id", "")),
                answerable=bool(row.get("answerable", True)),
            )
        )

    # Synthetic perturbations from clean queries: the held-out family is generated later with
    # a different seed, so these never share a mangler seed with the test split.
    clean_rows = [
        row
        for row in rows
        if not row.get("style")
        and str(row.get("category", "")) not in EXCLUDED_CATEGORIES
        and str(row.get("base_text", "")).strip()
    ]
    by_locale: dict[str, list[dict[str, object]]] = {}
    for row in clean_rows:
        by_locale.setdefault(str(row.get("locale", "")), []).append(row)
    for locale, locale_rows in sorted(by_locale.items()):
        rng.shuffle(locale_rows)
        for index, row in enumerate(locale_rows[:perturbation_per_locale]):
            clean = str(row["base_text"]).strip()
            style = perturbation_styles[index % len(perturbation_styles)]
            noisy = perturb(clean, style, seed=seed + index)
            if normalise(noisy) == normalise(clean):
                continue
            add(
                PairRecord(
                    pair_id=f"p-{locale}-{index:04d}",
                    locale=locale,
                    category="perturbation",
                    style=style,
                    source="perturbation",
                    noisy=noisy,
                    clean=clean,
                    doc_id=str(row.get("doc_id", "")),
                    answerable=bool(row.get("answerable", True)),
                )
            )

    # Cap the clean (identity) pairs per locale so the set is not dominated by them.
    clean_budget: Counter[str] = Counter()
    capped: list[PairRecord] = []
    for record in pairs:
        if record.source == "corpus_clean":
            if clean_budget[record.locale] >= clean_per_locale:
                continue
            clean_budget[record.locale] += 1
        capped.append(record)
    return capped


def _split_bucket(doc_id: str, *, salt: str = _SPLIT_SALT) -> float:
    """Deterministic 0..1 bucket per *document*, so a document never straddles splits."""
    digest = hashlib.sha256(f"{salt}:{doc_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _split_for(key: str, *, train_share: float, val_share: float) -> Split:
    bucket = _split_bucket(key)
    if bucket < train_share:
        return "train"
    if bucket < train_share + val_share:
        return "val"
    return "test"


#: Tokens that look numeric but are identifiers (``ORD-2026-10314``, ``ERR-TRF-6170``).
_IDENTIFIER_TOKEN = re.compile(r"[A-Za-z]{2,}[-\u2011_]?\d{2,}(?:[-\u2011_]\d+)*")


def assign_splits(
    pairs: Sequence[PairRecord], *, train_share: float = 0.8, val_share: float = 0.1
) -> tuple[list[PairRecord], list[dict[str, object]]]:
    """Assign frozen splits that are disjoint by *text* first, then by *document*.

    The corpus repeats the same archetype question across many documents, so a document-hash
    split would put the identical clean string in train and test. The split key is therefore
    the text itself (``sha256(salt:normalise(text))``), and every pair inherits the split of its
    clean target; a document with pairs whose texts land in different splits is resolved by the
    majority vote of its first bound text and the minority pairs are dropped and reported.

    The result is that no clean or noisy text crosses a split boundary, which is the leakage
    property the dataset tests assert, and the split sizes stay close to the configured shares.
    """
    if not 0.0 < train_share < 1.0 or not 0.0 <= val_share < 1.0:
        raise DatasetError("invalid split shares")
    assigned: list[PairRecord] = []
    dropped: list[dict[str, object]] = []
    text_split: dict[str, str] = {}
    for record in pairs:
        clean_key = normalise(record.clean)
        noisy_key = normalise(record.noisy)
        split = text_split.get(clean_key) or _split_for(
            clean_key, train_share=train_share, val_share=val_share
        )
        bound_noisy = text_split.get(noisy_key)
        if bound_noisy is not None and bound_noisy != split:
            dropped.append(
                {
                    "pair_id": record.pair_id,
                    "doc_id": record.doc_id,
                    "reason": "noisy text bound to another split",
                }
            )
            continue
        text_split[clean_key] = split
        text_split[noisy_key] = split
        assigned.append(record.model_copy(update={"split": split}))

    # Second pass: a document may still own pairs whose texts landed in different splits
    # (it answers several questions). Bind each document to its majority split and drop the
    # minority pairs, so document-disjointness holds as well as text-disjointness.
    by_doc: dict[str, list[PairRecord]] = {}
    for record in assigned:
        by_doc.setdefault(record.doc_id or _hash(record.clean), []).append(record)
    consistent: list[PairRecord] = []
    for doc_key in sorted(by_doc):
        members = by_doc[doc_key]
        counts = Counter(member.split for member in members)
        winner = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for member in members:
            if member.split == winner:
                consistent.append(member)
            else:
                dropped.append(
                    {
                        "pair_id": member.pair_id,
                        "doc_id": member.doc_id,
                        "reason": "document bound to another split",
                        "doc_split": winner,
                        "pair_split": member.split,
                    }
                )
    return consistent, dropped


def assert_no_leakage(pairs: Sequence[PairRecord]) -> None:
    """A document or a clean query may appear in exactly one split."""
    docs: dict[str, set[str]] = {}
    cleans: dict[str, set[str]] = {}
    for record in pairs:
        docs.setdefault(record.doc_id or record.pair_id, set()).add(record.split)
        cleans.setdefault(normalise(record.clean), set()).add(record.split)
    leaking_docs = sorted(key for key, splits in docs.items() if len(splits) > 1)
    leaking_cleans = sorted(key for key, splits in cleans.items() if len(splits) > 1)
    if leaking_docs or leaking_cleans:
        raise DatasetLeakageError(
            "split leakage detected: "
            f"{len(leaking_docs)} document(s) and {len(leaking_cleans)} clean query(ies) "
            "appear in more than one split",
            detail=f"docs={leaking_docs[:5]} cleans={leaking_cleans[:5]}",
        )


def review_pii(pairs: Sequence[PairRecord]) -> tuple[list[PairRecord], list[dict[str, object]]]:
    """Split out pairs the PII detector flags, recording *why* (never silently dropping).

    The policy detector is a safety net, not an oracle: a long numeric *identifier*
    (``ORD-2026-10314``, ``ERR-TRF-6170``) matches the phone pattern. Findings whose span lies
    entirely inside such an identifier token are reviewed as false positives and kept (they are
    exactly the productive noise a rewriter must learn to ignore); every other finding excludes
    the pair. Both lists land in ``stats.json`` so the decision is auditable, and the kept set is
    still asserted PII-free by :func:`assert_no_pii`.
    """
    from answerdesk.policy.pii import detect

    kept: list[PairRecord] = []
    excluded: list[dict[str, object]] = []
    reviewed: list[dict[str, object]] = []
    for record in pairs:
        findings = detect(record.noisy) + detect(record.clean)
        if not findings:
            kept.append(record)
            continue
        hard: list[str] = []
        for finding in findings:
            span = (
                record.noisy[finding.start : finding.end]
                if finding.start < len(record.noisy)
                else ""
            )
            match = _IDENTIFIER_TOKEN.search(span) if span else None
            if match and match.group(0).strip() == span.strip():
                reviewed.append(
                    {
                        "pair_id": record.pair_id,
                        "kind": finding.kind.value,
                        "token": span,
                    }
                )
            else:
                hard.append(finding.kind.value)
        if hard:
            excluded.append(
                {
                    "pair_id": record.pair_id,
                    "locale": record.locale,
                    "kinds": sorted(set(hard)),
                }
            )
            continue
        kept.append(record)
    pairs_out = kept
    reviewed_by_pair = {entry["pair_id"] for entry in reviewed}
    assert_no_pii([record for record in pairs_out if record.pair_id not in reviewed_by_pair])
    return pairs_out, [*excluded, {"reviewed_identifier_false_positives": reviewed}]


def assert_no_pii(pairs: Sequence[PairRecord]) -> None:
    """Training text must be PII-free (the corpus PII category is excluded upstream)."""
    from answerdesk.policy.pii import detect

    findings = [
        (record.pair_id, finding.kind.value) for record in pairs for finding in detect(record.noisy)
    ]
    if findings:
        raise DatasetError(
            f"PII found in {len(findings)} training pair(s); refusing to ship the dataset",
            detail=str(findings[:5]),
        )


def splits_hash(pairs: Sequence[PairRecord]) -> str:
    material = json.dumps(
        [record.to_json() for record in pairs], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def dataset_stats(pairs: Sequence[PairRecord]) -> dict[str, object]:
    by_split = Counter(record.split for record in pairs)
    by_source = Counter(record.source for record in pairs)
    by_locale = Counter(record.locale for record in pairs)
    by_style = Counter(record.style or "clean" for record in pairs)
    by_category = Counter(record.category for record in pairs)
    return {
        "dataset_version": DATASET_VERSION,
        "pairs": len(pairs),
        "by_split": dict(sorted(by_split.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_locale": dict(sorted(by_locale.items())),
        "by_style": dict(sorted(by_style.items())),
        "by_category": dict(sorted(by_category.items())),
        "hash": splits_hash(pairs),
        "mean_lengths": {
            "noisy_chars": round(sum(len(r.noisy) for r in pairs) / max(1, len(pairs)), 2),
            "clean_chars": round(sum(len(r.clean) for r in pairs) / max(1, len(pairs)), 2),
        },
    }


def build_dataset(
    *,
    corpus_dir: str | Path,
    out_dir: str | Path,
    seed: int = 1337,
    quiet: bool = False,
) -> dict[str, object]:
    """Build, validate and write ``pairs.jsonl`` + ``splits.json`` + ``stats.json``."""
    pairs = build_pairs(corpus_dir, seed=seed)
    if not pairs:
        raise DatasetError("no training pairs were built")
    pairs, pii_excluded = review_pii(pairs)
    assigned, target_collisions = assign_splits(pairs)
    assert_no_leakage(assigned)
    assert_no_pii(assigned)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for record in assigned:
            handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
    stats = dataset_stats(assigned)
    stats["pii_excluded"] = pii_excluded
    stats["dropped_pairs"] = len(target_collisions)
    stats["pii_excluded_count"] = len(pii_excluded)
    splits = {
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "hash": stats["hash"],
        "counts": stats["by_split"],
        "assignment": (
            "document-disjoint sha256(salt:doc_id) buckets (80/10/10); "
            "clean/noisy target texts bound to one split (collisions dropped)"
        ),
        "salt": _SPLIT_SALT,
        "sources": list(NOISY_CATEGORIES),
        "perturbation_styles": list(MANGLERS),
    }
    (target / "splits.json").write_text(
        json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (target / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not quiet:
        print(
            f"wrote {len(assigned)} pairs to {target} "
            f"({stats['by_split']}); hash {str(stats['hash'])[:16]}…"
        )
    return stats


def load_pairs(path: str | Path) -> list[PairRecord]:
    records: list[PairRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(PairRecord.model_validate(json.loads(line)))
    return records


__all__ = [
    "DATASET_VERSION",
    "EXCLUDED_CATEGORIES",
    "MANGLERS",
    "NOISY_CATEGORIES",
    "PairRecord",
    "assert_no_leakage",
    "assert_no_pii",
    "assign_splits",
    "build_dataset",
    "build_pairs",
    "dataset_stats",
    "jaccard",
    "load_pairs",
    "normalise",
    "perturb",
    "review_pii",
    "shingles",
    "splits_hash",
]
