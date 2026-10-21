"""Deterministic text utilities: hashing, normalisation, script detection, MinHash.

Everything here is pure Python (``hashlib``/``zlib``) and order-independent, so the
builder produces identical bytes on every machine and the verifier can recompute the same
numbers from the committed files (``docs/benchmark.md`` §5.4).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zlib
from collections.abc import Sequence
from typing import Any, Final

_WORD_RE: Final = re.compile(r"[^\W_]+", re.UNICODE)

_SCRIPTS: Final = ("Cyrillic", "Latin", "Han")

#: Expected writing system per benchmark language. ``uz`` is Latin in both sources
#: (Belebele ``uzn_Latn`` and the synthetic corpus); ``ru``/``ky``/``tg`` are Cyrillic.
EXPECTED_SCRIPT: Final = {
    "ru": "Cyrillic",
    "ky": "Cyrillic",
    "tg": "Cyrillic",
    "uz": "Latin",
    "zh": "Han",
    "tr": "Latin",
}


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Hex SHA-256 of UTF-8 text."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Any) -> str:
    """Hex SHA-256 of a file, read in chunks so a 1.5 GB archive is fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Canonical JSON: UTF-8 preserving, sorted keys, no insignificant whitespace."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalise_text(text: str) -> str:
    """Unicode NFC, LF line endings, collapsed whitespace, trimmed."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def normalise_question(text: str) -> str:
    """Case-folded, normalised question text — the dedup and leakage key."""
    return normalise_text(text).casefold()


def tokens(text: str) -> list[str]:
    """Unicode word tokens (letters/digits), case-folded."""
    return [match.group(0) for match in _WORD_RE.finditer(normalise_text(text).casefold())]


def content_tokens(text: str) -> list[str]:
    """Tokens that carry meaning for the lexical-anchor diagnostic (length > 2)."""
    return [token for token in tokens(text) if len(token) > 2]


def answer_anchor(answer: str, passage: str) -> float:
    """Fraction of the answer's content tokens that occur in the passage.

    A *diagnostic*, not a gate: Belebele options are paraphrases, so a low value does not
    mean the label is wrong — it means lexical anchoring cannot certify it
    (``docs/benchmark.md`` §6, limitation L3).
    """
    answer_tokens = content_tokens(answer)
    if not answer_tokens:
        return 0.0
    passage_tokens = set(content_tokens(passage))
    hits = sum(1 for token in answer_tokens if token in passage_tokens)
    return round(hits / len(answer_tokens), 4)


def _script_of(char: str) -> str | None:
    code = ord(char)
    if 0x0400 <= code <= 0x052F:  # Cyrillic + supplement
        return "Cyrillic"
    if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
        return "Han"
    if ("a" <= char.lower() <= "z") or 0x00C0 <= code <= 0x024F or 0x1E00 <= code <= 0x1EFF:
        return "Latin"
    return None


def script_profile(text: str) -> dict[str, int]:
    """Count letters per writing system (only Cyrillic, Latin and Han are tracked)."""
    profile = {script: 0 for script in _SCRIPTS}
    for char in text:
        script = _script_of(char)
        if script is not None:
            profile[script] += 1
    return profile


def script_share(text: str, script: str) -> float:
    """Share of letters written in ``script`` (0.0 when the text has no letters)."""
    profile = script_profile(text)
    total = sum(profile.values())
    if total == 0:
        return 0.0
    return round(profile[script] / total, 4)


def expected_script(language: str) -> str:
    """Writing system the language must predominantly use."""
    return EXPECTED_SCRIPT[language]


def script_ok(language: str, text: str, minimum: float = 0.5) -> bool:
    """True when the text is predominantly written in the language's script."""
    return script_share(text, expected_script(language)) >= minimum


def shingle_set(text: str, size: int = 5) -> frozenset[str]:
    """Deterministic near-duplicate fingerprint set.

    Word 5-grams for space-delimited scripts; character 5-grams for Han text, where
    whitespace tokenisation yields one long run.
    """
    normalised = normalise_text(text).casefold()
    if not normalised:
        return frozenset()
    if script_share(normalised, "Han") > 0.3:
        chars = [char for char in normalised if not char.isspace()]
        if len(chars) < size:
            return frozenset(["".join(chars)])
        return frozenset("".join(chars[index : index + size]) for index in range(len(chars) - size + 1))
    words = _WORD_RE.findall(normalised)
    if len(words) < size:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[index : index + size]) for index in range(len(words) - size + 1))


def _hash32(value: str, seed: int) -> int:
    """Fast deterministic 32-bit hash; ``seed`` makes the permutations independent."""
    return zlib.crc32(value.encode("utf-8"), (seed * 0x9E3779B1) & 0xFFFFFFFF)


def minhash_signature(text: str, num_hashes: int = 16, size: int = 5) -> tuple[int, ...]:
    """MinHash signature of a text (``num_hashes`` independent minima)."""
    shingles = shingle_set(text, size=size)
    if not shingles:
        return tuple(0xFFFFFFFF for _ in range(num_hashes))
    return tuple(min(_hash32(shingle, index) for shingle in shingles) for index in range(num_hashes))


def near_duplicate_pairs(
    texts: Sequence[str],
    *,
    threshold: float = 0.9,
    num_hashes: int = 16,
    bands: int = 4,
    size: int = 5,
) -> list[tuple[int, int, float]]:
    """Pairs whose exact Jaccard similarity is at least ``threshold``.

    LSH banding keeps this linear-ish: only items sharing a band become candidates, then
    the true Jaccard is computed on the shingle sets. Returned pairs are sorted by index.
    """
    if num_hashes % bands != 0:
        raise ValueError("num_hashes must be divisible by bands")
    rows = num_hashes // bands
    signatures = [minhash_signature(text, num_hashes=num_hashes, size=size) for text in texts]
    shingles = [shingle_set(text, size=size) for text in texts]

    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for index, signature in enumerate(signatures):
        for band in range(bands):
            key = (band, signature[band * rows : (band + 1) * rows])
            buckets.setdefault(key, []).append(index)

    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                first, second = members[left], members[right]
                candidates.add((first, second) if first < second else (second, first))

    duplicates: list[tuple[int, int, float]] = []
    for first, second in sorted(candidates):
        left, right = shingles[first], shingles[second]
        if not left or not right:
            continue
        union = len(left | right)
        if union == 0:
            continue
        jaccard = len(left & right) / union
        if jaccard >= threshold:
            duplicates.append((first, second, round(jaccard, 4)))
    return duplicates
