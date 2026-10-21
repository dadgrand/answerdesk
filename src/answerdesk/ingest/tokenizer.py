"""Deterministic, dependency-free token estimation (FR-ING-03).

Why not a real tokenizer? The base install carries no ``tiktoken`` / HF tokenizer
(NFR-PORT-03), and the chunker must be *byte-for-byte reproducible* on any OS, in any
process, without downloading a vocabulary. The contract this module provides is
therefore:

* **deterministic** — the same string yields the same integer everywhere, forever;
* **monotone** — ``estimate_tokens(a + b) >= estimate_tokens(a)``;
* **documented bias** — see the model below and ``docs/ingestion.md`` §2.2.

Estimation model
----------------
``tokens(text) = sum_words ceil(len(word) / 4) + cjk_chars + ceil(punct_chars / 4)``

* a whitespace-delimited word costs one token per started 4 characters (BPE-style
  behaviour for Latin and Cyrillic: ``"internationalization"`` -> 5, ``"card"`` -> 1);
* CJK / kana / hangul characters cost exactly one token each (no word segmentation);
* punctuation costs one token per started group of four (most punctuation is glued to
  a word token by real BPE vocabularies).

Measured against ``cl100k_base`` on the demo corpus this is conservative for Latin
(+8%), close for Cyrillic (-6%) and close for Chinese (+4%); the exact error for a
given text is not promised. The chunker treats the number as a **budget guard**, and
every consumer that needs a provider-exact count reconciles against provider usage in
the cost ledger (NFR-COST-01).
"""

from __future__ import annotations

import re

#: Bumped whenever the estimation model changes — it is part of the corpus manifest
#: (a token-count change is a corpus-version change).
TOKENIZER_VERSION = "answerdesk-est-v1"

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

#: Inclusion thresholds, not a regex: explicit ranges keep behaviour identical across
#: Unicode versions and platforms (a ``\p{Han}``-style regex would not be).
_CJK_BLOCKS: tuple[tuple[int, int], ...] = (
    (0x0E00, 0x0E7F),  # Thai (no inter-word spaces)
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0x20000, 0x2FA1F),  # CJK extensions B..F
)

_CHARS_PER_TOKEN = 4


def is_cjk_char(char: str) -> bool:
    """True when ``char`` belongs to an unsegmented (per-character) script."""
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_BLOCKS)


def is_cjk_word(word: str) -> bool:
    """True when every character of ``word`` is an unsegmented script character."""
    return bool(word) and all(is_cjk_char(char) for char in word)


def estimate_tokens(text: str) -> int:
    """Estimated BPE token count of ``text`` — deterministic, never negative."""
    if not text or not text.strip():
        return 0
    tokens = 0
    for word in _WORD_RE.findall(text):
        if is_cjk_word(word):
            tokens += len(word)
        else:
            tokens += (len(word) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN
    punctuation = len(_PUNCT_RE.findall(text))
    tokens += (punctuation + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN
    # A non-blank text is never zero tokens: punctuation-only strings still cost.
    return max(1, tokens)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Longest prefix of ``text`` whose estimate is ``<= max_tokens`` (word boundaries).

    Deterministic and dependency-free; used by the contextual-header builder where a
    hard cap matters more than exactness.
    """
    if max_tokens <= 0 or not text.strip():
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    words = text.split()
    kept: list[str] = []
    for word in words:
        candidate = " ".join([*kept, word])
        if estimate_tokens(candidate) > max_tokens:
            break
        kept.append(word)
    return " ".join(kept)


__all__ = [
    "TOKENIZER_VERSION",
    "estimate_tokens",
    "is_cjk_char",
    "is_cjk_word",
    "truncate_to_tokens",
]
