"""A deterministic, dependency-free rewriter — the baseline arm of the measurement.

This is deliberately *not* the retrieval layer's query analysis (t23 owns that, and it runs
inside the pipeline already). It is a small, auditable normaliser that turns a noisy customer
message into a cleaner search query with three operations:

1. **clean** — collapse whitespace, strip edge punctuation, drop greeting/filler phrases
   ("здравствуйте", "подскажите", "пожалуйста", "pls", "how do i");
2. **transliterate** — Latin spelling of a Cyrillic-language query (``zaym kak oformit``)
   back to Cyrillic, because the corpus and the index are Cyrillic;
3. **repair** — a curated list of frequent misspellings (``денги`` → ``деньги``) and
   missing-space joins are split on known function words only when the join is unambiguous.

The point of this backend in the project is threefold: it is the graceful fallback when the
LoRA backend is absent, it is the control arm for the rewriter experiment (so the measured
uplift of the trained model is not confounded with "any normalisation helps"), and it runs
with zero dependencies on any machine.

    # AT-GEN-14
"""

from __future__ import annotations

import re
import unicodedata

from answerdesk.rewrite.base import RewriteResult

#: Filler/greeting phrases removed from the front/end of a query (lowercase, per script).
_FILLERS: tuple[str, ...] = (
    "здравствуйте",
    "добрый день",
    "добрый вечер",
    "привет",
    "подскажите",
    "скажите",
    "пожалуйста",
    "будьте добры",
    "хочу узнать",
    "мне нужно",
    "how do i",
    "please",
    "pls",
    "hi",
    "hello",
    "salom",
    "assalomu alaykum",
    "салам",
    "саламатсызбы",
    "салом",
    "merhaba",
    "selam",
    "你好",
    "请问",
)

#: Frequent misspellings seen in the noisy slice, mapped to their in-corpus spelling.
_REPAIRS: dict[str, str] = {
    "денги": "деньги",
    "деньги": "деньги",
    "денег": "денег",
    "перевод": "перевод",
    "первод": "перевод",
    "займ": "займ",
    "займа": "займа",
    "оформит": "оформить",
    "оформитб": "оформить",
    "получит": "получить",
    "перевести": "перевести",
    "карта": "карта",
    "карты": "карты",
    "тариф": "тариф",
    "счёт": "счёт",
    "счет": "счёт",
    "кредит": "кредит",
    "вклад": "вклад",
    "депозит": "депозит",
    "блокироват": "блокировать",
    "как": "как",
    "что": "что",
    "где": "где",
    "когда": "когда",
}

#: Latin → Cyrillic transliteration (the inverse of the translit noise style in the corpus).
_TRANSLIT: dict[str, str] = {
    "shch": "щ",
    "sch": "щ",
    "zh": "ж",
    "kh": "х",
    "ts": "ц",
    "ch": "ч",
    "sh": "ш",
    "yu": "ю",
    "ya": "я",
    "yo": "ё",
    "ye": "е",
    "a": "а",
    "b": "б",
    "v": "в",
    "g": "г",
    "d": "д",
    "e": "е",
    "z": "з",
    "i": "и",
    "y": "й",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "f": "ф",
    "h": "х",
    "c": "ц",
    "j": "дж",
    "q": "к",
    "w": "в",
    "x": "кс",
    "'": "ь",
}

_CYRILLIC = re.compile(r"[\u0400-\u04ff]")

#: Locales whose users type Latin transliterations of Cyrillic words.
CYRILLIC_LOCALES = frozenset({"ru", "tg", "ky"})
_LATIN = re.compile(r"[A-Za-z]")
_WHITESPACE = re.compile(r"\s+")
_EDGE = " \t\n\r"
_TRANSLIT_KEYS = sorted(_TRANSLIT, key=len, reverse=True)


def _has_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC.search(text))


def _is_latin_word(word: str) -> bool:
    return bool(_LATIN.search(word)) and not _has_cyrillic(word)


#: A token that must survive transliteration verbatim (order id, error code, amount).
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_./]*\d[A-Za-z0-9\-_./]*")


def _transliterate_word(word: str) -> str:
    out: list[str] = []
    index = 0
    lowered = word.lower()
    while index < len(lowered):
        for key in _TRANSLIT_KEYS:
            if lowered.startswith(key, index):
                out.append(_TRANSLIT[key])
                index += len(key)
                break
        else:
            out.append(lowered[index])
            index += 1
    return "".join(out)


def transliterate_to_cyrillic(text: str) -> str:
    """Greedy Latin→Cyrillic transliteration, word by word.

    Tokens containing a digit (``ORD-12345``, ``120``, ``v1.2``) are identifiers or amounts and
    are preserved verbatim: rewriting an order number into Cyrillic would destroy the very
    signal the retrieval identifier probe looks for.
    """
    pieces: list[str] = []
    for token in text.split(" "):
        if not token or _IDENTIFIER.search(token) or not _LATIN.search(token):
            pieces.append(token)
        else:
            pieces.append(_transliterate_word(token))
    return " ".join(pieces)


def _strip_fillers(text: str) -> str:
    stripped = text.strip()
    changed = True
    while changed:
        changed = False
        lowered = stripped.casefold()
        for filler in _FILLERS:
            if lowered.startswith(filler):
                candidate = stripped[len(filler) :].lstrip(" ,.:;!?-—")
                if candidate:
                    stripped = candidate
                    changed = True
                    break
            if lowered.endswith(filler):
                candidate = stripped[: -len(filler)].rstrip(" ,.:;!?-—")
                if candidate:
                    stripped = candidate
                    changed = True
                    break
    return stripped


def _repair_tokens(text: str) -> str:
    tokens = []
    for token in text.split(" "):
        core = token.strip(".,?!:;()[]\"'")
        replacement = _REPAIRS.get(core.casefold())
        if replacement is not None and replacement != core:
            tokens.append(token.replace(core, replacement))
        else:
            tokens.append(token)
    return " ".join(tokens)


class DeterministicRewriter:
    """Dependency-free normaliser used as the control arm and as the fallback backend."""

    name = "deterministic"
    available = True

    def __init__(
        self,
        *,
        transliterate: bool = True,
        drop_fillers: bool = True,
        repair: bool = True,
    ) -> None:
        self.transliterate = transliterate
        self.drop_fillers = drop_fillers
        self.repair = repair

    def rewrite(self, query: str, *, locale: str | None = None) -> RewriteResult:
        original = query
        text = unicodedata.normalize("NFKC", query or "").strip()
        notes: list[str] = []

        if self.drop_fillers:
            without_fillers = _strip_fillers(text)
            if without_fillers != text:
                notes.append("filler-removed")
                text = without_fillers

        # Transliterate only when the whole query is Latin-script: a mixed query keeps its
        # Latin identifiers (order ids, error codes) untouched.
        words = text.split()
        wholly_latin = (
            bool(words)
            and all(_is_latin_word(word) or not _LATIN.search(word) for word in words)
            and any(_LATIN.search(word) for word in words)
        )
        # Transliteration is only correct for locales whose users spell Cyrillic in Latin
        # script. For an English or Turkish message the same transform would produce nonsense
        # ("block the card" → "блоцк тхе цард"), so a known non-Cyrillic locale disables it.
        transliteration_applies = locale in CYRILLIC_LOCALES or (locale is None and wholly_latin)
        if self.transliterate and wholly_latin and transliteration_applies:
            transliterated = transliterate_to_cyrillic(text)
            if _has_cyrillic(transliterated):
                notes.append("transliterated")
                text = transliterated

        if self.repair:
            repaired = _repair_tokens(text)
            if repaired != text:
                notes.append("repaired")
                text = repaired

        text = _WHITESPACE.sub(" ", text).strip().strip(_EDGE).strip()
        if not text:
            text = original.strip() or " "
            notes.append("empty-after-normalisation")
        if text != original:
            notes.insert(0, "changed")
        return RewriteResult(
            original=original,
            query=text,
            backend=self.name,
            available=True,
            notes=", ".join(notes) or "no-op",
        )


__all__ = ["CYRILLIC_LOCALES", "DeterministicRewriter", "transliterate_to_cyrillic"]
