"""Language detection for the six locales, and the "wrong language" notice.

FR-GEN-06 requires the answer language to equal the detected query language, and an
*explicit, visible* notice when the model cannot comply. So there are three jobs here:

* :func:`detect_language` — a deterministic script + cue-word detector for ru/tg/uz/ky/
  tr/zh/en. It is deliberately not a model: a language detector that fails open would
  silently answer in the wrong language, and detecting a locale is cheap.
* :func:`text_language` — the same detector, but allowed to answer ``None`` when the text
  carries too little signal (one word, digits only). A short query is not evidence.
* :func:`language_notice` — the localised sentence appended to the rendered answer.

The detector's accuracy is a release-gate measurement (NFR-ML-01/G-REL); in CI it is
exercised by per-locale fixtures whose expected detection is fixed by construction.

    # AT-GEN-06
"""

from __future__ import annotations

import re

#: Locales the product promises (requirements §10.1: ru, tg, uz, ky, tr, zh + en source).
SUPPORTED_LOCALES: tuple[str, ...] = ("en", "ru", "tg", "uz", "ky", "tr", "zh")

LOCALE_NAMES: dict[str, str] = {
    "en": "English",
    "ru": "Russian",
    "tg": "Tajik",
    "uz": "Uzbek",
    "ky": "Kyrgyz",
    "tr": "Turkish",
    "zh": "Chinese",
}

#: Shown when the model answered in a language other than the query language (FR-GEN-06).
LANGUAGE_NOTICES: dict[str, str] = {
    "en": "Note: this answer was produced in a different language than the question.",
    "ru": "Примечание: ответ сгенерирован не на языке вопроса.",
    "tg": "Эзоҳ: ҷавоб ба забони савол дода нашуд.",
    "uz": "Eslatma: javob savol tilida berilmadi.",
    "ky": "Эскертүү: жооп суроонун тилинде берилген жок.",
    "tr": "Not: yanıt sorunun dilinde üretilmedi.",
    "zh": "注意：回答所用语言与提问语言不同。",
}

_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
_LATIN = re.compile(r"[A-Za-z\u00c0-\u024f]")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_TG_LETTERS = set("ғӣқӯҳҷ")
_KY_LETTERS = set("ңөү")
_UZ_CYRILLIC_LETTERS = set("ўқғҳ")
_TR_LETTERS = set("ğışİ")
_UZ_LATIN_MARKS = ("oʻ", "gʻ", "o'", "g'")

_TG_CUES = {"барои", "мехоҳам", "чӣ", "нест", "кард", "шумо", "аз", "дар", "кумак"}
_KY_CUES = {
    "үчүн",
    "кандай",
    "канча",
    "эмне",
    "барбы",
    "жок",
    "менен",
    "болобу",
    "суроо",
    "убакыт",
    "кайтаруу",
}
_UZ_CYRILLIC_CUES = {"учун", "қандай", "нима", "борми", "йўқ", "билан", "сўров"}
_TR_CUES = {"için", "nasıl", "ne", "kadar", "kart", "ücret", "ücreti", "var", "mı", "mi"}
_UZ_LATIN_CUES = {
    "uchun",
    "qanday",
    "qancha",
    "nima",
    "bormi",
    "yoq",
    "bilan",
    "karta",
    "narxi",
    "qaytarish",
    "vaqt",
    "oladi",
}
_EN_CUES = {"the", "how", "what", "much", "does", "cost", "card", "fee", "is", "my"}

#: Keyboard variants of the Uzbek okina: real queries use all of them.
_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bb": "'", "\u02bc": "'", "`": "'"})


def normalise_apostrophes(text: str) -> str:
    """Unify ``‘ ’ ʻ ʼ ` `` so Uzbek ``to‘lov`` is detectable from any keyboard."""
    return text.translate(_APOSTROPHES)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE))


def detect_language(text: str) -> str:
    """Best-effort locale code for ``text``; always returns a supported locale."""
    detected = text_language(text)
    return detected if detected is not None else "en"


def text_language(text: str) -> str | None:
    """Detected locale, or ``None`` when the text carries no usable signal."""
    if not text or not text.strip():
        return None
    text = normalise_apostrophes(text)
    cjk = len(_CJK.findall(text))
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(_LATIN.findall(text))
    letters = cjk + cyrillic + latin
    if letters == 0:
        return None
    if cjk and cjk / letters >= 0.2:
        return "zh"
    words = _words(text)
    if cyrillic > latin:
        characters = set(text.lower())
        if characters & _TG_LETTERS:
            return "tg"
        if characters & _KY_LETTERS:
            return "ky"
        if words & _UZ_CYRILLIC_CUES:
            return "uz"
        if characters & _UZ_CYRILLIC_LETTERS and words & (_TG_CUES | _UZ_CYRILLIC_CUES):
            return "uz"
        if words & _TG_CUES:
            return "tg"
        if words & _KY_CUES:
            return "ky"
        return "ru"
    if latin > 0:
        lowered = text.lower()
        characters = set(lowered)
        if characters & _TR_LETTERS or words & _TR_CUES:
            return "tr"
        if any(mark in lowered for mark in _UZ_LATIN_MARKS) or words & _UZ_LATIN_CUES:
            return "uz"
        if latin <= 2 and cyrillic == 0 and not (words & _EN_CUES):
            return None
        return "en"
    return None


def language_name(code: str) -> str:
    """English name used in the prompt ("Answer in Russian")."""
    return LOCALE_NAMES.get(code, LOCALE_NAMES["en"])


def language_notice(code: str) -> str:
    """Localised visible notice for a non-complying answer (FR-GEN-06)."""
    return LANGUAGE_NOTICES.get(code, LANGUAGE_NOTICES["en"])


def languages_agree(expected: str, text: str) -> bool:
    """Whether ``text`` is written in ``expected`` (undecidable text counts as agreeing)."""
    observed = text_language(text)
    return observed is None or observed == expected


__all__ = [
    "LANGUAGE_NOTICES",
    "LOCALE_NAMES",
    "SUPPORTED_LOCALES",
    "detect_language",
    "language_name",
    "language_notice",
    "languages_agree",
    "normalise_apostrophes",
    "text_language",
]
