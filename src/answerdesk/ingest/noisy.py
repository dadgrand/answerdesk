"""Noisy free-form queries: telegraphic, run-on, dropped prepositions (t22).

Real customers do not type clean FAQ questions. This module holds the **fixed, verbatim**
noisy examples reported by the owner (so the exact reported cases are reproducible) and
the deterministic manglers that turn a clean question into a malformed one without
changing its meaning:

* ``telegraphic`` — content words only, no auxiliaries or punctuation
  (``как займ оформит``);
* ``run_on`` — two clauses glued without punctuation (``денги первод таджикистан сделал
  денги нет брату не дошли``);
* ``dropped_preposition`` — short prepositions removed (``деньги как получит``);
* ``typo_mix`` — the above plus a keystroke typo or transliteration.

Every noisy item keeps an intent label (the archetype its answer lives in) and a grounded
reply: the reply is the FAQ answer of the target document, so the t15 grounding gate covers
noisy queries exactly as it covers clean ones.
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass

#: Noisy style labels reported in the manifest and the query records.
NOISY_STYLES: tuple[str, ...] = (
    "telegraphic",
    "run_on",
    "dropped_preposition",
    "typo_mix",
    "translit",
)


@dataclass(frozen=True)
class NoisyExample:
    """One fixed noisy question from the owner's review, with its answer archetype."""

    locale: str
    text: str
    archetype: str
    style: str
    note: str
    answerable: bool = True


#: The owner's exact Russian examples (verbatim) and natural noisy translations.
OWNER_EXAMPLES: tuple[NoisyExample, ...] = (
    NoisyExample(
        locale="ru",
        text="как займ оформит",
        archetype="loan_credit",
        style="telegraphic",
        note="owner example 1: telegraphic, missing verb ending",
    ),
    NoisyExample(
        locale="ru",
        text="деньги как получит",
        archetype="remittance_tracking",
        style="dropped_preposition",
        note="owner example 2: dropped preposition, no punctuation",
    ),
    NoisyExample(
        locale="ru",
        text="денги первод таджикистан сделал денги нет брату не дошли",
        archetype="remittance_tracking",
        style="run_on",
        note="owner example 3: run-on, typos (денги, первод), no punctuation",
    ),
    NoisyExample(
        locale="ru",
        text="zaym kak oformit",
        archetype="loan_credit",
        style="translit",
        note="transliteration of owner example 1 (Latin keyboard)",
    ),
    NoisyExample(
        locale="ru",
        text="dengi kak poluchit",
        archetype="remittance_tracking",
        style="translit",
        note="transliteration of owner example 2",
    ),
    NoisyExample(
        locale="ru",
        text="perevod tadzhikistan dengi ne doshli",
        archetype="remittance_tracking",
        style="translit",
        note="transliteration of owner example 3",
    ),
    NoisyExample(
        locale="uz",
        text="kredit qanday olinadi",
        archetype="loan_credit",
        style="telegraphic",
        note="translation of owner example 1",
    ),
    NoisyExample(
        locale="uz",
        text="pul qanday olaman",
        archetype="remittance_tracking",
        style="dropped_preposition",
        note="translation of owner example 2",
    ),
    NoisyExample(
        locale="uz",
        text="pul otkazma tojikiston qildim pul yoq akamga yetmadi",
        archetype="remittance_tracking",
        style="run_on",
        note="translation of owner example 3, typos (otkazma, yoq)",
    ),
    NoisyExample(
        locale="tg",
        text="қарз чӣ тавр гирифтан",
        archetype="loan_credit",
        style="telegraphic",
        note="translation of owner example 1",
    ),
    NoisyExample(
        locale="tg",
        text="пул чӣ тавр гирифтан",
        archetype="remittance_tracking",
        style="dropped_preposition",
        note="translation of owner example 2",
    ),
    NoisyExample(
        locale="tg",
        text="пул интиқол тоҷикистон кардам пул нест ба бародарам нарасид",
        archetype="remittance_tracking",
        style="run_on",
        note="translation of owner example 3, no punctuation",
    ),
    NoisyExample(
        locale="ky",
        text="кредит кантип алуу",
        archetype="loan_credit",
        style="telegraphic",
        note="translation of owner example 1",
    ),
    NoisyExample(
        locale="ky",
        text="акча кантип алуу",
        archetype="remittance_tracking",
        style="dropped_preposition",
        note="translation of owner example 2",
    ),
    NoisyExample(
        locale="ky",
        text="акча которуу тажикстан кылдым акча жок агама жетпей калды",
        archetype="remittance_tracking",
        style="run_on",
        note="translation of owner example 3, no punctuation",
    ),
    NoisyExample(
        locale="tr",
        text="kredi nasil alinir",
        archetype="loan_credit",
        style="telegraphic",
        note="translation of owner example 1, ASCII typo",
    ),
    NoisyExample(
        locale="tr",
        text="para nasil alinir",
        archetype="remittance_tracking",
        style="dropped_preposition",
        note="translation of owner example 2",
    ),
    NoisyExample(
        locale="tr",
        text="para transfer tacikistan yaptim para yok kardesime ulasmadi",
        archetype="remittance_tracking",
        style="run_on",
        note="translation of owner example 3, no punctuation",
    ),
    NoisyExample(
        locale="zh",
        text="怎么贷款办理",
        archetype="loan_credit",
        style="telegraphic",
        note="translation of owner example 1",
    ),
    NoisyExample(
        locale="zh",
        text="钱怎么收到",
        archetype="remittance_tracking",
        style="dropped_preposition",
        note="translation of owner example 2",
    ),
    NoisyExample(
        locale="zh",
        text="转账塔吉克斯坦做了钱没有弟弟没收到",
        archetype="remittance_tracking",
        style="run_on",
        note="translation of owner example 3, no punctuation",
    ),
)

#: Short prepositions/particles per locale, removed by the ``dropped_preposition`` mangler.
PREPOSITIONS: dict[str, tuple[str, ...]] = {
    "ru": ("в", "во", "на", "по", "за", "из", "от", "до", "для", "с", "ко", "у"),
    "tg": ("дар", "ба", "аз", "то", "барои", "бо"),
    "uz": ("uchun", "bilan", "dan", "ga", "da", "orqali"),
    "ky": ("үчүн", "менен", "дан", "га", "да", "аркылуу"),
    "tr": ("için", "ile", "dan", "den", "e", "a", "de", "da"),
    "zh": (),
}

#: Filler/auxiliary words a telegraphic writer drops (per locale).
TELEGRAPHIC_DROP: dict[str, tuple[str, ...]] = {
    "ru": ("пожалуйста", "можно", "ли", "мне", "нужно", "как", "что", "будет", "если"),
    "tg": ("лутфан", "мешавад", "чӣ", "тавр", "бояд"),
    "uz": ("iltimos", "mumkin", "qanday", "kerak", "bo'ladi"),
    "ky": ("сураныч", "мүмкүн", "кандай", "керек", "болобу"),
    "tr": ("lütfen", "mümkün", "nasıl", "gerekir", "mi"),
    "zh": ("请问", "怎么", "如何", "可以", "吗"),
}


def strip_punctuation(text: str) -> str:
    """Remove sentence punctuation and collapse whitespace (run-on / telegraphic shape)."""
    cleaned = re.sub(r"[.,!?;:…。！？；：\"'«»()\[\]]", " ", text)
    return " ".join(cleaned.split()).strip()


def make_telegraphic(text: str, locale: str, rng: random.Random) -> str:
    """Content words only: punctuation and filler/auxiliary words removed."""
    words = strip_punctuation(text).split()
    drop = {word.casefold() for word in TELEGRAPHIC_DROP.get(locale, ())}
    kept = [word for word in words if word.casefold() not in drop]
    if len(kept) < 2:
        kept = words
    # a telegraphic message also loses the last word's ending punctuation and may end mid-word
    result = " ".join(kept).casefold() if locale in ("ru", "tg", "ky", "uz", "tr") else "".join(kept)
    if rng.random() < 0.5 and len(result) > 3:
        result = result[:-1] if result[-1].isalpha() else result
    return result


def make_run_on(text: str, locale: str, second: str | None = None) -> str:
    """Glue two clauses without punctuation, as a customer typing in one breath."""
    first = strip_punctuation(text).casefold()
    tail = strip_punctuation(second) if second else ""
    joined = f"{first} {tail}".strip() if tail else first
    if locale in ("zh",):
        joined = joined.replace(" ", "")
    return joined


def drop_prepositions(text: str, locale: str) -> str:
    """Remove short prepositions/particles (``деньги как получит`` style)."""
    dropped = PREPOSITIONS.get(locale, ())
    if not dropped:
        return strip_punctuation(text)
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(word) for word in dropped) + r")(?!\w)",
        re.IGNORECASE,
    )
    return " ".join(pattern.sub(" ", text).split())


def examples_for(locale: str) -> tuple[NoisyExample, ...]:
    """Fixed noisy examples of one locale (owner cases + translations + translit)."""
    return tuple(example for example in OWNER_EXAMPLES if example.locale == locale)


def noisy_styles_for(locale: str) -> tuple[str, ...]:
    """Styles that make sense for a locale (CJK has no short prepositions to drop)."""
    if locale == "zh":
        return ("telegraphic", "run_on", "typo_mix")
    return NOISY_STYLES


__all__ = [
    "NOISY_STYLES",
    "OWNER_EXAMPLES",
    "PREPOSITIONS",
    "NoisyExample",
    "drop_prepositions",
    "examples_for",
    "make_run_on",
    "make_telegraphic",
    "noisy_styles_for",
    "strip_punctuation",
]
