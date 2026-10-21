"""Deterministic surface-form variants used by the invariance check.

The same intent is asked five ways: clean, telegraphic (function words dropped), typo (one
character swapped in the longest content word), transliteration (Latin script) and code-switch
(a second locale's term for the same archetype injected into the sentence). Everything is
deterministic and offline so the invariance measurement is reproducible in CI; the corpus's
own noisy items remain the primary evidence for the *retrieval* arm (t23) and are not reused
here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

SCRIPTS = ("clean", "telegraphic", "typo", "translit", "code_switch")
#: function words per locale that a telegraphic user typically drops
FUNCTION_WORDS: dict[str, tuple[str, ...]] = {
    "ru": ("как", "ли", "что", "для", "или", "при", "это", "мне", "можно", "нужно"),
    "uz": ("qanday", "uchun", "va", "yoki", "bilan", "men", "bu", "kerak", "mumkin"),
    "tg": ("чӣ", "тавр", "барои", "ва", "ё", "бо", "ман", "ин", "лозим", "мумкин"),
    "ky": ("кантип", "үчүн", "жана", "же", "менен", "мен", "бул", "керек", "болот"),
    "zh": ("如何", "怎么", "的", "和", "或者", "我", "这个", "可以", "需要"),
    "tr": ("nasıl", "için", "ve", "veya", "ile", "ben", "bu", "gerekir", "mümkün"),
}
TRANS_LITERAL: dict[str, dict[str, str]] = {
    "ru": {
        "кредит": "kredit",
        "займ": "zaym",
        "перевод": "perevod",
        "деньги": "dengi",
        "тариф": "tarif",
        "счёт": "schet",
        "карта": "karta",
        "вклад": "vklad",
    },
    "uz": {"kredit": "кредит", "tarif": "тариф", "pul": "пул", "hisob": "ҳисоб"},
    "tg": {"қарз": "qarz", "пул": "pul", "тарҳ": "tarh", "ҳисоб": "hisob"},
    "ky": {"кредит": "kredit", "акча": "akcha", "эсеп": "esep", "тариф": "tarif"},
    "zh": {"贷款": "daikuan", "转账": "zhuanzhang", "费用": "feiyong", "账户": "zhanghu"},
    "tr": {"kredi": "kredi", "transfer": "transfer", "ücret": "ucret", "hesap": "hesap"},
}
_CODE_SWITCH_TERMS: dict[str, tuple[str, ...]] = {
    "ru": ("займ", "перевод", "тариф", "счёт"),
    "uz": ("kredit", "o‘tkazma", "tarif", "hisob"),
    "tg": ("қарз", "интиқол", "тариф", "ҳисоб"),
    "ky": ("кредит", "которуу", "тариф", "эсеп"),
    "zh": ("贷款", "转账", "费用", "账户"),
    "tr": ("kredi", "transfer", "ücret", "hesap"),
}


@dataclass(frozen=True)
class Variant:
    kind: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "text": self.text}


def _words(locale: str, text: str) -> list[str]:
    if locale == "zh":
        return list(text)
    return re.findall(r"[\w’'\-]+|[^\w\s]", text)


def telegraphic(locale: str, text: str) -> str:
    drops = {word.lower() for word in FUNCTION_WORDS.get(locale, ())}
    if locale == "zh":
        stripped = text
        for word in sorted(drops, key=len, reverse=True):
            stripped = stripped.replace(word, "")
        return re.sub(r"\s+", " ", stripped).strip()
    kept = [word for word in text.split() if word.lower().strip(".,?!;:") not in drops]
    return " ".join(kept) if len(kept) >= 2 else text


def typo(text: str, *, seed: int = 0) -> str:
    words = text.split()
    candidates = [index for index, word in enumerate(words) if len(word) >= 5]
    if not candidates:
        return text
    index = candidates[
        int(hashlib.sha256(f"{seed}:{text}".encode()).hexdigest(), 16) % len(candidates)
    ]
    word = words[index]
    position = len(word) // 2
    swapped = (
        word[:position]
        + word[position + 1 : position + 2]
        + word[position : position + 1]
        + word[position + 2 :]
    )
    words[index] = swapped
    return " ".join(words)


def translit(locale: str, text: str) -> str:
    mapping = TRANS_LITERAL.get(locale, {})
    result = text
    for source, target in mapping.items():
        result = re.sub(re.escape(source), target, result, flags=re.IGNORECASE)
    if result == text:
        # no mapped term in this question: romanise the whole sentence deterministically
        table = {
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
            "қ": "q",
            "ғ": "g",
            "ҳ": "h",
            "ӣ": "i",
            "ҷ": "j",
            "ӯ": "u",
            "ң": "ng",
            "ө": "o",
            "ү": "u",
            "ə": "a",
            "ı": "i",
            "ö": "o",
            "ü": "u",
            "ç": "ch",
            "ş": "sh",
        }
        result = "".join(table.get(character, character) for character in text.lower())
    return result


def code_switch(locale: str, text: str, *, other: str = "ru") -> str:
    """Inject the other locale's term for the same archetype into the sentence."""
    mine = _CODE_SWITCH_TERMS.get(locale, ())
    theirs = _CODE_SWITCH_TERMS.get(other, ())
    if not mine or not theirs:
        return text
    used = next((term for term in mine if term.lower() in text.lower()), mine[0])
    replacement = theirs[0]
    switched = re.sub(re.escape(used), replacement, text, count=1, flags=re.IGNORECASE)
    return switched if switched != text else f"{text} {replacement}"


def variants_for(locale: str, text: str, *, seed: int = 0, other: str = "ru") -> list[Variant]:
    """All five forms of one question, in a fixed order."""
    return [
        Variant("clean", text),
        Variant("telegraphic", telegraphic(locale, text)),
        Variant("typo", typo(text, seed=seed)),
        Variant("translit", translit(locale, text)),
        Variant("code_switch", code_switch(locale, text, other=other)),
    ]


__all__ = [
    "FUNCTION_WORDS",
    "SCRIPTS",
    "Variant",
    "code_switch",
    "telegraphic",
    "translit",
    "typo",
    "variants_for",
]
