"""Function words per shipped locale: one source of truth for cues, filtering and clauses.

Three consumers, one table, on purpose:

* **language identification** (:mod:`answerdesk.retrieval.query_analysis`) scores a text by
  how many function words it hits — a cue table that drifts from the filter table would make
  detection and noise handling disagree about what a function word is;
* **noise handling** (:mod:`answerdesk.retrieval.noisy`) needs the same words to build the
  *content-only* probe: telegraphic messages ("деньги как получит") are mostly function words
  and a bag-of-words score is diluted by them;
* **clause segmentation** needs a smaller table again: only the words that can *start* a new
  clause in a run-on message ("… сделал **а** деньги не дошли").

Nothing here is language-model-based. The lists were tuned on the checked-in query set
(96%+ accuracy per locale at 100% overall, see ``docs/retrieval.md`` §5.2) and are auditable
by eye, which is the property that matters for a filter nobody can train.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

#: Function words (articles, pronouns, prepositions, particles, copulas) per locale.
FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "ru": frozenset(
        [
            "что",
            "как",
            "для",
            "это",
            "или",
            "при",
            "после",
            "можно",
            "нужно",
            "если",
            "почему",
            "сколько",
            "какой",
            "какая",
            "какое",
            "где",
            "когда",
            "мне",
            "мой",
            "моя",
            "ваш",
            "ваша",
            "не",
            "на",
            "по",
            "с",
            "со",
            "из",
            "до",
            "за",
            "от",
            "у",
            "же",
            "ли",
            "бы",
            "есть",
            "нет",
            "будет",
            "стоит",
            "и",
            "в",
            "а",
            "но",
            "то",
            "уже",
            "тоже",
            "ещё",
            "еще",
        ]
    ),
    "tg": frozenset(
        [
            "ва",
            "барои",
            "чӣ",
            "чи",
            "кадом",
            "мебошад",
            "карда",
            "шуда",
            "бо",
            "аз",
            "дар",
            "куҷо",
            "чаро",
            "ман",
            "шумо",
            "оё",
            "ҳаст",
            "нест",
            "кунам",
            "метавонам",
            "дорад",
            "мешавад",
            "намешавад",
            "диҳад",
            "медиҳад",
            "аст",
            "чанд",
            "бояд",
            "шудааст",
            "кунад",
            "кардан",
            "буда",
            "ҳама",
            "дубора",
            "бораи",
            "бар",
        ]
    ),
    "ky": frozenset(
        [
            "жана",
            "менен",
            "үчүн",
            "канча",
            "болобу",
            "жок",
            "бар",
            "экен",
            "кантип",
            "ким",
            "эмне",
            "мен",
            "сиз",
            "ошондой",
            "болот",
            "болбойт",
            "керек",
            "турат",
            "алат",
            "чейин",
            "кийин",
            "аркылуу",
            "эмес",
            "кандай",
            "качан",
            "кайсы",
            "балким",
            "боюнча",
            "жатат",
            "эмнени",
            "билдирет",
            "катасы",
            "алам",
            "алабыз",
        ]
    ),
    "tr": frozenset(
        [
            "için",
            "nasıl",
            "neden",
            "niçin",
            "hangi",
            "bir",
            "ve",
            "ile",
            "bu",
            "ne",
            "mi",
            "mı",
            "mu",
            "mü",
            "yok",
            "var",
            "kadar",
            "daha",
            "olarak",
            "olur",
            "değil",
        ]
    ),
    "uz": frozenset(
        [
            "uchun",
            "qanday",
            "nima",
            "kerak",
            "bormi",
            "va",
            "bilan",
            "bu",
            "men",
            "siz",
            "yo'q",
            "bor",
            "mumkin",
            "qachon",
            "qayerda",
            "bo'ladi",
            "bo'lsa",
            "qancha",
            "yoki",
            "lar",
            "dan",
            "ga",
            "ni",
            "da",
            "xatosi",
            "nimani",
            "bildiradi",
            "turadi",
        ]
    ),
    "en": frozenset(
        [
            "the",
            "what",
            "how",
            "why",
            "when",
            "where",
            "which",
            "is",
            "are",
            "was",
            "can",
            "do",
            "does",
            "did",
            "my",
            "your",
            "our",
            "for",
            "with",
            "and",
            "to",
            "of",
            "in",
            "on",
            "at",
            "it",
            "this",
            "that",
        ]
    ),
}

#: Words that may *begin* a new clause in a run-on message. Deliberately smaller than
#: :data:`FUNCTION_WORDS`: splitting on every function word would shred a normal sentence.
CLAUSE_WORDS: dict[str, frozenset[str]] = {
    "ru": frozenset(
        [
            "а",
            "и",
            "но",
            "или",
            "потом",
            "затем",
            "поэтому",
            "если",
            "когда",
            "что",
            "чтобы",
            "тоже",
            "также",
        ]
    ),
    "tg": frozenset(["ва", "аммо", "ё", "ё", "ки", "чунки", "баъд"]),
    "ky": frozenset(["жана", "бирок", "же", "анткени", "кийин"]),
    "tr": frozenset(["ve", "ama", "veya", "sonra", "çünkü"]),
    "uz": frozenset(["va", "lekin", "yoki", "keyin", "chunki"]),
    "en": frozenset(["and", "but", "or", "then", "so", "if", "when", "because"]),
}

#: Locales written in Cyrillic — used by transliteration detection, not by scoring.
CYRILLIC_LOCALES: frozenset[str] = frozenset({"ru", "tg", "ky"})


def function_words(locale: str) -> frozenset[str]:
    """The function-word set for ``locale``; empty for an unknown locale."""
    return FUNCTION_WORDS.get(locale, frozenset())


def clause_words(locale: str) -> frozenset[str]:
    """The clause-boundary word set for ``locale``; empty for an unknown locale."""
    return CLAUSE_WORDS.get(locale, frozenset())


def all_function_words() -> frozenset[str]:
    """Union over every locale — the fallback when language identification says ``und``."""
    words: set[str] = set()
    for values in FUNCTION_WORDS.values():
        words |= values
    return frozenset(words)


def content_terms(
    terms: Iterable[str],
    *,
    locale: str = "und",
    min_chars: int = 2,
) -> list[str]:
    """Terms with function words removed, order preserved, input never mutated.

    This is a *view* used for an additional retrieval probe — it does not delete anything
    from the query itself (the additive-expansion invariant, ``FR-RET-06``): the raw query
    stays its own arm, and the filtered probe competes with it.
    """
    stop = function_words(locale) if locale != "und" else all_function_words()
    kept: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if len(term) < min_chars or term in stop or term in seen:
            continue
        seen.add(term)
        kept.append(term)
    return kept


def content_share(terms: Sequence[str], *, locale: str = "und") -> float:
    """Fraction of ``terms`` that are content words (0.0 for an empty sequence)."""
    if not terms:
        return 0.0
    return len(content_terms(terms, locale=locale)) / len(terms)


__all__ = [
    "CLAUSE_WORDS",
    "CYRILLIC_LOCALES",
    "FUNCTION_WORDS",
    "all_function_words",
    "clause_words",
    "content_share",
    "content_terms",
    "function_words",
]
