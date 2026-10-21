"""Locale content banks for the synthetic support corpus (FR-ING-10, t18).

One base module per locale holds the original six intents; an ``extra_<locale>`` module
adds six more archetypes (FAQ, fee table, app guide, fraud policy, account closure,
security notice) plus the shared sentence pool and the cross-locale code-switch pairs.
The merge happens here so every locale exposes a uniform :class:`LocaleBank`.

All content is **LLM-assisted, hand-authored in-repo data** and not native-speaker
validated — see ``docs/ingestion.md`` §12. Generation never calls the network; the banks
are the committed artifact.
"""

from __future__ import annotations

from dataclasses import replace
from types import ModuleType

from answerdesk.ingest.synthetic import IntentBank, LocaleBank
from answerdesk.ingest.synthetic_data import (
    extra_ky,
    extra_ru,
    extra_tg,
    extra_tr,
    extra_uz,
    extra_zh,
    wave3_ky,
    wave3_ru,
    wave3_tg,
    wave3_tr,
    wave3_uz,
    wave3_zh,
)
from answerdesk.ingest.synthetic_data.banks_ky import BANK as _KY
from answerdesk.ingest.synthetic_data.banks_ru import BANK as _RU
from answerdesk.ingest.synthetic_data.banks_tg import BANK as _TG
from answerdesk.ingest.synthetic_data.banks_tr import BANK as _TR
from answerdesk.ingest.synthetic_data.banks_uz import BANK as _UZ
from answerdesk.ingest.synthetic_data.banks_zh import BANK as _ZH


def _merge(base: LocaleBank, extra: ModuleType) -> LocaleBank:
    """Base bank + wave-2 archetypes, shared sentences and cross-locale code-switch pairs."""
    extra_intents: dict[str, IntentBank] = extra.EXTRA_INTENTS
    wave3 = getattr(extra, "WAVE3", None)
    if wave3 is not None:
        extra_intents = {**extra_intents, **wave3.EXTRA_INTENTS}
    return replace(
        base,
        intents={**base.intents, **extra_intents},
        generic_sentences=base.generic_sentences + extra.EXTRA_SHARED,
        code_switch_pairs=base.code_switch_pairs + extra.EXTRA_CODE_SWITCH,
    )


#: Every locale the golden set requires (§10.1), with wave-2 depth merged in.
BANKS: dict[str, LocaleBank] = {
    "ru": _merge(_RU, extra_ru),  # wave 2 + wave 3 (loan, deposit, remittance)
    "tg": _merge(_TG, extra_tg),
    "uz": _merge(_UZ, extra_uz),
    "ky": _merge(_KY, extra_ky),
    "tr": _merge(_TR, extra_tr),
    "zh": _merge(_ZH, extra_zh),
}

__all__ = ["BANKS"]
