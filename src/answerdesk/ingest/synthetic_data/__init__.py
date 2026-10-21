"""Locale content banks for the synthetic support corpus (FR-ING-10; t18, t22).

Three content layers are merged per locale so every :class:`LocaleBank` is uniform:

* ``banks_<locale>.py`` — wave 1: the six original intents (tariffs, cards, transfers,
  documents, KYC, complaints);
* ``extra_<locale>.py`` — wave 2: six more archetypes (FAQ, fee table, app guide, fraud
  policy, account closure, security notice) plus the shared sentence pool and the
  cross-locale code-switch pairs;
* ``wave3_<locale>.py`` — wave 3 (t22): consumer loan/credit, deposits/savings and
  remittance tracking, written with the surface forms customers actually type
  (``займ``, ``кредит``, ``вклад``, ``депозит``, ``деньги``, ``перевод``, ``Таджикистан``).

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


def _merge(base: LocaleBank, extra: ModuleType, wave3: ModuleType) -> LocaleBank:
    """Base bank + wave-2 archetypes/pool/code-switch + wave-3 archetypes."""
    intents: dict[str, IntentBank] = {
        **base.intents,
        **extra.EXTRA_INTENTS,
        **wave3.EXTRA_INTENTS,
    }
    return replace(
        base,
        intents=intents,
        generic_sentences=base.generic_sentences + extra.EXTRA_SHARED,
        code_switch_pairs=base.code_switch_pairs + extra.EXTRA_CODE_SWITCH,
    )


#: Every locale the golden set requires (§10.1), with waves 1–3 merged in.
BANKS: dict[str, LocaleBank] = {
    "ru": _merge(_RU, extra_ru, wave3_ru),
    "tg": _merge(_TG, extra_tg, wave3_tg),
    "uz": _merge(_UZ, extra_uz, wave3_uz),
    "ky": _merge(_KY, extra_ky, wave3_ky),
    "tr": _merge(_TR, extra_tr, wave3_tr),
    "zh": _merge(_ZH, extra_zh, wave3_zh),
}

__all__ = ["BANKS"]
