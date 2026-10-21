"""Consistency audit: cross-language agreement, surface-form invariance, logical coherence.

Modules:

* :mod:`answerdesk.eval.consistency.facts` — the transcreated fact set built from corpus v2.1;
* :mod:`answerdesk.eval.consistency.variants` — deterministic surface-form variants;
* :mod:`answerdesk.eval.consistency.values` — value extraction, fidelity and language ID;
* :mod:`answerdesk.eval.consistency.logical` — within/cross-document contradiction audit;
* :mod:`answerdesk.eval.consistency.agreement` — agreement, overlap and invariance metrics;
* :mod:`answerdesk.eval.consistency.runner` — artefacts under ``reports/consistency-<run>/``.
"""

from __future__ import annotations

__all__: list[str] = []
