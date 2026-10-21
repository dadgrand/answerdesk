"""Risk control v2: calibrated answerability, Learn-then-Test certificate, honest frontier.

Modules: :mod:`features` (decision-time features), :mod:`certificate` (CRC/LTT with a
Bentkus/Hoeffding bound), :mod:`model` (L2 logistic), :mod:`metrics` (AUROC/ECE/AURC/frontier),
:mod:`runner` (split, fit, certify, evaluate, artefacts, ``--check``).
"""

from __future__ import annotations

__all__: list[str] = []
