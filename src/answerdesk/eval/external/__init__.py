"""External validity: public benchmarks, reported with intervals and honest labels.

The internal frozen split is the primary evidence (``docs/results.md``); this package adds the
*external* arm the requirements ask for and keeps it strictly separate:

* :mod:`answerdesk.eval.external.datasets` — licence registry, cache and integrity checks;
* :mod:`answerdesk.eval.external.bench` — typed reader for the committed curated benchmark;
* :mod:`answerdesk.eval.external.retrieval` — BM25, dense and hybrid arms with paired CIs;
* :mod:`answerdesk.eval.external.qa` — judge-free multiple-choice closed-book vs RAG;
* :mod:`answerdesk.eval.external.calibration` — human-label template and Cohen's κ;
* :mod:`answerdesk.eval.external.runner` — artefacts under ``reports/external-<run>/``.
"""

from __future__ import annotations

from answerdesk.eval.external.datasets import ExternalDataError

__all__ = ["ExternalDataError"]
