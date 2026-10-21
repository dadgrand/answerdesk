"""Evaluation harness: golden sets, metrics, judge, statistics, reports.

Owning task: **t7**. Requirements: FR-EVL-01 … FR-EVL-11, NFR-DET-02, NFR-ML-01.

Module map — each module is also the answer to "where is that number computed?":

=========================  ==========================================================
``corpus``                 read-only view of ``data/corpus`` (chunks, queries, ids)
``splits``                 frozen document-level split + leakage assertion (§10.2)
``datasets``               seeded generators, evidence labels, manifests, validation
``arms``                   offline retrieval arms (BM25 / hashed-dense / RRF)
``fixture_provider``       deterministic ``FakeProvider`` that emits schema-valid drafts
``judge``                  rubric judge, position swap, metadata, κ plumbing
``metrics``                recall/nDCG/MRR, groundedness, citation P/R, escalation, ops
``calibration``            ECE (two binnings), Brier, risk–coverage, AURC
``stats``                  Wilson, paired bootstrap, McNemar, Cohen's κ
``runner``                 one golden case → one measured turn, per arm
``report``                 markdown tables, plots, ``latest`` pointer, honesty checks
``cli``                    ``answerdesk eval`` implementation
=========================  ==========================================================

Everything that crosses a module boundary comes from :mod:`answerdesk.contracts`; the
artifacts an evaluation writes are declared in :mod:`answerdesk.eval.models`.

Quick start (offline, no key, no network)::

    answerdesk eval --synthetic --provider fake --report reports/offline-smoke

    # AT-EVL-08
"""

from __future__ import annotations

from answerdesk.eval.calibration import CalibrationReport, summarise
from answerdesk.eval.corpus import Corpus, load_corpus
from answerdesk.eval.datasets import (
    GeneratedGolden,
    GoldenConfig,
    generate_golden,
    load_golden,
    write_golden,
)
from answerdesk.eval.errors import (
    CorpusUnavailableError,
    EvalDataError,
    EvalError,
    MissingLabelError,
    ReportError,
    SplitLeakageError,
    UnknownChunkError,
)
from answerdesk.eval.judge import RubricJudge, judge_metadata
from answerdesk.eval.metrics import (
    answer_cells,
    escalation_cells,
    mrr_at_k,
    ndcg_at_k,
    operational_cells,
    recall_all_at_k,
    recall_any_at_k,
    retrieval_cells,
)
from answerdesk.eval.models import MetricCell, RunHeader, RunResult, TurnRecord
from answerdesk.eval.report import LIMITATIONS, write_report
from answerdesk.eval.runner import EvalRunner, RunnerConfig, build_provider, deterministic_digest
from answerdesk.eval.stats import (
    bootstrap_interval,
    cohen_kappa,
    mcnemar,
    paired_bootstrap,
    wilson_interval,
)

__all__ = [
    "LIMITATIONS",
    "CalibrationReport",
    "Corpus",
    "CorpusUnavailableError",
    "EvalDataError",
    "EvalError",
    "EvalRunner",
    "GeneratedGolden",
    "GoldenConfig",
    "MetricCell",
    "MissingLabelError",
    "ReportError",
    "RubricJudge",
    "RunHeader",
    "RunResult",
    "RunnerConfig",
    "SplitLeakageError",
    "TurnRecord",
    "UnknownChunkError",
    "answer_cells",
    "bootstrap_interval",
    "build_provider",
    "cohen_kappa",
    "deterministic_digest",
    "escalation_cells",
    "generate_golden",
    "judge_metadata",
    "load_corpus",
    "load_golden",
    "mcnemar",
    "mrr_at_k",
    "ndcg_at_k",
    "operational_cells",
    "paired_bootstrap",
    "recall_all_at_k",
    "recall_any_at_k",
    "retrieval_cells",
    "summarise",
    "wilson_interval",
    "write_golden",
    "write_report",
]
