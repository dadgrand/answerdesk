"""Fixtures for the evaluation-harness tests.

Everything here is hermetic: a tiny generated corpus on ``tmp_path``, a tiny golden set built
by the real generator with reduced targets, and settings that point at those directories. No
test touches ``data/corpus`` or ``evals/`` — a frozen artifact changing would otherwise turn a
code test into a data test.

    # AT-EVL-01  # AT-EVL-02  # AT-DET-02
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest
from fixtures import tiny_config, write_tiny_corpus

from answerdesk.config import Settings
from answerdesk.contracts import EvalCase, EvalCategory, Split
from answerdesk.eval.corpus import Corpus, load_corpus
from answerdesk.eval.datasets import GeneratedGolden, generate_golden


@pytest.fixture()
def tiny_corpus(tmp_path: Path) -> Corpus:
    return load_corpus(write_tiny_corpus(tmp_path))


@pytest.fixture()
def tiny_golden(tmp_path: Path, tiny_corpus: Corpus) -> GeneratedGolden:
    return generate_golden(tiny_corpus, tiny_config(), strict=False)


@pytest.fixture()
def tiny_cases(tiny_golden: GeneratedGolden) -> list[EvalCase]:
    return list(tiny_golden.cases)


@pytest.fixture()
def make_turn() -> Callable[..., Any]:
    """Build a :class:`TurnRecord` without going through the pipeline."""
    from answerdesk.contracts import Action, ReasonCode
    from answerdesk.eval.models import TurnRecord

    counter = {"n": 0}

    def _make(**overrides: Any) -> TurnRecord:
        counter["n"] += 1
        index = counter["n"]
        answered = bool(overrides.get("answered", True))
        base: dict[str, Any] = {
            "case_id": f"case-{index:03d}",
            "locale": "ru",
            "category": EvalCategory.DIRECT,
            "arm": "arm-a",
            "answerable": True,
            "must_escalate": False,
            "expected_language": "ru",
            "action": Action.ANSWER if answered else Action.ESCALATE,
            "reason_code": ReasonCode.ANSWERED if answered else ReasonCode.NO_EVIDENCE,
            "answered": answered,
            "escalated": not answered,
            "ranked_chunk_ids": ["c1", "c2", "c3"],
            "gold_chunk_ids": ["c1"],
            "language": "ru",
            "answer_ok": answered,
        }
        if answered:
            base.setdefault("claims_total", 1)
            base.setdefault("claims_supported", 1)
            base.setdefault("groundedness", 1.0)
            base.setdefault("citation_precision", 1.0)
            base.setdefault("citation_recall", 1.0)
            base.setdefault("rubric_score", 5.0)
            base.setdefault("rubric_pass", True)
            base.setdefault("confidence", 0.9)
        base.update(overrides)
        return TurnRecord.model_validate(base)

    return _make


@pytest.fixture()
def settings_for(tmp_path: Path) -> Callable[..., Settings]:
    """Settings whose corpus/evals/reports roots live under ``tmp_path``."""

    def _make(**overrides: Any) -> Settings:
        settings = Settings()
        settings.app.corpus_dir = str(overrides.get("corpus_dir", tmp_path / "corpus"))
        settings.app.reports_dir = str(overrides.get("reports_dir", tmp_path / "reports"))
        settings.eval.split = str(overrides.get("split", tmp_path / "evals" / "splits" / "v1.json"))
        settings.eval.seed = int(overrides.get("seed", 424242))
        settings.eval.bootstrap_samples = int(overrides.get("bootstrap", 200))
        return settings

    return _make


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_case_ids(cases: Iterable[EvalCase], split: Split) -> list[str]:
    return sorted(case.case_id for case in cases if case.split is split)
