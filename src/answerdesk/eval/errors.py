"""Typed failures for the evaluation harness.

An evaluation that silently drops a broken case is worse than no evaluation: every
integrity failure here is loud, named, and carries the offending ids so the operator can
act on it (FR-EVL-03, §10.2).

    # AT-EVL-03
"""

from __future__ import annotations

from collections.abc import Sequence

from answerdesk.errors import AnswerdeskError


class EvalError(AnswerdeskError):
    """Base class for evaluation failures."""

    code = "eval_error"


class EvalDataError(EvalError):
    """The corpus or the golden set is missing, unreadable or malformed."""

    code = "eval_data_error"


class CorpusUnavailableError(EvalDataError):
    """The ingested corpus is not on disk — an actionable error, never an empty run."""

    code = "eval_corpus_missing"


class MissingLabelError(EvalDataError):
    """A golden case is missing a label class the metric needs (FR-EVL-01/03)."""

    code = "eval_missing_label"

    def __init__(self, case_ids: Sequence[str], detail: str) -> None:
        shown = ", ".join(list(case_ids)[:5])
        super().__init__(
            f"{len(list(case_ids))} golden case(s) are missing required labels: {shown}"
            + ("..." if len(list(case_ids)) > 5 else ""),
            detail=detail,
        )
        self.case_ids = list(case_ids)


class SplitLeakageError(EvalDataError):
    """A test-split item derives from a document that also feeds the dev split."""

    code = "eval_split_leakage"

    def __init__(
        self, case_ids: Sequence[str], document_ids: Sequence[str], detail: str = ""
    ) -> None:
        super().__init__(
            f"split leakage: {len(list(case_ids))} test case(s) derive from dev documents "
            f"({', '.join(sorted(set(document_ids))[:5])})",
            detail=detail or "the split unit is the document, not the query (§10.2)",
        )
        self.case_ids = list(case_ids)
        self.document_ids = list(document_ids)


class UnknownChunkError(EvalDataError):
    """A gold/label chunk id does not exist in the corpus (§10.2 loud-failure list)."""

    code = "eval_unknown_chunk"

    def __init__(self, case_ids: Sequence[str], chunk_ids: Sequence[str]) -> None:
        super().__init__(
            f"{len(list(chunk_ids))} gold chunk id(s) are not in the corpus: "
            f"{', '.join(sorted(set(chunk_ids))[:5])}",
            detail=f"cases: {', '.join(list(case_ids)[:5])}",
        )
        self.case_ids = list(case_ids)
        self.chunk_ids = list(chunk_ids)


class ReportError(EvalError):
    """A report cannot be written or would be dishonest (missing n, missing CI)."""

    code = "eval_report_error"


__all__ = [
    "CorpusUnavailableError",
    "EvalDataError",
    "EvalError",
    "MissingLabelError",
    "ReportError",
    "SplitLeakageError",
    "UnknownChunkError",
]
