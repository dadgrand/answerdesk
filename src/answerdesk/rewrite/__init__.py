"""Optional local query rewriter (t26): seam, backends, dataset builder, evaluation.

Nothing in this package is imported by the rest of answerdesk today except through
:func:`answerdesk.rewrite.backend.build_rewriter`, and that function imports no ML library.
The heavy path (torch/transformers/peft) lives behind `uv sync --extra train` and is only
touched by :mod:`answerdesk.rewrite.local_lora` when a caller actually instantiates it.

Public surface::

    from answerdesk.rewrite import build_rewriter, rewrite_status, load_rewrite_config
    rewriter = build_rewriter()                 # identity unless configs/rewrite enables it
    result = rewriter.rewrite("zaym kak oformit", locale="ru")

    # AT-GEN-13  # AT-GEN-16
"""

from __future__ import annotations

from answerdesk.rewrite.backend import RewriterStatus, build_rewriter, rewrite_status
from answerdesk.rewrite.base import (
    IdentityRewriter,
    QueryRewriter,
    RewriteResult,
    TimedRewriter,
)
from answerdesk.rewrite.config import (
    DEFAULT_CONFIG_PATH,
    RewriteConfig,
    default_config_path,
    load_rewrite_config,
)
from answerdesk.rewrite.deterministic import DeterministicRewriter
from answerdesk.rewrite.errors import (
    DatasetError,
    DatasetLeakageError,
    RewriteConfigError,
    RewriteError,
    RewriterUnavailable,
    RewriteTimeout,
)
from answerdesk.rewrite.evaluate import (
    EvaluationReport,
    evaluate_rewriter,
    paired_bootstrap,
    write_report,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DatasetError",
    "DatasetLeakageError",
    "DeterministicRewriter",
    "EvaluationReport",
    "IdentityRewriter",
    "QueryRewriter",
    "RewriteConfig",
    "RewriteConfigError",
    "RewriteError",
    "RewriteResult",
    "RewriteTimeout",
    "RewriterStatus",
    "RewriterUnavailable",
    "TimedRewriter",
    "build_rewriter",
    "default_config_path",
    "evaluate_rewriter",
    "load_rewrite_config",
    "paired_bootstrap",
    "rewrite_status",
    "write_report",
]
