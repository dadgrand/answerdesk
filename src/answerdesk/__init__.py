"""answerdesk — local-first, multilingual customer-support RAG+LLM reference stack.

This package is the implementation of the frozen specification in
``docs/requirements.md`` (document ``AD-REQ-v1.0``). Every public type that crosses a
module boundary lives in :mod:`answerdesk.contracts`; every runtime knob lives in
:mod:`answerdesk.config`.

The skeleton (this task) ships contracts, configuration, logging, the CLI surface and
deterministic test doubles only. Module internals are implemented by their owning tasks:

============================  ==================================
module                        owning task
============================  ==================================
``answerdesk.ingest``         t4
``answerdesk.index``          t5
``answerdesk.retrieval``      t5
``answerdesk.rerank``         t5
``answerdesk.llm``            t6
``answerdesk.generate``       t6
``answerdesk.policy``         t8
``answerdesk.cache``          t8
``answerdesk.service``        t9
``answerdesk.telemetry``      t9
``answerdesk.eval``           t7
============================  ==================================
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from answerdesk.contracts import SCHEMA_VERSION

__all__ = ["PACKAGE_VERSION", "SCHEMA_VERSION", "__version__"]

try:  # pragma: no cover - only hit in an uninstalled source tree
    _VERSION = _dist_version("answerdesk")
except PackageNotFoundError:  # pragma: no cover
    _VERSION = "0.0.0+unknown"

PACKAGE_VERSION: str = _VERSION
__version__: str = _VERSION
