"""Build tooling for ``answerdesk-bench``.

The package is import-only support for ``scripts/build_bench.py`` and for the offline
tests in ``tests/unit/bench``. It is deliberately kept outside ``src/answerdesk``: the
benchmark build is a repository asset pipeline, not a runtime feature of the library
(see ``docs/benchmark.md`` §1 and ``docs/requirements.md`` FR-EVL-01).

Modules
-------
config      Every constant that influences sampling or the on-disk layout.
normalize   Hashing, text normalisation, script detection and MinHash dedup.
schema      The item contract shared by the builder, the verifier and the tests.
templates   The corpus's localized PII / prompt-injection template bank (copied).
sources     Pinned downloads, the git-ignored cache and the derived Mr. TyDi pool.
build       Deterministic generation of the committed ``data/bench`` tree.
verify      The offline invariant checker behind ``--verify``.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
