"""The ingestion command surface used by the root CLI (FR-OPS-01, §17.2).

The root ``answerdesk`` command keeps the frozen typer signature and delegates here, so
all ingestion behaviour lives inside the package that owns it:

    answerdesk ingest --demo                # build the checked-in synthetic corpus
    answerdesk ingest ./some/kb --profile offline   # ingest a directory of documents

Scale knobs (FR-ING-07: the ≥ 3,000 docs / ≥ 20,000 chunks operating point must be
reachable from the same command) are environment variables rather than new CLI flags,
so grading the demo corpus and producing the scale corpus are literally the same call::

    ANSWERDESK_INGEST_DOCS_PER_LOCALE=500 answerdesk ingest --demo

``ANSWERDESK_INGEST_SEED`` changes the generator seed (the default is frozen, because it
is part of the corpus version).
"""

from __future__ import annotations

import os
from pathlib import Path

from answerdesk.config import Settings
from answerdesk.ingest.builder import BuildResult, build_demo, build_from_directory
from answerdesk.ingest.errors import IngestError
from answerdesk.ingest.synthetic import DEMO_SEED, SyntheticConfig

#: Environment variables that scale the demo build (documented in docs/ingestion.md).
ENV_DOCS_PER_LOCALE = "ANSWERDESK_INGEST_DOCS_PER_LOCALE"
ENV_SEED = "ANSWERDESK_INGEST_SEED"


def synthetic_config_from_env() -> SyntheticConfig:
    """Read the scale/seed knobs with the frozen defaults of the checked-in corpus."""
    docs_per_locale = _int_env(ENV_DOCS_PER_LOCALE, SyntheticConfig().docs_per_locale)
    seed = _int_env(ENV_SEED, DEMO_SEED)
    return SyntheticConfig(seed=seed, docs_per_locale=docs_per_locale)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise IngestError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise IngestError(f"{name} must be >= 1, got {value}")
    return value


def resolve_corpus_dir(settings: Settings) -> Path:
    """Corpus directory from the effective configuration (``app.corpus_dir``)."""
    return settings.resolve_path(settings.app.corpus_dir)


def run_ingest(
    settings: Settings,
    *,
    source: str | None = None,
    demo: bool = False,
    config: SyntheticConfig | None = None,
) -> BuildResult:
    """Build a corpus and return the result; raises :class:`IngestError` on misuse."""
    corpus_dir = resolve_corpus_dir(settings)
    if demo and source:
        raise IngestError("choose one: --demo builds the synthetic corpus, a path ingests files")
    if demo:
        return build_demo(corpus_dir, config=config or synthetic_config_from_env())
    if source:
        return build_from_directory(corpus_dir, source)
    raise IngestError("nothing to ingest: pass a source path (file or directory) or pass --demo")


def summary_lines(result: BuildResult) -> list[str]:
    """The operator-facing summary — counts, hashes and where the manifest lives."""
    manifest = result.manifest
    stats = manifest.stats
    lines = [
        f"corpus_version: {manifest.corpus_version}",
        f"documents: {stats.documents} (duplicates removed: {stats.duplicate_documents_removed})",
        f"chunks: {stats.chunks} (duplicate chunks removed: {stats.duplicate_chunks_removed})",
        f"tokens: {stats.tokens} (mean {stats.tokens_mean}/chunk, max {stats.tokens_max})",
        f"locales: {', '.join(sorted(stats.docs_per_locale))}",
        f"license: {manifest.license}",
        f"manifest: {result.manifest_path}",
        f"chunks_file: {result.chunks_path}",
    ]
    if result.queries_path:
        lines.append(f"queries_file: {result.queries_path}")
    return lines


def ingest_command(
    *,
    settings: Settings,
    source: str | None,
    demo: bool,
    echo: object | None = None,
) -> BuildResult:
    """Typer-facing entry point: run the build and print the summary."""
    result = run_ingest(settings, source=source, demo=demo)
    printer = echo if callable(echo) else print
    for line in summary_lines(result):
        printer(line)
    return result


__all__ = [
    "ENV_DOCS_PER_LOCALE",
    "ENV_SEED",
    "ingest_command",
    "resolve_corpus_dir",
    "run_ingest",
    "summary_lines",
    "synthetic_config_from_env",
]
