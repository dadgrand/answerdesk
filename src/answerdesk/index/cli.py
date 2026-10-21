"""The ``answerdesk index`` command surface (FR-OPS-02, §17.2 CMD-E).

The root CLI keeps the frozen typer signature and delegates here, exactly as ingestion
does, so index behaviour lives in the package that owns it:

    answerdesk index --demo              # ensure the demo corpus exists, then index it
    answerdesk index --demo --force      # rebuild even if the fingerprint matches

``--demo`` is self-healing: on a clean checkout with no corpus it builds the synthetic
corpus first (the same call as ``answerdesk ingest --demo``). That is deliberate — one
command that fails because of a missing sibling command is a worse contract than one
command that does the obvious thing and says so.
"""

from __future__ import annotations

from pathlib import Path

from answerdesk.config import Settings
from answerdesk.index.store import IndexBuildResult, build_indexes


def index_command(
    *,
    settings: Settings,
    demo: bool = False,
    force: bool = False,
    corpus_dir: Path | None = None,
    echo: object | None = None,
) -> IndexBuildResult:
    """Build or reuse the indexes and print the provenance the operator needs."""
    printer = echo if callable(echo) else print
    result = build_indexes(
        settings,
        corpus_dir=corpus_dir,
        demo=demo,
        force=force,
        echo=printer,
    )
    for line in summary_lines(result):
        printer(line)
    return result


def summary_lines(result: IndexBuildResult) -> list[str]:
    """Stable, greppable build report (and the numbers the docs quote)."""
    verb = "unchanged (reused)" if result.reused else "built"
    lines = [
        f"index: {verb}",
        f"corpus_version: {result.corpus_version}",
        f"index_version: {result.index_version}",
        f"chunks: {result.chunk_count}",
        f"embedder: {result.embedder.name} (dimension {result.embedder.dimension})",
        f"index_dir: {result.index_dir}",
    ]
    timings = result.timings_ms
    if timings:
        parts = [f"{name}={value}ms" for name, value in sorted(timings.items())]
        lines.append("timings_ms: " + ", ".join(parts))
    return lines


__all__ = ["index_command", "summary_lines"]
