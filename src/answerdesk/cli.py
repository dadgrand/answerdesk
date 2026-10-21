"""The ``answerdesk`` command-line surface (requirements FR-OPS-01, §17.2).

The skeleton wires the *whole* documented command surface — subcommand names, flags and
exit-code conventions — and delegates behaviour to the owning task. A stub exits with
code 2 and names the task that implements it, so the CLI never pretends to work.

    answerdesk version
    answerdesk config show|check
    answerdesk ingest --demo          (t4)
    answerdesk index --demo           (t5)
    answerdesk ask --fake "..."       (t6)
    answerdesk calibrate --synthetic  (t8)
    answerdesk serve --fake           (t9)
    answerdesk eval ...               (t7)
    answerdesk demo --smoke           (t14)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import typer

from answerdesk import PACKAGE_VERSION
from answerdesk.config import Settings, load_settings
from answerdesk.contracts import SCHEMA_VERSION
from answerdesk.errors import AnswerdeskError, ConfigError, NotYetImplemented
from answerdesk.logging import configure_logging

app = typer.Typer(
    name="answerdesk",
    help="Local-first, multilingual customer-support RAG+LLM reference stack.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
config_app = typer.Typer(
    help="Inspect and validate the effective configuration.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")


def _settings(config: Path | None, profile: str | None) -> Settings:
    """Load settings and configure logging from them; maps errors to a clean exit."""
    try:
        settings = load_settings(config_path=config, profile=profile)  # type: ignore[arg-type]
    except ConfigError as exc:
        typer.secho(f"config error: {exc.message}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    configure_logging(settings.logging.level, json_output=settings.logging.json_output)
    return settings


def _not_yet(surface: str, owner_task: str) -> NoReturn:
    """Exit 2 with a message that names the owning task — never a silent no-op."""
    error = NotYetImplemented(surface, owner_task)
    typer.secho(error.message, err=True, fg=typer.colors.YELLOW)
    raise typer.Exit(code=2)


# --------------------------------------------------------------------------------------
# Implemented in the skeleton
# --------------------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the package version and the frozen contracts schema version."""
    typer.echo(f"answerdesk {PACKAGE_VERSION} (contracts schema {SCHEMA_VERSION})")


@config_app.command("show")
def config_show(
    config: Path | None = typer.Option(None, "--config", help="Path to a YAML config file."),
    profile: str | None = typer.Option(None, "--profile", help="offline | api | local"),
) -> None:
    """Print the effective configuration with every secret redacted (FR-OPS-05)."""
    settings = _settings(config, profile)
    typer.echo(json.dumps(settings.redacted(), indent=2, ensure_ascii=False, sort_keys=True))


@config_app.command("check")
def config_check(
    config: Path | None = typer.Option(None, "--config", help="Path to a YAML config file."),
    profile: str | None = typer.Option(None, "--profile", help="offline | api | local"),
) -> None:
    """Validate that the configured provider can actually be constructed (FR-OPS-05)."""
    settings = _settings(config, profile)
    try:
        provider = settings.resolved_provider()
    except ConfigError as exc:
        typer.secho(f"config error: {exc.message}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(provider.redacted(), indent=2, ensure_ascii=False, sort_keys=True))


# --------------------------------------------------------------------------------------
# Delegated surfaces (owned by later tasks — stubs by design)
# --------------------------------------------------------------------------------------


@app.command()
def ingest(
    source: str | None = typer.Argument(None, help="File or directory to ingest."),
    demo: bool = typer.Option(False, "--demo", help="Build the checked-in synthetic demo corpus."),
    config: Path | None = typer.Option(None, "--config"),
    profile: str | None = typer.Option(None, "--profile"),
) -> None:
    """Ingest documents into the corpus and write the provenance manifest (t4)."""
    settings = _settings(config, profile)
    # Implemented by the owning package (FR-ING-01…10); the CLI only delegates.
    from answerdesk.ingest.cli import ingest_command
    from answerdesk.ingest.errors import IngestError

    try:
        ingest_command(settings=settings, source=source, demo=demo, echo=typer.echo)
    except IngestError as exc:
        typer.secho(f"ingest error: {exc.message}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc


@app.command()
def index(
    demo: bool = typer.Option(False, "--demo", help="Index the demo corpus."),
    force: bool = typer.Option(
        False, "--force", help="Rebuild even when the index fingerprint is unchanged."
    ),
    config: Path | None = typer.Option(None, "--config"),
    profile: str | None = typer.Option(None, "--profile"),
) -> None:
    """Build the BM25 and dense indexes and mint a corpus version (t5)."""
    settings = _settings(config, profile)
    # Implemented by the owning package (FR-OPS-02, FR-RET-01…03); the CLI only delegates.
    from answerdesk.index.cli import index_command

    try:
        index_command(settings=settings, demo=demo, force=force, echo=typer.echo)
    except AnswerdeskError as exc:
        typer.secho(f"index error: {exc.message}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc


@app.command()
def ask(
    question: str = typer.Argument(..., help="Customer question, in any supported locale."),
    fake: bool = typer.Option(False, "--fake", help="Use deterministic doubles, no network."),
    json_output: bool = typer.Option(
        False, "--json", help="Print the full generation result as JSON."
    ),
    config: Path | None = typer.Option(None, "--config"),
    profile: str | None = typer.Option(None, "--profile"),
) -> None:
    """Answer one question end-to-end and print the answer with citations (t6)."""
    settings = _settings(config, profile)
    # Implemented by the owning package (FR-GEN-01…12, FR-OPS-03); the CLI only delegates.
    from answerdesk.generate.ask import render_for_terminal, run_ask

    try:
        result = run_ask(question, settings=settings, fake=fake)
    except AnswerdeskError as exc:
        typer.secho(f"ask error: {exc.message}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    typer.echo(result.to_json() if json_output else render_for_terminal(result))


@app.command()
def calibrate(
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Calibrate on a synthetic score distribution."
    ),
    out: Path | None = typer.Option(None, "--out", help="Directory for calibration artifacts."),
    config: Path | None = typer.Option(None, "--config"),
    profile: str | None = typer.Option(None, "--profile"),
) -> None:
    """Fit the conformal calibration artifact and write provenance (t8)."""
    settings = _settings(config, profile)
    # Implemented by the owning package (FR-ESC-01…07); the CLI only delegates.
    from answerdesk.policy.calibrate import main as calibrate_main

    argv = ["--out", str(out or (settings.resolve_path(settings.app.reports_dir) / "policy"))]
    if synthetic:
        argv.insert(0, "--synthetic")
    raise typer.Exit(code=calibrate_main(argv))


@app.command()
def serve(
    fake: bool = typer.Option(False, "--fake", help="Serve the deterministic offline pipeline."),
    port: int = typer.Option(8765, "--port", help="Port to bind on 127.0.0.1."),
    healthcheck: bool = typer.Option(False, "--healthcheck", help="Start, probe /healthz, exit 0."),
    export_openapi: Path | None = typer.Option(
        None,
        "--export-openapi",
        help="Write the OpenAPI schema to this path and exit (no server is started).",
    ),
    config: Path | None = typer.Option(None, "--config"),
    profile: str | None = typer.Option(None, "--profile"),
) -> None:
    """Run the HTTP service with SSE streaming (t9)."""
    settings = _settings(config, profile)
    # Implemented by answerdesk.service (FR-SVC-01…08); the CLI only delegates.
    from answerdesk.service.server import run_server

    exit_code = run_server(
        settings=settings,
        fake=fake,
        port=port,
        healthcheck=healthcheck,
        export_openapi_path=export_openapi,
    )
    raise typer.Exit(code=exit_code)


@app.command()
def eval(  # the command is named `eval` on purpose (requirements §17.2)
    all_arms: bool = typer.Option(False, "--all", help="Run every ablation arm."),
    retrieval_only: bool = typer.Option(False, "--retrieval-only", help="Retrieval arms only."),
    synthetic: bool = typer.Option(False, "--synthetic", help="Use the synthetic golden set."),
    provider: str = typer.Option("fake", "--provider", help="fake | api"),
    judge_mode: str = typer.Option(
        "auto", "--judge", help="auto | fake | llm (auto follows the provider)."
    ),
    report: Path | None = typer.Option(None, "--report", help="Report output directory."),
    regenerate: bool = typer.Option(
        False, "--regenerate", help="Re-freeze the golden set from the corpus, then evaluate."
    ),
    limit: int | None = typer.Option(None, "--limit", help="Evaluate only the first N cases."),
    answer_limit: int | None = typer.Option(
        None, "--answer-limit", help="Generation arms: evaluate N cases (stratified by locale)."
    ),
    calibration: Path | None = typer.Option(
        None, "--calibration", help="Sealed policy calibration artifact (t8) for the bench."
    ),
    force_index: bool = typer.Option(
        False, "--force-index", help="Rebuild the retrieval index before the benchmark."
    ),
    replay: bool = typer.Option(
        False, "--replay", help="Force replay mode (deterministic doubles, no network)."
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Enforce the §10.1 golden composition (use --no-strict only for a smoke run).",
    ),
    secondary: bool = typer.Option(
        True, "--secondary/--no-secondary", help="Run the secondary generator arm too."
    ),
    split: str = typer.Option("test", "--split", help="test | dev | all."),
    config: Path | None = typer.Option(None, "--config"),
    profile: str | None = typer.Option(None, "--profile"),
) -> None:
    """Run the evaluation harness and write a report (t7)."""
    settings = _settings(config, profile)
    # Implemented by answerdesk.eval (FR-EVL-01…11); the CLI only delegates.
    from answerdesk.eval.cli import eval_command
    from answerdesk.eval.errors import EvalError

    try:
        exit_code = eval_command(
            settings=settings,
            emit=typer.echo,
            all_arms=all_arms,
            retrieval_only=retrieval_only,
            provider=provider,
            judge_mode=judge_mode,
            report_dir=report,
            regenerate=regenerate,
            limit=limit,
            split=split,
            answer_limit=answer_limit,
            calibration=calibration,
            force_index=force_index,
            replay=replay,
            secondary=secondary,
            strict=strict,
        )
    except EvalError as exc:
        typer.secho(f"eval error: {exc.message}", err=True, fg=typer.colors.RED)
        if exc.detail:
            typer.secho(f"  detail: {exc.detail}", err=True)
        raise typer.Exit(code=2) from exc
    if exit_code != 0:  # pragma: no cover - eval_command raises instead
        raise typer.Exit(code=exit_code)


@app.command()
def demo(
    smoke: bool = typer.Option(False, "--smoke", help="One-command offline end-to-end demo."),
    config: Path | None = typer.Option(None, "--config"),
    profile: str | None = typer.Option(None, "--profile"),
) -> None:
    """Run the offline end-to-end demo (ingest → index → ask → decision → report, t14)."""
    _settings(config, profile)
    _not_yet("demo", "t14")


def main() -> None:
    """Console-script entry point (``answerdesk = answerdesk.cli:main``)."""
    try:
        app(prog_name="answerdesk")
    except AnswerdeskError as exc:  # pragma: no cover - defensive top-level handler
        typer.secho(f"{exc.code}: {exc.message}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
