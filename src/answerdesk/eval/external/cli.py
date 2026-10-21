"""CLI surface for the external-validity arm (``answerdesk eval external …``)."""

from __future__ import annotations

import json
from pathlib import Path

from answerdesk.eval.external.bench import Bench
from answerdesk.eval.external.datasets import (
    REGISTRY,
    ExternalDataError,
    cache_root,
    load_curated,
)
from answerdesk.eval.external.runner import (
    BELEBELE_LANGUAGES,
    SUITE_ORDER,
    ExternalConfig,
    run_external,
)


def _list_suites(report: Path) -> int:
    record = load_curated(verify=True)
    bench = Bench(record)
    lines = [
        f"answerdesk-bench {bench.version}  licence set: {bench.manifest.get('license_map')}",
        f"manifest self_sha256: {record.manifest_sha256}",
        f"cache root (upstream datasets): {cache_root()}",
        "",
        "suite                 items  languages                          task",
    ]
    for suite in SUITE_ORDER:
        meta = bench.suites.get(suite, {})
        languages = ",".join(meta.get("languages", [])) or "-"
        lines.append(
            f"{suite:<21} {int(meta.get('items', 0)):<6} {languages:<34} {meta.get('task', '')}"
        )
    lines.append("")
    lines.append("upstream datasets (licence-gated, never vendored when redistribute=false):")
    for name, spec in REGISTRY.items():
        if spec.cache_dir is None:
            continue
        cached = (cache_root() / spec.cache_dir).is_dir()
        lines.append(
            f"  {name:<10} {spec.version:<6} {spec.licence:<28} "
            f"languages={','.join(spec.languages):<12} cached={'yes' if cached else 'no'}"
        )
    lines.append("")
    lines.append(
        "derived-label caveat: Belebele-derived retrieval qrels are derived from the "
        "reading-comprehension gold passage; uz/tg/ky retrieval labels are diagnostic only."
    )
    print("\n".join(lines))
    return 0


def external_command(
    *,
    list_suites: bool = False,
    run: bool = False,
    suites: tuple[str, ...] = SUITE_ORDER,
    languages: tuple[str, ...] = BELEBELE_LANGUAGES,
    limit: int | None = 150,
    report: Path | None = None,
    provider: str = "fake",
    answerer_model: str | None = None,
    embedder: str = "auto",
    samples: int = 1000,
    seed: int = 0,
    replay: bool = False,
    score_calibration: Path | None = None,
) -> int:
    report_dir = report or Path("reports/external-p0")
    if list_suites:
        return _list_suites(report_dir)
    if score_calibration is not None:
        from answerdesk.eval.external.calibration import score_sheets, write_kappa

        sheets = sorted(Path(score_calibration).glob("calibration-*.csv"))
        if not sheets:
            raise ExternalDataError(
                f"no calibration-*.csv sheets in {score_calibration}",
                detail="fill the template produced by --run, then score the same directory",
            )
        payload = score_sheets(sheets)
        path = write_kappa(Path(score_calibration), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"wrote {path}")
        return 0
    if not run:
        print("nothing to do: pass --run (and optionally --list-suites)")
        return 0
    config = ExternalConfig(
        run_id=report_dir.name,
        report_dir=report_dir,
        suites=tuple(suites),
        languages=tuple(languages),
        limit=limit,
        samples=samples,
        seed=seed,
        provider=provider,
        answerer_model=answerer_model,
        embedder=embedder,
        replay=replay,
    )
    summary = run_external(config)
    print(f"external run {config.run_id}: artefacts in {report_dir}")
    for suite in ("belebele_qa", "mrtydi_retrieval", "belebele_retrieval"):
        block = summary.get(suite)
        if block:
            print(f"  {suite}: {'done' if block else 'skipped'}")
    print(f"  judge gate: {summary.get('judge_gate', {}).get('status')}")
    return 0


__all__ = ["external_command"]
