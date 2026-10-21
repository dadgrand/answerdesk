"""CLI surface for the consistency audit (``answerdesk eval consistency …``)."""

from __future__ import annotations

import json
from pathlib import Path

from answerdesk.eval.consistency.facts import LOCALES
from answerdesk.eval.consistency.runner import (
    DEFAULT_RUN,
    ConsistencyConfig,
    build_set,
    run_answers,
    run_audit,
)


def consistency_command(
    *,
    build: bool = False,
    audit: bool = False,
    retrieval: bool = False,
    run: bool = False,
    replay: bool = False,
    report: Path | None = None,
    locales: tuple[str, ...] = LOCALES,
    facts_limit: int | None = None,
    answer_limit: int | None = None,
    variant_locales: tuple[str, ...] = ("ru",),
    provider: str = "fake",
    judge: bool = True,
    retriever: str = "cc@0.75",
    set_dir: Path | None = None,
) -> int:
    report_dir = report or Path("reports") / DEFAULT_RUN
    config = ConsistencyConfig(
        run_id=report_dir.name,
        report_dir=report_dir,
        locales=tuple(locales),
        facts_limit=facts_limit,
        answer_limit=answer_limit,
        variant_locales=tuple(variant_locales),
        provider=provider,
        retriever=retriever,
        replay=replay,
        judge=judge,
        set_dir=set_dir or ConsistencyConfig().set_dir,
    )
    if build:
        manifest = build_set(config)
        print(
            f"built {manifest['facts']} facts x {len(manifest['locales'])} locales "
            f"({manifest['instances']} instances) -> {config.set_dir}"
        )
        return 0
    if retrieval:
        from answerdesk.eval.consistency.runner import run_retrieval_invariance

        payload = run_retrieval_invariance(config)
        rows = payload["rows"]
        successes = sum(1 for row in rows if row["same_evidence"])
        print(f"retrieval invariance: {successes}/{len(rows)} pairs keep the same evidence")
        return 0
    if audit:
        payload = run_audit(config)
        print(
            json.dumps(
                {
                    "checked_documents": payload["checked_documents"],
                    "by_severity": payload["by_severity"],
                    "invariant_holds": payload["invariant_holds"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if payload["invariant_holds"] else 1
    if not run:
        print("nothing to do: pass --build, --audit or --run")
        return 0
    summary = run_answers(config)
    print(
        f"consistency run {config.run_id}: {summary['rows']['total']} rows, "
        f"pooled same-fact {summary['fact_agreement']['pooled']['rate']}, "
        f"evidence invariance {summary['surface_invariance']['evidence_invariance']['rate']}"
    )
    return 0


__all__ = ["consistency_command"]
