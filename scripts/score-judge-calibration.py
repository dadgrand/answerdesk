#!/usr/bin/env python
"""Score human judge-calibration sheets and update the mechanical judge gate.

Usage (from the ``answerdesk`` repository root)::

    uv run python scripts/score-judge-calibration.py --score    # score committed sheets
    uv run python scripts/score-judge-calibration.py --check    # offline self-check

``--score`` reads ``reports/external-p0/calibration/calibration-*.csv`` (override with
``--sheets``), validates the schema, computes pairwise and judge–human Cohen's κ with
bootstrap confidence intervals, writes ``reports/judge-gate/{gate.json,gate.md,
disagreements.csv}`` and exits 0. ``--check`` does the same in a temporary directory, proves
the dry-run fixtures flip the gate (and only they), and exits non-zero if any expectation
fails.

This command never writes human labels. A blank sheet set produces ``status=pending`` and
keeps every judged metric labelled ``diagnostic``; the gate opens mechanically only when real
labels satisfy ``evals/judge/gate-policy.json``. See ``docs/judge-calibration.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge_gate  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="score-judge-calibration.py",
        description="Offline human-κ scoring and the mechanical judge-calibration gate.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--score", action="store_true", help="score a sheet set and write the gate artefacts"
    )
    mode.add_argument(
        "--check", action="store_true", help="offline self-check (schemas, κ, gate transitions)"
    )
    mode.add_argument(
        "--label",
        action="store_true",
        help="print the consumer label for the committed gate (calibrated|diagnostic, fail-closed)",
    )
    parser.add_argument(
        "--sheets",
        type=Path,
        default=judge_gate.DEFAULT_SHEETS_DIR,
        help="directory with calibration-*.csv sheets",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=judge_gate.DEFAULT_OUT_DIR,
        help="gate artefact directory",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=judge_gate.POLICY_PATH,
        help="gate policy JSON",
    )
    parser.add_argument("--json", action="store_true", help="print the full artefact as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = _parser().parse_args(argv)
    if args.label:
        print(judge_gate.load_judged_label(args.out / judge_gate.GATE_FILE))
        return 0
    if args.check:
        result = judge_gate.check()
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            for item in result["checks"]:
                print(f"{'PASS' if item['ok'] else 'FAIL'}  {item['check']}: {item['detail']}")
            print(
                f"\n{sum(1 for item in result['checks'] if item['ok'])}/{len(result['checks'])} checks passed"
            )
        return 0 if result["ok"] else 1

    policy = judge_gate.Policy.load(args.policy)
    artefact = judge_gate.evaluate(args.sheets, policy)
    paths = judge_gate.write_artefacts(artefact, args.out)
    if args.json:
        print(
            json.dumps(
                {key: value for key, value in artefact.items() if key != "disagreement_rows"},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            f"judge gate: status={artefact['status']} judged_metrics_label={artefact['judged_metrics_label']} "
            f"publishable={artefact['publishable']} synthetic={artefact['synthetic']}"
        )
        print(
            f"items={artefact['pooled']['items']} labelled={artefact['pooled']['labelled']} "
            f"judge_kappa={artefact['pooled']['judge_kappa']} disagreements={artefact['disagreements']}"
        )
        for language, block in sorted(artefact["per_language"].items()):
            print(
                f"  {language}: rows={block['items']} labelled={block['labelled']} "
                f"judge_kappa={block['judge_kappa']} pairwise_agreement={block['pairwise_agreement']} "
                f"status={block['status']}"
            )
        for reason in artefact["reasons"]:
            print(f"  ! {reason}")
        for name, path in paths.items():
            print(f"wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
