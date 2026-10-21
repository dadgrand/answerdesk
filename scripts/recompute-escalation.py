"""Recompute the escalation block from a run's turn artefacts (no model calls).

Why this exists: the escalation analysis is a pure function of the per-turn records
(confidence, decision, reason code, correctness), so it can be recomputed after the fact —
for a new target risk, a new cost-ratio table, or when the benchmark was interrupted after
the answer arms had already produced their turns. It never regenerates or re-judges text.

    uv run python scripts/recompute-escalation.py --run reports/bench --arm primary
    uv run python scripts/recompute-escalation.py --run reports/bench --target-risk 0.02
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from answerdesk.config import Settings  # noqa: E402
from answerdesk.contracts import Action, EvalCase, EvalCategory, IntentBucket, ReasonCode, Split  # noqa: E402
from answerdesk.eval.bench import (  # noqa: E402
    BenchConfig,
    BenchState,
    run_escalation,
    select_operating_threshold,
)
from answerdesk.eval.corpus import load_corpus  # noqa: E402
from answerdesk.eval.models import TurnRecord  # noqa: E402


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _turn(row: dict[str, str], arm: str) -> TurnRecord:
    return TurnRecord(
        case_id=row["case_id"],
        locale=row["locale"],
        category=EvalCategory(row["category"]),
        arm=arm,
        answerable=row["answerable"] == "True",
        must_escalate=row["answerable"] != "True",
        expected_language=row["locale"],
        action=Action(row["action"]),
        reason_code=ReasonCode(row["reason_code"]),
        answered=row["answered"] == "True",
        escalated=row["answered"] != "True",
        confidence=float(row["p_correct"]),
        answer_ok=row["answer_ok"] == "True",
        language=row["locale"],
        claims_total=int(row.get("claims") or 0),
        gold_chunk_ids=[c for c in (row.get("gold") or "").split(";") if c],
        cited_chunk_ids=[c for c in (row.get("cited") or "").split(";") if c],
        intent_bucket=IntentBucket.INFORMATIONAL,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=REPO_ROOT / "reports" / "bench")
    parser.add_argument("--arm", default="primary")
    parser.add_argument("--target-risk", type=float, default=None)
    args = parser.parse_args(argv)
    settings = Settings()
    corpus = load_corpus(settings.resolve_path(settings.app.corpus_dir))
    target = args.target_risk
    if target is None:
        summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
        target = float(summary.get("config", {}).get("target_risk", 0.02))
    config = BenchConfig(run_id=args.run.name, target_risk=target)
    state = BenchState(
        settings=settings,
        config=config,
        corpus=corpus,
        test_cases=[],
        dev_cases=[],
    )
    test_rows = _rows(args.run / "answer" / args.arm / "turns.csv")
    turns = [_turn(row, args.arm) for row in test_rows]
    cases = [
        EvalCase(
            case_id=row["case_id"],
            locale=row["locale"],
            category=EvalCategory(row["category"]),
            question=row.get("question", ""),
            answerable=row["answerable"] == "True",
            split=Split.TEST,
            gold_chunk_ids=[c for c in (row.get("gold") or "").split(";") if c],
            reference_answer="reference" if row["answerable"] == "True" else None,
            metadata={
                "corpus_category": row.get("corpus_category", ""),
                "noisy_freeform": row.get("stratum") == "noisy",
            },
        )
        for row in test_rows
    ]
    dev_rows = _rows(args.run / "calibration" / "answer" / "primary" / "turns.csv")
    dev_operating: dict[str, Any] | None = None
    if dev_rows:
        dev_turns = [_turn(row, "dev") for row in dev_rows]
        dev_operating = select_operating_threshold(dev_turns, target)
    payload = run_escalation(
        state=state,
        turns=turns,
        cases=cases,
        out_dir=args.run / "_recomputed",
        arm=args.arm,
        dev_operating=dev_operating,
    )
    # publish next to the original artefacts so the report builder picks it up
    import shutil

    published = args.run / "escalation" / f"{args.arm}.json"
    published.parent.mkdir(parents=True, exist_ok=True)
    (args.run / "escalation" / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copyfile(args.run / "escalation" / "results.json", published)
    for name in ("curve.csv", "families.csv", "languages.csv", "cost_sensitivity.csv"):
        source = args.run / "_recomputed" / "escalation" / name
        if source.is_file():
            shutil.copyfile(source, args.run / "escalation" / name)
    plot = args.run / "_recomputed" / "escalation" / "risk_coverage.png"
    if plot.is_file():
        shutil.copyfile(plot, args.run / "escalation" / "risk_coverage.png")
    print(
        f"escalation for {args.arm}: confusion {payload['confusion']} "
        f"operating {payload['operating_point']} oracle {payload['oracle_upper_bound']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
