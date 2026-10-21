"""Recompute the retrieval ablation from the existing index (no generation, no judging).

The ablation is a pure function of the index, the frozen cases and the arm configuration, so
it can be refreshed without touching the answer arms — useful when the ranking window or a
fusion parameter is corrected after the fact. It rewrites ``retrieval/`` and patches the
``retrieval`` block of ``summary.json`` in place (everything else is left untouched).

    uv run python scripts/refresh-retrieval.py --run reports/bench --retrieval-limit 540
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from answerdesk.config import load_settings  # noqa: E402
from answerdesk.eval.bench import BenchConfig, prepare, run_retrieval  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=REPO_ROOT / "reports" / "bench")
    parser.add_argument("--retrieval-limit", type=int, default=540)
    parser.add_argument("--golden-version", default="v1")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument(
        "--only",
        choices=["retrieval", "chunking"],
        default="retrieval",
        help="which block to recompute",
    )
    args = parser.parse_args(argv)
    if args.only == "chunking":
        return run_chunking(args.run, limit=args.retrieval_limit, replay=args.replay)
    settings = load_settings(profile=None)
    config = BenchConfig(
        run_id=args.run.name,
        golden_version=args.golden_version,
        retrieval_limit=args.retrieval_limit,
        replay=args.replay,
        calibrate_on_dev=False,
        generator_arms=(),
    )
    state = prepare(settings, config, evals_root=REPO_ROOT / "evals")
    results = run_retrieval(state, out_dir=args.run)
    summary_path = args.run / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["retrieval"] = {
            arm: {
                "recall_any@10": _value(payload["cells"], "recall_any@10"),
                "recall_any@20": _value(payload["cells"], "recall_any@20"),
                "recall_all@20": _value(payload["cells"], "recall_all@20"),
                "nDCG@10": _value(payload["cells"], "nDCG@10"),
                "MRR@10": _value(payload["cells"], "MRR@10"),
                "n_labeled": payload["n_labeled"],
            }
            for arm, payload in results["arms"].items()
        }
        summary["retrieval_comparisons"] = results["comparisons"]
        summary["retrieval_refreshed"] = str(Path(__file__).name)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "retrieval refreshed:",
        {
            arm: _value(payload["cells"], "recall_any@20")
            for arm, payload in results["arms"].items()
        },
    )
    return 0


def _value(cells: Sequence[dict[str, object]], name: str) -> dict[str, object] | None:
    for cell in cells:
        if cell.get("name") == name and cell.get("locale") == "macro":
            return {
                "value": cell.get("value"),
                "ci_low": cell.get("ci_low"),
                "ci_high": cell.get("ci_high"),
                "n": cell.get("n"),
            }
    return None


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------------------
# Supplementary ablations (chunking) — separate entry point in the same script
# --------------------------------------------------------------------------------------


def run_chunking(run: Path, *, limit: int = 240, replay: bool = False) -> int:
    """Chunking variant block: structure-first vs naive fixed window."""
    from answerdesk.eval.bench import prepare, run_chunking_ablation

    settings = load_settings(profile=None)
    config = BenchConfig(
        run_id=run.name,
        retrieval_limit=limit,
        replay=replay,
        calibrate_on_dev=False,
        generator_arms=(),
    )
    state = prepare(settings, config, evals_root=REPO_ROOT / "evals")
    payload = run_chunking_ablation(state, out_dir=run, cases=state.test_cases, limit=limit)
    print("chunking ablation:", json.dumps(payload["rows"], ensure_ascii=False)[:400])
    return 0
