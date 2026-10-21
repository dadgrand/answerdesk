"""Build and verify the risk-control v2 artefact set.

# 1. feature table (one bounded reranker call per row; ~340 rows)
. ./scripts/live-env.ps1 neuraldeep
uv run python scripts/risk-v2.py --build-features

# 2. fit on dev, certify with Learn-then-Test, evaluate once on test
uv run python scripts/risk-v2.py --run

# 3. reproducibility gate (no network): re-derives every artefact from features.csv
uv run python scripts/risk-v2.py --check

# optional dev-only ablation: k=3 sampled drafts (bounded, documented)
uv run python scripts/risk-v2.py --self-consistency --limit 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from answerdesk.eval.risk import runner  # noqa: E402


def _reranker(settings) -> object:
    from answerdesk.rerank.api import api_reranker_from_settings

    return api_reranker_from_settings(settings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=runner.DEFAULT_OUT)
    parser.add_argument("--build-features", action="store_true")
    parser.add_argument("--build-bench-features", action="store_true")
    parser.add_argument(
        "--no-rerank", action="store_true", help="Skip the cross-encoder calls (rate limit)."
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-consistency", action="store_true")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=runner.ALPHA)
    parser.add_argument("--delta", type=float, default=runner.DELTA)
    args = parser.parse_args(argv)

    if args.build_features:
        from answerdesk.config import load_settings

        settings = load_settings(profile=None)
        try:
            reranker: object | None = _reranker(settings)
        except Exception as exc:  # no reranker configured: build without that feature, loudly
            print(f"reranker unavailable ({exc}); rerank_top stays 0.0", file=sys.stderr)
            reranker = None
        payload = runner.build_features(out_dir=args.run, reranker=reranker)
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    if args.build_bench_features:
        from answerdesk.config import load_settings
        from answerdesk.eval.risk import benchpop
        from answerdesk.eval.risk.features import write_rows

        settings = load_settings(profile=None)
        reranker = None
        if not args.no_rerank:
            try:
                reranker = _reranker(settings)
            except Exception as exc:
                print(f"reranker unavailable ({exc}); rerank_top stays 0.0", file=sys.stderr)
        else:
            print("rerank disabled by --no-rerank; rerank_top stays 0.0", file=sys.stderr)
        rows = benchpop.load_bench_rows(reranker=reranker)
        args.run.mkdir(parents=True, exist_ok=True)
        write_rows(args.run / "bench-features.csv", rows)
        print(json.dumps(benchpop.describe(rows), ensure_ascii=False))
        return 0
    if args.self_consistency:
        from answerdesk.eval.risk.selfconsistency import build_self_consistency

        payload = build_self_consistency(out_dir=args.run, limit=args.limit)
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    if args.evaluate or not args.check:
        summary = runner.evaluate(out_dir=args.run, alpha=args.alpha, delta=args.delta)
        primary = summary["populations"][summary["primary_population"]]
        print(
            json.dumps(
                {
                    "primary_population": summary["primary_population"],
                    "targets": primary["targets"],
                    "certificate_feasible": primary["certificate"]["feasible"],
                    "n_required": summary["n_required"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    return runner.check(args.run, alpha=args.alpha, delta=args.delta)


if __name__ == "__main__":
    raise SystemExit(main())
