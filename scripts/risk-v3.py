"""Risk-control v3 CLI: persist bench answers, build features, attempt the 5% certificate.

. ./scripts/live-env.ps1 neuraldeep -Judge deepseek
uv run python scripts/risk-v3.py --persist-dev --limit 200
uv run python scripts/risk-v3.py --persist-test --limit 200
uv run python scripts/risk-v3.py --features
uv run python scripts/risk-v3.py --self-consistency --limit 25
uv run python scripts/risk-v3.py --evaluate
uv run python scripts/risk-v3.py --check       # offline, no network
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from answerdesk.eval.risk import v3  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=v3.DEFAULT_OUT)
    parser.add_argument("--persist-dev", action="store_true")
    parser.add_argument("--persist-test", action="store_true")
    parser.add_argument("--rejudge-dev", action="store_true")
    parser.add_argument("--rejudge-test", action="store_true")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--self-consistency", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=v3.ALPHA)
    parser.add_argument("--delta", type=float, default=v3.DELTA)
    args = parser.parse_args(argv)
    config = v3.V3Config(out_dir=args.run, alpha=args.alpha, delta=args.delta)

    if args.persist_dev or args.persist_test:
        from answerdesk.eval.risk import bench_answers

        payload = bench_answers.persist(
            out_dir=args.run,
            split="dev" if args.persist_dev else "test",
            limit=args.limit,
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    if args.rejudge_dev or args.rejudge_test:
        from answerdesk.eval.risk import bench_answers

        payload = bench_answers.rejudge_references(
            out_dir=args.run, split="dev" if args.rejudge_dev else "test", limit=args.limit
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    if args.self_consistency:
        from answerdesk.eval.risk.selfconsistency import build_self_consistency

        payload = build_self_consistency(out_dir=args.run, limit=args.limit or 25)
        print(
            json.dumps(
                {
                    "questions": payload["questions"],
                    "calls": payload["calls"],
                    "mean_agreement": payload["mean_agreement"],
                    "model": payload["model"],
                    "duration_s": payload["duration_s"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.features:
        print(json.dumps(v3.build_features(out_dir=args.run), ensure_ascii=False))
        return 0
    if args.check:
        return v3.check(args.run, config=config)
    summary = v3.evaluate(config=config)
    bench = summary["populations"]["bench"]
    print(
        json.dumps(
            {
                "targets": bench["targets"],
                "certificate_feasible": bench["certificate"]["feasible"],
                "certificate_threshold": bench["certificate"]["threshold"],
                "test": {
                    "auroc": bench["test_report"]["auroc"],
                    "coverage_at_risk": bench["test_report"]["coverage_at_risk"]["coverage"],
                    "realised_risk_at_certified": bench["test_report"]["certified_operating_point"][
                        "realised_risk"
                    ],
                },
                "n_required": summary["n_required"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
