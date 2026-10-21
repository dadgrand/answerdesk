#!/usr/bin/env python
"""Build, fetch and verify ``answerdesk-bench``.

Usage (from the ``answerdesk`` repository root)::

    uv run python scripts/build_bench.py --fetch     # network: fill .cache/answerdesk-bench/
    uv run python scripts/build_bench.py --build     # offline: write
                                                     # data/bench/answerdesk-bench/v0.1/
    uv run python scripts/build_bench.py --verify    # offline: check the committed release

``--fetch`` is the only command that opens a socket. It downloads the pinned sources,
checks every SHA-256 against ``sources.lock.json`` and writes the git-ignored cache.
``--build`` is a pure function of that cache plus the repository's synthetic corpus and
t11 golden split; it writes canonical JSONL and a manifest that hashes every file.
``--verify`` re-reads the release, recomputes every invariant and exits non-zero on the
first class of failure it finds (``docs/benchmark.md`` §5, §7).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench.build import assemble
from bench.config import BENCH_DIR, CACHE_DIR, CORPUS_DIR, GOLDEN_DIR, SOURCE_LOCK, SPLITS_FILE
from bench.sources import check_cache, check_local_sources, fetch, load_lock
from bench.verify import summarise, verify


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_bench.py",
        description=(
            "answerdesk-bench v0.1: fetch, build or verify the curated multilingual benchmark."
        ),
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--fetch", action="store_true", help="download the pinned sources into the cache (network)"
    )
    action.add_argument(
        "--build", action="store_true", help="build the release from the cache (offline)"
    )
    action.add_argument(
        "--verify", action="store_true", help="verify the committed release (offline)"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download or re-derive even when the cache is warm"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=CACHE_DIR, help="download cache (git-ignored)"
    )
    parser.add_argument("--bench-dir", type=Path, default=BENCH_DIR, help="release directory")
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR, help="synthetic corpus root")
    parser.add_argument(
        "--golden-dir", type=Path, default=GOLDEN_DIR, help="t11 golden split directory"
    )
    parser.add_argument(
        "--splits-file", type=Path, default=SPLITS_FILE, help="t11 frozen split file"
    )
    parser.add_argument("--lock", type=Path, default=SOURCE_LOCK, help="pinned source lock")
    parser.add_argument(
        "--lax-cache",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = _parser().parse_args(argv)
    lock = load_lock(args.lock)

    if args.fetch:
        fetch(args.cache_dir, lock, force=args.force)
        problems = [
            *check_cache(args.cache_dir, lock),
            *check_local_sources(lock, args.corpus_dir.parent),
        ]
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
        print(f"cache ready: {args.cache_dir}")
        return 0

    if args.build:
        _files, manifest = assemble(
            cache_dir=args.cache_dir,
            corpus_dir=args.corpus_dir,
            golden_dir=args.golden_dir,
            splits_file=args.splits_file,
            lock_path=args.lock,
            out_dir=args.bench_dir,
            lax_cache=args.lax_cache,
        )
        counts = manifest["counts"]
        print(
            f"built {counts['items']} items ({counts['retrieval_queries']} retrieval queries, "
            f"{counts['qa_items']} QA, {counts['domain_items']} domain) in {args.bench_dir}"
        )
        print(f"manifest self_sha256={manifest['self_sha256']}")
        return 0

    verdicts = verify(
        args.bench_dir,
        corpus_dir=args.corpus_dir,
        golden_dir=args.golden_dir,
        splits_file=args.splits_file,
        lock_path=args.lock,
        lock=lock,
    )
    print(summarise(verdicts))
    return 0 if all(verdict.status != "fail" for verdict in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
