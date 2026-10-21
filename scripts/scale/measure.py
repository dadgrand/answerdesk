"""Operating-point scale run for NFR-SCALE-01 (R-05, t42).

Builds a 3,000-document / 20,000+-chunk corpus from the same seeded generator that
produces the demo corpus (with the same grounding and identifier gates), builds the BM25 +
dense indexes, and measures:

* ingest wall-clock (corpus generation + write),
* index build wall-clock and index size on disk,
* retrieval latency p50 / p95 / mean over a deterministic sample of the corpus queries,
* peak resident memory of the process.

Everything is offline: the default embedder is the deterministic ``HashingEmbedder``
(``embedding.backend: hashing``), so the run needs no API key and no model download.
``--embedder local`` switches to the ``local`` extra when a sentence-transformers model is
installed.

Usage
-----
    uv run python scripts/scale/measure.py --smoke        # CI-sized probe, ~seconds
    uv run python scripts/scale/measure.py                # operating point (500 docs/locale)
    uv run python scripts/scale/measure.py --docs-per-locale 600 --queries 300

Artifacts
---------
``reports/scale/last-run.json`` (committed) and ``reports/scale/last-run.md`` are written on
every non-``--no-write`` run; the report embeds the exact command line, hardware profile,
corpus identity, timings, latency distribution and the NFR-SCALE-01 verdict. The scale
corpus and its index live under ``data/scale/`` (git-ignored), never in ``data/corpus``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "src"))

from answerdesk.config import Settings, load_settings  # noqa: E402
from answerdesk.index.embedders import default_embedder  # noqa: E402
from answerdesk.index.store import build_indexes, load_indexes  # noqa: E402
from answerdesk.ingest.builder import build_demo  # noqa: E402
from answerdesk.ingest.synthetic import SyntheticConfig  # noqa: E402
from answerdesk.retrieval.pipeline import build_pipeline  # noqa: E402

SCALE_ROOT = REPO_ROOT / "data" / "scale"
SCALE_CORPUS = SCALE_ROOT / "corpus"
SCALE_INDEX = SCALE_ROOT / "index"
REPORT_DIR = REPO_ROOT / "reports" / "scale"

#: NFR-SCALE-01 thresholds (docs/requirements.md §5).
TARGET_DOCUMENTS = 3_000
TARGET_CHUNKS = 20_000
TARGET_INDEX_BUILD_S = 600.0
TARGET_PEAK_RSS_GB = 4.0
TARGET_RETRIEVAL_P95_MS = 500.0


# ----------------------------------------------------------------------------------
# hardware / memory
# ----------------------------------------------------------------------------------
def _peak_rss_bytes() -> int:
    """Peak resident set size of this process (best effort, platform specific)."""
    try:  # Linux
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    if os.name == "nt":  # Windows: Psapi.GetProcessMemoryInfo, PeakWorkingSetSize
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_Counters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
        except Exception:  # pragma: no cover - non-standard Windows environments
            pass
    try:  # cross-platform fallback when psutil is present
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().peak_wset or 0)
    except Exception:
        return 0


def _hardware_profile() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "implementation": platform.python_implementation(),
    }


# ----------------------------------------------------------------------------------
# measurement helpers
# ----------------------------------------------------------------------------------
def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile (no interpolation; small samples stay honest)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(fraction * len(ordered))))
    return float(ordered[rank - 1])


def _finding_count(block: object) -> int | None:
    """Count findings in a manifest block that may be a model or a plain dict."""
    if block is None:
        return None
    if isinstance(block, dict):
        findings = block.get("findings")
    else:
        findings = getattr(block, "findings", None)
    if findings is None:
        return None
    return len(findings)


def _dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@dataclass(frozen=True)
class RunConfig:
    label: str
    docs_per_locale: int
    queries: int
    seed: int
    k: int
    write: bool
    embedder: str
    repeat: int = 1


def _scale_settings(*, embedder: str) -> Settings:
    """Settings pointed at data/scale, offline by default (no network, no API key).

    ``load_settings`` applies the profile *after* the YAML/env merge, so the scale paths
    and the embedder backend are set with ``model_copy`` on the resolved offline settings —
    the profile still guarantees no network access at import time.
    """
    base = load_settings(profile="offline")
    app = base.app.model_copy(
        update={"corpus_dir": str(SCALE_CORPUS), "index_dir": str(SCALE_INDEX)}
    )
    embedding = base.embedding.model_copy(update={"backend": embedder})
    return base.model_copy(update={"app": app, "embedding": embedding})


def _sample_queries(corpus_dir: Path, limit: int) -> list[str]:
    """Deterministic, category-spread sample of the corpus queries."""
    path = corpus_dir / "queries" / "queries.jsonl"
    if not path.is_file():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not records:
        return []
    if limit >= len(records):
        return [str(record["text"]) for record in records]
    stride = len(records) / limit
    chosen = [records[int(index * stride)] for index in range(limit)]
    return [str(record["text"]) for record in chosen]


def run(config: RunConfig) -> dict[str, Any]:
    """Execute one scale run and return its report (also written to reports/scale)."""
    SCALE_CORPUS.mkdir(parents=True, exist_ok=True)
    settings = _scale_settings(embedder=config.embedder)

    # 1. ingest: the same seeded generator + the same grounding/identifier gates.
    started = time.perf_counter()
    build = build_demo(
        SCALE_CORPUS,
        config=SyntheticConfig(docs_per_locale=config.docs_per_locale, seed=config.seed),
    )
    ingest_s = time.perf_counter() - started
    manifest = build.manifest
    corpus_stats = manifest.stats

    # 2. index build (BM25 + dense), measured separately from ingest.
    embedder = default_embedder(settings)
    started = time.perf_counter()
    index_result = build_indexes(
        settings,
        corpus_dir=SCALE_CORPUS,
        force=True,
        embedder=embedder,
        echo=lambda line: print(f"  index: {line}"),
    )
    index_s = time.perf_counter() - started
    index_dir = Path(index_result.index_dir)
    index_bytes = _dir_size(index_dir)

    # 3. retrieval latency at the operating point.
    bundle = load_indexes(settings, corpus_dir=SCALE_CORPUS, embedder=embedder, index_dir=index_dir)
    pipeline = build_pipeline(settings, bundle)
    queries = _sample_queries(SCALE_CORPUS, config.queries)
    latencies: list[float] = []
    stage_totals: dict[str, float] = {}
    for query in queries:
        started = time.perf_counter()
        trace, timings = pipeline.retrieve_with_timings(query, final_k=config.k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed_ms)
        assert trace.packed is not None
        for stage, value in timings.items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + value

    peak_rss = _peak_rss_bytes()
    report: dict[str, Any] = {
        "task": "t42 (R-05, NFR-SCALE-01)",
        "label": config.label,
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "cwd": str(REPO_ROOT),
        "hardware": _hardware_profile(),
        "config": {
            "docs_per_locale": config.docs_per_locale,
            "locales": list(manifest.generator.locales if manifest.generator else []),
            "seed": config.seed,
            "queries_measured": len(queries),
            "final_k": config.k,
            "embedding_backend": embedder.name,
            "embedding_dimension": embedder.dimension,
            "fuser": settings.retrieval.fuser,
            "bm25": {"k1": 1.2, "b": 0.75},
            "chunking": dict(manifest.chunking) if manifest.chunking else {},
        },
        "corpus": {
            "corpus_version": manifest.corpus_version,
            "schema_version": manifest.schema_version,
            "documents": corpus_stats.documents,
            "chunks": corpus_stats.chunks,
            "tokens": corpus_stats.tokens,
            "queries": manifest.queries.get("count", 0),
            "license": manifest.license,
            "path": str(SCALE_CORPUS.relative_to(REPO_ROOT)),
            "grounding_findings": _finding_count(manifest.reference_grounding),
            "identifier_findings": _finding_count(manifest.identifier_scan),
        },
        "timings": {
            "ingest_s": round(ingest_s, 3),
            "index_build_s": round(index_s, 3),
            "index_build_reported_ms": index_result.timings_ms,
            "retrieval": {
                "count": len(latencies),
                "p50_ms": round(_percentile(latencies, 0.50), 3),
                "p95_ms": round(_percentile(latencies, 0.95), 3),
                "mean_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
                "max_ms": round(max(latencies), 3) if latencies else 0.0,
                "stage_mean_ms": {
                    stage: round(total / max(1, len(latencies)), 3)
                    for stage, total in sorted(stage_totals.items())
                },
            },
        },
        "sizes": {
            "index_bytes": index_bytes,
            "index_mib": round(index_bytes / (1024 * 1024), 2),
            "corpus_bytes": _dir_size(SCALE_CORPUS),
            "index_dir": str(index_dir.relative_to(REPO_ROOT)),
        },
        "resources": {
            "peak_rss_bytes": peak_rss,
            "peak_rss_gb": round(peak_rss / (1024**3), 3) if peak_rss else None,
        },
    }
    report["nfr_scale_01"] = _verdict(report)
    if config.write:
        _write_reports(report, stem="smoke-run" if config.label == "smoke" else "last-run")
    return report


def _verdict(report: dict[str, Any]) -> dict[str, Any]:
    """Compare the measured numbers with the NFR-SCALE-01 thresholds."""
    corpus = report["corpus"]
    timings = report["timings"]
    resources = report["resources"]
    checks = {
        "documents": corpus["documents"] >= TARGET_DOCUMENTS,
        "chunks": corpus["chunks"] >= TARGET_CHUNKS,
        "index_build": timings["index_build_s"] <= TARGET_INDEX_BUILD_S,
        "retrieval_p95": timings["retrieval"]["p95_ms"] <= TARGET_RETRIEVAL_P95_MS,
    }
    if resources["peak_rss_gb"] is not None:
        checks["peak_rss"] = resources["peak_rss_gb"] <= TARGET_PEAK_RSS_GB
    missing = [name for name, ok in checks.items() if not ok]
    return {
        "requirement": "NFR-SCALE-01",
        "status": "measured" if not missing else "miss",
        "checks": checks,
        "failed": missing,
        "targets": {
            "documents": TARGET_DOCUMENTS,
            "chunks": TARGET_CHUNKS,
            "index_build_s": TARGET_INDEX_BUILD_S,
            "peak_rss_gb": TARGET_PEAK_RSS_GB,
            "retrieval_p95_ms": TARGET_RETRIEVAL_P95_MS,
        },
    }


def _write_reports(report: dict[str, Any], *, stem: str = "last-run") -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"{stem}.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_path = REPORT_DIR / f"{stem}.md"
    md_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(f"\nreport: {json_path.relative_to(REPO_ROOT)} + {md_path.relative_to(REPO_ROOT)}")


def _markdown(report: dict[str, Any]) -> str:
    corpus = report["corpus"]
    timings = report["timings"]
    retrieval = timings["retrieval"]
    sizes = report["sizes"]
    resources = report["resources"]
    verdict = report["nfr_scale_01"]
    hardware = report["hardware"]
    config = report["config"]
    targets = verdict["targets"]
    checks = verdict["checks"]
    peak = resources["peak_rss_gb"]
    peak_text = f"{peak}" if peak is not None else "not measurable on this platform"
    lines = [
        f"# Scale run — {report[chr(39) + chr(39)] if False else report['label']}",
        "",
        f"Command: `{report['command']}`",
        "",
        "## Verdict",
        "",
        f"**NFR-SCALE-01: {verdict['status'].upper()}** — "
        f"failed checks: {', '.join(verdict['failed']) or 'none'}",
        "",
        "| Check | Target | Measured | Pass |",
        "|---|---|---|---|",
        f"| Documents | >= {targets['documents']:,} | {corpus['documents']:,} "
        f"| {checks['documents']} |",
        f"| Chunks | >= {targets['chunks']:,} | {corpus['chunks']:,} | {checks['chunks']} |",
        f"| Index build | <= {targets['index_build_s']:.0f} s | "
        f"{timings['index_build_s']:.1f} s | {checks['index_build']} |",
        f"| Retrieval p95 | <= {targets['retrieval_p95_ms']:.0f} ms | "
        f"{retrieval['p95_ms']:.1f} ms | {checks['retrieval_p95']} |",
        f"| Peak RSS | <= {targets['peak_rss_gb']:.1f} GB | {peak_text} "
        f"| {checks.get('peak_rss', 'n/a')} |",
        "",
        "## Hardware",
        "",
        f"- {hardware['platform']} ({hardware['machine']}, {hardware['cpu_count']} CPUs)",
        f"- Python {hardware['python']} ({hardware['implementation']})",
        "",
        "## Configuration",
        "",
        f"- documents per locale: {config['docs_per_locale']}, "
        f"locales: {', '.join(config['locales'])}, seed {config['seed']}",
        f"- embedder: `{config['embedding_backend']}` (dim {config['embedding_dimension']}), "
        f"fuser `{config['fuser']}`, final_k {config['final_k']}",
        f"- chunking: {json.dumps(config['chunking'], ensure_ascii=False)}",
        "",
        "## Corpus",
        "",
        f"- corpus_version `{corpus['corpus_version']}` (schema {corpus['schema_version']}), "
        f"{corpus['documents']:,} documents, {corpus['chunks']:,} chunks, "
        f"{corpus['tokens']:,} tokens, {corpus['queries']:,} queries, license {corpus['license']}",
        f"- grounding findings: {corpus['grounding_findings']}, "
        f"identifier findings: {corpus['identifier_findings']}",
        f"- path: `{corpus['path']}` (git-ignored)",
        "",
        "## Numbers",
        "",
        f"- ingest: {timings['ingest_s']:.1f} s",
        f"- index build: {timings['index_build_s']:.1f} s "
        f"(reported: {json.dumps(timings['index_build_reported_ms'])} ms)",
        f"- index size: {sizes['index_mib']:.1f} MiB ({sizes['index_bytes']:,} bytes)",
        f"- retrieval ({retrieval['count']} queries): p50 {retrieval['p50_ms']:.1f} ms, "
        f"p95 {retrieval['p95_ms']:.1f} ms, mean {retrieval['mean_ms']:.1f} ms, "
        f"max {retrieval['max_ms']:.1f} ms",
        f"- stage means: {json.dumps(retrieval['stage_mean_ms'])}",
        f"- peak RSS: {peak_text} GB",
        "",
        _spread_markdown(report),
    ]
    return "\n".join(lines)


def _spread_markdown(report: dict[str, Any]) -> str:
    """Render the repeat-run spread table (empty when only one run was measured)."""
    spread = report.get("spread")
    if not spread:
        return ""
    rows = [
        ("ingest (s)", spread["ingest_s"]),
        ("index build (s)", spread["index_build_s"]),
        ("retrieval p50 (ms)", spread["retrieval_p50_ms"]),
        ("retrieval p95 (ms)", spread["retrieval_p95_ms"]),
        ("peak RSS (GB)", spread["peak_rss_gb"]),
    ]
    lines = [
        f"## Spread over {spread['runs']} runs",
        "",
        "| Metric | min | median | max |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {name} | {value['min']:.3f} | {value['median']:.3f} | {value['max']:.3f} |"
        for name, value in rows
    ]
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="NFR-SCALE-01 operating-point scale run (t42)")
    parser.add_argument(
        "--smoke", action="store_true", help="small CI probe instead of the operating point"
    )
    parser.add_argument("--docs-per-locale", type=int, default=None)
    parser.add_argument("--queries", type=int, default=None, help="queries to time")
    parser.add_argument("--seed", type=int, default=SyntheticConfig().seed)
    parser.add_argument("--k", type=int, default=8, help="final_k for the retrieval measurement")
    parser.add_argument(
        "--embedder",
        choices=("hashing", "bge_m3", "qwen3", "api", "local"),
        default="hashing",
        help="embedding backend; 'local' is an alias for the bge_m3 sentence-transformers path",
    )
    parser.add_argument("--no-write", action="store_true", help="do not touch reports/scale")
    parser.add_argument(
        "--repeat", type=int, default=1, help="run the measurement N times and report the spread"
    )
    args = parser.parse_args(argv)
    docs = args.docs_per_locale if args.docs_per_locale is not None else (12 if args.smoke else 500)
    queries = args.queries if args.queries is not None else (24 if args.smoke else 240)
    embedder = "bge_m3" if args.embedder == "local" else args.embedder
    return RunConfig(
        label="smoke" if args.smoke else "operating-point",
        docs_per_locale=docs,
        queries=queries,
        seed=args.seed,
        k=args.k,
        write=not args.no_write,
        embedder=embedder,
        repeat=args.repeat,
    )


def _spread(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Min / median / max of the headline numbers across repeated runs (t42)."""

    def sample(getter: Any) -> dict[str, float]:
        values = [float(getter(report)) for report in reports]
        return {
            "min": round(min(values), 3),
            "median": round(statistics.median(values), 3),
            "max": round(max(values), 3),
            "n": len(values),
        }

    return {
        "runs": len(reports),
        "ingest_s": sample(lambda r: r["timings"]["ingest_s"]),
        "index_build_s": sample(lambda r: r["timings"]["index_build_s"]),
        "retrieval_p50_ms": sample(lambda r: r["timings"]["retrieval"]["p50_ms"]),
        "retrieval_p95_ms": sample(lambda r: r["timings"]["retrieval"]["p95_ms"]),
        "peak_rss_gb": sample(lambda r: r["resources"]["peak_rss_gb"] or 0.0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    config = _parse_args(argv)
    print(
        f"scale run '{config.label}': {config.docs_per_locale} docs/locale, "
        f"{config.queries} timed queries, embedder={config.embedder}, seed={config.seed}, "
        f"repeat={config.repeat}"
    )
    try:
        reports = [run(config) for _ in range(max(1, config.repeat))]
    except Exception as exc:  # the script must explain, not traceback
        print(f"\nscale run failed: {type(exc).__name__}: {exc}")
        if config.embedder != "hashing":
            print(
                "hint: the hashing embedder is the offline default; a real model needs the "
                "`local` extra (sentence-transformers) and `--embedder api` needs a key "
                "(see docs/scale.md §6)."
            )
        return 2
    report = reports[-1]
    if len(reports) > 1:
        report["spread"] = _spread(reports)
        report["runs"] = [
            {
                "ingest_s": item["timings"]["ingest_s"],
                "index_build_s": item["timings"]["index_build_s"],
                "retrieval_p95_ms": item["timings"]["retrieval"]["p95_ms"],
                "peak_rss_gb": item["resources"]["peak_rss_gb"],
                "index_bytes": item["sizes"]["index_bytes"],
            }
            for item in reports
        ]
        if config.write:
            _write_reports(report, stem="smoke-run" if config.label == "smoke" else "last-run")
    verdict = report["nfr_scale_01"]
    corpus = report["corpus"]
    timings = report["timings"]
    print(
        f"\ncorpus: {corpus['documents']:,} documents / {corpus['chunks']:,} chunks "
        f"({corpus['corpus_version']})"
    )
    print(
        f"ingest {timings['ingest_s']:.1f}s | index {timings['index_build_s']:.1f}s "
        f"({report['sizes']['index_mib']:.1f} MiB) | "
        f"retrieval p50 {timings['retrieval']['p50_ms']:.1f}ms "
        f"p95 {timings['retrieval']['p95_ms']:.1f}ms"
    )
    print(f"NFR-SCALE-01: {verdict['status'].upper()} (failed: {verdict['failed'] or 'none'})")
    if config.label != "smoke" and verdict["status"] != "measured":
        print("operating point not reached — see docs/scale.md for the honest MISS")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
