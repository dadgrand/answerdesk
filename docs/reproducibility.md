# Reproducibility

Everything this repository claims is meant to be re-runnable on a laptop, offline, without an API
key. This document is the operator's guide to that promise: what is pinned, what is deterministic,
which command regenerates which artifact, and what cannot be reproduced without provider keys.

The rule the repository enforces: **no number exists without an artifact and a command**. If a
number in a document is not produced by a committed script from committed inputs, it is a bug.

---

## 1. What the reference run used

| item | value |
|---|---|
| OS | Windows 11 (x86_64) |
| Python | 3.12.10 (uv-managed) |
| Package manager | `uv` (see `uv.lock`) |
| GPU | none required; CPU-only path is the default |
| Node (console) | 20+ (verified with Node 24 / npm 11) |
| Corpus | `corpus-77d7d081813738a6` — 810 documents, 7,229 chunks |
| Golden set | `evals/golden/v1` (seed 20261021; dev 216 / test 530) |
| Benchmark release | `answerdesk-bench` 0.1.0, manifest `self_sha256 = 65de8498…` |
| Live models (primary arm) | generator `qwen3.6-35b-a3b-noreason`, cross-family judge `deepseek-flash` |

The full provenance block (corpus, golden, split, bench, pricing, summary hashes) is printed at the
top of [`docs/results.md`](results.md) on every build. If one input changes without a re-run, the hash
chain no longer matches and `build-results.py --check` fails.

## 2. What is pinned

| input | pinned by | how the pin is enforced |
|---|---|---|
| Python dependencies | `uv.lock` | CI runs `uv sync --extra dev --frozen` |
| Console dependencies | `web/package-lock.json` | `npm ci` |
| Corpus content | `data/corpus/manifest.json` + sha256 | printed in every report header |
| Golden set & splits | `evals/golden/v1/manifest.json`, `evals/splits/v1.json` | hashes printed; regeneration must reproduce them |
| Benchmark items | `data/bench/.../manifest.json` (`self_sha256`) | adapter verifies file by file |
| Pricing | `configs/pricing.yaml` with `retrieved_at` | cost table and header |
| Model ids | run header | compared in later runs |

Generated corpora and benchmark files are **LF-canonical** so hashes do not depend on the platform
(`.gitattributes`, `core.autocrlf` handling in the build scripts).

## 3. Determinism: what is guaranteed and what is not

Deterministic (and tested):

- the synthetic corpus generator (same seed → byte-identical documents, queries and manifest);
- golden-set regeneration from the corpus (`--regenerate` with the same seed → same hashes);
- the hashing embedder (BLAKE2b, not Python's randomised `hash()`), so rankings are reproducible across
  processes and machines;
- metric computation, bootstrap and McNemar given the same inputs and seed (`eval.seed = 1337`);
- the offline test suite: no network, no keys — an autouse socket guard fails any test that tries to
  open a connection;
- the console's mock mode and component tests (MSW, no backend).

Not deterministic (and therefore reported as distributions, with n and percentiles):

- hosted model outputs (temperature 0 still varies across provider deployments);
- judge outputs and their latency tail;
- wall-clock latency and cost — always tied to the recorded hardware and pricing snapshot;
- anything that depends on which provider deployment answers.

Replay mode exists for this reason: `--replay` / `provider=fake` runs the full pipeline on recorded
doubles and labels every artifact `REPLAY-MODE`. Replay numbers may never be presented as live ones —
that rule is enforced in review and stated in the results header.

## 4. Command matrix

Run from the repository root. `pwsh` on Windows; the Python commands are cross-platform.

| Goal | Command | Needs network/keys | Writes |
|---|---|---|---|
| Install (light) | `uv sync --extra dev` | first time only | `.venv/` |
| Offline gate (same as CI) | `pwsh -File scripts/ci-offline.ps1` | no | `.pytest_cache/`, caches |
| One-command demo | `pwsh -File scripts/demo.ps1` | no | `data/`, `reports/demo-t12/` |
| Build corpus | `uv run answerdesk ingest --demo` | no | `data/corpus/` |
| Build index | `uv run answerdesk index --demo` | no | `data/index/<corpus-version>/` |
| Ask one question | `uv run answerdesk ask --fake "<question>"` | no | nothing |
| Calibrate (synthetic) | `uv run answerdesk calibrate --synthetic --out reports/policy-smoke` | no | `reports/policy-smoke/` |
| Eval smoke (offline) | `uv run answerdesk eval --synthetic --provider fake --report reports/offline-smoke` | no | `reports/offline-smoke/` |
| Service health check | `$env:NO_PROXY='*'; uv run --extra api answerdesk serve --fake --port 8765 --healthcheck` | no | `logs/` |
| Live evidence package | `pwsh -File scripts/run-bench.ps1` | **yes** (`.secrets/*.key`) | `reports/bench/`, **rewrites `docs/results.md`** |
| Replay evidence package | `pwsh -File scripts/run-bench.ps1 -Replay` | no | `reports/bench/` (labelled replay), rewrites `docs/results.md` |
| Rebuild results doc | `uv run python scripts/build-results.py --run reports/bench --out docs/results.md` | no | `docs/results.md` |
| Verify results doc is current | `uv run python scripts/build-results.py --run reports/bench --out docs/results.md --check` | no | nothing (exit 1 if stale) |
| External arm, list suites | `uv run answerdesk eval external --list-suites` | no | nothing |
| External arm, run | `uv run answerdesk eval external --run --languages ru --answerer-model <model>` | **yes** (+ upstream download) | `reports/external-<run>/` |
| Load / concurrency check | `uv run python scripts/load.py --smoke` (CI) or `--full` | no | `reports/load*` if `--json` given |
| Console install / test / build | `cd web && npm ci && npm run test && npm run build` | `npm ci` needs the registry (or a warm npm cache) | `web/node_modules/`, `web/dist/` |
| Console screenshots | `cd web && npm run screenshots` | no (needs the Python service running) | `docs/assets/*.png` |

> `scripts/run-bench.ps1` is intentionally a **rewriter**: its last step regenerates
> `docs/results.md` from the artifacts it just produced. Run it in a clean tree (or on a branch) — a
> replay run would otherwise replace live numbers with replay numbers in the working copy.

## 5. Which command reproduces which table

| Table | Artifact | Producer |
|---|---|---|
| Retrieval ablation + paired comparisons | `reports/bench/retrieval/` | `answerdesk eval --all …` |
| Answer quality / escalation / per-language | `reports/bench/answer/`, `.../escalation/` | same |
| Calibration verdict, scorer study | `reports/bench/escalation/`, `.../summary.json` | same + `scripts/fit-calibration.py`, `scripts/recompute-escalation.py` |
| Chunking ablation | `reports/bench/retrieval/` | same |
| Semantic cache | `reports/bench/cache/` | same + `scripts/recompute-escalation.py` |
| Latency / cost | `reports/bench/ops/` | same |
| External validity | `reports/external-<run>/` | `answerdesk eval external --run` + `scripts/build-external.py` |
| Load & concurrency | `reports/load*/` | `scripts/load.py --full --json …` |

The documents themselves (`docs/results.md`, `docs/external-validity.md`, `docs/benchmark.md`) are
build products. Editing them by hand is a defect: change the script or the artifacts instead.

## 6. Troubleshooting

**`httpx` fails with `Unknown scheme for proxy URL … socks4://`.** Windows proxy settings are read by
`httpx`; a SOCKS URL it cannot drive breaks client construction. Set `$env:NO_PROXY='*'` (scripts do
this) or per-host `NO_PROXY=localhost,127.0.0.1`.

**`uv sync` cannot reach PyPI.** A machine behind a SOCKS proxy needs `HTTPS_PROXY`/`ALL_PROXY` set for
uv (e.g. `socks5h://127.0.0.1:10808`). With a warm `uv.lock` + cache, `uv sync` works with no network
at all; only re-locking needs the index.

**`npm ci` fails offline.** The npm cache must already contain the tarballs (`npm ci --offline` proves
it). Otherwise `npm ci` needs the registry.

**`index --demo` says the fingerprint is unchanged.** That is the content-addressed cache working; pass
`--force` to rebuild (the live benchmark does).

**`build-results.py --check` fails.** `docs/results.md` no longer matches its artifacts: regenerate it
from `reports/bench/`, do not edit the document.

**`serve` starts but `/healthz` shows `calibration: missing`.** The service needs a sealed calibration
artifact; run `answerdesk calibrate --synthetic --out reports/policy-smoke` first (the demo script does
it).

**A model call fails with a schema error.** Some providers reject `response_format=json_schema`; the
provider layer falls back and the run is labelled *schema-fallback, not directly comparable* (this is
what the secondary DeepSeek arm does). Do not compare such an arm to a strict-JSON arm.

**Torch/CUDA on Windows.** Install CPU wheels with `uv sync --extra local`; for CUDA use
`uv pip install torch --index-url https://download.pytorch.org/whl/cu124` (match your driver). Details
in [`docs/architecture.md`](architecture.md) §11.

## 7. What cannot be reproduced without keys

| Claim | Needs |
|---|---|
| Live answer quality, judged metrics, escalation behaviour | a generator key and a **cross-family** judge key |
| Cost per answer | provider billing (the current run recorded a free tariff, i.e. 0.0) |
| External arm (`eval external --run`) | answerer key + one-time upstream download of Belebele/Mr. TyDi |
| Human-judge κ | human annotators (the protocol and label sheets ship; labels do not) |

Without keys the honest options are: run the offline smoke, run `run-bench.ps1 -Replay` and label the
output as replay, or read the committed artifacts. There is no fourth option in which a live-looking
number is produced without a live call.
