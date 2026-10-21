# Release-candidate review — answerdesk v0.1.0 (t13)

**Audit type:** independent release-candidate review (honesty, tests, security, reproducibility,
documentation accuracy).
**Auditor:** reviewer (independent of the implementation tasks).
**Audited tree:** `answerdesk/` at HEAD `fdb137a` (round 1), re-verified at HEAD `2dfb713`
(round-2 core fixes), `2c0c9dc` (after t44) and `795c73f` (final pushed HEAD, with this report
committed). The final bytes are `795c73f`.
**Status:** **formal verdict: FAIL (needs_revision).** The working-tree offline gate is green
(`pwsh -File scripts/ci-offline.ps1` → "offline gate: OK", 1,555 passed / 1 skipped, coverage 78.40%,
all checkers green — reproducing the captain's claim), but the pushed HEAD is **not green on a clean
checkout**: full suite = **1 failed, 1,547 passed, 2 skipped, 6 errors** (R-28). That is the one open
high finding; R-01, R-02, R-03, R-04, R-05 and R-06 are verified fixed. The fix is small and
test-only (one wrong skip guard plus one missing corpus guard), but it must land before the release
can be reported as green — GitHub Actions checks out the repository without the git-ignored corpus,
so its `pytest` steps fail exactly as reproduced below.

---

## 0. What was audited, and what was not

Executed on this machine (Windows 11, Python 3.12.10, uv 0.11.15, Node 24.16.0 / npm 11.13.0):

| Area | Evidence produced |
|---|---|
| Quickstart | `uv sync --extra dev`; `ingest --demo` (7.8 s); `index --demo`; `ask --fake` (output matches README byte-for-byte); `eval --synthetic --provider fake` (12.1 s, deterministic digest); `scripts/demo.ps1` end-to-end; `--healthcheck` |
| Service | offline service booted, SSE stream inspected, PII and injection turns traced, cache hit verified, `/metrics` and `/healthz` read, OpenAPI export |
| Web | `npm run test` (61/61), `npm run build` (tsc + vite), screenshots inspected |
| Numbers | recomputed coverage/risk/rubric/citation precision/language match/per-language accuracy/retrieval recall+nDCG/external Belebele & Mr. TyDi from committed artifacts; Wilson intervals recomputed independently; provenance hashes verified; `build-results.py --check` and `build-external.py --check` |
| Tests | full suite (1,467 collected: 4 failed, 1,462 passed, 1 skipped), mutation-style breakage on 8 behaviours across 6 modules, vacuous-test AST scan, skip inventory |
| Security | secret scan of tracked files and full git history, live PII/trace inspection, injection probe, dependency/licence inspection |

Not executable in this sandbox, and therefore **not verified**:

* live-provider arms (no API keys; the committed artifacts were audited instead);
* `pip-audit` (no network egress; DNS is blocked). This is itself a finding (R-08);
* `npm ci` from a cold cache / `npm run screenshots` (browser + running service; the committed
  screenshots and `screenshots.json` were inspected instead);
* GitHub Actions runs (no CI execution history available locally); CI claims were checked by
  reading `.github/workflows/ci.yml` and by running the same commands locally.

The working tree was red for two reasons at audit time, both traced to in-flight/known work rather
than to the committed content:

1. t25's uncommitted consistency files: 23 `ruff` findings and one failing test
   (`tests/unit/eval/test_consistency.py::test_cross_document_conflict_fails_the_invariant`), plus
   `cli.py` formatting;
2. three proxy tests fail when the environment sets `NO_PROXY='*'` — which is the workaround the
   README tells Windows users to use (finding R-16).

In round 1, a clean LF checkout of HEAD (`fdb137a`) with no proxy environment variables was
`ruff` / `format` / `mypy` clean and **1,453 passed, 2 skipped, 0 failed** (the two documented skips
in §2.2). On a clean `git archive` copy of HEAD `2dfb713` the same run is **2 failed, 1,526 passed,
2 skipped**; after t44 landed (`2c0c9dc`) it is **2 failed, 1,546 passed, 2 skipped, 6 errors**; at
the final pushed HEAD `795c73f` (with this report committed) it is **1 failed, 1,547 passed,
2 skipped, 6 errors**. All failures/errors are tests that read a corpus that only a developer working
tree has (R-28), not the round-1 findings.

Round 2 (post-fix verification, HEAD `2dfb713`) additionally exercised:

| Check | Command / artifact | Result |
|---|---|---|
| Traceability gate | `uv run python scripts/check-traceability.py` | exit 0: **121 rows · 121/121 AT ids marked · 154 paths resolve**; notes 11 markers outside §16 (R-32) |
| Number gate | `uv run python scripts/check-numbers.py` | exit 0: 1,834 quantities, 3 documented exceptions |
| Licence gate | `uv run python scripts/check-licenses.py` | exit 0: 57 installed distributions, all permissive |
| LAT-01 miss | `docs/results.md` §"NFR-LAT-01 verdict" + `docs/requirements.md` CC-2026-10-21-b | PASS: MISS recorded, requirement not softened |
| Numeric guard | mutation M6 re-run on a clean `git archive` copy of `2dfb713` | baseline 69 passed; mutated **19 failed** → R-04 closed |
| Scale evidence | `reports/scale/last-run.{json,md}` (committed) | 3,000 docs / 26,068 chunks; index 7.7 s; retrieval p95 11.8 ms; peak RSS 0.487 GB; 3 runs |
| Provenance | `build-results.py --check` and `build-external.py --check` on a clean LF `git archive` copy of `2dfb713` | exit 0 on both (R-03 closed) |

Round 3 (final HEAD `795c73f`): the working-tree offline gate was re-run end-to-end
(`pwsh -File scripts/ci-offline.ps1`, no proxy env) and the clean-checkout suite was re-run from a
`git archive` copy. Working tree: **offline gate: OK** — ruff/format/mypy clean, traceability
121/121, numbers green, licences 57/57, coverage **78.40%** (gate 70%), full suite **1,555 passed,
1 skipped** (the loopback-only socket test). Clean checkout: **1 failed, 1,547 passed, 2 skipped,
6 errors** (R-28). The second result is the one that matters for delivery, because
`.github/workflows/ci.yml` runs on a fresh clone with no `data/corpus/chunks.jsonl`.

---

## 1. Requirements traceability

### 1.1 Method

The traceability matrix (`docs/requirements.md` §16) claims 121 rows (87 FR + 34 NFR) and says
`scripts/check-traceability.py` (CMD-T, NFR-REP-03) asserts that **every** `AT-` id appears as a
marker and that **every** module path in §16 exists. The auditor re-implemented that assertion
independently (AT ids from the matrix table rows, markers scanned in `src/`, `tests/`, `scripts/`,
`web/`), then sampled the requirement→code→test chain per module.

### 1.2 Result

| Check | Result |
|---|---|
| AT ids in §16 | 121 (matches 87 FR + 34 NFR) |
| AT ids with a `# AT-…` marker anywhere | **100 / 121** |
| AT ids with no marker at all | **21** (list below) |
| Module paths named in §16 that do not exist | **3 / 33** (`policy/risk.py`, `policy/cost_model.py`, `scripts/check-traceability.py`) |
| `scripts/check-traceability.py` present | **No** — the command exits 2 (`can't open file`) |
| `tests/smoke/test_traceability.py` | Checks only 12 t3-owned ids; does not implement the matrix-wide gate |

Missing markers: `AT-A11Y-01`, `AT-DOC-02`, `AT-LAT-04`, `AT-QUAL-02`, `AT-QUAL-03`, `AT-REP-01`,
`AT-REP-02`, `AT-REP-04`, `AT-SCALE-01`, `AT-SEC-04`, `AT-SEC-05`, and all eleven `AT-UI-01…11`.

The matrix is therefore not enforced today, and 21 requirements have no accept-test marker to point
at. Several of those gaps are real evidence gaps, not bookkeeping (§2–§5): NFR-SCALE-01,
NFR-SEC-05, NFR-QUAL-02, NFR-A11Y-01, NFR-SEC-04, FR-OPS-06.

### 1.3 Requirements that are met with verifiable evidence (spot-checked)

| Requirement | Code | Evidence seen / run |
|---|---|---|
| FR-ING-01…05, FR-ING-08 | `ingest/loaders.py`, `chunker.py`, `canonical.py` | `ingest --demo` reproduces 810 docs / 7,229 chunks / `corpus-77d7d081813738a6`; unit tests pass |
| FR-RET-01…10 | `index/`, `retrieval/`, `rerank/` | ablation artifacts recomputed (bm25 0.996, dense 0.950, rrf@60 0.992, cc@0.75 0.975) |
| FR-GEN-01…12 | `generate/`, `llm/` | `ask --fake` output matches README; 7/8 mutations caught; numeric guard has a gap (R-04) |
| FR-ESC-01…10 | `policy/` | conformal quantile mutation caught; runtime PII/injection guard fires with zero provider calls |
| FR-CAC-01…09 | `cache/` | entity-guard mutation caught; second identical turn = exact cache hit |
| FR-SVC-01…08 | `service/`, `telemetry/` | SSE sequence `retrieval → tokens* → citations → decision → done`, trace on every event; 65 service tests pass |
| FR-EVL-01…11 | `eval/` | two offline eval runs produce the same deterministic digest `dfbb4924…`; `build-results --check` passes on this checkout |
| FR-OPS-01…05 | `cli.py`, `config.py` | all documented subcommands exist except `demo` (R-11) |
| NFR-DET-01/02 | `tests/conftest.py`, `eval/runner.py` | autouse socket guard blocks outbound connections; `test_runner_is_deterministic_apart_from_timing` and the eval digest pass |
| NFR-LAT-02/03 | `retrieval/`, service | retrieval p95 2.54 ms ≪ 200 ms; offline service p95 test asserts ≤ 1,500 ms |
| NFR-COST-01/03 | `llm/cost.py`, `telemetry/` | token counts and per-stage timings present in traces; `usd=0` explained by the recorded free tariff |
| NFR-PORT-01 | `llm/providers.py`, `presets.py` | one OpenAI-compatible client; presets are data; local profile needs no key |
| NFR-SEC-01/02 | `policy/`, `telemetry/` | injection fixtures (78 lines in `evals/golden/v1/adversarial/injection.jsonl`); PII redaction verified live (see §4) |
| NFR-OBS-01/02 | `telemetry/`, `web/` | `/metrics` scrape verified; analytics screenshot + `screenshots.json` `"mode": "live"` |
| NFR-REP-02 | manifests | corpus/golden/split/bench/pricing hashes all match the files **on this checkout** — but not on an LF checkout (R-03) |
| NFR-ML-01 | golden set / reports | six locales present in every per-language table |

Requirements with a **not-met or unverifiable** status are listed as findings R-02, R-04, R-05,
R-07–R-10, R-18 and are not counted as met above.

---

## 2. Test quality

### 2.1 Mutation-style breakage (NFR-QUAL-03 / CL-QUAL-01)

Run on a clean `git archive` copy of HEAD (never the working tree), one semantic change at a time,
with the owning test subset run before and after:

| # | Module | Deliberate break | Tests before | Tests after | Caught |
|---|---|---|---|---|---|
| M1 | `retrieval/fusion.py` | flat multi-item channel normalises to 1.0 instead of 0.0 | 37 passed | 2 failed | yes |
| M2 | `policy/conformal.py` | drop the `(n+1)` finite-sample correction | 27 passed | 5 failed | yes |
| M3 | `policy/pii.py` | detector finds nothing | 71 passed | 30 failed | yes |
| M4 | `cache/semantic_cache.py` | entity-mismatch guard disabled | 19 passed | 1 failed | yes |
| M5 | `ingest/chunker.py` | chunks lose their heading path | 18 passed | 3 failed | yes |
| M6 | `generate/verifier.py` | numeric-guard branch disabled (`elif not numeric_ok` → `elif False`) | 41 passed | **41 passed** | **NO** (R-04) |
| M7 | `generate/verifier.py` | a claim with no citation is `supported` | 28 passed | 1 failed | yes |
| M8 | `telemetry/redaction.py` | PII passes through unredacted | 18 passed | 10 failed | yes |

M6 is a real hole, not a bad test selection: with the mutation in place, a claim whose text is
99%+ lexically supported but carries a **wrong number** (`999` vs evidence `500`) is returned as
`supported` and would be displayed. The guard is implemented, but no test drives it through the
verifier at its enforcement point; `test_numeric_guard.py` tests `numeric_violations()` directly and
`test_repair.py` happens to trip the coverage threshold first.

### 2.2 Vacuous tests and skips

* AST scan of all 1,214 test functions: **0** empty bodies, **0** "no assert, no call" bodies.
* Skips are conditional and documented (2 skipped in a clean checkout: the loopback HTTP test and
  the 1.6 GB bench-cache test). `ANSWERDESK_REQUIRE_API=1` turns missing API-extras into failures.
* Clean-checkout run of HEAD without proxy env: round 1 (`fdb137a`) **1,453 passed, 2 skipped,
  0 failed**; round 2 (`2dfb713`) **2 failed, 1,526 passed, 2 skipped**; after t44 (`2c0c9dc`)
  **2 failed, 1,546 passed, 2 skipped, 6 errors**; final `795c73f` **1 failed, 1,547 passed,
  2 skipped, 6 errors** — the remaining tests read a corpus that only a developer tree has (R-28).
  With `NO_PROXY='*'`: 3 proxy tests fail (R-16). Audit-time working tree round 1 (HEAD + t25 in
  flight + `NO_PROXY='*'`): 4 failed, 1,462 passed, 1 skipped.

### 2.3 Coverage (NFR-QUAL-02)

Full offline suite with branch coverage:

| Package | Line+branch coverage |
|---|---|
| `src/answerdesk/retrieval` | 90% |
| `src/answerdesk/policy` | 89% |
| `src/answerdesk/cache` | 84% |
| `src/answerdesk/eval` | **70%** (at the threshold) |
| four-package total | 78% |

The numbers pass the ≥ 70% requirement on this checkout, but **CI has no coverage step**, so the
threshold is not enforced (R-07).

---

## 3. Numerical honesty

### 3.1 Recomputed from committed artifacts (independent of the report builders)

| Reported | Source artifact | Recomputed | Verdict |
|---|---|---|---|
| coverage 0.740 [0.675, 0.796] | `reports/bench/answer/primary/turns.csv` (148/200) | 0.7400; Wilson [0.675, 0.796] | match |
| auto-answer coverage 0.841 [0.780, 0.888] | same (148/176 answerable) | 0.8409; Wilson [0.780, 0.888] | match |
| realised risk 0.223 [0.163, 0.297] | same (33/148) | 0.22297; Wilson [0.163, 0.297] | match |
| rubric 4.692 [4.576, 4.790] (n=141) | same | mean 4.692 (n=141) | match |
| citation precision 0.671 [0.610, 0.727] | same + `cells.json` | 0.6714 | match |
| language agreement 0.986 [0.952, 0.996] (n=148) | same (146/148 answered) | 0.9865 | match |
| must-escalate recall 1.000 [0.862, 1.000] (24/24) | same | 1.000; Wilson [0.862, 1.000] | match |
| escalation precision 0.462 [0.333, 0.595] (24/52) | same | 0.4615; Wilson [0.333, 0.595] | match |
| false escalation 0.159 | same (28/176) | 0.1591 | match |
| per-language coverage/recall/accuracy | same | all six languages match to 3 dp, including zh accuracy 0.500 [0.307, 0.693] = 11/22 answered | match |
| secondary "61/148 answered turns with no parsed claims" | `turns-secondary.csv` | 61/148 (primary 7/148) | match |
| retrieval arms | `reports/bench/retrieval/*.csv` | bm25 0.996 / dense 0.950 / rrf@60 0.992 / cc@0.75 0.975; nDCG 0.934 / 0.672 / 0.856 / 0.809 | match |
| Belebele ru/uz/tg/ky/zh and deltas | `reports/external-p0/summary.json` | every accuracy, delta and CI matches `docs/external-validity.md` to 6 dp | match |
| Mr. TyDi nDCG@10 0.6036 → 0.7472, Δ +0.1435 [0.1290, 0.1581] | same | matches | match |
| provenance hashes | committed files | corpus `da3396…`, golden `01e95c…`, split `2342e1…`, bench self `65de84…`, pricing `a341c5…`, summary `fce1ae…` all match **this checkout** | match, with R-03 caveat |

No result depends on a hidden key: no key-shaped string exists in any committed report, dataset or
doc; `reports/bench/summary.json` records model ids and families, not credentials; the run metadata
is explicit that the tariff was free (USD 0.0) and why.

The two headline honesty statements the team asked to check are accurate:

* risk control **did not hold** (0.22297 vs α = 0.05; `risk_control_held=false`; no calibrated
  wording ships);
* the zh external result is a **no-claim** (Δ −0.0083, CI [−0.0833, +0.0750] crosses zero) and the
  secondary arm is labelled **not directly comparable** (schema fallback, 61/148 empty claims).

### 3.2 Reproducibility checks on the evidence documents

* `uv run python scripts/build-results.py --run reports/bench --out docs/results.md --check` →
  passes on this Windows checkout ("docs/results.md matches the artefacts").
* The **same command on a clean `git archive` (LF) checkout fails**: "docs/results.md is out of
  date". The four affected hashes change when CRLF is normalised (golden `01e95c…`→`2fd9eb…`,
  split `2342e1…`→`643a28…`, pricing `a341c5…`→`30c26f…`, summary `fce1ae…`→`fe547d…`). This
  contradicts "Every hash above is of a committed file" and the reproducibility document's
  LF-canonical claim — finding R-03.
* `build-external.py --check` passes on both checkouts.
* Two offline eval runs produce the identical digest
  `dfbb4924fae717b5fe68bbdbcab8ed8e379c36a9fa47c2c2db9ee753bc688a48`; this is real determinism.
* **Stale committed artifact:** `reports/offline-smoke/` was generated against
  `corpus-ab58e8c0f5cf506b`, 360 test cases, digest `69dd478a…`, wall clock 4.5 s. The documented
  command now produces `corpus-77d7d081813738a6`, 530 cases, digest `dfbb4924…`, 9.0 s — so the
  committed replay evidence does not correspond to the shipped corpus and any run of the documented
  command dirties the tree (R-12).

---

## 4. Security

| Check | Result |
|---|---|
| Secret scan, tracked files (`git grep` for key/secret/token plus provider key shapes) | **0 findings.** Only deliberate redaction fixtures (`AKIAIOSFODNN7EXAMPLE`, a fake `ghp_…`, PEM headers) in tests that prove redaction. |
| Secret scan, full git history (all 34 commits, added lines) | **0 findings** beyond the same fixtures. `.env`, `.secrets/` never committed; both are gitignored. |
| `.env.example` | Placeholder-only; empty key values. |
| PII redaction in logs/traces (live) | Sent phone + test card + e-mail: response escalated `PII_DETECTED` with `input_tokens=0` (no provider call); trace records `query: "Мой телефон [PHONE], карта [CARD], email [EMAIL] …"`; `grep` for all three raw values across `logs/traces/*.jsonl` and logs: **0 hits**. |
| Token counts survive redaction | Normal turn: `input_tokens=364`, `output_tokens=45` recorded. |
| PII in provider body | Covered by `tests/property/test_injection_and_pii.py`, `test_fallback_redaction.py`, `test_error_redaction.py` (t24/t35 audit trail in `docs/audit-t24.md`); not re-run against a live provider (no key). |
| Prompt injection (live) | "Ignore all previous instructions …" → `escalate/INJECTION_SUSPECTED`, `input_tokens=0`. 78 injection fixtures in the committed adversarial set; delimiter escaping covered by `test_delimiter_neutralisation.py`. |
| Decision path (CL-SEC-01) | `DecisionEngine.evaluate` consumes guards, `p̂`, config, and the generator's block-only signal; guards receive the query and passages but can only escalate/refuse (`GuardVerdict` validator forbids `answer`). No document text or model output selects an answer route. |
| Known, documented limitations | L-7 (risk tier is client-declared; `max(declared, detected)` scheduled) and L-9 (trace redaction is untested defence in depth) are documented in `docs/robustness.md`; the former medium leak L-8 was fixed in t35 with positive tests. |
| Dependency audit | **Not run and not committed** — `pip-audit` cannot be installed offline (DNS blocked), no audit report exists, and no `NOTICE`/licence report is committed although the tree relies on Apache-2.0-licensed components transitively (matplotlib bundles Apache-2.0 ColorBrewer/SheenBidi). NFR-SEC-05 / NFR-REP-04 unmet (R-08). |

---

## 5. Reproducibility — commands as documented

Round-1 execution on `fdb137a`. Round-2 re-checks (traceability/numbers/licences/scale/provenance)
are summarised in §0 and §6.2; the commands marked "fail" below are exactly the ones the round-2
fixes addressed.

| Documented command | Result |
|---|---|
| `uv sync --extra dev` | pass (cached, 0.8 s) |
| `uv run answerdesk ingest --demo` | pass, 7.8 s, 810/7,229, version matches docs |
| `uv run answerdesk index --demo` | pass, reused; 7,229 chunks |
| `uv run answerdesk ask --fake "Сколько стоит обслуживание карты?"` | pass; output identical to the README example, including `evidence_score=0.940`, `input=314 output=45` |
| `uv run answerdesk eval --synthetic --provider fake --report reports/offline-smoke` | pass, 12.1 s, deterministic digest stable across runs |
| `pwsh -File scripts/demo.ps1` | pass end-to-end (corpus → index → ask → calibration → eval → health check) |
| `uv run --extra api answerdesk serve --fake --port 8765` + `GET /healthz` | pass; reports dependencies without secrets |
| `uv run --extra api answerdesk serve --export-openapi <path>` | pass |
| `uv run answerdesk eval external --list-suites` | pass, offline |
| `uv run python scripts/build-results.py … --check` | pass here, **fails on a clean LF checkout** (R-03) |
| `uv run python scripts/build-external.py … --check` | pass |
| `uv run answerdesk config check --profile local` | pass |
| `uv run answerdesk --profile local config check` (README/architecture/generation wording) | **fail**: `No such option: --profile` (R-15) |
| `uv run answerdesk demo --smoke` (CMD-U / §17.3) | **fail**: exit 2, "implemented by task t14" (R-11) |
| `uv run python scripts/check-traceability.py` (CMD-T / §17.3) | **fail**: file does not exist (R-01) |
| `cd web && npm run test` | pass, 61/61, 2.35 s |
| `cd web && npm run build` | pass (tsc --noEmit + vite build) |
| `ruff check .`, `ruff format --check .`, `mypy src/answerdesk` on HEAD copy | pass (clean copy; the live tree is red only for t25's files) |
| `ruff check .`, `ruff format --check .` on the audit-time working tree | fail: 23 lint findings + 7 unformatted files, all in t25's new consistency files plus `cli.py` (in flight; not recorded against the release) |
| clean-checkout `pytest -q` (HEAD, no proxy env) | **1,453 passed, 2 skipped** |
| full `pytest -q` on the audit-time working tree | 4 failed / 1,462 passed / 1 skipped (t25 in flight + `NO_PROXY='*'`; see §0) |

---

## 6. Findings

Severity scale: blocker (must not publish), high (must fix before publish), medium (fix or formally
defer with an owner), low (cleanup).

| id | severity | problem (short) | file |
|---|---|---|---|
| R-01 | high | `scripts/check-traceability.py` missing; traceability gate cannot run; 21/121 AT markers absent; 3/33 matrix module paths do not exist | `scripts/`, `docs/requirements.md:423` |
| R-02 | high | NFR-LAT-01 missed and not reported (generation p50 2618 ms / p95 14,117 ms vs ≤1500/≤4000 ms) | `docs/results.md:424`, `README.md:278` |
| R-03 | high | Provenance hashes are CRLF-dependent; `build-results.py --check` fails on a clean LF checkout | `scripts/build-results.py:53` |
| R-04 | high | Numeric guard (INV-04/FR-GEN-12) untested at the verifier; disabling it lets a wrong-number claim display as supported | `src/answerdesk/generate/verifier.py:238` |
| R-05 | high | NFR-SCALE-01 has no scale-run artifact though it is a MUST G-REL | `docs/requirements.md:431` |
| R-06 | medium | 21 requirements have no `AT-` marker; all UI/a11y/security/evidence gates are unmarked | `docs/requirements.md:1160-1235` |
| R-07 | medium | NFR-QUAL-02 coverage gate absent from CI; eval sits exactly at 70% | `.github/workflows/ci.yml:90` |
| R-08 | medium | NFR-SEC-05/NFR-REP-04: no dependency audit report, no NOTICE/licence report; CMD-Z not runnable offline | `docs/requirements.md:415,424` |
| R-09 | medium | NFR-SEC-04 has no XSS DOM test (React escaping mitigates the risk) | `docs/requirements.md:414` |
| R-10 | medium | NFR-A11Y-01 has no automated a11y check or committed report | `docs/requirements.md:430` |
| R-11 | medium | `answerdesk demo --smoke` is a stub (exit 2); §17.3 acceptance sequence cannot run | `src/answerdesk/cli.py`, `docs/requirements.md:1311` |
| R-12 | medium | Committed `reports/offline-smoke` is stale (old corpus, 360 vs 530 cases) | `reports/offline-smoke/report.md:11` |
| R-13 | medium | Generated results: `duration 0.0 s`; secondary arm attributed to the qwen generator | `docs/results.md:19,250` |
| R-14 | medium | `docs/architecture.md` §12 is a stale skeleton-status section; §6 storage table describes unused FTS5/vectors.npy | `docs/architecture.md:254,378` |
| R-15 | medium | Documented `answerdesk --profile …` commands use a non-existent global flag | `README.md:134`, `docs/generation.md:71` |
| R-16 | medium | Three tests fail under the README's own `NO_PROXY='*'` workaround | `tests/unit/retrieval/test_reranker_proxy.py:109` |
| R-17 | medium | "Upstream data never vendored"/"Belebele text never enters this repository" is false: CC BY-SA items and passages are committed | `README.md:292`, `docs/external-validity.md:24` |
| R-18 | medium | NFR-PORT-02/CL-REL-02: no clean-checkout quickstart transcript in `docs/results.md` | `docs/requirements.md:409` |
| R-19 | low | §10.7 limitations are not verbatim in `docs/results.md` (they are in `eval-methodology.md`) | `docs/results.md:465` |
| R-20 | low | `citation_recall` note in artifacts describes chunk recall while code computes statement-level ALCE recall | `src/answerdesk/eval/metrics.py:468` |
| R-21 | low | Two different "coverage" denominators; README gloss "84% of turns answered" is true only of answerable turns | `README.md:244`, `docs/results.md:335` |
| R-22 | low | Absolute local path committed in an artifact and a generated doc | `reports/external-p0/summary.json:645` |
| R-23 | low | README promises Python 3.13 and macOS; CI covers 3.11/3.12 on Linux+Windows only | `README.md:12` |
| R-24 | low | `docs/reproducibility.md` says the demo writes `reports/demo-t12/`; it writes `reports/demo/` | `docs/reproducibility.md:77` |
| R-25 | low | Dead public method `uncited_claim_indexes` (never called, not tested) | `src/answerdesk/generate/citations.py:92` |
| R-26 | low | NFR-DET-03 asks for two workflows with recorded durations; there is one workflow with two jobs and no durations | `docs/requirements.md:407` |
| R-27 | low | `/healthz` and traces expose absolute host paths (no secrets, but host layout) | `src/answerdesk/service/runtime.py:377` |
| R-28 | high | Tests fail on a clean checkout/CI because they read untracked artefacts: the consistency test needs `data/corpus/chunks.jsonl`, the locale-purity suite's guard checks the wrong file, and the mutation-audit test needs `docs/review-report.md` (fixed by `795c73f`) | `tests/unit/eval/test_consistency.py:305`, `tests/integration/retrieval/test_locale_purity.py:34`, `tests/smoke/test_requirements_evidence.py` |
| R-29 | medium | The new gates (traceability, numbers, licences, coverage) run in `scripts/ci-offline.ps1` but **not** in `.github/workflows/ci.yml`, so GitHub Actions does not enforce them | `.github/workflows/ci.yml` |
| R-30 | medium | Residual limitation: spelled-out numbers ("сто двадцать") are invisible to the digit guard; only digit-bearing claims are checked | `src/answerdesk/generate/numeric.py` (behaviour), documented from this audit |
| R-31 | low | Residual limitation: a numeric-guard rejection is reported to the operator as `CITATION_FAILED`; the numeric detail only lives in `numeric_violations_pre_repair` on the trace | `src/answerdesk/generate/pipeline.py:337-339` |
| R-32 | low | 11 acceptance-test markers (`AT-CNS-01…04`, `AT-GEN-13…19`) exist in the tree but have no §16 row; the checker reports them as a note, not a failure | `docs/requirements.md` §16 / CC-2026-10-21-c note |
| R-33 | low | `pytest -q -m smoke` (the CI fast-gate command) fails when the `api` extra is installed: `test_base_install_does_not_import_heavy_dependencies` asserts on process-global `sys.modules`, which an earlier-collected service conftest pollutes with `fastapi` | `tests/smoke/test_package.py:60` |

### 6.2 Round-2 status (post-fix verification, HEAD `2dfb713`)

| Finding | Round-2 status | Evidence |
|---|---|---|
| R-01 traceability gate | **resolved (verified)** | `scripts/check-traceability.py` exists, exits 0: 121 rows, 121/121 AT ids marked, 154 path references checked; `scripts/check-numbers.py` and `scripts/check-licenses.py` run; requirements CC-2026-10-21-a/c/d record the path fixes and marker evidence. Residual wiring gap → R-29 |
| R-02 NFR-LAT-01 miss | **resolved (verified)** | `docs/results.md` §"NFR-LAT-01 verdict": p50 2618.22 / p95 14116.76 ms vs 1500/4000 ms, **MISS**, with "not amended" wording; requirements CC-2026-10-21-b. README still does not link the verdict (cosmetic) |
| R-03 CRLF provenance | **resolved (verified)** | `build-results.py --check` and `build-external.py --check` exit 0 on a clean LF `git archive` copy of `2dfb713`; the provenance block now publishes LF digests and says so |
| R-04 numeric guard | **resolved (verified)** | New `tests/integration/generation/test_verifier_numeric.py` (+281 lines) and `tests/unit/generate/test_numeric_guard_contract.py`; mutation M6 re-run on a clean copy: baseline 69 passed, mutated **19 failed** |
| R-05 NFR-SCALE-01 | **resolved (verified)** | committed `reports/scale/last-run.{json,md}`: 3,000 docs / 26,068 chunks, index 7.7 s, retrieval p95 11.8 ms, peak RSS 0.487 GB, 3 runs |
| R-06 missing markers | **resolved (verified by checker)** | 121/121 marked; UI ids traced in `tests/smoke/test_ui_traceability.py`, repo-level ids in `tests/smoke/test_requirements_evidence.py`. Residual unregistered markers → R-32 |
| R-07 coverage gate | **partially resolved** | Gate added to `scripts/ci-offline.ps1` (`--cov-fail-under=70`); still absent from `.github/workflows/ci.yml` → R-29 |
| R-08 dependency/licence | **partially resolved** | `scripts/check-licenses.py` exits 0 over 57 installed distributions (licence half); the networked `pip-audit` vulnerability half is explicitly a release step, not a CI claim (requirements NFR-SEC-05 honesty note, docs/decisions.md CC-e) |
| R-09/R-10 XSS/a11y | **open** | No XSS DOM test and no axe/pa11y run were added; `test_ui_traceability.py` only asserts that the vitest cases exist. Both remain medium |
| R-11…R-27 | **open** | Not addressed in this round (see §8) |
| R-28 clean-checkout suite | **open — high at final HEAD** | Clean `git archive` of `2dfb713`: 2 failed / 1,526 passed / 2 skipped; `2c0c9dc`: 2 failed / 1,546 passed / 2 skipped / 6 errors; **final `795c73f`: 1 failed / 1,547 passed / 2 skipped / 6 errors**. Committing this report removed the mutation-audit failure; the remaining failure is the consistency replay test (missing git-ignored `data/corpus/chunks.jsonl`) and the six `test_locale_purity` errors, whose `_require_inputs()` guards on `manifest.json` instead of `chunks.jsonl`. All pass in the working tree |
| R-29 CI wiring | **open — medium** | `.github/workflows/ci.yml` still has no step for the checkers or coverage (verified at `795c73f`); only `scripts/ci-offline.ps1` runs them |
| R-30/R-31 residual numeric limitations | **new — medium/low** | Confirmed by probe/code read; must stay in the report even after R-04 is closed |


### R-01 — the traceability gate does not exist (high)

**Round-2 status: resolved (verified).** The script now exists and exits 0 on `2dfb713`
(121 rows · 121/121 AT ids · 154 path references); `check-numbers.py` and `check-licenses.py` were
added and run. Resolved in requirements CC-2026-10-21-a/c/d. Residual: the checkers are not wired
into the GitHub Actions workflow (R-29) and 11 markers remain outside §16 (R-32).

**Problem (round 1, for the record).** `docs/requirements.md` (NFR-REP-03, CMD-T, §17.3), `CONTRIBUTING.md:58` and
`docs/design-decisions.md:130` all state that `scripts/check-traceability.py` parses §16 and asserts
every `AT-` marker and module path. The file does not exist in HEAD or anywhere in history; running
the documented command exits 2 (`can't open file`). `tests/smoke/test_traceability.py:4` describes
the script as existing "(t12)" but implements only a 12-id subset. Independently reimplementing the
check shows the rot the missing gate would have caught: 21/121 AT ids have no marker, and three
module paths named in the matrix do not exist (`policy/risk.py`, `policy/cost_model.py`,
`scripts/check-traceability.py`). The same commit has no `REP-01` number-grep script either.

**Required fix.** Add `scripts/check-traceability.py` implementing exactly the §16 contract (parse
the matrix, assert every AT id appears as `# AT-…` in `src|tests|scripts`, assert every module path
resolves, exit non-zero listing gaps), wire it into CI and §17.3, add the missing 21 markers, and
fix or amend the three module paths. Add the NFR-REP-01 number-check script or amend that
requirement.

### R-02 — NFR-LAT-01 is missed and the miss is not reported (high)

**Round-2 status: resolved (verified).** `docs/results.md` now carries an "NFR-LAT-01 verdict"
section (p50 2618.22 / p95 14116.76 ms, generation stage, judge excluded, verdict **MISS**, requirement
explicitly *not* amended) and requirements CC-2026-10-21-b records the annotation. The README's
operational table still does not link the verdict; that is cosmetic and can ride with the docs pass.

**Problem (round 1, for the record).** NFR-LAT-01 (MUST, G-REL) requires p50 ≤ 1,500 ms and p95 ≤ 4,000 ms server-side,
arrival → final byte, API provider, n ≥ 200. The committed primary-arm measurement is generation
p50 **2,618.22 ms**, p95 **14,116.76 ms** (n = 200; `reports/bench/ops/ops-primary.json`), and the
total including the offline judge is 7,748/55,107 ms. `docs/results.md` and the README print these
numbers without stating that the requirement target is missed, and no requirement amendment is
recorded. `docs/ops.md:482` still lists ≤4 s / ≤1.5 s as the operating target with no measured value
or deviation note. The release rule in §17.3 ("every G-REL threshold is met or the requirement was
formally amended") is therefore not satisfied.

**Required fix.** Either (a) report the LAT-01 verdict next to the operational table (target vs
measured, both p50/p95, and the judge-excluded basis) and amend/annotate NFR-LAT-01 with the
measured reality and an owner, or (b) produce a run that meets the target (smaller/local model,
streaming TTFT measurement) and record it. Do not leave a MUST target silently missed.

### R-03 — provenance hashes are platform-dependent (high)

**Round-2 status: resolved (verified).** Provenance hashing is canonical-LF and the published block
says so; `build-results.py --check` and `build-external.py --check` both exit 0 on a clean LF
`git archive` copy of `2dfb713`. Commits 54ed1b1 / fa3d6ba / 2dfb713.

**Problem (round 1, for the record).** `scripts/build-results.py` hashes raw file bytes (`_sha256`, lines 53–54 and 172).
The committed `docs/results.md` was generated on a working tree where `evals/golden/v1/manifest.json`,
`evals/splits/v1.json`, `configs/pricing.yaml` and `reports/bench/summary.json` had CRLF line endings
(`git ls-files --eol` shows `i/lf w/crlf`), so the hashes in the provenance block are CRLF hashes. On
a clean `git archive` checkout (LF, which is what Linux/CI and every fresh clone get) the same
command regenerates different hashes and `--check` fails ("docs/results.md is out of date"). This
breaks the README's "determinism" claim, CL-REL-02, and NFR-REP-02's "hashes stable across re-runs".
The fact that it passes on this Windows checkout masks the defect.

**Required fix.** Normalise before hashing (`read_bytes().replace(b"\r\n", b"\n")`, or hash a
canonical JSON projection), regenerate `docs/results.md`, and add a test that runs the `--check`
path on LF-normalised inputs (or in a Linux CI job). Document the canonicalisation in
`docs/reproducibility.md`.

### R-04 — the numeric guard is not tested where it is enforced (high)

**Round-2 status: resolved (verified).** New verifier-level and pipeline-level tests
(`tests/integration/generation/test_verifier_numeric.py`, 281 lines;
`tests/unit/generate/test_numeric_guard_contract.py`, 131 lines) pin the guard where it is enforced.
The original mutation M6 was re-run on a clean copy of `2dfb713`: baseline 69 passed, mutated
**19 failed**. Residual limitations R-30 (spelled-out numbers) and R-31 (reason code) remain.

**Problem (round 1, for the record).** FR-GEN-12 / INV-04 / CR-9 require that a number or date in a claim appears in a cited
passage, enforced deterministically in `LexicalVerifier.check`. Mutation M6 (`elif not numeric_ok:` →
`elif False:`) leaves the entire generation test set green (41/41), while an independent probe shows
the behaviour change: a claim with 1.000 lexical coverage but `999` where the evidence says `500`
returns `supported` instead of `unsupported`. `test_numeric_guard.py` only tests
`numeric_violations()`; `test_repair.py`'s wrong-number fixture is already `unsupported` through the
coverage branch. NFR-QUAL-03 requires zero undetected breakages; this one is safety-relevant because
it is exactly the "plausible wrong number displayed" failure the guard exists to prevent.

**Required fix.** Add an integration test that builds a claim whose only defect is a mismatched
number with high lexical coverage and asserts the verifier returns `unsupported` (and the pipeline
drops/escalates). Keep `test_numeric_guard.py` as the unit layer.

### R-05 — NFR-SCALE-01 is not evidenced (high)

**Round-2 status: resolved (verified).** Committed `reports/scale/last-run.{json,md}` and
`smoke-run.{json,md}` (commit 8e248a1): 3,000 documents / 26,068 chunks, index build 7.7 s, retrieval
p95 11.8 ms (240 queries), peak RSS 0.487 GB, three runs with the spread recorded, hardware recorded,
CC0-1.0 corpus, git-ignored corpus path stated. Requirement satisfied.

**Problem (round 1, for the record).** NFR-SCALE-01 (MUST, G-REL) requires a recorded run at ≥3,000 docs / ≥20,000 chunks
with index build ≤10 min, peak RSS ≤4 GB and retrieval p95 ≤500 ms. There is no scale artifact in
`reports/`, no scale harness output, and no scale section in `docs/results.md`; `docs/ingestion.md:593-595`
defers it to t11, whose committed evidence package contains no such run. The matrix maps it to
CMD-R and AT-SCALE-01, and that AT marker is absent.

**Required fix.** Run `ANSWERDESK_INGEST_DOCS_PER_LOCALE=500 answerdesk ingest --demo`, build the
index, measure build time/RSS/retrieval p95 on the reference machine, commit the artifact under
`reports/scale/`, add the `AT-SCALE-01` marker and a results.md row; or formally amend
NFR-SCALE-01 to a SHOULD/deferred status with an owner.

### R-06 — 21 acceptance-test markers absent (medium)

**Round-2 status: resolved (verified by the checker).** All 121 AT ids now have markers; the UI ids
are traced through `tests/smoke/test_ui_traceability.py`, the repository-level ids through
`tests/smoke/test_requirements_evidence.py`, and `check-traceability.py` exits 0. Residual: 11
markers still have no §16 row (R-32).

**Problem (round 1, for the record).** The missing markers are listed in §1.2. Most are genuine evidence gaps: UI ATs
(`AT-UI-01…11`), `AT-A11Y-01`, `AT-SEC-04`, `AT-LAT-04`, `AT-QUAL-02`, `AT-QUAL-03`, `AT-REP-01/02/04`,
`AT-SCALE-01`, `AT-SEC-05`, `AT-DOC-02`. Web tests exist and pass, but nothing in the repository ties
them to the requirements they satisfy, and the matrix is the stated traceability contract.

**Required fix.** Add the markers at the tests/sections that actually prove each requirement, or
amend the matrix where a requirement is intentionally proven by a checklist item (then say so in the
row). This should be automated by R-01.

### R-07 — coverage threshold is not enforced (medium)

**Round-2 status: partially resolved.** A `--cov … --cov-fail-under=70` step now exists in
`scripts/ci-offline.ps1`; it is still absent from `.github/workflows/ci.yml` (R-29). The reported
aggregate differs by run/test set — the captain's message says 78%, requirements CC-2026-10-21-d
records 80.7% — and both exceed the threshold; an independent partial-suite run on a clean copy gave
retrieval 90 / policy 89 / cache 83 / **eval 67** (eval below threshold in that subset), which is why
the full-suite configuration and a real CI step matter.

**Problem (round 1, for the record).** NFR-QUAL-02 (MUST) requires a coverage report in CI and a ≥70% threshold for
`retrieval`, `policy`, `cache`, `eval`. `.github/workflows/ci.yml` runs `pytest -q` with no
`--cov`, `pyproject.toml` has no `fail_under`, and no CI job produces a coverage artifact. Measured
now: 90/89/84/70% (eval is exactly at the line), so the gate is needed to keep it there.

**Required fix.** Add `--cov=src/answerdesk/{retrieval,policy,cache,eval} --cov-fail-under=70` (or
per-package gates) to the full CI job, upload the report, and add the `AT-QUAL-02` marker.

### R-08 — dependency and licence audit missing (medium)

**Problem.** NFR-SEC-05 requires a committed dependency-audit report with 0 high/critical findings;
NFR-REP-04 requires a licence report and Apache-2.0 attribution in `NOTICE`. There is no audit
artifact, no `NOTICE`, no licence-report script (`CMD-Z` names "the license report script", which
does not exist), and `pip-audit` cannot run in this offline sandbox. Direct dependencies look
permissive (MIT/BSD/PSF/Apache-2.0), but the bundled Apache-2.0 components in matplotlib alone make
the "0 disallowed licences" claim unverified.

**Required fix.** Add a `pip-audit` job (or a committed report with a date and environment) and a
licence-report script + `NOTICE` listing Apache-2.0 components; record the results in
`docs/reproducibility.md` and add `AT-SEC-05`/`AT-REP-04`.

### R-09 — no XSS DOM test (medium)

**Problem.** NFR-SEC-04 requires "vitest DOM test with script payload passes". There is no such
test (`web/src/**` has no XSS/`<script>` fixture). The risk is mitigated because `AnswerCard.tsx`
renders text through React text nodes (no `dangerouslySetInnerHTML`), but the requirement's test is
absent and `AT-SEC-04` has no marker.

**Required fix.** Add a vitest DOM test that feeds a `<script>`/`onerror` payload through a mocked
`/v1/chat` answer and asserts it is rendered as text (no element/execution), and mark `AT-SEC-04`.

### R-10 — no automated accessibility check (medium, SHOULD)

**Problem.** NFR-A11Y-01 requires "automated a11y check with 0 critical violations" and a committed
a11y report. No axe/pa11y dependency, test, or report exists; accessibility is implemented (landmarks,
aria attributes, focus handling) but not verified by tooling.

**Required fix.** Add `vitest-axe`/`axe-core` to a component test, commit the report, and add
`AT-A11Y-01`.

### R-11 — `answerdesk demo --smoke` is a stub (medium)

**Problem.** FR-OPS-06 and CMD-U require a one-command offline demo; the CLI command exists but
exits 2 with "implemented by task t14", and §17.3 includes it in the release acceptance sequence.
`scripts/demo.ps1` does work (verified), so the README's primary path is fine; the documented
CLI path is not.

**Required fix.** Wire `demo --smoke` to the same steps as `scripts/demo.ps1` (owner t14) and add
`AT-OPS-06`; until then mark CMD-U as pending in §17.3.

### R-12 — committed offline-smoke evidence is stale (medium)

**Problem.** `reports/offline-smoke/` was last committed in the initial scaffold (`de14a71`) against
`corpus-ab58e8c0f5cf506b` with 360 cases and digest `69dd478a…`. The documented command now produces
`corpus-77d7d081813738a6`, 530 cases, digest `dfbb4924…` and dirties eight committed files. A reader
following the README reproduces different bytes from the committed evidence.

**Required fix.** Regenerate and commit `reports/offline-smoke/` with the current corpus (or move it
to a clearly-labelled archived run and stop committing it), and state in the README that the command
rewrites the artifact.

### R-13 — generated results doc misattributes the secondary arm and reports a zero duration (medium)

**Problem.** `scripts/build-results.py:311` hardcodes "Generator family **qwen**" and the primary
model names for every arm section, so the `secondary` section of `docs/results.md:250` says the qwen
model produced answers that were actually generated by `deepseek-flash` (the header table at line 22
correctly says deepseek-flash). The run header also reports `duration | 0.0 s` for a multi-hour phased
run. Both are generated defects, not typo-level, because the document is advertised as machine-built.

**Required fix.** Render the family/model sentence from `summary.models[arm]` and either measure
duration or print "not measured (phased execution)". Regenerate the doc and re-run `--check`.

### R-14 — architecture document is partly stale (medium)

**Problem.** `docs/architecture.md` §12 "Skeleton status" still describes the t3 skeleton: modules
are "placeholder `__init__.py` only", CLI bodies "exit 2", the repo map contains only
`contracts/config/errors/logging/cli/testing`. The implementation is complete (169 source files in
the mypy scope; full service and console). §6's storage table also describes an SQLite FTS5 lexical index and
`data/index/vectors.npy`; the implementation uses `bm25.json`/`dense.json` under a content-addressed
directory and no FTS5 code exists. A release reader is misled about both the status and the storage
design.

**Required fix.** Replace §12 with a current status table (or move it under a clearly historical
heading and link the release status in `docs/results.md`), and update §6 to the implemented index
layout.

### R-15 — documented profile commands are invalid (medium)

**Problem.** Five documented commands put `--profile` before the subcommand, but `--profile` is a
per-command option, not an app-level one: `answerdesk --profile local config check` fails with
`No such option: --profile` (verified). Affected: `README.md:134-135`, `README.ru.md:127-128`,
`docs/architecture.md:369`, `docs/generation.md:71`. The correct form
(`answerdesk config check --profile local`) works. This violates CL-DOC-01 ("zero invented flags").

**Required fix.** Move `--profile` after the subcommand in all five places (or add a real global
option), then re-run the documented commands during the release pass.

### R-16 — the README's proxy workaround breaks three tests (medium)

**Problem.** With `NO_PROXY='*'` in the environment — exactly what `README.md:111-113` tells Windows
users to prefix commands with, and what the offline scripts assume — these tests fail because they
assert the ambient-proxy fallback path and the environment makes the proxy usable:
`tests/unit/index/test_api_proxy.py::test_httpx_alone_still_raises_on_this_proxy`,
`…::test_build_http_client_falls_back_when_the_ambient_proxy_is_unusable`,
`tests/unit/retrieval/test_reranker_proxy.py::test_reranker_client_is_built_lazily_through_the_proxy_tolerant_factory`
(`KeyError: 'trust_env'`). Without `NO_PROXY` they pass (verified both ways). CI does not set it, so
CI is green while the documented local path is red.

**Required fix.** Make the tests set/clear the proxy environment explicitly (`monkeypatch`) instead
of inheriting it, or have `scripts/ci-offline.ps1`/README clear `NO_PROXY` for the test step.

### R-17 — "upstream data never vendored" is inaccurate (medium)

**Problem.** `README.md:292` says "upstream data never vendored" and
`docs/external-validity.md:24` says Belebele's "text never enters this repository or these reports —
only item ids, aggregate metrics and counts leave the adapter". In fact
`data/bench/answerdesk-bench/v0.1/belebele/qa/*.jsonl` contains 150 verbatim Belebele
questions+options per language and `belebele/retrieval/*/corpus.jsonl` contains 488 FLORES passages
per language (CC BY-SA 4.0). The commit is licence-compliant — `ATTRIBUTION.md`, the licence text,
modification notes and per-item provenance all ship — but the claim is false, and it understates the
share-alike obligations a redistributor inherits.

**Required fix.** Reword both places: upstream *archives* are not vendored; a curated CC BY-SA 4.0
subset with attribution is committed under `data/bench/…` (or stop committing it, which would be a
larger change), and point to the dataset card.

### R-18 — no clean-checkout transcript (medium)

**Problem.** NFR-PORT-02 requires "clean-checkout quickstart transcript in `docs/results.md`" and
CL-REL-02 requires a committed transcript. `docs/results.md` contains no transcript and no other
document does. This audit executed the quickstart end-to-end (§5) but did not add a transcript to the
release package (that is a documentation task, not the auditor's).

**Required fix.** Add a short transcript block (commands + key output lines, captured on a fresh
clone) to `docs/results.md` or `docs/reproducibility.md`, and point CL-REL-02 at it.

### R-19 — §10.7 limitations are not verbatim in results.md (low)

**Problem.** §10.7 says the ten limitations "must appear verbatim in `docs/results.md` and
`docs/eval-methodology.md`". `eval-methodology.md:326-343` has them; `docs/results.md:465-477` has a
different eleven-item list (more specific, but not the normative wording). CL-HON-01 checks for the
verbatim list.

**Required fix.** Either include the §10.7 list verbatim in the generated results doc (in addition
to the run-specific list) or amend §10.7/CL-HON-01 to reference the run-specific list.

### R-20 — citation-recall note contradicts its implementation (low)

**Problem.** `src/answerdesk/eval/metrics.py:468` describes `citation_recall` as "fraction of
required evidence chunks actually cited", while `eval/runner.py:428-445` computes the statement-level
ALCE definition (supported+cited claims / claims) and the README describes it that way. The note is
carried into `reports/bench/answer/primary/cells.json`. Additionally, per-claim support data is not
committed, so the 0.211 value is traceable to the aggregate artifact but not independently
recomputable.

**Required fix.** Correct the note; optionally commit a per-claim support column so the metric can
be recomputed from `turns.csv`.

### R-21 — coverage denominators are ambiguous (low)

**Problem.** `docs/results.md:223` reports "coverage 0.841 (n=176)" (answerable coverage), while
line 335 reports "test coverage 0.740 [0.675, 0.796]" (all turns). README:244 glosses 0.841 as
"84% of turns answered without a human", which is only true of answerable turns; overall it is 74%.
Both numbers are correct, but the label is ambiguous and the gloss is misleading.

**Required fix.** Use distinct names ("answerable coverage" vs "overall coverage") everywhere and
fix the README gloss.

### R-22 — absolute local path in committed artifacts (low)

`reports/external-p0/summary.json:645` and `docs/external-validity.md:114` contain
`D:/Documents/deepseek-harness/default-workspace/answerdesk/data/bench/…`. Not a secret, but it
leaks the author's machine layout and makes the doc look machine-specific.

**Required fix.** Store/print the dataset root as a repo-relative path.

### R-23 — support matrix overclaims (low)

`README.md:12` promises Python 3.11–3.13 and macOS; CI tests 3.11/3.12 on Ubuntu and Windows only.
`requires-python` allows 3.13, so the claim is plausible but unverified; macOS is not CI-verified at
all (NFR-PORT-02 only requires Windows+Linux).

**Required fix.** Either add 3.13 and macOS CI legs or narrow the README line.

### R-24 — demo output path mismatch (low)

`docs/reproducibility.md:77` says the demo writes `reports/demo-t12/`; `scripts/demo.ps1:24` uses
`reports/demo`. Trivial but it is a documented behavior claim.

### R-25 — dead method (low)

`generate/citations.py:92` `uncited_claim_indexes()` is never called and its mutation is not caught
by any test. Remove it or use it in the repair path with a test.

### R-26 — CI topology wording (low)

NFR-DET-03 asks for "two workflows, measured durations recorded". The repository has one workflow
with a fast and a full job (functionally equivalent) and no recorded durations.

**Required fix.** Amend the requirement wording or split the workflows and record durations.

### R-27 — host paths in health output (low)

`/healthz` returns absolute paths for the calibration artifact, cache thresholds and trace file
(e.g. `D:\Documents\…\reports\policy-smoke\calibration-artifact.json`). No secrets are exposed, but
a public-facing deployment would reveal host layout.

**Required fix.** Return repo-relative paths or basenames in the health payload (keep the full path
in logs if needed).

### R-28 — the suite is not green on a clean checkout (high, still open at final HEAD)

**Problem.** A full run on a clean `git archive` checkout with no proxy env is **2 failed, 1,526
passed, 2 skipped** at `2dfb713`, **2 failed, 1,546 passed, 2 skipped, 6 errors** at `2c0c9dc`
(after t44), and **1 failed, 1,547 passed, 2 skipped, 6 errors** at the final pushed HEAD `795c73f`.
Committing this report (`795c73f`) fixed the third bullet below; the other two causes remain. They
all read artefacts which exist only in a developer working tree:

* `tests/unit/eval/test_consistency.py::test_runner_replay_is_offline_and_writes_artefacts`
  ends in `CorpusUnavailableError: corpus artifacts are missing: …/data/corpus/chunks.jsonl`.
  `chunks.jsonl` is gitignored (derived ingest output); the test neither builds the corpus nor skips.
* `tests/integration/retrieval/test_locale_purity.py` (six tests, all fixture **errors**) documents
  "without either it skips rather than failing", but `_require_inputs()` checks
  `CORPUS / "manifest.json"` (committed) instead of `CORPUS / "chunks.jsonl"` (not committed). The
  module fixture then raises `CorpusMissingError: no chunks at …/chunks.jsonl`.
* ~~`tests/smoke/test_requirements_evidence.py::test_qual_03_mutation_audit_is_recorded` reads
  `docs/review-report.md`; it failed while the report was untracked.~~ **Fixed by `795c73f`** (the
  report is committed), though the test would still fail on any clone that strips documentation —
  a `pytest.skip` when the file is absent is still advisable.

All of them pass in the working tree (verified: 7 passed for locale-purity, 2 passed for the other
two in the repo), so a gate run there misses the problem. `scripts/ci-offline.ps1` runs `pytest -q`
before anything builds the corpus, so the local gate fails on a clean clone too; GitHub Actions
certainly does (its `pytest -q` step has no ingest step before it).

**Required fix.** (a) make the consistency test build its corpus fixture or `pytest.skip` with the
standard "checked-in corpus not built in this tree" reason; (b) fix `_require_inputs()` in the
locale-purity test to check `chunks.jsonl` (and catch `CorpusMissingError` for the fixture), so it
skips on a clean clone; (c) optionally make the mutation-audit test skip when the review report is
absent. Re-run the full suite on a clean clone afterwards — that is the only way this class of bug
stays fixed.

### R-29 — the new gates are not in the GitHub Actions workflow (medium, new in round 2)

**Problem.** `scripts/ci-offline.ps1` now runs `check-traceability.py`, `check-numbers.py`,
`check-licenses.py` and a `--cov-fail-under=70` step. `.github/workflows/ci.yml` is unchanged: it runs
ruff, format, mypy, `pytest -q`, load smoke, and imports — no checker and no coverage. NFR-REP-01,
NFR-REP-03, NFR-REP-04 and NFR-QUAL-02 are written as **CI** gates (`G-CI`); a pull request can
currently merge without them. The requirement text was adjusted to describe the licence half as an
offline gate (CC-2026-10-21-e), but the traceability/number/coverage half is still claimed as CI.

**Required fix.** Add the three checker steps and the coverage step to the `fast`/`test` jobs in
`.github/workflows/ci.yml` (or make the workflow invoke `scripts/ci-offline.ps1`), and update
`tests/smoke/test_ci_workflow.py` to assert the steps exist. Without that, the local script is a
convenience, not an enforced gate.

### R-30 — spelled-out numbers are invisible to the numeric guard (medium, residual limitation)

**Problem.** INV-04/FR-GEN-12 canonicalise digit forms; a claim that writes a number in words is
not checked against the cited evidence. Reproduced directly:

```text
numeric_violations("Комиссия за перевод составляет сто двадцать рублей.",
                   ["Комиссия за перевод составляет 120 рублей."])  ->  []
canonical_numeric_tokens("сто двадцать")  ->  set()
canonical_numeric_tokens("120")           ->  {'120'}
```

So a model that renders the right amount as words is safe, but a model that renders a **wrong**
amount as words passes the guard that exists to prevent exactly that. This is a known limitation,
not a regression; it must stay in the release notes next to the R-04 fix.

**Required fix (documentation now, code later).** State the limitation in `docs/generation.md` /
`docs/math.md` next to the guard's contract (and in the README limitations if the guard is described
as covering "every number"). The code fix is a numeral-word lexicon per locale plus a
claim-side extractor; it is a feature, not a patch, and should be scheduled explicitly.

### R-31 — a numeric rejection is surfaced as `CITATION_FAILED` (low, residual limitation)

**Problem.** When no claim survives verification the pipeline chooses `CITATION_FAILED` if any check
has `numeric_ok == False` or `quote_ok == False`, else `VERIFIER_REJECT`
(`src/answerdesk/generate/pipeline.py:337-339`). The operator therefore sees the same reason code for
a wrong number, a broken quote and a citation-integrity failure; the numeric detail exists only as
`numeric_violations_pre_repair` on the trace. Requirements FR-GEN-12 permits
"drops the claim or escalates with `CITATION_FAILED`", so this is compliant, but it makes the
operator-facing explanation less useful than the trace.

**Required fix.** Either add a `NUMERIC_MISMATCH` reason code (and update the closed enum +
docs/requirements reason-code table) or document in `docs/escalation.md` that numeric rejections are
reported as `CITATION_FAILED` with the count on the trace.

### R-32 — 11 markers sit outside the §16 matrix (low, new in round 2)

**Problem.** `check-traceability.py` passes but prints: "11 marker id(s) are not in §16 (stale marker
or typo): AT-CNS-01…04, AT-GEN-13…19". The new consistency suite and the experimental query-rewriter
have acceptance markers without requirement rows, and the checker treats them as a note. The
requirements document acknowledges this as an open item (CC-2026-10-21-c note), so it is recorded
rather than hidden — but the matrix is no longer a complete map of the repository's evidence.

**Required fix.** Add §16 rows (or a §16.12 experimental/out-of-scope table) for these behaviours, or
mark the suites explicitly as experimental without requirement IDs. Make the checker fail when a
marker is outside the matrix once the rows exist.

### R-33 — the smoke subset is order-dependent (low, new in round 3)

**Problem.** `.github/workflows/ci.yml`'s fast job runs `uv run pytest -q -m smoke` with the dev extra.
With the `api` extra installed (the documented setup for the service), the same command fails:

```text
FAILED tests/smoke/test_package.py::test_base_install_does_not_import_heavy_dependencies
E  Left contains one more item: 'fastapi'
```

`test_base_install_does_not_import_heavy_dependencies` inspects the process-global `sys.modules`, and
collecting the service tests imports `fastapi` via `tests/integration/service/conftest.py`
(`pytest.importorskip`). The test therefore passes only when the service tests are not collected in
the same process (CI's dev-only fast job) and fails for a developer who followed the README's
`uv sync --extra api` step. It is not the cause of R-28 (the full suite happens to pass), but it is
the same class of environment-dependent gate.

**Required fix.** Run the base-install check in a subprocess (`python -c "import answerdesk; …"`)
instead of inspecting the pytest process's module table, or mark it to run in its own process, so the
result does not depend on what else was collected.

---

## 7. What held up (worth keeping)

* **The round-2 fixes are real, not papered over.** The traceability gate now runs and reports
  121/121; the LAT-01 miss is printed as MISS next to the target it misses; provenance is canonical
  LF and survives a clean checkout; the numeric guard is pinned by tests that fail when the branch is
  removed (19 failures); and the scale run is a committed artifact with three repetitions and its
  hardware/config recorded.
* **Numbers are honest.** Every recomputable headline value matches its artifact; CIs are real
  Wilson/bootstrap intervals (recomputed independently); the negative calibration result is stated
  in the headline and in `summary.json`; zh is a no-claim; the secondary arm is labelled not
  comparable; the free tariff and 0.0 USD are explained rather than hidden.
* **Determinism.** Two offline eval runs give the same digest; the corpus rebuild is byte-stable;
  the hashing embedder is process-independent; `build-results.py --check` works on the canonical
  checkout.
* **Security boundaries are real, not asserted.** Live PII and injection probes both stop before a
  provider call and leave no raw value in traces; the mutation sweep proves the PII detector, the
  entity guard, the conformal quantile, the fusion degeneracy rule and heading structure are
  load-bearing tests.
* **The service and console are genuine.** SSE event order and trace ids verified live; cache hit
  verified; 61 web tests and a production build pass; screenshots show real service data with a
  trace id and model/version headers.
* **Lint/types are clean on the committed bytes** (ruff check, ruff format, mypy over 169 files).
* **Documentation is unusually honest about limitations** (risk control, judge κ, synthetic corpus,
  citation recall, disabled rewriter) even where this audit found inaccuracies.

---

## 8. Required actions before the verdict can be `pass`

**Closed and verified (R-01, R-02, R-03, R-04, R-05, R-06).** The review report itself was committed
in `795c73f`, which removed one of the three R-28 failures.

**Still required:**

1. **R-28 (high):** make the full suite green on a clean checkout — fix the locale-purity
   `_require_inputs()` guard to test `chunks.jsonl`, and make the consistency replay test skip or
   build the missing corpus. This is the only open high finding and the only thing between this
   revision and a `pass`.
2. **R-29 (medium):** put `check-traceability.py`, `check-numbers.py`, `check-licenses.py` and the
   coverage step into `.github/workflows/ci.yml`; a local PowerShell script is not a CI gate.
3. **R-30 / R-31 (residual limitations):** document spelled-out numbers as outside the digit guard
   and state that numeric rejections surface as `CITATION_FAILED` with the count on the trace (or add
   a dedicated reason code). They are recorded in this report, not yet in `docs/`.
4. **R-33 (low):** make the base-install check run in a subprocess so `pytest -m smoke` is
   order-independent with the `api` extra installed.
5. R-09/R-10: add the XSS DOM test and an automated a11y check (still open).
6. R-11: wire `demo --smoke` (t14) or mark CMD-U pending in §17.3.
7. R-12/R-13/R-14/R-15/R-17: regenerate/repair the documentation and the committed replay artifact.
8. R-16: make the proxy tests environment-independent.
9. R-18: commit the clean-checkout transcript.
10. R-32: register (or explicitly exclude) the 11 out-of-matrix markers.

Low findings R-19…R-27 may be fixed in the same pass or explicitly deferred with an owner; they do
not on their own block a verdict.

---

## 9. Audit trail (reproduction)

Everything below was run from `answerdesk/` on the machine described in §0.

```powershell
uv sync --extra dev
uv sync --extra dev --extra api --frozen
uv run answerdesk ingest --demo                      # 7.8 s, 810 docs / 7,229 chunks
uv run answerdesk index --demo
uv run answerdesk ask --fake "Сколько стоит обслуживание карты?"
uv run answerdesk eval --synthetic --provider fake --report reports/offline-smoke
pwsh -File scripts/demo.ps1
uv run --extra api answerdesk serve --fake --port 8765 --healthcheck
uv run --extra api answerdesk serve --export-openapi $env:TEMP/openapi.json
uv run answerdesk eval external --list-suites
uv run answerdesk demo --smoke                       # round 1: stub (R-11)
uv run python scripts/check-traceability.py          # round 1: missing (R-01); round 2: exit 0, 121/121
uv run python scripts/check-numbers.py               # round 2: exit 0 (1,834 quantities)
uv run python scripts/check-licenses.py              # round 2: exit 0 (57 distributions)
uv run answerdesk --profile local config check       # round 1: fails (R-15)
uv run answerdesk config check --profile local       # works
uv run python scripts/build-results.py --run reports/bench --out docs/results.md --check
uv run python scripts/build-external.py --run reports/external-p0 --check
uv run pytest -q
uv run ruff check . ; uv run ruff format --check . ; uv run mypy src/answerdesk
cd web ; npm run test ; npm run build
```

Round-3 verification at the final pushed HEAD `795c73f` (LF `git archive` copy, no proxy env):

```powershell
# from the copy, with PYTHONPATH=<copy>/src, ANSWERDESK_REQUIRE_API=1
python -m pytest -q          # 1 failed, 1547 passed, 2 skipped, 6 errors   -> R-28
python -m pytest -q tests/unit   # 1 failed, 1046 passed, 1 skipped         -> the CI fast gate fails too
```

```powershell
# in the repository working tree (has data/corpus/chunks.jsonl), no proxy env
pwsh -File scripts/ci-offline.ps1
# ==> uv sync --extra dev --extra api ... ==> pytest -q
# offline gate: OK
# 1555 passed, 1 skipped in 262.54s
# Required test coverage of 70% reached. Total coverage: 78.40%
```

Round-2 verification on clean `git archive` copies of HEAD `2dfb713` and `2c0c9dc` (LF, no proxy env):

```powershell
# from the copy, with PYTHONPATH=<copy>/src
python scripts/check-traceability.py                     # exit 0
python scripts/build-results.py --run reports/bench --out docs/results.md --check   # exit 0
python scripts/build-external.py --run reports/external-p0 --check                  # exit 0
python -m pytest -q                                      # 2dfb713: 2 failed, 1526 passed, 2 skipped
                                                          # 2c0c9dc: 2 failed, 1546 passed, 2 skipped, 6 errors -> R-28
```

The final-HEAD failure is
`tests/unit/eval/test_consistency.py::test_runner_replay_is_offline_and_writes_artefacts`
(missing git-ignored `data/corpus/chunks.jsonl`); the six errors are all of
`tests/integration/retrieval/test_locale_purity.py`, whose skip guard checks `manifest.json` instead
of `chunks.jsonl`. The mutation-audit failure from the earlier rounds disappeared when this report
was committed in `795c73f`. All of them pass in the working tree.

Recomputations and probes were run as throwaway scripts under the session's `_tmp/t13/` (not
committed): AT-marker scan, provenance-hash comparison, Wilson intervals, per-language metric
recomputation, retrieval pooling, external-summary extraction, vacuous-test AST scan, the mutation
harness (8 mutations, baselines included), the round-2 M6 re-run against `2dfb713`, and the
numeric-guard/word-numeral probes. Clean LF copies of HEAD were created with
`git archive HEAD | tar -x` for platform-independence checks; mutations were applied only inside
those copies.

### Audit limitations

* No live provider run (no keys, no network): live-arm honesty was judged from committed artifacts.
* No dependency-vulnerability scan (network blocked). The licence half now runs
  (`scripts/check-licenses.py`, 57 distributions, all permissive); the vulnerability half is
  documented as a networked release step (see R-08 and docs/decisions.md CC-2026-10-21-e).
* The working-tree gate was run on the final HEAD `795c73f`; the clean-checkout suite was run on a
  `git archive` copy of the same commit. No further commit landed between the two runs.
