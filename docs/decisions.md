# Decision and repair register

Short, append-only records of decisions taken **after** the requirements were frozen, and of
repairs made to the release candidate. The product contract stays `docs/requirements.md`; this
file is the narrative the reviewer and the next maintainer need: what was wrong, what was
changed, and what evidence exists now.

Rules for this file: one row per decision or repair, each row names its evidence (a command, a
test, an artifact), and a row is never rewritten — a reversal gets a new row. Change-control
ids (`CC-…`) refer to `docs/requirements.md` §18.5; ADRs refer to `docs/design-decisions.md`.

| id | date | kind | decision / repair | evidence | status |
|---|---|---|---|---|---|
| CC-2026-10-21-a | 2026-10-21 | repair | The §16 traceability matrix pointed at six module paths that do not exist (`generate/prompts/`, `policy/risk.py`, `policy/cost_model.py`, `web/tests/`, bare `presets.py`, `NOTICE`) and at `scripts/check-traceability.py`, which had never been written. The paths now name the real modules, and the checker exists and is executed by the test suite | `uv run python scripts/check-traceability.py` → 121/121 marked, 0 missing paths; `tests/smoke/test_traceability.py` seeds both failure modes | done |
| CC-2026-10-21-b | 2026-10-21 | decision | **NFR-LAT-01 is missed and stays missed.** The policy decision is to report the miss rather than amend the target: the requirement is a MUST, and lowering it after seeing 2,618/14,117 ms would be exactly the kind of retro-fitting this project refuses. The verdict lives next to the latency table in `docs/results.md`, with the basis (generation stage, judge excluded, n=200) and what would change it | `docs/results.md` §"NFR-LAT-01 verdict"; `scripts/build-results.py --check` | done |
| CC-2026-10-21-c | 2026-10-21 | decision | **Evidence classes for the 21 unmarked requirements.** UI ids are traced to their vitest cases (the browser gate runs `npm run test`, the Python gate asserts the case exists); repository-level ids are traced to executable checks; where no honest automated proof exists in this checkout (a dedicated axe run, a 3,000-document scale run, a networked `pip-audit`), the matrix says so instead of asserting it | `tests/smoke/test_ui_traceability.py`, `tests/smoke/test_requirements_evidence.py`, §5 NFR rows | done |
| CC-2026-10-21-d | 2026-10-21 | decision | **Two new gates became executable rather than aspirational:** a number checker (NFR-REP-01) and a coverage threshold (NFR-QUAL-02). Both run inside `scripts/ci-offline.ps1`, so CI exercises them through `pytest`/the offline gate. The number checker's exception list is deliberately tiny and prints on every run | `scripts/check-numbers.py` (1813 quantities, 3 documented exception patterns); coverage measured 85.5 / 73.6 / 90.0 / 92.5 % (aggregate 80.7 %) | done |
| CC-2026-10-21-e | 2026-10-21 | decision | **Licence audit is split by what can be proven offline.** `scripts/check-licenses.py` audits the installed distribution tree from metadata, falling back to bundled licence text, and rejects GPL/AGPL/SSPL/BUSL/Commons-Clause/non-commercial terms. `pip-audit` (vulnerability half) needs network and remains a release step, documented in `docs/reproducibility.md` instead of being claimed as CI | `uv run python scripts/check-licenses.py --quiet` → all 57 installed distributions permissive | done |
| O-2026-10-21-1 | 2026-10-21 | open | Markers `AT-CNS-01…04` (consistency suite) and `AT-GEN-13…19` (experimental query-rewriter) have no §16 row: those behaviours are untracked by the matrix. They need a requirement amendment or removal — recorded, not deleted | `scripts/check-traceability.py` "note: marker id(s) are not in §16" | open |

## Why this file exists

The review of the release candidate (`docs/review-report.md`) found that the repository claimed a
traceability gate that did not exist, a latency requirement that was missed without being
reported, and a number-checking discipline that lived only in prose. Each of those is a
*process* failure, not a code failure: the code was fine, the bookkeeping was aspirational.

The response is mechanical rather than narrative: two scripts that fail, a coverage threshold
that fails, a verdict row that prints `MISS`, and this register so the next person can see why.
A claim that is only in a document is still a claim; a claim with a failing gate behind it is
evidence.
