# Judge calibration and the mechanical judge gate

The stack grades answers with an LLM judge (`rubric-v1`,
[`evals/judge/rubric_v1.md`](../evals/judge/rubric_v1.md)). An LLM judge is not ground truth:
until it has been shown to agree with humans *on this corpus*, every judged number is a
diagnostic, not a quality claim. This document specifies the human-calibration protocol, the
offline scoring tool and the mechanical gate that flips the label only when real labels clear
the threshold.

**Current state: `pending`.** The committed sheets contain no human labels, so
`reports/judge-gate/gate.json` reports `status=pending`, `judged_metrics_label=diagnostic`,
and every judged metric in [`docs/results.md`](results.md) and
[`reports/external-p0/report.md`](../reports/external-p0/report.md) stays labelled diagnostic.
No human labels are fabricated or committed.

## 1. Artefacts

| Path | Role |
|---|---|
| `evals/judge/rubric_v1.md` | the rubric the judge applies; the executable copy is `src/answerdesk/eval/judge.py:RUBRIC_TEXT` |
| `evals/judge/gate-policy.json` | languages, 150 items/language, label set, κ ≥ 0.6, pairwise agreement ≥ 0.7, bootstrap CI settings |
| `reports/external-p0/calibration/calibration-<lang>.csv` | the blank 150-item sheets per language, produced by `answerdesk eval external --run` |
| `reports/external-p0/calibration/PROTOCOL.md` | the short protocol shipped next to the sheets |
| `scripts/judge_gate.py` | scoring library: schema validation, Cohen's κ, bootstrap CIs, gate decision, artefact writer, `load_judged_label()` |
| `scripts/score-judge-calibration.py` | the CLI (`--score`, `--check`) |
| `reports/judge-gate/gate.json` | the current verdict: status, per-language κ with CIs, pooled metrics, reasons, input hashes |
| `reports/judge-gate/gate.md` | the same verdict as a table |
| `reports/judge-gate/disagreements.csv` | annotator A ≠ B rows for adjudication |
| `tests/unit/eval/fixtures/judge-gate/` | **synthetic** dry-run fixtures that prove score + gate transition; never human data |

## 2. Protocol

### 2.1 Sample

150 items per language (`ru`, `uz`, `tg`, `ky`, `zh`), sampled deterministically by item id
from the Belebele test split of `answerdesk-bench` (see
[`docs/benchmark.md`](benchmark.md)). The sheet references the licensed benchmark file in
`item_ref` rather than copying CC BY-SA question text into the report directory.

### 2.2 Labelling

1. Two annotators label every item independently, in a fixed order, as `supported`,
   `unsupported` or `unclear`. They see the question, the retrieved evidence and the answer —
   **not** the judge label, which stays hidden on the sheet until both passes are returned.
2. `human_label_a` and `human_label_b` are the two independent labels; `judge_label` is the
   pre-computed judge output and must not be edited.
3. If both annotators agree (`A == B`), that label is the reference. If they disagree, the
   row is adjudicated to `adjudicated_label` by a third pass; unresolved disagreements block
   the gate (they are never silently dropped).
4. Pairwise agreement is the share of items where `A == B`. Below **0.7**, the protocol calls
   for a rubric clarification round before κ is reported; the tool blocks the gate instead of
   publishing a κ computed on an unclear rubric.

### 2.3 What κ means here

* **Judge κ** — Cohen's κ between the judge and the adjudicated reference, binary
  (`supported` vs everything else), per language and pooled. This is the number the gate
  checks.
* **Pairwise κ** — Cohen's κ between annotator A and annotator B, on the three labels, per
  language and pooled. Reported for context; the gate checks raw pairwise agreement, as in
  the protocol above.
* Both come with a **95 % percentile bootstrap CI** (2,000 resamples, fixed seed
  `20261021`, so the interval is reproducible). κ is `null` when it is undefined (constant
  labels or fewer than two scored items); the gate treats that as not calibrated.

## 3. Commands

```bash
cd answerdesk

# 1. Produce the blank sheets (already committed; regenerate only with a new benchmark run)
uv run answerdesk eval external --run --report reports/external-p0

# 2. Two humans fill the sheets in place (do not edit judge_label)

# 3. Score: validates, computes κ + CIs, writes the gate artefacts
uv run python scripts/score-judge-calibration.py --score

# 4. Optional self-check: committed state is pending, fixtures prove open/blocked transitions
uv run python scripts/score-judge-calibration.py --check
```

`--score` reads `reports/external-p0/calibration/calibration-*.csv` by default; use
`--sheets <dir>`, `--out <dir>` and `--policy <file>` for a different sheet set (the dry-run
fixtures use their own 30-item policy). `--json` prints the full artefact.

The t19 helper `answerdesk eval external --score-calibration <dir>` remains available for a
quick judge-vs-reference κ; it does not validate the sheet set, compute pairwise agreement or
CIs, or decide the gate. **The canonical gate computation is
`scripts/score-judge-calibration.py`.**

## 4. Gate semantics

| Status | Meaning | `judged_metrics_label` |
|---|---|---|
| `pending` | sheets exist but carry no human labels (all cells blank) | `diagnostic` |
| `blocked` | labels incomplete, schema invalid, unresolved disagreements, pairwise agreement < 0.7, or judge κ < 0.6 in any language | `diagnostic` |
| `open` | every language has a complete, valid, adjudicated sheet; pairwise agreement ≥ 0.7; judge κ ≥ 0.6 per language | `calibrated` |

Additional artefact fields:

* `publishable` — `true` only for `open` **and** non-synthetic input;
* `synthetic` — `true` when the input carries the `SYNTHETIC-DRY-RUN` marker; synthetic runs
  can be `open` (that is how the transition is proven) but are never `publishable`;
* `inputs` — path + SHA-256 of every sheet the verdict was computed from;
* `policy` — a copy of the thresholds used, so a verdict cannot be separated from its policy;
* `reasons` — the exact blocking reasons, one per line, surfaced in `gate.md`.

Consumers must use the fail-closed helper:

```python
from judge_gate import load_judged_label

label = load_judged_label()   # "calibrated" only for an open, non-synthetic, publishable gate
```

A missing, corrupt or synthetic gate returns `diagnostic`. Reports must print `diagnostic`
next to any judged metric unless `load_judged_label()` returns `calibrated`.

**The flip is mechanical.** Filling the sheets and re-running `--score` is the only action
needed; there is no code change, allow-list or flag to flip. Conversely, deleting a sheet,
leaving a disagreement unadjudicated or dropping κ below 0.6 re-blocks the gate on the next
run.

## 5. Adjudication workflow

1. `--score` writes `reports/judge-gate/disagreements.csv` (rows where `A != B`), with
   `reason=unresolved` for rows that still have no `adjudicated_label`. `gate.md` reports the
   count per language.
2. The adjudicator reads the item in the benchmark file named by `item_ref`, decides the
   reference label, and writes `adjudicated_label` in the sheet (never `human_label_a/b`).
3. Re-run `--score`. When no unresolved disagreements remain and thresholds are met, the gate
   opens (real labels) or stays synthetic-open (dry-run).

## 6. Dry-run fixtures

`tests/unit/eval/fixtures/judge-gate/` contains three synthetic sheet sets:
`high/` (good agreement → `open`), `low/` (poor judge and annotator agreement → `blocked`)
and `partial/` (blank `zh` sheet → `blocked`). Every row carries `SYNTHETIC-DRY-RUN` in
`notes`, the directory contains its own 30-item `policy.json`, and the README there repeats
that these are not human labels. `scripts/score-judge-calibration.py --check` exercises all
three plus malformed-sheet rejection, determinism and fail-closed behaviour; tests in
`tests/unit/eval/test_judge_gate.py` assert the same. A synthetic `open` verdict is marked
`synthetic: true` / `publishable: false` and must never be cited as calibration.

## 7. Failure modes the tool refuses

* missing columns, unknown labels, duplicate `item_id`, empty sheet, wrong row count
  (≠ 150 per language), missing language sheet, extra language — `blocked` with reasons;
* partial labels (some rows filled) — `blocked` as incomplete, never scored on a subset;
* disagreement without adjudication — `blocked` as unresolved;
* constant labels or < 2 scored rows — κ `null`, gate stays closed;
* missing/corrupt `gate.json` — consumer helper returns `diagnostic` (fail-closed);
* stale gate vs changed sheets — `--check` recomputes and compares; CI fails when the
  committed `gate.json` no longer matches the committed sheets.

## 8. Maintenance

* Changing a threshold or the sample size is an edit to `evals/judge/gate-policy.json`
  followed by `--score`; the verdict records the policy inline.
* A new benchmark version changes the sheet templates; regenerate them with the external arm,
  re-run `--score`, and keep the old sheets until adjudication is complete.
* The judged-metric label in `docs/results.md` and `reports/external-p0/report.md` must be
  updated only after `load_judged_label()` returns `calibrated`; until then the pending wording
  is the correct wording.
* The gate is offline: no network, no API key, no model call. Everything it needs is the
  sheet set and the policy.
