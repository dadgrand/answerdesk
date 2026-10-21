# `evals/judge` — the judge-calibration gate

Everything in this directory is about one rule: **a judged metric may be labelled
`calibrated` only after the LLM judge has been shown to agree with two blind human
annotators on this corpus.**

| Path | What it is |
|---|---|
| `rubric_v1.md` | the grading rubric the judge applies (kept in sync with `src/answerdesk/eval/judge.py`) |
| `gate-policy.json` | the mechanical gate parameters: languages, sheet size, labels, κ ≥ 0.6, pairwise agreement ≥ 0.7, bootstrap CI settings |
| `../reports/judge-gate/gate.json` | the current gate verdict (status `pending` / `blocked` / `open`, per-language κ with CIs, reasons) |
| `../reports/judge-gate/gate.md` | the same verdict as a human-readable table |
| `../reports/judge-gate/disagreements.csv` | annotator A ≠ B rows that need adjudication |
| `../reports/external-p0/calibration/` | the 150-item blank labelling sheets per language produced by the external arm |
| `../../tests/unit/eval/fixtures/judge-gate/` | **synthetic** dry-run fixtures that prove the computation and the gate transition; never human data |

## Current state

`reports/judge-gate/gate.json` is `pending`: no human labels are present in this checkout,
so every judged metric in the external and benchmark reports stays **diagnostic**. The gate
flips mechanically (no code change) once real labels satisfy the policy; until then,
`scripts/judge_gate.py:load_judged_label()` returns `diagnostic` — and it fails closed if the
gate file is missing, stale or synthetic.

## Commands

```bash
# Fill the blank sheets in reports/external-p0/calibration/ (two annotators, blind), then:
uv run python scripts/score-judge-calibration.py --score          # writes reports/judge-gate/*
uv run python scripts/score-judge-calibration.py --check         # same self-check (canonical entry point)
```

The scoring command validates every sheet (columns, labels, unique item ids, 150 rows per
language), computes pairwise and judge–human κ with deterministic bootstrap CIs, writes the
disagreement rows for adjudication, and records the gate decision together with the SHA-256
of every input sheet.

Full protocol, adjudication workflow and gate semantics: [`docs/judge-calibration.md`](../../docs/judge-calibration.md).

## Rules for contributors

* **Never commit human labels you did not collect.** Blank sheets are the honest default;
  the gate stays `pending`.
* Never copy labels from an LLM or from another model into `human_label_*` columns.
* Dry-run fixtures are marked `SYNTHETIC-DRY-RUN` in every row and in their directory name;
  a gate computed from them may be `open` but is `synthetic: true` and `publishable: false`.
* Changing a threshold means editing `gate-policy.json` and re-running `--score`; the
  decision must remain traceable to the policy values stored inside `gate.json`.
