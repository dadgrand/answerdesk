# `evals/` — frozen evaluation artifacts

Everything in this directory is **generated** by `answerdesk.eval.datasets` from the ingested
corpus with a fixed seed. Do not hand-edit: the same seed and corpus reproduce the files
byte-for-byte, so a diff means the generator (or the corpus) changed and must be reviewed.

Regenerate deliberately:

```bash
cd answerdesk
uv run answerdesk eval --synthetic --provider fake --report reports/offline-smoke --regenerate
```

## Tree

| Path | Content |
|---|---|
| `splits/v1.json` | frozen document-level split (§10.2) with its self-hash; a hand edit is rejected on load |
| `golden/v1/test.jsonl` | the stratified test split (360 cases: 60 per locale) |
| `golden/v1/dev.jsonl` | the dev split (216 cases: 36 per locale) — the only place parameters may be fitted (`INV-09`) |
| `golden/v1/manifest.json` | provenance: seed, corpus hash, split hash, per-file sha256 and item counts, every §10.1 composition check with its target/actual/verdict, and the 36-cell locale × category matrix |
| `golden/v1/adversarial/near_miss.jsonl` | cache near-miss replay pairs (same intent, near-identical wording, different acceptable evidence) |
| `golden/v1/adversarial/injection.jsonl` | prompt-injection cases (§13.5/NFR-SEC-01) |
| `golden/v1/adversarial/pii.jsonl` | PII-bearing cases with reserved-range values only (§13.4) |
| `golden/v1/adversarial/perturbation.jsonl` | typo / code-switch / numeric / identifier perturbations for RT-9 |
| `golden/v1/adversarial/claims.jsonl` | claim-level verifier labels: verbatim sentences (`supported`) and mutated twins (`unsupported`) |
| `judge/rubric_v1.md` | the rubric the judge uses, versioned; the code holds the same text (`eval.judge.RUBRIC_TEXT`) |

## Rules

* **Labels are machine-derived** in `golden/v1`: `metadata.machine_label` is `true` on every
  generated case. §10.3 requires human verification for any gate claim, so these artifacts
  validate the harness and never authorise a quality number.
* **Retrieval labels are equivalence classes**, not single documents: every FAQ chunk whose
  question normalises to the same stem (digits masked) is acceptable evidence. See
  `docs/eval-methodology.md` §2.3.
* **Licence**: the corpus is synthetic CC0-1.0; these files inherit it. No third-party text.
* Adversarial sets are not part of the stratified test count; they are inputs for the
  escalation, security and cache measurements.
* A corpus rebuild invalidates the manifest: `load_golden` compares the `corpus_hash` and
  fails loudly when the corpus changed after the freeze.

## What is *not* here

Human verification labels, judge–human calibration labels (`judge/human_v1.jsonl`), the
conformal calibration artifact (`artifacts/`, t8) and the evidence reports (`reports/bench`,
t11). Their absence is deliberate: they are human or system outputs, not generated fixtures.
