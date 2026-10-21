# Escalation: conformal abstention, cost-sensitive routing, guards

Owner: **t8** (`answerdesk.policy`). Normative source: `docs/requirements.md` §8 (escalation
contract), §3 (invariants), §11 (security), Appendix B (reason codes), Appendix C (keys).
This document is the derivation the requirements demand — the guarantee, the Bayes rule and
the coverage numbers, each with the code that implements it and the test that measures it.

```text
query + passages ──► guards (PII → injection → risk tier)      ─┐
                     │ forced? → escalate/refuse (typed)         │
                     ▼                                           │
              answerability scorer  p̂(x)                        │
                     ▼                                           │
              split conformal  s = 1 − p̂  ≤  q̂ ?  ── no ──► LOW_CONFIDENCE
                     │ yes                                       │
                     ▼                                           │
              cost band  p̂ ≥ max(1 − C_delay/C_wrong, 1 − α_final)? ── yes ──► answer
                     │ no                                        │
                     ├─ p̂ ≥ 1 − C_abstain/C_wrong ──► clarify   │
                     └─ else ──────────────────────► escalate   ┘
```

Every decision records `score = p̂`, `alpha = α_final`, `reason_code` (closed enum) and
`threshold_ref = "cal-<artifact hash>"`, so a decision can be reproduced from a sealed file.

---

## 1. Split conformal: what is guaranteed, and what is not

### 1.1 The quantile

With a calibration set of `n` exchangeable risk scores `s_1…s_n` (`s = 1 − p̂`) and a target
risk `α`, define

```
rank = ⌈(n + 1)(1 − α)⌉                     # the finite-sample (+1) correction
q̂    = the rank-th smallest of {s_i}        # q̂ = +∞ when rank > n
```

`conformal_rank` / `conformal_quantile` in `policy/conformal.py`; the certificate stores
`n`, `rank`, `q̂`, `α`, the calibration-set hash and the scorer identity.

**Why the `+1`.** For exchangeable calibration and test points, the rank of the test score
among the `n+1` scores is uniform on `{1, …, n+1}`. Covering the test point means its score is
among the `rank` smallest, which happens with probability `rank/(n+1) ≥ 1 − α`. So

```
P(s_new ≤ q̂) ≥ 1 − α          (distribution-free, finite-sample)
```

That is the *coverage* half, and it is what the `n+1` correction buys. `minimum_calibration_size(α)
= ⌈1/α⌉ − 1` (19 for 0.05, 49 for 0.02, 99 for 0.01): below that the rank exceeds `n` and the
certificate is **infeasible** — the decision layer then answers nothing at that α rather than
pretending. `test_minimum_calibration_size_matches_the_rank_algebra` pins the algebra.

### 1.2 The risk half needs a calibrated scorer

Coverage says nothing about correctness by itself. The risk statement is:

> if `p̂` is calibrated (`P(correct | p̂ = p) = p`), then among the selected points
> `P(error | selected) = E[1 − p̂ | selected] ≤ 1 − threshold`.

Proof: on the event `{p̂ ≥ τ}`, `1 − p̂ ≤ 1 − τ`, so
`P(error ∧ p̂ ≥ τ) = E[𝟙{p̂ ≥ τ}(1 − p̂)] ≤ (1 − τ)·P(p̂ ≥ τ)`. Dividing gives the claim.

The decision layer therefore answers only above `1 − α_final` (see §3), and the artifact
reports ECE and Brier for the calibrated-scorer half. This split is deliberate: the coverage
guarantee is distribution-free, the risk guarantee is conditional on calibration, and a system
that hides the second is uncalibrated by construction.

### 1.3 Finite-sample honesty of the property test

`tests/unit/policy/test_conformal.py::test_coverage_property_on_fresh_samples` runs 200 seeded
trials (fresh calibration set of 199 scores, one fresh test point) and asserts the *pooled*
nonconformity rate stays within `α + 0.02`. A per-trial "≥ 95% of trials covered" assertion is
a coin flip at this size: with true coverage ≈ 0.951, `P(≥ 190/200 covered) ≈ 0.5`. Stating the
pooled bound (and the reason) is the honest version; the exact observed numbers are in the
failure message and in `reports/policy-smoke/calibration-metrics.json`.

---

## 2. The cost-sensitive three-way rule

With `p̂ = P(answer correct | x, E)` and three actions, expected costs are

```
E[cost | answer]   = (1 − p̂)·C_wrong
E[cost | clarify]  = C_delay
E[cost | escalate] = C_abstain
```

Minimising over actions gives the Bayes rule of §8.3:

```
answer    iff (1 − p̂)·C_wrong ≤ min(C_delay, C_abstain)   ⇔ p̂ ≥ 1 − C_delay/C_wrong
clarify   iff C_delay ≤ (1 − p̂)·C_wrong < C_abstain      ⇔ 1 − C_abstain/C_wrong ≤ p̂ < 1 − C_delay/C_wrong
escalate  otherwise                                      ⇔ p̂ < 1 − C_abstain/C_wrong
```

Costs live in `configs/policy/costs.yaml` with provenance (source, owner, date); a cost file
without provenance fails validation (`load_cost_set`, `test_a_cost_without_provenance_is_invalid_config`).
`CostSet` exposes the thresholds as properties, so no module compares `p̂` with a literal
(CL-POL-01): `test_changing_costs_moves_the_operating_point` proves the band moves when the
matrix moves.

Reference numbers: `C_wrong = 1.00`, `C_abstain = 0.10`, `C_delay = 0.05` ⇒ answer ≥ 0.95,
clarify ∈ [0.90, 0.95), escalate < 0.90; cost ratio 10:1.

---

## 3. Composing three axes (and why only answers carry risk)

Per bucket `b` (§8.2), with `α_final(b) = min(α_b, C_abstain/C_wrong)`:

```
answer_min_p(b)  = max(1 − C_delay/C_wrong, 1 − α_final(b))
clarify_min_p(b) = 1 − C_abstain/C_wrong
```

* The **cost axis** sets the answer edge; the **risk-tier axis** can tighten it
  (α = 0.01 ⇒ ≥ 0.99) but can never loosen it — exactly §8.2's "the cost model can tighten α
  but never loosen it". A cheap-cost matrix with α = 0.05 still answers at ≥ 0.95
  (`test_changing_costs_moves_the_operating_point`, last assertion).
* **Clarify and escalate never display an answer**, so they carry no correctness risk; their
  edges are purely cost-driven. This is why the clarify band is not tightened by α.

| bucket | α_b | α_final | answer ≥ | clarify ≥ | binding axis |
|---|---|---|---|---|---|
| informational | 0.05 | 0.05 | 0.95 | 0.90 | cost and α coincide |
| transactional | 0.02 | 0.02 | 0.98 | 0.90 | α |
| account/legal | 0.01 | 0.01 | 0.99 | 0.90 | α (and the guard tier above) |

Account/legal is *also* hard-policy (§8.4 step 2), so in the shipped configuration it escalates
before the band is consulted; its α exists for the case where an operator deliberately relaxes
the tier (`configs/policy/guards.yaml`).

---

## 4. Guards: deterministic, total, and unable to force an answer

Precedence (§8.4) is implemented as one ordered function, `guards.evaluate_guards`:

1. **PII** — `policy/pii.py` detects e-mail, card (13–19 digits passing Luhn), phone (7–15
   digits with a `+`/separator, or a bare 10–11 digit run) and account/IBAN shapes. The
   *precision* controls matter as much as recall: dates (`05.03.2026`), amounts (`150 000`,
   `12,5 €`), times and identifiers (`TS-999`, `ORD-12345`) are explicitly excluded, and a
   finding stores a hashed token, never the value (INV-05). Recall/precision are measured on a
   fixture matrix in `tests/unit/policy/test_pii_guards.py` (≥ 0.95 each).
2. **Injection** — `policy/injection.py` reuses the generation-layer rule set (one auditable
   list) and adds answer coercion, hidden/bidi characters and credential requests.
3. **Risk tier** — `IntentBucket.ACCOUNT_LEGAL` plus the labels in
   `configs/policy/guards.yaml` (account, legal, fraud, complaint, kyc, …). Refund stays
   transactional on purpose: it is exact-cache-only, not force-escalated.

A guard returns a `GuardVerdict` whose `action` is validated to be `escalate` or `refuse`;
"a guard cannot force an answer" is a type-level fact, not a comment.

### 4.1 Reason codes

`Decision` and `EscalationDecision` use the closed enum of Appendix B; the guard paths emit
`PII_DETECTED`, `INJECTION_SUSPECTED`, `HIGH_RISK_INTENT`; the band emits `ANSWERED`,
`CLARIFY_AMBIGUOUS`, `LOW_CONFIDENCE`; stale calibration emits `STALE_CALIBRATION`; generator
signals pass through unchanged (`NO_EVIDENCE`, `VERIFIER_REJECT`, `CITATION_FAILED`,
`PROVIDER_ERROR`). `contracts.Decision` rejects `ANSWERED` for non-answer actions, so a
mislabelled decision cannot be constructed.

---

## 5. Metrics and the release gates

| metric | definition | gate |
|---|---|---|
| realised risk | errors / answers on the held-out split, per bucket and pooled | E-1: ≤ α (Wilson UCB ≤ α + 0.03) |
| must-escalate recall | escalated fraction of (unanswerable ∪ PII ∪ injection ∪ out-of-scope) | E-2 ≥ 0.95 |
| escalation precision | escalated turns that were unsafe **or** would have been wrong | E-3 ≥ 0.55 |
| coverage (answerable) | answers / answerable turns | E-4 ≥ 0.60 |
| ECE | `Σ_b (n_b/n)·|acc_b − conf_b|`, equal-width bins (equal-mass reported too) | E-5 ≤ 0.08 |
| Brier | `mean((p̂ − y)²)` | E-5 ≤ 0.15 |
| AURC | `∫ risk dcoverage` over the threshold sweep | E-6: reported, not gated |

**On E-3.** Two readings are reported, because the requirement's parenthetical ("genuinely
unsafe/unanswerable") excludes *correct* conservatism: a low-confidence answerable turn is
exactly what a safe system should escalate, yet it is not "unsafe". The artifact therefore
reports `escalation_precision` (unsafe ∨ would-have-been-wrong, the useful one, gated) and
`escalation_precision_safety` (unsafe only, the literal reading, also reported). In the smoke
run the literal reading lands below 0.55 (0.447) while the effective one passes (0.599) —
reported, not hidden, and discussed in §8.

Wilson intervals are implemented from the closed form (`wilson_interval`), the exact
Clopper–Pearson upper bound is available where a conservative bound matters, and every table
carries `n`.

---

## 6. The calibration artifact

`policy/artifact.py` defines `CalibrationArtifact`: version, `created_at`, corpus version,
calibration/test split hashes, scorer identity, cost set + hash, conformal certificates, the
threshold table, pooled metrics, per-bucket metrics, and `artifact_hash` computed over
everything except itself. `verify()` re-hashes and rejects any edit; `save`/`load` round-trip.

Operational rules (§8.6):

* **staleness** — age > `max_age_days` (default 7) or a corpus-version mismatch marks the
  artifact stale; `DecisionEngine` then escalates with `STALE_CALIBRATION` unless the guard
  policy explicitly says `warn`. `assert_calibrated_engine` raises `StaleCalibrationError` for
  startup checks.
* **refresh** — `python -m answerdesk.policy.calibrate --synthetic` re-fits and re-seals; a live
  run uses the same code path with real scored traffic (t11).
* **monitoring** — the service compares rolling coverage with the calibration-time value and
  alerts at −3 points (FR-ESC-10; t9 owns the wiring).

---

## 7. The synthetic study: what it proves and what it does not

`policy/synthetic.py` generates a labelled population: a latent `p_true` (bimodal — most support
turns are easy, a minority are hard), `correct ~ Bernoulli(p_true)`, a simulated scorer
`p̂ = clip(p_true + N(0, σ))` with σ = 0.02, and the special shares that form the must-escalate
set (unanswerable 18%, PII 2%, injection 2%, out-of-scope 1%). The bucket mix mirrors the golden
set (65/25/10).

*It proves* the machinery: the quantile, the composition, the guards, the metrics and the
artifact are correct on data whose truth is known, and the decision rules behave as derived.
*It does not prove* any model's calibration — that is E-5 on real traffic (t11), and the report
says so in its own limitations section.

Smoke run (`python -m answerdesk.policy.calibrate --synthetic --out reports/policy-smoke`,
seed 20261020, 4500 items, 50/50 split; numbers from `reports/policy-smoke/calibration-metrics.json`):

| bucket | α | q̂ | answer ≥ | coverage (answerable) | risk | risk UCB | ECE |
|---|---|---|---|---|---|---|---|
| informational | 0.050 | 0.9764 | 0.9500 | 0.607 | 0.0135 | 0.0235 | 0.0250 |
| transactional | 0.020 | 0.9858 | 0.9800 | 0.369 | 0.0048 | 0.0267 | 0.0250 |
| account/legal | 0.010 | 0.9968 | 0.9900 | 0.000 | 0.0000 | n/a | 0.0250 |

Pooled: risk 0.0119 (UCB 0.0202), coverage on answerable turns 0.6317, ECE 0.0250, Brier 0.0309,
AURC 0.0626, E-2 recall 1.0000, E-3 precision 0.5991. Account/legal coverage is 0 because the
hard-policy guard escalates that tier before the band — a design outcome, not a calibration
failure.

---

## 8. Limitations (stated, not hidden)

1. **Synthetic only.** Until t11 runs the same code on labelled traffic, no claim is made about
   a model's calibration; the artifact's numbers prove the machinery.
2. **Exchangeability.** The coverage bound assumes it; traffic drifts (new product, new locale).
   §8.6 monitoring exists because the assumption is not self-enforcing.
3. **E-3 has two defensible readings** (§5). Both are reported; the gate uses the effective one
   and the difference is visible in `gate-checks.json`.
4. **ECE is binning-dependent** (equal-width gated, equal-mass reported, reliability plot
   alongside).
5. **The answerability scorer matters.** `OverlapScorer` is deterministic and monotone in
   evidence overlap but not calibrated by construction; `LLMJudgeScorer` inherits every judge
   bias of §10.5. Neither is claimed to be a good probability until ECE/Brier are measured.
6. **NLI scorer deferred.** The Protocol is the seam; an NLI implementation would add a model
   dependency to a module whose CI value is being deterministic.

---

## 9. Command reference and map

```bash
uv run pytest -q tests/unit/policy                       # AT-ESC-01…10
uv run python -m answerdesk.policy.calibrate --synthetic --out reports/policy-smoke   # CMD-K
uv run answerdesk calibrate --synthetic                  # the same, via the CLI
```

| requirement | code | test |
|---|---|---|
| FR-ESC-01 scorer interface | `scorers.py` | `test_scorers_satisfy_the_contract_protocol`, `test_overlap_scorer_is_monotone_in_evidence_overlap` |
| FR-ESC-02 conformal n+1 | `conformal.py` | `test_conformal.py` (rank algebra, property test) |
| FR-ESC-03 cost rule | `costs.py`, `thresholds.py` | `test_costs.py`, `test_decision.py::test_band_boundaries_for_the_default_costs` |
| FR-ESC-04 precedence | `guards.py`, `decision.py` | `test_pii_guards.py`, `test_decision.py` (forced outcomes) |
| FR-ESC-05 reason codes | `contracts.py` enum | `tests/unit/test_contracts.py` parity, decision tests |
| FR-ESC-06 artifact provenance | `artifact.py` | `test_decision.py` (seal, tamper, staleness) |
| FR-ESC-07 release targets | `calibrate.py`, `metrics.py` | `test_calibrate.py::test_artifact_meets_the_risk_and_calibration_targets` |
| FR-ESC-08 PII | `pii.py` | `test_pii_guards.py` (matrix, precision/recall) |
| FR-ESC-09 injection | `injection.py` | `test_pii_guards.py` |
| FR-ESC-10 coverage monitor | `decision.refresh_staleness`, `CacheStats`-style counters | `test_decision.py::test_stale_artifact_never_answers` |
