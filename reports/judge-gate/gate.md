# Judge calibration gate

- **Status:** `pending`
- **Judged metrics:** `diagnostic`
- **Publishable:** `false`
- **Synthetic input:** `false`
- **Source:** `reports/external-p0/calibration`
- **Generated:** 2026-10-21T16:11:01+00:00

## Per language

| language | rows | labelled | judge κ [95% CI] | pairwise agreement | pairwise κ | disagreements | status |
|---|---|---|---|---|---|---|---|
| ky | 150 | 0 | — | — | — | 0 | blank |
| ru | 150 | 0 | — | — | — | 0 | blank |
| tg | 150 | 0 | — | — | — | 0 | blank |
| uz | 150 | 0 | — | — | — | 0 | blank |
| zh | 150 | 0 | — | — | — | 0 | blank |

## Pooled

- rows: 750 (labelled 0)
- judge κ: None None
- pairwise agreement: None
- pairwise κ: None None

## Reasons

- human labels absent for: ky, ru, tg, uz, zh

## How to read this

`judged_metrics_label` is `calibrated` only when status is `open`; a missing,
corrupt or stale gate is treated as `diagnostic` by `load_judged_label` (fail-closed).
Synthetic dry-run fixtures can reach `open` to prove the mechanics but are never
`publishable`: real human labels are required for a reportable claim.
