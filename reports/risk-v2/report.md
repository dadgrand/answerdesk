# Risk control v2 (run `risk-v2`)

Primary population: **consistency** — the consistency population carries the full answer text, so all five decision-time features are available; the bench population lacks two of them and its reranker feature was rate-limited at build time, and it fails the AUROC target as a result

Labels: cross-family LLM judge (deepseek-flash), not human labels; alpha 0.05, delta 0.1. Rows needed for a 5% certificate: Bentkus 46.1, Hoeffding 460.5 (at zero observed risk).

## Population `consistency` (133 dev / {'evaluated_once': True, 'auroc': 0.964713, 'ece_equal_mass': 0.057409, 'ece_equal_width': 0.052773, 'aurc': 0.446281, 'coverage_at_risk': {'alpha': 0.05, 'coverage': 0.115942, 'risk': 0.041667, 'risk_ci': [0.007393, 0.202418], 'threshold': 0.819824, 'n_accepted': 24, 'label': 'oracle frontier point (chosen on test for reporting only, not deployable)'}, 'certified_operating_point': {'threshold': 0.968445, 'n': 207, 'n_accepted': 1, 'coverage': 0.004831, 'wrong': 0, 'realised_risk': 0.0, 'risk_ci': [0.0, 0.793451]}} rows test)

- base error: dev 0.691729, test 0.758454
- AUROC 0.964713 (target >= 0.75: True)
- ECE (equal-mass) 0.057409 (target <= 0.08: True)
- coverage at realised risk <= 0.05: 0.115942 (target >= 0.40: False)
- AURC 0.446281
- certificate feasible: False; threshold 0.968445; calibration risk 1.0 (UCB 1.0) over 133 rows
- certified operating point on test: coverage 0.004831, realised risk 0.0
- largest feasible alpha certificate: alpha 0.25, feasible True, coverage 0.082707

## Population `bench` (200 dev / {'evaluated_once': True, 'auroc': 0.664808, 'ece_equal_mass': 0.186036, 'ece_equal_width': 0.177332, 'aurc': 0.343192, 'coverage_at_risk': {'alpha': 0.05, 'coverage': 0.0, 'risk': None, 'threshold': None, 'n_accepted': 0, 'label': 'oracle frontier: no prefix meets the target'}, 'certified_operating_point': {'threshold': 0.999948, 'n': 200, 'n_accepted': 0, 'coverage': 0.0, 'wrong': 0, 'realised_risk': None, 'risk_ci': None}} rows test)

- base error: dev 0.465, test 0.425
- AUROC 0.664808 (target >= 0.75: False)
- ECE (equal-mass) 0.186036 (target <= 0.08: False)
- coverage at realised risk <= 0.05: 0.0 (target >= 0.40: False)
- AURC 0.343192
- certificate feasible: False; threshold 0.999948; calibration risk 1.0 (UCB 1.0) over 200 rows
- certified operating point on test: coverage 0.0, realised risk None
- largest feasible alpha certificate: alpha 0.25, feasible False, coverage 0.0

## Caveats

- labels are cross-family LLM judgments, not human labels; the certificate covers the judge's notion of correctness
- the certificate is over the calibration draw and the selection rule; test risk is reported separately and is not covered by it
- the consistency population's base error is dominated by sibling-document ambiguity (CNS-2); the bench population is the product's own frozen test split
- two decision-time features (numeric consistency, claim support) need the full answer text and are unavailable on the bench population
