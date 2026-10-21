# Risk control v3 (run `risk-v3`)

Bench population: 485 dev / 159 test turns, re-answered with the shipped pipeline and re-judged cross-family.

- AUROC 0.828645 (target >= 0.80: True)
- ECE 0.068949 (target <= 0.08: True)
- coverage at realised risk <= 0.05: 0.012579 (target >= 0.20: False)
- certificate at alpha 0.05: ABSENT (threshold 1.0, calibration risk 1.0, UCB 1.0, n=485)
- certified operating point on test: coverage 0.0, realised risk None (Wilson None)
- rows needed for a 5% certificate: Bentkus 46.1, Hoeffding 460.5

## Caveats

- labels are cross-family LLM judgments, not human labels
- the certificate covers the calibration draw and the selection rule; test risk is reported separately and is not covered by it
- dev calibration is the union of the re-answered bench dev split and the consistency run's dev facts; both use the same feature schema
