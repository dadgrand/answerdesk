# Escalation calibration report (synthetic)

* artifact: `cal-f70b9bf8edb0` (sha256 `f70b9bf8edb0a672…`)
* created: 2026-10-20T22:15:19+00:00
* scorer: `synthetic-population` synthetic-population-1.0.0
* calibration hash: `4adbe992f74965fc…`; test hash: `264dd60c1017bc2b…`
* costs: C_wrong=1.0, C_abstain=0.1, C_delay=0.05 (cost hash `7fd95740cbdd…`)
* split: synthetic-0.50/0.50; n_calibration=2249, n_test=2251

## Pooled metrics

| metric | value | target | pass |
|---|---|---|---|
| E-1 risk<=alpha [informational] | 0.0135 | 0.0500 | yes |
| E-1 risk UCB<=alpha+0.03 [informational] | 0.0235 | 0.0800 | yes |
| E-1 risk<=alpha [transactional] | 0.0048 | 0.0200 | yes |
| E-1 risk UCB<=alpha+0.03 [transactional] | 0.0267 | 0.0500 | yes |
| E-1 risk<=alpha [account_legal] | 0.0000 | 0.0100 | yes |
| E-1 risk UCB<=alpha+0.03 [account_legal] (n/a: 0 answered) | 0.0000 | 0.0400 | yes |
| E-1 pooled risk<=0.05 | 0.0119 | 0.0500 | yes |
| E-2 must-escalate recall>=0.95 | 1.0000 | 0.9500 | yes |
| E-3 escalation precision>=0.55 (unsafe or would-be-wrong) | 0.5991 | 0.5500 | yes |
| E-3b escalation precision (safety-only reading) | 0.4468 | 0.5500 | **NO** |
| E-4 coverage on answerable items>=0.60 | 0.6317 | 0.6000 | yes |
| E-5 ECE(width)<=0.08 | 0.0250 | 0.0800 | yes |
| E-5 Brier<=0.15 | 0.0309 | 0.1500 | yes |
| E-6 AURC (diagnostic) | 0.0626 | — | — |

## Per bucket

| bucket | alpha | q_hat | answer_min_p | coverage(ans.) | risk | UCB | ECE | Brier |
|---|---|---|---|---|---|---|---|---|
| informational | 0.050 | 0.9764 | 0.9500 | 0.789 | 0.0135 | 0.0235 | 0.0263 | 0.0308 |
| transactional | 0.020 | 0.9858 | 0.9800 | 0.475 | 0.0048 | 0.0267 | 0.0365 | 0.0307 |
| account_legal | 0.010 | 0.9968 | 0.9900 | 0.000 | 0.0000 | 1.0000 | 0.0353 | 0.0323 |

## Honest limitations

1. This is a **synthetic** study: `p̂` is a simulated calibrated scorer, so
   the numbers prove the machinery (quantile, thresholds, guards, metrics), not
   a model's calibration. t11 re-runs the same code on real scored traffic;
   until then no quality claim is made.
2. ECE depends on the binning rule; both equal-width and equal-mass values are in
   `calibration-metrics.json`, and the reliability plot shows the bins.
3. The conformal risk statement additionally needs a calibrated scorer; the
   coverage guarantee (1 − α) is distribution-free only under exchangeability
   (§8.6 monitoring).
