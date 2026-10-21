# Query-rewriter retrieval comparison

backend: `deterministic` · cases: 870 · corpus: `corpus-77d7d081813738a6` · index: `index-dc9909c648367626`

| slice | n | recall@20 | recall@20 rewritten | Δ@20 | 95% CI | Δ@10 | 95% CI |
|---|---|---|---|---|---|---|---|
| noisy_freeform | 474 | 0.8228 | 0.8249 | +0.0021 | [-0.0063, +0.0105] | -0.0021 | [-0.0105, +0.0042] |
| held_out | 156 | 0.9679 | 0.9679 | +0.0000 | [-0.0192, +0.0192] | +0.0256 | [+0.0064, +0.0513] |
| clean | 240 | 0.9208 | 0.8917 | -0.0292 | [-0.0542, -0.0083] | -0.0167 | [-0.0333, -0.0042] |

clean-slice non-regression gate (Δ@20 ≥ −0.01): **FAILED** (Δ = -0.0292)

latency per query: p50 0.025 ms, p95 0.045 ms (wall-clock of the rewrite call only (retrieval excluded))

bootstrap: 1000 resamples, seed 20261021
