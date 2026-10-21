# Query-rewriter retrieval comparison

backend: `local_lora` · cases: 870 · corpus: `corpus-77d7d081813738a6` · index: `index-dc9909c648367626`

| slice | n | recall@20 | recall@20 rewritten | Δ@20 | 95% CI | Δ@10 | 95% CI |
|---|---|---|---|---|---|---|---|
| noisy_freeform | 474 | 0.8228 | 0.8186 | -0.0042 | [-0.0316, +0.0232] | +0.0570 | [+0.0169, +0.0992] |
| held_out | 156 | 0.9679 | 0.7885 | -0.1795 | [-0.2500, -0.1154] | -0.2885 | [-0.3718, -0.2051] |
| clean | 240 | 0.9208 | 0.8000 | -0.1208 | [-0.1667, -0.0792] | -0.1500 | [-0.2042, -0.1000] |

clean-slice non-regression gate (Δ@20 ≥ −0.01): **FAILED** (Δ = -0.1208)

latency per query: p50 110.649 ms, p95 149.792 ms (batched amortised (batch generation divided by items); not single-request latency)

bootstrap: 1000 resamples, seed 20261021
