# Query-rewriter retrieval comparison

backend: `local_lora` · cases: 870 · corpus: `corpus-77d7d081813738a6` · index: `index-dc9909c648367626`

| slice | n | recall@20 | recall@20 rewritten | Δ@20 | 95% CI | Δ@10 | 95% CI |
|---|---|---|---|---|---|---|---|
| noisy_freeform | 474 | 0.8228 | 0.8312 | +0.0084 | [-0.0211, +0.0359] | +0.0633 | [+0.0253, +0.1076] |
| held_out | 156 | 0.9679 | 0.7949 | -0.1731 | [-0.2436, -0.1090] | -0.2885 | [-0.3718, -0.2115] |
| clean | 240 | 0.9208 | 0.7958 | -0.1250 | [-0.1708, -0.0833] | -0.1500 | [-0.2042, -0.1000] |

clean-slice non-regression gate (Δ@20 ≥ −0.01): **FAILED** (Δ = -0.1250)

latency per query: p50 122.211 ms, p95 152.653 ms (batched amortised (batch generation divided by items); not single-request latency)

bootstrap: 1000 resamples, seed 20261021
