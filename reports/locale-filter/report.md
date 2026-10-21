# Locale-aware retrieval (CNS-6): before / after

- corpus: `corpus-77d7d081813738a6` (chunk hash `852b9574323a258c`)
- query set: consistency run `reports/consistency-p0` — 378 clean instances, 6 locales, digest `3b6d35b833ab099c`
- packed context window: 12 chunks
- policy: same-locale first, cross-locale fallback when the window cannot be filled (min same-locale share 0.5, min language confidence 0.5)

## Evidence-locale purity

| metric | before (policy off) | after (policy on) | Δ |
|---|---|---|---|
| chunk purity (pooled) | 0.769 | 0.994 | +0.226 |
| instances fully same-locale | 0.201 | 0.979 | +0.778 |
| instances ≥90% same-locale | 0.341 | 0.979 | +0.637 |
| `recall_any@20` (doc-level) | 0.984 | 0.984 | +0.000 |
| `recall_any@10` (doc-level) | 0.820 | 0.820 | +0.000 |

Target: chunk purity ≥ **0.90** — **PASS**.

## Per locale

| locale | instances | chunk purity before | chunk purity after | fully pure before | fully pure after |
|---|---|---|---|---|---|
| ky | 63 | 0.853 | 0.992 | 0.254 | 0.952 |
| ru | 63 | 0.717 | 1.000 | 0.111 | 1.000 |
| tg | 63 | 0.833 | 1.000 | 0.349 | 1.000 |
| tr | 63 | 0.807 | 1.000 | 0.286 | 1.000 |
| uz | 63 | 0.710 | 0.974 | 0.095 | 0.921 |
| zh | 63 | 0.692 | 1.000 | 0.111 | 1.000 |

## Guards

- surface-form invariance: 756/378 (rate 1.000) — same packed context for a casings/spacing/punctuation variant
- ablation non-regression (`recall_any@20` on clean `direct` questions, n=600):

| arm | policy off | policy on | Δ |
|---|---|---|---|
| bm25 | 0.968 | 0.968 | +0.000 |
| dense | 0.475 | 0.475 | +0.000 |
| hybrid-rrf | 0.927 | 0.927 | +0.000 |

Tolerance: each arm must not drop more than 0.02 — **PASS**.
- determinism: packed-context digest identical across two runs — **PASS**

## Gates

- purity: PASS
- invariance: PASS
- ablation_guard: PASS
- packed_recall: PASS
- determinism: PASS

Overall: **PASS**
