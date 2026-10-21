# External validity run `external-p0`

Hardware: Windows 11 / Python 3.12.10

Curated benchmark: answerdesk-bench 0.1.0 (manifest `65de8498b494b653…`)

## Belebele-derived QA (judge-free, option-letter scoring)

Answerer: `deepseek-chat`

| language | n | closed-book | RAG | delta | 95% CI | crosses zero |
|---|---|---|---|---|---|---|
| ky | 120 | 0.458333 | 0.741667 | 0.283333 | [0.175, 0.391667] | no |
| ru | 120 | 0.583333 | 0.85 | 0.266667 | [0.175, 0.358333] | no |
| tg | 120 | 0.533333 | 0.791667 | 0.258333 | [0.158333, 0.358333] | no |
| uz | 120 | 0.541667 | 0.775 | 0.233333 | [0.133333, 0.333333] | no |
| zh | 120 | 0.7 | 0.691667 | -0.008333 | [-0.083333, 0.075] | yes |

## Mr. TyDi ru (native qrels)

n = 994 queries, 2428 passages, fusion {'alpha_bm25': 0.5, 'family': 'convex combination', 'normalisation': 'min-max'}

| arm | nDCG@10 | recall@10 | MRR@10 |
|---|---|---|---|
| bm25 | 0.603617 | 0.72837 | 0.601402 |
| hybrid | 0.747153 | 0.8833 | 0.728749 |

Paired bootstrap nDCG@10 delta: 0.143536 [0.129003, 0.158084] (crosses zero: False)

## Belebele-derived retrieval (diagnostic only)

- ky: n=120, bm25 nDCG@10 0.737558, hybrid 0.827194, delta 0.089637 [0.056524, 0.122829], crosses zero False — **derived labels, diagnostic**
- ru: n=120, bm25 nDCG@10 0.724983, hybrid 0.860306, delta 0.135323 [0.097471, 0.17497], crosses zero False — **derived labels, diagnostic**
- tg: n=120, bm25 nDCG@10 0.729744, hybrid 0.839775, delta 0.110032 [0.077857, 0.149057], crosses zero False — **derived labels, diagnostic**
- uz: n=120, bm25 nDCG@10 0.681534, hybrid 0.857894, delta 0.17636 [0.125321, 0.235669], crosses zero False — **derived labels, diagnostic**
- zh: n=120, bm25 nDCG@10 0.066397, hybrid 0.969015, delta 0.902618 [0.856783, 0.943801], crosses zero False — **derived labels, diagnostic**

## Judge calibration

Status: pending-human-labels

no human labels are present in this checkout; every judged metric in this arm is diagnostic until calibration-kappa.json exists with κ >= 0.6 per language

## Non-claims

- no state-of-the-art claim of any kind
- no retrieval-quality claim for uz/tg/ky (derived labels only)
- no claim whose bootstrap interval crosses zero
- no calibrated-judge claim until the human-label κ file exists
