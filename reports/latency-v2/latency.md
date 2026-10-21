# Latency v2 — measured arms

Generated 2026-10-21T16:32:44+00:00 · mode `live` · profile `api` · measured here: **True**

Generator: `qwen3.6-35b-a3b-noreason` · judge: `deepseek-chat` · host: win32 / python 3.12.10

| arm | n | TTFT p50 | TTFT p95 | total p50 | total p95 | in tok p50 | out tok mean | USD/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 20 | 673 ms | 965 ms | 2382 ms | 7474 ms | 1618 | 268.4 | n/a (unpriced) |
| budget | 20 | 707 ms | 2410 ms | 1728 ms | 3038 ms | 1530 | 105.3 | n/a (unpriced) |
| cascade | 20 | 638 ms | 12187 ms | 2840 ms | 13948 ms | 1494 | 239.2 | n/a (unpriced) |

## Quality

| arm | n | rubric mean | rubric p50 | groundedness | answered | judge failures |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 20 | 3.85 | 5.00 | 0.882 | 17 | 0 |
| budget | 20 | 4.65 | 5.00 | 0.950 | 20 | 0 |
| cascade | 20 | 4.30 | 5.00 | 0.895 | 19 | 0 |

## Retrieval (must not regress)

n=20 · p50 2.72 ms · p95 5.05 ms · target p95 ≤ 15 ms · retrieval stage only; unchanged by this task

## Cascade

n=20 · small=0 (deepseek-chat) · large=20 (qwen3.6-35b-a3b-noreason) · easy threshold 0.71 · min margin 0.00

reasons: low_confidence=20

## Checks

* output-token reduction: **60.7%** (target ≥ 30%) — met
* quality drop: **-0.80** rubric points (budget ≤ 0.2) — within budget
* cascade quality drop: **-0.45** rubric points (budget ≤ 2) — within budget
* retrieval p95: 5.05 ms — ok
* TTFT recorded for every streamed arm: True

## Notes

* Both arms stream from the same endpoint with the same retrieval; only the output budget (cap, claim limit, compact schema, extractive instruction) differs.
* TTFT is measured inside the stream (first non-empty delta), not derived from total.
* Judge: cross-family rubric judge (deepseek-chat) over 20 items per arm.
* Cascade not exercised: no item cleared the derived threshold on this sample, so the cascade row is a second large-model pass. The router itself is unit-tested; a calibrated p_hat (t8 engine) is required for a meaningful split.

## Commands

* `check`: `python -m answerdesk.llm.latency_report --check`
* `live`: `python -m answerdesk.llm.latency_report --run --live --limit 24`
