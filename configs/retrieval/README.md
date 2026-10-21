# Retrieval configuration (owner: t5)

Two files, two purposes. Neither is loaded by `answerdesk.config.Settings` — the frozen
configuration surface lives in `configs/default.yaml` (`retrieval:` section) — and that is
deliberate: an ablation arm must never be able to change the shipped pipeline by editing a
file the service also reads.

| file | what it controls | who reads it |
|---|---|---|
| `ablations.yaml` | the six arms, their channels and optional stages, and the evaluation cut-offs | `python -m answerdesk.retrieval.ablation --arms configs/retrieval/ablations.yaml` |
| `fusion.yaml` | cold-start fusion parameters, the fitting grids/method, the artifact paths, and the post-fusion stage budgets | the ablation runner (documented contract) and humans reviewing `docs/retrieval.md` |

## Why the shipped defaults are RRF, not convex combination

`cold_start.fuser: rrf` with `rrf_k: 60` is the untuned default (requirements D-03). RRF is
rank-based, so it cannot be damaged by an uncalibrated score scale — which is the situation
on a fresh checkout before any dev-split fit has run. The convex combination is the *tuned*
path: `fit.method: coordinate_search` searches `rrf_k` and `alpha` on the dev split and
persists the winner, and only then does `retrieval.fuser: cc` become the right setting.

The falsification clause in requirements §6.4 applies: if the fitted convex combination
does not beat RRF on our own dev data (RT-7), RRF stays the default and the ADR records it.

## Artifacts are outputs, not inputs

`fit.artifacts.*` points at `data/index/ablation/` — a git-ignored runtime directory. A
fitted `fusion.json` carries the corpus hash, both split hashes, the embedder identity, the
optimised metric and the full grid it searched, so a number in `docs/results.md` can always
be traced back to the parameters that produced it (NFR-REP-02). Committing a fitted artifact
into `configs/` would let the configuration claim provenance it cannot prove.

## Regenerating everything

```bash
uv run answerdesk ingest --demo                     # corpus + query set
uv run answerdesk index --demo                      # BM25 + exact dense index
uv run python -m answerdesk.retrieval.ablation      # fit on dev, table on test
```

Add `--bootstrap 1000` for the paired confidence interval on the RT-7 uplift, and
`--no-fit` to see the cold-start parameters instead of the fitted ones.
