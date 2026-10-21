# `reports/offline-smoke/` — the offline harness smoke run

**This directory is a build product.** Every file in it except this README is rewritten by

```bash
uv run answerdesk eval --synthetic --provider fake --report reports/offline-smoke
```

so a reviewer comparing two runs should diff the directory, and a reviewer reading one number
should read it from `report.md` §1, not from a copy of a copy.

| field | value |
|---|---|
| run date (regenerated) | 2026-10-21 |
| corpus version | `corpus-77d7d081813738a6` (corpus v2.1: 648 files, 5,739 chunks, 6 locales) |
| corpus hash (sha256) | `ae30066e61e1dc1402b4a51a12f4c4f66335d72c207073b496ef5cd9e5378a36` |
| golden version / hash | `v1` / `ad1a9a3aeeaa88c75b7ae1725993dcb9856a640eb5e428b5576cde44aae73f13` |
| cases | 530 (6 locales) |
| mode | REPLAY-MODE — deterministic doubles, no model call, no network, no key |
| digest | see `report.md` (deterministic digest printed at the end of the run) |

What the run proves and what it does not (the same caveat the report opens with): it validates
the harness end to end — corpus packing, retrieval arms, prompts, verifier, decision plumbing and
artefact writing — on machine-labelled synthetic data. It is **not** a quality measurement of the
model: no number here may be presented as a live-model or production result (§10.7 item 9,
NFR-REP-01). The live evidence package is `reports/bench/` (`docs/results.md`).

The previous copy of this directory was generated on the corpus v1 (360 documents, 746 cases);
it went stale when corpus v2.1 landed in t18 and was regenerated in t50 against the current
corpus, with the date and version recorded above.
