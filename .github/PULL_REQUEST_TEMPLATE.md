## What and why

<!-- One paragraph: the behaviour change, and the problem it solves. -->

**Requirement / invariant:** <!-- e.g. FR-CAC-03, INV-02, or "docs only" -->
**Kind:** <!-- plumbing | metric/eval | threshold/policy | contract/config (needs amendment + ADR) | docs | perf -->

## How it was verified

<!-- Paste the commands you ran and their result. The gate below must be green. -->

```
uv run pytest -q
uv run ruff check src tests
uv run mypy src/answerdesk
```

If this changes a number in `docs/results.md`, say which artifact and command regenerated it:
`reports/<run>/…`, `scripts/build-results.py --run … --out docs/results.md --check`.

## Contracts, config and docs

- [ ] No contract, config key, CLI flag or metric definition changed.
      *(If one did: it is listed below with the requirement amendment and ADR.)*
- [ ] Cross-module types still come from `answerdesk.contracts`; no ad-hoc dicts added.
- [ ] Traceability markers `# AT-xxx-nn` added/updated for new behaviour.
- [ ] Docs updated in this PR (module doc, `docs/design-decisions.md` if a decision changed, README if
      a public command or claim changed).
- [ ] No new runtime dependency, or the dependency is behind an extra with a licence entry
      (MIT/BSD/Apache-2.0/ISC/PSF; unmodified MPL-2.0 only).

## Honesty checklist

- [ ] Every claim in this PR is backed by an artifact in the diff (measurement, test output, trace).
- [ ] No number was hand-edited into a document.
- [ ] Judged metrics carry their κ caveat; replay-mode numbers are labelled.
- [ ] Negative results are reported, not deleted.
- [ ] No secret, provider key or real customer/employer data in the diff.

## Reviewer notes

<!-- Anything that needs a human judgement call: a trade-off, a deliberately deferred fix, a follow-up issue. -->
