# Judge-gate dry-run fixtures (SYNTHETIC — never human data)

Every CSV in this directory is a **synthetic dry-run fixture**, generated to prove that the
scoring pipeline and the mechanical gate in `scripts/judge_gate.py` work:

* every row carries `SYNTHETIC-DRY-RUN` in `notes`, and the directory names (`high/`, `low/`,
  `partial/`) are recognised as synthetic by the scorer;
* the artefacts computed from them are marked `synthetic: true`, `publishable: false`, and
  must never be cited as human calibration or as a quality claim;
* `high/` has good judge–human agreement and pairwise agreement (gate opens),
  `low/` has deliberately poor judge and annotator agreement (gate blocks),
  `partial/` leaves the `zh` sheet blank (gate blocks as incomplete).

The real sheets live in `reports/external-p0/calibration/` and are blank until two human
annotators fill them. This directory exists only for `tests/unit/eval/test_judge_gate.py` and
`uv run python scripts/score-judge-calibration.py --check`.
