# Judge-calibration protocol (external arm)

Template size: 150 items per language, sampled deterministically by item id from the
test split of the suite, for: ru, zh, uz, tg, ky.

0. Read each item's question and evidence from the licensed benchmark file named in
   `item_ref` (the sheet itself carries no CC BY-SA text).
1. Two annotators label every item independently as `supported`, `unsupported` or
   `unclear`, seeing the question, the retrieved evidence and the answer — **not** the
   judge's label, which is hidden until both sheets are returned.
2. Disagreements are adjudicated to `adjudicated_label`; the adjudicated sheet is the
   reference. A second-pass agreement rate below 0.7 triggers a rubric clarification
   round before any κ is reported.
3. `answerdesk eval external --score-calibration <dir>` computes Cohen's κ between
   the judge (binary: supported vs the rest) and the adjudicated labels, per language
   and overall, and writes `calibration-kappa.json`.
4. **Until that file exists with κ ≥ 0.6 in every language, every judged metric in the
   external report is labelled diagnostic and must not be quoted as a quality claim.**
   The external arm's primary numbers (multiple-choice accuracy, retrieval nDCG/recall)
   are judge-free by construction, so they are unaffected by this gate.
