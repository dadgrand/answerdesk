# Attribution — Belebele-derived subset

Everything under `belebele/` is derived from **Belebele**, and is distributed under
**CC BY-SA 4.0** — the same licence as the original. The full licence text is in
`LICENSE-CC-BY-SA-4.0.txt`.

## Original work

* **Dataset:** Belebele — *The Belebele Benchmark: a Parallel Reading Comprehension Dataset
  in 122 Language Variants* (Bandarkar, Liang, Muller, Artetxe, Husa, Goyal, Zettlemoyer,
  Koyejo, 2023; arXiv:2308.16884).
* **Distribution:** <https://huggingface.co/datasets/facebook/belebele>, commit
  `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2`, release `ds=2023-05-20`.
* **Licence of the original:** CC BY-SA 4.0 (<https://creativecommons.org/licenses/by-sa/4.0/>).
* **Files used:** `data/rus_Cyrl.jsonl`, `data/uzn_Latn.jsonl`, `data/tgk_Cyrl.jsonl`,
  `data/kir_Cyrl.jsonl`, `data/zho_Hans.jsonl`. Their SHA-256 hashes are pinned in
  `../sources.lock.json` and repeated in every item's `provenance.source_sha256`.

## What we changed (CC BY-SA 4.0 §3(a)(1)(B) — indicate modifications)

* Deduplicated the FLORES passages of each language variant into one pool per language
  (488 passages each) and assigned derived ids `bb-<lang>-pNNNN` in first-appearance order.
* Sampled 150 of 900 questions per language by a seeded hash, and assigned each question the
  split of its gold passage (30 dev / 120 test).
* Added derived fields: `item_id`, `suite`, `category`, `split`, `gold_evidence_ids`,
  `answer_anchor`, and provenance metadata.
* Derived retrieval relevance labels: a question's gold passage is used as its single qrel.
  **Belebele contains no retrieval labels; this is our derivation, not an upstream
  annotation.**
* Removed (did not ship) questions whose four options are not distinct, and did not ship the
  raw archive or any non-sampled question text.

No question, option or passage text was edited, translated or reordered. The `link` field of
each passage is kept as `source_link`.

## Share-alike notice

This directory is a derivative work of Belebele and is licensed **CC BY-SA 4.0**. If you
redistribute it, or a modified version of it, you must:

1. give appropriate credit to Belebele and to this benchmark (this file, the dataset card and
   the item provenance satisfy that);
2. indicate the changes (listed above);
3. license your derivative under CC BY-SA 4.0 or a licence compatible with it.

The repository's MIT licence applies to the *code* of `answerdesk` only. It does **not**
apply to the contents of this directory. Mixing these files into an MIT-licensed dataset
would misrepresent their licence.

## Suggested citation

```bibtex
@inproceedings{bandarkar2023belebele,
  title     = {The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants},
  author    = {Bandarkar, Lucas and Liang, Davis and Muller, Benjamin and Artetxe, Mikel and
               Husa, Satya and Goyal, Naman and Zettlemoyer, Luke and Koyejo, Sanmi},
  booktitle = {arXiv:2308.16884},
  year      = {2023}
}
```
