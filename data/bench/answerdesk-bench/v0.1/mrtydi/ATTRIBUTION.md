# Attribution — Mr. TyDi ru subset

Everything under `mrtydi/` is derived from **Mr. TyDi** (Russian), and is distributed under
**Apache-2.0** — the licence the dataset declares. The full licence text is in
`LICENSE-APACHE-2.0.txt`.

## Original work

* **Dataset:** Mr. TyDi — *Mr. TyDi: A Multi-lingual Benchmark for Dense Retrieval* (Zhang,
  Thakur, Ogundepo, Kamalloo, Alfonso-Hermelo, Li, Liu, Rezagholizadeh, Lin, 2021;
  arXiv:2108.08787).
* **Queries and relevance judgments:** <https://huggingface.co/datasets/castorini/mr-tydi>,
  commit `1d43c80218d06d0ef80f5b172ccabd848b948bc1` (v1.1), files
  `mrtydi-v1.1-russian/{dev,test}.jsonl.gz`.
* **Passage text:** <https://huggingface.co/datasets/castorini/mr-tydi-corpus>, commit
  `3a3aa212bbe94a8cc0dc858710a3dad49d532054`, file
  `mrtydi-v1.1-russian/corpus.jsonl.gz` (1.6 GB, not redistributed).
* **Declared licence:** Apache-2.0 on both Hugging Face repositories. SHA-256 hashes are
  pinned in `../sources.lock.json`.

## What we changed

* Extracted only the 2,428 passages that are positive evidence of a dev or test query; the
  full corpus is not redistributed.
* Dropped duplicate query strings (2 of 2,371) so the suite satisfies the benchmark's
  near-duplicate invariant; both splits are otherwise complete.
* Added derived ids (`mrtydi-ru:<docid>`), `item_id`, `suite`, `category`, `split`,
  provenance metadata and a `title` field.

Query text, passage text and relevance labels are unchanged. No JMESPath/regex rewriting,
translation or reordering was applied.

## Residual licence note (recorded, not hidden)

Mr. TyDi's passages are Wikipedia text. The dataset declares Apache-2.0 and we redistribute
under that declaration, but Wikipedia text is itself CC BY-SA. This is a known ambiguity in
the upstream release, not something this benchmark can resolve. If your use is sensitive to
it, either fetch the corpus yourself with `--fetch` and do not redistribute `mrtydi/`, or use
`belebele/` and `domain/` only. The repository's MIT licence applies to code only.

## Suggested citation

```bibtex
@inproceedings{zhang2021mrtydi,
  title     = {Mr. TyDi: A Multi-lingual Benchmark for Dense Retrieval},
  author    = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and
               Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
  booktitle = {arXiv:2108.08787},
  year      = {2021}
}
```
