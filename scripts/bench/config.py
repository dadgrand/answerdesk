"""Constants for the ``answerdesk-bench`` build.

One source of truth: the builder, the verifier, the tests and the dataset card all read
these values, and the manifest records them, so a reader can check the sampling rule
instead of trusting prose (``docs/benchmark.md`` §3).

Nothing here is random: sampling ranks items by ``sha256(seed + role + id)`` so the same
seed always selects the same items, independent of filesystem or JSON ordering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

#: Benchmark identity. ``VERSION_DIR`` is the on-disk directory name under ``data/bench``.
BENCH_NAME: Final = "answerdesk-bench"
VERSION: Final = "0.1.0"
VERSION_DIR: Final = "v0.1"
SCHEMA_VERSION: Final = "1.0"
RELEASE_DATE: Final = "2026-10-21"

#: Seed for every sampling and split decision. Changing it is a benchmark version bump.
SEED: Final = 20261021

#: Filesystem anchor: ``<repo>/scripts/bench/config.py`` -> ``<repo>``.
PACKAGE_ROOT: Final = Path(__file__).resolve().parents[1]
REPO_ROOT: Final = PACKAGE_ROOT.parent
CACHE_DIR: Final = REPO_ROOT / ".cache" / "answerdesk-bench"
BENCH_DIR: Final = REPO_ROOT / "data" / "bench" / BENCH_NAME / VERSION_DIR
SOURCE_LOCK: Final = BENCH_DIR / "sources.lock.json"
CORPUS_DIR: Final = REPO_ROOT / "data" / "corpus"
GOLDEN_DIR: Final = REPO_ROOT / "evals" / "golden" / "v1"
SPLITS_FILE: Final = REPO_ROOT / "evals" / "splits" / "v1.json"
DOCS_DIR: Final = REPO_ROOT / "docs"

#: Belebele language variants (ISO code -> FLORES-200 code used by the HF release).
BELEBELE_LANGS: Final = {
    "ru": "rus_Cyrl",
    "uz": "uzn_Latn",
    "tg": "tgk_Cyrl",
    "ky": "kir_Cyrl",
    "zh": "zho_Hans",
}

#: Languages of the domain slice (the corpus has one extra locale, ``tr``).
DOMAIN_LANGS: Final = ("ru", "uz", "tg", "ky", "zh", "tr")

#: Every language that appears anywhere in the benchmark.
ALL_LANGS: Final = ("ru", "uz", "tg", "ky", "zh", "tr")

#: Belebele sampling: questions per language, split by the passage they belong to.
BELEBELE_TEST_PER_LANG: Final = 120
BELEBELE_DEV_PER_LANG: Final = 30
#: One passage in five is assigned to ``dev``; questions inherit their passage's split.
BELEBELE_SPLIT_MODULUS: Final = 5

#: Domain slice quotas per locale. Sums to 40 items per locale (240 total).
DOMAIN_QUOTAS: Final = {
    "direct": 8,
    "numeric_date": 8,
    "multi_hop": 8,
    "pii": 6,
    "injection": 6,
    "unanswerable": 4,
}
#: One item in five is assigned to ``dev`` within each category, by seeded rank.
DOMAIN_DEV_MODULUS: Final = 5

#: MinHash dedup parameters. 16 hashes / 4 bands of 4 rows catches ~0.9+ Jaccard pairs.
MINHASH_HASHES: Final = 16
MINHASH_BANDS: Final = 4
MINHASH_THRESHOLD: Final = 0.90
SHINGLE_SIZE: Final = 5

#: A question may legitimately mix scripts (acronyms, transliterated names); below this
#: share of the expected script it is treated as a language/encoding error.
MIN_SCRIPT_SHARE: Final = 0.30

#: Verifier floors. The committed tree must clear all of them.
MIN_TOTAL_ITEMS: Final = 1200
MIN_RETRIEVAL_QUERIES_PER_LANG: Final = 100
MIN_QA_ITEMS_PER_LANG: Final = 100
MIN_DOMAIN_ITEMS: Final = 200
MIN_DOMAIN_ITEMS_PER_CATEGORY: Final = 20

#: Datasets that must never enter this tree (licence-blocked, ``research/answerdesk``).
FORBIDDEN_DATASETS: Final = ("rgb", "crud-rag", "crud_rag", "crudrag")

#: Source identifiers used in provenance records.
SOURCE_BELEBELE: Final = "facebook/belebele"
SOURCE_MRTYDI: Final = "castorini/mr-tydi"
SOURCE_SYNTHETIC: Final = "answerdesk-synthetic"

#: Licence ids per suite, repeated by ``schema.py`` (kept in sync by a test).
SUITE_LICENSES: Final = {
    "belebele_retrieval": "CC-BY-SA-4.0",
    "belebele_qa": "CC-BY-SA-4.0",
    "mrtydi_retrieval": "Apache-2.0",
    "domain": "CC0-1.0",
}

#: Licence files copied from the pinned cache into the committed tree.
LICENSE_FILES: Final = (
    ("raw/licenses/CC-BY-SA-4.0.txt", "belebele/LICENSE-CC-BY-SA-4.0.txt"),
    ("raw/licenses/APACHE-2.0.txt", "mrtydi/LICENSE-APACHE-2.0.txt"),
    ("raw/licenses/CC0-1.0.txt", "domain/LICENSE-CC0-1.0.txt"),
)
