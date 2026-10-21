"""Licence and attribution integrity — the part a downstream user is most likely to get wrong."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bench.config import SUITE_LICENSES

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_licence_texts_are_present_and_identifiable(bench_dir: Path) -> None:
    expectations = {
        "belebele/LICENSE-CC-BY-SA-4.0.txt": "Attribution-ShareAlike 4.0 International",
        "mrtydi/LICENSE-APACHE-2.0.txt": "Apache License",
        "domain/LICENSE-CC0-1.0.txt": "CC0 1.0",
        "LICENSES/MIT.txt": "MIT License",
    }
    for relative, marker in expectations.items():
        text = (bench_dir / relative).read_text(encoding="utf-8")
        assert marker in text, relative


def test_repository_licence_is_mit() -> None:
    text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Permission is hereby granted" in text


def test_mit_copy_is_the_repository_licence(bench_dir: Path) -> None:
    assert (bench_dir / "LICENSES" / "MIT.txt").read_bytes() == (REPO_ROOT / "LICENSE").read_bytes()


def test_belebele_subset_carries_attribution_and_share_alike(bench_dir: Path) -> None:
    attribution = (bench_dir / "belebele" / "ATTRIBUTION.md").read_text(encoding="utf-8").casefold()
    assert "belebele" in attribution
    assert "cc by-sa 4.0" in attribution
    assert "share-alike" in attribution
    assert "2308.16884" in attribution
    assert "derived" in attribution
    readme = (bench_dir / "belebele" / "README.md").read_text(encoding="utf-8")
    assert "CC BY-SA 4.0" in readme
    assert "not MIT" in readme


def test_mrtydi_subset_carries_apache_attribution(bench_dir: Path) -> None:
    attribution = (bench_dir / "mrtydi" / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "Mr. TyDi" in attribution
    assert "Apache-2.0" in attribution
    assert "2108.08787" in attribution
    assert "Wikipedia" in attribution  # the residual licence risk is stated, not hidden


def test_dataset_card_names_every_source_and_version(bench_dir: Path, manifest: dict[str, Any]) -> None:
    card = (bench_dir / "DATASET_CARD.md").read_text(encoding="utf-8")
    for source in manifest["sources"]:
        assert source["dataset"] in card or source["id"] in card
        assert source["version"] in card, (source["id"], source["version"])
    for requirement in (
        "CC BY-SA 4.0",
        "Apache-2.0",
        "CC0-1.0",
        "Limitations",
        "Sampling",
        "Intended use",
        "Citation",
    ):
        assert requirement in card, requirement


def test_dataset_card_does_not_misrepresent_the_repository_licence(bench_dir: Path) -> None:
    card = (bench_dir / "DATASET_CARD.md").read_text(encoding="utf-8")
    assert "The repository is MIT; this dataset is not" in card
    assert "The repository's MIT licence applies to the *code*" in card
    assert "data is MIT" not in card
    license_map = (bench_dir / "LICENSES" / "README.md").read_text(encoding="utf-8")
    assert "the data is not uniformly MIT" in license_map


def test_item_licence_matches_its_directory(items: list[dict[str, Any]]) -> None:
    for item in items:
        assert item["license"] == SUITE_LICENSES[item["suite"]]
        if item["suite"] == "belebele_retrieval" or item["suite"] == "belebele_qa":
            assert item["license"] == "CC-BY-SA-4.0"
        elif item["suite"] == "mrtydi_retrieval":
            assert item["license"] == "Apache-2.0"
        else:
            assert item["license"] == "CC0-1.0"


def test_no_forbidden_dataset_enters_provenance(items: list[dict[str, Any]]) -> None:
    forbidden = ("rgb", "crud-rag", "crud_rag", "crudrag")
    for item in items:
        dataset = str(item["provenance"]["dataset"]).casefold()
        assert not any(name in dataset for name in forbidden), item["item_id"]


def test_no_raw_archive_or_oversized_file_is_committed(bench_dir: Path) -> None:
    for path in bench_dir.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix not in (".gz", ".zip", ".tar", ".tgz", ".bz2", ".xz"), path
        assert path.stat().st_size < 8 * 1024 * 1024, path
        assert not (path.name.startswith("belebele_") and path.name.endswith(".jsonl")), path
