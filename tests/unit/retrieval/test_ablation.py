"""The ablation runner: arm loading, the metrics table, the fitted artifact, leakage.

# AT-RET-10  # AT-RET-05
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerdesk.config import load_settings
from answerdesk.errors import ConfigError
from answerdesk.retrieval.ablation import (
    DEFAULT_ARMS,
    PRIMARY_ARM,
    AblationReport,
    load_arms,
    run_ablation,
)
from answerdesk.retrieval.fit import FusionArtifact, LeakageError, assert_disjoint

REPO_ROOT = Path(__file__).resolve().parents[3]
ARMS_FILE = REPO_ROOT / "configs" / "retrieval" / "ablations.yaml"


def _settings(tmp_path: Path):
    return load_settings(
        profile="offline",
        overrides={
            "app": {
                "corpus_dir": str(tmp_path / "corpus"),
                "index_dir": str(tmp_path / "index"),
            }
        },
    )


@pytest.fixture(scope="module")
def ablation_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[AblationReport, Path]:
    """One real run of the runner on a six-document corpus (module scope: it is the cost)."""
    tmp_path = tmp_path_factory.mktemp("ablation")
    settings = _settings(tmp_path)
    import os

    previous = os.environ.get("ANSWERDESK_INGEST_DOCS_PER_LOCALE")
    os.environ["ANSWERDESK_INGEST_DOCS_PER_LOCALE"] = "1"
    try:
        report = run_ablation(
            settings,
            out_dir=tmp_path / "out",
            fit=True,
            bootstrap=50,
            latency_sample=2,
            echo=lambda _message: None,
        )
    finally:
        if previous is None:
            os.environ.pop("ANSWERDESK_INGEST_DOCS_PER_LOCALE", None)
        else:
            os.environ["ANSWERDESK_INGEST_DOCS_PER_LOCALE"] = previous
    return report, tmp_path / "out"


@pytest.fixture()
def ablation_report(ablation_run: tuple[AblationReport, Path]) -> AblationReport:
    return ablation_run[0]


@pytest.fixture()
def ablation_out(ablation_run: tuple[AblationReport, Path]) -> Path:
    return ablation_run[1]


def test_the_shipped_arms_file_matches_the_built_in_arms() -> None:
    """The YAML is the documented surface; the code default must not drift from it."""
    arms = load_arms(ARMS_FILE)
    assert [arm.name for arm in arms] == [arm.name for arm in DEFAULT_ARMS]
    for loaded, builtin in zip(arms, DEFAULT_ARMS, strict=True):
        assert loaded.fuser == builtin.fuser
        assert loaded.channels == builtin.channels
        assert loaded.rerank == builtin.rerank
        assert loaded.mmr == builtin.mmr


def test_load_arms_rejects_a_broken_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ConfigError, match="no arms file"):
        load_arms(missing)

    broken = tmp_path / "broken.yaml"
    broken.write_text("arms: not-a-list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'arms' list"):
        load_arms(broken)

    unknown_fuser = tmp_path / "fuser.yaml"
    unknown_fuser.write_text("arms:\n  - name: x\n    fuser: bm25-plus\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown fuser"):
        load_arms(unknown_fuser)

    duplicate = tmp_path / "dup.yaml"
    duplicate.write_text(
        "arms:\n  - name: x\n    fuser: rrf\n  - name: x\n    fuser: cc\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="duplicate arm names"):
        load_arms(duplicate)


def test_report_covers_all_six_arms(ablation_report: AblationReport) -> None:
    assert [arm.name for arm in ablation_report.arms] == [arm.name for arm in DEFAULT_ARMS]
    for arm in ablation_report.arms:
        assert arm.n > 0
        assert 0.0 <= arm.recall_any_at_20 <= 1.0
        assert 0.0 <= arm.recall_all_at_20 <= 1.0
        assert 0.0 <= arm.ndcg_at_10 <= 1.0
        assert 0.0 <= arm.mrr_at_10 <= 1.0
        assert arm.per_locale, f"{arm.name} has no per-locale rows (NFR-ML-01)"


def test_report_contains_a_primary_arm_and_uplift(ablation_report: AblationReport) -> None:
    assert ablation_report.primary_arm == PRIMARY_ARM
    assert any(arm.name == PRIMARY_ARM for arm in ablation_report.arms)
    assert ablation_report.hybrid_uplift is not None
    assert len(ablation_report.hybrid_uplift_ci) == 2
    assert ablation_report.hybrid_uplift_ci[0] <= ablation_report.hybrid_uplift_ci[1]


def test_report_is_explicitly_a_machinery_check(ablation_report: AblationReport) -> None:
    """§6.3 step 2: hashing-embedder numbers may never be presented as quality."""
    assert ablation_report.embedder.name.startswith("hashing")
    assert any("machinery check" in note for note in ablation_report.notes)


def test_the_fitted_artifact_is_on_disk_with_provenance(
    ablation_report: AblationReport, ablation_out: Path
) -> None:
    assert ablation_report.fitted is not None
    artifact_path = ablation_out / "fusion.json"
    assert artifact_path.is_file()
    artifact = FusionArtifact.load(artifact_path)
    assert artifact.split.name == "dev"
    assert artifact.excluded_split.name == "test"
    assert artifact.split.hash == ablation_report.dev_split.hash
    assert artifact.excluded_split.hash == ablation_report.eval_split.hash
    assert artifact.params.fuser in ("rrf", "cc")
    assert artifact.method == "coordinate_search"
    assert artifact.search_history


def test_the_reported_numbers_come_from_the_test_split(ablation_report: AblationReport) -> None:
    """INV-09: the dev split is used for fitting only; the table is the test split."""
    assert ablation_report.eval_split.name == "test"
    assert ablation_report.eval_split.hash != ablation_report.dev_split.hash
    assert ablation_report.eval_split.n_items > 0
    for arm in ablation_report.arms:
        assert arm.n == ablation_report.eval_split.n_items
    assert ablation_report.fitted is not None
    assert ablation_report.fitted.split.hash == ablation_report.dev_split.hash


def test_report_artifacts_are_written(ablation_report: AblationReport, ablation_out: Path) -> None:
    out = ablation_out
    report_path = out / "retrieval-ablation.json"
    table_path = out / "retrieval-metrics.md"
    assert report_path.is_file() and table_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["eval_split"]["name"] == "test"
    assert len(payload["arms"]) == len(DEFAULT_ARMS)
    table = table_path.read_text(encoding="utf-8")
    for arm in ablation_report.arms:
        assert arm.name in table
    assert "**caveat**" in table
    assert "Per-locale" in table


def test_no_leakage_between_the_splits(ablation_report: AblationReport) -> None:
    """The runner would have raised during fitting; this asserts the invariant explicitly."""
    dev_hash = ablation_report.dev_split.hash
    test_hash = ablation_report.eval_split.hash
    assert dev_hash != test_hash
    with pytest.raises(LeakageError):
        assert_disjoint(["shared"], ["shared"])


def test_rerank_and_mmr_arms_run_the_optional_stages(ablation_report: AblationReport) -> None:
    by_name = {arm.name: arm for arm in ablation_report.arms}
    assert by_name["hybrid-convex+rerank"].rerank == "lexical-overlap-v1"
    assert by_name["hybrid-convex+mmr"].mmr_lambda is not None
    assert by_name["bm25"].rerank is None and by_name["bm25"].mmr_lambda is None
