"""Fitting on the dev split, the persisted artifact, and the leakage guard (INV-09).

# AT-RET-05
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerdesk.index.store import EmbedderInfo
from answerdesk.retrieval.fit import (
    RRF_K_GRID,
    CorpusProvenance,
    FusionArtifact,
    FusionObservation,
    LeakageError,
    SplitProvenance,
    assert_disjoint,
    coordinate_search,
    fit_fusion_artifact,
    fit_logistic_weights,
    split_hash,
)
from answerdesk.retrieval.fusion import FusionParams, Normalization

DEV_IDS = [f"dev-{index}" for index in range(12)]
TEST_IDS = [f"test-{index}" for index in range(8)]
CORPUS = CorpusProvenance(version="corpus-test", hash="a" * 64)
EMBEDDER = EmbedderInfo(name="hashing-32", dimension=32)


def _split(name: str, ids: list[str]) -> SplitProvenance:
    return SplitProvenance(name=name, hash=split_hash(ids), n_items=len(ids))  # type: ignore[arg-type]


def _dev_split() -> SplitProvenance:
    return _split("dev", DEV_IDS)


def _test_split() -> SplitProvenance:
    return _split("test", TEST_IDS)


def test_split_hash_is_order_independent_and_deduplicated() -> None:
    assert split_hash(["a", "b"]) == split_hash(["b", "a", "a"])
    assert split_hash(["a"]) != split_hash(["a", "b"])
    assert len(split_hash(["a"])) == 64


def test_assert_disjoint_raises_on_any_overlap() -> None:
    assert_disjoint(["a", "b"], ["c"])
    with pytest.raises(LeakageError, match="INV-09"):
        assert_disjoint(["a", "b"], ["b", "c"])


def test_fitting_refuses_overlapping_splits() -> None:
    """§6.5's leakage check: a test item inside the fitting input is a hard error."""
    with pytest.raises(LeakageError, match="INV-09"):
        fit_fusion_artifact(
            evaluate=lambda _params: 0.5,
            dev_split=_split("dev", ["a", "leak"]),
            test_split=_split("test", ["leak"]),
            dev_item_ids=["a", "leak"],
            test_item_ids=["leak"],
            corpus=CORPUS,
            embedder=EMBEDDER,
        )


def test_fitting_refuses_a_provenance_hash_that_does_not_match_the_ids() -> None:
    with pytest.raises(LeakageError, match="does not match the fitting ids"):
        fit_fusion_artifact(
            evaluate=lambda _params: 0.5,
            dev_split=_split("dev", ["other"]),
            test_split=_test_split(),
            dev_item_ids=DEV_IDS,
            test_item_ids=TEST_IDS,
            corpus=CORPUS,
            embedder=EMBEDDER,
        )


def test_coordinate_search_finds_the_true_optimum() -> None:
    """A synthetic objective with a known best point must be recovered exactly."""

    def evaluate(params: FusionParams) -> float:
        if params.fuser == "rrf":
            return 0.5 - abs(params.rrf_k - 40) / 1000.0
        return 0.9 - abs((params.alpha or 0.0) - 0.7)

    params, value, history = coordinate_search(evaluate)
    assert params.fuser == "cc"
    assert params.alpha == pytest.approx(0.7)
    assert value == pytest.approx(0.9)
    assert len(history) == len(RRF_K_GRID) + 8
    assert {row["fuser"] for row in history} == {"rrf", "cc"}


def test_coordinate_search_prefers_rrf_when_convex_cannot_win() -> None:
    """Documented tie-break reversal: if CC never beats RRF, RRF ships."""

    def evaluate(params: FusionParams) -> float:
        return 0.8 if params.fuser == "rrf" else 0.1

    params, value, _history = coordinate_search(evaluate, rrf_grid=(60,), alpha_grid=(0.5,))
    assert params.fuser == "rrf"
    assert params.rrf_k == 60
    assert value == pytest.approx(0.8)


def test_coordinate_search_uses_the_requested_normalisation() -> None:
    params, _value, _history = coordinate_search(
        lambda params: 1.0 if params.normalization is Normalization.ZSCORE else 0.0,
        rrf_grid=(60,),
        alpha_grid=(0.5,),
        normalization=Normalization.ZSCORE,
    )
    assert params.fuser == "cc"
    assert params.normalization is Normalization.ZSCORE


def test_logistic_weights_learn_which_channel_predicts_relevance() -> None:
    """bm25 separates gold from noise, dense is constant: bm25 must win the weight."""
    observations: list[FusionObservation] = []
    for index in range(60):
        gold = index % 2 == 0
        observations.append(
            FusionObservation(
                features={"bm25": 1.0 if gold else 0.0, "dense": 0.5},
                label=1 if gold else 0,
            )
        )
    weights = fit_logistic_weights(observations)
    assert weights["bm25"] > weights["dense"]
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert fit_logistic_weights(observations) == weights  # deterministic


def test_logistic_weights_survive_an_imbalanced_base_rate() -> None:
    """Gold candidates are rare (~8 of 50); the fit must not collapse to equal weights.

    Without an intercept the optimiser reaches the low base rate only by making every weight
    negative, which the non-negativity clip turns into the equal-weight fallback. This test
    is the regression guard for exactly that bug.
    """
    observations: list[FusionObservation] = []
    for index in range(200):
        gold = index % 10 == 0  # 10% positive rate
        observations.append(
            FusionObservation(
                features={"bm25": 0.9 if gold else 0.1, "dense": 0.5},
                label=1 if gold else 0,
            )
        )
    weights = fit_logistic_weights(observations)
    assert weights["bm25"] > weights["dense"], weights
    assert weights["bm25"] > 0.7


def test_logistic_weights_are_used_by_the_artifact_path(tmp_path: Path) -> None:
    """The learned weights are persisted, not just computed (FR-RET-05)."""
    observations = [
        FusionObservation(
            features={"bm25": 1.0 if index % 4 == 0 else 0.0, "dense": 0.4},
            label=1 if index % 4 == 0 else 0,
        )
        for index in range(80)
    ]
    artifact = fit_fusion_artifact(
        evaluate=lambda _params: 0.0,
        dev_split=_dev_split(),
        test_split=_test_split(),
        dev_item_ids=DEV_IDS,
        test_item_ids=TEST_IDS,
        corpus=CORPUS,
        embedder=EMBEDDER,
        observations=observations,
        method="logistic_regression",
    )
    assert artifact.method == "logistic_regression"
    assert artifact.params.fuser == "cc"
    assert artifact.params.weights
    assert sum(artifact.params.weights.values()) == pytest.approx(1.0, abs=1e-6)
    loaded = FusionArtifact.load(artifact.save(tmp_path / "fusion.json"))
    assert loaded.params.weights == artifact.params.weights
    assert loaded.search_history


def test_logistic_weights_fall_back_to_equal_weights_without_data() -> None:
    assert fit_logistic_weights([]) == {"bm25": 0.5, "dense": 0.5}


def test_artifact_roundtrip_records_provenance(tmp_path: Path) -> None:
    artifact = fit_fusion_artifact(
        evaluate=lambda params: 0.42 if params.fuser == "cc" else 0.2,
        dev_split=_dev_split(),
        test_split=_test_split(),
        dev_item_ids=DEV_IDS,
        test_item_ids=TEST_IDS,
        corpus=CORPUS,
        embedder=EMBEDDER,
        fitted_at="2026-01-01T00:00:00Z",
    )
    path = artifact.save(tmp_path / "fusion.json")
    loaded = FusionArtifact.load(path)
    assert loaded.model_dump() == artifact.model_dump()
    assert loaded.split.hash == split_hash(DEV_IDS)
    assert loaded.excluded_split.hash == split_hash(TEST_IDS)
    assert loaded.corpus.version == "corpus-test"
    assert loaded.embedder.dimension == 32
    assert loaded.fitted_at == "2026-01-01T00:00:00Z"
    assert loaded.metric_value == pytest.approx(0.42)
    assert loaded.search_history


def test_artifact_fingerprint_ignores_the_timestamp() -> None:
    def build(timestamp: str) -> FusionArtifact:
        return fit_fusion_artifact(
            evaluate=lambda params: 0.3,
            dev_split=_dev_split(),
            test_split=_test_split(),
            dev_item_ids=DEV_IDS,
            test_item_ids=TEST_IDS,
            corpus=CORPUS,
            embedder=EMBEDDER,
            fitted_at=timestamp,
        )

    first, second = build("2026-01-01T00:00:00Z"), build("2026-02-02T00:00:00Z")
    assert first.fingerprint() == second.fingerprint()
    assert first.fitted_at != second.fitted_at


def test_artifact_rejects_one_split_for_fitting_and_reporting() -> None:
    with pytest.raises(ValueError, match="fitted and reported on one split"):
        FusionArtifact(
            artifact_id="fusion-x",
            method="coordinate_search",
            metric="nDCG@10",
            metric_value=0.5,
            params=FusionParams(fuser="rrf", rrf_k=60),
            split=_dev_split(),
            excluded_split=SplitProvenance(
                name="test", hash=split_hash(DEV_IDS), n_items=len(DEV_IDS)
            ),
            corpus=CORPUS,
            embedder=EMBEDDER,
        )


def test_artifact_load_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LeakageError, match="ablation"):
        FusionArtifact.load(tmp_path / "nope.json")


def test_artifact_json_forbids_unknown_keys(tmp_path: Path) -> None:
    artifact = fit_fusion_artifact(
        evaluate=lambda params: 0.3,
        dev_split=_dev_split(),
        test_split=_test_split(),
        dev_item_ids=DEV_IDS,
        test_item_ids=TEST_IDS,
        corpus=CORPUS,
        embedder=EMBEDDER,
    )
    payload = artifact.model_dump(mode="json")
    payload["injected"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="injected"):
        FusionArtifact.load(path)
