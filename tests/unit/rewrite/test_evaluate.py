"""The measurement harness: paired bootstrap, recall deltas, latency and the clean gate.

# AT-GEN-17
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from answerdesk.rewrite import IdentityRewriter, evaluate_rewriter, write_report
from answerdesk.rewrite.evaluate import (
    BOOTSTRAP_SEED,
    CLEAN_DROP_TOLERANCE,
    RECALL_KS,
    EvalCase,
    paired_bootstrap,
)


class FakeSearcher:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[str, int]] = []

    def search_ids(self, query: str, k: int) -> list[str]:
        self.calls.append((query, k))
        return list(self.mapping.get(query, []))[:k]


class MarkerRewriter:
    """Appends a marker; only the marked query hits, so the delta is exactly +1 per case."""

    name = "marker"

    def __init__(self, marker: str = " fixed") -> None:
        self.marker = marker

    @property
    def available(self) -> bool:
        return True

    def rewrite(self, query: str, *, locale: str | None = None):
        from answerdesk.rewrite.base import RewriteResult

        return RewriteResult(
            original=query, query=f"{query}{self.marker}", backend=self.name, available=True
        )


def test_recall_ks_cover_the_release_gate() -> None:
    assert 10 in RECALL_KS and 20 in RECALL_KS
    assert pytest.approx(0.01) == CLEAN_DROP_TOLERANCE


def test_paired_bootstrap_is_centred_on_the_mean_difference() -> None:
    baseline = [0] * 80 + [1] * 20
    candidate = [1] * 80 + [0] * 20
    mean, low, high = paired_bootstrap(baseline, candidate, samples=500, seed=7)
    assert mean == pytest.approx(0.6)
    assert low <= mean <= high
    assert low > 0.0, "with n=100 and a 60-point swing the interval must exclude zero"


def test_paired_bootstrap_is_deterministic_for_a_seed() -> None:
    baseline = [index % 3 == 0 for index in range(60)]
    candidate = [index % 4 == 0 for index in range(60)]
    first = paired_bootstrap(baseline, candidate, samples=200, seed=BOOTSTRAP_SEED)
    second = paired_bootstrap(baseline, candidate, samples=200, seed=BOOTSTRAP_SEED)
    assert first == second
    assert paired_bootstrap(baseline, candidate, samples=200, seed=1) != first


def test_paired_bootstrap_handles_an_empty_slice() -> None:
    assert paired_bootstrap([], []) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        paired_bootstrap([1], [1, 0])


def test_evaluate_rewriter_measures_an_uplift(
    make_cases: Callable[..., list[EvalCase]],
) -> None:
    noisy = make_cases(
        [
            ("карта сколько стоит", ["chunk-1"]),
            ("займ оформит", ["chunk-2"]),
            ("перевод не дошёл", ["chunk-3"]),
            ("вклад процент", ["chunk-4"]),
        ]
    )
    searcher = FakeSearcher(
        {
            # Baseline misses everything; the rewritten query hits everything.
            "карта сколько стоит": [],
            "займ оформит": [],
            "перевод не дошёл": [],
            "вклад процент": [],
            "карта сколько стоит fixed": ["chunk-1"],
            "займ оформит fixed": ["chunk-2"],
            "перевод не дошёл fixed": ["chunk-3"],
            "вклад процент fixed": ["chunk-4"],
        }
    )
    report, traces = evaluate_rewriter(
        rewriter=MarkerRewriter(),
        slices={"noisy_freeform": noisy},
        searcher=searcher,
        samples=200,
        seed=3,
    )
    entry = report.slice_by_name("noisy_freeform")
    assert entry is not None
    assert entry["recall@20"] == pytest.approx(0.0)
    assert entry["recall@20_rewritten"] == pytest.approx(1.0)
    assert entry["delta@20"] == pytest.approx(1.0)
    assert entry["improved@20"] == 4 and entry["worsened@20"] == 0
    assert report.n_cases == 4
    assert traces["noisy_freeform"][0]["rewritten"].endswith("fixed")
    assert searcher.calls, "the harness must actually search"


def test_identity_baseline_has_zero_delta(make_cases: Callable[..., list[EvalCase]]) -> None:
    cases = make_cases([("вопрос", ["chunk-1"]), ("второй", ["chunk-2"])])
    searcher = FakeSearcher({"вопрос": ["chunk-1"], "второй": []})
    report, _ = evaluate_rewriter(
        rewriter=IdentityRewriter(),
        slices={"noisy_freeform": cases, "clean": cases},
        searcher=searcher,
        samples=50,
    )
    for entry in report.slices:
        assert entry["delta@20"] == pytest.approx(0.0)
    assert report.clean_gate_passed is True


def test_clean_non_regression_gate_fails_on_a_regression(
    make_cases: Callable[..., list[EvalCase]],
) -> None:
    """A rewriter that helps noisy input but breaks clean input must fail the gate."""
    noisy = make_cases([("грязный запрос", ["chunk-1"])])
    clean = make_cases([("чистый запрос", ["chunk-2"])])
    searcher = FakeSearcher(
        {
            "грязный запрос": [],
            "грязный запрос fixed": ["chunk-1"],
            "чистый запрос": ["chunk-2"],
            "чистый запрос fixed": [],
        }
    )
    report, _ = evaluate_rewriter(
        rewriter=MarkerRewriter(),
        slices={"noisy_freeform": noisy, "clean": clean},
        searcher=searcher,
        samples=50,
    )
    assert report.slice_by_name("noisy_freeform")["delta@20"] == pytest.approx(1.0)
    assert report.clean_gate_passed is False
    assert report.clean_gate_delta <= -CLEAN_DROP_TOLERANCE


def test_report_is_written_with_traces(
    make_cases: Callable[..., list[EvalCase]], tmp_path: Path
) -> None:
    cases = make_cases([("карта", ["chunk-1"])])
    searcher = FakeSearcher({"карта": ["chunk-1"], "карта fixed": ["chunk-1"]})
    report, traces = evaluate_rewriter(
        rewriter=MarkerRewriter(),
        slices={"noisy_freeform": cases, "clean": cases},
        searcher=searcher,
        corpus_version="corpus-test",
        index_version="index-test",
        samples=50,
    )
    written = write_report(report, tmp_path, traces=traces)
    names = {path.name for path in written}
    assert names == {"rewriter-eval.json", "rewriter-eval.md"}
    payload = json.loads((tmp_path / "rewriter-eval.json").read_text(encoding="utf-8"))
    assert payload["corpus_version"] == "corpus-test"
    assert payload["traces"]["noisy_freeform"][0]["case_id"] == "case-0"
    markdown = (tmp_path / "rewriter-eval.md").read_text(encoding="utf-8")
    assert "recall@20" in markdown and "clean-slice non-regression gate" in markdown


def test_empty_slices_are_skipped() -> None:
    report, traces = evaluate_rewriter(
        rewriter=IdentityRewriter(),
        slices={"noisy_freeform": [], "clean": []},
        searcher=FakeSearcher({}),
        samples=10,
    )
    assert report.n_cases == 0
    assert report.slices == []
    assert traces == {}
