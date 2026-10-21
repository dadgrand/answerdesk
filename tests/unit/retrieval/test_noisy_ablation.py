"""The noisy-input ablation: deltas, the clean non-regression gate, and the table.

Runs on a hand-built mini corpus (no generated corpus, no network), so the *machinery* is
covered offline; the measured numbers on the frozen corpus come from
``python -m answerdesk.retrieval.ablation`` (see ``docs/retrieval.md`` §15.3).

    # AT-RET-10
"""

from __future__ import annotations

from pathlib import Path

from answerdesk.contracts import Chunk
from answerdesk.index.bm25_index import BM25Index
from answerdesk.index.store import ChunkStore, EmbedderInfo, IndexBundle
from answerdesk.index.vector_index import ExactVectorIndex
from answerdesk.retrieval.ablation import (
    NoiseAblationReport,
    _format_noise_table,
    run_noise_ablation,
)
from answerdesk.retrieval.datasets import NOISY_CATEGORY, RetrievalCase
from answerdesk.retrieval.fusion import FusionParams
from answerdesk.retrieval.pipeline import PipelineConfig
from answerdesk.testing.doubles import HashingEmbedder

DOCS: dict[str, str] = {
    "doc-loan": (
        "Оформление займа: заявка, ставка и досрочное погашение. Займ оформляется в приложении."
    ),
    "doc-remit": (
        "Отслеживание перевода: статус розыска и сроки. Если деньги не дошли, подаётся заявка."
    ),
    "doc-card": ("Обслуживание карты: стоимость и перевыпуск. Карта блокируется при подозрении."),
}


def _chunk(doc_id: str, text: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::c{index:04d}",
        doc_id=doc_id,
        text=text,
        heading_path=["KB"],
        breadcrumb="KB",
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
    )


def _bundle() -> IndexBundle:
    chunks = [_chunk(doc_id, text, 0) for doc_id, text in DOCS.items()]
    store = ChunkStore.from_chunks(chunks, corpus_version="corpus-noise-test")
    embedder = HashingEmbedder(dimension=32)
    bm25 = BM25Index.from_chunks(chunks, text_for=ChunkStore.text_for)
    dense = ExactVectorIndex.from_chunks(chunks, embedder, text_for=ChunkStore.text_for)
    from answerdesk.index.bm25_index import BM25Params
    from answerdesk.index.store import IndexManifest, compute_index_version

    manifest = IndexManifest(
        corpus_version="corpus-noise-test",
        index_version=compute_index_version(
            corpus_version="corpus-noise-test",
            chunks_sha256=store.fingerprint(),
            embedder=EmbedderInfo(name=embedder.name, dimension=32),
            bm25_params=BM25Params(),
        ),
        chunk_count=len(chunks),
        chunks_sha256=store.fingerprint(),
        embedder=EmbedderInfo(name=embedder.name, dimension=32),
    )
    return IndexBundle(store, bm25, dense, manifest)


def _case(
    case_id: str, query: str, doc_id: str, *, category: str, style: str = ""
) -> RetrievalCase:
    return RetrievalCase(
        case_id=case_id,
        locale="ru",
        category=category,
        query=query,
        gold_document=doc_id,
        gold_chunk_ids=[f"{doc_id}::c0000"],
        style=style,
    )


def _cases() -> list[RetrievalCase]:
    return [
        _case("n1", "займ оформит как", "doc-loan", category=NOISY_CATEGORY, style="telegraphic"),
        _case(
            "n2",
            "первод денги не дошли",
            "doc-remit",
            category=NOISY_CATEGORY,
            style="typo_mix",
        ),
        _case("c1", "как оформить займ в приложении", "doc-loan", category="direct"),
        _case("c2", "стоимость обслуживания карты", "doc-card", category="direct"),
        _case("c3", "статус розыска перевода", "doc-remit", category="direct"),
    ]


def test_noise_ablation_reports_both_slices_and_all_arms() -> None:
    report = run_noise_ablation(
        index=_bundle(),
        cases=_cases(),
        params=FusionParams(fuser="rrf", rrf_k=60),
        config=PipelineConfig(top_k=10, gate_k=5, final_k=3),
        bootstrap=50,
        clean_sample=10,
    )
    assert isinstance(report, NoiseAblationReport)
    assert report.noisy_n == 2
    assert report.clean_n == 3
    assert {(cell.slice, cell.condition, cell.arm) for cell in report.cells} == {
        (slice_name, condition, arm)
        for slice_name in ("noisy", "clean")
        for condition in ("baseline", "noisy")
        for arm in ("bm25", "dense", "hybrid-rrf")
    }
    assert len(report.deltas) == 6
    for delta in report.deltas:
        assert -1.0 <= delta.delta_at_20 <= 1.0
    assert report.by_style  # telegraphic + typo_mix


def test_noise_ablation_clean_gate_is_reported() -> None:
    report = run_noise_ablation(
        index=_bundle(),
        cases=_cases(),
        params=FusionParams(fuser="rrf", rrf_k=60),
        config=PipelineConfig(top_k=10, gate_k=5, final_k=3),
        bootstrap=0,
        clean_sample=10,
    )
    assert report.clean_threshold == 0.01
    assert isinstance(report.clean_non_regression_ok, bool)
    assert report.bootstrap_samples == 0
    assert all(delta.ci_low is None for delta in report.deltas)


def test_noise_ablation_records_bootstrap_intervals_when_requested() -> None:
    report = run_noise_ablation(
        index=_bundle(),
        cases=_cases(),
        params=FusionParams(fuser="rrf", rrf_k=60),
        config=PipelineConfig(top_k=10, gate_k=5, final_k=3),
        bootstrap=50,
        clean_sample=10,
    )
    for delta in report.deltas:
        assert delta.ci_low is not None and delta.ci_high is not None
        assert delta.ci_low <= delta.ci_high


def test_noise_ablation_is_deterministic() -> None:
    kwargs = dict(
        index=_bundle(),
        cases=_cases(),
        params=FusionParams(fuser="rrf", rrf_k=60),
        config=PipelineConfig(top_k=10, gate_k=5, final_k=3),
        bootstrap=20,
        clean_sample=10,
    )
    first = run_noise_ablation(**kwargs).model_dump()  # type: ignore[arg-type]
    second = run_noise_ablation(**kwargs).model_dump()  # type: ignore[arg-type]
    assert first == second


def test_noise_ablation_skips_gracefully_without_a_noisy_slice() -> None:
    clean_only = [case for case in _cases() if case.category != NOISY_CATEGORY]
    report = run_noise_ablation(
        index=_bundle(),
        cases=clean_only,
        params=FusionParams(fuser="rrf", rrf_k=60),
        config=PipelineConfig(top_k=10, gate_k=5, final_k=3),
    )
    assert report.cells == []
    assert any("skipped" in note for note in report.notes)
    assert report.clean_non_regression_ok is True


def test_noise_table_documents_the_gate_and_the_styles() -> None:
    report = run_noise_ablation(
        index=_bundle(),
        cases=_cases(),
        params=FusionParams(fuser="rrf", rrf_k=60),
        config=PipelineConfig(top_k=10, gate_k=5, final_k=3),
        bootstrap=20,
        clean_sample=10,
    )
    lines = "\n".join(_format_noise_table(report))
    assert "Noisy-input ablation" in lines
    assert "clean non-regression gate" in lines
    assert "| noisy | bm25 |" in lines
    assert "Per-style" in lines
    assert "telegraphic" in lines
    assert _format_noise_table(None) == []


def test_the_measured_defaults_are_the_shipped_ones() -> None:
    """Guard the config the ablation validated: focus on, clause/expansion promotion off."""
    config = PipelineConfig()
    assert config.noisy_probes is True
    assert config.focus_probe is True
    assert config.clause_probes is False
    assert config.expansion_probe is False
    assert config.rewrite_probe is True


def test_reports_reference_the_shipped_config_file() -> None:
    path = Path(__file__).resolve().parents[3] / "configs" / "retrieval" / "noisy.yaml"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for knob in ("focus_probe", "clause_probes", "expansion_probe", "rewrite", "enabled"):
        assert knob in text
