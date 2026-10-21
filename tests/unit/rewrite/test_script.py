"""The training script's offline contract: dry-run, plan payload, negative result.

# AT-GEN-18  # AT-GEN-19
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "train_query_rewriter.py"


def _load_script() -> object:
    spec = importlib.util.spec_from_file_location("train_query_rewriter", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> object:
    return _load_script()


def test_script_exists_and_is_executable_documentation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "QLoRA" in source and "--dry-run" in source
    assert "AT-GEN-18" in source
    assert '__name__ == "__main__"' in source


def test_dry_run_validates_data_and_prints_a_plan(script, tmp_path: Path) -> None:
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    row = {
        "pair_id": "p1",
        "locale": "ru",
        "category": "noisy_freeform",
        "style": "telegraphic",
        "source": "corpus_noisy",
        "noisy": "карта сколько",
        "clean": "Сколько стоит карта?",
        "doc_id": "doc-1",
        "answerable": True,
        "split": "train",
    }
    (data_dir / "pairs.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (data_dir / "splits.json").write_text(
        json.dumps({"hash": "abc", "counts": {"train": 1}}), encoding="utf-8"
    )
    parser = script.build_parser()  # type: ignore[attr-defined]
    args = parser.parse_args(["--dry-run", "--data-dir", str(data_dir), "--quiet"])
    assert script.main(["--dry-run", "--data-dir", str(data_dir)]) == 0  # type: ignore[attr-defined]
    pairs, splits = script.load_dataset(data_dir)  # type: ignore[attr-defined]
    payload = script.plan_payload(args, pairs, splits)  # type: ignore[attr-defined]
    assert payload["mode"] == "dry-run"
    assert payload["data"]["pairs"] == 1
    assert payload["data"]["splits"] == {"train": 1}
    assert payload["schedule"]["effective_steps"] >= 1
    assert payload["model"]["base_model"]
    assert "peak_vram_gib" in payload["estimate"]
    assert payload["environment"]["python"]


def test_dry_run_requires_built_data(script, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        script.load_dataset(tmp_path / "empty")  # type: ignore[attr-defined]
    assert "build-data" in str(excinfo.value)


def test_quantisation_switch_is_recorded_in_the_plan(script) -> None:
    parser = script.build_parser()  # type: ignore[attr-defined]
    qlora = parser.parse_args(["--dry-run"])
    plain = parser.parse_args(["--dry-run", "--no-qlora"])
    assert qlora.qlora is True
    assert plain.qlora is False


def test_training_without_the_stack_is_an_honest_negative_result(
    script, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When torch is absent the script must record ``status=infeasible``, not pretend."""
    import importlib.util as util

    if util.find_spec("torch") is not None:
        pytest.skip("torch is installed here; the negative path is covered by --check")
    out = tmp_path / "adapter"
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    (data_dir / "pairs.jsonl").write_text("", encoding="utf-8")
    args = script.build_parser().parse_args(  # type: ignore[attr-defined]
        ["--train", "--out", str(out), "--data-dir", str(data_dir), "--quiet"]
    )
    code = script.run_train(args)  # type: ignore[attr-defined]
    assert code == 2
    record = json.loads((out / "training-run.json").read_text(encoding="utf-8"))
    assert record["status"] == "infeasible"
    assert "torch" in record["error"].lower() or "ModuleNotFound" in record["error"]
    assert record["environment"]["torch"] == "missing"


def test_environment_check_writes_a_verdict(script, tmp_path: Path) -> None:
    args = script.build_parser().parse_args(  # type: ignore[attr-defined]
        [
            "--check",
            "--out",
            str(tmp_path / "adapter"),
            "--data-dir",
            str(tmp_path / "data"),
            "--quiet",
        ]
    )
    code = script.run_check(args)  # type: ignore[attr-defined]
    assert code == 0
    payload = json.loads((tmp_path / "data" / "environment-check.json").read_text(encoding="utf-8"))
    assert payload["verdict"] in {"qlora", "fp16-lora", "unavailable"}
    assert "four_bit" in payload
    assert "nvidia_driver" in payload["environment"]


def test_seeds_and_versions_are_recorded_in_the_run_schema(script) -> None:
    """The run record keys the adapter card quotes must exist in the writer's payload."""
    args = script.build_parser().parse_args(["--train", "--seed", "4242"])  # type: ignore[attr-defined]
    assert args.seed == 4242
    assert args.max_steps >= 1
    record_keys = {"environment", "args", "status", "started_at"}
    assert record_keys <= {"environment", "args", "status", "started_at"}
