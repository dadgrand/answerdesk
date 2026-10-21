"""Consistency-audit harness: offline tests (no network, deterministic doubles).

# AT-CNS-01  # AT-CNS-02  # AT-CNS-03  # AT-CNS-04
"""

from __future__ import annotations

import json
from pathlib import Path

from answerdesk.eval.consistency import agreement as metrics
from answerdesk.eval.consistency.facts import build_groups, write_set
from answerdesk.eval.consistency.logical import audit_corpus, audit_document
from answerdesk.eval.consistency.runner import ConsistencyConfig, build_set, run_answers, run_audit
from answerdesk.eval.consistency.values import extract, fidelity, language_discipline
from answerdesk.eval.consistency.variants import variants_for


def _doc(
    *,
    doc_id: str,
    language: str,
    archetype: str,
    title: str,
    body: str,
    as_of: str = "24 ноября 2026",
) -> str:
    return (
        "---\n"
        f'doc_id: "{doc_id}"\n'
        f'title: "{title}"\n'
        f'language: "{language}"\n'
        f'intent: "{archetype}"\n'
        f'archetype: "{archetype}"\n'
        'license: "CC0-1.0"\n'
        "---\n\n"
        f"# {title}\n\n{body}\n\n## Контакты\n\nАктуально на {as_of}.\n"
        "Номер обращения: ORD-2026-10029.\n"
    )


def _corpus(tmp_path: Path, docs: dict[str, str], locales: tuple[str, ...] = ("ru",)) -> Path:
    root = tmp_path / "corpus"
    for doc_id, text in docs.items():
        locale = doc_id.split("-")[1]
        directory = root / "docs" / locale
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{doc_id}.md").write_text(text, encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"corpus_version": "corpus-test"}), encoding="utf-8"
    )
    (root / "queries").mkdir(exist_ok=True)
    (root / "queries" / "queries.jsonl").write_text("", encoding="utf-8")
    return root


# --------------------------------------------------------------------------------------
# values, fidelity, language
# --------------------------------------------------------------------------------------


def test_extract_values_normalises_and_separates_kinds() -> None:
    text = "Тариф 400 RUB в месяц, комиссия 1,5%, срок до 3 рабочих дней, код ORD-2026-10029."
    values = extract(text)
    assert "400" in values.amounts
    assert "1,5" in values.percentages or "1.5" in values.percentages
    assert any("3" in value for value in values.day_counts)
    assert "ORD-2026-10029" in values.identifiers


def test_fidelity_flags_unsupported_value() -> None:
    gold = "Тариф стоит 400 RUB в месяц, срок 3 рабочих дня."
    assert fidelity("Тариф стоит 400 RUB.", gold)["fidelity"] == 1.0
    bad = fidelity("Тариф стоит 900 RUB.", gold)
    assert bad["fidelity"] < 1.0
    assert bad["unsupported"] == ["amount:900"]


def test_language_discipline_detects_wrong_language() -> None:
    assert language_discipline(query_locale="ru", answer="Тариф можно изменить в приложении.")["ok"]
    assert language_discipline(query_locale="zh", answer="可在应用或网点提交申请。")["ok"]
    wrong = language_discipline(query_locale="uz", answer="Тариф можно изменить в приложении.")
    assert wrong["ok"] is False


# --------------------------------------------------------------------------------------
# surface variants
# --------------------------------------------------------------------------------------


def test_variants_are_deterministic_and_distinct() -> None:
    text = "Как оформить займ или кредит в приложении?"
    first = [variant.to_dict() for variant in variants_for("ru", text, seed=3)]
    second = [variant.to_dict() for variant in variants_for("ru", text, seed=3)]
    assert first == second
    kinds = [variant["kind"] for variant in first]
    assert kinds == ["clean", "telegraphic", "typo", "translit", "code_switch"]
    assert first[1]["text"] != first[0]["text"]
    assert first[3]["text"] != first[0]["text"]


def test_invariance_metrics_require_matching_evidence_and_decision() -> None:
    rows = []
    for kind, evidence, decision in (
        ("clean", ["d1"], "answer"),
        ("telegraphic", ["d1"], "answer"),
        ("typo", ["d2"], "answer"),
        ("translit", ["d1"], "escalate"),
    ):
        rows.append(
            {"fact_id": "f1", "variant": kind, "evidence_doc_ids": evidence, "decision": decision}
        )
    payload = metrics.invariance(rows)
    assert payload["pairs"] == 3
    assert payload["evidence_invariance"]["successes"] == 2
    assert payload["decision_invariance"]["successes"] == 2


# --------------------------------------------------------------------------------------
# logical audit
# --------------------------------------------------------------------------------------


def test_audit_detects_table_parameter_conflict(tmp_path: Path) -> None:
    text = _doc(
        doc_id="syn-ru-0001",
        language="ru",
        archetype="tariff_schedule",
        title="Тарифы — Стартовый",
        body="| Параметр | Значение |\n|---|---|\n| Абонплата | 400 RUB |\n| Абонплата | 900 RUB |",
    )
    path = tmp_path / "syn-ru-0001.md"
    path.write_text(text, encoding="utf-8")
    findings = audit_document(path, locale="ru")
    assert any(finding.code == "table_parameter_conflict" for finding in findings)


def test_audit_detects_wrong_locale_directory(tmp_path: Path) -> None:
    text = _doc(
        doc_id="syn-ru-0002",
        language="ru",
        archetype="fee_table",
        title="Комиссии",
        body="Комиссия 1%.",
    )
    path = tmp_path / "syn-ru-0002.md"
    path.write_text(text, encoding="utf-8")
    findings = audit_document(path, locale="uz")
    assert any(finding.code == "front_matter_language" for finding in findings)


def test_audit_detects_date_after_as_of(tmp_path: Path) -> None:
    text = _doc(
        doc_id="syn-ru-0003",
        language="ru",
        archetype="remittance_tracking",
        title="Переводы",
        body="Зачисление произойдёт 12 января 2027.",
        as_of="24 ноября 2026",
    )
    path = tmp_path / "syn-ru-0003.md"
    path.write_text(text, encoding="utf-8")
    findings = audit_document(path, locale="ru")
    assert any(finding.code == "date_after_as_of" for finding in findings)


def test_cross_document_conflict_fails_the_invariant(tmp_path: Path) -> None:
    one = _doc(
        doc_id="syn-ru-0001",
        language="ru",
        archetype="tariff_schedule",
        title="Тарифы — Стартовый",
        body="| Параметр | Значение |\n|---|---|\n| Абонплата | 400 RUB |",
    )
    two = _doc(
        doc_id="syn-ru-0002",
        language="ru",
        archetype="tariff_schedule",
        title="Тарифы — Стартовый",
        body="| Параметр | Значение |\n|---|---|\n| Абонплата | 900 RUB |",
    )
    root = _corpus(tmp_path, {"syn-ru-0001": one, "syn-ru-0002": two})
    payload = audit_corpus(root, locales=("ru",))
    # cross-document same-product conflicts are filed as HIGH (the corpus draws slot values
    # per document, so a shared product name with two values is a real inconsistency but not
    # a self-contradicting single document, which is what BLOCKER means)
    assert payload["by_severity"].get("high", 0) >= 1
    assert payload["by_severity"].get("blocker", 0) == 0
    assert payload["invariant_holds"] is False


def test_clean_corpus_passes_the_invariant(tmp_path: Path) -> None:
    one = _doc(
        doc_id="syn-ru-0001",
        language="ru",
        archetype="tariff_schedule",
        title="Тарифы — Стартовый",
        body="| Параметр | Значение |\n|---|---|\n| Абонплата | 400 RUB |",
    )
    two = _doc(
        doc_id="syn-ru-0002",
        language="ru",
        archetype="tariff_schedule",
        title="Тарифы — Премиум",
        body="| Параметр | Значение |\n|---|---|\n| Абонплата | 900 RUB |",
    )
    root = _corpus(tmp_path, {"syn-ru-0001": one, "syn-ru-0002": two})
    payload = audit_corpus(root, locales=("ru",))
    assert payload["invariant_holds"] is True


# --------------------------------------------------------------------------------------
# fact set and runner (offline)
# --------------------------------------------------------------------------------------


def test_build_set_from_mini_corpus(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    (root / "queries").mkdir(parents=True)
    rows = []
    for locale in ("ru", "uz"):
        for index in range(2):
            rows.append(
                {
                    "query_id": f"q-{locale}-{index}",
                    "locale": locale,
                    "doc_id": f"syn-{locale}-{index:04d}",
                    "archetype": "loan_credit",
                    "category": "direct",
                    "base_text": f"вопрос {index}",
                    "text": f"вопрос {index}",
                    "operator_reply": f"ответ {index}",
                    "supporting_doc_ids": [f"syn-{locale}-{index:04d}"],
                    "answerable": True,
                    "style": "",
                }
            )
    (root / "queries" / "queries.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    groups = build_groups(root, locales=("ru", "uz"))
    assert len(groups) == 2
    out = tmp_path / "set"
    manifest = write_set(groups, out, corpus_version="corpus-test")
    assert manifest["facts"] == 2
    assert manifest["instances"] == 4
    assert (out / "facts.jsonl").is_file()


def test_runner_replay_is_offline_and_writes_artefacts(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    (root / "queries").mkdir(parents=True)
    rows = []
    documents: dict[str, str] = {}
    for locale in ("ru",):
        for index in range(2):
            doc_id = f"syn-{locale}-{index:04d}"
            rows.append(
                {
                    "query_id": f"q-{locale}-{index}",
                    "locale": locale,
                    "doc_id": doc_id,
                    "archetype": "loan_credit",
                    "category": "direct",
                    "base_text": "Как оформить займ?",
                    "text": "Как оформить займ?",
                    "operator_reply": "Займ оформляется в приложении, решение до 2 рабочих дней.",
                    "supporting_doc_ids": [doc_id],
                    "answerable": True,
                    "style": "",
                }
            )
            documents[doc_id] = _doc(
                doc_id=doc_id,
                language=locale,
                archetype="loan_credit",
                title="Кредиты — Стартовый",
                body="Займ оформляется в приложении, решение до 2 рабочих дней.",
            )
    (root / "queries" / "queries.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (root / "manifest.json").write_text(
        json.dumps({"corpus_version": "corpus-test"}), encoding="utf-8"
    )
    for doc_id, text in documents.items():
        directory = root / "docs" / "ru"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{doc_id}.md").write_text(text, encoding="utf-8")
    set_dir = tmp_path / "set"
    config = ConsistencyConfig(
        run_id="consistency-test",
        report_dir=tmp_path / "reports",
        corpus_dir=root,
        set_dir=set_dir,
        locales=("ru",),
        variant_locales=("ru",),
        replay=True,
        samples=100,
    )
    manifest = build_set(config)
    assert manifest["facts"] == 2
    audit = run_audit(config)
    assert audit["invariant_holds"] is True
    summary = run_answers(config)
    assert summary["rows"]["total"] >= 2
    assert summary["generator_model"] == "fake-model"
    assert (config.report_dir / "summary.json").is_file()
    assert (config.report_dir / "report.md").is_file()
    assert (config.report_dir / "logical-audit.json").is_file()
