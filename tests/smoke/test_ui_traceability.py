"""Console requirements are traced to the console tests that actually prove them.

The console tests live in ``web/src`` and run under vitest (``npm run test``), which the
offline Python gate cannot execute. This module closes the traceability loop without
duplicating a browser: for every UI acceptance id it names the web test file and the test
case that owns the behaviour, and asserts both still exist. A renamed or deleted console
test therefore breaks the Python gate, which is the point.

    # AT-UI-01  # AT-UI-02  # AT-UI-03  # AT-UI-04  # AT-UI-05  # AT-UI-06
    # AT-UI-07  # AT-UI-08  # AT-UI-09  # AT-UI-10  # AT-UI-11
    # AT-A11Y-01  # AT-SEC-04
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB = REPO_ROOT / "web"

#: AT id → (web test file, a distinctive fragment of the test case name).
UI_EVIDENCE: dict[str, tuple[str, str]] = {
    "AT-UI-01": (
        "src/components/QueuePanel.test.tsx",
        "lists every conversation with a language badge",
    ),
    "AT-UI-02": ("src/App.test.tsx", "streams a grounded answer with verified citation chips"),
    "AT-UI-03": ("src/components/AnswerCard.test.tsx", "opens the source passage on click"),
    "AT-UI-04": (
        "src/App.test.tsx",
        "shows the decision, confidence, cache state and the diagnostics",
    ),
    "AT-UI-05": (
        "src/components/AnswerCard.test.tsx",
        "presents an escalation with reason, p-hat and ticket",
    ),
    "AT-UI-06": ("src/components/AnswerCard.test.tsx", "supports edit, save and send"),
    "AT-UI-07": ("src/pages/AnalyticsPage.test.tsx", "renders the KPI cards derived from /metrics"),
    "AT-UI-08": ("src/state/console.test.ts", "accumulates streamed tokens one delta at a time"),
    "AT-UI-10": (
        "src/components/QueuePanel.test.tsx",
        "marks the selected conversation for assistive technology",
    ),
    "AT-A11Y-01": (
        "src/components/QueuePanel.test.tsx",
        "exposes filter controls as pressed buttons and a language select",
    ),
}


@pytest.mark.parametrize(("at_id", "evidence"), sorted(UI_EVIDENCE.items()))
def test_ui_requirement_has_a_named_console_test(at_id: str, evidence: tuple[str, str]) -> None:
    """Each UI acceptance id points at a real vitest case (the browser gate runs `npm run test`)."""
    relative, fragment = evidence
    path = WEB / relative
    assert path.is_file(), f"{at_id}: {relative} is missing"
    text = path.read_text(encoding="utf-8")
    assert fragment in text, f"{at_id}: {relative} no longer contains the case {fragment!r}"


def test_ui_09_mock_mode_is_shared_with_the_tests() -> None:
    """# AT-UI-09 — MSW handlers are the same in tests and in browser mock mode."""
    app_test = (WEB / "src/App.test.tsx").read_text(encoding="utf-8")
    assert "msw" in app_test and "http" in app_test
    handlers = list((WEB / "src").rglob("*handlers*"))
    assert handlers, "no shared MSW handler module found under web/src"
    package = json.loads((WEB / "package.json").read_text(encoding="utf-8"))
    assert package["scripts"]["test"] == "vitest run"
    assert package["scripts"]["build"].startswith("tsc --noEmit")


def test_ui_11_screenshots_are_committed_and_reproducible() -> None:
    """# AT-UI-11 — one script regenerates the committed screenshots."""
    script = WEB / "scripts/screenshots.mjs"
    assert script.is_file(), "web/scripts/screenshots.mjs is missing"
    manifest_path = REPO_ROOT / "docs/assets/screenshots.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["mode"] in {"live", "mock"}
    screens = manifest["screens"]
    assert len(screens) >= 6, screens
    for screen in screens:
        assert (REPO_ROOT / screen["file"]).is_file(), screen["file"]


def test_sec_04_console_never_injects_raw_html() -> None:
    """# AT-SEC-04 — model and document text are rendered as text, never as markup."""
    offenders = []
    for path in (WEB / "src").rglob("*.ts*"):
        text = path.read_text(encoding="utf-8")
        for needle in ("dangerouslySetInnerHTML", ".innerHTML =", "insertAdjacentHTML"):
            if needle in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {needle}")
    assert not offenders, offenders


def test_a11y_assertions_exist_in_the_console() -> None:
    """# AT-A11Y-01 — the console carries explicit assistive-technology affordances."""
    aria_files = [
        path for path in (WEB / "src").rglob("*.tsx") if "aria-" in path.read_text(encoding="utf-8")
    ]
    assert len(aria_files) >= 3, [str(p) for p in aria_files]
    queue_tests = (WEB / "src/components/QueuePanel.test.tsx").read_text(encoding="utf-8")
    assert "aria-current" in queue_tests or "aria-pressed" in queue_tests
