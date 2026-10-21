# Offline quality gate — the same commands CI runs, for a local machine.
#
#   pwsh -File scripts/ci-offline.ps1
#
# No network, no API key, no Docker. The socket guard in tests/conftest.py fails any test
# that tries to open a connection (docs/requirements.md NFR-DET-01).
# AT-DET-01, AT-DET-03, AT-QUAL-01

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    $env:ANSWERDESK_PROFILE = "offline"

    Write-Host "==> uv sync --extra dev --extra api" -ForegroundColor Cyan
    uv sync --extra dev --extra api
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

    Write-Host "==> answerdesk ingest --demo (offline corpus)" -ForegroundColor Cyan
    uv run answerdesk ingest --demo
    if ( -ne 0) { throw "corpus build failed" }

    Write-Host "==> ruff check src tests" -ForegroundColor Cyan
    uv run ruff check src tests
    if ($LASTEXITCODE -ne 0) { throw "ruff check failed" }

    Write-Host "==> ruff format --check src tests" -ForegroundColor Cyan
    uv run ruff format --check src tests
    if ($LASTEXITCODE -ne 0) { throw "ruff format check failed" }

    Write-Host "==> mypy src/answerdesk" -ForegroundColor Cyan
    uv run mypy src/answerdesk
    if ($LASTEXITCODE -ne 0) { throw "mypy failed" }

    Write-Host "==> traceability matrix (CMD-T)" -ForegroundColor Cyan
    uv run python scripts/check-traceability.py
    if ($LASTEXITCODE -ne 0) { throw "traceability check failed" }

    Write-Host "==> published numbers (CMD-NUM)" -ForegroundColor Cyan
    uv run python scripts/check-numbers.py
    if ($LASTEXITCODE -ne 0) { throw "number check failed" }

    Write-Host "==> licence audit (CMD-LIC)" -ForegroundColor Cyan
    uv run python scripts/check-licenses.py --quiet
    if ($LASTEXITCODE -ne 0) { throw "licence audit failed" }

    Write-Host "==> pytest -q" -ForegroundColor Cyan
    uv run pytest -q
    if ($LASTEXITCODE -ne 0) { throw "pytest failed" }

    # NFR-QUAL-02: line coverage >= 70% for the four math packages. Runs after the plain
    # suite so a coverage failure is attributable to the threshold, not to a test failure.
    Write-Host "==> coverage gate >= 70% (retrieval, policy, cache, eval)" -ForegroundColor Cyan
    uv run pytest -q `
        --cov=src/answerdesk/retrieval --cov=src/answerdesk/policy `
        --cov=src/answerdesk/cache --cov=src/answerdesk/eval `
        --cov-report=term --cov-fail-under=70
    if ($LASTEXITCODE -ne 0) { throw "coverage gate failed" }

    Write-Host "==> contracts import smoke" -ForegroundColor Cyan
    uv run python -c "import answerdesk; from answerdesk.contracts import Answer; print('ok')"
    if ($LASTEXITCODE -ne 0) { throw "contracts import failed" }

    Write-Host "offline gate: OK" -ForegroundColor Green
}
finally {
    Pop-Location
}
