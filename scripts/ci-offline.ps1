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

    Write-Host "==> uv sync --extra dev" -ForegroundColor Cyan
    uv sync --extra dev
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

    Write-Host "==> ruff check src tests" -ForegroundColor Cyan
    uv run ruff check src tests
    if ($LASTEXITCODE -ne 0) { throw "ruff check failed" }

    Write-Host "==> ruff format --check src tests" -ForegroundColor Cyan
    uv run ruff format --check src tests
    if ($LASTEXITCODE -ne 0) { throw "ruff format check failed" }

    Write-Host "==> mypy src/answerdesk" -ForegroundColor Cyan
    uv run mypy src/answerdesk
    if ($LASTEXITCODE -ne 0) { throw "mypy failed" }

    Write-Host "==> pytest -q" -ForegroundColor Cyan
    uv run pytest -q
    if ($LASTEXITCODE -ne 0) { throw "pytest failed" }

    Write-Host "==> contracts import smoke" -ForegroundColor Cyan
    uv run python -c "import answerdesk; from answerdesk.contracts import Answer; print('ok')"
    if ($LASTEXITCODE -ne 0) { throw "contracts import failed" }

    Write-Host "offline gate: OK" -ForegroundColor Green
}
finally {
    Pop-Location
}
