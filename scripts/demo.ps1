# One-command offline demo for answerdesk.
#
#   pwsh -File scripts/demo.ps1
#
# What it does, in order: install the light environment, build the synthetic corpus, build the
# index, ask one question end-to-end, fit a calibration artifact, run the offline evaluation
# smoke, and health-check the HTTP service. No API key, no network, no Docker.
#
# It writes only under data/, reports/demo/ and logs/ (all gitignored runtime artifacts).
# AT-OPS-06 (one-command demo), NFR-DET-01 (offline, keyless).

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    # httpx reads the Windows proxy settings; a SOCKS URL it cannot drive breaks client
    # construction even for a fake provider. Loopback traffic must bypass it.
    $env:NO_PROXY = 'localhost,127.0.0.1,::1'
    $env:no_proxy = $env:NO_PROXY
    $env:ANSWERDESK_PROFILE = 'offline'

    $run = 'reports/demo'
    $question = 'Сколько стоит обслуживание карты?'

    function Step([string]$label, [scriptblock]$body) {
        Write-Host "==> $label" -ForegroundColor Cyan
        & $body
        if ($LASTEXITCODE -ne 0) { throw "step failed: $label (exit $LASTEXITCODE)" }
    }

    Step '1/6 install (light: no torch, no service deps)' { uv sync --extra dev }
    Step '2/6 build the synthetic corpus (6 locales)' { uv run answerdesk ingest --demo }
    Step '3/6 build BM25 + dense indexes' { uv run answerdesk index --demo }
    Step '4/6 ask one question end-to-end' { uv run answerdesk ask --fake $question }
    Step '5/6 fit the calibration artifact' {
        uv run answerdesk calibrate --synthetic --out "$run/policy"
    }
    Step '6/6 offline evaluation smoke' {
        uv run answerdesk eval --synthetic --provider fake --report "$run/offline-smoke"
    }
    Step 'health check the HTTP service' {
        uv run --extra api answerdesk serve --fake --port 8765 --healthcheck
    }

    Write-Host ''
    Write-Host 'offline demo: OK' -ForegroundColor Green
    Write-Host "  evaluation report : $run/offline-smoke/report.md"
    Write-Host "  calibration       : $run/policy/"
    Write-Host ''
    Write-Host 'Next steps'
    Write-Host '  service + console : uv run --extra api answerdesk serve --fake --port 8765'
    Write-Host '                      cd web && npm ci && npm run dev   (http://localhost:5173)'
    Write-Host '  real model        : cp .env.example .env, set a key, then --profile api'
    Write-Host '  live evidence     : pwsh -File scripts/run-bench.ps1   (needs keys; rewrites docs/results.md)'
}
finally {
    Pop-Location
}
