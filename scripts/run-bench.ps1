# One-command reproduction of the t11 evidence package.
#
#   .\scripts\run-bench.ps1                      # live: qwen primary, deepseek judge+secondary
#   .\scripts\run-bench.ps1 -Replay              # deterministic doubles, REPLAY-MODE artefacts
#   .\scripts\run-bench.ps1 -AnswerLimit 120     # smaller generation sample
#
# Steps, in order: environment → corpus → index (hosted bge-m3, cached) → frozen golden set
# → benchmark suite → reasoning side study → docs/results.md (strictly from artefacts).
#
# Keys come from the git-ignored .secrets/ directory via scripts/live-env.ps1; nothing here
# writes a secret into the repository, a report or a log.
param(
    [switch]$Replay,
    [int]$AnswerLimit = 400,
    [int]$RetrievalLimit = 540,
    [int]$SideStudyCases = 32,
    [string]$Run = 'reports/bench',
    [switch]$SkipIndex
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    if ($Replay) {
        . .\scripts\live-env.ps1 offline
        $provider = 'fake'
    }
    else {
        . .\scripts\live-env.ps1 neuraldeep -Judge deepseek -Secondary deepseek
        $provider = 'api'
    }

    Write-Host '== corpus (deterministic build)'
    uv run answerdesk ingest --demo

    if (-not $Replay -and -not $SkipIndex) {
        Write-Host '== index (hosted bge-m3, content-addressed cache under data/index)'
        uv run answerdesk index --demo --force
    }

    Write-Host "== benchmark (provider=$provider, run=$Run)"
    uv run answerdesk eval --all --provider $provider --report $Run `
        --answer-limit $AnswerLimit --retrieval-limit $RetrievalLimit --regenerate

    if (-not $Replay) {
        Write-Host '== reasoning-variant side study'
        uv run python scripts/reasoning-side-study.py --limit $SideStudyCases --budgets 8192 16384 --run $Run
    }

    Write-Host '== docs/results.md (generated from artefacts only)'
    uv run python scripts/build-results.py --run $Run --out docs/results.md
    uv run python scripts/build-results.py --run $Run --out docs/results.md --check
}
finally {
    Pop-Location
}
