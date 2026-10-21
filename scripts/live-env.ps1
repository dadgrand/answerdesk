# Live-provider setup for benchmark runs.
#
# Dot-source it so the variables land in your shell:
#   . .\scripts\live-env.ps1 neuraldeep
#   . .\scripts\live-env.ps1 deepseek
#   . .\scripts\live-env.ps1 offline
#
# Keys live in the gitignored .secrets/<provider>.key files and are never written
# into the repository, docs or reports. `answerdesk config show --profile api`
# prints them redacted.
param(
    [ValidateSet('neuraldeep', 'neuraldeep2', 'deepseek', 'offline')]
    [string]$Provider = 'neuraldeep',
    # Cross-family judge (requirements §10.5): default none = judge on the generator
    # (diagnostic only). Example: . .\scripts\live-env.ps1 neuraldeep -Judge deepseek
    [ValidateSet('none', 'neuraldeep', 'neuraldeep2', 'deepseek')]
    [string]$Judge = 'none',
    # Secondary generator arm reported side by side with the primary.
    [ValidateSet('none', 'neuraldeep', 'neuraldeep2', 'deepseek')]
    [string]$Secondary = 'deepseek'
)

$repo = Split-Path -Parent $PSScriptRoot

# Windows' WinINET proxy settings (read by httpx when trust_env=True) point at a SOCKS
# proxy httpx cannot drive without the socks extra. The providers are reachable directly,
# so bypass the proxy for the API hosts explicitly.
$env:NO_PROXY = 'api.neuraldeep.ru,api.deepseek.com,localhost,127.0.0.1'
$env:no_proxy = $env:NO_PROXY

function Read-Key([string]$name) {
    $path = Join-Path $repo ".secrets\$name.key"
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing provider key file: $path" }
    return (Get-Content -LiteralPath $path -Raw).Trim()
}

if ($Provider -eq 'offline') {
    $env:ANSWERDESK_PROFILE = 'offline'
    Write-Host 'answerdesk: offline profile (deterministic doubles, no network)'
    return
}

$keyFile = Join-Path $repo ".secrets\$Provider.key"
if (-not (Test-Path -LiteralPath $keyFile)) {
    throw "Missing provider key file: $keyFile"
}
$key = (Get-Content -LiteralPath $keyFile -Raw).Trim()

$env:ANSWERDESK_PROFILE = 'api'
$env:ANSWERDESK_LLM__API_KEY = $key

# --- judge provider (independent of the generator) -------------------------------
if ($Judge -ne 'none') {
    $judgeKey = Read-Key $Judge
    $env:ANSWERDESK_EVAL__JUDGE_API_KEY = $judgeKey
    if ($Judge -eq 'deepseek') {
        $env:ANSWERDESK_EVAL__JUDGE_PRESET = 'deepseek'
        $env:ANSWERDESK_EVAL__JUDGE_BASE_URL = 'https://api.deepseek.com/v1'
        $env:ANSWERDESK_EVAL__JUDGE_MODEL = 'deepseek-flash'
        $env:ANSWERDESK_EVAL__JUDGE_FAMILY = 'deepseek'
    }
    else {
        $env:ANSWERDESK_EVAL__JUDGE_PRESET = 'openai_compatible'
        $env:ANSWERDESK_EVAL__JUDGE_BASE_URL = 'https://api.neuraldeep.ru/v1'
        $env:ANSWERDESK_EVAL__JUDGE_MODEL = 'qwen3.6-35b-a3b-noreason'
        $env:ANSWERDESK_EVAL__JUDGE_FAMILY = 'qwen'
    }
    Write-Host "answerdesk: judge $Judge ($($env:ANSWERDESK_EVAL__JUDGE_MODEL))"
}

# --- secondary generator arm ----------------------------------------------------
if ($Secondary -ne 'none') {
    $secondaryKey = Read-Key $Secondary
    if ($Secondary -eq 'deepseek') {
        $env:DEEPSEEK_API_KEY = $secondaryKey
        $env:ANSWERDESK_EVAL__SECONDARY_PRESET = 'deepseek'
        $env:ANSWERDESK_EVAL__SECONDARY_MODEL = 'deepseek-flash'
        $env:ANSWERDESK_EVAL__SECONDARY_BASE_URL = 'https://api.deepseek.com/v1'
        $env:ANSWERDESK_EVAL__SECONDARY_API_KEY_ENV = 'DEEPSEEK_API_KEY'
    }
    else {
        $env:ANSWERDESK_EVAL__SECONDARY_PRESET = 'openai_compatible'
        $env:ANSWERDESK_EVAL__SECONDARY_MODEL = 'qwen3.6-35b-a3b-noreason'
        $env:ANSWERDESK_EVAL__SECONDARY_BASE_URL = 'https://api.neuraldeep.ru/v1'
        $env:ANSWERDESK_EVAL__SECONDARY_API_KEY = $secondaryKey
    }
    Write-Host "answerdesk: secondary $Secondary ($($env:ANSWERDESK_EVAL__SECONDARY_MODEL))"
}

if ($Provider -in @('neuraldeep', 'neuraldeep2')) {
    # Primary generator: Qwen3.6-35B-A3B served by NeuralDeep (OpenAI-compatible).
    # `neuraldeep2` is a second subscription used to rotate around the free tariff's
    # chat-session cooldown; the model and endpoints are identical.
    $env:ANSWERDESK_LLM__PRESET = 'openai_compatible'
    $env:ANSWERDESK_LLM__BASE_URL = 'https://api.neuraldeep.ru/v1'
    # The reasoning variant spends the whole JSON budget on its trace (measured:
    # 4096-token traces, PROVIDER_ERROR after the strict re-ask). Generation therefore
    # uses the same release's no-reason variant; docs/results.md records both ids.
    $env:ANSWERDESK_LLM__MODEL = 'qwen3.6-35b-a3b-noreason'
    $env:ANSWERDESK_LLM__MAX_TOKENS = '1024'
    # Hosted bge-m3 embeddings + bge-reranker (t16 adapters). One NeuralDeep key
    # serves the generator, embeddings and reranker; it is read from .secrets and
    # never committed. Secrets are redacted by `answerdesk config show`.
    $env:OPENAI_COMPATIBLE_API_KEY = $key
    $env:ANSWERDESK_EMBEDDING__BACKEND = 'api'
    $env:ANSWERDESK_EMBEDDING__API_BASE_URL = 'https://api.neuraldeep.ru/v1'
    $env:ANSWERDESK_EMBEDDING__API_MODEL = 'bge-m3'
    $env:ANSWERDESK_EMBEDDING__API_DIMENSION = '1024'
    $env:ANSWERDESK_EMBEDDING__BATCH_SIZE = '32'
    # The free tariff rate-limits bursts: pace embedding/query calls (the benchmark reads
    # this value through eval.min_call_interval_s, and the cached embedder honours it).
    $env:ANSWERDESK_EVAL__MIN_CALL_INTERVAL_S = '1.1'
    $env:ANSWERDESK_RETRIEVAL__RERANK__ENABLED = 'true'
    $env:ANSWERDESK_RETRIEVAL__RERANK__BACKEND = 'api'
    $env:ANSWERDESK_RETRIEVAL__RERANK__API_BASE_URL = 'https://api.neuraldeep.ru/v1'
    $env:ANSWERDESK_RETRIEVAL__RERANK__API_MODEL = 'bge-reranker'
    Write-Host 'answerdesk: neuraldeep | qwen3.6-35b-a3b | bge-m3 + bge-reranker (api)'
}
else {
    # Secondary/comparison arm: DeepSeek flash. Embeddings and reranking are only served
    # by NeuralDeep, so they are wired from the second subscription (the generator stays on
    # DeepSeek); without this the benchmark cannot build/load the index in this mode.
    $env:ANSWERDESK_LLM__PRESET = 'deepseek'
    $env:ANSWERDESK_LLM__MODEL = 'deepseek-flash'
    $env:DEEPSEEK_API_KEY = $key
    if (Test-Path -LiteralPath (Join-Path $repo '.secrets\neuraldeep2.key')) {
        $embedKey = Read-Key 'neuraldeep2'
        $env:ANSWERDESK_EMBEDDING__BACKEND = 'api'
        $env:ANSWERDESK_EMBEDDING__API_KEY = $embedKey
        $env:ANSWERDESK_EMBEDDING__API_BASE_URL = 'https://api.neuraldeep.ru/v1'
        $env:ANSWERDESK_EMBEDDING__API_MODEL = 'bge-m3'
        $env:ANSWERDESK_EMBEDDING__API_DIMENSION = '1024'
        $env:ANSWERDESK_EMBEDDING__BATCH_SIZE = '32'
        $env:ANSWERDESK_RETRIEVAL__RERANK__ENABLED = 'true'
        $env:ANSWERDESK_RETRIEVAL__RERANK__BACKEND = 'api'
        $env:ANSWERDESK_RETRIEVAL__RERANK__API_KEY = $embedKey
        $env:ANSWERDESK_RETRIEVAL__RERANK__API_BASE_URL = 'https://api.neuraldeep.ru/v1'
        $env:ANSWERDESK_RETRIEVAL__RERANK__API_MODEL = 'bge-reranker'
    }
    Write-Host 'answerdesk: deepseek | generator deepseek-flash'
}
