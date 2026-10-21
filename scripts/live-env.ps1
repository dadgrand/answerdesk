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
    [ValidateSet('neuraldeep', 'deepseek', 'offline')]
    [string]$Provider = 'neuraldeep'
)

$repo = Split-Path -Parent $PSScriptRoot

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

if ($Provider -eq 'neuraldeep') {
    # Primary generator: Qwen3.6-35B-A3B served by NeuralDeep (OpenAI-compatible).
    $env:ANSWERDESK_LLM__PRESET = 'openai_compatible'
    $env:ANSWERDESK_LLM__BASE_URL = 'https://api.neuraldeep.ru/v1'
    $env:ANSWERDESK_LLM__MODEL = 'qwen3.6-35b-a3b'
    $env:ANSWERDESK_LLM__MAX_TOKENS = '4096'
    # Hosted bge-m3 embeddings + bge-reranker (t16 adapters). One NeuralDeep key
    # serves the generator, embeddings and reranker; it is read from .secrets and
    # never committed. Secrets are redacted by `answerdesk config show`.
    $env:OPENAI_COMPATIBLE_API_KEY = $key
    $env:ANSWERDESK_EMBEDDING__BACKEND = 'api'
    $env:ANSWERDESK_EMBEDDING__API_BASE_URL = 'https://api.neuraldeep.ru/v1'
    $env:ANSWERDESK_EMBEDDING__API_MODEL = 'bge-m3'
    $env:ANSWERDESK_EMBEDDING__API_DIMENSION = '1024'
    $env:ANSWERDESK_EMBEDDING__BATCH_SIZE = '32'
    $env:ANSWERDESK_RETRIEVAL__RERANK__ENABLED = 'true'
    $env:ANSWERDESK_RETRIEVAL__RERANK__BACKEND = 'api'
    $env:ANSWERDESK_RETRIEVAL__RERANK__API_BASE_URL = 'https://api.neuraldeep.ru/v1'
    $env:ANSWERDESK_RETRIEVAL__RERANK__API_MODEL = 'bge-reranker'
    Write-Host 'answerdesk: neuraldeep | qwen3.6-35b-a3b | bge-m3 + bge-reranker (api)'
}
else {
    # Secondary/comparison arm: DeepSeek flash.
    $env:ANSWERDESK_LLM__PRESET = 'deepseek'
    $env:ANSWERDESK_LLM__MODEL = 'deepseek-flash'
    $env:DEEPSEEK_API_KEY = $key
    Write-Host 'answerdesk: deepseek | generator deepseek-flash'
}
