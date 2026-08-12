param(
    [Parameter(Mandatory = $true)][string]$STRATEGY,
    [Parameter(Mandatory = $true)][string]$COMMANDER_MODEL,
    [string]$EVOLUTION_MODEL = "",
    [string]$DIFFICULTIES = "harder,veryhard,cheatvision,cheatmoney,cheatinsane",
    [int]$MATCHES = 10,
    [int]$CONCURRENCY = 5,
    [int]$MAX_GENERATIONS = 10,
    [string]$RUN_DIR = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Error "Python virtual environment not found: $python"
    exit 1
}

$arguments = @(
    "-m", "evolution",
    "--strategy", $STRATEGY,
    "--commander-model", $COMMANDER_MODEL,
    "--difficulties", $DIFFICULTIES,
    "--matches", $MATCHES,
    "--concurrency", $CONCURRENCY,
    "--max-generations", $MAX_GENERATIONS
)
if (-not [string]::IsNullOrWhiteSpace($EVOLUTION_MODEL)) {
    $arguments += @("--evolution-model", $EVOLUTION_MODEL)
}
if (-not [string]::IsNullOrWhiteSpace($RUN_DIR)) {
    $arguments += @("--run-dir", $RUN_DIR)
}

Set-Location $repoRoot
& $python @arguments
exit $LASTEXITCODE
