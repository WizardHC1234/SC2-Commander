$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$scriptDir = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptDir)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).ProviderPath
Set-Location -LiteralPath $repoRoot

if (-not $env:SC2PATH -or -not (Test-Path -LiteralPath $env:SC2PATH)) {
    $env:SC2PATH = "D:\StarCraft II"
}

$MAP_NAME = "KairosJunctionLE"
$ENEMY_RACE = "terran"
# $ENEMY_RACE = "protoss"
$ENEMY_DIFFICULTY = "harder"
$ENEMY_BUILD = "macro"
$BOT_RACE = "terran"
$FORCE_STRATEGY = "tank_opt1"
# $COMMANDER_MODEL = "qwen3-32b"
$COMMANDER_MODEL = "deepseek-v4-flash"
# $COMMANDER_MODEL = "qwen3.5-27b"

$configPath = Join-Path $repoRoot "llm\config.json"
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    Write-Error "Missing llm/config.json. Copy llm/config.example.json and fill in keys."
    exit 1
}

function Test-UsablePython([string]$ExePath) {
    if ([string]::IsNullOrWhiteSpace($ExePath)) { return $false }
    if (-not (Test-Path -LiteralPath $ExePath -PathType Leaf)) { return $false }
    if ($ExePath -like "*\WindowsApps\python.exe") { return $false }
    & $ExePath -c "import openai" 1>$null 2>$null
    return ($LASTEXITCODE -eq 0)
}

$candidates = @()
if (-not [string]::IsNullOrWhiteSpace($env:VIRTUAL_ENV)) {
    $candidates += (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}
$candidates += (Join-Path $repoRoot "venv\Scripts\python.exe")
$cmd = Get-Command "python.exe" -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if ($cmd -and $cmd.Source) {
    $candidates += [string]$cmd.Source
}

$PYTHON_EXE = ""
foreach ($cand in $candidates) {
    $path = [string]$cand
    Write-Output "Trying Python: $path"
    if (Test-UsablePython $path) {
        $PYTHON_EXE = $path
        break
    }
}

if ([string]::IsNullOrWhiteSpace($PYTHON_EXE)) {
    $expected = Join-Path $repoRoot "venv\Scripts\python.exe"
    Write-Error "No usable Python found (need import openai). Expected: $expected"
    exit 1
}

Write-Output "Using Python: $PYTHON_EXE"
Write-Output "Repo root: $repoRoot"

$runScript = Join-Path $repoRoot "run_vs_ai.py"
$pyArgs = @(
    $runScript,
    "--my-bot-name", "commander",
    "--map-name", $MAP_NAME,
    "--bot-race", $BOT_RACE,
    "--enemy-race", $ENEMY_RACE,
    "--enemy-difficulty", $ENEMY_DIFFICULTY,
    "--enemy-build", $ENEMY_BUILD,
    "--force-strategy", $FORCE_STRATEGY,
    "--commander-model", $COMMANDER_MODEL
)

& $PYTHON_EXE @pyArgs
exit $LASTEXITCODE