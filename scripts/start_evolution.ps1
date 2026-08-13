# SC2-Commander 自动策略进化（Windows）
# 直接改下面「配置区」后执行: .\scripts\start_evolution.ps1
# 命令行参数若传入会覆盖配置区对应项。
param(
    [string]$STRATEGY = "",
    [string]$COMMANDER_MODEL = "",
    [string]$EVOLUTION_MODEL = "",
    [string]$DIFFICULTIES = "",
    [Nullable[int]]$MATCHES = $null,
    [Nullable[int]]$CONCURRENCY = $null,
    [Nullable[int]]$MAX_GENERATIONS = $null,
    [string]$RUN_DIR = ""
)

# =============================================================================
# 配置区（按需修改）
# =============================================================================
$CFG_STRATEGY = "tank"
$CFG_COMMANDER_MODEL = "deepseek-v4-flash"
$CFG_EVOLUTION_MODEL = "deepseek-v4-flash"   # 空字符串 = 与 commander 相同
$CFG_DIFFICULTIES = "harder,veryhard,cheatvision,cheatmoney,cheatinsane"
$CFG_MATCHES = 10
$CFG_CONCURRENCY = 5
$CFG_MAX_GENERATIONS = 10
$CFG_RUN_DIR = ""                       # 续跑时填 evolution_runs\... 路径
# =============================================================================

$ErrorActionPreference = "Stop"

if (-not [string]::IsNullOrWhiteSpace($STRATEGY)) { $CFG_STRATEGY = $STRATEGY }
if (-not [string]::IsNullOrWhiteSpace($COMMANDER_MODEL)) { $CFG_COMMANDER_MODEL = $COMMANDER_MODEL }
if (-not [string]::IsNullOrWhiteSpace($EVOLUTION_MODEL)) { $CFG_EVOLUTION_MODEL = $EVOLUTION_MODEL }
if (-not [string]::IsNullOrWhiteSpace($DIFFICULTIES)) { $CFG_DIFFICULTIES = $DIFFICULTIES }
if ($null -ne $MATCHES) { $CFG_MATCHES = [int]$MATCHES }
if ($null -ne $CONCURRENCY) { $CFG_CONCURRENCY = [int]$CONCURRENCY }
if ($null -ne $MAX_GENERATIONS) { $CFG_MAX_GENERATIONS = [int]$MAX_GENERATIONS }
if (-not [string]::IsNullOrWhiteSpace($RUN_DIR)) { $CFG_RUN_DIR = $RUN_DIR }

if ([string]::IsNullOrWhiteSpace($CFG_STRATEGY) -or [string]::IsNullOrWhiteSpace($CFG_COMMANDER_MODEL)) {
    Write-Error "STRATEGY and COMMANDER_MODEL must be set (in config block or via -STRATEGY / -COMMANDER_MODEL)."
    exit 1
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Error "Python virtual environment not found: $python"
    exit 1
}

Write-Host "Strategy     : $CFG_STRATEGY"
Write-Host "Commander    : $CFG_COMMANDER_MODEL"
if (-not [string]::IsNullOrWhiteSpace($CFG_EVOLUTION_MODEL)) {
    Write-Host "Evolution    : $CFG_EVOLUTION_MODEL"
}
Write-Host "Difficulties : $CFG_DIFFICULTIES"
Write-Host "Matches/gen  : $CFG_MATCHES, concurrency=$CFG_CONCURRENCY, max_gen=$CFG_MAX_GENERATIONS"
if (-not [string]::IsNullOrWhiteSpace($CFG_RUN_DIR)) {
    Write-Host "Resume dir   : $CFG_RUN_DIR"
}

$arguments = @(
    "-m", "evolution",
    "--strategy", $CFG_STRATEGY,
    "--commander-model", $CFG_COMMANDER_MODEL,
    "--difficulties", $CFG_DIFFICULTIES,
    "--matches", $CFG_MATCHES,
    "--concurrency", $CFG_CONCURRENCY,
    "--max-generations", $CFG_MAX_GENERATIONS
)
if (-not [string]::IsNullOrWhiteSpace($CFG_EVOLUTION_MODEL)) {
    $arguments += @("--evolution-model", $CFG_EVOLUTION_MODEL)
}
if (-not [string]::IsNullOrWhiteSpace($CFG_RUN_DIR)) {
    $arguments += @("--run-dir", $CFG_RUN_DIR)
}

Set-Location $repoRoot
& $python @arguments
exit $LASTEXITCODE
