# SC2-Commander 自动策略进化（Windows）
# 直接改下面「配置区」后执行: .\scripts\start_evolution.ps1
# 命令行参数若传入会覆盖配置区对应项。
param(
    [string]$STRATEGY = "",
    [string]$COMMANDER_MODEL = "",
    [string]$EVOLUTION_MODEL = "",
    [string]$DIFFICULTIES = "",
    [Nullable[int]]$MATCHES = $null,
    [Nullable[int]]$CANDIDATE_MATCHES = $null,
    [Nullable[int]]$CONCURRENCY = $null,
    [Nullable[int]]$MAX_GENERATIONS = $null,
    [Nullable[int]]$MAX_GENERATIONS_PER_DIFFICULTY = $null,
    [Nullable[double]]$MASTERY_SCORE_THRESHOLD = $null,
    [string]$RUN_DIR = "",
    [string]$BASELINE_BATCH_DIR = ""
)

# =============================================================================
# 配置区（按需修改）
# =============================================================================
$CFG_STRATEGY = "tank"
$CFG_COMMANDER_MODEL = "deepseek-v4-flash"
$CFG_EVOLUTION_MODEL = "deepseek-v4-flash"   # 空字符串 = 与 commander 相同
$CFG_DIFFICULTIES = "harder,veryhard,cheatvision,cheatmoney,cheatinsane"
$CFG_MATCHES = 10
$CFG_CANDIDATE_MATCHES = 10
$CFG_CONCURRENCY = 5
$CFG_MAX_GENERATIONS = 50
$CFG_MAX_GENERATIONS_PER_DIFFICULTY = 10
$CFG_MASTERY_SCORE_THRESHOLD = 0.90
$CFG_RUN_DIR = ""                       # 续跑时填 evolution_runs\... 路径
$CFG_BASELINE_BATCH_DIR = ""            # 新 run 可复用已完成基线批次
# =============================================================================

$ErrorActionPreference = "Stop"

if (-not [string]::IsNullOrWhiteSpace($STRATEGY)) { $CFG_STRATEGY = $STRATEGY }
if (-not [string]::IsNullOrWhiteSpace($COMMANDER_MODEL)) { $CFG_COMMANDER_MODEL = $COMMANDER_MODEL }
if (-not [string]::IsNullOrWhiteSpace($EVOLUTION_MODEL)) { $CFG_EVOLUTION_MODEL = $EVOLUTION_MODEL }
if (-not [string]::IsNullOrWhiteSpace($DIFFICULTIES)) { $CFG_DIFFICULTIES = $DIFFICULTIES }
if ($null -ne $MATCHES) { $CFG_MATCHES = [int]$MATCHES }
if ($null -ne $CANDIDATE_MATCHES) { $CFG_CANDIDATE_MATCHES = [int]$CANDIDATE_MATCHES }
if ($null -ne $CONCURRENCY) { $CFG_CONCURRENCY = [int]$CONCURRENCY }
if ($null -ne $MAX_GENERATIONS) { $CFG_MAX_GENERATIONS = [int]$MAX_GENERATIONS }
if ($null -ne $MAX_GENERATIONS_PER_DIFFICULTY) { $CFG_MAX_GENERATIONS_PER_DIFFICULTY = [int]$MAX_GENERATIONS_PER_DIFFICULTY }
if ($null -ne $MASTERY_SCORE_THRESHOLD) { $CFG_MASTERY_SCORE_THRESHOLD = [double]$MASTERY_SCORE_THRESHOLD }
if (-not [string]::IsNullOrWhiteSpace($RUN_DIR)) { $CFG_RUN_DIR = $RUN_DIR }
if (-not [string]::IsNullOrWhiteSpace($BASELINE_BATCH_DIR)) { $CFG_BASELINE_BATCH_DIR = $BASELINE_BATCH_DIR }

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
Write-Host "Candidate    : $CFG_CANDIDATE_MATCHES evaluation games"
Write-Host "Selection    : candidate score > champion score"
Write-Host "Tie          : inconclusive; Champion retained"
Write-Host "Mastery      : Champion score > $CFG_MASTERY_SCORE_THRESHOLD"
Write-Host "Posterior    : diagnostic only"
Write-Host "Budget       : per-diff gens=$CFG_MAX_GENERATIONS_PER_DIFFICULTY, max_gen=$CFG_MAX_GENERATIONS"
if (-not [string]::IsNullOrWhiteSpace($CFG_RUN_DIR)) {
    Write-Host "Resume dir   : $CFG_RUN_DIR"
}
if (-not [string]::IsNullOrWhiteSpace($CFG_BASELINE_BATCH_DIR)) {
    Write-Host "Baseline     : $CFG_BASELINE_BATCH_DIR"
}

$arguments = @(
    "-m", "evolution",
    "--strategy", $CFG_STRATEGY,
    "--commander-model", $CFG_COMMANDER_MODEL,
    "--difficulties", $CFG_DIFFICULTIES,
    "--matches", $CFG_MATCHES,
    "--candidate-matches", $CFG_CANDIDATE_MATCHES,
    "--concurrency", $CFG_CONCURRENCY,
    "--max-total-generations", $CFG_MAX_GENERATIONS,
    "--max-generations-per-difficulty", $CFG_MAX_GENERATIONS_PER_DIFFICULTY,
    "--mastery-score-threshold", $CFG_MASTERY_SCORE_THRESHOLD
)
if (-not [string]::IsNullOrWhiteSpace($CFG_EVOLUTION_MODEL)) {
    $arguments += @("--evolution-model", $CFG_EVOLUTION_MODEL)
}
if (-not [string]::IsNullOrWhiteSpace($CFG_RUN_DIR)) {
    $arguments += @("--run-dir", $CFG_RUN_DIR)
}
if (-not [string]::IsNullOrWhiteSpace($CFG_BASELINE_BATCH_DIR)) {
    $arguments += @("--baseline-batch-dir", $CFG_BASELINE_BATCH_DIR)
}

Set-Location $repoRoot
& $python @arguments
exit $LASTEXITCODE
