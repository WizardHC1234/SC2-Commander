param(
    [string]$COMMANDER_MODEL = "kimi-k2.5",
    [string]$EVOLUTION_MODEL = "kimi-k2.5",
    [string]$DIFFICULTIES = "harder,veryhard,cheatvision,cheatmoney,cheatinsane",
    # Do not name this MATCHES: PowerShell's automatic $Matches is a Hashtable
    # written by -match/-notmatch and would clash with [int]$MATCHES.
    [int]$MATCH_COUNT = 10,
    [int]$CANDIDATE_MATCHES = 10,
    [int]$CONFIRMATION_MATCHES = 4,
    [int]$CONCURRENCY_PER_STRATEGY = 2,
    [int]$MAX_GENERATIONS = 10,
    [double]$MASTERY_SCORE_THRESHOLD = 0.90,
    [int]$ANALYSIS_BATCH_GAMES = 10,
    [int]$MAX_ANALYSIS_GAMES_PER_GENERATION = 10,
    [ValidateSet("multi_match", "single_failure")]
    [string]$ANALYSIS_EXPERIENCE_MODE = "multi_match",
    [int]$ANALYSIS_SAMPLE_SEED = 0,
    [ValidateSet("enabled", "disabled")]
    [string]$KNOWLEDGE_MODE = "enabled",
    [string]$RECORDS_GROUP = "",
    [string]$MARINE_BASELINE_BATCH_DIR = "",
    [string]$TANK_BASELINE_BATCH_DIR = "",
    [string]$BATTLECRUISER_BASELINE_BATCH_DIR = "",
    [string]$RUN_STAMP = "",
    [switch]$WAIT
)

# Start three new runs:
#   .\scripts\start_three_strategy_evolution.ps1
# Resume all three after an interruption:
#   .\scripts\start_three_strategy_evolution.ps1 -RUN_STAMP 20260826_140000

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python virtual environment not found: $python"
}

$strategies = @("marine", "tank", "battlecruiser")
$baselineByStrategy = @{
    marine = $MARINE_BASELINE_BATCH_DIR
    tank = $TANK_BASELINE_BATCH_DIR
    battlecruiser = $BATTLECRUISER_BASELINE_BATCH_DIR
}
foreach ($strategy in $strategies) {
    $strategyFile = Join-Path $repoRoot "skills\terran\$strategy\strategy.md"
    if (-not (Test-Path -LiteralPath $strategyFile -PathType Leaf)) {
        throw "Strategy file not found: $strategyFile"
    }
    $baselineDir = [string]$baselineByStrategy[$strategy]
    if (-not [string]::IsNullOrWhiteSpace($baselineDir) -and -not (Test-Path -LiteralPath $baselineDir -PathType Container)) {
        throw "Baseline batch directory not found for ${strategy}: $baselineDir"
    }
}

$stamp = if ([string]::IsNullOrWhiteSpace($RUN_STAMP)) {
    Get-Date -Format "yyyyMMdd_HHmmss"
} else {
    $RUN_STAMP.Trim()
}
if (-not [regex]::IsMatch($stamp, '^[A-Za-z0-9_-]+$')) {
    throw "RUN_STAMP may contain only letters, numbers, underscores, and hyphens."
}
$launchStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$launcherDir = Join-Path $repoRoot "evolution_runs\three_strategy_launcher\$stamp\$launchStamp"
New-Item -ItemType Directory -Path $launcherDir -Force | Out-Null

$processes = @()
foreach ($strategy in $strategies) {
    $runDir = Join-Path $repoRoot "evolution_runs\$strategy\$stamp"
    $recordsRoot = Join-Path $repoRoot "game_records_evol"
    if (-not [string]::IsNullOrWhiteSpace($RECORDS_GROUP)) {
        $recordsRoot = Join-Path $recordsRoot $RECORDS_GROUP.Trim()
    }
    $recordsDir = Join-Path $recordsRoot $strategy
    New-Item -ItemType Directory -Path $recordsDir -Force | Out-Null

    $stdout = Join-Path $launcherDir "$strategy.stdout.log"
    $stderr = Join-Path $launcherDir "$strategy.stderr.log"
    $arguments = @(
        "-m", "evolution",
        "--strategy", $strategy,
        "--commander-model", $COMMANDER_MODEL,
        "--evolution-model", $EVOLUTION_MODEL,
        "--difficulties", $DIFFICULTIES,
        "--matches", $MATCH_COUNT,
        "--candidate-matches", $CANDIDATE_MATCHES,
        "--confirmation-matches", $CONFIRMATION_MATCHES,
        "--concurrency", $CONCURRENCY_PER_STRATEGY,
        "--max-total-generations", $MAX_GENERATIONS,
        "--max-generations-per-difficulty", $MAX_GENERATIONS,
        "--require-full-generation-budget",
        "--mastery-score-threshold", $MASTERY_SCORE_THRESHOLD,
        "--analysis-batch-games", $ANALYSIS_BATCH_GAMES,
        "--max-analysis-games-per-generation", $MAX_ANALYSIS_GAMES_PER_GENERATION,
        "--analysis-experience-mode", $ANALYSIS_EXPERIENCE_MODE,
        "--analysis-sample-seed", $ANALYSIS_SAMPLE_SEED,
        "--knowledge-mode", $KNOWLEDGE_MODE,
        "--run-dir", $runDir,
        "--records-dir", $recordsDir
    )
    $baselineDir = [string]$baselineByStrategy[$strategy]
    if (-not [string]::IsNullOrWhiteSpace($baselineDir)) {
        $arguments += @("--baseline-batch-dir", $baselineDir)
    }

    $process = Start-Process `
        -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru

    $processes += [pscustomobject]@{
        strategy = $strategy
        process_id = $process.Id
        run_dir = $runDir
        records_dir = $recordsDir
        stdout = $stdout
        stderr = $stderr
    }
    Write-Host "Started $strategy evolution (PID=$($process.Id), concurrency=$CONCURRENCY_PER_STRATEGY)"
}

$manifestPath = Join-Path $launcherDir "launcher.json"
$manifest = [ordered]@{
    created_at = (Get-Date).ToString("o")
    commander_model = $COMMANDER_MODEL
    evolution_model = $EVOLUTION_MODEL
    max_generations = $MAX_GENERATIONS
    require_full_generation_budget = $true
    concurrency_per_strategy = $CONCURRENCY_PER_STRATEGY
    analysis_experience_mode = $ANALYSIS_EXPERIENCE_MODE
    analysis_sample_seed = $ANALYSIS_SAMPLE_SEED
    knowledge_mode = $KNOWLEDGE_MODE
    records_group = $RECORDS_GROUP
    baseline_batches = $baselineByStrategy
    processes = $processes
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8

Write-Host "Run stamp: $stamp"
Write-Host "Launcher manifest: $manifestPath"
Write-Host "Analysis mode: $ANALYSIS_EXPERIENCE_MODE; knowledge mode: $KNOWLEDGE_MODE"
Write-Host "Match records: $recordsRoot\<strategy>"

if ($WAIT) {
    Write-Host "Waiting for all three evolution processes..."
    foreach ($item in $processes) {
        Wait-Process -Id $item.process_id
        $finished = Get-Process -Id $item.process_id -ErrorAction SilentlyContinue
        if ($null -eq $finished) {
            Write-Host "$($item.strategy) process finished. See $($item.stdout) and $($item.stderr)."
        }
    }
}
