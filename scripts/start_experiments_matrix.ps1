$ErrorActionPreference = "Stop"

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not $env:SC2PATH -or -not (Test-Path -LiteralPath $env:SC2PATH)) {
    $env:SC2PATH = "D:\StarCraft II"
}

# =============================================================================
# start_experiments_matrix.ps1
# 按实验列表批量跑对局；每组实验委托 scripts/start_batch.ps1。
# =============================================================================

# =============================================================================
# 1. Shared game config
# =============================================================================
$MY_BOT_NAME = "commander"
$MAP_NAME = "KairosJunctionLE"
$REAL_TIME = "0"
$BOT_RACE = "terran"
$BOT_INSTRUCT = ""

# Default strategy when an experiment row omits Strategy.
$FORCE_STRATEGY = "mid_tank"

# =============================================================================
# 2. Commander model (must exist in llm/config.json -> llm_agents_pool)
# =============================================================================
$COMMANDER_MODEL = "qwen3-32b"
# $COMMANDER_MODEL = "kimi-k2.5"
# $COMMANDER_MODEL = "deepseek-v4-flash"

# =============================================================================
# 3. Run control
# =============================================================================
$DEFAULT_MATCHES_PER_EXPERIMENT = 10
$CONCURRENCY = 2

# =============================================================================
# 4. Experiment list
# =============================================================================
# EnemyRace       : terran / zerg / protoss / random
# EnemyDifficulty : veryeasy / easy / medium / mediumhard / hard / harder /
#                   veryhard / cheatvision / cheatmoney / cheatinsane
# EnemyBuild      : random / rush / timing / power / macro / air
# Strategy        : skills/<BOT_RACE>/<Strategy>；省略则用 $FORCE_STRATEGY
# Matches         : 省略则用 $DEFAULT_MATCHES_PER_EXPERIMENT
$EXPERIMENTS = @(
    @{ EnemyRace = "terran"; EnemyDifficulty = "hard"; EnemyBuild = "macro"; Strategy = "early_marine"; Matches = 10 }
    @{ EnemyRace = "terran"; EnemyDifficulty = "hard"; EnemyBuild = "macro"; Strategy = "mid_tank"; Matches = 10 }
    @{ EnemyRace = "terran"; EnemyDifficulty = "hard"; EnemyBuild = "macro"; Strategy = "late_battlecruiser"; Matches = 10 }

    # @{ EnemyRace = "terran"; EnemyDifficulty = "veryhard"; EnemyBuild = "macro"; Strategy = "mid_tank"; Matches = 10 }
    # @{ EnemyRace = "zerg";   EnemyDifficulty = "hard";     EnemyBuild = "macro"; Strategy = "mid_tank"; Matches = 10 }
)

function Get-SafeName {
    param([Parameter(Mandatory = $true)][string]$Value)
    return ($Value -replace '[^a-zA-Z0-9_-]', '_')
}

function Get-RepoRoot {
    $scriptDir = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
        (Get-Location).Path
    } else {
        $PSScriptRoot
    }
    return (Resolve-Path (Join-Path $scriptDir "..")).Path
}

function Test-Config {
    if ($CONCURRENCY -le 0) {
        Write-Error "CONCURRENCY must be greater than 0."
        exit 1
    }

    if ($DEFAULT_MATCHES_PER_EXPERIMENT -lt 0) {
        Write-Error "DEFAULT_MATCHES_PER_EXPERIMENT cannot be negative."
        exit 1
    }

    if ([string]::IsNullOrWhiteSpace($COMMANDER_MODEL)) {
        Write-Error "COMMANDER_MODEL cannot be empty."
        exit 1
    }

    if (-not $EXPERIMENTS -or $EXPERIMENTS.Count -eq 0) {
        Write-Error "EXPERIMENTS is empty. Add at least one experiment."
        exit 1
    }
}

$currentPath = Get-RepoRoot
Set-Location -Path $currentPath

$batchScript = Join-Path $PSScriptRoot "start_batch.ps1"
if (-not (Test-Path -LiteralPath $batchScript -PathType Leaf)) {
    Write-Error "Batch script not found: $batchScript"
    exit 1
}

$powerShellCommand = Get-Command "powershell.exe" -ErrorAction SilentlyContinue
if (-not $powerShellCommand) {
    Write-Error "powershell.exe was not found."
    exit 1
}
$powerShellExe = $powerShellCommand.Source

Test-Config

Write-Host ""
Write-Host "Preparing to run $($EXPERIMENTS.Count) experiment(s)."
Write-Host "Work dir: $currentPath"
Write-Host "Batch launcher: $batchScript"
Write-Host "Commander model: $COMMANDER_MODEL"

$totalFailures = 0
for ($experimentIndex = 0; $experimentIndex -lt $EXPERIMENTS.Count; $experimentIndex++) {
    $experiment = $EXPERIMENTS[$experimentIndex]
    $displayIndex = $experimentIndex + 1

    $enemyRace = [string]$experiment.EnemyRace
    $enemyDifficulty = [string]$experiment.EnemyDifficulty
    $enemyBuild = [string]$experiment.EnemyBuild
    $strategy = $FORCE_STRATEGY
    if ($experiment.ContainsKey("Strategy") -and $null -ne $experiment.Strategy) {
        $strategy = [string]$experiment.Strategy
    }

    if ([string]::IsNullOrWhiteSpace($enemyRace) -or
        [string]::IsNullOrWhiteSpace($enemyDifficulty) -or
        [string]::IsNullOrWhiteSpace($enemyBuild)) {
        Write-Error "Experiment #$displayIndex is missing EnemyRace, EnemyDifficulty, or EnemyBuild."
        exit 1
    }

    if ([string]::IsNullOrWhiteSpace($strategy)) {
        Write-Error "Experiment #$displayIndex has an empty Strategy."
        exit 1
    }

    if ($strategy -ne "none") {
        $strategyPath = Join-Path $currentPath "skills\$BOT_RACE\$strategy"
        if (-not (Test-Path -LiteralPath $strategyPath -PathType Container)) {
            Write-Error "Strategy folder not found for experiment #${displayIndex}: $strategyPath"
            exit 1
        }
    }

    $matches = $DEFAULT_MATCHES_PER_EXPERIMENT
    if ($experiment.ContainsKey("Matches") -and $null -ne $experiment.Matches) {
        $matches = [int]$experiment.Matches
    }

    if ($matches -le 0) {
        Write-Host "Skipping experiment #$displayIndex because Matches=$matches."
        continue
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $safeMap = Get-SafeName $MAP_NAME
    $safeBotRace = Get-SafeName $BOT_RACE
    $safeEnemyRace = Get-SafeName $enemyRace
    $safeDifficulty = Get-SafeName $enemyDifficulty
    $safeBuild = Get-SafeName $enemyBuild
    $safeStrategy = Get-SafeName $strategy
    $safeModel = Get-SafeName $COMMANDER_MODEL
    $batchName = "batch_${timestamp}_e${displayIndex}_${safeMap}_${safeBotRace}v${safeEnemyRace}_${safeDifficulty}_${safeBuild}_${safeStrategy}_${safeModel}"

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Experiment $displayIndex / $($EXPERIMENTS.Count)"
    Write-Host "=================================================="
    Write-Host "Enemy AI : $enemyRace | difficulty=$enemyDifficulty | build=$enemyBuild"
    Write-Host "Strategy : $strategy"
    Write-Host "Model    : $COMMANDER_MODEL"
    Write-Host "Run      : $matches matches, concurrency=$CONCURRENCY"
    Write-Host "Batch    : $batchName"
    Write-Host "=================================================="

    $batchArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $batchScript,
        "-MY_BOT_NAME", $MY_BOT_NAME,
        "-MAP_NAME", $MAP_NAME,
        "-REAL_TIME", $REAL_TIME,
        "-ENEMY_RACE", $enemyRace,
        "-ENEMY_DIFFICULTY", $enemyDifficulty,
        "-ENEMY_BUILD", $enemyBuild,
        "-BOT_RACE", $BOT_RACE,
        "-FORCE_STRATEGY", $strategy,
        "-BOT_INSTRUCT", $BOT_INSTRUCT,
        "-COMMANDER_MODEL", $COMMANDER_MODEL,
        "-TOTAL_MATCHES", $matches,
        "-CONCURRENCY", $CONCURRENCY,
        "-BATCH_NAME", $batchName
    )

    & $powerShellExe @batchArgs
    $batchExitCode = $LASTEXITCODE

    if ($batchExitCode -ne 0) {
        $totalFailures++
        Write-Host "Experiment $displayIndex failed with exit code $batchExitCode."
    }
    else {
        Write-Host "Experiment $displayIndex finished successfully."
    }
}

Write-Host ""
if ($totalFailures -gt 0) {
    Write-Error "All experiments finished, but $totalFailures experiment(s) failed. Check the log files above."
    exit 1
}

Write-Host "All experiments finished successfully."
