$ErrorActionPreference = "Stop"

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not $env:SC2PATH -or -not (Test-Path -LiteralPath $env:SC2PATH)) {
    $env:SC2PATH = "D:\StarCraft II"
}

# =============================================================================
# start_experiments_matrix.ps1
# 按实验列表批量跑对局；每组实验委�?scripts/start_batch.ps1�?# =============================================================================

# =============================================================================
# 1. Shared game config
# =============================================================================
$MY_BOT_NAME = "commander"
$MAP_NAME = "KairosJunctionLE"
$REAL_TIME = "0"
$BOT_RACE = "terran"
$BOT_INSTRUCT = ""

# Default strategy when an experiment row omits Strategy.
$FORCE_STRATEGY = "tank"

# =============================================================================
# 2. Commander model (must exist in llm/config.json -> llm_agents_pool)
# =============================================================================
# $COMMANDER_MODEL = "qwen3-32b"
# $COMMANDER_MODEL = "qwen3.5-27b"
$COMMANDER_MODEL = "kimi-k2.5"
# $COMMANDER_MODEL = "deepseek-v4-flash"
# $COMMANDER_MODEL = "qwen3-32b-reasoning"

# =============================================================================
# 3. Run control
# =============================================================================
$DEFAULT_MATCHES_PER_EXPERIMENT = 10
$CONCURRENCY = 5

# =============================================================================
# 4. Experiment list
# =============================================================================
# EnemyRace       : terran / zerg / protoss / random
# EnemyDifficulty : veryeasy / easy / medium / mediumhard / hard / harder /
#                   veryhard / cheatvision / cheatmoney / cheatinsane
# EnemyBuild      : random / rush / timing / power / macro / air
# Strategy        : skills/<BOT_RACE>/<Strategy>
# Matches         : omit to use $DEFAULT_MATCHES_PER_EXPERIMENT
# Model           : 省略则用 $COMMANDER_MODEL（须存在于 llm/config.json）
$EXPERIMENTS = @(

    # @{ EnemyRace = "terran";  EnemyDifficulty = "harder"; EnemyBuild = "macro";  Strategy = "tank_opt1"; Matches = 10 }


    # @{ EnemyRace = "terran";  EnemyDifficulty = "veryeasy"; EnemyBuild = "macro";  Strategy = "marine"; Matches = 20 }
    # @{ EnemyRace = "terran";  EnemyDifficulty = "veryeasy"; EnemyBuild = "macro";  Strategy = "tank"; Matches = 20 }
    # @{ EnemyRace = "terran";  EnemyDifficulty = "veryeasy"; EnemyBuild = "macro";  Strategy = "battlecruiser"; Matches = 20 }
    # @{ EnemyRace = "terran";  EnemyDifficulty = "easy"; EnemyBuild = "macro";  Strategy = "marine"; Matches = 20 }
    # @{ EnemyRace = "terran";  EnemyDifficulty = "easy"; EnemyBuild = "macro";  Strategy = "tank"; Matches = 20 }
    # @{ EnemyRace = "terran";  EnemyDifficulty = "easy"; EnemyBuild = "macro";  Strategy = "battlecruiser"; Matches = 20 }
    # @{ EnemyRace = "terran";  EnemyDifficulty = "medium"; EnemyBuild = "macro";  Strategy = "marine"; Matches = 20 }
    # @{ EnemyRace = "terran";  EnemyDifficulty = "medium"; EnemyBuild = "macro";  Strategy = "tank"; Matches = 20 }
    # @{ EnemyRace = "terran";  EnemyDifficulty = "medium"; EnemyBuild = "macro";  Strategy = "battlecruiser"; Matches = 20 }

    @{ EnemyRace="terran"; EnemyDifficulty="mediumhard"; EnemyBuild="macro"; Strategy="none"; Matches=20;}
    @{ EnemyRace="terran"; EnemyDifficulty="hard"; EnemyBuild="macro"; Strategy="none"; Matches=20;}
    @{ EnemyRace="terran"; EnemyDifficulty="harder"; EnemyBuild="macro"; Strategy="none"; Matches=20;}
    @{ EnemyRace="terran"; EnemyDifficulty="veryhard"; EnemyBuild="macro"; Strategy="none"; Matches=20;}
    


    @{ EnemyRace = "terran";  EnemyDifficulty = "mediumhard"; EnemyBuild = "macro";  Strategy = "marine"; Matches = 20  }
    @{ EnemyRace = "terran";  EnemyDifficulty = "mediumhard"; EnemyBuild = "macro";  Strategy = "tank"; Matches = 20}
    @{ EnemyRace = "terran";  EnemyDifficulty = "mediumhard"; EnemyBuild = "macro";  Strategy = "battlecruiser"; Matches = 20 }
    @{ EnemyRace = "terran";  EnemyDifficulty = "hard"; EnemyBuild = "macro"; Strategy = "marine"; Matches = 20 }
    @{ EnemyRace = "terran";  EnemyDifficulty = "hard"; EnemyBuild = "macro"; Strategy = "tank"; Matches = 20  }
    @{ EnemyRace = "terran";  EnemyDifficulty = "hard"; EnemyBuild = "macro"; Strategy = "battlecruiser"; Matches = 20 }
    @{ EnemyRace = "terran";  EnemyDifficulty = "harder"; EnemyBuild = "macro"; Strategy = "marine"; Matches = 20 }
    @{ EnemyRace = "terran";  EnemyDifficulty = "harder"; EnemyBuild = "macro";  Strategy = "tank"; Matches = 20  }
    @{ EnemyRace = "terran";  EnemyDifficulty = "harder"; EnemyBuild = "macro";    Strategy = "battlecruiser"; Matches = 20 }
    @{ EnemyRace = "terran";  EnemyDifficulty = "veryhard"; EnemyBuild = "macro";  Strategy = "marine"; Matches = 20  }
    @{ EnemyRace = "terran";  EnemyDifficulty = "veryhard"; EnemyBuild = "macro";  Strategy = "tank"; Matches = 20  }
    @{ EnemyRace = "terran";  EnemyDifficulty = "veryhard"; EnemyBuild = "macro";    Strategy = "battlecruiser"; Matches = 20  }

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
Write-Host "Commander model (default): $COMMANDER_MODEL"

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

    $model = $COMMANDER_MODEL
    if ($experiment.ContainsKey("Model") -and -not [string]::IsNullOrWhiteSpace([string]$experiment.Model)) {
        $model = [string]$experiment.Model
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

    if ([string]::IsNullOrWhiteSpace($model)) {
        Write-Error "Experiment #$displayIndex has an empty Model (and COMMANDER_MODEL is blank)."
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
    $safeModel = Get-SafeName $model
    $batchName = "batch_${timestamp}_e${displayIndex}_${safeMap}_${safeBotRace}v${safeEnemyRace}_${safeDifficulty}_${safeBuild}_${safeStrategy}_${safeModel}"

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Experiment $displayIndex / $($EXPERIMENTS.Count)"
    Write-Host "=================================================="
    Write-Host "Enemy AI : $enemyRace | difficulty=$enemyDifficulty | build=$enemyBuild"
    Write-Host "Strategy : $strategy"
    Write-Host "Model    : $model"
    Write-Host "Run      : $matches matches, concurrency=$CONCURRENCY"
    Write-Host "Batch    : $batchName"
    Write-Host "=================================================="

    # Empty string args are dropped by powershell.exe -File, so omit optional
    # -BOT_INSTRUCT when blank (start_batch.ps1 already defaults it to "").
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
        "-COMMANDER_MODEL", $model,
        "-TOTAL_MATCHES", $matches,
        "-CONCURRENCY", $CONCURRENCY,
        "-BATCH_NAME", $batchName
    )
    if (-not [string]::IsNullOrWhiteSpace($BOT_INSTRUCT)) {
        $batchArgs += @("-BOT_INSTRUCT", $BOT_INSTRUCT)
    }

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
