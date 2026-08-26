$ErrorActionPreference = "Stop"

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not $env:SC2PATH -or -not (Test-Path -LiteralPath $env:SC2PATH)) {
    $env:SC2PATH = "D:\StarCraft II"
}

# =============================================================================
# start_skill_bot_matrix.ps1
# Batch-run skill-aligned deterministic bots (skill_marine / skill_tank / skill_bc)
# Records go under game_records_bot/<marine_bot|tank_bot|battlecruiser_bot>/...
# =============================================================================

# =============================================================================
# 1. Shared game config
# =============================================================================
$MAP_NAME = "KairosJunctionLE"
$REAL_TIME = "0"
$BOT_RACE = "terran"
$BOT_INSTRUCT = ""
# Script bots do not read strategy.md.
$FORCE_STRATEGY = "none"
# Unused by skill bots; kept so match_id / batch naming stay valid.
$COMMANDER_MODEL = "na"

# =============================================================================
# 2. Records + run control
# =============================================================================
$OUTPUT_BASE_DIR = "game_records_bot"
$DEFAULT_MATCHES_PER_EXPERIMENT = 20
$CONCURRENCY = 5

# =============================================================================
# 3. Experiment list
# BotName: skill_marine | skill_tank | skill_bc  (see bot_loader/bot_definitions.py)
# =============================================================================
$EXPERIMENTS = @(
    # mediumhard
    @{ BotName = "skill_marine"; EnemyRace = "terran"; EnemyDifficulty = "mediumhard"; EnemyBuild = "macro"; Matches = 20 }
    @{ BotName = "skill_tank";   EnemyRace = "terran"; EnemyDifficulty = "mediumhard"; EnemyBuild = "macro"; Matches = 20 }
    @{ BotName = "skill_bc";     EnemyRace = "terran"; EnemyDifficulty = "mediumhard"; EnemyBuild = "macro"; Matches = 20 }
    # hard
    @{ BotName = "skill_marine"; EnemyRace = "terran"; EnemyDifficulty = "hard"; EnemyBuild = "macro"; Matches = 20 }
    @{ BotName = "skill_tank";   EnemyRace = "terran"; EnemyDifficulty = "hard"; EnemyBuild = "macro"; Matches = 20 }
    @{ BotName = "skill_bc";     EnemyRace = "terran"; EnemyDifficulty = "hard"; EnemyBuild = "macro"; Matches = 20 }
    # harder
    @{ BotName = "skill_marine"; EnemyRace = "terran"; EnemyDifficulty = "harder"; EnemyBuild = "macro"; Matches = 20 }
    @{ BotName = "skill_tank";   EnemyRace = "terran"; EnemyDifficulty = "harder"; EnemyBuild = "macro"; Matches = 20 }
    @{ BotName = "skill_bc";     EnemyRace = "terran"; EnemyDifficulty = "harder"; EnemyBuild = "macro"; Matches = 20 }
    # veryhard
    @{ BotName = "skill_marine"; EnemyRace = "terran"; EnemyDifficulty = "veryhard"; EnemyBuild = "macro"; Matches = 20 }
    @{ BotName = "skill_tank";   EnemyRace = "terran"; EnemyDifficulty = "veryhard"; EnemyBuild = "macro"; Matches = 20 }
    @{ BotName = "skill_bc";     EnemyRace = "terran"; EnemyDifficulty = "veryhard"; EnemyBuild = "macro"; Matches = 20 }
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
    if (-not $EXPERIMENTS -or $EXPERIMENTS.Count -eq 0) {
        Write-Error "EXPERIMENTS is empty. Add at least one experiment."
        exit 1
    }
}

$allowedBots = @{
    "skill_marine" = "marine_bot"
    "skill_tank"   = "tank_bot"
    "skill_bc"     = "battlecruiser_bot"
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

if ([System.IO.Path]::IsPathRooted($OUTPUT_BASE_DIR)) {
    $recordsRoot = $OUTPUT_BASE_DIR
} else {
    $recordsRoot = Join-Path $currentPath $OUTPUT_BASE_DIR
}
New-Item -ItemType Directory -Path $recordsRoot -Force | Out-Null
foreach ($style in ($allowedBots.Values | Select-Object -Unique)) {
    New-Item -ItemType Directory -Path (Join-Path $recordsRoot $style) -Force | Out-Null
}

Write-Host ""
Write-Host "Preparing to run $($EXPERIMENTS.Count) skill-bot experiment(s)."
Write-Host "Work dir : $currentPath"
Write-Host "Records  : $recordsRoot\<marine_bot|tank_bot|battlecruiser_bot>\..."
Write-Host "Strategy : $FORCE_STRATEGY (script bots ignore strategy.md)"
Write-Host "Batch    : $batchScript"

$totalFailures = 0
for ($experimentIndex = 0; $experimentIndex -lt $EXPERIMENTS.Count; $experimentIndex++) {
    $experiment = $EXPERIMENTS[$experimentIndex]
    $displayIndex = $experimentIndex + 1

    $botName = [string]$experiment.BotName
    $enemyRace = [string]$experiment.EnemyRace
    $enemyDifficulty = [string]$experiment.EnemyDifficulty
    $enemyBuild = [string]$experiment.EnemyBuild

    if ([string]::IsNullOrWhiteSpace($botName) -or
        [string]::IsNullOrWhiteSpace($enemyRace) -or
        [string]::IsNullOrWhiteSpace($enemyDifficulty) -or
        [string]::IsNullOrWhiteSpace($enemyBuild)) {
        Write-Error "Experiment #$displayIndex is missing BotName / EnemyRace / EnemyDifficulty / EnemyBuild."
        exit 1
    }

    if (-not $allowedBots.ContainsKey($botName)) {
        Write-Error ("Experiment #$displayIndex has unknown BotName='{0}'. Allowed: {1}" -f `
            $botName, (($allowedBots.Keys | Sort-Object) -join ", "))
        exit 1
    }

    $strategyStyle = [string]$allowedBots[$botName]
    $recordsAbs = Join-Path $recordsRoot $strategyStyle
    New-Item -ItemType Directory -Path $recordsAbs -Force | Out-Null

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
    $safeStyle = Get-SafeName $strategyStyle
    $batchName = "batch_${timestamp}_e${displayIndex}_${safeMap}_${safeBotRace}v${safeEnemyRace}_${safeDifficulty}_${safeBuild}_${safeStyle}"

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Experiment $displayIndex / $($EXPERIMENTS.Count)"
    Write-Host "=================================================="
    Write-Host "Bot      : $botName"
    Write-Host "Strategy : $strategyStyle"
    Write-Host "Enemy AI : $enemyRace | difficulty=$enemyDifficulty | build=$enemyBuild"
    Write-Host "Run      : $matches matches, concurrency=$CONCURRENCY"
    Write-Host "Batch    : $batchName"
    Write-Host "Records  : $recordsAbs"
    Write-Host "=================================================="

    $batchArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $batchScript,
        "-MY_BOT_NAME", $botName,
        "-MAP_NAME", $MAP_NAME,
        "-REAL_TIME", $REAL_TIME,
        "-ENEMY_RACE", $enemyRace,
        "-ENEMY_DIFFICULTY", $enemyDifficulty,
        "-ENEMY_BUILD", $enemyBuild,
        "-BOT_RACE", $BOT_RACE,
        "-FORCE_STRATEGY", $FORCE_STRATEGY,
        "-COMMANDER_MODEL", $COMMANDER_MODEL,
        "-TOTAL_MATCHES", $matches,
        "-CONCURRENCY", $CONCURRENCY,
        "-BATCH_NAME", $batchName,
        "-OUTPUT_BASE_DIR", $recordsAbs
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
    Write-Error "All experiments finished, but $totalFailures experiment(s) failed. Check logs under $recordsRoot."
    exit 1
}

Write-Host "All skill-bot experiments finished successfully."
Write-Host "Records: $recordsRoot\<marine_bot|tank_bot|battlecruiser_bot>\"

