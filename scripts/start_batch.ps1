param(
    [string]$MY_BOT_NAME = "commander",
    [string]$MAP_NAME = "KairosJunctionLE",
    [string]$REAL_TIME = "0",

    [string]$ENEMY_RACE = "terran",
    [string]$ENEMY_DIFFICULTY = "hard",
    [string]$ENEMY_BUILD = "macro",

    [string]$BOT_RACE = "terran",
    [string]$FORCE_STRATEGY = "tank",
    [string]$BOT_INSTRUCT = "",

    [string]$COMMANDER_MODEL = "qwen3-32b",

    [int]$TOTAL_MATCHES = 20,
    [int]$CONCURRENCY = 2,
    [int]$START_INDEX = 0,

    [string]$BATCH_NAME = ""
)

$ErrorActionPreference = "Stop"

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not $env:SC2PATH -or -not (Test-Path -LiteralPath $env:SC2PATH)) {
    $env:SC2PATH = "D:\StarCraft II"
}

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

function Test-BatchConfig {
    param([Parameter(Mandatory = $true)][string]$WorkDir)

    if ($CONCURRENCY -le 0) {
        Write-Error "CONCURRENCY must be greater than 0."
        exit 1
    }

    if ($START_INDEX -lt 0) {
        Write-Error "START_INDEX cannot be negative."
        exit 1
    }

    if ($TOTAL_MATCHES -le 0) {
        Write-Host "TOTAL_MATCHES=$TOTAL_MATCHES, nothing to run."
        exit 0
    }

    if ([string]::IsNullOrWhiteSpace($FORCE_STRATEGY)) {
        Write-Error "FORCE_STRATEGY cannot be empty."
        exit 1
    }

    if ($FORCE_STRATEGY -ne "none") {
        # Evolution sets SC2_STRATEGY_ROOT to evolution_runs/.../strategies so
        # candidates are not written under skills/<race>/. Prefer that overlay.
        $candidates = @()
        $overlay = [string]$env:SC2_STRATEGY_ROOT
        if (-not [string]::IsNullOrWhiteSpace($overlay)) {
            $candidates += (Join-Path $overlay $FORCE_STRATEGY)
        }
        $candidates += (Join-Path $WorkDir "skills\$BOT_RACE\$FORCE_STRATEGY")

        $resolved = $null
        foreach ($candidate in $candidates) {
            $md = Join-Path $candidate "strategy.md"
            if (Test-Path -LiteralPath $md -PathType Leaf) {
                $resolved = $candidate
                break
            }
        }
        if (-not $resolved) {
            Write-Error ("Strategy folder not found for '{0}'. Searched: {1}" -f `
                $FORCE_STRATEGY, ($candidates -join "; "))
            exit 1
        }
    }

    if (-not (Test-Path -LiteralPath (Join-Path $WorkDir "llm\config.json") -PathType Leaf)) {
        Write-Error "Missing llm\config.json. Copy llm\config.example.json and fill in keys."
        exit 1
    }
}

function Get-PythonExe {
    param([Parameter(Mandatory = $true)][string]$WorkDir)

    $candidates = @()

    if (-not [string]::IsNullOrWhiteSpace($env:VIRTUAL_ENV)) {
        $activeVenvPython = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
        if (Test-Path -LiteralPath $activeVenvPython -PathType Leaf) {
            $candidates += $activeVenvPython
        }
    }

    $venvPython = Join-Path $WorkDir "venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $candidates += $venvPython
    }

    $pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $candidates += $pythonCmd.Source
    }

    $candidates = @($candidates | Select-Object -Unique)
    if ($candidates.Count -eq 0) {
        Write-Error "python was not found. Install Python or add it to PATH."
        exit 1
    }

    foreach ($candidate in $candidates) {
        try {
            & $candidate -c "import openai" *> $null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Python: $candidate"
                return $candidate
            }
        }
        catch {
            # Try next candidate.
        }
    }

    Write-Error "No Python candidate can import openai. Tried: $($candidates -join ', ')"
    exit 1
}

function New-DefaultBatchName {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $safeMap = Get-SafeName $MAP_NAME
    $safeBotRace = Get-SafeName $BOT_RACE
    $safeEnemyRace = Get-SafeName $ENEMY_RACE
    $safeDifficulty = Get-SafeName $ENEMY_DIFFICULTY
    $safeBuild = Get-SafeName $ENEMY_BUILD
    $safeStrategy = Get-SafeName $FORCE_STRATEGY
    $safeModel = Get-SafeName $COMMANDER_MODEL
    return "batch_${timestamp}_${safeMap}_${safeBotRace}v${safeEnemyRace}_${safeDifficulty}_${safeBuild}_${safeStrategy}_${safeModel}"
}

function New-RunVsAiArgs {
    param(
        [Parameter(Mandatory = $true)][int]$RunIndex,
        [Parameter(Mandatory = $true)][string]$RecordRoot,
        [Parameter(Mandatory = $true)][string]$BatchName,
        [string]$RecordDirFile = ""
    )

    $pythonArgs = @(
        "run_vs_ai.py",
        "--my-bot-name", $MY_BOT_NAME,
        "--map-name", $MAP_NAME,
        "--enemy-race", $ENEMY_RACE,
        "--enemy-difficulty", $ENEMY_DIFFICULTY,
        "--enemy-build", $ENEMY_BUILD,
        "--bot-race", $BOT_RACE,
        "--commander-model", $COMMANDER_MODEL,
        "--force-strategy", $FORCE_STRATEGY,
        "--batch-name", $BatchName,
        "--run-index", $RunIndex,
        "--output-base-dir", $RecordRoot,
        "--skip-version-update"
    )

    if (-not [string]::IsNullOrWhiteSpace($BOT_INSTRUCT)) {
        $pythonArgs += @("--bot-instruct", $BOT_INSTRUCT)
    }

    if (-not [string]::IsNullOrWhiteSpace($RecordDirFile)) {
        $pythonArgs += @("--record-dir-file", $RecordDirFile)
    }

    if ($REAL_TIME -eq "1" -or $REAL_TIME -eq "true" -or $REAL_TIME -eq "True") {
        $pythonArgs += "--real-time"
    }

    return $pythonArgs
}

function Move-MatchConsoleLog {
    param(
        [Parameter(Mandatory = $true)][string]$OutFile,
        [Parameter(Mandatory = $true)][string]$RecordDirFile
    )

    if (-not (Test-Path -LiteralPath $RecordDirFile -PathType Leaf)) {
        return $OutFile
    }

    try {
        $matchDir = (Get-Content -LiteralPath $RecordDirFile -Raw).Trim()
        Remove-Item -LiteralPath $RecordDirFile -Force -ErrorAction SilentlyContinue

        if ([string]::IsNullOrWhiteSpace($matchDir) -or
            -not (Test-Path -LiteralPath $matchDir -PathType Container)) {
            return $OutFile
        }

        $matchId = Split-Path -Leaf $matchDir
        $canonicalLog = Join-Path $matchDir "$matchId.log"

        if (Test-Path -LiteralPath $canonicalLog -PathType Leaf) {
            if ((Test-Path -LiteralPath $OutFile -PathType Leaf) -and
                ($OutFile -ne $canonicalLog)) {
                Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
            }
            return $canonicalLog
        }

        if (Test-Path -LiteralPath $OutFile -PathType Leaf) {
            Move-Item -LiteralPath $OutFile -Destination $canonicalLog -Force
            return $canonicalLog
        }
        return $OutFile
    }
    catch {
        "Could not archive match log: $_" | Out-File -LiteralPath $OutFile -Append -Encoding utf8
        return $OutFile
    }
}

$currentPath = Get-RepoRoot
Set-Location -Path $currentPath

Test-BatchConfig -WorkDir $currentPath
$PYTHON_EXE = Get-PythonExe -WorkDir $currentPath
$recordRoot = Join-Path $currentPath "game_records"
$sc2Path = $env:SC2PATH

Write-Host ""
Write-Host "=================================================="
Write-Host "SC2-Commander experiment batch"
Write-Host "=================================================="
Write-Host "Bot      : $MY_BOT_NAME ($BOT_RACE)"
Write-Host "Enemy    : $ENEMY_RACE | difficulty=$ENEMY_DIFFICULTY | build=$ENEMY_BUILD"
Write-Host "Map      : $MAP_NAME"
Write-Host "Strategy : $FORCE_STRATEGY"
Write-Host "Model    : $COMMANDER_MODEL"
Write-Host "Run      : $TOTAL_MATCHES matches, concurrency=$CONCURRENCY"
Write-Host "=================================================="
Write-Host ""

if ($TOTAL_MATCHES -le 1) {
    if ([string]::IsNullOrWhiteSpace($BATCH_NAME)) {
        $BATCH_NAME = New-DefaultBatchName
    }

    Write-Host "Starting one match."
    Write-Host "Batch: $BATCH_NAME"
    $safeBatchTempName = Get-SafeName $BATCH_NAME
    $singleLogDir = Join-Path ([System.IO.Path]::GetTempPath()) "sc2-commander\$safeBatchTempName-$PID"
    New-Item -ItemType Directory -Path $singleLogDir -Force | Out-Null
    $outFile = Join-Path $singleLogDir "fg_run_0.log"
    $recordDirFile = Join-Path $singleLogDir ".record_dir_0.txt"
    $pythonArgs = New-RunVsAiArgs -RunIndex $START_INDEX -RecordRoot $recordRoot -BatchName $BATCH_NAME -RecordDirFile $recordDirFile
    & $PYTHON_EXE @pythonArgs *>> $outFile
    $exitCode = $LASTEXITCODE
    $outFile = Move-MatchConsoleLog -OutFile $outFile -RecordDirFile $recordDirFile

    Write-Host "Log: $outFile"
    if ($exitCode -ne 0) {
        Write-Error "Single match failed with exit code $exitCode."
        exit $exitCode
    }
    if ((Get-ChildItem -LiteralPath $singleLogDir -Force | Measure-Object).Count -eq 0) {
        Remove-Item -LiteralPath $singleLogDir
    }
    return
}

if ([string]::IsNullOrWhiteSpace($BATCH_NAME)) {
    $BATCH_NAME = New-DefaultBatchName
}

$safeBatchTempName = Get-SafeName $BATCH_NAME
$logDir = Join-Path ([System.IO.Path]::GetTempPath()) "sc2-commander\$safeBatchTempName-$PID"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

Write-Host "Starting batch: $BATCH_NAME"
Write-Host "Temporary logs (failed startup only): $logDir"
Write-Host ""

$jobs = @{}
$failedMatches = 0

for ($i = 0; $i -lt $TOTAL_MATCHES; $i++) {
    while (($jobs.Values | Where-Object { $_.State -eq 'Running' }).Count -ge $CONCURRENCY) {
        $completedJob = Wait-Job -Any -Job @($jobs.Values) 2>$null
        if ($completedJob) {
            $result = Receive-Job $completedJob
            Remove-Job $completedJob
            $jobs.Remove($completedJob.Id)

            foreach ($item in @($result)) {
                if ($item -and $null -ne $item.ExitCode) {
                    Write-Host "[Job $($item.Index)] completed, exit=$($item.ExitCode), log=$($item.LogFile)"
                    if ([int]$item.ExitCode -ne 0) {
                        $failedMatches++
                    }
                }
                elseif ($item) {
                    Write-Host $item
                }
            }
        }
    }

    $matchIndex = $START_INDEX + $i
    $job = Start-Job -ScriptBlock {
        param(
            $idx,
            $myBot,
            $map,
            $realTime,
            $eRace,
            $eDiff,
            $eBuild,
            $instruct,
            $bRace,
            $commanderM,
            $fStrat,
            $rRoot,
            $bName,
            $lDir,
            $workDir,
            $pyExe,
            $sc2
        )

        $ErrorActionPreference = "Continue"
        $env:PYTHONIOENCODING = "utf-8"
        $env:SC2PATH = $sc2
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

        Set-Location -Path $workDir

        $outFile = Join-Path $lDir "fg_run_${idx}.log"
        $recordDirFile = Join-Path $lDir ".record_dir_${idx}.txt"
        try {
            $pythonArgs = @(
                "run_vs_ai.py",
                "--my-bot-name", $myBot,
                "--map-name", $map,
                "--enemy-race", $eRace,
                "--enemy-difficulty", $eDiff,
                "--enemy-build", $eBuild,
                "--bot-race", $bRace,
                "--commander-model", $commanderM,
                "--force-strategy", $fStrat,
                "--batch-name", $bName,
                "--run-index", $idx,
                "--output-base-dir", $rRoot,
                "--record-dir-file", $recordDirFile,
                "--skip-version-update"
            )

            if (-not [string]::IsNullOrWhiteSpace($instruct)) {
                $pythonArgs += @("--bot-instruct", $instruct)
            }

            if ($realTime -eq "1" -or $realTime -eq "true" -or $realTime -eq "True") {
                $pythonArgs += "--real-time"
            }

            & $pyExe @pythonArgs *>> $outFile
            $exitCode = $LASTEXITCODE
        }
        catch {
            "Job $idx exception: $_" | Out-File -LiteralPath $outFile -Append -Encoding utf8
            $exitCode = 1
        }

        if (Test-Path -LiteralPath $recordDirFile -PathType Leaf) {
            try {
                $matchDir = (Get-Content -LiteralPath $recordDirFile -Raw).Trim()
                Remove-Item -LiteralPath $recordDirFile -Force -ErrorAction SilentlyContinue
                if (-not [string]::IsNullOrWhiteSpace($matchDir) -and
                    (Test-Path -LiteralPath $matchDir -PathType Container)) {
                    $matchId = Split-Path -Leaf $matchDir
                    $canonicalLog = Join-Path $matchDir "$matchId.log"
                    if (Test-Path -LiteralPath $canonicalLog -PathType Leaf) {
                        if ((Test-Path -LiteralPath $outFile -PathType Leaf) -and
                            ($outFile -ne $canonicalLog)) {
                            Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue
                        }
                        $outFile = $canonicalLog
                    }
                    elseif (Test-Path -LiteralPath $outFile -PathType Leaf) {
                        Move-Item -LiteralPath $outFile -Destination $canonicalLog -Force
                        $outFile = $canonicalLog
                    }
                }
            }
            catch {
                "Could not archive match log: $_" | Out-File -LiteralPath $outFile -Append -Encoding utf8
            }
        }

        [pscustomobject]@{
            Index = $idx
            ExitCode = $exitCode
            LogFile = $outFile
        }
    } -ArgumentList $matchIndex, $MY_BOT_NAME, $MAP_NAME, $REAL_TIME, $ENEMY_RACE, $ENEMY_DIFFICULTY, $ENEMY_BUILD, `
        $BOT_INSTRUCT, $BOT_RACE, $COMMANDER_MODEL, `
        $FORCE_STRATEGY, $recordRoot, $BATCH_NAME, $logDir, $currentPath, $PYTHON_EXE, $sc2Path

    $jobs[$job.Id] = $job
    Write-Host "  Submitted match $i (Job ID: $($job.Id))"
}

Write-Host "Waiting for all batch jobs..."
$jobs.Values | Wait-Job | Out-Null
foreach ($j in @($jobs.Values)) {
    $result = Receive-Job $j
    Remove-Job $j

    foreach ($item in @($result)) {
        if ($item -and $null -ne $item.ExitCode) {
            Write-Host "[Job $($item.Index)] completed, exit=$($item.ExitCode), log=$($item.LogFile)"
            if ([int]$item.ExitCode -ne 0) {
                $failedMatches++
            }
        }
        elseif ($item) {
            Write-Host $item
        }
    }
}

if ($failedMatches -gt 0) {
    Write-Error "Batch finished, but $failedMatches match(es) failed. Check the log files above."
    exit 1
}

if ((Get-ChildItem -LiteralPath $logDir -Force | Measure-Object).Count -eq 0) {
    Remove-Item -LiteralPath $logDir
}
Write-Host "Batch finished successfully."
