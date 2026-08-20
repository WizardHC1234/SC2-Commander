param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [string[]]$Records = @(),

    [string]$SpecDir = "tools\strategy_specs",

    [string]$OutDir = "",

    [switch]$NoPlots
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python was not found: $python"
}

$resolvedRunDir = (Resolve-Path -LiteralPath $RunDir).Path
if (-not $OutDir) {
    $runName = Split-Path -Leaf $resolvedRunDir
    $OutDir = Join-Path $projectRoot "analysis_results\evolution\$runName"
}
$resolvedOutDir = [IO.Path]::GetFullPath($OutDir)
$consistencyCsv = ""

if ($Records.Count -gt 0) {
    $consistencyDir = Join-Path $resolvedOutDir "strategy_execution"
    $metricArgs = @(
        (Join-Path $projectRoot "tools\strategy_execution_metrics.py"),
        "--input"
    )
    $metricArgs += $Records
    $metricArgs += @(
        "--spec-dir", (Join-Path $projectRoot $SpecDir),
        "--difficulties", "all",
        "--out-dir", $consistencyDir
    )
    & $python @metricArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Strategy-consistency calculation failed with exit code $LASTEXITCODE"
    }
    $consistencyCsv = Join-Path $consistencyDir "per_game.csv"
}

$reportArgs = @(
    (Join-Path $projectRoot "tools\evolution_report.py"),
    "--run-dir", $resolvedRunDir,
    "--out-dir", $resolvedOutDir
)
if ($consistencyCsv) {
    $reportArgs += @("--consistency-csv", $consistencyCsv)
}
if ($NoPlots) {
    $reportArgs += "--no-plots"
}

& $python @reportArgs
if ($LASTEXITCODE -ne 0) {
    throw "Evolution report generation failed with exit code $LASTEXITCODE"
}

Write-Host "Report output: $resolvedOutDir"
