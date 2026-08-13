$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Root = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $PSScriptRoot "organize_batch_json.py"

$Python = $null
$venvPython = Join-Path $Root "venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $Python = $venvPython
}
if (-not $Python -and -not [string]::IsNullOrWhiteSpace($env:VIRTUAL_ENV)) {
    $active = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    if (Test-Path -LiteralPath $active -PathType Leaf) {
        $Python = $active
    }
}
if (-not $Python) {
    foreach ($name in @("python", "python3", "py")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -notlike "*\WindowsApps\python.exe") {
            $Python = $cmd.Source
            break
        }
    }
}
if (-not $Python) {
    throw "Python not found. Use the repo venv or install Python first."
}

Set-Location $Root
& $Python $Script @args
exit $LASTEXITCODE
