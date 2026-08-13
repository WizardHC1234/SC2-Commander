$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# =============================================================================
# export_sft_jsonl.ps1
# Export Victory decision turns from game_records_json to SFT jsonl.
# Edit the config below, then run: .\tools\export_sft_jsonl.ps1
# Use @() for "no filter" on that axis.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Paths (empty string = repo default)
# -----------------------------------------------------------------------------
$RECORDS_DIR = ""   # default: <repo>/game_records_json
$OUT_PATH = ""      # default: <repo>/sft_data/sft.jsonl

# -----------------------------------------------------------------------------
# 2. Filters (add/remove strings; use @() for no filter)
# Options:
#   models:       kimi-k2.5 / qwen3-32b / qwen3.5-27b / deepseek-v4-flash
#   strategies:   marine / tank / battlecruiser
#   races:        terran / zerg / protoss
#   difficulties: mediumhard / hard / harder / veryhard
#   styles:       macro / rush / timing
# -----------------------------------------------------------------------------
$MODELS = @(
    "deepseek-v4-flash"
)

$STRATEGIES = @(
    "marine",
    "tank",
    "battlecruiser"
)

$RACES = @(
    "terran"
)

$DIFFICULTIES = @(
    "mediumhard",
    "hard",
    "harder",
    "veryhard"
)

$STYLES = @(
    "macro"
)

# -----------------------------------------------------------------------------
# 3. Export rules
# -----------------------------------------------------------------------------
$MIN_WINRATE = $null          # e.g. 0.7 ; $null = no limit
$TOOL_MODE = ""               # e.g. "json" ; "" = any
$LIMIT = 0                    # 0 = no limit
$DRY_RUN = $false
$INCLUDE_LOSSES = $false
$ALLOW_UNACCEPTED = $false

# =============================================================================
# Runtime (usually no need to edit)
# =============================================================================

$Root = Split-Path -Parent $PSScriptRoot
$PyScript = Join-Path $PSScriptRoot "export_sft_jsonl.py"

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

function Add-FilterArgs {
    param(
        [System.Collections.Generic.List[string]]$Target,
        [string]$Flag,
        [AllowNull()]
        [string[]]$Values
    )
    if ($null -eq $Values) {
        return
    }
    foreach ($value in $Values) {
        $text = [string]$value
        if (-not [string]::IsNullOrWhiteSpace($text)) {
            [void]$Target.Add($Flag)
            [void]$Target.Add($text.Trim())
        }
    }
}

$pyArgs = New-Object 'System.Collections.Generic.List[string]'
if (-not [string]::IsNullOrWhiteSpace($RECORDS_DIR)) {
    [void]$pyArgs.Add("--records-dir")
    [void]$pyArgs.Add($RECORDS_DIR)
}
if (-not [string]::IsNullOrWhiteSpace($OUT_PATH)) {
    [void]$pyArgs.Add("--out")
    [void]$pyArgs.Add($OUT_PATH)
}

Add-FilterArgs -Target $pyArgs -Flag "--model" -Values $MODELS
Add-FilterArgs -Target $pyArgs -Flag "--strategy" -Values $STRATEGIES
Add-FilterArgs -Target $pyArgs -Flag "--race" -Values $RACES
Add-FilterArgs -Target $pyArgs -Flag "--difficulty" -Values $DIFFICULTIES
Add-FilterArgs -Target $pyArgs -Flag "--style" -Values $STYLES

if ($null -ne $MIN_WINRATE) {
    [void]$pyArgs.Add("--min-winrate")
    [void]$pyArgs.Add([string]$MIN_WINRATE)
}
if (-not [string]::IsNullOrWhiteSpace($TOOL_MODE)) {
    [void]$pyArgs.Add("--tool-mode")
    [void]$pyArgs.Add($TOOL_MODE)
}
if ($LIMIT -gt 0) {
    [void]$pyArgs.Add("--limit")
    [void]$pyArgs.Add([string]$LIMIT)
}
if ($DRY_RUN) { [void]$pyArgs.Add("--dry-run") }
if ($INCLUDE_LOSSES) { [void]$pyArgs.Add("--include-losses") }
if ($ALLOW_UNACCEPTED) { [void]$pyArgs.Add("--allow-unaccepted") }

foreach ($extra in $args) {
    [void]$pyArgs.Add([string]$extra)
}

Write-Host "Python : $Python"
Write-Host ("Config : model=[{0}] strategy=[{1}] race=[{2}] difficulty=[{3}] style=[{4}]" -f `
    ($MODELS -join ','), ($STRATEGIES -join ','), ($RACES -join ','), `
    ($DIFFICULTIES -join ','), ($STYLES -join ','))
Write-Host ("Args   : {0}" -f ($pyArgs -join ' '))

Set-Location $Root
& $Python $PyScript @($pyArgs.ToArray())
exit $LASTEXITCODE
