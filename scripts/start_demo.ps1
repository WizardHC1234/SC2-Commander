$ErrorActionPreference = "Stop"

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not $env:SC2PATH -or -not (Test-Path -LiteralPath $env:SC2PATH)) {
    $env:SC2PATH = "D:\StarCraft II"
}

$MAP_NAME = "KairosJunctionLE"
$ENEMY_RACE = "terran"
$ENEMY_DIFFICULTY = "hard"
$ENEMY_BUILD = "macro"
$BOT_RACE = "terran"
$FORCE_STRATEGY = "mid_tank"
$COMMANDER_MODEL = "qwen3-32b"

$pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "未找到 python。请先安装并加入 PATH。"
    exit 1
}
$PYTHON_EXE = $pythonCmd.Source

if (-not (Test-Path ".\llm\config.json")) {
    Write-Error "缺少 llm\config.json。请复制 llm\config.example.json 并填入密钥。"
    exit 1
}

& $PYTHON_EXE -c "import openai" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "当前 Python 缺少 openai：$PYTHON_EXE"
    exit 1
}

& $PYTHON_EXE run_vs_ai.py `
  --my-bot-name commander `
  --map-name $MAP_NAME `
  --bot-race $BOT_RACE `
  --enemy-race $ENEMY_RACE `
  --enemy-difficulty $ENEMY_DIFFICULTY `
  --enemy-build $ENEMY_BUILD `
  --force-strategy $FORCE_STRATEGY `
  --commander-model $COMMANDER_MODEL
