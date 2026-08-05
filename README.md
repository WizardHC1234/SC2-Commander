# SC2-Commander

自用：用**单个 LLM Agent**执行 `strategy.md`，通过 OpenAI 兼容的 tool / JSON tool 调用下达宏观与军队命令；采矿、微操、本地防守等仍由 Sharpy 自动化处理。

## 功能摘要

- 单 Agent，每 **20s** 决策一次
- 可执行命令均为 tool（训练/建造/研究/扩张/调兵/扫描/探图）
- 每轮**整表替换**活跃目标
- Demo：Kairos Junction + `mid_tank` + 内置 AI
- 内置人族策略：`early_marine`、`mid_tank`、`late_battlecruiser`

## 目录

```text
SC2-Commander/
  commander/     # Bot、提示词、tool、观测、军队执行
  llm/           # API 调用；config.example.json（密钥勿提交）
  skills/        # 策略与 Action 空间（terran/early_marine|mid_tank|late_battlecruiser）
  scripts/       # 启动脚本
  docs/          # 方案笔记
  sharpy/        # Sharpy 运行时
  bot_loader/    # 对局启动
  run_vs_ai.py   # 对内置 AI 入口
```

## 环境

- StarCraft II，并设置环境变量 `SC2PATH`
- Python 3.8+，依赖见 `requirements.txt`（含 `burnysc2`、`openai` 等）
- 本机可能还需可用的 `sc2pathlib` 原生扩展（`.pyd` / `.so` 不入库，需自行编译或拷贝）

## 配置

```powershell
Copy-Item llm\config.example.json llm\config.json
# 编辑 llm\config.json，填入 api_url / api_key / model_name
```

`llm/config.json` 含密钥，**不要提交**。

可选：在模型条目中设置 `"tool_mode": "json"`（本地 vLLM 未开 auto tool choice 时使用）。

## 运行

```powershell
.\scripts\start_demo.ps1
```

或：

```powershell
python run_vs_ai.py --force-strategy mid_tank --commander-model qwen3-32b
python run_vs_ai.py --force-strategy early_marine --commander-model qwen3-32b
python run_vs_ai.py --force-strategy late_battlecruiser --commander-model qwen3-32b
```

常用参数：`--map-name`、`--enemy-race`、`--enemy-difficulty`、`--commander-model`、`--force-strategy`。

对局记录默认写入本地 `game_records/`（已 gitignore）。

## 相关

- Sharpy 自用副本：https://github.com/WizardHC1234/sharpy-sc2
