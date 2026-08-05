# SC2-Commander

自用：用**单个 LLM Agent**执行 `strategy.md`，通过 OpenAI 兼容的 tool / JSON tool 调用下达宏观与军队命令；采矿、微操、本地防守等仍由 Sharpy 自动化处理。

## 功能摘要

- 单 Agent，默认每 **60s** 决策一次（可改）
- 可执行命令均为 tool（训练/建造/研究/扩张/调兵/扫描/探图）
- 每轮**整表替换**活跃目标；tool 顺序即资源优先级
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

- StarCraft II，并设置环境变量 `SC2PATH`（未设置时脚本会回退到 `D:\StarCraft II`）
- Python 3.8+，依赖见 `requirements.txt`（含 `burnysc2`、`openai` 等）
- 本机可能还需可用的 `sc2pathlib` 原生扩展（`.pyd` / `.so` 不入库，需自行编译或拷贝）

## 配置

### LLM（必配）

```powershell
Copy-Item llm\config.example.json llm\config.json
# 编辑 llm\config.json，填入 api_url / api_key / model_name
```

`llm/config.json` 含密钥，**不要提交**。仓库只带 `llm/config.example.json`。

配置结构示例：

```json
{
  "llm_agents_pool": {
    "qwen3-32b": {
      "api_url": "http://127.0.0.1:8000/v1",
      "api_key": "EMPTY",
      "model_name": "qwen3-32b",
      "temperature": 0.5,
      "tool_mode": "json"
    }
  }
}
```

说明：

- 启动时用 `--commander-model <key>`，key 必须与 `llm_agents_pool` 里的条目名一致
- 本地 vLLM 若未开 auto tool choice，在对应条目加 `"tool_mode": "json"`（用回复正文里的 JSON tool_calls）
- 支持原生 OpenAI `tool_calls` 的服务可不设 `tool_mode`

### 决策间隔

在 [`commander/bot.py`](commander/bot.py) 中修改：

```python
class CommanderBot(KnowledgeBot):
    DECISION_INTERVAL: float = 60.0  # 秒；当前默认 60
```

改完后需重新启动对局才会生效。

### 对局默认值

[`run_vs_ai.py`](run_vs_ai.py) 顶部的 `DEFAULT_*`（地图、对手难度、默认策略、默认模型等）可直接改；CLI 参数会覆盖这些默认值。

常用脚本里也可改：

- [`scripts/start_demo.ps1`](scripts/start_demo.ps1)：单局 demo
- [`scripts/start_batch.ps1`](scripts/start_batch.ps1) / [`scripts/start_experiments_matrix.ps1`](scripts/start_experiments_matrix.ps1)：批量实验的模型、策略、并发、局数

### 不入库的本地数据

`.gitignore` 会忽略：`llm/config.json`、`game_records/`、`*.SC2Replay`、`*.log`、`analysis_results/`、venv、本地二进制等。

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

### 批量对局

单组配置（策略 / 难度 / 局数 / 并发）：

```powershell
.\scripts\start_batch.ps1 -FORCE_STRATEGY mid_tank -ENEMY_DIFFICULTY hard -TOTAL_MATCHES 10 -CONCURRENCY 2 -COMMANDER_MODEL qwen3-32b
```

多组实验矩阵（编辑脚本内 `$EXPERIMENTS` 列表）：

```powershell
.\scripts\start_experiments_matrix.ps1
```

记录写入 `game_records/<batch_name>/`（已 gitignore）。

## 相关

- Sharpy 自用副本：https://github.com/WizardHC1234/sharpy-sc2
