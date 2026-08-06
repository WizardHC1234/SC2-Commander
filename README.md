# SC2-Commander

自用：用**单个 LLM Agent**执行 `strategy.md`，通过 OpenAI 兼容的 tool / JSON tool 调用下达宏观与军队命令；采矿、微操、本地防守等仍由 Sharpy 自动化处理。

## 功能摘要

- 单 Agent，**事件驱动决策**（开局 bootstrap + 模型 `set_wake_event`；漏写则 now+60 弱保底；另有独立 now+60 截止保险丝）
- 可执行命令均为 tool（训练/建造/研究/扩张/调兵/扫描/探图/唤醒条件）
- 每轮**整表替换**活跃目标；tool 顺序即资源优先级
- 军队侧：`move_group`（zone + movement mode）+ `scanner_sweep` / `scout`；微操在 Python
- Demo：Kairos Junction + `tank` + 内置 AI
- 内置人族策略：`marine`、`tank`、`battlecruiser`（旧名 `early_marine` / `mid_tank` / `late_battlecruiser` 仍作别名）

## 目录

```text
SC2-Commander/
  commander/     # Bot、提示词、tool、观测、军队执行、唤醒
  llm/           # API 调用；config.example.json（密钥勿提交）
  skills/        # 策略与 Action 空间（terran/marine|tank|battlecruiser）
  scripts/       # 启动脚本（demo / batch / 实验矩阵）
  tools/         # 批局统计等辅助脚本
  docs/          # 方案笔记
  sharpy/        # Sharpy 运行时
  bot_loader/    # 对局启动
  tests/         # 单元测试
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
      "is_reasoning": true,
      "tool_mode": "json"
    }
  }
}
```

说明：

- 启动时用 `--commander-model <key>`，key 必须与 `llm_agents_pool` 里的条目名一致
- **默认走 JSON tool_mode**（把工具名+描述写入 prompt，模型在正文里吐 `tool_calls` JSON）
- 需要原生 OpenAI `tools=` 时，在模型配置里设 `"tool_mode": "native"`；若服务端不支持，运行时自动回退到 JSON，同一进程内后续决策会记住该回退
- `is_reasoning`：对 Qwen 等会注入 `chat_template_kwargs.enable_thinking`；推理模型开 `true`，普通补全用 `false`

### 提示词与策略

- System 提示在 [`commander/prompts.py`](commander/prompts.py)：SC2 通则、宏执行/唤醒、宏决策序、军队 zone/mode、军队规则、输出格式
- 具体开局与门控以 `skills/terran/<strategy>/strategy.md` 为准；Action 名与描述来自 [`skills/terran/Action.py`](skills/terran/Action.py)

### 决策调度

决策不再固定轮询。每轮模型必须调用 `set_wake_event` 声明下次唤醒条件；漏写或非法条件时，运行时自动挂 `game_time_at_least=now+60` 弱保底。另：**无论模型是否写了唤醒条件，运行时都会武装 `now+60` 截止保险丝**，避免条件长期不满足时睡死。

相关常量在 [`commander/bot.py`](commander/bot.py)：

```python
class CommanderBot(KnowledgeBot):
    OBS_RECORD_INTERVAL: float = 60.0  # 仅观测落盘，不触发 LLM
    WAKE_COOLDOWN: float = 2.0         # 两次决策最小间隔
```

谓词白名单与求值见 [`commander/wake_events.py`](commander/wake_events.py)。`FALLBACK_DELAY_SECONDS = 60` 控制截止保险丝。观测里的 `[Runtime Decision Trigger]` 带 `woken_by=`（具体谓词或 `runtime_deadline_fuse`）。

### 对局记录与训练字段

对局 JSON（`game_records/.../*.json`）中 `interactions` 对每轮 Commander 决策额外保存：

- `messages`：本轮初始 system/user（可直接当 SFT 输入）
- `messages_transcript`：含反射重试的完整对话
- `assistant_content` / `tool_calls`：最终输出
- `reflection_rounds`：被拒的中间轮（issues + 原文）
- `woken_by`：本轮实际触发的唤醒条件
- `tool_mode` / `usage` / `usage_total` / `accepted`
- `strategy_id` / `strategy_hash`；`metadata` 含终局 `result` 与策略/模型

批局耗时统计可用：

```powershell
python tools/batch_time_stats.py
python tools/batch_time_stats.py --group-by strategy
```

### 对局默认值

[`run_vs_ai.py`](run_vs_ai.py) 顶部的 `DEFAULT_*`（地图、对手难度、默认策略、默认模型等）可直接改；CLI 参数会覆盖这些默认值。当前默认策略为 `tank`。

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
python run_vs_ai.py --force-strategy tank --commander-model qwen3-32b
python run_vs_ai.py --force-strategy marine --commander-model qwen3-32b
python run_vs_ai.py --force-strategy battlecruiser --commander-model qwen3-32b
```

常用参数：`--map-name`、`--enemy-race`、`--enemy-difficulty`、`--commander-model`、`--force-strategy`。

### 批量对局

单组配置（策略 / 难度 / 局数 / 并发）：

```powershell
.\scripts\start_batch.ps1 -FORCE_STRATEGY tank -ENEMY_DIFFICULTY hard -TOTAL_MATCHES 10 -CONCURRENCY 2 -COMMANDER_MODEL qwen3-32b
```

多组实验矩阵（编辑脚本内 `$EXPERIMENTS` 列表）：

```powershell
.\scripts\start_experiments_matrix.ps1
```

记录写入 `game_records/<batch_name>/`（已 gitignore）。

## 相关

- Sharpy 自用副本：https://github.com/WizardHC1234/sharpy-sc2
