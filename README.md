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
  commander/     # Bot、提示词、tool、观测、执行层，以及 races/<race> 适配器
  evol_agent/    # 离线读取对局记录并生成优化版 strategy.md
  llm/           # API 调用；config.example.json（密钥勿提交）
  skills/        # 仅保存各族的 strategy.md 策略内容
  scripts/       # 启动脚本（demo / batch / 实验矩阵）
  tools/         # 批局统计等辅助脚本
  docs/          # 方案笔记
  sharpy/        # Sharpy 运行时
  bot_loader/    # 对局启动
  tests/         # 单元测试
  run_vs_ai.py   # 对内置 AI 入口
```

## 环境

### 前置

- StarCraft II，并设置环境变量 `SC2PATH`
  - Windows：未设置时脚本会回退到 `D:\StarCraft II`
  - Linux：通常指向安装目录，例如 `~/StarCraftII` 或 Battle.net 安装路径
- Python 3.8+（推荐 3.9）
  - Windows：勿用 Microsoft Store 的 `WindowsApps\python.exe` 占位符
  - Linux：一般用发行版自带的 `python3`（缺包时安装 `python3-venv` / `python3-pip`）
- 本机可能还需可用的 `sc2pathlib` 原生扩展（`.pyd` / `.so` 不入库，需自行编译或拷贝）

### 创建 / 激活 venv（Windows）

仓库脚本默认使用仓库根目录下的 `venv\`（例如 `.\venv\Scripts\python.exe`）。首次：

```powershell
cd D:\path\to\SC2-Commander

# 用本机真实 Python 创建虚拟环境（按你的安装路径改）
python -m venv venv
# 或： py -3.9 -m venv venv
# 或： D:\Softwares\miniconda3\envs\my_env\python.exe -m venv venv

.\venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

开发/测试额外依赖（可选）：

```powershell
pip install -r requirements.dev.txt
```

之后每次开新终端：

```powershell
cd D:\path\to\SC2-Commander
.\venv\Scripts\Activate.ps1
```

不激活也可以直接调用解释器：

```powershell
.\venv\Scripts\python.exe run_vs_ai.py --force-strategy tank --commander-model qwen3-32b
.\venv\Scripts\python.exe -m evol_agent.cli --help
```

### 创建 / 激活 venv（Linux）

在仓库根目录创建 `venv/`（解释器路径为 `venv/bin/python`）：

```bash
cd /path/to/SC2-Commander

# 若提示缺少 venv 模块：sudo apt install python3-venv  （Debian/Ubuntu）
python3 -m venv venv
# 或指定版本：python3.9 -m venv venv

source venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

开发/测试额外依赖（可选）：

```bash
pip install -r requirements.dev.txt
```

之后每次开新终端：

```bash
cd /path/to/SC2-Commander
source venv/bin/activate
```

不激活也可以直接调用解释器：

```bash
./venv/bin/python run_vs_ai.py --force-strategy tank --commander-model qwen3-32b
./venv/bin/python -m evol_agent.cli --help
```

可选：把 `SC2PATH` 写进 shell 配置或当前会话：

```bash
export SC2PATH="$HOME/StarCraftII"
```

说明：Linux 上请用 `python` / `./venv/bin/python` 直接跑对局；仓库里的 `scripts/*.ps1` 面向 Windows PowerShell。

### 说明（两端通用）

- `venv/` 已在 `.gitignore` 中，不要提交
- Windows：`scripts\start_demo.ps1` / `start_batch.ps1` 会优先找 `venv\Scripts\python.exe`（其次当前 `$env:VIRTUAL_ENV`）；`start_evolution.ps1` **要求**仓库根已有该 venv
- 若 `import openai` / `import sc2` 失败，先确认用的是对应平台的 venv 解释器，再重装 `requirements.txt`

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
- 具体开局与门控以 `skills/terran/<strategy>/strategy.md` 为准；动作执行、静态数据和工具描述统一定义在 [`commander/races/terran/actions.py`](commander/races/terran/actions.py)

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

### 离线策略优化

`evol_agent/` 可读取当前单 Commander 的对局记录，执行跨局分析，并只生成新的 `strategy.md` 候选。进化流程把候选保存在对应 run 的 `strategies/` 以及对局记录目录里，避免在 `skills/` 下堆积 `_optN` 文件夹。它与 Commander 共用 `llm/config.json`，静态成本与前置条件来自动作目录；策略文件使用 `# Summary` 和 `# Details` 两部分。

```powershell
.\venv\Scripts\python.exe -m evol_agent.cli --batch-dir game_records\<batch_name> --strategy tank
```

详细说明见 [`evol_agent/README.md`](evol_agent/README.md)。

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
