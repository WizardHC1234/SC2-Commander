# EvolAgent

EvolAgent 是 SC2-Commander 的离线策略优化器。它读取多场对局记录，总结可重复的问题，按需查询内置 SC2 静态知识库，最后只生成新的 `skills/<race>/<strategy>_optN/strategy.md`。它不会修改 Commander 的 Observation、动作空间或运行时代码。

## 与当前运行时的关系

当前项目由一个 Commander 同时完成宏观、军队和唤醒决策。EvolAgent 直接读取当前录像中的：

- `metadata.strategy_id`：本局策略。
- `strategy_forced` / `strategy_tool_selection`：策略和动作目录选择。
- `commander_bootstrap` / `wake_event` / `wake_fallback_timeout`：Commander 决策触发。
- `observation`：完整 schema 2.0 状态。
- `macro_tasks`、`army_policy`、`tool_calls`、`wake_event`：本轮实际决策。
- `issues`、`reflection_issues`、`accepted`：验证和重试结果。

旧版 Strategy Coordinator、Macro Planner、Army Planner 的录像格式仍可读取，但只用于兼容历史数据。

## 优化流程

```text
对局事实总结（最多并行 3 个子任务）
  -> 跨局 Analysis：定位可重复、可由当前运行时执行的问题
  -> Knowledge：按实体和需求类型确定性读取内置 SC2 数据库
  -> Optimization：重写 strategy.md
  -> Deterministic Validator：检查格式与可执行边界
```

策略输出只允许两个一级标题：

```markdown
# Summary
一段简短的策略概述。

# Details
* Opening: 使用可观察条件和绝对目标描述开局。
* Main Attack Gate: 描述集结后的主力进攻条件。
```

不要添加 `# Resource Costs` 或 `# Required Tools`。矿物、Gas、人口、建造/研究时间、生产者和前置条件由 Commander 的动作目录元数据提供，不在策略文档中重复。

军队规则必须遵守当前语义：持久主力是唯一作战主体；`group_1` 是新生产单位离主力较远时形成的临时增援组，应汇入主力或前往主力当前目标，不能独立发起进攻、骚扰或搜索路线。

## 运行

模型配置统一读取项目根目录的 `llm/config.json`。`--model` 必须是其中 `llm_agents_pool` 的 key。

```powershell
# 使用内置 SC2 静态知识增强
.\venv\Scripts\python.exe -m evol_agent.cli `
  --batch-dir "D:\path\to\game_records\batch_name" `
  --strategy tank `
  --knowledge-mode enabled

# 消融组：不查询静态知识
.\venv\Scripts\python.exe -m evol_agent.cli `
  --batch-dir "D:\path\to\game_records\batch_name" `
  --strategy tank `
  --knowledge-mode disabled

# 只分析，不生成策略
.\venv\Scripts\python.exe -m evol_agent.cli `
  --batch-dir "D:\path\to\game_records\batch_name" `
  --strategy tank `
  --dry-run
```

当前内置 Terran 策略为 `marine`、`tank`、`battlecruiser`。默认输出目录是 `skills/terran/<strategy>_optN/`；目录生成后可直接传给 `run_vs_ai.py --force-strategy <strategy>_optN`，不需要 `registry.json`。

## 赛后敌方真值

`run_vs_ai.py` 默认在比赛结束后重放同目录的 `.SC2Replay`，按 Commander 的决策帧提取玩家 2 自己的完整资源、人口、单位、建筑、升级和生产状态，保存为同名前缀的 `.enemy_truth.json`。原始对局 JSON 不会被改写；固定时间表会把迷雾下的 `enemy` 和赛后的 `opponent_truth_after_match` 分开呈现，避免把赛后信息误当成 Commander 当时已知的信息。

可以使用 `--no-extract-enemy-truth` 关闭自动提取。旧批次可单独补齐：

```powershell
.\venv\Scripts\python.exe -m evol_agent.analysis.replay_truth `
  --batch-dir "D:\path\to\game_records\batch_name"
```

## 内置 SC2 数据

`evol_agent/sc2_data_agent/` 内含 `data_sc2_260701` 数据集。EvolAgent 的知识问题使用 `entities` 和 `needs` 直接读取实体卡、能力、生产/科技条件以及克制/协同关系，不再经过多轮 LLM 工具规划。Commander 已有动作的成本、时间、生产位置和依赖以动作目录元数据为准；数据库只补充能力效果和关系知识。

该数据集不包含 Commander 的动作替换语义、`group_1` 汇入逻辑、移动模式、唤醒条件或 Sharpy 微操行为。这些问题必须依据当前 `RUNTIME_CONTRACT` 和对局证据判断，不能交给静态知识 Agent 猜测。

运行日志写入 `evol_agent/logs/`，该目录不应提交到版本库。
