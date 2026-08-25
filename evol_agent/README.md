# EvolAgent

EvolAgent 是 SC2-Commander 的离线候选策略生成器。它读取一批完整对局记录，分析当前自然语言策略的主要瓶颈，按需查询内置 SC2 静态知识库，最后只生成一个新的 `strategy.md` 候选。它不修改 Commander、Observation、动作空间或模型参数，也不自行宣称候选已经优化成功。

## 当前流程

```text
完整对局记录
  -> 每局固定时间表的事实总结（并行）
  -> Cross-match Discovery：成功机制、反复问题、未知静态事实
  -> 确定性 SC2 知识查询（按需）
  -> Cross-match Decision：从决定性交战倒推、比较 2–4 个竞争解释（支持/反证）后选出一个 priority_problem、一个因果 hypothesis、一个 plan.direction
  -> Optimizer：为实现同一假设修改所有必要依赖段落
  -> StrategyPatchValidator：同一假设、无第二优化目标、保持策略身份、内部一致
  -> 基础格式、可执行边界和 200 人口检查
  -> 保存不可覆盖的新候选目录
```

跨局 Decision 在选定最终 hypothesis 前比较当前证据真正支持的竞争解释，并检查“瓶颈已解决后是否仍失败”。当多个问题都有证据时，它采用 SC2-aware reasoning preference：先检查组建完成的 army package 面对已观察敌军是否可战，再检查相对 power-spike timing、production/resource synchronization、economy/recovery、upgrade multiplier，以及信息是否真正改变战略决策。该顺序不是固定 enum 或确定性 selector，证据可以改变优先级。达到或超过 attack gate 后仍反复惨败时，不把更快达到同一门槛当作充分解释；首战基本合理但无法重建时，才优先考虑经济与恢复。

每代只验证一个由对局证据支持的主要因果假设，并把它实现为一个 coherent strategy package：为了让假设可执行、资源可行、前置完整且内部一致，可以同步修改多个 paragraph 和战略维度；但不能夹带与该假设无关的第二优化目标。`plan.direction` 描述这个完整策略包，而不是 single lever 或 paragraph 名称。不增加固定优化类别 enum。知识库只在静态事实能区分解释时查询；对局时机、策略选择和 Commander 行为由对局证据与运行时边界决定。`considered_explanations` 留在 `analysis.json`，不写入长期 experiment_history。

每个实验还会预注册 `mechanism_prediction`：候选应改变的可观察中间状态、构成有效测试所需的最低实质变化、预期比赛效果、决定性交战应改善的 `combat_success_measure` 和真正的反证条件。外层评测结束后，Post-experiment Audit 会重新比较 Parent 与 Candidate 的完整对局摘要，沿“策略规则—Commander 决策—实际应用指令—后续状态”检查机制是否真正发生，并分别记录 `implementation_verdict`、`hypothesis_verdict`、`mechanism_evidence` 与 `combat_evidence`。胜率上升不会自动把假设标记为 supported；只有最低机制变化已实现且决定性交战证据与预测一致时才允许支持该假设。依赖底层微操、单位变形、装载或技能施放的候选会被标记为 `execution_invalid`，不能晋级。

策略优化的首要方向是赢下决定性交战，或以足够兵力保留通过该交战并继续取胜计划。更早进攻、减少资源积压、提高生产同步、达到数量门槛和增加侦察都只是中间机制，必须明确说明其如何改善首战结果或兵力保留率，不能作为独立的最终优化目标。

## 职责边界

EvolAgent 负责提出候选，不负责判断候选是否更强。候选必须由外层实验流程实际进行 10 局评测：胜率高于当前 Champion 才能接受，否则保留 Champion，并把失败原因作为后续优化经验。

策略目录是不可变版本。进化跑次把候选写到
`evolution_runs/<strategy>/<timestamp>/strategies/<base>_optN/`，并在
`game_records/` 的 batch/match 目录旁保存 `strategy.md`。单独调用 EvolAgent
且未指定 `--output-dir` 时，仍会使用下一个可用的
`skills/<race>/<base>_optN/`。已经包含文件的候选目录不会被覆盖。

## 策略文件

候选目录只包含 `strategy.md`，格式固定为：

```markdown
# Summary

一段简短的策略概述。

# Details

* Opening and Economy: ...
* Production: ...
* Main Attack Gate: ...
```

不要添加 `Resource Costs` 或 `Required Tools`。资源、人口、建造时间、生产者和前置条件来自 Commander 动作目录元数据。

## 运行

```powershell
.\venv\Scripts\python.exe -m evol_agent.cli `
  --batch-dir "D:\path\to\game_records\batch_name" `
  --strategy tank `
  --knowledge-mode enabled
```

只运行分析、不生成候选：

```powershell
.\venv\Scripts\python.exe -m evol_agent.cli `
  --batch-dir "D:\path\to\game_records\batch_name" `
  --strategy tank `
  --dry-run
```

中断后可从日志目录恢复：

```powershell
.\venv\Scripts\python.exe -m evol_agent.cli `
  --resume "D:\User\HC\Desktop\Code\SC2-Commander\evol_agent\logs\tank\YYYYMMDD_HHMMSS"
```

运行日志保存在 `evol_agent/logs/<strategy>/<timestamp>/`。其中 `analysis.json` 保存单一优化假设，`knowledge_trace.json` 保存知识查询，`improvement.json` 保存候选内容和修改理由，`context.json` 保存完整运行上下文。

自动进化进入下一代时，如果 Parent 未改变而证据池只增加了确认局，系统会从最新兼容 checkpoint 复用已有逐场总结，只总结新增记录。上一轮跨局分析会作为可修正的分析种子与实验历史一起传入；新一轮仍以当前完整证据集为准，不会把旧分析当作额外对局重复计数。

## 敌方赛后真实状态

`run_vs_ai.py` 默认从 `.SC2Replay` 提取对手视角状态并保存为 `.enemy_truth.json`。逐局固定时间表会把迷雾下的 `enemy` 与 `opponent_truth_after_match` 分开，避免把赛后真值误当成 Commander 当时已知的信息。
