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
  -> First-commitment Feasibility Audit：程序模拟采矿、采气、前置、补给和生产队列，计算门槛编成最早可行时间
  -> StrategyPatchValidator：同一假设、无第二优化目标、保持策略身份、内部一致
  -> 基础格式、可执行边界和 200 人口检查
  -> 保存不可覆盖的新候选目录
```

跨局 Decision 先从策略文本识别打法风格、核心取胜机制和相对强势窗口，再沿“承诺与接敌时机—双方接敌兵力—首战保留—增援与恢复—生产瓶颈”重建胜负链条。它同时比较等待前、实际接敌和等待后的敌军变化：更大的己方军队不一定更强，因为等待也可能让敌方克制兵种成型。该过程不是固定类别排序，最终只选择证据最充分且能通过策略文本修复的一个断点。

Optimizer 先生成完整候选，再从 Parent 与 Candidate 中抽取首次有效进攻所需的经济目标、单位门槛、生产槽和科技前置。确定性 Terran 模拟器从项目实际采用的 8 SCV、13 补给开局状态出发，使用 Commander 动作元数据与 Sharpy 相同的矿气收入近似，逐事件处理 SCV 分配与饱和、建造占用、扩张、Refinery、补给、附属建筑和生产队列，分别计算两个策略的 `earliest_feasible_time`。该时间只表示目标编成最早可以完成，不叠加模型决策、集结、移动或经验延迟。校验器再依据最小时间增量和对局中同期敌方成长判断修改是否破坏策略的关键窗口；不合理的候选会返回 Optimizer 重新生成。报告保存在候选的 `deterministic_feasibility_audit` 中。

每代只验证一个由对局证据支持的主要因果假设，并把它实现为一个 coherent strategy package：为了让假设可执行、资源可行、前置完整且内部一致，可以同步修改多个 paragraph 和战略维度；但不能夹带与该假设无关的第二优化目标。`plan.direction` 描述这个完整策略包，而不是 single lever 或 paragraph 名称。不增加固定优化类别 enum。知识库只在静态事实能区分解释时查询；对局时机、策略选择和 Commander 行为由对局证据与运行时边界决定。`considered_explanations` 留在 `analysis.json`，不写入长期 experiment_history。

每个实验还会预注册 `mechanism_prediction`：候选应改变的可观察中间状态、构成有效测试所需的最低实质变化、预期比赛效果、决定性交战应改善的 `combat_success_measure` 和真正的反证条件。外层评测结束后，Post-experiment Audit 会重新比较 Parent 与 Candidate 的完整对局摘要，沿“策略规则—Commander 决策—实际应用指令—后续状态”检查机制是否真正发生，并分别记录 `implementation_verdict`、`hypothesis_verdict`、`mechanism_evidence` 与 `combat_evidence`。胜率上升不会自动把假设标记为 supported；只有最低机制变化已实现且决定性交战证据与预测一致时才允许支持该假设。依赖底层微操、单位变形、装载或技能施放的候选会被标记为 `execution_invalid`，不能晋级。

策略优化的首要方向是赢下决定性交战，或以足够兵力保留通过该交战并继续取胜计划。更早进攻、减少资源积压、提高生产同步、达到数量门槛和增加侦察都只是中间机制，必须明确说明其如何改善首战结果或兵力保留率，不能作为独立的最终优化目标。

## 职责边界

EvolAgent 负责提出候选，不负责判断候选是否更强。每轮始终以唯一的官方 Champion 作为候选生成和评分基线。候选必须由外层实验流程实际进行 10 局评测：得分严格高于当前 Champion 才能接受；其他候选只作为后续分析经验，不会成为下一轮的文本父策略。

策略目录是不可变版本。进化跑次把候选写到
`evolution_runs/<strategy>/<timestamp>/strategies/<base>_optN/`，并在
`game_records/` 的 batch/match 目录旁保存 `strategy.md`。单独调用 EvolAgent
且未指定 `--output-dir` 时，仍会使用下一个可用的
`skills/<race>/<base>_optN/`。已经包含文件的候选目录不会被覆盖。

## 策略文件

候选目录只包含 `strategy.md`，格式固定为：

```markdown
# Summary

A short description of the strategy's economy, army style, power stage, and win plan.

# Details

* Opening and Economy: ...
* Expansion: ...
* Production: ...
* Technology: ...
* Scouting: ...
* Scans: ...
* Pre-Attack Army Posture: ...
* Main Attack Gate: ...
* Attack Objective: ...
* Engagement and Reinforcement: ...
* Recovery and Cleanup: ...
* Ultimate Goal: ...
```

`Main Attack Gate` 只定义首次进攻，不能作为恢复阶段反复使用的门槛。`Scouting` 和 `Scans` 服务于目标选择与残局清理，不能成为隐藏进攻门槛。每局上限为 30 分钟，Evol Agent 会把及时取得胜利作为全局分析条件。不要添加 `Resource Costs` 或 `Required Tools`。资源、人口、建造时间、生产者和前置条件来自 Commander 动作目录元数据。

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
