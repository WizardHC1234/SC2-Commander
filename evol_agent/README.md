# EvolAgent

EvolAgent 是 SC2-Commander 的离线候选策略生成器。它读取一批完整对局记录，分析当前自然语言策略的主要瓶颈，按需查询内置 SC2 静态知识库，最后只生成一个新的 `strategy.md` 候选。它不修改 Commander、Observation、动作空间或模型参数，也不自行宣称候选已经优化成功。

## 当前流程

```text
完整对局记录
  -> 每局固定时间表的事实总结（并行）
  -> 一次跨局分析
       - 当前策略如何取胜
       - 必须保留什么
       - 一个主要可控问题
       - 一个连贯优化假设
       - 0 到 5 个必要知识问题
  -> 确定性 SC2 知识查询（按需）
  -> 一次候选策略生成
  -> 基础格式、可执行边界和 200 人口检查
  -> 保存不可覆盖的新候选目录
```

跨局分析不会再经过 `diagnose -> finish_analysis` 两次 LLM 转换。知识库只提供单位、建筑和升级的静态事实；对局时机、策略选择和 Commander 行为由对局证据与运行时边界决定。

## 职责边界

EvolAgent 负责提出候选，不负责判断候选是否更强。候选必须由外层实验流程实际进行 10 局评测：胜率高于当前 Champion 才能接受，否则保留 Champion，并把失败原因作为后续优化经验。

策略目录是不可变版本。默认输出使用下一个可用的 `skills/<race>/<base>_optN/`。已经包含文件的候选目录不会被覆盖，因此相同名称不会再混入多个不同策略内容。

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

## 敌方赛后真实状态

`run_vs_ai.py` 默认从 `.SC2Replay` 提取对手视角状态并保存为 `.enemy_truth.json`。逐局固定时间表会把迷雾下的 `enemy` 与 `opponent_truth_after_match` 分开，避免把赛后真值误当成 Commander 当时已知的信息。
