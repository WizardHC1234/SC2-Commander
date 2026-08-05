# 方案讨论（已落地骨架）

## 已决议

1. 单 Agent；扁平 tool；OpenAI `tool_calls`
2. 整表替换；scout 不调 = 取消
3. 保留自动化；保留 `strategy.md`
4. 决策间隔 20s
5. Demo：Kairos + mid_tank + 内置 AI
6. **目录按 Commander 重设计**，不沿用旧多 Agent 布局；**不单独拆 army 子包**

## 当前包布局

见仓库根 `README.md`。核心在 `commander/` 扁平模块：`tools` / `agent` / `bot` / `macro_exec` / `combat_*`。

## 决议记录

- 2026-08-05：接口决议齐套；开始搭自包含仓
- 2026-08-05：按用户要求重设计目录；取消 army 独立子包
