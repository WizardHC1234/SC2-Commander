from __future__ import annotations

from pathlib import Path


# 路径配置：全部从当前文件位置推导，避免依赖程序的启动目录。
PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # evol_agent 包目录。
PROJECT_ROOT = PACKAGE_ROOT.parent  # SC2-Commander 项目根目录。
SKILL_ROOT = PROJECT_ROOT / "skills"  # 各种族策略文件的根目录。
OPTIMIZATION_LOG_DIR = PACKAGE_ROOT / "logs"  # Evol Agent 的运行日志目录。

# 优化范围：Agent 只读取和重写 strategy.md。
SKILL_FILES = ["strategy.md"]

# strategy.md 必须包含的一级标题，用于校验统一策略格式。
REQUIRED_TOP_HEADINGS = [
    "# Summary",
    "# Details",
]

# ---------------------------------------------------------------------------
# 模型配置（统一入口）
# key 必须存在于项目根目录 llm/config.json 的 llm_agents_pool。
# 改 DEFAULT_MODEL 即可切换整条 EvolAgent 默认链路；也可按角色单独覆盖。
# ---------------------------------------------------------------------------
# DEFAULT_MODEL = "kimi-k2.5"
# DEFAULT_MODEL = "qwen3.5-27b"
DEFAULT_MODEL = "deepseek-v4-flash"

# Analysis Agent（含 Match SubAgent）默认模型。
DEFAULT_ANALYSIS_MODEL = DEFAULT_MODEL

# Optimization Agent 默认模型。
DEFAULT_OPTIMIZATION_MODEL = DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Think / Reasoning 开关（统一入口）
# True：调用时传 is_reasoning / enable_reasoning；False：关闭。
# ---------------------------------------------------------------------------
# Match Summary Sub-Agent：默认关闭（只做事实总结）。
MATCH_SUBAGENT_ENABLE_REASONING = False

# Analysis Agent（diagnose / finish_analysis）。
ANALYSIS_ENABLE_REASONING = False

# Optimization Agent（draft / revise）。
OPTIMIZATION_ENABLE_REASONING = True

# 知识增强消融配置：enabled 使用确定性数据库查询；disabled 跳过。
DEFAULT_KNOWLEDGE_MODE = "enabled"
KNOWLEDGE_MODES = ("enabled", "disabled")


def resolve_model(explicit: str = "", *, role: str = "default") -> str:
    """Resolve an API pool model key.

    Non-empty ``explicit`` always wins. Otherwise the role-specific default
    from this module is used.
    """
    value = str(explicit or "").strip()
    if value:
        return value
    if role == "analysis":
        return DEFAULT_ANALYSIS_MODEL
    if role == "optimization":
        return DEFAULT_OPTIMIZATION_MODEL
    return DEFAULT_MODEL

# 一次结构化 LLM 请求最多尝试 3 次；首次、第二次失败后分别等待 2 秒和 5 秒。
LLM_CALL_MAX_ATTEMPTS = 3
LLM_CALL_RETRY_DELAYS_SECONDS = (2.0, 5.0)

# Analysis Agent 或 Optimization Agent 单次循环最多执行的决策步数。
MAX_EVOL_AGENT_STEPS = 10

# Analysis Agent 一轮跨对局分析最多保留的问题数量。
MAX_DIAGNOSED_PROBLEMS = 10

# 同时运行的 Match SubAgent 最大数量，用于限制并发模型请求。
MAX_CONCURRENT_MATCH_SUBAGENTS = 3

# 一轮最多执行多少个确定性知识查询。
MAX_KNOWLEDGE_QUERIES = 10

# 优化候选未通过格式或内容校验后，最多允许模型额外修订 3 次。
# 因此加上第一次候选，最多会校验 4 个候选版本。
MAX_VALIDATION_RETRIES = 3
