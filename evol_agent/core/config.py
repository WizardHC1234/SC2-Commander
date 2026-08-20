from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SKILL_ROOT = PROJECT_ROOT / "skills"
OPTIMIZATION_LOG_DIR = PACKAGE_ROOT / "logs"

# EvolAgent reads and rewrites only the natural-language strategy.
SKILL_FILES = ["strategy.md"]
REQUIRED_TOP_HEADINGS = ["# Summary", "# Details"]

# Keys must exist in llm/config.json -> llm_agents_pool. A non-empty CLI
# --model value overrides both role defaults.
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_ANALYSIS_MODEL = DEFAULT_MODEL
DEFAULT_OPTIMIZATION_MODEL = DEFAULT_MODEL

MATCH_SUBAGENT_ENABLE_REASONING = False
ANALYSIS_ENABLE_REASONING = False
OPTIMIZATION_ENABLE_REASONING = False

DEFAULT_KNOWLEDGE_MODE = "enabled"
KNOWLEDGE_MODES = ("enabled", "disabled")


def resolve_model(explicit: str = "", *, role: str = "default") -> str:
    value = str(explicit or "").strip()
    if value:
        return value
    if role == "analysis":
        return DEFAULT_ANALYSIS_MODEL
    if role == "optimization":
        return DEFAULT_OPTIMIZATION_MODEL
    return DEFAULT_MODEL


# JSON transport retries happen inside call_json_llm.
LLM_CALL_MAX_ATTEMPTS = 2
LLM_CALL_RETRY_DELAYS_SECONDS = (2.0, 5.0)
# Bound each provider request so a live process cannot wait forever on a
# half-open connection. Large batch-analysis prompts can legitimately take
# several minutes on reasoning models, so keep this above Commander turn time.
# Bound each provider request so a live process cannot wait forever on a
# half-open connection. Large batch-analysis prompts can legitimately take
# several minutes on reasoning models, so keep this above Commander turn time.
LLM_CALL_TIMEOUT_SECONDS = 300.0

# Match summaries are independent and can be produced concurrently.
MAX_CONCURRENT_MATCH_SUBAGENTS = 3

# Knowledge lookup is optional: match evidence drives the analysis, and static
# SC2 facts are queried only when they can change a candidate decision.
MIN_KNOWLEDGE_QUERIES = 0
MAX_KNOWLEDGE_QUERIES = 4

# Candidate retries feed every structural, deterministic-knowledge, basic, and
# semantic validation error back to the optimizer. Keep this bounded: retries
# are for repairing a concrete error, not for unconstrained strategy search.
MAX_VALIDATION_RETRIES = 4
