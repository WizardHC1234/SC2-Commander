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
OPTIMIZATION_ENABLE_REASONING = True

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
LLM_CALL_MAX_ATTEMPTS = 3
LLM_CALL_RETRY_DELAYS_SECONDS = (2.0, 5.0)

# Match summaries are independent and can be produced concurrently.
MAX_CONCURRENT_MATCH_SUBAGENTS = 3

# Batch analysis may ask no questions; when needed it can ask up to five
# focused deterministic knowledge questions.
MAX_KNOWLEDGE_QUERIES = 5

# Candidate retries are only for basic strategy.md validation failures.
MAX_VALIDATION_RETRIES = 3
