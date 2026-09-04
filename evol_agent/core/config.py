from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SKILL_ROOT = PROJECT_ROOT / "skills"
OPTIMIZATION_LOG_DIR = PACKAGE_ROOT / "logs"

# Evolution writes candidates here instead of skills/<race>/. Child match
# processes inherit this via the environment so Commander can load them.
STRATEGY_ROOT_ENV = "SC2_STRATEGY_ROOT"
STRATEGY_FOLDER_ALIASES = {
    "early_marine": "marine",
    "mid_tank": "tank",
    "late_battlecruiser": "battlecruiser",
}


def canonical_strategy_folder(name: str) -> str:
    key = str(name or "").strip()
    if not key:
        return key
    return STRATEGY_FOLDER_ALIASES.get(key.lower(), key)


def resolve_skill_dir(
    strategy_name: str,
    race: str = "terran",
    *,
    overlay_root: str | Path | None = None,
    skill_root: str | Path | None = None,
) -> Path:
    """Prefer a run-local strategy overlay, then skills/<race>/<name>."""
    folder = canonical_strategy_folder(strategy_name)
    roots: list[Path] = []
    if overlay_root:
        roots.append(Path(overlay_root))
    env_root = str(os.environ.get(STRATEGY_ROOT_ENV) or "").strip()
    if env_root:
        roots.append(Path(env_root))
    for root in roots:
        candidate = root / folder
        if (candidate / "strategy.md").is_file():
            return candidate
    skills = Path(skill_root) if skill_root is not None else SKILL_ROOT
    return skills / str(race or "terran").strip().lower() / folder

# EvolAgent reads and rewrites only the natural-language strategy.
SKILL_FILES = ["strategy.md"]
REQUIRED_TOP_HEADINGS = ["# Summary", "# Details"]
# Read compatibility for candidates/checkpoints briefly produced by the retired
# three-section experiment. New candidates use REQUIRED_TOP_HEADINGS above.
STRUCTURED_TOP_HEADINGS = ["# Goal", "# Macro", "# Combat"]
STRUCTURED_STRATEGY_FIELDS = {
    "Goal": ["Strategy Style", "Core Objective", "Key Principle"],
    "Macro": ["Economy and Expansion", "Production", "Technology", "Ultimate Goal"],
    "Combat": [
        "Pre-Attack Army Posture",
        "Scouting and Information",
        "Main Attack Gate",
        "Attack Objective",
        "Engagement and Reinforcement",
        "Recovery and Cleanup",
    ],
}

# Keys must exist in llm/config.json -> llm_agents_pool. A non-empty CLI
# --model value overrides both role defaults.
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_ANALYSIS_MODEL = DEFAULT_MODEL
DEFAULT_OPTIMIZATION_MODEL = DEFAULT_MODEL

# All EvolAgent stages keep think off. DeepSeek think on discovery/timing/
# package/writer calls stalls for minutes.
MATCH_SUBAGENT_ENABLE_REASONING = False
CROSS_MATCH_DISCOVERY_ENABLE_REASONING = False
OPTIMIZATION_PACKAGE_GENERATION_ENABLE_REASONING = False
OPTIMIZATION_PACKAGE_SELECTION_ENABLE_REASONING = False
CANDIDATE_GENERATION_ENABLE_REASONING = False
PARENT_TIMING_PACKAGE_EXTRACTION_ENABLE_REASONING = False
CONTACT_TIMING_EXTRACTION_ENABLE_REASONING = False
STRATEGY_SEMANTIC_VALIDATION_ENABLE_REASONING = False
MECHANISM_HISTORY_ENABLE_REASONING = False
EXPERIMENT_AUDIT_ENABLE_REASONING = False

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
# Reasoning calls from Kimi and similar models can exceed five minutes. Keep a
# finite bound, but leave enough room for a complete strategic judgment.
LLM_CALL_TIMEOUT_SECONDS = 480.0

# Match summaries are independent and can be produced concurrently.
MAX_CONCURRENT_MATCH_SUBAGENTS = 3

# Discovery may request a small number of deterministic SC2 facts. When it makes
# a production, timing, upgrade, or matchup claim but omits the query, the
# analysis loop adds one grounded fallback query.
MAX_KNOWLEDGE_QUERIES = 5

# Candidate retries feed every structural, deterministic-knowledge, basic, and
# semantic validation error back to the optimizer. Keep this bounded: retries
# are for repairing a concrete error, not for unconstrained strategy search.
MAX_VALIDATION_RETRIES = 4
