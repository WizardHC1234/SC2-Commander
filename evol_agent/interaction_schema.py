"""Interaction schema helpers for current and legacy match records.

The current runtime has one ``commander`` agent.  The multi-agent constants
below are retained only so historical records remain readable by EvolAgent.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Current single-Commander schema (primary)
# ---------------------------------------------------------------------------
CURRENT_AGENT = "commander"
CURRENT_FIELD_OBSERVATION = "observation"

TRIGGER_STRATEGY_FORCED = "strategy_forced"
TRIGGER_STRATEGY_TOOL_SELECTION = "strategy_tool_selection"
TRIGGER_COMMANDER_BOOTSTRAP = "commander_bootstrap"
TRIGGER_WAKE_EVENT = "wake_event"
TRIGGER_WAKE_FALLBACK_TIMEOUT = "wake_fallback_timeout"
TRIGGER_AUTO_RETREAT = "auto_retreat_triggered"

CURRENT_COMMANDER_TRIGGERS = frozenset(
    {
        TRIGGER_COMMANDER_BOOTSTRAP,
        TRIGGER_WAKE_EVENT,
        TRIGGER_WAKE_FALLBACK_TIMEOUT,
        TRIGGER_AUTO_RETREAT,
    }
)
CURRENT_RECORD_TRIGGERS = frozenset(
    {
        TRIGGER_STRATEGY_FORCED,
        TRIGGER_STRATEGY_TOOL_SELECTION,
        *CURRENT_COMMANDER_TRIGGERS,
    }
)


def is_current_commander_interaction(interaction: dict[str, Any] | None) -> bool:
    """Return whether an interaction belongs to the current runtime schema."""
    if not isinstance(interaction, dict):
        return False
    trigger = str(interaction.get("trigger_reason") or "")
    return bool(
        interaction.get("agent") == CURRENT_AGENT
        or isinstance(interaction.get(CURRENT_FIELD_OBSERVATION), dict)
        or trigger in CURRENT_RECORD_TRIGGERS
    )


def current_agent_role_for_trigger(trigger: Any) -> str:
    """Return the current framework role represented by a record event."""
    name = str(trigger or "unknown")
    if name == TRIGGER_STRATEGY_FORCED:
        return "init"
    if name == TRIGGER_STRATEGY_TOOL_SELECTION:
        return "selector"
    if name in CURRENT_COMMANDER_TRIGGERS:
        return CURRENT_AGENT
    return "-"


# ---------------------------------------------------------------------------
# Legacy multi-agent keys (read compatibility only)
# ---------------------------------------------------------------------------
STRATEGY_COORDINATOR_STRATEGY = "strategy_coordinator_strategy"
STRATEGY_COORDINATOR_BUILD_DIRECTIVE = "strategy_coordinator_build_directive"
STRATEGY_COORDINATOR_ARMY_DIRECTIVE = "strategy_coordinator_army_directive"
STRATEGY_COORDINATOR_COORDINATION = "strategy_coordinator_coordination"
STRATEGY_COORDINATOR_INITIAL = "strategy_coordinator_initial"

MACRO_PLANNER_INPUT_PREVIOUS_TASKS = "macro_planner_input_previous_tasks"
MACRO_PLANNER_RAW_RESPONSE = "macro_planner_raw_response"
MACRO_PLANNER_OUTPUT_NEW_TASKS = "macro_planner_output_new_tasks"

ACTION_TRANSLATOR_TRANSLATIONS = "action_translator_translations"

# Canonical trigger_reason values
TRIGGER_STRATEGY_COORDINATOR_INITIAL = "strategy_coordinator_initial"
TRIGGER_STRATEGY_COORDINATOR_INITIAL_FORCED = "strategy_coordinator_initial_forced"
TRIGGER_MACRO_PLANNER_POLL = "macro_planner_poll"
TRIGGER_ARMY_PLANNER_POLL = "army_control_agent_poll"

# ---------------------------------------------------------------------------
# Read aliases: canonical first, then legacy
# ---------------------------------------------------------------------------
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    STRATEGY_COORDINATOR_STRATEGY: (
        STRATEGY_COORDINATOR_STRATEGY,
        "top_agent_strategy",
    ),
    STRATEGY_COORDINATOR_BUILD_DIRECTIVE: (
        STRATEGY_COORDINATOR_BUILD_DIRECTIVE,
        "top_agent_build_directive",
        "top_agent_focus",
    ),
    STRATEGY_COORDINATOR_ARMY_DIRECTIVE: (
        STRATEGY_COORDINATOR_ARMY_DIRECTIVE,
        "top_agent_army_directive",
    ),
    STRATEGY_COORDINATOR_COORDINATION: (
        STRATEGY_COORDINATOR_COORDINATION,
        "top_agent_coordination",
        "top_agent_phase_assessment",
    ),
    STRATEGY_COORDINATOR_INITIAL: (
        STRATEGY_COORDINATOR_INITIAL,
        "top_agent_initial",
    ),
    MACRO_PLANNER_INPUT_PREVIOUS_TASKS: (
        MACRO_PLANNER_INPUT_PREVIOUS_TASKS,
        "mid_agent_input_previous_tasks",
    ),
    MACRO_PLANNER_RAW_RESPONSE: (
        MACRO_PLANNER_RAW_RESPONSE,
        "mid_agent_raw_response",
    ),
    MACRO_PLANNER_OUTPUT_NEW_TASKS: (
        MACRO_PLANNER_OUTPUT_NEW_TASKS,
        "mid_agent_output_new_tasks",
    ),
    ACTION_TRANSLATOR_TRANSLATIONS: (
        ACTION_TRANSLATOR_TRANSLATIONS,
        "down_agent_translations",
    ),
}


def interaction_get(
    interaction: dict[str, Any] | None,
    canonical_key: str,
    default: Any = None,
) -> Any:
    """Return the first present value among canonical + legacy aliases."""
    if not isinstance(interaction, dict):
        return default
    for key in FIELD_ALIASES.get(canonical_key, (canonical_key,)):
        if key in interaction and interaction[key] is not None:
            return interaction[key]
    return default


def interaction_get_dict(
    interaction: dict[str, Any] | None,
    canonical_key: str,
) -> dict[str, Any]:
    value = interaction_get(interaction, canonical_key, default={})
    return value if isinstance(value, dict) else {}


def interaction_get_list(
    interaction: dict[str, Any] | None,
    canonical_key: str,
) -> list[Any]:
    value = interaction_get(interaction, canonical_key, default=[])
    return value if isinstance(value, list) else []


def interaction_get_str(
    interaction: dict[str, Any] | None,
    canonical_key: str,
    default: str = "",
) -> str:
    value = interaction_get(interaction, canonical_key, default=default)
    if value is None:
        return default
    return str(value)


def normalize_trigger_reason(raw_trigger: Any) -> str:
    """Keep current triggers intact and normalize historical trigger aliases."""
    trigger = str(raw_trigger or "unknown").strip() or "unknown"
    if trigger in CURRENT_RECORD_TRIGGERS:
        return trigger
    if trigger in {"poll", "mid_agent_poll", TRIGGER_MACRO_PLANNER_POLL}:
        return TRIGGER_MACRO_PLANNER_POLL
    if trigger in {
        "top_agent_initial_t0",
        TRIGGER_STRATEGY_COORDINATOR_INITIAL,
    }:
        return TRIGGER_STRATEGY_COORDINATOR_INITIAL
    if trigger in {
        "top_agent_initial_t0_forced",
        TRIGGER_STRATEGY_COORDINATOR_INITIAL_FORCED,
    }:
        return TRIGGER_STRATEGY_COORDINATOR_INITIAL_FORCED
    return trigger
