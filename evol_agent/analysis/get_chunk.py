"""Build compact strategy-optimization evidence from recorded SC2 interactions.

The extractor consumes the current single-Commander schema 2.0 ``observation``
and also retains compatibility with legacy multi-agent ``observation_view``
records.
"""

from __future__ import annotations

import json
import re
from typing import Any

from commander.observation import render_observation

from ..interaction_schema import (
    ACTION_TRANSLATOR_TRANSLATIONS,
    MACRO_PLANNER_INPUT_PREVIOUS_TASKS,
    MACRO_PLANNER_OUTPUT_NEW_TASKS,
    MACRO_PLANNER_RAW_RESPONSE,
    STRATEGY_COORDINATOR_ARMY_DIRECTIVE,
    STRATEGY_COORDINATOR_BUILD_DIRECTIVE,
    STRATEGY_COORDINATOR_COORDINATION,
    STRATEGY_COORDINATOR_INITIAL,
    STRATEGY_COORDINATOR_STRATEGY,
    TRIGGER_ARMY_PLANNER_POLL,
    TRIGGER_MACRO_PLANNER_POLL,
    current_agent_role_for_trigger,
    interaction_get,
    interaction_get_dict,
    interaction_get_list,
    interaction_get_str,
    is_current_commander_interaction,
    normalize_trigger_reason,
)


def _compact_completed(completed: dict) -> str:
    """将 completed 建筑 dict 压缩为简短描述，按数量降序。"""
    if not isinstance(completed, dict) or not completed:
        return ""
    items = sorted(completed.items(), key=lambda kv: (-_sortable_count(kv[1]), kv[0]))
    parts = [f"{k}={v}" for k, v in items]
    return " | ".join(parts)


def _compact_enemy(enemy_composition: dict) -> str:
    """将敌方单位压缩。"""
    if not isinstance(enemy_composition, dict) or not enemy_composition:
        return "(unknown)"
    items = sorted(enemy_composition.items(), key=lambda kv: (-_sortable_count(kv[1]), kv[0]))
    parts = [f"{k}={v}" for k, v in items]
    return ", ".join(parts)

def _compact_contents(contents: Any) -> str:
    """Render one zone-content cell without using the table delimiter."""
    data = _as_dict(contents)
    if not data:
        return "none"
    items = sorted(data.items(), key=lambda kv: (-_sortable_count(kv[1]), kv[0]))
    return ",".join(f"{key}={value}" for key, value in items)



def _sortable_count(value: Any) -> float:
    """Return a sortable numeric count without trusting JSON value types."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_dict(value: Any) -> dict:
    """Treat missing or malformed JSON objects as empty objects."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    """Treat missing or malformed JSON arrays as empty arrays."""
    return value if isinstance(value, list) else []


def is_strategy_coordination_trigger(trigger: str) -> bool:
    """True for Strategy Coordinator decision steps in the current runtime."""
    return (
        trigger.startswith("strategy_coordination")
        or trigger.startswith("top_agent_poll")
        or trigger.endswith("_coordination_poll")
    )


def agent_role_for_trigger(trigger: str, *, has_initial: bool = False) -> str:
    """Compact agent role used by the chunk catalog."""
    if has_initial or "initial" in trigger:
        return "init"
    if is_strategy_coordination_trigger(trigger) or (
        trigger.startswith("top_agent") and "initial" not in trigger
    ):
        return "coordinator"
    if trigger in {TRIGGER_MACRO_PLANNER_POLL, "mid_agent_poll", "poll"}:
        return "macro"
    if trigger == TRIGGER_ARMY_PLANNER_POLL:
        return "army"
    return "-"


def _compact_actions(translations: Any) -> str:
    """Compact down-agent translations into issued action summaries."""
    parts = []
    for item in _as_list(translations):
        parsed = _as_dict(item.get("parsed")) if isinstance(item, dict) else {}
        if parsed:
            action = parsed.get("action", "?")
            target = parsed.get("to_count")
            parts.append(f"{action}->{target}" if target is not None else str(action))
        elif isinstance(item, dict) and item.get("raw"):
            parts.append(str(item["raw"]))
        else:
            parts.append(str(item))
    return " | ".join(parts)


def _compact_policy_execution(execution: Any) -> str:
    army = _as_dict(execution)
    policy = _as_dict(army.get("last_policy"))
    if not policy:
        return ""
    parts = []
    for raw in _as_list(policy.get("commands")):
        command = _as_dict(raw)
        parts.append(
            f"{command.get('group_id', '?')} "
            f"{command.get('movement_mode', '?')} "
            f"{command.get('destination_zone_id', '?')}"
        )
    parts.append(f"scan {policy.get('scan_zone_id') or 'none'}")
    parts.append(f"scout {policy.get('scout_zone_id') or 'none'}")
    issues = _as_list(army.get("last_command_issues"))
    return (
        f"execution_status={army.get('status', 'unknown')}; current_policy={' | '.join(parts)}; "
        f"policy_age={army.get('policy_age_seconds')}s; "
        f"last_command_issues={' | '.join(str(item) for item in issues) if issues else 'none'}"
    )


def _compact_mid_execution(execution: Any) -> str:
    mid = _as_dict(execution)
    if not mid:
        return ""
    tasks = " | ".join(str(item) for item in _as_list(mid.get("last_tasks"))) or "none"
    actions = " | ".join(
        f"{item.get('action', '?')}->{item.get('to_count', '?')}"
        for item in (_as_dict(raw) for raw in _as_list(mid.get("active_macro_tasks")))
    ) or "none"
    issues = _as_list(mid.get("last_issues") or mid.get("last_translation_issues"))
    return (
        f"execution_status={mid.get('status', 'unknown')}; last_tasks={tasks}; active_macro_tasks={actions}; "
        f"last_update={mid.get('last_update_seconds_ago')}s; "
        f"last_issues={' | '.join(str(item) for item in issues) if issues else 'none'}"
    )

def _compact_reasoning(reasoning: Any) -> str:
    """Keep narrative reasoning while omitting the duplicated JSON task block."""
    text = str(reasoning or "").strip()
    fence_pos = text.find("```")
    if fence_pos >= 0:
        text = text[:fence_pos].rstrip()
    else:
        task_json = re.search(
            r"(?:^|\n\s*\n)\s*(?:\[\s*)?\{\s*[\r\n ]*\"tasks\"\s*:",
            text,
        )
        if task_json:
            text = text[:task_json.start()].rstrip()
    return text


def _compact_army_decision(interaction: dict) -> dict:
    """Extract the Army Planner result without copying its large prompt payload."""
    if normalize_trigger_reason(interaction.get("trigger_reason")) != TRIGGER_ARMY_PLANNER_POLL:
        return {}

    policy = _as_dict(interaction.get("army_control_agent_policy"))
    parsed = _as_dict(policy.get("parsed"))
    commands = []
    for raw_command in _as_list(parsed.get("commands")):
        command = _as_dict(raw_command)
        group_id = command.get("group_id", "?")
        movement_mode = command.get("movement_mode", "?")
        zone_id = command.get("destination_zone_id", "?")
        rendered = f"{group_id}:{movement_mode}->{zone_id}"
        focus = _as_list(command.get("focus_target_types"))
        if focus:
            rendered += f"; focus={','.join(str(item) for item in focus)}"
        commands.append(rendered)

    issues = [str(item) for item in _as_list(policy.get("command_issues"))]
    error = policy.get("error")
    directive = interaction_get_str(
        interaction, STRATEGY_COORDINATOR_ARMY_DIRECTIVE
    )

    target_zone_ids = []
    for raw_command in _as_list(parsed.get("commands")):
        zone_id = _as_dict(raw_command).get("destination_zone_id")
        if zone_id and zone_id not in target_zone_ids:
            target_zone_ids.append(zone_id)
    for zone_id in (parsed.get("scan_zone_id"), parsed.get("scout_zone_id")):
        if zone_id and zone_id not in target_zone_ids:
            target_zone_ids.append(zone_id)

    observation = _as_dict(interaction.get("observation_view"))
    observation = _as_dict(observation.get("army_control")) or observation
    zones_by_id = {
        zone.get("zone_id"): zone
        for zone in (
            _as_dict(item)
            for item in _as_list(observation.get("zones"))
        )
        if zone.get("zone_id")
    }
    target_zones = [
        zones_by_id[zone_id]
        for zone_id in target_zone_ids
        if zone_id in zones_by_id
    ]
    signature = json.dumps(parsed, sort_keys=True, ensure_ascii=False) if parsed else ""
    return {
        "directive": directive,
        "commands": commands,
        "scan_zone_id": parsed.get("scan_zone_id"),
        "scout_zone_id": parsed.get("scout_zone_id"),
        "target_zones": target_zones,
        "issues": issues,
        "error": str(error) if error else "",
        "signature": signature,
    }


def _execution_history_lines(execution: Any) -> list[str]:
    """Render sub-agent decisions collected since the previous coordination decision."""
    history = _as_dict(execution)
    mid_events = _as_list(history.get("mid"))
    army_events = _as_list(history.get("army"))
    if not mid_events and not army_events:
        return []
    lines = [
        "[Execution Window] "
        f"since={history.get('window_start_game_time_seconds', 'unknown')}s; "
        f"mid_decisions={len(mid_events)}; army_decisions={len(army_events)}"
    ]
    for raw in mid_events:
        event = _as_dict(raw)
        tasks = " | ".join(str(item) for item in _as_list(event.get("tasks"))) or "none"
        issues = " | ".join(str(item) for item in _as_list(event.get("issues"))) or "none"
        lines.append(
            "[Macro Planner Decision In Window] "
            f"time={event.get('game_time_seconds')}s; status={event.get('status', 'unknown')}; "
            f"tasks={tasks}; issues={issues}"
        )
    for raw in army_events:
        event = _as_dict(raw)
        commands = []
        for raw_command in _as_list(event.get("commands")):
            command = _as_dict(raw_command)
            commands.append(
                f"{command.get('group_id', '?')}:{command.get('movement_mode', '?')}"
                f"->{command.get('destination_zone_id', '?')}"
            )
        issues = " | ".join(str(item) for item in _as_list(event.get("issues"))) or "none"
        lines.append(
            "[Army Decision In Window] "
            f"time={event.get('game_time_seconds')}s; status={event.get('status', 'unknown')}; "
            f"applied={event.get('applied')}; commands={' | '.join(commands) or 'none'}; "
            f"scan={event.get('scan_zone_id') or 'none'}; "
            f"scout={event.get('scout_zone_id') or 'none'}; issues={issues}"
        )
    return lines


def _current_army_decision(interaction: dict, observation: dict) -> dict:
    policy = _as_dict(interaction.get("army_policy"))
    intent = _as_dict(policy.get("intent"))
    commands = []
    target_zone_ids = []
    for raw_command in _as_list(policy.get("commands")):
        command = _as_dict(raw_command)
        zone_id = command.get("destination_zone_id")
        rendered = (
            f"{command.get('group_id', '?')}:"
            f"{command.get('movement_mode', '?')}->{zone_id or '?'}"
        )
        if command.get("retreat_ratio") is not None:
            rendered += f"; retreat_ratio={command.get('retreat_ratio')}"
        commands.append(rendered)
        if zone_id and zone_id not in target_zone_ids:
            target_zone_ids.append(zone_id)
    for zone_id in (policy.get("scan_zone_id"), policy.get("scout_zone_id")):
        if zone_id and zone_id not in target_zone_ids:
            target_zone_ids.append(zone_id)

    control = _as_dict(observation.get("army_control"))
    zones_by_id = {
        zone.get("zone_id"): zone
        for zone in (_as_dict(item) for item in _as_list(control.get("zones")))
        if zone.get("zone_id")
    }
    issues = [
        str(item)
        for item in (
            _as_list(interaction.get("issues"))
            + _as_list(interaction.get("reflection_issues"))
        )
    ]
    return {
        "intent": intent,
        "directive": "",
        "commands": commands,
        "scan_zone_id": policy.get("scan_zone_id"),
        "scout_zone_id": policy.get("scout_zone_id"),
        "target_zones": [
            zones_by_id[zone_id]
            for zone_id in target_zone_ids
            if zone_id in zones_by_id
        ],
        "target_zone_ids": target_zone_ids,
        "issues": list(dict.fromkeys(issues)),
        "error": str(interaction.get("error") or ""),
        "signature": json.dumps(policy, sort_keys=True, ensure_ascii=False),
    }
def _current_decision_lines(
    interaction: dict,
    *,
    reasoning_source: Any,
    macro_tasks: list,
    tool_call_summary: str,
    army: dict,
) -> list[str]:
    lines: list[str] = []
    reasoning = _compact_reasoning(reasoning_source)
    if reasoning:
        lines.append(f"[Commander Reasoning] {reasoning}")
    if macro_tasks:
        lines.append(
            "[Commander Macro Targets] "
            + " | ".join(
                f"{item.get('action', '?')}->{item.get('to_count', '?')}"
                for item in (_as_dict(raw) for raw in macro_tasks)
            )
        )
    if tool_call_summary:
        lines.append(f"[Commander Tool Calls] {tool_call_summary}")
    if army.get("commands"):
        lines.append("[Commander Army Commands] " + " | ".join(army["commands"]))
    if army.get("intent"):
        lines.append(
            "[Commander Army Intent] "
            f"{army['intent'].get('mode', '?')}->{army['intent'].get('zone_id', '?')}"
        )
    if army.get("scan_zone_id"):
        lines.append(f"[Commander Scan] {army['scan_zone_id']}")
    if army.get("scout_zone_id"):
        lines.append(f"[Commander Scout] {army['scout_zone_id']}")
    wake = _as_dict(interaction.get("wake_event"))
    if wake:
        lines.append(f"[Wake Event] {json.dumps(wake, ensure_ascii=False)}")
    if army.get("issues"):
        lines.append("[Decision Issues] " + " | ".join(army["issues"]))
    if interaction.get("accepted") is False:
        lines.append("[Decision Rejected]")
    if interaction.get("error"):
        lines.append(f"[Decision Error] {str(interaction.get('error'))}")
    return lines


def _extract_current_commander_chunk(interaction: dict) -> dict:
    """Convert one current single-Commander interaction into optimizer evidence."""
    try:
        game_time = float(interaction.get("game_time", 0) or 0)
    except (TypeError, ValueError):
        game_time = 0.0
    trigger = str(interaction.get("trigger_reason") or "unknown")
    strategy = str(
        interaction.get("strategy_id")
        or interaction.get("forced_strategy")
        or ""
    )
    obs = _as_dict(interaction.get("observation"))
    economy = _as_dict(obs.get("economy"))
    own = _as_dict(obs.get("own_forces"))
    production = _as_dict(obs.get("production"))
    technology = _as_dict(obs.get("technology"))
    enemy = _as_dict(obs.get("enemy"))
    map_control = _as_dict(obs.get("map_control"))
    combat = _as_dict(obs.get("combat"))
    threat_flags = _as_dict(obs.get("threat_flags"))
    execution = _as_dict(obs.get("execution"))
    army = _current_army_decision(interaction, obs)

    supply_used = economy.get("supply_used", own.get("supply_used"))
    supply_cap = economy.get("supply_cap", own.get("supply_cap"))
    supply_free = economy.get("supply_free", own.get("supply_free"))
    state = {
        "game_time": game_time,
        "trigger": trigger,
        "view_type": "commander" if obs else "none",
        "phase": "",
        "strategy": strategy,
        "workers": economy.get("workers"),
        "ideal_workers": economy.get("ideal_workers"),
        "army_supply": own.get("army_supply"),
        "supply": (
            f"{supply_used}/{supply_cap}"
            if supply_used is not None or supply_cap is not None
            else None
        ),
        "supply_left": supply_free,
        "minerals": economy.get("minerals"),
        "vespene": economy.get("vespene"),
        "income_min": economy.get("mineral_income"),
        "income_gas": economy.get("vespene_income"),
        "own_bases": map_control.get("own_base_count", economy.get("own_base_count")),
        "active_mining_bases": map_control.get("active_mining_base_count"),
        "enemy_bases_known": map_control.get("known_enemy_base_count"),
        "neutral_expansions": map_control.get("neutral_expansion_count"),
        "power_own": combat.get("our_army_power"),
        "power_enemy": combat.get("enemy_army_power"),
        "visible_enemy_power": combat.get("visible_enemy_army_power"),
        "combat_advantage": combat.get("advantage_predicted"),
        "army_advantage": combat.get("army_advantage"),
        "income_advantage": combat.get("income_advantage"),
        "army_control_advantage": combat.get("army_control_advantage"),
        "own_lost_minerals": combat.get("own_lost_minerals"),
        "own_lost_gas": combat.get("own_lost_gas"),
        "enemy_lost_minerals": combat.get("enemy_lost_minerals"),
        "enemy_lost_gas": combat.get("enemy_lost_gas"),
        "enemy_air": combat.get("enemy_air"),
        "enemy_cloak": threat_flags.get("cloak_or_burrow_threat"),
        "enemy_proxy": threat_flags.get("has_proxy_buildings"),
        "enemy_rushing": threat_flags.get("is_rushing"),
        "enemy_rush_build": threat_flags.get("rush_build"),
        "enemy_seen_seconds_ago": enemy.get("seconds_since_last_seen"),
        "upgrades": _as_list(technology.get("completed_upgrades")),
        "completed_buildings": _compact_completed(production.get("completed", {})),
        "under_construction": _compact_completed(production.get("under_construction", {})),
        "active_queues": _compact_completed(production.get("active_queues", {})),
        "workers_en_route": _compact_completed(production.get("workers_en_route", {})),
        "enemy_visible": _compact_enemy(enemy.get("visible_composition", {})),
        "enemy_known": _compact_enemy(
            enemy.get("known_combat_composition")
            or enemy.get("known_composition", {})
        ),
    }

    macro_tasks = _as_list(interaction.get("macro_tasks"))
    tool_calls = _as_list(interaction.get("tool_calls"))
    issued = " | ".join(
        f"{item.get('name', '?')}({json.dumps(item.get('arguments') or {}, ensure_ascii=False)})"
        for item in (_as_dict(raw) for raw in tool_calls)
    )
    reasoning_source = (
        interaction.get("assistant_content") or interaction.get("content") or ""
    )
    reasoning = _compact_reasoning(reasoning_source)
    tool_selection = {}
    if trigger == "strategy_tool_selection":
        tool_selection = {
            "selected_tools": _as_list(interaction.get("selected_tools")),
            "semantic_tools": _as_list(interaction.get("semantic_tools")),
            "dependency_tools": _as_list(interaction.get("dependency_tools")),
            "baseline_tools": _as_list(interaction.get("baseline_tools")),
            "selected_tool_count": interaction.get("selected_tool_count"),
            "full_tool_count": interaction.get("full_tool_count"),
            "fallback_used": interaction.get("fallback_used"),
            "fallback_reason": interaction.get("fallback_reason"),
            "dependency_error": interaction.get("dependency_error"),
        }
    decision = {
        "reasoning": reasoning,
        "macro_tasks": macro_tasks,
        "tool_calls": tool_calls,
        "tool_call_summary": issued,
        "army": army,
        "wake_event": _as_dict(interaction.get("wake_event")),
        "accepted": interaction.get("accepted"),
        "issues": list(
            dict.fromkeys(
                str(item)
                for item in (
                    _as_list(interaction.get("issues"))
                    + _as_list(interaction.get("reflection_issues"))
                )
            )
        ),
        "execution": execution,
        "tool_selection": tool_selection,
        "error": interaction.get("error"),
    }

    initial = {}
    if trigger == "strategy_forced":
        initial = {
            "race": "",
            "instruct": "",
            "forced_strategy": interaction.get("forced_strategy"),
            "selected_strategy": interaction.get("strategy_id"),
            "strategy_description": interaction.get("strategy_description", ""),
        }

    lines = [f"=== Step {game_time:.0f}s [{trigger}] ==="]
    if initial:
        lines.append(f"[Strategy] {strategy or 'unknown'}")
    if trigger == "strategy_tool_selection":
        lines.append(
            "[Selected Tools] "
            + ", ".join(str(item) for item in _as_list(interaction.get("selected_tools")))
        )
        lines.append(
            "[Tool Selection Sources] "
            f"semantic={','.join(map(str, tool_selection['semantic_tools'])) or 'none'}; "
            f"dependencies={','.join(map(str, tool_selection['dependency_tools'])) or 'none'}; "
            f"baseline={','.join(map(str, tool_selection['baseline_tools'])) or 'none'}; "
            f"selected={tool_selection['selected_tool_count']}/"
            f"{tool_selection['full_tool_count']}; "
            f"fallback_used={tool_selection['fallback_used']}"
        )
        if tool_selection["fallback_reason"] or tool_selection["dependency_error"]:
            lines.append(
                "[Tool Selection Issues] "
                f"fallback_reason={tool_selection['fallback_reason'] or 'none'}; "
                f"dependency_error={tool_selection['dependency_error'] or 'none'}"
            )
    if obs:
        # Prefer the exact observation supplied to Commander.  This prevents
        # EvolAgent from drifting when the live observation renderer changes.
        observation_text = str(interaction.get("text_observation") or "").strip()
        if not observation_text:
            observation_text = render_observation(obs).strip()
        lines.extend(["[Commander Observation]", observation_text])
    lines.extend(
        _current_decision_lines(
            interaction,
            reasoning_source=reasoning_source,
            macro_tasks=macro_tasks,
            tool_call_summary=issued,
            army=army,
        )
    )
    agent_role = current_agent_role_for_trigger(trigger)
    if agent_role == "-" and interaction.get("agent") == "commander":
        agent_role = "commander"
    return {
        "game_time": game_time,
        "trigger": trigger,
        "raw_trigger": trigger,
        "view_type": "commander" if obs else "none",
        "phase": "",
        "agent_role": agent_role,
        "state": state,
        "decision": decision,
        "high_level": {},
        "initial": initial,
        "strategic_signature": strategy,
        "mid_signature": json.dumps(macro_tasks, sort_keys=True, default=str),
        "army_signature": army["signature"],
        "enemy_signature": json.dumps(
            [state["enemy_visible"], state["enemy_known"], state["enemy_bases_known"]],
            sort_keys=True,
            default=str,
        ),
        "army_observation": obs,
        "text": "\n".join(lines),
    }


def extract_interaction_chunk(interaction: dict) -> dict:
    """Convert one recorded interaction into compact optimizer evidence."""
    if is_current_commander_interaction(interaction):
        return _extract_current_commander_chunk(interaction)
    try:
        game_time = float(interaction.get("game_time", 0) or 0)
    except (TypeError, ValueError):
        game_time = 0.0
    raw_trigger = str(interaction.get("trigger_reason") or "unknown")
    trigger = normalize_trigger_reason(raw_trigger)
    # Legacy field retained for old records; current runtime no longer emits phase.
    phase = str(interaction.get("top_agent_phase") or "")
    strategy = interaction_get_str(interaction, STRATEGY_COORDINATOR_STRATEGY)

    obs = _as_dict(interaction.get("observation_view"))
    if obs and obs.get("schema_version") != "2.0":
        obs = {}
    view_type = str(
        interaction.get("observation_view_type")
        or obs.get("view_type")
        or "none"
    )

    economy = _as_dict(obs.get("economy"))
    own = _as_dict(obs.get("own_forces"))
    production = _as_dict(obs.get("production"))
    readiness = _as_dict(obs.get("military_readiness"))
    if not production and readiness:
        production = {
            "completed": (
                _as_dict(readiness.get("completed_units_and_structures"))
                or _as_dict(readiness.get("completed"))
            ),
            "under_construction": _as_dict(readiness.get("under_construction")),
            "active_queues": {},
            "workers_en_route": {},
        }
    technology = _as_dict(obs.get("technology")) or _as_dict(readiness.get("technology"))
    enemy = _as_dict(obs.get("enemy"))
    map_control = _as_dict(obs.get("map_control"))
    combat = _as_dict(obs.get("combat"))
    threat_flags = _as_dict(obs.get("threat_flags"))
    execution = _as_dict(obs.get("execution"))
    mid_execution = _as_dict(execution.get("mid"))
    army_execution = _as_dict(execution.get("army"))
    execution_history = _as_dict(execution.get("since_last_top_decision"))

    supply_used = economy.get("supply_used", own.get("supply_used"))
    supply_cap = economy.get("supply_cap", own.get("supply_cap"))
    supply_free = economy.get("supply_free", own.get("supply_free"))
    supply_text = (
        f"{supply_used}/{supply_cap}"
        if supply_used is not None or supply_cap is not None
        else None
    )
    state = {
        "game_time": game_time,
        "trigger": trigger,
        "view_type": view_type,
        "phase": phase,
        "strategy": strategy,
        "workers": economy.get("workers"),
        "ideal_workers": economy.get("ideal_workers"),
        "army_supply": own.get("army_supply"),
        "supply": supply_text,
        "supply_left": supply_free,
        "minerals": economy.get("minerals"),
        "vespene": economy.get("vespene"),
        "income_min": economy.get("mineral_income"),
        "income_gas": economy.get("vespene_income"),
        "own_bases": map_control.get("own_base_count"),
        "enemy_bases_known": map_control.get("known_enemy_base_count"),
        "neutral_expansions": map_control.get("neutral_expansion_count"),
        "power_own": combat.get("our_army_power"),
        "power_enemy": combat.get("enemy_army_power"),
        "visible_enemy_power": combat.get("visible_enemy_army_power"),
        "combat_advantage": combat.get("advantage_predicted"),
        "army_advantage": combat.get("army_advantage"),
        "income_advantage": combat.get("income_advantage"),
        "army_control_advantage": combat.get("army_control_advantage"),
        "own_lost_minerals": combat.get("own_lost_minerals"),
        "own_lost_gas": combat.get("own_lost_gas"),
        "enemy_lost_minerals": combat.get("enemy_lost_minerals"),
        "enemy_lost_gas": combat.get("enemy_lost_gas"),
        "enemy_air": combat.get("enemy_air"),
        "enemy_cloak": threat_flags.get("cloak_or_burrow_threat"),
        "enemy_proxy": threat_flags.get("has_proxy_buildings"),
        "enemy_rushing": threat_flags.get("is_rushing"),
        "enemy_rush_build": threat_flags.get("rush_build"),
        "enemy_seen_seconds_ago": enemy.get("seconds_since_last_seen"),
        "upgrades": _as_list(technology.get("completed_upgrades")),
        "completed_buildings": _compact_completed(production.get("completed", {})),
        "under_construction": _compact_completed(production.get("under_construction", {})),
        "active_queues": _compact_completed(production.get("active_queues", {})),
        "workers_en_route": _compact_completed(production.get("workers_en_route", {})),
        "enemy_visible": _compact_enemy(enemy.get("visible_composition", {})),
        "enemy_known": _compact_enemy(
            enemy.get("known_combat_composition") or enemy.get("known_composition", {})
        ),
    }

    build_directive = interaction_get_str(
        interaction, STRATEGY_COORDINATOR_BUILD_DIRECTIVE
    )
    army_directive = interaction_get_str(
        interaction, STRATEGY_COORDINATOR_ARMY_DIRECTIVE
    )
    decision = {
        "top_focus": interaction.get("top_agent_focus", ""),
        "top_build_directive": build_directive,
        "top_army_directive": army_directive,
        "prev_tasks": interaction_get_list(
            interaction, MACRO_PLANNER_INPUT_PREVIOUS_TASKS
        ),
        "reasoning": _compact_reasoning(
            interaction_get(interaction, MACRO_PLANNER_RAW_RESPONSE, default="")
        ),
        "new_tasks": (
            interaction_get_list(interaction, MACRO_PLANNER_OUTPUT_NEW_TASKS)
            or _as_list(mid_execution.get("last_tasks"))
        ),
        "active_tasks": (
            _as_list(interaction.get("active_tasks_after_refresh"))
            or _as_list(mid_execution.get("active_macro_tasks"))
        ),
        "actions_issued": _compact_actions(
            interaction_get_list(interaction, ACTION_TRANSLATOR_TRANSLATIONS)
        ),
        "army": _compact_army_decision(interaction),
        "mid_execution": _compact_mid_execution(mid_execution),
        "army_execution": _compact_policy_execution(army_execution),
        "error": interaction.get("error"),
    }

    coordination_record = interaction_get_dict(
        interaction, STRATEGY_COORDINATOR_COORDINATION
    )
    parsed_assessment = _as_dict(coordination_record.get("parsed"))
    high_level = {}
    if parsed_assessment:
        high_level = {
            "assessed_build_directive": (
                parsed_assessment.get("build_directive")
                or parsed_assessment.get("focus")
            ),
            "assessed_army_directive": (
                parsed_assessment.get("army_directive")
                or army_directive
            ),
        }

    top_initial = interaction_get_dict(interaction, STRATEGY_COORDINATOR_INITIAL)
    initial = {}
    if top_initial:
        initial = {
            "race": top_initial.get("race"),
            "instruct": top_initial.get("instruct"),
            "forced_strategy": top_initial.get("forced_strategy"),
            "selected_strategy": top_initial.get("selected_strategy"),
            "strategy_description": top_initial.get("strategy_description", ""),
        }

    def value(raw: Any, fmt: str = "") -> str:
        if raw is None:
            return "unknown"
        try:
            return format(raw, fmt) if fmt else str(raw)
        except (ValueError, TypeError):
            return str(raw)

    lines = [f"=== Step {game_time:.0f}s [{trigger}] ==="]
    if initial:
        initial_strategy = initial.get("forced_strategy") or initial.get("selected_strategy") or "unknown"
        lines.append(
            f"Initial strategy: {initial_strategy} | "
            f"{str(initial.get('instruct') or '')}"
        )
    if not obs:
        if not initial:
            lines.append("(no schema 2.0 observation_view)")
    else:
        lines.append(f"[Observation View] {view_type}")
        if economy:
            lines.append(
                f"Economy: {value(state['workers'])}/{value(state['ideal_workers'])} workers, "
                f"{value(state['minerals'])}$/{value(state['vespene'])}g "
                f"({value(state['income_min'], '.0f')}$/min, "
                f"{value(state['income_gas'], '.0f')}g/min) "
                f"Supply: {value(state['supply'])} ({value(state['supply_left'])} free)"
            )
        elif supply_text is not None:
            lines.append(f"Supply: {value(state['supply'])} ({value(state['supply_left'])} free)")

        if own:
            own_parts = [f"army_supply={value(state['army_supply'])}"]
            if state["power_own"] is not None:
                own_parts.append(f"global_power={value(state['power_own'], '.2f')}")
            lines.append("Military: " + "; ".join(own_parts))
        if combat:
            combat_parts = []
            for label, key in (
                ("predicted", "combat_advantage"),
                ("army_adv", "army_advantage"),
                ("income_adv", "income_advantage"),
                ("army_control_adv", "army_control_advantage"),
            ):
                if state[key] is not None:
                    combat_parts.append(f"{label}={state[key]}")
            if state["power_enemy"] is not None:
                combat_parts.append(f"enemy_power={value(state['power_enemy'], '.2f')}")
            if state["visible_enemy_power"] is not None:
                combat_parts.append(f"visible_enemy_power={value(state['visible_enemy_power'], '.2f')}")
            if combat_parts:
                lines.append("Combat: " + "; ".join(combat_parts))
        if map_control:
            lines.append(
                f"Bases: own={value(state['own_bases'])}, "
                f"enemy_known={value(state['enemy_bases_known'])}, "
                f"neutral={value(state['neutral_expansions'])}"
            )
        if state["completed_buildings"]:
            lines.append(f"Completed: {state['completed_buildings']}")
        infrastructure = []
        if state["under_construction"]:
            infrastructure.append(f"building={state['under_construction']}")
        if state["active_queues"]:
            infrastructure.append(f"queues={state['active_queues']}")
        if state["workers_en_route"]:
            infrastructure.append(f"en_route={state['workers_en_route']}")
        if infrastructure:
            lines.append("Infrastructure: " + " | ".join(infrastructure))
        if state["enemy_visible"] != "(unknown)":
            lines.append(f"Enemy visible now: {state['enemy_visible']}")
        if state["enemy_known"] != "(unknown)":
            lines.append(f"Enemy known or remembered: {state['enemy_known']}")
        if state["enemy_seen_seconds_ago"] is not None:
            lines.append(f"Last enemy sighting: {value(state['enemy_seen_seconds_ago'], '.1f')}s ago")

        losses = (
            state["own_lost_minerals"], state["own_lost_gas"],
            state["enemy_lost_minerals"], state["enemy_lost_gas"],
        )
        if any(item not in (None, 0, 0.0) for item in losses):
            lines.append(
                "Losses: "
                f"own={value(losses[0])}min/{value(losses[1])}gas; "
                f"enemy={value(losses[2])}min/{value(losses[3])}gas"
            )
        if state["upgrades"]:
            lines.append("Completed upgrades: " + ", ".join(str(item) for item in state["upgrades"]))
        threats = []
        if state["enemy_cloak"]:
            threats.append("cloak_or_burrow")
        if state["enemy_proxy"]:
            threats.append("proxy_buildings")
        if state["enemy_rushing"]:
            threats.append(f"rush={state['enemy_rush_build'] or 'detected'}")
        if state["enemy_air"] not in (None, "NoAir"):
            threats.append(f"air={state['enemy_air']}")
        if threats:
            lines.append("Threats: " + " | ".join(threats))

    if strategy and (
        is_strategy_coordination_trigger(trigger) or trigger.startswith("top_agent")
    ):
        lines.append(f"[Strategy Coordinator Strategy] {strategy}")
    if phase:
        lines.append(f"[Strategy Coordinator Phase] {phase}")
    if trigger == "army_control_agent_poll":
        lines.append(
            "[Strategy Coordinator Army Directive] "
            + (str(decision["top_army_directive"]) or "none")
        )
    if high_level:
        if high_level.get("assessed_build_directive"):
            lines.append(
                "[Strategy Coordinator Build Directive] "
                f"{str(high_level['assessed_build_directive'])}"
            )
        if high_level.get("assessed_army_directive"):
            lines.append(
                "[Strategy Coordinator Army Directive] "
                f"{str(high_level['assessed_army_directive'])}"
            )
    elif is_strategy_coordination_trigger(trigger) or (
        trigger != "army_control_agent_poll" and decision["top_build_directive"]
    ):
        if decision["top_build_directive"]:
            lines.append(
                "[Strategy Coordinator Build Directive] "
                f"{str(decision['top_build_directive'])}"
            )
        if decision["top_army_directive"] and trigger != "army_control_agent_poll":
            lines.append(
                "[Strategy Coordinator Army Directive] "
                f"{str(decision['top_army_directive'])}"
            )

    if decision["reasoning"]:
        lines.append(f"[Macro Planner Reasoning] {decision['reasoning']}")
    if decision["prev_tasks"]:
        lines.append("[Macro Planner Previous Tasks] " + " | ".join(str(item) for item in decision["prev_tasks"]))
    if decision["new_tasks"]:
        lines.append("[Macro Planner New Tasks] " + " | ".join(str(item) for item in decision["new_tasks"]))
    active_text = ""
    if decision["active_tasks"]:
        active_text = " | ".join(
            f"{item.get('action', '?')}->{item.get('to_count', '?')}"
            if isinstance(item, dict) else str(item)
            for item in decision["active_tasks"]
        )
        lines.append(f"[Action Translator Active Commands] {active_text}")
    if decision["actions_issued"] and decision["actions_issued"] != active_text:
        lines.append(f"[Action Translator Actions Issued] {decision['actions_issued']}")
    if decision["mid_execution"]:
        lines.append(f"[Macro Planner Execution] {decision['mid_execution']}")
    if decision["army_execution"]:
        lines.append(f"[Army Execution] {decision['army_execution']}")
    lines.extend(_execution_history_lines(execution_history))

    army = decision["army"]
    if army:
        army_control = _as_dict(obs.get("army_control"))
        capabilities = _as_dict(obs.get("capabilities"))
        scan_state = _as_dict(capabilities.get("scan"))
        scout_state = _as_dict(capabilities.get("scv_scout"))
        lines.append(
            "[Army Status] "
            f"controlled={army_control.get('controlled_combat_units')}; "
            f"nearest_zone={army_control.get('army_nearest_zone')}; "
            f"threatened_zones={army_control.get('threatened_zone_ids') or []}"
        )
        lines.append(
            "[Army Orbital] "
            f"count={scan_state.get('orbital_count')}; energies={scan_state.get('orbital_energies') or []}; "
            f"scan_ready={scan_state.get('available_scan_count')}; "
            f"last_zone={scan_state.get('last_target_zone_id')}; "
            f"last_result={scan_state.get('last_result')}; seconds_ago={scan_state.get('last_result_seconds_ago')}"
        )
        lines.append(
            "[Army SCV Status] "
            f"workers={scout_state.get('worker_count')}; active={scout_state.get('active')}; "
            f"active_zone={scout_state.get('active_target_zone_id')}; "
            f"last_zone={scout_state.get('last_target_zone_id')}; "
            f"last_result={scout_state.get('last_result')}; seconds_ago={scout_state.get('last_result_seconds_ago')}"
        )
        lines.append(
            "[Army Unit Types] "
            f"own={_compact_completed(own.get('combat_composition', {})) or 'none'}; "
            f"visible_enemy={_compact_completed(enemy.get('visible_composition', {})) or 'none'}; "
            f"known_enemy={_compact_completed(enemy.get('known_combat_composition', {})) or 'none'}"
        )
        for raw_group in _as_list(army_control.get("groups")):
            group = _as_dict(raw_group)
            command = _as_dict(group.get("current_command"))
            command_text = (
                f"{command.get('movement_mode', 'none')}->{command.get('destination_zone_id', 'none')}"
                if command else "none"
            )
            lines.append(
                "[Army Group] "
                f"id={group.get('group_id')}; role={group.get('role')}; units={group.get('unit_count')}; "
                f"power={group.get('power')}; nearest_zone={group.get('nearest_zone_id')}; "
                f"types={_compact_completed(group.get('unit_type_counts', {})) or 'none'}; "
                f"nearby_enemy_count={group.get('nearby_enemy_count')}; "
                f"nearby_enemy_power={group.get('nearby_enemy_power')}; fragmented={group.get('is_fragmented')}; "
                f"command={command_text}; command_age={group.get('command_age_seconds')}s; "
                f"search_target={group.get('search_target_zone_id') or 'none'}; "
                f"searched_zones={group.get('searched_zone_ids') or []}"
            )

        selected_zone_ids = {
            zone.get("zone_id") for zone in army.get("target_zones", []) if zone.get("zone_id")
        }
        zones = _as_list(army_control.get("zones"))
        if zones:
            lines.append(
                "[Army Zones] columns="
                "id|selected|owner|role|vision|distance_from_army|distance_to_own_main|"
                "distance_to_enemy_main|own_contents|visible_enemy_contents|last_seen_enemy_contents|"
                "enemy_information_age_seconds|combat_power_balance|under_attack; "
                "bool=1/0; missing=-"
            )
        def cell(raw: Any) -> str:
            if raw is None:
                return "-"
            if isinstance(raw, bool):
                return "1" if raw else "0"
            return str(raw)
        for raw_zone in zones:
            zone = _as_dict(raw_zone)
            zone_id = zone.get("zone_id")
            values = [
                zone_id, zone_id in selected_zone_ids, zone.get("owner"), zone.get("zone_role"),
                zone.get("vision_state"), zone.get("distance_from_army"),
                zone.get("distance_to_own_main"), zone.get("distance_to_enemy_main"),
                _compact_contents(zone.get("own_contents")),
                _compact_contents(zone.get("visible_enemy_contents")),
                _compact_contents(zone.get("last_seen_enemy_contents")),
                zone.get("enemy_information_age_seconds"), zone.get("combat_power_balance"),
                zone.get("under_attack"),
            ]
            lines.append("[Z] " + "|".join(cell(item) for item in values))
        lines.append(f"[Army Commands] {' | '.join(army['commands']) if army['commands'] else 'none'}")
        lines.append(
            f"[Army Scan Command] scan_zone_id={army['scan_zone_id'] or 'none'}"
        )
        lines.append(
            f"[Army Scout Command] scout_zone_id={army['scout_zone_id'] or 'none'}"
        )
        if army["issues"]:
            lines.append(f"[Army Command Issues] {' | '.join(army['issues'])}")
        if army["error"]:
            lines.append(f"[Army Error] {army['error']}")
    if decision["error"]:
        lines.append(f"Error: {str(decision['error'])}")

    strategic_signature = json.dumps(
        [phase, decision["top_build_directive"], decision["top_army_directive"]],
        ensure_ascii=False, sort_keys=True,
    )
    mid_signature = json.dumps(
        [decision["new_tasks"], decision["active_tasks"]],
        ensure_ascii=False, sort_keys=True, default=str,
    )
    enemy_signature = json.dumps(
        [enemy.get("visible_composition"), enemy.get("known_combat_composition"),
         map_control.get("known_enemy_base_count")],
        ensure_ascii=False, sort_keys=True, default=str,
    )
    agent_role = agent_role_for_trigger(trigger, has_initial=bool(initial))
    return {
        "game_time": game_time,
        "trigger": trigger,
        "raw_trigger": raw_trigger,
        "view_type": view_type,
        "phase": phase,
        "agent_role": agent_role,
        "state": state,
        "decision": decision,
        "high_level": high_level,
        "initial": initial,
        "strategic_signature": strategic_signature,
        "mid_signature": mid_signature,
        "army_signature": decision["army"].get("signature", ""),
        "enemy_signature": enemy_signature,
        "army_observation": obs if trigger == "army_control_agent_poll" else {},
        "text": "\n".join(lines),
    }


def format_metadata_text(metadata: dict) -> str:
    """Render match-level conditions needed to compare optimization evidence."""
    data = metadata or {}
    parts = [
        f"Map: {data.get('map_name', '?')}",
        f"Matchup: {data.get('matchup', '?')}",
        f"Own race: {data.get('my_race', '?')}",
        f"Enemy race: {data.get('enemy_race', '?')}",
        f"Duration: {data.get('game_duration_formatted', '?')}",
        f"Result: {data.get('result', '?')}",
        f"Opponent: {data.get('opponent_id', '?')}",
    ]
    if data.get("interval_seconds") is not None:
        parts.append(f"Record interval: {data.get('interval_seconds')}s")
    return " | ".join(parts)


def extract_chunks(data: dict) -> dict:
    """将整局对战记录的 interactions 提取为 chunk 列表。

    Args:
        data: 从 JSON 加载的完整对战记录 dict（含 metadata 和 interactions）

    Returns:
        dict with keys:
          - metadata: 原始 metadata dict
          - metadata_text: metadata 的文本描述（一行）
          - chunks: list[dict]，每个 dict 是 extract_interaction_chunk 的返回值
    """
    metadata = data.get("metadata", {})
    interactions = data.get("interactions", [])
    chunks = []
    for interaction in interactions:
        chunks.append(extract_interaction_chunk(interaction))
    return {
        "metadata": metadata,
        "metadata_text": format_metadata_text(metadata),
        "chunks": chunks,
    }

