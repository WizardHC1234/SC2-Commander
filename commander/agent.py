"""One Commander decision cycle: tool calls → macro tasks + army policy."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from llm.caller import call_openai_detailed, load_agent_pool

from commander.prompts import build_commander_messages
from commander.tools import (
    apply_tool_calls,
    build_commander_tools,
    normalize_tool_calls,
    parse_tool_calls_from_content,
)

logger = logging.getLogger("commander.agent")

_NATIVE_TOOL_CHOICE_ERRORS = (
    "enable-auto-tool-choice",
    "tool-call-parser",
    "tool_choice",
    "tool choice",
)


def _resolve_tool_mode(model_key: str) -> str:
    pool = (load_agent_pool().get("llm_agents_pool") or {}).get(model_key) or {}
    mode = str(pool.get("tool_mode") or "native").strip().lower()
    return mode if mode in {"native", "json"} else "native"


def _is_native_tools_unsupported(error: str) -> bool:
    text = (error or "").lower()
    return any(token in text for token in _NATIVE_TOOL_CHOICE_ERRORS)


def _pack_outcome(
    *,
    tool_calls: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    policy: Any,
    issues: List[str],
    error: str,
    result: Dict[str, Any],
    tool_mode: str,
) -> Dict[str, Any]:
    army_summary = {
        "commands": [
            {
                "group_id": c.group_id,
                "destination_zone_id": c.destination_zone_id,
                "movement_mode": c.movement_mode,
            }
            for c in policy.commands
        ],
        "scan_zone_id": policy.scan_zone_id,
        "scout_zone_id": policy.scout_zone_id,
    }
    return {
        "tool_calls": tool_calls,
        "tasks": tasks,
        "policy": policy,
        "army_summary": army_summary,
        "issues": issues,
        "error": error,
        "latency_seconds": result.get("latency_seconds"),
        "finish_reason": result.get("finish_reason"),
        "content": result.get("content") or "",
        "tool_mode": tool_mode,
    }


def run_commander_decision(
    *,
    race: str,
    strategy_description: str,
    observation_text: str,
    previous_macro_tasks: List[Dict[str, Any]],
    previous_army_summary: Optional[Dict[str, Any]],
    action_space: Dict[str, str],
    model_key: str,
    full_observation: Optional[Dict[str, Any]] = None,
    ensure_addon_parents=None,
) -> Dict[str, Any]:
    """Call the model and return applied macro/army results."""
    tool_mode = _resolve_tool_mode(model_key)
    if tool_mode == "json":
        return _run_json_mode(
            race=race,
            strategy_description=strategy_description,
            observation_text=observation_text,
            previous_macro_tasks=previous_macro_tasks,
            previous_army_summary=previous_army_summary,
            action_space=action_space,
            model_key=model_key,
            full_observation=full_observation,
            ensure_addon_parents=ensure_addon_parents,
        )

    # Native OpenAI tools. Do NOT send tool_choice="auto" — many vLLM servers
    # reject it unless started with --enable-auto-tool-choice.
    messages = build_commander_messages(
        race=race,
        strategy_description=strategy_description,
        observation_text=observation_text,
        previous_macro_tasks=previous_macro_tasks,
        previous_army_summary=previous_army_summary,
        tool_mode="native",
        action_space=action_space,
    )
    tools = build_commander_tools(action_space)
    result = call_openai_detailed(
        messages,
        model_key=model_key,
        tools=tools,
        timeout=120.0,
    )
    error = result.get("error") or ""
    if error and _is_native_tools_unsupported(error):
        logger.warning(
            "Native tools unsupported for %s (%s); falling back to JSON tool_mode.",
            model_key,
            error[:160],
        )
        return _run_json_mode(
            race=race,
            strategy_description=strategy_description,
            observation_text=observation_text,
            previous_macro_tasks=previous_macro_tasks,
            previous_army_summary=previous_army_summary,
            action_space=action_space,
            model_key=model_key,
            full_observation=full_observation,
            ensure_addon_parents=ensure_addon_parents,
        )

    raw_message = result.get("raw_message") or {}
    tool_calls = normalize_tool_calls(raw_message.get("tool_calls"))
    if not tool_calls and (result.get("content") or ""):
        # Some servers put tools in content even when tools= was sent.
        tool_calls = parse_tool_calls_from_content(result.get("content") or "")

    tasks, policy, issues = apply_tool_calls(
        tool_calls,
        legal_action_keys=set(action_space.keys()),
    )
    if full_observation is not None and callable(ensure_addon_parents):
        tasks = ensure_addon_parents(tasks, full_observation)
    return _pack_outcome(
        tool_calls=tool_calls,
        tasks=tasks,
        policy=policy,
        issues=issues,
        error=error,
        result=result,
        tool_mode="native",
    )


def _run_json_mode(
    *,
    race: str,
    strategy_description: str,
    observation_text: str,
    previous_macro_tasks: List[Dict[str, Any]],
    previous_army_summary: Optional[Dict[str, Any]],
    action_space: Dict[str, str],
    model_key: str,
    full_observation: Optional[Dict[str, Any]],
    ensure_addon_parents,
) -> Dict[str, Any]:
    messages = build_commander_messages(
        race=race,
        strategy_description=strategy_description,
        observation_text=observation_text,
        previous_macro_tasks=previous_macro_tasks,
        previous_army_summary=previous_army_summary,
        tool_mode="json",
        action_space=action_space,
    )
    result = call_openai_detailed(
        messages,
        model_key=model_key,
        timeout=120.0,
    )
    error = result.get("error") or ""
    content = result.get("content") or ""
    tool_calls = parse_tool_calls_from_content(content)
    tasks, policy, issues = apply_tool_calls(
        tool_calls,
        legal_action_keys=set(action_space.keys()),
    )
    if full_observation is not None and callable(ensure_addon_parents):
        tasks = ensure_addon_parents(tasks, full_observation)
    if not tool_calls and not error:
        issues = list(issues) + ["no_tool_calls_parsed"]
    return _pack_outcome(
        tool_calls=tool_calls,
        tasks=tasks,
        policy=policy,
        issues=issues,
        error=error,
        result=result,
        tool_mode="json",
    )
