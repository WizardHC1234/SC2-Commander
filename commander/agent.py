"""One Commander decision cycle: tool calls → macro tasks + army policy."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from llm.caller import call_openai_detailed

from commander.prompts import build_commander_messages
from commander.tools import (
    _macro_keys,
    apply_tool_calls,
    army_group_ids_from_observation,
    parse_tool_calls_from_content,
    validate_army_tools_for_cycle,
)
from commander.wake_events import (
    MAX_WAKE_REFLECTION_RETRIES,
    format_wake_reflection_feedback,
    validate_wake_for_cycle,
)


logger = logging.getLogger("commander.agent")


def _usage_as_dict(result: Dict[str, Any]) -> Dict[str, Any]:
    usage = result.get("usage") or {}
    if isinstance(usage, dict):
        return dict(usage)
    for method_name in ("model_dump", "dict"):
        method = getattr(usage, method_name, None)
        if callable(method):
            try:
                dumped = method()
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                pass
    return {}


def _sum_usage(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, int] = {}
    for part in parts:
        if not isinstance(part, dict):
            continue
        for key, value in part.items():
            try:
                totals[key] = int(totals.get(key, 0)) + int(value)
            except (TypeError, ValueError):
                continue
    return totals


def _pack_outcome(
    *,
    tool_calls: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    policy: Any,
    issues: List[str],
    error: str,
    result: Dict[str, Any],
    wake_event: Optional[Dict[str, Any]] = None,
    reflection_retries: int = 0,
    reflection_issues: Optional[List[str]] = None,
    prompt_messages: Optional[List[Dict[str, str]]] = None,
    messages_transcript: Optional[List[Dict[str, str]]] = None,
    reflection_rounds: Optional[List[Dict[str, Any]]] = None,
    usage_parts: Optional[List[Dict[str, Any]]] = None,
    accepted: bool = True,
) -> Dict[str, Any]:
    army_summary = {
        "commands": [
            {
                "group_id": c.group_id,
                "destination_zone_id": c.destination_zone_id,
                "movement_mode": c.movement_mode,
                "retreat_ratio": c.retreat_ratio,
            }
            for c in policy.commands
        ],
        "scan_zone_id": policy.scan_zone_id,
        "scout_zone_id": policy.scout_zone_id,
    }
    content = result.get("content") or ""
    usage = _usage_as_dict(result)
    parts = list(usage_parts or [])
    usage_total = _sum_usage(parts) if parts else usage
    transcript = list(messages_transcript or prompt_messages or [])
    # Append the final assistant turn so the transcript is a complete chat.
    if content or tool_calls:
        final_assistant = content
        if not final_assistant and tool_calls:
            final_assistant = (
                "(Native tool_calls; see tool_calls / raw_message fields.)"
            )
        if not transcript or transcript[-1].get("role") != "assistant":
            transcript = list(transcript)
            transcript.append(
                {"role": "assistant", "content": final_assistant or "(empty)"}
            )
    packed: Dict[str, Any] = {
        "tool_calls": tool_calls,
        "tasks": tasks,
        "policy": policy,
        "army_summary": army_summary,
        "wake_event": wake_event,
        "issues": issues,
        "error": error,
        "latency_seconds": result.get("latency_seconds"),
        "finish_reason": result.get("finish_reason"),
        "content": content,
        "assistant_content": content,
        "tool_mode": "json",
        "reflection_retries": reflection_retries,
        "reflection_issues": list(reflection_issues or []),
        "messages": deepcopy(prompt_messages or []),
        "messages_transcript": deepcopy(transcript),
        "reflection_rounds": list(reflection_rounds or []),
        "usage": usage,
        "usage_total": usage_total,
        "accepted": bool(accepted),
    }
    return packed


def _apply_parsed_calls(
    tool_calls: List[Dict[str, Any]],
    *,
    macro_space: Dict[str, str],
    full_observation: Optional[Dict[str, Any]],
    ensure_addon_parents,
) -> Tuple[List[Dict[str, Any]], Any, List[str], Optional[Dict[str, Any]]]:
    tasks, policy, issues, wake_event = apply_tool_calls(
        tool_calls,
        legal_action_keys=set(macro_space.keys()),
    )
    if full_observation is not None and callable(ensure_addon_parents):
        tasks = ensure_addon_parents(tasks, full_observation)
    return tasks, policy, issues, wake_event


def _is_wake_blocking_issue(item: Any) -> bool:
    text = str(item)
    return (
        text.startswith("wake_")
        or "wake_unreachable" in text
        or "wake_forbidden" in text
        or "wake_event" in text
    )


def _blocking_decision_issues(
    *,
    wake_event: Optional[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    policy: Any,
    apply_issues: List[str],
    legal_macro_keys: Optional[List[str]] = None,
    full_observation: Optional[Dict[str, Any]] = None,
) -> List[str]:
    macro_actions = [
        str(task.get("action") or "")
        for task in tasks
        if isinstance(task, dict) and task.get("action")
    ]
    blocking = validate_wake_for_cycle(
        wake_event,
        macro_actions=macro_actions,
        apply_issues=apply_issues,
        legal_macro_keys=legal_macro_keys,
    )
    blocking.extend(
        validate_army_tools_for_cycle(
            policy,
            required_group_ids=army_group_ids_from_observation(full_observation),
        )
    )
    return blocking


def run_commander_decision(
    *,
    race: str,
    strategy_description: str,
    observation_text: str,
    action_space: Dict[str, str],
    model_key: str,
    full_observation: Optional[Dict[str, Any]] = None,
    ensure_addon_parents=None,
    runtime_hint: str = "",
    map_topology_text: str = "",
) -> Dict[str, Any]:
    """Call the model and return applied macro/army results."""
    return _run_with_wake_reflection(
        race=race,
        strategy_description=strategy_description,
        observation_text=observation_text,
        runtime_hint=runtime_hint,
        map_topology_text=map_topology_text,
        action_space=action_space,
        model_key=model_key,
        full_observation=full_observation,
        ensure_addon_parents=ensure_addon_parents,
    )


def _run_with_wake_reflection(
    *,
    race: str,
    strategy_description: str,
    observation_text: str,
    action_space: Dict[str, str],
    model_key: str,
    full_observation: Optional[Dict[str, Any]],
    ensure_addon_parents,
    runtime_hint: str,
    map_topology_text: str,
) -> Dict[str, Any]:
    macro_space = _macro_keys(action_space)
    messages = build_commander_messages(
        race=race,
        strategy_description=strategy_description,
        observation_text=observation_text,
        runtime_hint=runtime_hint,
        map_topology_text=map_topology_text,
        action_space=action_space,
    )
    prompt_messages = deepcopy(messages)

    reflection_retries = 0
    reflection_issues: List[str] = []
    reflection_rounds: List[Dict[str, Any]] = []
    usage_parts: List[Dict[str, Any]] = []
    last_result: Dict[str, Any] = {}
    last_error = ""
    latency_total = 0.0

    while True:
        result = call_openai_detailed(messages, model_key=model_key, timeout=120.0)
        last_result = result
        last_error = result.get("error") or ""
        usage = _usage_as_dict(result)
        if usage:
            usage_parts.append(usage)
        try:
            latency_total += float(result.get("latency_seconds") or 0.0)
        except (TypeError, ValueError):
            pass

        content = result.get("content") or ""
        tool_calls = parse_tool_calls_from_content(content)

        tasks, policy, issues, wake_event = _apply_parsed_calls(
            tool_calls,
            macro_space=macro_space,
            full_observation=full_observation,
            ensure_addon_parents=ensure_addon_parents,
        )
        if not tool_calls and not last_error:
            issues = list(issues) + ["no_tool_calls_parsed"]

        blocking = _blocking_decision_issues(
            wake_event=wake_event,
            tasks=tasks,
            policy=policy,
            apply_issues=issues,
            legal_macro_keys=list(macro_space.keys()),
            full_observation=full_observation,
        )
        if not blocking:
            packed = _pack_outcome(
                tool_calls=tool_calls,
                tasks=tasks,
                policy=policy,
                issues=issues,
                error=last_error,
                result=last_result,
                wake_event=wake_event,
                reflection_retries=reflection_retries,
                reflection_issues=reflection_issues,
                prompt_messages=prompt_messages,
                messages_transcript=messages,
                reflection_rounds=reflection_rounds,
                usage_parts=usage_parts,
                accepted=True,
            )
            packed["latency_seconds"] = latency_total or packed.get(
                "latency_seconds"
            )
            return packed

        if reflection_retries >= MAX_WAKE_REFLECTION_RETRIES or last_error:
            issues = list(issues) + [
                f"decision_reflection_exhausted:{item}" for item in blocking
            ]
            logger.warning(
                "Decision validation still failing after %s reflection(s): %s",
                reflection_retries,
                blocking[:5],
            )
            non_wake_blocking = [
                item for item in blocking if not _is_wake_blocking_issue(item)
            ]
            if not non_wake_blocking and tool_calls and not last_error:
                # Wake-only failure: the macro/army tools themselves passed
                # validation, so apply them and let the bot arm the fallback
                # wake. Inheriting the previous decision here would discard a
                # valid tool set and can deadlock macro on a stale decision.
                # (Empty parses / call errors still take the inherit path so
                # active tasks are never wiped by a contentless response.)
                packed = _pack_outcome(
                    tool_calls=tool_calls,
                    tasks=tasks,
                    policy=policy,
                    issues=issues,
                    error=last_error,
                    result=last_result,
                    wake_event=None,
                    reflection_retries=reflection_retries,
                    reflection_issues=blocking,
                    prompt_messages=prompt_messages,
                    messages_transcript=messages,
                    reflection_rounds=reflection_rounds,
                    usage_parts=usage_parts,
                    accepted=True,
                )
                packed["latency_seconds"] = latency_total or packed.get(
                    "latency_seconds"
                )
                return packed
            packed = _pack_outcome(
                tool_calls=tool_calls,
                tasks=tasks,
                policy=policy,
                issues=issues,
                error=last_error,
                result=last_result,
                wake_event=None,
                reflection_retries=reflection_retries,
                reflection_issues=blocking,
                prompt_messages=prompt_messages,
                messages_transcript=messages,
                reflection_rounds=reflection_rounds,
                usage_parts=usage_parts,
                accepted=False,
            )
            packed["latency_seconds"] = latency_total or packed.get(
                "latency_seconds"
            )
            return packed

        feedback = format_wake_reflection_feedback(
            blocking_issues=blocking,
            previous_tool_calls=tool_calls,
        )
        reflection_issues = list(blocking)
        assistant_content = (result.get("content") or "").strip()
        if not assistant_content and tool_calls:
            assistant_content = (
                "(Previous response used tool_calls; see rejected list in the "
                "validation message.)"
            )
        reflection_rounds.append(
            {
                "round": reflection_retries,
                "accepted": False,
                "assistant_content": assistant_content or "(empty)",
                "tool_calls": deepcopy(tool_calls),
                "issues": list(blocking),
                "usage": usage,
                "finish_reason": result.get("finish_reason") or "",
                "error": last_error,
            }
        )
        reflection_retries += 1
        logger.info(
            "Decision validation failed (%s); requesting model reflection retry #%s",
            blocking[:3],
            reflection_retries,
        )

        # Keep a compact transcript of the rejected answer for reflection.
        messages = list(messages)
        messages.append({"role": "assistant", "content": assistant_content or "(empty)"})
        messages.append({"role": "user", "content": feedback})
