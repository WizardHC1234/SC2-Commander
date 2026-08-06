"""One Commander decision cycle: tool calls → macro tasks + army policy."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from llm.caller import call_openai_detailed, load_agent_pool

from commander.prompts import build_commander_messages
from commander.tools import (
    NON_MACRO_TOOL_NAMES,
    apply_tool_calls,
    army_group_ids_from_observation,
    build_commander_tools,
    normalize_tool_calls,
    parse_tool_calls_from_content,
    validate_army_tools_for_cycle,
)
from commander.wake_events import (
    MAX_WAKE_REFLECTION_RETRIES,
    format_wake_reflection_feedback,
    validate_wake_for_cycle,
)


def _macro_action_space(action_space: Dict[str, str]) -> Dict[str, str]:
    return {
        key: description
        for key, description in action_space.items()
        if key not in NON_MACRO_TOOL_NAMES
    }

logger = logging.getLogger("commander.agent")

_NATIVE_TOOL_CHOICE_ERRORS = (
    "enable-auto-tool-choice",
    "tool-call-parser",
    "tool_choice",
    "tool choice",
    "does not support tools",
    "does not support function",
    "tools is not supported",
    "function calling is not enabled",
    "tool use is not supported",
    "unsupported.*tool",
)

# After a model proves native tools are unavailable, keep using JSON for the
# rest of this process so later decisions do not pay a failed native attempt.
_json_fallback_models: Set[str] = set()


def _resolve_preferred_tool_mode(model_key: str) -> str:
    """Prefer native tools= API; JSON only when forced or already known broken.

    Config ``tool_mode: "json"`` still forces JSON (escape hatch). Otherwise
    default is native; auto-fallback records the model in
    ``_json_fallback_models`` after an unsupported-tools error.
    """
    if model_key in _json_fallback_models:
        return "json"
    pool = (load_agent_pool().get("llm_agents_pool") or {}).get(model_key) or {}
    mode = str(pool.get("tool_mode") or "native").strip().lower()
    if mode == "json":
        return "json"
    return "native"


def _is_native_tools_unsupported(error: str) -> bool:
    text = (error or "").lower()
    if not text:
        return False
    if any(token in text for token in _NATIVE_TOOL_CHOICE_ERRORS):
        return True
    # Broad catch for OpenAI-compatible servers that reject the tools field.
    if "tool" in text and any(
        word in text
        for word in ("unsupported", "not support", "invalid", "unknown")
    ):
        return True
    return False


def _pack_outcome(
    *,
    tool_calls: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    policy: Any,
    issues: List[str],
    error: str,
    result: Dict[str, Any],
    tool_mode: str,
    wake_event: Optional[Dict[str, Any]] = None,
    reflection_retries: int = 0,
    reflection_issues: Optional[List[str]] = None,
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
        "wake_event": wake_event,
        "issues": issues,
        "error": error,
        "latency_seconds": result.get("latency_seconds"),
        "finish_reason": result.get("finish_reason"),
        "content": result.get("content") or "",
        "tool_mode": tool_mode,
        "reflection_retries": reflection_retries,
        "reflection_issues": list(reflection_issues or []),
    }


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
    previous_macro_tasks: List[Dict[str, Any]],
    previous_army_summary: Optional[Dict[str, Any]],
    action_space: Dict[str, str],
    model_key: str,
    full_observation: Optional[Dict[str, Any]] = None,
    ensure_addon_parents=None,
    runtime_hint: str = "",
) -> Dict[str, Any]:
    """Call the model and return applied macro/army results."""
    tool_mode = _resolve_preferred_tool_mode(model_key)
    if tool_mode == "json":
        return _run_with_wake_reflection(
            race=race,
            strategy_description=strategy_description,
            observation_text=observation_text,
            previous_macro_tasks=previous_macro_tasks,
            previous_army_summary=previous_army_summary,
            runtime_hint=runtime_hint,
            action_space=action_space,
            model_key=model_key,
            full_observation=full_observation,
            ensure_addon_parents=ensure_addon_parents,
            tool_mode="json",
        )

    # Default: native OpenAI tools=. Do NOT send tool_choice="auto" — many
    # vLLM servers reject it unless started with --enable-auto-tool-choice.
    outcome = _run_with_wake_reflection(
        race=race,
        strategy_description=strategy_description,
        observation_text=observation_text,
        previous_macro_tasks=previous_macro_tasks,
        previous_army_summary=previous_army_summary,
        runtime_hint=runtime_hint,
        action_space=action_space,
        model_key=model_key,
        full_observation=full_observation,
        ensure_addon_parents=ensure_addon_parents,
        tool_mode="native",
    )
    error = outcome.get("error") or ""
    if error and _is_native_tools_unsupported(error):
        _json_fallback_models.add(model_key)
        logger.warning(
            "Native tools unsupported for %s (%s); falling back to JSON "
            "tool_mode for this process.",
            model_key,
            error[:160],
        )
        return _run_with_wake_reflection(
            race=race,
            strategy_description=strategy_description,
            observation_text=observation_text,
            previous_macro_tasks=previous_macro_tasks,
            previous_army_summary=previous_army_summary,
            runtime_hint=runtime_hint,
            action_space=action_space,
            model_key=model_key,
            full_observation=full_observation,
            ensure_addon_parents=ensure_addon_parents,
            tool_mode="json",
        )
    return outcome


def _run_with_wake_reflection(
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
    runtime_hint: str,
    tool_mode: str,
) -> Dict[str, Any]:
    macro_space = _macro_action_space(action_space)
    # JSON mode has no native tools payload, so pass the full Action.py catalog
    # (macro + army + meta descriptions) into the prompt. Native mode only needs
    # macro names in the prompt text; tool descriptions arrive via tools=.
    prompt_action_space = (
        action_space if tool_mode == "json" else macro_space
    )
    messages = build_commander_messages(
        race=race,
        strategy_description=strategy_description,
        observation_text=observation_text,
        previous_macro_tasks=previous_macro_tasks,
        previous_army_summary=previous_army_summary,
        runtime_hint=runtime_hint,
        tool_mode=tool_mode,
        action_space=prompt_action_space,
    )
    tools = build_commander_tools(action_space) if tool_mode == "native" else None

    reflection_retries = 0
    reflection_issues: List[str] = []
    last_result: Dict[str, Any] = {}
    last_error = ""
    latency_total = 0.0

    while True:
        call_kwargs: Dict[str, Any] = {
            "model_key": model_key,
            "timeout": 120.0,
        }
        if tools is not None:
            call_kwargs["tools"] = tools
        result = call_openai_detailed(messages, **call_kwargs)
        last_result = result
        last_error = result.get("error") or ""
        try:
            latency_total += float(result.get("latency_seconds") or 0.0)
        except (TypeError, ValueError):
            pass

        if tool_mode == "native":
            raw_message = result.get("raw_message") or {}
            tool_calls = normalize_tool_calls(raw_message.get("tool_calls"))
            if not tool_calls and (result.get("content") or ""):
                tool_calls = parse_tool_calls_from_content(
                    result.get("content") or ""
                )
        else:
            content = result.get("content") or ""
            tool_calls = parse_tool_calls_from_content(content)

        tasks, policy, issues, wake_event = _apply_parsed_calls(
            tool_calls,
            macro_space=macro_space,
            full_observation=full_observation,
            ensure_addon_parents=ensure_addon_parents,
        )
        if tool_mode == "json" and not tool_calls and not last_error:
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
                tool_mode=tool_mode,
                wake_event=wake_event,
                reflection_retries=reflection_retries,
                reflection_issues=reflection_issues,
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
            wake_ok = not any(
                str(item).startswith("wake_")
                or "wake_unreachable" in str(item)
                or "wake_forbidden" in str(item)
                or "wake_event" in str(item)
                for item in blocking
            )
            packed = _pack_outcome(
                tool_calls=tool_calls,
                tasks=tasks,
                policy=policy,
                issues=issues,
                error=last_error,
                result=last_result,
                tool_mode=tool_mode,
                wake_event=wake_event if wake_ok else None,
                reflection_retries=reflection_retries,
                reflection_issues=blocking,
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
        reflection_retries += 1
        logger.info(
            "Decision validation failed (%s); requesting model reflection retry #%s",
            blocking[:3],
            reflection_retries,
        )

        # Keep a compact transcript of the rejected answer for reflection.
        assistant_content = (result.get("content") or "").strip()
        if not assistant_content and tool_calls:
            assistant_content = (
                "(Previous response used tool_calls; see rejected list in the "
                "validation message.)"
            )
        messages = list(messages)
        messages.append({"role": "assistant", "content": assistant_content or "(empty)"})
        messages.append({"role": "user", "content": feedback})
