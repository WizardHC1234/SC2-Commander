"""Commander Agent: OpenAI tool_calls schema and application helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from commander.combat_policy import (
    ALLOWED_MOVEMENT_MODES,
    MOVE_TYPE_BY_MOVEMENT_MODE,
    ArmyControlPolicy,
    ArmyGroupCommand,
)

# Must match Action.py entries with type == "army".
ARMY_TOOL_NAMES = frozenset({"move_group", "scanner_sweep", "scout"})


def _macro_keys(action_space: Dict[str, str]) -> Dict[str, str]:
    return {
        name: description
        for name, description in action_space.items()
        if name not in ARMY_TOOL_NAMES
    }


def build_macro_tools(action_space: Dict[str, str]) -> List[Dict[str, Any]]:
    """One OpenAI function tool per macro Action.py key (to_count)."""
    tools: List[Dict[str, Any]] = []
    for name, description in _macro_keys(action_space).items():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description or f"Set absolute target for {name}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to_count": {
                                "type": "integer",
                                "minimum": 1,
                                "description": (
                                    "Absolute target count on the field "
                                    "(including under construction). "
                                    "For expand: desired active mineral-bearing bases."
                                ),
                            }
                        },
                        "required": ["to_count"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


def build_army_tools(action_space: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Army-control tools; descriptions prefer Action.py registry when provided."""
    space = action_space or {}
    modes = sorted(ALLOWED_MOVEMENT_MODES)
    return [
        {
            "type": "function",
            "function": {
                "name": "move_group",
                "description": space.get(
                    "move_group",
                    (
                        "Command one army_group to a destination zone with a movement mode. "
                        "Call exactly once per group_id in army_groups; omit when empty."
                    ),
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "group_id": {
                            "type": "string",
                            "description": (
                                "Exact group_id from this cycle's army_groups "
                                "(e.g. group_0). Must exist in the observation."
                            ),
                        },
                        "destination_zone_id": {
                            "type": "string",
                            "description": (
                                "Exact zone_id from the observation zone table "
                                "(e.g. zone_5). Must exist in the observation."
                            ),
                        },
                        "movement_mode": {
                            "type": "string",
                            "enum": modes,
                            "description": (
                                "How the group moves: regroup (safe gather), push, "
                                "assault, harass, defensive_retreat, panic_retreat, "
                                "or search_and_destroy (only with a runtime hint)."
                            ),
                        },
                    },
                    "required": [
                        "group_id",
                        "destination_zone_id",
                        "movement_mode",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scanner_sweep",
                "description": space.get(
                    "scanner_sweep",
                    (
                        "Request one Scanner Sweep on a zone (50 Orbital energy). "
                        "Omit to request no scan."
                    ),
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "zone_id": {
                            "type": "string",
                            "description": (
                                "Exact zone_id from the observation (e.g. zone_5). "
                                "Must exist in the observation."
                            ),
                        }
                    },
                    "required": ["zone_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scout",
                "description": space.get(
                    "scout",
                    (
                        "Send or keep one SCV zone scout. Omit to cancel any active scout."
                    ),
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "zone_id": {
                            "type": "string",
                            "description": (
                                "Exact zone_id from the observation (e.g. zone_3). "
                                "If a scout is already active, repeat the same zone_id "
                                "to preserve it."
                            ),
                        }
                    },
                    "required": ["zone_id"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def build_commander_tools(action_space: Dict[str, str]) -> List[Dict[str, Any]]:
    """Flat tool list from unified Action registry (macro + army control)."""
    return build_macro_tools(action_space) + build_army_tools(action_space)


def normalize_tool_calls(raw_tool_calls: Any) -> List[Dict[str, Any]]:
    """Normalize SDK / dict tool_calls into [{id,name,arguments_obj}]."""
    import json

    if not raw_tool_calls:
        return []
    out: List[Dict[str, Any]] = []
    for item in raw_tool_calls:
        if item is None:
            continue
        if isinstance(item, dict):
            fn = item.get("function") or {}
            name = fn.get("name") or item.get("name")
            args_raw = fn.get("arguments") or item.get("arguments") or "{}"
            call_id = item.get("id") or ""
        else:
            fn = getattr(item, "function", None)
            name = getattr(fn, "name", None) if fn is not None else getattr(item, "name", None)
            args_raw = (
                getattr(fn, "arguments", None) if fn is not None else getattr(item, "arguments", None)
            ) or "{}"
            call_id = getattr(item, "id", "") or ""
        if not isinstance(name, str) or not name:
            continue
        if isinstance(args_raw, dict):
            args_obj = args_raw
        else:
            try:
                args_obj = json.loads(args_raw)
            except Exception:
                args_obj = {}
            if not isinstance(args_obj, dict):
                args_obj = {}
        out.append({"id": call_id, "name": name, "arguments": args_obj})
    return out


def parse_tool_calls_from_content(text: str) -> List[Dict[str, Any]]:
    """Parse JSON tool_calls embedded in assistant content (vLLM / no native tools)."""
    import json
    import re

    if not text:
        return []
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    data: Any = None
    try:
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        return []

    raw_calls = data.get("tool_calls")
    if raw_calls is None and isinstance(data.get("tools"), list):
        raw_calls = data.get("tools")
    if not isinstance(raw_calls, list):
        # Single call object: {"name": "...", "arguments": {...}}
        if "name" in data:
            raw_calls = [data]
        else:
            return []

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_calls):
        if not isinstance(item, dict):
            continue
        name = item.get("name") or (item.get("function") or {}).get("name")
        args = item.get("arguments")
        if args is None:
            args = (item.get("function") or {}).get("arguments")
        if args is None and "to_count" in item:
            args = {k: v for k, v in item.items() if k != "name"}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(args, dict):
            args = {}
        normalized.append(
            {"id": str(item.get("id") or f"json_{idx}"), "name": name, "arguments": args}
        )
    return normalized


def apply_tool_calls(
    tool_calls: Sequence[Dict[str, Any]],
    *,
    legal_action_keys: Set[str],
) -> Tuple[List[Dict[str, Any]], ArmyControlPolicy, List[str]]:
    """Full-replace apply: macro tasks + army policy. Returns (tasks, policy, issues)."""
    issues: List[str] = []
    macro_by_key: Dict[str, int] = {}
    group_commands: Dict[str, ArmyGroupCommand] = {}
    scan_zone_id: Optional[str] = None
    scout_zone_id: Optional[str] = None
    saw_scan = False
    saw_scout = False

    for call in tool_calls:
        name = call.get("name") or ""
        args = call.get("arguments") or {}
        if not isinstance(args, dict):
            issues.append(f"invalid_args:{name}")
            continue

        if name in legal_action_keys:
            raw = args.get("to_count")
            try:
                to_count = int(raw)
            except (TypeError, ValueError):
                issues.append(f"bad_to_count:{name}")
                continue
            if to_count <= 0:
                issues.append(f"non_positive_to_count:{name}")
                continue
            # Last write wins for to_count; move key to the end so list order
            # matches the LLM's resource-priority order (last occurrence).
            if name in macro_by_key:
                del macro_by_key[name]
            macro_by_key[name] = to_count
            continue

        if name == "move_group":
            try:
                cmd = _parse_move_group(args)
            except ValueError as exc:
                issues.append(f"move_group:{exc}")
                continue
            group_commands[cmd.group_id] = cmd
            continue

        if name == "scanner_sweep":
            zone = _optional_zone(args.get("zone_id"), "zone_id")
            if zone is None:
                issues.append("scanner_sweep:bad_zone")
                continue
            scan_zone_id = zone
            saw_scan = True
            continue

        if name == "scout":
            zone = _optional_zone(args.get("zone_id"), "zone_id")
            if zone is None:
                issues.append("scout:bad_zone")
                continue
            scout_zone_id = zone
            saw_scout = True
            continue

        issues.append(f"unknown_tool:{name}")

    # Keep LLM tool-call order as resource priority (dict insertion order).
    tasks = [
        {"action": key, "to_count": count}
        for key, count in macro_by_key.items()
    ]

    policy = ArmyControlPolicy(
        commands=list(group_commands.values()),
        scan_zone_id=scan_zone_id if saw_scan else None,
        scout_zone_id=scout_zone_id if saw_scout else None,
    )
    return tasks, policy, issues


def _parse_move_group(args: Dict[str, Any]) -> ArmyGroupCommand:
    group_id = str(args.get("group_id") or "").strip()
    zone_id = str(args.get("destination_zone_id") or "").strip()
    mode = str(args.get("movement_mode") or "").strip()
    if not group_id.startswith("group_") or not group_id[6:].isdigit():
        raise ValueError("bad_group_id")
    if not zone_id.startswith("zone_") or not zone_id[5:].isdigit():
        raise ValueError("bad_zone_id")
    if mode not in ALLOWED_MOVEMENT_MODES:
        raise ValueError("bad_movement_mode")
    return ArmyGroupCommand(
        group_id=group_id,
        destination_zone_id=zone_id,
        movement_mode=mode,
        move_type=MOVE_TYPE_BY_MOVEMENT_MODE[mode],
    )


def _optional_zone(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized.startswith("zone_") or not normalized[5:].isdigit():
        return None
    return normalized
