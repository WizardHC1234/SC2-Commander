"""Commander Agent: JSON tool_calls parsing and application helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from commander.combat_policy import (
    ALLOWED_MOVEMENT_MODES,
    MOVE_TYPE_BY_MOVEMENT_MODE,
    ArmyControlPolicy,
    ArmyGroupCommand,
)
from commander.retreat_policy import clamp_retreat_ratio
from commander.wake_events import normalize_wake_event

# Must match race action-registry entries with type == "army" / "meta".
ARMY_TOOL_NAMES = frozenset({"move_group", "scanner_sweep", "scout"})
META_TOOL_NAMES = frozenset({"set_wake_event"})
NON_MACRO_TOOL_NAMES = ARMY_TOOL_NAMES | META_TOOL_NAMES


def _macro_keys(action_space: Dict[str, str]) -> Dict[str, str]:
    return {
        name: description
        for name, description in action_space.items()
        if name not in NON_MACRO_TOOL_NAMES
    }


def _extract_tool_calls_payload(data: Any) -> Optional[Dict[str, Any]]:
    """Pick a dict that looks like tool_calls JSON from json / json_repair output."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        for item in data:
            found = _extract_tool_calls_payload(item)
            if found is not None and (
                "tool_calls" in found
                or "tools" in found
                or "name" in found
            ):
                return found
        for item in data:
            found = _extract_tool_calls_payload(item)
            if found is not None:
                return found
    return None


def _loads_tool_json(text: str) -> Any:
    """Strict json.loads first; fall back to json_repair for LLM-broken JSON."""
    import json
    import re

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        blob = match.group(0)
        try:
            return json.loads(blob)
        except Exception:
            pass
        try:
            from json_repair import loads as repair_loads

            return repair_loads(blob)
        except Exception:
            pass

    try:
        from json_repair import loads as repair_loads

        return repair_loads(text)
    except Exception:
        return None


def parse_tool_calls_from_content(text: str) -> List[Dict[str, Any]]:
    """Parse JSON tool_calls embedded in assistant content.

    Falls back to ``json_repair`` when the model emits broken braces/commas, which
    is common with JSON tool_mode (e.g. ``\"seconds\":1438}}]``).
    """
    import json
    import re

    if not text:
        return []
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    data = _extract_tool_calls_payload(_loads_tool_json(cleaned))
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
                try:
                    from json_repair import loads as repair_loads

                    args = repair_loads(args)
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


def army_group_ids_from_observation(
    full_observation: Optional[Dict[str, Any]],
) -> List[str]:
    """Extract current army_groups ids from the canonical full observation."""
    if not isinstance(full_observation, dict):
        return []
    army = full_observation.get("army_control")
    if not isinstance(army, dict):
        return []
    ids: List[str] = []
    for group in army.get("groups") or []:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("group_id") or "").strip()
        if group_id:
            ids.append(group_id)
    return ids


def validate_army_tools_for_cycle(
    policy: ArmyControlPolicy,
    *,
    required_group_ids: Sequence[str],
) -> List[str]:
    """Blocking army issues: every observed group must receive move_group."""
    required = [str(gid).strip() for gid in required_group_ids if str(gid).strip()]
    if not required:
        return []
    commanded = {
        str(command.group_id).strip()
        for command in (policy.commands or [])
        if getattr(command, "group_id", None)
    }
    missing = [gid for gid in required if gid not in commanded]
    if not missing:
        return []
    if len(missing) == len(required):
        return [
            "army_move_group:missing — army_groups is non-empty so emit exactly "
            f"one move_group per group_id ({', '.join(required)}); typically "
            "hold at a safe defensive zone while production continues"
        ]
    return [
        "army_move_group:incomplete — missing move_group for "
        f"{', '.join(missing)} (required: {', '.join(required)})"
    ]


def apply_tool_calls(
    tool_calls: Sequence[Dict[str, Any]],
    *,
    legal_action_keys: Set[str],
) -> Tuple[List[Dict[str, Any]], ArmyControlPolicy, List[str], Optional[Dict[str, Any]]]:
    """Full-replace apply: macro tasks + army policy + optional wake event.

    Returns ``(tasks, policy, issues, wake_event)``. ``wake_event`` is None when
    ``set_wake_event`` was omitted or fully invalid (caller may apply fallback).
    """
    issues: List[str] = []
    macro_by_key: Dict[str, int] = {}
    group_commands: Dict[str, ArmyGroupCommand] = {}
    scan_zone_id: Optional[str] = None
    scout_zone_id: Optional[str] = None
    saw_scan = False
    saw_scout = False
    wake_event: Optional[Dict[str, Any]] = None
    saw_wake = False

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
            zone = _optional_zone(args.get("zone_id"))
            if zone is None:
                issues.append("scanner_sweep:bad_zone")
                continue
            scan_zone_id = zone
            saw_scan = True
            continue

        if name == "scout":
            zone = _optional_zone(args.get("zone_id"))
            if zone is None:
                issues.append("scout:bad_zone")
                continue
            scout_zone_id = zone
            saw_scout = True
            continue

        if name == "set_wake_event":
            saw_wake = True
            event, wake_issues = normalize_wake_event(args)
            issues.extend(wake_issues)
            if event is not None:
                wake_event = event
            continue

        issues.append(f"unknown_tool:{name}")

    if not saw_wake:
        issues.append("wake_event:missing")

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
    return tasks, policy, issues, wake_event


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
        retreat_ratio=_parse_retreat_ratio(args.get("retreat_ratio")),
    )


def _parse_retreat_ratio(value: Any) -> Optional[float]:
    """Soft-parse: a bad value falls back to the runtime default instead of
    dropping the whole move_group command."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return round(clamp_retreat_ratio(float(value)), 2)
    except (TypeError, ValueError):
        return None


def _optional_zone(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized.startswith("zone_") or not normalized[5:].isdigit():
        return None
    return normalized
