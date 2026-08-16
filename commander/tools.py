"""Commander Agent: JSON tool_calls parsing and application helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from commander.combat_policy import (
    ArmyIntent,
    ArmyControlPolicy,
)
from commander.wake_events import normalize_wake_event

# Must match race action-registry entries with type == "army" / "meta".
ARMY_TOOL_NAMES = frozenset(
    {"army_intent", "scanner_sweep", "scout"}
)
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


def validate_army_tools_for_cycle(
    policy: ArmyControlPolicy,
) -> List[str]:
    """Require one whole-army intent, including before combat units exist."""
    if policy.army_intent is not None:
        return []
    return [
        "army_intent:missing — emit exactly one army_intent every decision "
        "cycle, including while the army is still being produced"
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
    army_intent: Optional[ArmyIntent] = None
    army_intent_count = 0
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

        if name == "army_intent":
            army_intent_count += 1
            try:
                army_intent = _parse_army_intent(args)
            except ValueError as exc:
                issues.append(f"army_intent:{exc}")
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

    if army_intent_count > 1:
        issues.append("army_intent:duplicate")
        army_intent = None

    policy = ArmyControlPolicy(
        commands=[],
        army_intent=army_intent,
        scan_zone_id=scan_zone_id if saw_scan else None,
        scout_zone_id=scout_zone_id if saw_scout else None,
    )
    return tasks, policy, issues, wake_event


def _parse_army_intent(args: Dict[str, Any]) -> ArmyIntent:
    mode = str(args.get("mode") or args.get("intent") or "").strip().lower()
    if mode not in {"hold", "attack", "regroup", "cleanup"}:
        raise ValueError("bad_mode")
    zone_id = _optional_zone(args.get("zone_id"))
    if zone_id is None:
        raise ValueError("bad_zone_id")
    return ArmyIntent(mode=mode, zone_id=zone_id)


def _optional_zone(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized.startswith("zone_") or not normalized[5:].isdigit():
        return None
    return normalized
