"""Model-authored wake events: validate, evaluate, rising-edge helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

ALLOWED_LOGICS = frozenset({"all", "any"})

UNIT_COUNT_TYPES = frozenset({"unit_count_at_least", "unit_count_less_than"})
ARMY_GROUP_COUNT_TYPES = frozenset(
    {"army_group_count_at_least", "army_group_count_less_than"}
)
MOVEMENT_MODE_TYPES = frozenset({"movement_mode_in", "movement_mode_not_in"})
OBJECTIVE_IS_TYPES = frozenset({"objective_status_is"})
OBJECTIVE_BECAME_TYPES = frozenset({"objective_status_became"})
OBJECTIVE_TYPES = OBJECTIVE_IS_TYPES | OBJECTIVE_BECAME_TYPES
OBJECTIVE_STATUS_VALUES = frozenset(
    {
        "none",
        "unknown",
        "enemy_present",
        "confirmed_clear",
        "reached_unconfirmed",
        "en_route_unconfirmed",
    }
)
DESTINATION_TYPES = frozenset({"destination_reached"})
SCOUT_TYPES = frozenset({"scout_result_is", "scout_just_finished"})
# Rejected as wake conditions:
# - scout_*: SCV scouts often killed en route (unreliable timers).
# - movement_mode_*: modes only change via Commander move_group (circular).
# - army_group_count_*: after the first group appears, at_least is usually already
#   true; less_than has no practical progress-wake use.
# - objective_status_is: level check; if already true at arm, rising-edge never
#   fires — use objective_status_became instead.
DISABLED_WAKE_TYPES = frozenset(
    SCOUT_TYPES
    | MOVEMENT_MODE_TYPES
    | ARMY_GROUP_COUNT_TYPES
    | OBJECTIVE_IS_TYPES
)
BOOL_FLAG_TYPES = frozenset({"scan_ready", "cleanup_hint_present"})
TIME_TYPES = frozenset({"game_time_at_least"})
SUPPLY_TYPES = frozenset({"supply_left_at_most"})

# Legacy abbreviated names still accepted and rewritten on normalize.
PREDICATE_TYPE_ALIASES = {
    "unit_count_gte": "unit_count_at_least",
    "unit_count_lt": "unit_count_less_than",
    "game_time_gte": "game_time_at_least",
    "supply_left_lte": "supply_left_at_most",
    "army_group_count_gte": "army_group_count_at_least",
    "army_group_count_lt": "army_group_count_less_than",
}

ALLOWED_CONDITION_TYPES = (
    UNIT_COUNT_TYPES
    | ARMY_GROUP_COUNT_TYPES
    | MOVEMENT_MODE_TYPES
    | OBJECTIVE_TYPES
    | DESTINATION_TYPES
    | SCOUT_TYPES
    | BOOL_FLAG_TYPES
    | TIME_TYPES
    | SUPPLY_TYPES
)

# Runtime emits "completed"; accept legacy "reached" from older records/prompts.
SCOUT_RESULT_ALIASES = {
    "reached": "completed",
    "complete": "completed",
    "completed": "completed",
    "killed_en_route": "killed_en_route",
    "interrupted": "interrupted",
}
SCOUT_TERMINAL_RESULTS = frozenset(
    {"completed", "killed_en_route", "interrupted"}
)

FALLBACK_DELAY_SECONDS = 60.0
MAX_WAKE_REFLECTION_RETRIES = 1


def _disabled_wake_reason(ctype: str) -> str:
    if ctype in SCOUT_TYPES:
        return "scout_wake_disabled"
    if ctype in MOVEMENT_MODE_TYPES:
        return "movement_mode_wake_disabled"
    if ctype in ARMY_GROUP_COUNT_TYPES:
        return "army_group_count_wake_disabled"
    if ctype in OBJECTIVE_IS_TYPES:
        return "objective_status_is_wake_disabled"
    return "wake_disabled"


def normalize_scout_result(value: Any) -> str:
    text = str(value or "").strip().lower()
    return SCOUT_RESULT_ALIASES.get(text, text)


def fallback_wake_event(now: float, *, delay: float = FALLBACK_DELAY_SECONDS) -> Dict[str, Any]:
    """Weak safety net when the model omits set_wake_event."""
    return {
        "logic": "any",
        "conditions": [
            {
                "type": "game_time_at_least",
                "seconds": float(now) + float(delay),
            }
        ],
    }


def _unit_has_train_tool(unit: str, train_stems: set) -> bool:
    raw = unit.replace("_", "").replace(" ", "").lower()
    if not raw:
        return False
    # SiegeTankSieged / HellionTank etc. → base combat form.
    for suffix in ("sieged", "burrowed", "flying"):
        if raw.endswith(suffix) and len(raw) > len(suffix):
            raw = raw[: -len(suffix)]
            break
    if raw in train_stems:
        return True
    return any(raw.startswith(stem) or stem.startswith(raw) for stem in train_stems)


def _train_stems_from_macro(macro_action_keys: Sequence[str]) -> set:
    return {
        key[len("train_") :].replace("_", "").lower()
        for key in macro_action_keys
        if isinstance(key, str) and key.startswith("train_")
    }


def _needed_train_keys(unit: str, macro_action_keys: Sequence[str]) -> List[str]:
    """Best-effort train_* keys that could produce ``unit``."""
    token = unit.replace("_", "").replace(" ", "").lower()
    for suffix in ("sieged", "burrowed", "flying"):
        if token.endswith(suffix) and len(token) > len(suffix):
            token = token[: -len(suffix)]
            break
    matches: List[str] = []
    for key in macro_action_keys:
        if not isinstance(key, str) or not key.startswith("train_"):
            continue
        stem = key[len("train_") :].replace("_", "").lower()
        if not stem:
            continue
        if token == stem or token.startswith(stem) or stem.startswith(token):
            matches.append(key)
    return matches


def validate_wake_for_cycle(
    wake_event: Optional[Dict[str, Any]],
    *,
    macro_actions: Sequence[str],
    apply_issues: Optional[Sequence[str]] = None,
    legal_macro_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    """Blocking wake issues that should trigger model reflection (not silent drop)."""
    blocking: List[str] = []
    apply_issues = list(apply_issues or [])
    macro_list = [str(key) for key in macro_actions if key]
    macro_set = set(macro_list)
    train_stems = _train_stems_from_macro(macro_list)
    suggest_keys = [
        str(key)
        for key in (legal_macro_keys if legal_macro_keys is not None else macro_list)
        if key
    ]

    if wake_event is None:
        if any(item.startswith("wake_event:missing") for item in apply_issues):
            blocking.append("wake_event:missing")
        elif any("_wake_disabled" in item for item in apply_issues):
            blocking.append(
                "wake_event:disabled_predicate — do not use scout_result_is, "
                "scout_just_finished, movement_mode_in, movement_mode_not_in, "
                "army_group_count_at_least, army_group_count_less_than, or "
                "objective_status_is; prefer unit_count_at_least / "
                "unit_count_less_than / supply_left_at_most / "
                "objective_status_became / destination_reached / scan_ready / "
                "cleanup_hint_present / game_time_at_least"
            )
        elif any("no_valid_conditions" in item for item in apply_issues):
            blocking.append("wake_event:no_valid_conditions")
        else:
            blocking.append("wake_event:missing_or_invalid")
        return blocking

    conditions = wake_event.get("conditions") if isinstance(wake_event, dict) else None
    if not isinstance(conditions, list) or not conditions:
        blocking.append("wake_event:no_valid_conditions")
        return blocking

    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        ctype = str(cond.get("type") or "")
        if ctype not in UNIT_COUNT_TYPES:
            continue
        unit = str(cond.get("unit") or "").strip()
        if _unit_has_train_tool(unit, train_stems):
            continue
        needed = _needed_train_keys(unit, suggest_keys)
        if needed:
            blocking.append(
                f"wake_unreachable:{ctype}:{unit or '?'} — include "
                f"{'|'.join(needed)} in this cycle's tool_calls before waking "
                f"on that unit count (or wake on infrastructure/time instead)"
            )
        else:
            blocking.append(
                f"wake_unreachable:{ctype}:{unit or '?'} — no matching train_* "
                "action in the legal macro set"
            )
    # Scout predicates should already be stripped; surface if somehow present.
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        ctype = str(cond.get("type") or "")
        if ctype in DISABLED_WAKE_TYPES:
            blocking.append(
                f"wake_forbidden:{ctype} — "
                f"{_disabled_wake_reason(ctype).replace('_wake_disabled', '')} "
                "wake predicates are disabled"
            )
    return blocking


def format_wake_reflection_feedback(
    *,
    blocking_issues: Sequence[str],
    previous_tool_calls: Sequence[Dict[str, Any]],
) -> str:
    """User-turn text asking the model to rethink and re-emit tool_calls."""
    lines = [
        "[Decision Validation Failed — Reflect and Correct]",
        "Your previous tool_calls were rejected. Fix EVERY issue below and emit a "
        "COMPLETE corrected tool_calls set for this same decision cycle.",
        "",
        "Issues:",
    ]
    for issue in blocking_issues:
        lines.append(f"- {issue}")
    lines.extend(
        [
            "",
            "Rules:",
            "- When army_groups is non-empty, emit exactly one move_group for every "
            "group_id in this cycle (regroup to a safe staging zone is fine before "
            "the attack gate). Do not omit army tools while waiting on production.",
            "- set_wake_event is required and must be reachable from the macro tools "
            "you emit in THIS response.",
            "- If you wake on unit_count_at_least or unit_count_less_than for a "
            "unit, include the matching train tool in the same tool_calls "
            "(example failure: Marine at_least 45 without train_marine).",
            "- Do not use scout_result_is, scout_just_finished, movement_mode_in, "
            "movement_mode_not_in, army_group_count_at_least, "
            "army_group_count_less_than, or objective_status_is as wake "
            "conditions (use objective_status_became when you need an "
            "objective-status edge).",
            "- While infrastructure is still missing, prefer game_time_at_least a "
            "short time ahead, supply_left_at_most, or other currently achievable "
            "predicates — not an attack-gate unit count you are not producing yet. "
            "objective_status_became is only for army destination statuses "
            "(confirmed_clear, enemy_present, ...), never for buildings/research.",
            "- Keep all still-valid macro and army tools; do not shrink to only "
            "set_wake_event.",
            "",
            "Previous rejected tool_calls:",
            json.dumps(
                [
                    {
                        "name": call.get("name"),
                        "arguments": call.get("arguments") or {},
                    }
                    for call in previous_tool_calls
                ],
                ensure_ascii=False,
            ),
        ]
    )
    return "\n".join(lines)


def normalize_wake_event(raw: Any) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Validate and normalize one composite wake event.

    Returns ``(event_or_none, issues)``. Invalid input yields ``None``.
    """
    issues: List[str] = []
    if raw is None:
        return None, ["wake_event:missing"]
    if not isinstance(raw, dict):
        return None, ["wake_event:not_object"]

    logic = str(raw.get("logic") or "all").strip().lower()
    if logic not in ALLOWED_LOGICS:
        issues.append(f"wake_event:bad_logic:{logic}")
        return None, issues

    conditions_raw = raw.get("conditions")
    if not isinstance(conditions_raw, list) or not conditions_raw:
        issues.append("wake_event:empty_conditions")
        return None, issues

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(conditions_raw):
        cond, cond_issues = _normalize_condition(item, idx)
        issues.extend(cond_issues)
        if cond is not None:
            normalized.append(cond)

    if not normalized:
        issues.append("wake_event:no_valid_conditions")
        return None, issues

    return {"logic": logic, "conditions": normalized}, issues


def _normalize_condition(
    item: Any, idx: int
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    issues: List[str] = []
    prefix = f"wake_cond[{idx}]"
    if not isinstance(item, dict):
        return None, [f"{prefix}:not_object"]

    ctype = str(item.get("type") or "").strip()
    ctype = PREDICATE_TYPE_ALIASES.get(ctype, ctype)
    if ctype in DISABLED_WAKE_TYPES:
        reason = _disabled_wake_reason(ctype)
        return None, [f"{prefix}:{reason}:{ctype}"]
    if ctype not in ALLOWED_CONDITION_TYPES:
        return None, [f"{prefix}:bad_type:{ctype or '?'}"]

    if ctype in UNIT_COUNT_TYPES:
        unit = str(item.get("unit") or "").strip()
        if not unit:
            return None, [f"{prefix}:missing_unit"]
        try:
            count = int(item.get("count"))
        except (TypeError, ValueError):
            return None, [f"{prefix}:bad_count"]
        if count < 0:
            return None, [f"{prefix}:negative_count"]
        return {"type": ctype, "unit": unit, "count": count}, issues

    if ctype in ARMY_GROUP_COUNT_TYPES:
        try:
            count = int(item.get("count"))
        except (TypeError, ValueError):
            return None, [f"{prefix}:bad_count"]
        if count < 0:
            return None, [f"{prefix}:negative_count"]
        return {"type": ctype, "count": count}, issues

    if ctype in MOVEMENT_MODE_TYPES:
        modes = _string_list(item.get("modes"))
        if not modes:
            return None, [f"{prefix}:empty_modes"]
        return {"type": ctype, "modes": modes}, issues

    if ctype in OBJECTIVE_TYPES:
        status = str(item.get("status") or "").strip()
        if not status:
            return None, [f"{prefix}:missing_status"]
        if status not in OBJECTIVE_STATUS_VALUES:
            return None, [
                f"{prefix}:bad_objective_status:{status} — use army destination "
                "status only "
                f"({', '.join(sorted(OBJECTIVE_STATUS_VALUES))}), not building "
                "or research completion"
            ]
        return {"type": ctype, "status": status}, issues

    if ctype in DESTINATION_TYPES:
        return {"type": ctype}, issues

    if ctype == "scout_result_is":
        result = normalize_scout_result(item.get("result"))
        if not result:
            return None, [f"{prefix}:missing_result"]
        return {"type": ctype, "result": result}, issues

    if ctype == "scout_just_finished":
        return {"type": ctype}, issues

    if ctype in BOOL_FLAG_TYPES:
        return {"type": ctype}, issues

    if ctype in TIME_TYPES:
        try:
            seconds = float(item.get("seconds"))
        except (TypeError, ValueError):
            return None, [f"{prefix}:bad_seconds"]
        if seconds < 0:
            return None, [f"{prefix}:negative_seconds"]
        return {"type": ctype, "seconds": seconds}, issues

    if ctype in SUPPLY_TYPES:
        try:
            count = int(item.get("count"))
        except (TypeError, ValueError):
            return None, [f"{prefix}:bad_count"]
        if count < 0:
            return None, [f"{prefix}:negative_count"]
        return {"type": ctype, "count": count}, issues

    return None, [f"{prefix}:unsupported"]


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, (list, tuple)):
        return []
    out: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def evaluate_wake_event(event: Optional[Dict[str, Any]], snapshot: Dict[str, Any]) -> bool:
    """Return whether the composite wake event is currently satisfied."""
    if not event or not isinstance(event, dict):
        return False
    conditions = event.get("conditions") or []
    if not isinstance(conditions, list) or not conditions:
        return False
    logic = str(event.get("logic") or "all").strip().lower()
    results = [_evaluate_condition(cond, snapshot) for cond in conditions]
    if logic == "any":
        return any(results)
    return all(results)


def rising_edge(satisfied: bool, prev_satisfied: Optional[bool]) -> bool:
    """True only on false→true (or None→true treated as no prior true)."""
    return bool(satisfied) and not bool(prev_satisfied)


def compact_wake_event(event: Optional[Dict[str, Any]]) -> str:
    if not event:
        return "{}"
    try:
        return json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return "{}"


def format_wake_condition(cond: Any) -> str:
    """One-line label for a wake predicate (for trigger hints / logs)."""
    if not isinstance(cond, dict):
        return "invalid_condition"
    ctype = str(cond.get("type") or "").strip()
    ctype = PREDICATE_TYPE_ALIASES.get(ctype, ctype)
    if ctype in UNIT_COUNT_TYPES:
        unit = str(cond.get("unit") or "?").strip() or "?"
        count = cond.get("count")
        op = ">=" if ctype == "unit_count_at_least" else "<"
        return f"{ctype}({unit}{op}{count})"
    if ctype in TIME_TYPES:
        return f"{ctype}(seconds={cond.get('seconds')})"
    if ctype in SUPPLY_TYPES:
        return f"{ctype}(count={cond.get('count')})"
    if ctype in OBJECTIVE_TYPES:
        return f"{ctype}(status={cond.get('status')})"
    if ctype in DESTINATION_TYPES or ctype in BOOL_FLAG_TYPES:
        return ctype
    if ctype in MOVEMENT_MODE_TYPES:
        modes = cond.get("modes")
        return f"{ctype}(modes={modes})"
    if ctype in ARMY_GROUP_COUNT_TYPES:
        return f"{ctype}(count={cond.get('count')})"
    if ctype in SCOUT_TYPES:
        if ctype == "scout_result_is":
            return f"{ctype}(result={cond.get('result')})"
        return ctype
    return ctype or "unknown_condition"


def list_satisfied_wake_conditions(
    event: Optional[Dict[str, Any]],
    snapshot: Dict[str, Any],
) -> List[str]:
    """Labels of conditions that are true under the current snapshot."""
    if not event or not isinstance(event, dict):
        return []
    conditions = event.get("conditions") or []
    if not isinstance(conditions, list):
        return []
    out: List[str] = []
    for cond in conditions:
        if _evaluate_condition(cond, snapshot):
            out.append(format_wake_condition(cond))
    return out


def build_trigger_hint(
    *,
    reason: str,
    event: Optional[Dict[str, Any]] = None,
    fired_conditions: Optional[Sequence[str]] = None,
) -> str:
    """Explain why this Commander cycle was scheduled.

    ``fired_conditions`` should be the predicates that are true at fire time
    (or a synthetic label such as ``runtime_deadline_fuse``).
    """
    lines = [
        "[Runtime Decision Trigger]",
        f"reason={reason}",
    ]
    fired = [str(item).strip() for item in (fired_conditions or []) if str(item).strip()]
    if fired:
        lines.append(f"woken_by={'; '.join(fired)}")
    elif reason == "wake_fallback_timeout":
        lines.append("woken_by=runtime_deadline_fuse")
    if event:
        logic = str(event.get("logic") or "any").strip().lower() or "any"
        lines.append(f"armed_logic={logic}")
        lines.append(f"armed_event={compact_wake_event(event)}")
    lines.append(
        "Use woken_by as the completed checkpoint that caused this wake; "
        "reassess from the current observation and strategy."
    )
    return "\n".join(lines)


def build_wake_snapshot(
    *,
    time_seconds: float,
    supply_used: int = 0,
    supply_cap: int = 0,
    own_unit_type_counts: Optional[Dict[str, int]] = None,
    army_groups: Optional[Sequence[Dict[str, Any]]] = None,
    army_summary: Optional[Dict[str, Any]] = None,
    available_zones: Optional[Sequence[Dict[str, Any]]] = None,
    last_scout_result: str = "",
    last_scout_result_time: Optional[float] = None,
    scan_ready: bool = False,
    cleanup_hint_present: bool = False,
    wake_armed_at: Optional[float] = None,
    baseline_objective_status: str = "",
) -> Dict[str, Any]:
    """Assemble a cheap evaluation snapshot (no observation text)."""
    groups = [g for g in (army_groups or []) if isinstance(g, dict)]
    zones = [z for z in (available_zones or []) if isinstance(z, dict)]
    summary = army_summary if isinstance(army_summary, dict) else {}
    modes = _modes_from_summary_or_groups(summary, groups)
    main = _main_group(groups)
    objective_status, destination_reached = _main_objective(main, zones)
    scout_result = normalize_scout_result(last_scout_result)
    return {
        "time_seconds": float(time_seconds),
        "supply_used": int(supply_used),
        "supply_cap": int(supply_cap),
        "own_unit_type_counts": dict(own_unit_type_counts or {}),
        "army_group_count": len(groups),
        "movement_modes": modes,
        "objective_status": objective_status,
        "destination_reached": destination_reached,
        "last_scout_result": scout_result,
        "last_scout_result_time": (
            float(last_scout_result_time)
            if last_scout_result_time is not None
            else None
        ),
        "scan_ready": bool(scan_ready),
        "cleanup_hint_present": bool(cleanup_hint_present),
        "wake_armed_at": (
            float(wake_armed_at) if wake_armed_at is not None else None
        ),
        "baseline_objective_status": str(baseline_objective_status or ""),
    }


def _modes_from_summary_or_groups(
    summary: Dict[str, Any], groups: Sequence[Dict[str, Any]]
) -> List[str]:
    modes: List[str] = []
    for command in summary.get("commands") or []:
        if not isinstance(command, dict):
            continue
        mode = str(command.get("movement_mode") or "").strip()
        if mode:
            modes.append(mode)
    if modes:
        return modes
    for group in groups:
        command = group.get("current_command") or {}
        if not isinstance(command, dict):
            continue
        mode = str(command.get("movement_mode") or "").strip()
        if mode:
            modes.append(mode)
    return modes


def _main_group(groups: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for group in groups:
        if group.get("role") == "main_force":
            return group
    return groups[0] if groups else None


def _main_objective(
    main: Optional[Dict[str, Any]], zones: Sequence[Dict[str, Any]]
) -> Tuple[str, bool]:
    if not main:
        return "none", False
    command = main.get("current_command") or {}
    if not isinstance(command, dict):
        return "none", False
    destination = str(command.get("destination_zone_id") or "").strip()
    if not destination:
        return "none", False
    nearest = str(main.get("nearest_zone_id") or "").strip()
    destination_reached = bool(nearest and nearest == destination)
    zone_by_id = {
        str(zone.get("zone_id")): zone
        for zone in zones
        if zone.get("zone_id") is not None
    }
    zone = zone_by_id.get(destination)
    if not zone:
        return "unknown", destination_reached
    if _zone_has_enemy_evidence(zone):
        return "enemy_present", destination_reached
    if str(zone.get("vision_state") or "") == "visible":
        return "confirmed_clear", destination_reached
    if destination_reached:
        return "reached_unconfirmed", destination_reached
    return "en_route_unconfirmed", destination_reached


def _zone_has_enemy_evidence(zone: Dict[str, Any]) -> bool:
    visible_contents = zone.get("visible_enemy_contents") or {}
    remembered = zone.get("last_seen_enemy_contents") or {}
    return bool(
        (isinstance(visible_contents, dict) and visible_contents)
        or (isinstance(remembered, dict) and remembered)
        or int(zone.get("visible_enemy_units", 0) or 0) > 0
        or int(zone.get("remembered_enemy_units", 0) or 0) > 0
        or float(zone.get("visible_enemy_power", 0.0) or 0.0) > 0.0
        or float(zone.get("remembered_enemy_power", 0.0) or 0.0) > 0.0
        or float(zone.get("enemy_static_power", 0.0) or 0.0) > 0.0
    )


def _unit_count(counts: Dict[str, int], unit: str) -> int:
    want = unit.strip().lower()
    total = 0
    for name, value in counts.items():
        if str(name).strip().lower() == want:
            try:
                total += int(value)
            except (TypeError, ValueError):
                continue
    return total


def _evaluate_condition(cond: Any, snapshot: Dict[str, Any]) -> bool:
    if not isinstance(cond, dict):
        return False
    ctype = str(cond.get("type") or "")
    ctype = PREDICATE_TYPE_ALIASES.get(ctype, ctype)
    if ctype == "unit_count_at_least":
        return _unit_count(
            snapshot.get("own_unit_type_counts") or {},
            str(cond.get("unit") or ""),
        ) >= int(cond.get("count") or 0)
    if ctype == "unit_count_less_than":
        return _unit_count(
            snapshot.get("own_unit_type_counts") or {},
            str(cond.get("unit") or ""),
        ) < int(cond.get("count") or 0)
    if ctype == "army_group_count_at_least":
        return int(snapshot.get("army_group_count") or 0) >= int(cond.get("count") or 0)
    if ctype == "army_group_count_less_than":
        return int(snapshot.get("army_group_count") or 0) < int(cond.get("count") or 0)
    if ctype == "movement_mode_in":
        modes = set(snapshot.get("movement_modes") or [])
        wanted = set(cond.get("modes") or [])
        return bool(modes & wanted)
    if ctype == "movement_mode_not_in":
        modes = set(snapshot.get("movement_modes") or [])
        forbidden = set(cond.get("modes") or [])
        if not modes:
            return True
        return not bool(modes & forbidden)
    if ctype == "objective_status_is":
        return str(snapshot.get("objective_status") or "") == str(
            cond.get("status") or ""
        )
    if ctype == "objective_status_became":
        wanted = str(cond.get("status") or "")
        current = str(snapshot.get("objective_status") or "")
        baseline = str(snapshot.get("baseline_objective_status") or "")
        return bool(wanted) and current == wanted and current != baseline
    if ctype == "destination_reached":
        return bool(snapshot.get("destination_reached"))
    if ctype == "scout_result_is":
        current = normalize_scout_result(snapshot.get("last_scout_result"))
        wanted = normalize_scout_result(cond.get("result"))
        return bool(wanted) and current == wanted
    if ctype == "scout_just_finished":
        current = normalize_scout_result(snapshot.get("last_scout_result"))
        if current not in SCOUT_TERMINAL_RESULTS:
            return False
        result_time = snapshot.get("last_scout_result_time")
        armed_at = snapshot.get("wake_armed_at")
        if result_time is None or armed_at is None:
            return False
        # Strictly after this wake was armed — ignores stale prior scout results.
        return float(result_time) > float(armed_at)
    if ctype == "scan_ready":
        return bool(snapshot.get("scan_ready"))
    if ctype == "cleanup_hint_present":
        return bool(snapshot.get("cleanup_hint_present"))
    if ctype == "game_time_at_least":
        return float(snapshot.get("time_seconds") or 0.0) >= float(
            cond.get("seconds") or 0.0
        )
    if ctype == "supply_left_at_most":
        left = int(snapshot.get("supply_cap") or 0) - int(
            snapshot.get("supply_used") or 0
        )
        return left <= int(cond.get("count") or 0)
    return False
