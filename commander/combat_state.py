import json
import logging
import os
from dataclasses import asdict, is_dataclass
import re
import time as _wall_time
from typing import Any, Dict, List, Optional

from sc2.ids.unit_typeid import UnitTypeId

from llm.caller import call_openai
from commander.combat_policy import (
    ArmyControlPolicy,
    ArmyGroupCommand,
    parse_army_control_policy,
)


logger = logging.getLogger("commander.combat_state")

FALLBACK_ARMY_CONTROL_POLICY = ArmyControlPolicy(commands=[])

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _env_truthy(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def build_strategy_directive(act: Any) -> str:
    """Build Army context from the strategy selected by the Strategy Coordinator and its live directive."""
    ai = getattr(act, "ai", None)
    strategy = str(getattr(ai, "selected_strategy", "") or "").strip()
    strategy_description = str(
        getattr(ai, "strategy_description", "") or ""
    ).strip()
    army_directive = str(
        getattr(ai, "current_army_directive", "") or ""
    ).strip()

    return chr(10).join(
        [
            f"strategy={strategy or 'unknown'}",
            "strategy_description="
            f"{strategy_description or 'No Strategy Coordinator strategy description available.'}",
            "army_directive="
            f"{army_directive or 'No Army directive emitted by Strategy Coordinator yet.'}",
        ]
    )


ENEMY_MAIN_ZONE_ID = "zone_15"
ENDGAME_CLEANUP_REMAINING_SECONDS = 5 * 60


def _main_force_nearest_zone_id(state: Dict[str, Any]) -> str:
    for group in state.get("army_groups", []) or []:
        if str(group.get("role") or "") == "main_force":
            return str(group.get("nearest_zone_id") or "").strip()
    return str(state.get("army_nearest_zone") or "").strip()


def _update_peak_known_enemy_bases(act: Any, state: Dict[str, Any]) -> int:
    known = int(state.get("known_enemy_bases", 0) or 0)
    peak = int(getattr(act, "_peak_known_enemy_bases", 0) or 0)
    if known > peak:
        setattr(act, "_peak_known_enemy_bases", known)
        peak = known
    return peak


def _update_main_force_reached_enemy_main(act: Any, state: Dict[str, Any]) -> bool:
    """True once a combat force has reached the enemy main.

    A spatially broad main-force group can have its center remain in the enemy
    natural while many of its units are already fighting in the enemy main.
    Use zone-local combat presence as additional evidence so group clustering
    cannot prevent the endgame cleanup transition.
    """
    if bool(getattr(act, "_main_force_reached_enemy_main", False)):
        return True
    if _main_force_nearest_zone_id(state) == ENEMY_MAIN_ZONE_ID:
        setattr(act, "_main_force_reached_enemy_main", True)
        return True
    for zone in state.get("available_zones", []) or []:
        zone_id = str(zone.get("zone_id") or "").strip()
        zone_role = str(zone.get("zone_role") or "").strip()
        if zone_id != ENEMY_MAIN_ZONE_ID and zone_role != "enemy_main":
            continue
        if float(zone.get("own_combat_power", 0.0) or 0.0) > 0.0:
            setattr(act, "_main_force_reached_enemy_main", True)
            return True
    return False


def should_recommend_cleanup(
    state: Dict[str, Any],
    *,
    peak_known_enemy_bases: int = 0,
    main_force_reached_enemy_main: bool = False,
) -> bool:
    """True only after the main force reached enemy main and bases are cleared.

    Early game often has known_enemy_bases=0 simply because nothing has been
    scouted yet; that must not trigger cleanup / search_and_destroy.
    Cleanup also must not start before the main force has been to zone_15.
    """
    if int(state.get("controlled_combat_units", 0) or 0) <= 0:
        return False
    if not bool(main_force_reached_enemy_main):
        return False
    if int(peak_known_enemy_bases or 0) < 1:
        return False
    known_enemy_bases = int(state.get("known_enemy_bases", 0) or 0)
    visible_enemy_power = float(state.get("visible_enemy_power", 0.0) or 0.0)
    return known_enemy_bases <= 0 and visible_enemy_power <= 0.0


def should_recommend_endgame_cleanup(
    state: Dict[str, Any],
    *,
    game_time_seconds: float,
    game_time_limit_seconds: float,
) -> bool:
    """True in the final minutes when any combat units remain to sweep the map."""
    if int(state.get("controlled_combat_units", 0) or 0) <= 0:
        return False
    limit = float(game_time_limit_seconds or 0.0)
    if limit <= 0:
        return False
    remaining = limit - float(game_time_seconds or 0.0)
    return remaining <= float(ENDGAME_CLEANUP_REMAINING_SECONDS)


def _game_time_limit_seconds(act: Any) -> float:
    ai = getattr(act, "ai", None)
    limit = getattr(ai, "game_time_limit_seconds", None) if ai is not None else None
    try:
        value = float(limit)
    except (TypeError, ValueError):
        value = 0.0
    if value > 0:
        return value
    try:
        from commander.match_defaults import DEFAULT_GAME_TIME_LIMIT_SECONDS

        return float(DEFAULT_GAME_TIME_LIMIT_SECONDS)
    except Exception:
        return 0.0


def build_cleanup_runtime_hint(
    act: Any,
    state: Dict[str, Any],
    *,
    game_time_seconds: Optional[float] = None,
) -> str:
    """Inject an explicit map-clear cue only while cleanup conditions hold now.

    Program-side situation check only; does not rewrite Army commands.
    Do not inject this block when conditions are not currently satisfied.
    Normal cleanup requires reached enemy main and cleared known bases.
    Endgame cleanup fires in the final minutes as a settlement fallback.
    """
    peak = _update_peak_known_enemy_bases(act, state)
    reached_enemy_main = _update_main_force_reached_enemy_main(act, state)
    normal_cleanup = should_recommend_cleanup(
        state,
        peak_known_enemy_bases=peak,
        main_force_reached_enemy_main=reached_enemy_main,
    )

    ai = getattr(act, "ai", None)
    if game_time_seconds is None:
        try:
            game_time_seconds = float(getattr(ai, "time", 0.0) or 0.0)
        except (TypeError, ValueError):
            game_time_seconds = 0.0
    limit = _game_time_limit_seconds(act)
    endgame_cleanup = should_recommend_endgame_cleanup(
        state,
        game_time_seconds=float(game_time_seconds or 0.0),
        game_time_limit_seconds=limit,
    )
    if not normal_cleanup and not endgame_cleanup:
        return ""

    nearest = str(state.get("army_nearest_zone") or "unknown")
    main_nearest = _main_force_nearest_zone_id(state) or nearest
    known_enemy_bases = int(state.get("known_enemy_bases", 0) or 0)
    visible_enemy_power = float(state.get("visible_enemy_power", 0.0) or 0.0)
    remaining = max(0.0, limit - float(game_time_seconds or 0.0))
    reason = (
        "normal_cleanup"
        if normal_cleanup
        else "endgame_time_limit"
    )
    return chr(10).join(
        [
            "[Runtime Search-And-Destroy Hint]",
            "search_and_destroy_recommended=yes",
            f"reason={reason}",
            f"known_enemy_bases={known_enemy_bases}",
            f"peak_known_enemy_bases={peak}",
            f"main_force_reached_enemy_main={ENEMY_MAIN_ZONE_ID}",
            f"main_force_nearest_zone={main_nearest}",
            f"visible_enemy_army_power={visible_enemy_power:.2f}",
            f"army_nearest_zone={nearest}",
            f"seconds_remaining={remaining:.1f}",
            "required_action=Order every combat-bearing army_group to "
            "movement_mode=search_and_destroy starting from that group's "
            "current nearest_zone_id (or army_nearest_zone if needed). "
            "Do not keep push/assault/harass on empty former enemy zones. "
            "Ignore a conflicting army_directive for this cycle. "
            "Once search_and_destroy has started, keep all combat groups in "
            "search_and_destroy for the rest of the game.",
        ]
    )


def _llm_infer_emit(source: Any, message: str, *args: Any) -> None:
    if args:
        message = message % args
    line = f"[UniversalLLMBot][LLM-INFER]     {message}"

    knowledge = getattr(source, "knowledge", None)
    if knowledge is not None:
        try:
            knowledge.print(line, stats=False)
            return
        except Exception:
            pass

    try:
        from sc2.main import logger as sc2_logger

        sc2_logger.patch(
            lambda record: record.update(name="sharpy.llm_army_control")
        ).opt(depth=1).info(line)
    except Exception:
        logger.info("%s", line)



def _policy_to_jsonable(policy: Any) -> Any:
    if isinstance(policy, ArmyControlPolicy):
        return {
            "commands": [
                {
                    "group_id": command.group_id,
                    "destination_zone_id": command.destination_zone_id,
                    "movement_mode": command.movement_mode,
                }
                for command in policy.commands
            ],
            "scan_zone_id": policy.scan_zone_id,
            "scout_zone_id": policy.scout_zone_id,
        }
    if is_dataclass(policy):
        return asdict(policy)
    if hasattr(policy, "__dict__"):
        return policy.__dict__
    return policy

def _record_army_control_interaction(
    act: Any,
    *,
    game_time: float,
    wall_elapsed_seconds: float,
    model_key: str,
    messages: list,
    observation: str,
    state: Dict[str, Any],
    response: str,
    policy: ArmyControlPolicy,
    issues: list,
    error: Optional[Exception] = None,
    observation_full: Optional[Dict[str, Any]] = None,
    observation_view: Optional[Dict[str, Any]] = None,
) -> None:
    recorder = getattr(
        getattr(act, "ai", None), "llm_observation_recorder", None
    )
    append_func = getattr(recorder, "record_llm_interaction", None)
    if append_func is None:
        return
    try:
        append_func(
            {
                "game_time": round(game_time, 2),
                "trigger_reason": "army_control_agent_poll",
                "wall_elapsed_seconds": round(wall_elapsed_seconds, 3),
                "strategy_coordinator_strategy": getattr(
                    act.ai, "selected_strategy", ""
                ),
                "strategy_coordinator_army_directive": getattr(
                    act.ai, "current_army_directive", ""
                ),
                "observation_text": observation,
                "observation_full": observation_full,
                "observation_view": observation_view,
                "observation_view_type": "combat",
                "army_control_agent_policy": {
                    "messages_sent": [dict(message) for message in messages],
                    "raw_response": response,
                    "command_issues": list(issues),
                    "parsed": _policy_to_jsonable(policy),
                    "wall_elapsed_seconds": round(
                        wall_elapsed_seconds, 3
                    ),
                    "error": repr(error) if error is not None else None,
                },
            }
        )
    except Exception as exc:
        logger.warning("Unable to record Army Control interaction: %s", exc)


def _persist_combat_execution(
    act: Any,
    policy: ArmyControlPolicy,
    issues: list,
    now: float,
    *,
    applied: bool,
    status: str,
) -> None:
    """Expose only the policy that actually remains in force."""
    ai = getattr(act, "ai", None)
    if ai is None:
        return
    previous = dict(getattr(ai, "llm_combat_execution_state", {}) or {})
    if applied or not previous.get("last_policy"):
        previous["last_policy"] = _policy_to_jsonable(policy)
        previous["policy_applied_game_time"] = now
    previous["last_command_issues"] = [
        str(issue) for issue in issues if str(issue).strip()
    ]
    previous["status"] = status
    ai.llm_combat_execution_state = previous

    append_event = getattr(ai, "_append_execution_event", None)
    if append_event is None:
        append_event = getattr(ai, "_append_top_execution_event", None)
    if callable(append_event):
        policy_data = _policy_to_jsonable(policy)
        if not isinstance(policy_data, dict):
            policy_data = {}
        append_event(
            "combat",
            {
                "game_time_seconds": round(float(now), 1),
                "status": status,
                "applied": bool(applied),
                "commands": list(policy_data.get("commands") or []),
                "scan_zone_id": policy_data.get("scan_zone_id"),
                "scout_zone_id": policy_data.get("scout_zone_id"),
                "issues": list(previous["last_command_issues"]),
            },
        )


def _validate_policy_for_state(
    policy: ArmyControlPolicy,
    state: Dict[str, Any],
) -> None:
    zones = {
        zone["zone_id"]: zone
        for zone in state.get("available_zones", [])
    }
    groups = {
        group["group_id"]: group
        for group in state.get("army_groups", [])
    }
    active_ids = set(groups)
    commanded_ids = {command.group_id for command in policy.commands}
    if commanded_ids != active_ids:
        raise ValueError(
            "commands must cover every army_group exactly once; "
            f"expected {sorted(active_ids)}, got {sorted(commanded_ids)}"
        )
    for field_name, zone_id in (
        ("scan_zone_id", policy.scan_zone_id),
        ("scout_zone_id", policy.scout_zone_id),
    ):
        if zone_id is not None and zone_id not in zones:
            raise ValueError(
                f"{field_name} {zone_id!r} is absent from available_zones"
            )
    for command in policy.commands:
        group = groups.get(command.group_id)
        if group is None:
            raise ValueError(
                f"group_id {command.group_id!r} is absent from army_groups"
            )
        selected = zones.get(command.destination_zone_id)
        if selected is None:
            raise ValueError(
                f"destination_zone_id {command.destination_zone_id!r} "
                "is absent from available_zones"
            )
        owner = selected["owner"]
        if (
            command.movement_mode == "regroup"
            and not _safe_regroup_zone(selected, group)
        ):
            raise ValueError(
                "regroup requires a safe own or neutral zone"
            )
        if (
            command.movement_mode
            in {"defensive_retreat", "panic_retreat"}
            and owner != "own"
        ):
            raise ValueError(
                f"{command.movement_mode} requires an own zone, "
                f"got {owner!r}"
            )

def _safe_regroup_zone(
    zone: Dict[str, Any],
    group: Optional[Dict[str, Any]] = None,
) -> bool:
    if (
        group
        and str(group.get("nearest_zone_id") or "")
        == str(zone.get("zone_id") or "")
        and (
            int(group.get("nearby_enemy_count", 0) or 0) > 0
            or float(group.get("nearby_enemy_power", 0.0) or 0.0) > 0.0
        )
    ):
        return False
    if zone.get("owner") == "own":
        return not zone.get("under_attack", False)
    return (
        zone.get("owner") == "neutral"
        and not zone.get("under_attack", False)
        and int(zone.get("known_enemy_units", 0) or 0) == 0
        and float(zone.get("known_enemy_power", 0.0) or 0.0) <= 0
        and float(zone.get("enemy_static_power", 0.0) or 0.0) <= 0
    )


def _cleanup_hint_allows_search_and_destroy(cleanup_hint: str) -> bool:
    return "[Runtime Search-And-Destroy Hint]" in str(cleanup_hint or "")


def _nearest_safe_regroup_zone_id(
    zones: Dict[str, Dict[str, Any]],
    *,
    preferred_zone_id: str = "",
    group: Optional[Dict[str, Any]] = None,
    prefer_own: bool = False,
) -> Optional[str]:
    preferred = str(preferred_zone_id or "").strip()
    if preferred:
        zone = zones.get(preferred)
        if zone is not None and _safe_regroup_zone(zone, group):
            return preferred
    candidates = [
        zone
        for zone in zones.values()
        if _safe_regroup_zone(zone, group)
    ]
    if not candidates:
        return None
    return str(
        min(
            candidates,
            key=lambda item: (
                0 if prefer_own and item.get("owner") == "own" else 1,
                float(item.get("distance_from_army", float("inf"))),
            ),
        ).get("zone_id")
        or ""
    ) or None


def _replace_command(
    command: ArmyGroupCommand,
    destination_zone_id: str,
    movement_mode: str,
) -> ArmyGroupCommand:
    return parse_army_control_policy(
        {
            "commands": [
                {
                    "group_id": command.group_id,
                    "destination_zone_id": destination_zone_id,
                    "movement_mode": movement_mode,
                }
            ]
        }
    ).commands[0]


def _make_group_command(
    group_id: str,
    destination_zone_id: str,
    movement_mode: str,
) -> ArmyGroupCommand:
    return parse_army_control_policy(
        {
            "commands": [
                {
                    "group_id": group_id,
                    "destination_zone_id": destination_zone_id,
                    "movement_mode": movement_mode,
                }
            ]
        }
    ).commands[0]


def _main_group_id_from_state(groups: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for group_id, group in groups.items():
        if group.get("role") == "main_force":
            return str(group_id)
    if "group_0" in groups:
        return "group_0"
    if groups:
        return next(iter(groups))
    return None


def _parse_policy_commands_individually(
    text: str,
) -> tuple[ArmyControlPolicy, list]:
    _require_decision_explanation(text)
    data = _json_from_llm_response(text)
    allowed_fields = {"commands", "scan_zone_id", "scout_zone_id"}
    if (
        not set(data).issubset(allowed_fields)
        or not isinstance(data.get("commands"), list)
    ):
        raise ValueError(
            "army control response must contain commands and optional scan_zone_id/scout_zone_id"
        )
    requests = parse_army_control_policy(
        {
            "commands": [],
            "scan_zone_id": data.get("scan_zone_id"),
            "scout_zone_id": data.get("scout_zone_id"),
        }
    )

    parsed = []
    issues = []
    seen_groups = set()
    for index, raw_command in enumerate(data["commands"][:3]):
        try:
            command = parse_army_control_policy(
                {"commands": [raw_command]}
            ).commands[0]
        except Exception as exc:
            issues.append(f"command #{index + 1} rejected: {exc}")
            continue
        if command.group_id in seen_groups:
            issues.append(
                f"command #{index + 1} rejected: duplicate "
                f"group_id {command.group_id}"
            )
            continue
        seen_groups.add(command.group_id)
        parsed.append(command)
    if len(data["commands"]) > 3:
        issues.append("commands after the first three were ignored")
    return ArmyControlPolicy(parsed, requests.scan_zone_id, requests.scout_zone_id), issues


def _fill_missing_group_commands(
    commands: List[ArmyGroupCommand],
    groups: Dict[str, Dict[str, Any]],
    zones: Dict[str, Dict[str, Any]],
    state: Dict[str, Any],
) -> tuple[List[ArmyGroupCommand], list]:
    """Ensure every active army_group has a command when possible."""
    by_id = {command.group_id: command for command in commands}
    issues: list = []
    main_id = _main_group_id_from_state(groups)
    active_ids = list(groups)

    if main_id and main_id not in by_id and by_id:
        source = next(iter(by_id.values()))
        by_id[main_id] = _make_group_command(
            main_id,
            source.destination_zone_id,
            source.movement_mode,
        )
        issues.append(
            f"{main_id} filled: copied from {source.group_id}"
        )

    main_command = by_id.get(main_id) if main_id else None
    for group_id in active_ids:
        if group_id in by_id:
            continue
        group = groups.get(group_id)
        if main_command is not None:
            by_id[group_id] = _make_group_command(
                group_id,
                main_command.destination_zone_id,
                main_command.movement_mode,
            )
            issues.append(
                f"{group_id} filled: copied from {main_id}"
            )
            continue
        preferred = str(
            (group or {}).get("nearest_zone_id")
            or state.get("army_nearest_zone")
            or ""
        )
        destination = _nearest_safe_regroup_zone_id(
            zones,
            preferred_zone_id=preferred,
            group=group,
            prefer_own=True,
        )
        if destination is None:
            issues.append(
                f"{group_id} could not be filled: no safe regroup zone"
            )
            continue
        by_id[group_id] = _make_group_command(
            group_id,
            destination,
            "regroup",
        )
        issues.append(
            f"{group_id} filled: regroup at {destination}"
        )

    ordered = [
        by_id[group_id]
        for group_id in sorted(
            by_id,
            key=lambda value: int(str(value).removeprefix("group_") or "0"),
        )
        if group_id in groups
    ]
    return ordered, issues


def _unify_search_and_destroy_commands(
    commands: List[ArmyGroupCommand],
    groups: Dict[str, Dict[str, Any]],
) -> tuple[List[ArmyGroupCommand], list]:
    search = next(
        (
            command
            for command in commands
            if command.movement_mode == "search_and_destroy"
        ),
        None,
    )
    if search is None or not groups:
        return commands, []

    issues = []
    by_id = {command.group_id: command for command in commands}
    for group_id in groups:
        current = by_id.get(group_id)
        if (
            current is not None
            and current.movement_mode == "search_and_destroy"
            and current.destination_zone_id == search.destination_zone_id
        ):
            continue
        by_id[group_id] = _make_group_command(
            group_id,
            search.destination_zone_id,
            "search_and_destroy",
        )
        issues.append(
            f"{group_id} filled: unified search_and_destroy at "
            f"{search.destination_zone_id}"
        )
    ordered = [
        by_id[group_id]
        for group_id in sorted(
            by_id,
            key=lambda value: int(str(value).removeprefix("group_") or "0"),
        )
        if group_id in groups
    ]
    return ordered, issues


def _repair_policy_for_state(
    policy: ArmyControlPolicy,
    state: Dict[str, Any],
    *,
    cleanup_hint: str = "",
) -> tuple[ArmyControlPolicy, list]:
    groups = {
        group.get("group_id"): group
        for group in state.get("army_groups", [])
    }
    zones = {
        zone.get("zone_id"): zone
        for zone in state.get("available_zones", [])
    }
    allow_search_and_destroy = _cleanup_hint_allows_search_and_destroy(
        cleanup_hint
    )
    repaired = []
    issues = []
    for command in policy.commands:
        group = groups.get(command.group_id)
        if group is None:
            issues.append(
                f"{command.group_id} rejected because it is not active"
            )
            continue
        zone = zones.get(command.destination_zone_id)
        if zone is None:
            issues.append(
                f"{command.group_id} rejected because "
                f"{command.destination_zone_id} is unavailable"
            )
            continue

        candidate = command
        if candidate.movement_mode in {
            "defensive_retreat",
            "panic_retreat",
        } and zone.get("owner") != "own":
            own_candidates = [
                item
                for item in zones.values()
                if item.get("owner") == "own"
                and not item.get("under_attack", False)
            ]
            if not own_candidates:
                issues.append(
                    f"{candidate.group_id} retreat rejected: no safe own zone"
                )
                continue
            destination = min(
                own_candidates,
                key=lambda item: float(
                    item.get("distance_from_army", float("inf"))
                ),
            ).get("zone_id")
            candidate = _replace_command(
                candidate, destination, candidate.movement_mode
            )
            issues.append(
                f"{candidate.group_id} retreat destination repaired "
                f"to {destination}"
            )
            zone = zones[destination]

        if (
            candidate.movement_mode == "search_and_destroy"
            and not allow_search_and_destroy
        ):
            preferred = str(
                (group or {}).get("nearest_zone_id")
                or state.get("army_nearest_zone")
                or ""
            )
            destination = _nearest_safe_regroup_zone_id(
                zones,
                preferred_zone_id=preferred,
                group=group,
            )
            if destination is None:
                issues.append(
                    f"{candidate.group_id} search_and_destroy rejected: "
                    "no Runtime Search-And-Destroy Hint and no safe regroup zone"
                )
                continue
            candidate = _replace_command(
                candidate, destination, "regroup"
            )
            issues.append(
                f"{candidate.group_id} search_and_destroy repaired to "
                f"regroup at {destination} (missing Runtime Search-And-Destroy Hint)"
            )
            zone = zones[destination]

        if (
            candidate.movement_mode == "regroup"
            and not _safe_regroup_zone(zone, group)
        ):
            destination = _nearest_safe_regroup_zone_id(
                zones,
                group=group,
                prefer_own=True,
            )
            if destination is None:
                issues.append(
                    f"{candidate.group_id} regroup rejected: no currently safe gather zone"
                )
                continue
            candidate = _replace_command(candidate, destination, "regroup")
            issues.append(
                f"{candidate.group_id} regroup destination repaired "
                f"to {destination} because the previous target is no longer safe"
            )

        repaired.append(candidate)

    filled, fill_issues = _fill_missing_group_commands(
        repaired,
        groups,
        zones,
        state,
    )
    issues.extend(fill_issues)
    unified, unify_issues = _unify_search_and_destroy_commands(
        filled,
        groups,
    )
    issues.extend(unify_issues)

    return ArmyControlPolicy(
        unified, policy.scan_zone_id, policy.scout_zone_id
    ), issues

def _revalidate_policy_for_state(
    policy: ArmyControlPolicy,
    state: Dict[str, Any],
    *,
    cleanup_hint: str = "",
) -> tuple[ArmyControlPolicy, list]:
    """Recheck a previously accepted policy against the current snapshot."""
    # A search-and-destroy command was already admitted by a runtime hint at
    # its decision boundary. Keep it valid until the next Army decision; an
    # enemy becoming visible during the sweep is not a reason to turn the
    # cached command into regroup between decision cycles.
    effective_cleanup_hint = cleanup_hint
    if any(
        command.movement_mode == "search_and_destroy"
        for command in policy.commands
    ):
        effective_cleanup_hint = "[Runtime Search-And-Destroy Hint]"
    repaired, issues = _repair_policy_for_state(
        policy,
        state,
        cleanup_hint=effective_cleanup_hint,
    )
    _validate_policy_for_state(repaired, state)
    if policy.commands and not repaired.commands:
        raise ValueError("all cached army commands are invalid in the current state")
    return repaired, issues


def _fallback_policy_for_state(state: Dict[str, Any]) -> ArmyControlPolicy:
    return FALLBACK_ARMY_CONTROL_POLICY

def _print_llm_request(
    source: Any,
    model_key: str,
    now: float,
) -> None:
    _llm_infer_emit(
        source,
        "calling Army Planner (policy) at t=%.1fs model=%s...",
        now,
        model_key,
    )


def _print_llm_result(
    source: Any,
    now: float,
    observation: str,
    response: str,
    policy: ArmyControlPolicy,
    error: Optional[Exception] = None,
) -> None:
    _llm_infer_emit(source, "Army Planner done at t=%.1fs.", now)
    _llm_infer_emit(
        source,
        "Army Planner observation:\n%s",
        str(observation or "").strip(),
    )
    _llm_infer_emit(
        source,
        "Army Planner output: %r",
        (response or "").strip(),
    )
    if error is not None:
        _llm_infer_emit(source, "Army Planner parse error: %s", error)
    _llm_infer_emit(
        source,
        "Army Planner parsed policy: %s",
        json.dumps(
            _policy_to_jsonable(policy),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def _json_from_llm_response(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("empty LLM army control response")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        match = _JSON_OBJECT_RE.search(cleaned)
        if not match:
            raise ValueError("LLM army control response does not contain JSON")
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("LLM army control response must be a JSON object")
    return data


def _require_decision_explanation(text: str) -> str:
    """Return the natural-language paragraph before JSON or reject it."""
    cleaned = str(text or "").strip()
    match = _JSON_OBJECT_RE.search(cleaned)
    if not match:
        raise ValueError("LLM army control response does not contain JSON")
    explanation = cleaned[: match.start()].strip()
    explanation = re.sub(
        r"```(?:json)?\s*$",
        "",
        explanation,
        flags=re.IGNORECASE,
    ).strip()
    if not explanation:
        raise ValueError(
            "army control response must include one concise plain-text "
            "decision explanation before JSON"
        )
    return explanation


def parse_llm_army_control_response(text: str) -> ArmyControlPolicy:
    _require_decision_explanation(text)
    return parse_army_control_policy(_json_from_llm_response(text))


class BlockingLLMArmyControlProvider:
    def __init__(
        self,
        model_key: Optional[str] = None,
        decision_interval_seconds: float = 20.0,
        retry_interval_seconds: float = 2.0,
    ):
        self.model_key = (
            model_key
            or os.environ.get("LLM_ARMY_CONTROL_MODEL")
            or "DeepSeek-V4-pro"
        )
        self.decision_interval_seconds = decision_interval_seconds
        self.retry_interval_seconds = retry_interval_seconds
        self.last_valid_policy: Optional[ArmyControlPolicy] = None
        self.next_decision_time = 0.0
        self.last_observation_state: Optional[Dict[str, Any]] = None
        self.verbose = _env_truthy("LLM_ARMY_CONTROL_VERBOSE", True)

    def get_policy(self, act: Any) -> ArmyControlPolicy:
        now = float(getattr(act.ai, "time", 0.0))
        state = collect_army_control_state(act)
        cleanup_hint = build_cleanup_runtime_hint(
            act,
            state,
            game_time_seconds=now,
        )

        if self.last_valid_policy is not None and now < self.next_decision_time:
            try:
                cached_policy, cached_issues = _revalidate_policy_for_state(
                    self.last_valid_policy,
                    state,
                    cleanup_hint=cleanup_hint,
                )
                self.last_valid_policy = cached_policy
                if cached_issues:
                    logger.warning(
                        "Cached Army policy repaired for current state: %s",
                        "; ".join(cached_issues),
                    )
                return cached_policy
            except Exception as exc:
                logger.warning(
                    "Cached Army policy is stale; requesting a fresh decision. err=%s",
                    exc,
                )
                self.last_valid_policy = None
                self.next_decision_time = now

        observation = ""
        observation_full = None
        observation_view = None
        recorder = getattr(act.ai, "llm_observation_recorder", None)
        capture = getattr(recorder, "capture_observation_bundle", None)
        try:
            if not callable(capture):
                raise RuntimeError("unified Observation API unavailable")
            observation, observation_full, observation_view = capture(
                "combat", army_state=state
            )
        except Exception as exc:
            # Do not rebuild the removed legacy global + Army observation.
            # A short unavailable marker lets the normal retry/fallback policy
            # protect the game without silently switching schemas.
            logger.warning("Unable to build unified Army View: %s", exc)
            observation = "(unified Army View unavailable)"
            observation_full = None
            observation_view = None
        strategy_directive = build_strategy_directive(act)
        messages = self._build_messages(
            observation,
            strategy_directive,
            cleanup_hint=cleanup_hint,
        )
        if cleanup_hint and self.verbose:
            _llm_infer_emit(
                act,
                "Army Planner search_and_destroy hint active "
                "(known_enemy_bases=%s, visible_enemy_power=%.2f).",
                int(state.get("known_enemy_bases", 0) or 0),
                float(state.get("visible_enemy_power", 0.0) or 0.0),
            )
        if self.verbose:
            _print_llm_request(act, self.model_key, now)

        request_started = _wall_time.monotonic()
        response = call_openai(
            messages=messages,
            model_key=self.model_key,
            temperature=0.5,
        )

        issues = []
        try:
            parsed_policy, issues = _parse_policy_commands_individually(
                response
            )
            policy, repair_issues = _repair_policy_for_state(
                parsed_policy,
                state,
                cleanup_hint=cleanup_hint,
            )
            issues.extend(repair_issues)
            _validate_policy_for_state(policy, state)
            if not policy.commands and issues:
                raise ValueError("; ".join(issues))
            if issues:
                logger.warning(
                    "Army control commands repaired/rejected: %s",
                    "; ".join(issues),
                )
                if self.verbose:
                    _llm_infer_emit(
                        act,
                        "Army Planner command repairs: %s",
                        "; ".join(issues),
                    )
        except Exception as exc:
            logger.warning(
                "Invalid LLM army control policy; using fallback/last valid. "
                "err=%s response=%r",
                exc,
                response,
            )
            self.next_decision_time = now + self.retry_interval_seconds
            policy = _fallback_policy_for_state(state)
            reused_last_valid = False
            if self.last_valid_policy is not None:
                try:
                    policy, fallback_repairs = _revalidate_policy_for_state(
                        self.last_valid_policy,
                        state,
                        cleanup_hint=cleanup_hint,
                    )
                    failure_issues = list(issues) + list(fallback_repairs)
                    reused_last_valid = True
                    self.last_valid_policy = policy
                except Exception as cached_exc:
                    failure_issues = list(issues)
                    failure_issues.append(
                        f"cached policy rejected: {cached_exc}"
                    )
                    self.last_valid_policy = None
            else:
                failure_issues = list(issues)
            if self.verbose:
                _print_llm_result(
                    act,
                    now,
                    observation,
                    response,
                    policy,
                    exc,
                )
            if str(exc) not in failure_issues:
                failure_issues.append(str(exc))
            _persist_combat_execution(
                act, policy, failure_issues, now,
                applied=not reused_last_valid,
                status="fallback_active",
            )
            _record_army_control_interaction(
                act,
                game_time=now,
                wall_elapsed_seconds=(
                    _wall_time.monotonic() - request_started
                ),
                model_key=self.model_key,
                messages=messages,
                observation=observation,
                state=state,
                response=response,
                policy=policy,
                issues=failure_issues,
                error=exc,
                observation_full=observation_full,
                observation_view=observation_view,
            )
            return policy

        self.last_valid_policy = policy
        self.last_observation_state = _state_snapshot(state)
        self.next_decision_time = now + self.decision_interval_seconds
        _persist_combat_execution(
            act, policy, issues, now,
            applied=True,
            status="executing_with_repairs" if issues else "executing",
        )
        if self.verbose:
            _print_llm_result(
                act, now, observation, response, policy
            )
        _record_army_control_interaction(
            act,
            game_time=now,
            wall_elapsed_seconds=_wall_time.monotonic() - request_started,
            model_key=self.model_key,
            messages=messages,
            observation=observation,
            state=state,
            response=response,
            policy=policy,
            issues=issues,
            observation_full=observation_full,
            observation_view=observation_view,
        )
        return policy

    def _build_messages(
        self,
        observation: str,
        strategy_directive: str = "",
        cleanup_hint: str = "",
    ) -> list:

        system_msg = """You are the Army Planner for a StarCraft II bot that may use any race, strategy, or unit composition.
Treat the full strategy as authoritative and the current army_directive as high-level guidance until the next Coordinator update; independently convert both and the latest Army View into current Army Control commands.
You control each army_group's destination zone and movement or combat mode, one Scanner Sweep request, and at most one SCV zone-scout request.
You do not control production, economy, general worker allocation, upgrades, expansions, unit tags, coordinates, or individual combat units.

Output format:
1. Begin with exactly one concise, natural plain-text paragraph explaining the current evidence and why the selected army actions are appropriate. Do not label the paragraph and do not use bullets.
2. Leave one blank line, then output one JSON object with this exact schema:
{"commands":[{"group_id":"group_0","destination_zone_id":"zone_5","movement_mode":"push"},{"group_id":"group_1","destination_zone_id":"zone_3","movement_mode":"regroup"}],"scan_zone_id":"zone_5","scout_zone_id":null}
When only one army_group is present, output exactly that one group command. When army_groups is empty, output commands:[].

The explanatory paragraph is required. A response that begins with "{" or contains only JSON is invalid. Do not output markdown, comments, coordinates, extra fields, or more than one JSON object.

Zone table:
- The columns= line defines the | separated field order; row_count is the number of following zone rows.
- own_non_army_contents excludes controlled combat units already represented in army_groups; never add zone contents to a group's composition.
- distance_from_army uses the current controlled-army center. distance_to_own_main and distance_to_enemy_main use fixed map landmarks; none of these fields selects an objective or proves safety.
- vision_state reports current visibility. visible_enemy_contents is visible now; last_seen_enemy_contents is remembered under fog; enemy_information_age_seconds reports its age or no_enemy_record.
- A fogged or partially_visible zone with no visible enemies is not confirmed empty.

Output rules:
- commands: exactly one object per group_id currently present in army_groups, with no duplicate group_id and no omitted groups. Length must equal the number of army_groups (zero when army_groups is empty).
- group_id: exactly one group_id currently present in army_groups.
- destination_zone_id: exactly one zone_id currently present in available_zones.
- movement_mode: regroup, push, assault, harass, defensive_retreat, panic_retreat, search_and_destroy.
- scan_zone_id: an existing zone_id to request one Scanner Sweep, or null.
- scout_zone_id: an existing zone_id to start or preserve one SCV reconnaissance task, or null. If scv_scout_active is yes, repeat active_scout_target_zone_id every cycle to preserve the scout; null cancels the active scout.

Movement semantics:
- regroup: move toward a safe own or neutral-zone gather point while preserving cohesion. A neutral zone is safe only when it has no known enemy units, enemy power, static defense, or active threat.
- push: advance through or toward the selected zone, taking limited forward fights without chasing targets behind the advance. The selected zone may be own, neutral, or enemy.
- assault: actively attack toward an enemy or useful neutral zone.
- harass: pressure a vulnerable enemy or useful neutral objective while avoiding the enemy main force and unfavorable committed engagements.
- defensive_retreat: withdraw to an own zone while allowing defensive fire.
- panic_retreat: escape to an own zone with survival as the priority.
- search_and_destroy: begin at the selected zone and automatically sweep different expansion zones while this mode remains active. Visible enemy structures are attacked first.

Decision rules:
- Act as a strategy executor. Use the current army_directive to prioritize the full strategy rather than as permission to bypass it; treat required conditions that cannot be confirmed from the Army View as unsatisfied.
- Use only the supplied Army View and treat masked information as unknown. Completed and under-construction units, structures, and technology are prerequisite evidence only; never output macro tasks.
- Army View exposes one persistent main_force and, when needed, one temporary reinforcement group. Main-force membership does not split because its formation spreads; newly produced or surviving non-main units remain reinforcement until they physically rejoin it. fragmented=yes means that no connected component contains at least 80% of the group's combat power; a broad connected formation or a small straggler is not fragmented.
- Treat main_force as the single operational force. Whenever reinforcement is present, still output a command for main_force in the same cycle; never command only reinforcement. Unless an immediate local threat requires retreat, direct reinforcement to converge on it: regroup toward the main force's current safe zone before an offensive, or move toward the same current objective after the offensive begins. Do not give reinforcement an independent attack, harass, or search route; reunited units merge into main_force automatically.
- Base every decision on the current Army View. A previous offensive or regroup order is historical context, not permission to repeat it when the current group distribution, local combat situation, or objective state has changed.
- When the strategy requires a concentrated force, use the current spatial distribution and local threats to decide whether groups should gather, reinforce a progressing force, continue the current objective, or recover. Do not infer readiness or progress from an old command alone.
- Evaluate strategy attack-composition readiness from the combined combat units across all current army_groups, excluding units still in production. If that combined force would meet the strategy gate only by ignoring separated reinforcement or detached combat units, treat the army as not yet attack-ready and merge first.
- Before initiating a planned offensive, explicitly compare each numeric attack-gate component with completed living units in the explanation; every component must be satisfied, and being nearly ready or having a favorable estimated advantage is insufficient. Once a valid offensive begins, use current progress and the strategy recovery conditions rather than automatically reapplying the opening gate after each loss.
- Do not recall a forward group solely because newly produced reinforcements form another group. Keep it advancing only while current evidence shows that it can make progress, and use current strategy conditions to decide whether other groups should reinforce, gather, or recover.
- Do not select an unsafe enemy zone as an ordinary regroup point. Use push or assault for an active enemy objective; use regroup only for a currently safe own or neutral gather zone.
- Clear local advantage at the active enemy objective is evidence that the forward group can still make progress; maintain its pressure while reinforcements travel forward.
- current_destination_reached and current_objective_status summarize the evidence for each group's existing destination. confirmed_clear means the destination is currently visible with no enemy presence.
- Do not begin search_and_destroy from missing vision or "no enemy is visible" alone. Begin or continue search_and_destroy only when a [Runtime Search-And-Destroy Hint] is present in the user message; follow that hint and ignore a conflicting army_directive for that cycle.
- Choose Scanner Sweep and SCV reconnaissance actions from the full strategy and the current Army View; do not wait for the army_directive to prescribe either action.
- Scanner Sweep costs 50 Orbital energy. Request one only when available_scanner_sweep_count is greater than 0 and missing vision materially affects the current army decision; otherwise use null. When necessary information cannot be obtained safely by ground scouting, prefer a Scanner Sweep if one is available.
- Only one SCV scout may be active. While scv_scout_active=yes, keep repeating active_scout_target_zone_id every cycle to preserve that task unless intentional cancellation is required; do not switch to another target mid-task.
- After an SCV scout reaches its target, is killed, or is interrupted, reassess the strategy, the latest scout result, zone vision and information age before choosing another target. A resolved task does not automatically require a replacement scout.
- Treat a recently completed scout that reached a zone and found no relevant enemy presence as completed even after that zone becomes fogged. Do not immediately scout the same empty zone again; choose the next strategy-relevant zone or use null unless new or sufficiently stale information materially affects a current decision.
- If last_scout_result=killed_en_route, do not automatically resend another SCV along the same route. Reassess whether the information is still necessary and whether the route is reasonably safe; use a Scanner Sweep when the information is necessary, the route is unsafe, and a sweep is available.
- If the selected strategy explicitly requests an opening SCV scout and the scout history shows no attempt, choose the strategy-specified target even when no army group exists or the army is not ready to attack. Postpone it only when the current observation shows a concrete route threat; lack of confirmed safety alone is not evidence of danger.
- Fog alone is not a reason to dispatch an SCV outside an explicit strategy scout objective. Reconnaissance must answer a current strategy decision and must not delay a supportable offensive whose prerequisites are already satisfied.
- Treat neutral_expansion zones as possible hidden enemy bases because they are neutral mineral expansion sites. Scout one when locating the next objective, checking a strategy-relevant expansion, or resolving sufficiently stale information; prioritize never-observed or strategically relevant sites using their distance fields. Do not mechanically cycle through every neutral expansion during the opening or interrupt a progressing offensive merely to scout them.
- During an ongoing forward operation, prioritize reconnaissance of the current or next strategy objective when needed; do not mechanically restart an already resolved opening scout of the enemy main.
- Use each group's role, composition, power, location, nearby enemies, fragmentation, current command, and the fixed Zone distance fields to compare destinations. Choose objectives from the strategy and current situation.
- Treat zone_id as an identifier only; adjacent zone numbers do not imply adjacent map positions.
- During the opening scout, follow the selected strategy's first reconnaissance objective. Scouting only the enemy natural is not sufficient when the strategy explicitly requests information from the enemy main.

Before output, verify that every command follows the strategy, uses an existing group and zone, respects unconfirmed conditions, and remains justified by the current Army View rather than only by a previous command.

Sharpy handles pathfinding, internal grouping, movement execution, abilities, formations, and unit-level micro."""

        parts = []
        if cleanup_hint:
            parts.extend([cleanup_hint.strip(), ""])
        parts.extend(
            [
                "[Full Strategy And Coordinator Army Guidance]",
                strategy_directive
                or "No Coordinator army guidance available.",
                "",
                "[Army Planner Observation At Decision Time]",
                observation,
            ]
        )
        user_msg = chr(10).join(parts)
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]


def build_army_control_observation(
    act: Any,
    state: Optional[Dict[str, Any]] = None,
    previous_state: Optional[Dict[str, Any]] = None,
) -> str:
    state = state or collect_army_control_state(act)
    previous_state = previous_state or {}

    recent_own_minerals = max(0, state["own_lost_minerals"] - previous_state.get("own_lost_minerals", state["own_lost_minerals"]))
    recent_own_gas = max(0, state["own_lost_gas"] - previous_state.get("own_lost_gas", state["own_lost_gas"]))
    recent_enemy_minerals = max(0, state["enemy_lost_minerals"] - previous_state.get("enemy_lost_minerals", state["enemy_lost_minerals"]))
    recent_enemy_gas = max(0, state["enemy_lost_gas"] - previous_state.get("enemy_lost_gas", state["enemy_lost_gas"]))

    lines = [
        "[Army Situation]",
        f"Controlled combat units={state['controlled_combat_units']}; idle_or_moving={state['idle_or_moving']}; attacking_or_moving={state['attacking_or_moving']}.",
        f"Threatened own zones: {_format_list(state['threatened_zone_ids'])}.",
        f"Controlled army power: own={state['own_army_power']:.2f}; visible enemy={state['visible_enemy_power']:.2f}; ratio={state['power_ratio']:.2f}; assessment={state['army_advantage']}.",
        f"Army center: nearest_zone={state['army_nearest_zone']}; position={state['army_position']}.",
        f"Recent losses since previous Army decision: own={recent_own_minerals} minerals/{recent_own_gas} gas; enemy={recent_enemy_minerals} minerals/{recent_enemy_gas} gas.",
        f"Known enemy types: {_format_counts(state['known_enemy_type_counts'])}.",
        f"Orbital scan status: orbitals={state.get('orbital_count', 0)}; energies={state.get('orbital_energies', [])}; scan_ready={state.get('scan_ready', 0)}.",
        f"SCV scout status: workers={state.get('scv_worker_count', 0)}; active_scout={state.get('active_scv_scout', False)}; scout_zone_id={state.get('active_scv_scout_zone_id')}; last_target_zone={state.get('last_scv_scout_zone_id')}; last_result={state.get('last_scv_scout_result')}; last_result_seconds_ago={state.get('last_scv_scout_result_seconds_ago')}.",
        "",
        "[Army Groups]",
    ]
    lines.extend(_format_army_group(group) for group in state.get("army_groups", []))
    if not state.get("army_groups"):
        lines.append("none")

    zone_groups = (("Own Zones", "own"), ("Enemy Zones", "enemy"), ("Neutral Zones", "neutral"))
    for heading, owner in zone_groups:
        lines.extend(["", f"[{heading}]"])
        zones = [zone for zone in state.get("available_zones", []) if zone.get("owner") == owner]
        zones.sort(
            key=lambda zone: float(
                zone.get("distance_from_army", float("inf"))
            )
        )
        lines.extend(_format_zone(zone) for zone in zones)
        if not zones:
            lines.append("none")
    return "\n".join(lines)


def _format_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))


def _format_list(values: list) -> str:
    return ", ".join(str(value) for value in values) if values else "none"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _format_army_group(group: Dict[str, Any]) -> str:
    command = group.get("current_command") or {}
    return (
        f"{group.get('group_id', 'unknown')} ({group.get('role', 'unknown')}): "
        f"unit_count={group.get('unit_count', 0)}; composition={_format_counts(group.get('unit_type_counts', {}))}; "
        f"power={group.get('power', 0.0)}; nearest_zone={group.get('nearest_zone_id', 'unknown')}; "
        f"fragmented={_yes_no(group.get('is_fragmented'))}; nearby_enemy_count={group.get('nearby_enemy_count', 0)}; "
        f"nearby_enemy_power={group.get('nearby_enemy_power', 0.0)}; nearby_enemy_composition={_format_counts(group.get('nearby_enemy_type_counts', {}))}; "
        f"current_command={command.get('movement_mode', 'none')} to {command.get('destination_zone_id', 'none')}; "
        f"search_target={group.get('search_target_zone_id') or 'none'}; "
        f"searched_zones={_format_list(group.get('searched_zone_ids', []))}; command_age={group.get('command_age_seconds', 0.0)}s."
    )


def _format_zone(zone: Dict[str, Any]) -> str:
    return (
        f"{zone.get('zone_id', 'unknown')} ({zone.get('zone_role', 'unknown')}): "
        f"owner={zone.get('owner', 'unknown')}; "
        f"distance_from_army={zone.get('distance_from_army', 'unknown')}; "
        f"distance_to_own_main={zone.get('distance_to_own_main', 'unknown')}; "
        f"distance_to_enemy_main={zone.get('distance_to_enemy_main', 'unknown')}; "
        f"own_units={zone.get('own_units', 0)}; own_power={zone.get('own_power', 0.0)}; "
        f"known_enemy_units={zone.get('known_enemy_units', 0)}; known_enemy_power={zone.get('known_enemy_power', 0.0)}; "
        f"enemy_static_power={zone.get('enemy_static_power', 0.0)}; power_balance={zone.get('power_balance', 0.0)}; "
        f"on_gather_route={_yes_no(zone.get('on_gather_route'))}; "
        f"under_attack={_yes_no(zone.get('under_attack'))}."
    )

def collect_army_control_state(act: Any) -> Dict[str, Any]:
    ai = act.ai
    roles = act.roles

    idle_or_moving = roles.free_units.filter(
        lambda unit: act.unit_values.should_attack(unit)
    )
    attacking_or_moving = roles.attacking_units.filter(
        lambda unit: act.unit_values.should_attack(unit)
    )
    controlled_units = idle_or_moving.copy()
    controlled_units.extend(
        attacking_or_moving.tags_not_in(controlled_units.tags)
    )

    enemy_units = getattr(ai, "all_enemy_units", [])
    visible_enemy_units, remembered_enemy_units = _split_enemy_units(enemy_units)
    close_enemies = (
        visible_enemy_units.closer_than(28, controlled_units.center)
        if controlled_units and visible_enemy_units
        else []
    )
    enemy_combat_units = _enemy_combat_units(act, enemy_units)
    visible_enemy_combat_units = _enemy_combat_units(act, visible_enemy_units)

    zones = list(act.zone_manager.expansion_zones)
    own_bases = sum(1 for zone in zones if getattr(zone, "is_ours", False))
    enemy_bases = sum(1 for zone in zones if getattr(zone, "is_enemys", False))
    threatened_zone_ids = [
        f"zone_{index}"
        for index, zone in enumerate(zones)
        if getattr(zone, "is_ours", False)
        and (
            getattr(zone, "is_under_attack", False)
            or bool(getattr(zone, "known_enemy_units", []))
        )
    ]

    army_position = controlled_units.center if controlled_units else None
    nearest_zone = _nearest_zone_name(army_position, zones)
    army_groups = act.get_army_group_states(controlled_units)
    available_zones = _available_zones(
        act,
        army_position,
        zones,
        set(getattr(act.zone_manager, "gather_points", [])),
        float(getattr(ai, "time", 0.0)),
        controlled_unit_tags=set(getattr(controlled_units, "tags", set())),
    )
    scout_state = act.get_scv_scout_state()
    scan_state = act.get_scan_state()

    own_power = _total_power(act, controlled_units)
    enemy_power = _total_power(act, visible_enemy_combat_units)
    power_ratio = own_power / max(enemy_power, 0.1)
    own_lost_minerals, own_lost_gas = _lost_resources(act, own=True)
    enemy_lost_minerals, enemy_lost_gas = _lost_resources(act, own=False)

    try:
        orbitals = ai.structures(UnitTypeId.ORBITALCOMMAND).ready
    except (AttributeError, TypeError):
        orbitals = []
    orbital_energies = [round(float(orbital.energy), 1) for orbital in orbitals]

    own_race = getattr(getattr(ai, "race", None), "name", "UNKNOWN")
    enemy_race = getattr(
        getattr(act.knowledge, "enemy_race", None),
        "name",
        "UNKNOWN",
    )

    return {
        "time_seconds": float(getattr(ai, "time", 0.0)),
        "own_race": own_race,
        "enemy_race": enemy_race,
        "supply_used": getattr(ai, "supply_used", 0),
        "supply_cap": getattr(ai, "supply_cap", 0),
        "army_supply": getattr(ai, "supply_army", 0),
        "controlled_combat_units": len(controlled_units),
        "idle_or_moving": len(idle_or_moving),
        "attacking_or_moving": len(attacking_or_moving),
        "known_enemy_units": len(enemy_units),
        "visible_enemy_units": len(visible_enemy_units),
        "close_enemy_units": len(close_enemies),
        "remembered_enemy_units": len(remembered_enemy_units),
        "own_bases": own_bases,
        "known_enemy_bases": enemy_bases,
        "orbital_count": len(orbitals),
        "orbital_energies": orbital_energies,
        "scan_ready": sum(1 for energy in orbital_energies if energy >= 50),
        "last_scan_zone_id": scan_state["last_target_zone"],
        "last_scan_result": scan_state["last_result"],
        "last_scan_result_seconds_ago": scan_state["last_result_seconds_ago"],
        "scv_worker_count": scout_state["workers"],
        "active_scv_scout": scout_state["active_scout"],
        "active_scv_scout_zone_id": scout_state["scout_zone_id"],
        "last_scv_scout_zone_id": scout_state["last_target_zone"],
        "last_scv_scout_result": scout_state["last_result"],
        "last_scv_scout_result_seconds_ago": scout_state[
            "last_result_seconds_ago"
        ],
        "threatened_zone_ids": threatened_zone_ids,
        "own_army_power": own_power,
        "visible_enemy_power": enemy_power,
        "power_ratio": power_ratio,
        "army_advantage": _advantage_label(power_ratio, enemy_power),
        "army_nearest_zone": nearest_zone,
        "army_position": _point_text(army_position),
        "available_zones": available_zones,
        "army_groups": army_groups,
        "own_lost_minerals": own_lost_minerals,
        "own_lost_gas": own_lost_gas,
        "enemy_lost_minerals": enemy_lost_minerals,
        "enemy_lost_gas": enemy_lost_gas,
        "own_unit_type_counts": _unit_counts(controlled_units),
        "close_enemy_type_counts": _unit_counts(close_enemies),
        "known_enemy_type_counts": _unit_counts(enemy_combat_units),
    }


def _state_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "own_lost_minerals": state["own_lost_minerals"],
        "own_lost_gas": state["own_lost_gas"],
        "enemy_lost_minerals": state["enemy_lost_minerals"],
        "enemy_lost_gas": state["enemy_lost_gas"],
    }


def _is_remembered_enemy(unit: Any) -> bool:
    return bool(
        getattr(unit, "is_memory", False)
        or getattr(unit, "is_snapshot", False)
    )


def _split_enemy_units(enemy_units: Any) -> tuple:
    if hasattr(enemy_units, "filter"):
        return (
            enemy_units.filter(lambda unit: not _is_remembered_enemy(unit)),
            enemy_units.filter(_is_remembered_enemy),
        )
    return (
        [unit for unit in enemy_units if not _is_remembered_enemy(unit)],
        [unit for unit in enemy_units if _is_remembered_enemy(unit)],
    )


def _enemy_combat_units(act: Any, enemy_units: Any) -> Any:
    if not hasattr(enemy_units, "filter"):
        return enemy_units
    return enemy_units.filter(
        lambda unit: not getattr(unit, "is_structure", False)
        and not act.unit_values.is_worker(unit)
    )


def _total_power(act: Any, units: Any) -> float:
    if not units:
        return 0.0
    try:
        return float(act.unit_values.calc_total_power(units).power)
    except Exception:
        return float(len(units))


def _lost_resources(act: Any, own: bool) -> tuple:
    manager = getattr(act, "lost_units_manager", None)
    if manager is None:
        return 0, 0
    method_name = (
        "calculate_own_lost_resources"
        if own
        else "calculate_enemy_lost_resources"
    )
    try:
        minerals, gas = getattr(manager, method_name)()
        return int(minerals), int(gas)
    except Exception:
        return 0, 0


def _advantage_label(power_ratio: float, enemy_power: float) -> str:
    if enemy_power <= 0:
        return "no_visible_enemy_army"
    if power_ratio >= 1.6:
        return "strong_advantage"
    if power_ratio >= 1.15:
        return "advantage"
    if power_ratio >= 0.85:
        return "even"
    if power_ratio >= 0.6:
        return "disadvantage"
    return "strong_disadvantage"


def _nearest_zone_name(position: Any, zones: list) -> str:
    if position is None or not zones:
        return "unknown"
    index, zone = min(
        enumerate(zones),
        key=lambda item: position.distance_to(item[1].center_location),
    )
    return f"zone_{index}"


def _zone_vision_state(
    ai: Any,
    zone: Any,
    own_units: Any,
    visible_enemy_units: Any,
    remembered_enemy_units: Any,
    last_confirmed_at: float,
) -> str:
    points = [
        getattr(zone, "center_location", None),
        getattr(zone, "mineral_line_center", None),
        getattr(zone, "gather_point", None),
    ]
    visible_checks = []
    for point in points:
        if point is None:
            continue
        try:
            visible_checks.append(bool(ai.is_visible(point)))
        except Exception:
            visible_checks.append(False)

    if visible_checks and all(visible_checks):
        return "visible"
    if (
        any(visible_checks)
        or bool(own_units)
        or bool(visible_enemy_units)
    ):
        return "partially_visible"
    was_previously_observed = (
        last_confirmed_at >= 0.0
        or float(getattr(zone, "last_scouted_center", -1.0)) >= 0.0
        or float(getattr(zone, "last_scouted_mineral_line", -1.0)) >= 0.0
        or bool(remembered_enemy_units)
    )
    if was_previously_observed:
        return "fogged"
    return "never_observed"


def _zone_enemy_information_age(
    remembered_enemy_units: Any,
    visible_enemy_units: Any,
    vision_state: str,
    current_time: float,
    last_confirmed_at: float,
) -> Optional[float]:
    remembered_ages = []
    for unit in remembered_enemy_units:
        try:
            remembered_ages.append(float(getattr(unit, "age")))
        except (AttributeError, TypeError, ValueError):
            continue
    if remembered_ages:
        return round(max(remembered_ages), 1)
    if visible_enemy_units or vision_state == "visible":
        return 0.0
    if last_confirmed_at >= 0.0:
        return round(max(0.0, current_time - last_confirmed_at), 1)
    return None


def _available_zones(
    act: Any,
    position: Any,
    zones: list,
    gather_points: set,
    current_time: float,
    controlled_unit_tags: Optional[set] = None,
) -> list:
    result = []
    controlled_unit_tags = controlled_unit_tags or set()
    zone_count = len(zones)
    own_main = zones[0] if zones else None
    enemy_main = zones[-1] if zones else None
    for index, zone in enumerate(zones):
        owner = "own" if getattr(zone, "is_ours", False) else (
            "enemy" if getattr(zone, "is_enemys", False) else "neutral"
        )
        known_enemy_units = getattr(zone, "known_enemy_units", [])
        visible_enemy_units, remembered_enemy_units = _split_enemy_units(
            known_enemy_units
        )
        own_units = getattr(zone, "our_units", [])
        if hasattr(own_units, "tags_not_in"):
            own_non_army_units = own_units.tags_not_in(controlled_unit_tags)
        else:
            own_non_army_units = [
                unit
                for unit in own_units
                if getattr(unit, "tag", None) not in controlled_unit_tags
            ]
        own_combat_units = _enemy_combat_units(act, own_units)
        visible_enemy_combat_units = _enemy_combat_units(
            act, visible_enemy_units
        )
        remembered_enemy_combat_units = _enemy_combat_units(
            act, remembered_enemy_units
        )
        own_combat_power = _total_power(act, own_combat_units)
        visible_enemy_combat_power = _total_power(act, visible_enemy_combat_units)
        remembered_enemy_combat_power = _total_power(act, remembered_enemy_combat_units)
        enemy_static_power = _power_value(getattr(zone, "enemy_static_power", None))
        combat_power_balance = round(
            own_combat_power - visible_enemy_combat_power
            - remembered_enemy_combat_power - enemy_static_power,
            2,
        )
        last_military_state_confirmed_at = float(
            getattr(zone, "last_military_state_confirmed_at", -1.0)
        )
        vision_state = _zone_vision_state(
            act.ai,
            zone,
            own_units,
            visible_enemy_units,
            remembered_enemy_units,
            last_military_state_confirmed_at,
        )
        enemy_information_age_seconds = _zone_enemy_information_age(
            remembered_enemy_units,
            visible_enemy_units,
            vision_state,
            current_time,
            last_military_state_confirmed_at,
        )
        under_attack = (
            owner == "own"
            and bool(getattr(zone, "is_under_attack", False))
        )
        item = {
            "zone_id": f"zone_{index}",
            "owner": owner,
            "zone_role": _zone_role(index, zone_count, owner),
            "under_attack": under_attack,
            "on_gather_route": index in gather_points,
            "own_units": len(own_units),
            "own_non_army_units": len(own_non_army_units),
            "known_enemy_units": len(known_enemy_units),
            "visible_enemy_units": len(visible_enemy_units),
            "remembered_enemy_units": len(remembered_enemy_units),
            "own_contents": _unit_counts(own_units),
            "own_non_army_contents": _unit_counts(own_non_army_units),
            "visible_enemy_contents": _unit_counts(visible_enemy_units),
            "last_seen_enemy_contents": _unit_counts(remembered_enemy_units),
            "vision_state": vision_state,
            "enemy_information_age_seconds": enemy_information_age_seconds,
            "own_combat_power": own_combat_power,
            "visible_enemy_power": visible_enemy_combat_power,
            "remembered_enemy_power": remembered_enemy_combat_power,
            "combat_power_balance": combat_power_balance,
            "own_power": _power_value(getattr(zone, "our_power", None)),
            "known_enemy_power": _power_value(
                getattr(zone, "known_enemy_power", None)
            ),
            "enemy_static_power": enemy_static_power,
        }
        item["power_balance"] = round(
            item["own_power"] - item["known_enemy_power"], 2
        )
        if position is not None:
            distance_from_army = round(
                position.distance_to(zone.center_location), 1
            )
            item["distance_from_army"] = distance_from_army
        if own_main is not None:
            item["distance_to_own_main"] = round(
                zone.center_location.distance_to(own_main.center_location), 1
            )
        if enemy_main is not None:
            item["distance_to_enemy_main"] = round(
                zone.center_location.distance_to(enemy_main.center_location), 1
            )
        result.append(item)
    return result


def _zone_role(index: int, zone_count: int, owner: str) -> str:
    if index == 0:
        return "own_main"
    if index == 1:
        return "own_natural"
    if index == zone_count - 1:
        return "enemy_main"
    if index == zone_count - 2:
        return "enemy_natural"
    if owner == "own":
        return "own_expansion"
    if owner == "enemy":
        return "enemy_expansion"
    return "neutral_expansion"


def _power_value(power: Any) -> float:
    return round(float(getattr(power, "power", 0.0)), 2)


def _point_text(point: Any) -> str:
    if point is None:
        return "unknown"
    return f"({point.x:.1f},{point.y:.1f})"


def _unit_counts(units: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for unit in units:
        type_id = getattr(unit, "type_id", None)
        name = getattr(type_id, "name", str(type_id or "UNKNOWN"))
        counts[name] = counts.get(name, 0) + 1
    return counts
