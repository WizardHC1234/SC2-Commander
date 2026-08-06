"""Unified structured observations and role-specific text views.

The collectors that know about Sharpy remain in their existing modules.  This
module owns the stable schema, masking rules, and rendering only; it never
reads live game state itself.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional


SCHEMA_VERSION = "2.0"

_UPGRADE_DISPLAY_ALIASES = {
    # SC2 reports Battlecruiser Weapon Refit with this legacy/internal name.
    # Present the strategy-facing name so planners can match it to Yamato.
    "BATTLECRUISERENABLESPECIALIZATIONS": "YAMATOCANNON",
}


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _normalise_upgrade_names(values: Iterable[Any]) -> List[str]:
    names = {
        _UPGRADE_DISPLAY_ALIASES.get(str(value), str(value))
        for value in values
    }
    return sorted(names)


def _active_mining_base_count(minerals: Dict[str, Any]) -> int:
    """Match ``Zone.has_minerals`` / Sharpy ``Expand`` base-count semantics."""
    count = 0
    for status in ("Full", "Plenty", "Limited"):
        try:
            count += max(0, int(minerals.get(status, 0) or 0))
        except (TypeError, ValueError):
            continue
    return count


def _age(now: float, timestamp: Any) -> Optional[float]:
    try:
        if timestamp is None:
            return None
        return round(max(0.0, now - float(timestamp)), 1)
    except (TypeError, ValueError):
        return None


def _optional_nonneg_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def _game_time_line(
    time: Dict[str, Any],
    *,
    extra_fields: str = "",
) -> str:
    """Render the shared [Game] time fields; omit limit when unset."""
    parts = [f"game_time_seconds={_value(time.get('seconds'))}"]
    limit = time.get("limit_seconds")
    if limit is not None:
        parts.append(f"game_time_limit_seconds={_value(limit)}")
        parts.append(
            f"seconds_remaining={_value(time.get('seconds_remaining'))}"
        )
    if extra_fields:
        parts.append(extra_fields)
    return "; ".join(parts)


def _normalise_macro_execution(raw: Any, now: float) -> Dict[str, Any]:
    state = _dict(raw)
    tasks = [_dict(task) for task in _list(state.get("active_macro_tasks"))]
    issues = list(_list(state.get("last_issues")))
    task_statuses = []
    for task in tasks:
        task_status = str(
            task.get("status")
            or ("failed" if task.get("disabled") else "active_unsatisfied")
        )
        # Normalise records produced before the status name was clarified.
        if task_status == "executing":
            task_status = "active_unsatisfied"
        task["status"] = task_status
        task_statuses.append(task_status)
    failed_count = sum(
        status in {"failed", "disabled"}
        for status in task_statuses
    )
    if tasks and failed_count == len(tasks):
        status = "failed"
    elif failed_count or issues:
        status = "active_with_issues" if tasks else "failed"
    elif tasks and all(
        item in {"completed", "target_satisfied"} for item in task_statuses
    ):
        status = "target_satisfied"
    elif tasks:
        status = "active_unsatisfied"
    else:
        status = "idle"
    return {
        "status": status,
        "last_tasks": list(_list(state.get("last_tasks"))),
        "active_macro_tasks": tasks,
        "last_update_seconds_ago": _age(
            now, state.get("last_update_game_time")
        ),
        "last_issues": issues,
    }


def _normalise_combat_execution(raw: Any, now: float) -> Dict[str, Any]:
    state = _dict(raw)
    policy = deepcopy(_dict(state.get("last_policy")))
    issues = list(_list(state.get("last_command_issues")))
    status = state.get("status")
    if not status:
        if not policy:
            status = "idle"
        elif issues:
            status = "executing_with_issues"
        else:
            status = "executing"
    return {
        "status": status,
        "last_policy": deepcopy(_dict(state.get("last_policy"))),
        "policy_age_seconds": _age(
            now, state.get("policy_applied_game_time")
        ),
        "last_command_issues": list(
            _list(state.get("last_command_issues"))
        ),
    }


def _normalise_execution_history(raw: Any) -> Dict[str, Any]:
    history = _dict(raw)
    try:
        window_start = float(
            history.get("window_start_game_time_seconds", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        window_start = 0.0
    macro_events = _list(history.get("macro"))
    combat_events = _list(history.get("combat"))
    return {
        "window_start_game_time_seconds": round(window_start, 1),
        "macro": deepcopy(
            [item for item in macro_events if isinstance(item, dict)]
        ),
        "combat": deepcopy(
            [item for item in combat_events if isinstance(item, dict)]
        ),
    }


def _normalise_previous_decision(raw: Any) -> Optional[Dict[str, Any]]:
    state = _dict(raw)
    if not state:
        return None
    macro_commands = [
        _dict(item)
        for item in _list(state.get("macro_commands"))
        if isinstance(item, dict) and item.get("name")
    ]
    army_commands = [
        _dict(item)
        for item in _list(state.get("army_commands"))
        if isinstance(item, dict) and item.get("group_id")
    ]
    if not macro_commands and not army_commands and not state.get("wake_event"):
        return None
    try:
        game_time = float(state.get("game_time_seconds"))
    except (TypeError, ValueError):
        game_time = None
    return {
        "game_time_seconds": (
            round(game_time, 1) if game_time is not None else None
        ),
        "macro_commands": macro_commands,
        "army_commands": army_commands,
        "scan_zone_id": state.get("scan_zone_id"),
        "scout_zone_id": state.get("scout_zone_id"),
        "wake_event": deepcopy(_dict(state.get("wake_event"))) or None,
        "issues": [
            str(item) for item in _list(state.get("issues")) if str(item).strip()
        ],
    }


def build_full_observation(
    legacy_snapshot: Dict[str, Any],
    *,
    army_state: Optional[Dict[str, Any]] = None,
    macro_execution: Optional[Dict[str, Any]] = None,
    combat_execution: Optional[Dict[str, Any]] = None,
    previous_decision: Optional[Dict[str, Any]] = None,
    execution_history: Optional[Dict[str, Any]] = None,
    game_loop: Optional[int] = None,
    game_time_limit_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert existing collectors into the one canonical observation schema."""
    legacy = _dict(legacy_snapshot)
    economy = _dict(legacy.get("economy"))
    legacy_own = _dict(legacy.get("own_forces"))
    enemy = _dict(legacy.get("enemy"))
    map_control = _dict(legacy.get("map_control"))
    combat = _dict(legacy.get("combat"))
    flags = _dict(legacy.get("memory_flags"))
    army = _dict(army_state)

    now = float(legacy.get("time", army.get("time_seconds", 0.0)) or 0.0)
    loop = int(game_loop) if game_loop is not None else None
    snapshot_id = f"game_loop:{loop}" if loop is not None else f"time:{now:.2f}"
    limit = _optional_nonneg_float(
        game_time_limit_seconds
        if game_time_limit_seconds is not None
        else legacy.get("game_time_limit_seconds")
    )
    seconds_remaining = (
        round(max(0.0, float(limit) - now), 1) if limit is not None else None
    )

    groups = deepcopy(_list(army.get("army_groups")))
    zones = deepcopy(_list(army.get("available_zones")))
    _derive_army_objective_context(groups, zones)
    current_commands = []
    for group in groups:
        if not isinstance(group, dict) or not group.get("current_command"):
            continue
        command = deepcopy(group["current_command"])
        command["group_id"] = group.get("group_id")
        command["command_age_seconds"] = group.get("command_age_seconds")
        current_commands.append(command)

    completed = deepcopy(_dict(legacy_own.get("completed")))
    under_construction = deepcopy(
        _dict(legacy_own.get("under_construction"))
    )
    workers_en_route = deepcopy(_dict(legacy_own.get("workers_en_route")))
    active_queues = deepcopy(_dict(legacy_own.get("active_queues")))
    upgrades = _normalise_upgrade_names(_list(legacy.get("upgrades")))
    upgrades_in_progress = _normalise_upgrade_names(
        str(key)[len("Researching "):]
        for key in active_queues
        if str(key).startswith("Researching ")
    )

    combat_state = deepcopy(combat)
    combat_state.update(
        {
            "controlled_own_army_power": army.get("own_army_power"),
            "visible_enemy_army_power": army.get("visible_enemy_power"),
            "controlled_to_visible_enemy_power_ratio": army.get("power_ratio"),
            "army_control_advantage": army.get("army_advantage"),
        }
    )

    own_base_minerals = _dict(map_control.get("own_base_minerals"))
    active_mining_base_count = _active_mining_base_count(own_base_minerals)

    result = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "time": {
            "game_loop": loop,
            "seconds": now,
            "formatted": legacy.get("time_formatted") or _format_time(now),
            "limit_seconds": limit,
            "seconds_remaining": seconds_remaining,
        },
        "economy": {
            "minerals": economy.get("minerals"),
            "vespene": economy.get("vespene"),
            "mineral_income": economy.get("minerals_per_min"),
            "vespene_income": economy.get("vespene_per_min"),
            "supply_used": economy.get("supply_used"),
            "supply_cap": economy.get("supply_cap"),
            "supply_free": economy.get("supply_left"),
            "workers": economy.get("supply_workers"),
            "ideal_workers": economy.get("ideal_worker_count"),
            "own_base_count": map_control.get("own_bases"),
            "own_base_minerals": deepcopy(
                _dict(map_control.get("own_base_minerals"))
            ),
            "own_base_gas": deepcopy(
                _dict(map_control.get("own_base_gas"))
            ),
        },
        "own_forces": {
            "race": army.get("own_race"),
            "army_supply": economy.get("supply_army"),
            "supply_used": economy.get("supply_used"),
            "supply_cap": economy.get("supply_cap"),
            "supply_free": economy.get("supply_left"),
            "army_power": combat.get("our_army_power"),
            # Ready combat units on the field only (attack-gate relevant).
            "combat_composition": deepcopy(
                _dict(army.get("own_unit_type_counts"))
            ),
            # Units currently in production queues (not yet living).
            "training_combat_composition": _training_unit_counts(active_queues),
            "completed_counts": completed,
        },
        "production": {
            "completed": completed,
            "under_construction": under_construction,
            "workers_en_route": workers_en_route,
            "active_queues": active_queues,
        },
        "technology": {
            "completed_upgrades": upgrades,
            "upgrades_in_progress": upgrades_in_progress,
        },
        "enemy": {
            "race": army.get("enemy_race"),
            "known_composition": deepcopy(_dict(enemy.get("composition"))),
            "visible_composition": deepcopy(
                _dict(army.get("close_enemy_type_counts"))
            ),
            "known_combat_composition": deepcopy(
                _dict(army.get("known_enemy_type_counts"))
            ),
            "known_base_count": map_control.get("known_enemy_bases"),
            "last_observation_time": enemy.get("last_observation_time"),
            "seconds_since_last_seen": enemy.get("seconds_since_last_seen"),
            "macro_build": (
                flags.get("macro_build")
                if flags.get("macro_build") not in (None, "", "StandardMacro")
                else "unclassified"
            ),
            "known_types": sorted(_dict(enemy.get("composition")).keys()),
        },
        "map_control": {
            "own_base_count": map_control.get("own_bases"),
            "active_mining_base_count": active_mining_base_count,
            "known_enemy_base_count": map_control.get("known_enemy_bases"),
            "neutral_expansion_count": map_control.get("neutral_expansions"),
            "own_base_minerals": deepcopy(
                _dict(map_control.get("own_base_minerals"))
            ),
            "own_base_gas": deepcopy(
                _dict(map_control.get("own_base_gas"))
            ),
        },
        "combat": combat_state,
        "threat_flags": {
            "is_rushing": bool(flags.get("is_rushing", False)),
            "rush_build": flags.get("rush_build"),
            "macro_build": flags.get("macro_build"),
            "cloak_or_burrow_threat": bool(
                flags.get("enemy_cloak_threat", False)
            ),
            "has_proxy_buildings": bool(
                flags.get("has_proxy_buildings", False)
            ),
            "remembered_enemy_units": flags.get("remembered_enemy_units"),
            "own_zones_under_attack": list(
                _list(army.get("threatened_zone_ids"))
            ),
            "supply_blocked": (
                economy.get("supply_left") is not None
                and economy.get("supply_left") <= 0
            ),
        },
        "army_control": {
            "groups": groups,
            "zones": zones,
            "current_commands": current_commands,
            "threatened_zone_ids": list(
                _list(army.get("threatened_zone_ids"))
            ),
            "controlled_combat_units": army.get("controlled_combat_units", 0),
            "idle_or_moving_combat_units": army.get("idle_or_moving", 0),
            "attacking_or_moving_combat_units": army.get("attacking_or_moving", 0),
            "known_enemy_unit_count": army.get("known_enemy_units", 0),
            "visible_enemy_unit_count": army.get("visible_enemy_units", 0),
            "remembered_enemy_unit_count": army.get("remembered_enemy_units", 0),
            "visible_enemy_unit_count_near_army": army.get("close_enemy_units", 0),
            "army_nearest_zone": army.get("army_nearest_zone"),
            "zone_topology": deepcopy(_dict(army.get("zone_topology"))),
        },
        "capabilities": {
            "scan": {
                "orbital_count": army.get("orbital_count", 0),
                "orbital_energies": list(
                    _list(army.get("orbital_energies"))
                ),
                "available_scan_count": army.get("scan_ready", 0),
                "scan_energy_cost": 50,
                "last_target_zone_id": army.get("last_scan_zone_id"),
                "last_result": army.get("last_scan_result"),
                "last_result_seconds_ago": army.get(
                    "last_scan_result_seconds_ago"
                ),
            },
            "scv_scout": {
                "worker_count": army.get(
                    "scv_worker_count", economy.get("supply_workers")
                ),
                "active": bool(army.get("active_scv_scout", False)),
                "active_scout_count": (
                    1 if army.get("active_scv_scout", False) else 0
                ),
                "active_target_zone_id": army.get(
                    "active_scv_scout_zone_id"
                ),
                "last_target_zone_id": army.get("last_scv_scout_zone_id"),
                "last_result": army.get("last_scv_scout_result"),
                "last_result_seconds_ago": army.get(
                    "last_scv_scout_result_seconds_ago"
                ),
            },
        },
        "execution": {
            "macro": _normalise_macro_execution(macro_execution, now),
            "combat": _normalise_combat_execution(combat_execution, now),
            "previous_decision": _normalise_previous_decision(previous_decision),
            "since_last_decision": _normalise_execution_history(
                execution_history
            ),
        },
    }
    return result


def _derive_army_objective_context(
    groups: list,
    zones: list,
) -> None:
    """Add factual target-state summaries without choosing Army actions."""
    zone_by_id = {
        str(zone.get("zone_id")): zone
        for zone in zones
        if isinstance(zone, dict) and zone.get("zone_id") is not None
    }
    for group in groups:
        if not isinstance(group, dict):
            continue
        command = _dict(group.get("current_command"))
        destination = str(command.get("destination_zone_id") or "").strip()
        if not destination:
            group["current_destination_reached"] = None
            group["current_objective_status"] = "none"
            continue

        destination_reached = (
            str(group.get("nearest_zone_id") or "").strip() == destination
        )
        group["current_destination_reached"] = destination_reached
        zone = _dict(zone_by_id.get(destination))
        if not zone:
            group["current_objective_status"] = "unknown"
            continue

        enemy_present = _zone_has_enemy_evidence(zone)
        if enemy_present:
            status = "enemy_present"
        elif str(zone.get("vision_state") or "") == "visible":
            status = "confirmed_clear"
        elif destination_reached:
            status = "reached_unconfirmed"
        else:
            status = "en_route_unconfirmed"
        group["current_objective_status"] = status

def _zone_has_enemy_evidence(zone: Dict[str, Any]) -> bool:
    return bool(
        _dict(zone.get("visible_enemy_contents"))
        or _dict(zone.get("last_seen_enemy_contents"))
        or int(zone.get("visible_enemy_units", 0) or 0) > 0
        or int(zone.get("remembered_enemy_units", 0) or 0) > 0
        or _numeric(zone.get("visible_enemy_power")) > 0.0
        or _numeric(zone.get("remembered_enemy_power")) > 0.0
        or _numeric(zone.get("enemy_static_power")) > 0.0
    )


def mask_observation(
    full_observation: Dict[str, Any],
    view_type: str,
) -> Dict[str, Any]:
    """Return a deep-copied role view without reading live game state."""
    full = _dict(full_observation)
    # Accept legacy view_type aliases from the multi-agent era.
    if view_type == "top":
        view_type = "full"
    elif view_type == "mid":
        view_type = "macro"
    elif view_type == "army":
        view_type = "combat"
    common = {
        "schema_version": full.get("schema_version", SCHEMA_VERSION),
        "snapshot_id": full.get("snapshot_id"),
        "view_type": view_type,
    }
    if view_type == "full":
        view = deepcopy(full)
        view["view_type"] = "full"
        return view

    if view_type == "macro":
        view = {
            **common,
            "time": deepcopy(_dict(full.get("time"))),
            "economy": deepcopy(_dict(full.get("economy"))),
            "own_forces": deepcopy(_dict(full.get("own_forces"))),
            "production": deepcopy(_dict(full.get("production"))),
            "technology": deepcopy(_dict(full.get("technology"))),
            "enemy": deepcopy(_dict(full.get("enemy"))),
            "map_control": deepcopy(_dict(full.get("map_control"))),
            "execution": {
                "macro": deepcopy(
                    _dict(_dict(full.get("execution")).get("macro"))
                )
            },
        }
        return view

    if view_type == "combat":
        own = _dict(full.get("own_forces"))
        production = _dict(full.get("production"))
        combat_composition = _dict(own.get("combat_composition"))
        completed = _dict(production.get("completed"))
        military_completed = {
            unit_name: completed.get(unit_name, count)
            for unit_name, count in combat_composition.items()
        }
        view = {
            **common,
            "time": deepcopy(_dict(full.get("time"))),
            "own_forces": {
                "army_supply": own.get("army_supply"),
                "supply_used": own.get("supply_used"),
                "supply_cap": own.get("supply_cap"),
                "supply_free": own.get("supply_free"),
                "army_power": own.get("army_power"),
                "race": own.get("race"),
                "combat_composition": deepcopy(
                    combat_composition
                ),
                "training_combat_composition": deepcopy(
                    _dict(own.get("training_combat_composition"))
                ),
            },
            "military_readiness": {
                "completed": military_completed,
                "completed_units_and_structures": deepcopy(completed),
                "under_construction": deepcopy(
                    _dict(production.get("under_construction"))
                ),
                "technology": deepcopy(_dict(full.get("technology"))),
            },
            "enemy": deepcopy(_dict(full.get("enemy"))),
            "combat": deepcopy(_dict(full.get("combat"))),
            "threat_flags": deepcopy(_dict(full.get("threat_flags"))),
            "army_control": deepcopy(_dict(full.get("army_control"))),
            "capabilities": deepcopy(_dict(full.get("capabilities"))),
            "execution": {
                "combat": deepcopy(
                    _dict(_dict(full.get("execution")).get("combat"))
                )
            },
        }
        return view

    raise ValueError(f"unknown observation view_type: {view_type!r}")


def render_observation(view: Dict[str, Any], view_type: str) -> str:
    if view_type == "top":
        view_type = "full"
    elif view_type == "mid":
        view_type = "macro"
    elif view_type == "army":
        view_type = "combat"
    if view_type == "full":
        return _render_full(view)
    if view_type == "macro":
        return _render_macro(view)
    if view_type == "combat":
        return _render_combat(view)
    raise ValueError(f"unknown observation view_type: {view_type!r}")


def _value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _format_time(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _counts(counts: Any) -> str:
    data = _dict(counts)
    if not data:
        return "none"
    return ", ".join(
        f"{value} {key}"
        for key, value in sorted(
            data.items(), key=lambda item: (-_numeric(item[1]), str(item[0]))
        )
    )


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _items(values: Iterable[Any]) -> str:
    rendered = [str(value) for value in values if value not in (None, "")]
    return " | ".join(rendered) if rendered else "none"


def _economy_line(economy: Dict[str, Any]) -> str:
    return (
        f"worker_count={_value(economy.get('workers'))}; "
        f"ideal_worker_count={_value(economy.get('ideal_workers'))}; "
        f"minerals={_value(economy.get('minerals'))}; vespene_gas={_value(economy.get('vespene'))}; "
        f"mineral_income_per_minute={_value(economy.get('mineral_income'))}; "
        f"vespene_gas_income_per_minute={_value(economy.get('vespene_income'))}; "
        f"supply_used={_value(economy.get('supply_used'))}; "
        f"supply_capacity={_value(economy.get('supply_cap'))}; "
        f"supply_available={_value(economy.get('supply_free'))}"
    )


def _production_lines(production: Dict[str, Any]) -> List[str]:
    return [
        f"completed_units_and_structures={_counts(production.get('completed'))}",
        f"units_and_structures_under_construction={_counts(production.get('under_construction'))}",
        f"workers_en_route_to_build={_counts(production.get('workers_en_route'))}",
        f"active_production_and_research_queues={_counts(production.get('active_queues'))}",
    ]


def _combat_line(combat: Dict[str, Any]) -> str:
    global_own = combat.get("our_army_power")
    controlled_own = combat.get("controlled_own_army_power")
    parts = [
        f"global_army_advantage={_value(combat.get('army_advantage'))}",
        f"global_income_advantage={_value(combat.get('income_advantage'))}",
        f"global_predicted_combat_outcome={_value(combat.get('advantage_predicted'))}",
        f"global_own_army_power={_value(global_own)}",
        f"global_enemy_army_power={_value(combat.get('enemy_army_power'))}",
    ]
    # Skip controlled own power when it matches global (common duplicate).
    if controlled_own is not None and abs(
        _numeric(controlled_own) - _numeric(global_own)
    ) > 1e-6:
        parts.append(f"controlled_own_army_power={_value(controlled_own)}")
    if combat.get("visible_enemy_army_power") is not None:
        parts.append(
            f"visible_enemy_army_power={_value(combat.get('visible_enemy_army_power'))}"
        )
    parts.extend(
        [
            f"enemy_air_threat_level={_value(combat.get('enemy_air'))}",
            f"own_lost_minerals={_value(combat.get('own_lost_minerals'))}",
            f"own_lost_vespene_gas={_value(combat.get('own_lost_gas'))}",
            f"enemy_lost_minerals={_value(combat.get('enemy_lost_minerals'))}",
            f"enemy_lost_vespene_gas={_value(combat.get('enemy_lost_gas'))}",
        ]
    )
    return "; ".join(parts)

def _enemy_lines(
    enemy: Dict[str, Any],
    *,
    include_base_count: bool = True,
) -> List[str]:
    observation_age = (
        f"seconds_since_enemy_army_last_seen={_value(enemy.get('seconds_since_last_seen'))}"
    )
    if include_base_count:
        observation_age = (
            f"known_enemy_base_count={_value(enemy.get('known_base_count'))}; "
            + observation_age
        )
    return [
        f"known_enemy_composition={_counts(enemy.get('known_composition'))}",
        f"visible_enemy_composition_near_army={_counts(enemy.get('visible_composition'))}",
        f"known_enemy_combat_composition={_counts(enemy.get('known_combat_composition'))}",
        f"enemy_macro_build={_value(enemy.get('macro_build'))}",
        observation_age,
    ]


def _threat_line(flags: Dict[str, Any]) -> str:
    rush_build = flags.get("rush_build") if flags.get("is_rushing") else "none"
    return (
        f"enemy_rush_build={_value(rush_build)}; "
        f"enemy_cloak_or_burrow_threat={_value(flags.get('cloak_or_burrow_threat'))}; "
        f"enemy_proxy_buildings_detected={_value(flags.get('has_proxy_buildings'))}; "
        f"own_zone_ids_under_attack={_items(_list(flags.get('own_zones_under_attack')))}"
    )


def _group_lines(army_control: Dict[str, Any]) -> List[str]:
    groups = [_dict(group) for group in _list(army_control.get("groups"))]
    if not groups:
        return ["none"]
    lines = []
    for group in groups:
        command = _dict(group.get("current_command"))
        lines.append(
            f"group_id={_value(group.get('group_id'))}; role={_value(group.get('role'))}; "
            f"unit_count={_value(group.get('unit_count', 0))}; "
            f"unit_composition={_counts(group.get('unit_type_counts'))}; "
            f"army_group_power={_value(group.get('power'))}; "
            f"nearest_zone_id={_value(group.get('nearest_zone_id'))}; "
            f"is_fragmented={_value(group.get('is_fragmented'))}; "
            f"nearby_enemy_count={_value(group.get('nearby_enemy_count', 0))}; "
            f"nearby_enemy_army_power={_value(group.get('nearby_enemy_power', 0.0))}; "
            f"nearby_enemy_composition={_counts(group.get('nearby_enemy_type_counts'))}; "
            f"current_movement_mode={_value(command.get('movement_mode'))}; "
            f"current_destination_zone_id={_value(command.get('destination_zone_id'))}; "
            f"current_destination_reached={_value(group.get('current_destination_reached'))}; "
            f"current_objective_status={_value(group.get('current_objective_status'))}; "
            f"search_target_zone_id={_value(group.get('search_target_zone_id'))}; "
            f"searched_zone_ids={_items(_list(group.get('searched_zone_ids')))}; "
            f"seconds_since_current_command={_value(group.get('command_age_seconds'))}."
        )
    return lines


def format_map_topology(topology: Dict[str, Any]) -> str:
    """Render the static map topology block for the system prompt."""
    topology = _dict(topology)
    topo_zones = [_dict(zone) for zone in _list(topology.get("zones"))]
    primary_route = _list(topology.get("primary_route"))

    lines: List[str] = ["[Map Topology]"]
    if primary_route:
        lines.append(f"primary_route={_items(primary_route)}")
    if topo_zones:
        lines.append("zone_id|role|ramp|island|route|neighbors|path_to_enemy_main")
        for zone in topo_zones:
            adjacent = "; ".join(
                f"{item.get('zone_id')}({item.get('path_distance')})"
                for item in _list(zone.get("neighbors"))
            ) or "none"
            lines.append(
                "|".join(
                    str(value)
                    for value in (
                        _value(zone.get("zone_id")),
                        _value(zone.get("zone_role")),
                        "yes" if zone.get("has_ramp") else "no",
                        "yes" if zone.get("is_island") else "no",
                        "yes" if zone.get("on_primary_route") else "no",
                        adjacent,
                        _value(zone.get("path_distance_to_enemy_main")),
                    )
                )
            )
    else:
        lines.append("none")
    return "\n".join(lines)


def _zone_lines(army_control: Dict[str, Any]) -> List[str]:
    zones = [_dict(zone) for zone in _list(army_control.get("zones"))]

    lines: List[str] = ["[Zone State Table]"]
    if not zones:
        lines.append("none")
        return lines

    def compact_row(values: Iterable[Any]) -> str:
        return "|".join(
            str(value).replace("|", "/").replace(chr(10), " ")
            for value in values
        )

    lines.append(f"row_count={len(zones)}")
    lines.append(
        "columns="
        + compact_row(
            (
                "zone_id",
                "owner",
                "zone_role",
                "vision_state",
                "own_contents",
                "visible_enemy_contents",
                "last_seen_enemy_contents",
                "enemy_information_age_seconds",
                "under_attack",
            )
        )
    )

    owner_order = {"own": 0, "enemy": 1, "neutral": 2}
    zones.sort(
        key=lambda zone: (
            owner_order.get(str(zone.get("owner")), 3),
            _zone_number(zone.get("zone_id")),
        )
    )
    for zone in zones:
        lines.append(
            compact_row(
                (
                    _value(zone.get("zone_id")),
                    _value(zone.get("owner")),
                    _value(zone.get("zone_role")),
                    _value(zone.get("vision_state")),
                    _counts(
                        zone.get(
                            "own_non_army_contents",
                            zone.get("own_contents"),
                        )
                    ),
                    _counts(zone.get("visible_enemy_contents")),
                    _counts(zone.get("last_seen_enemy_contents")),
                    "no_enemy_record"
                    if zone.get("enemy_information_age_seconds") is None
                    else _value(zone.get("enemy_information_age_seconds")),
                    _value(zone.get("under_attack")),
                )
            )
        )
    return lines

def _zone_number(zone_id: Any) -> int:
    text = str(zone_id)
    try:
        return int(text[5:] if text.startswith("zone_") else text)
    except (TypeError, ValueError):
        return 10**9


def _scan_lines(
    capabilities: Dict[str, Any],
    *,
    include_worker_count: bool = True,
) -> List[str]:
    scan = _dict(capabilities.get("scan"))
    scout = _dict(capabilities.get("scv_scout"))
    worker_count = (
        f"total_worker_count={_value(scout.get('worker_count'))}; "
        if include_worker_count else ""
    )
    return [
        "[Orbital Scanner Sweep Capability]",
        (
            f"orbital_energy_values={_items(_list(scan.get('orbital_energies')))}; "
            f"available_scanner_sweep_count={_value(scan.get('available_scan_count'))}; "
            f"last_scan_target_zone_id={_value(scan.get('last_target_zone_id'))}; "
            f"last_scan_result={_value(scan.get('last_result'))}; "
            f"seconds_since_last_scan_result={_value(scan.get('last_result_seconds_ago'))}"
        ),
        "",
        "[SCV Scout]",
        (
            worker_count +
            f"scv_scout_active={_value(scout.get('active'))}; "
            f"active_scout_target_zone_id={_value(scout.get('active_target_zone_id'))}; "
            f"last_scout_target_zone_id={_value(scout.get('last_target_zone_id'))}; "
            f"last_scout_result={_value(scout.get('last_result'))}; "
            f"seconds_since_last_scout_result={_value(scout.get('last_result_seconds_ago'))}"
        ),
    ]


def _macro_execution_lines(
    execution: Dict[str, Any],
    *,
    include_last_tasks: bool = True,
) -> List[str]:
    macro = _dict(execution.get("macro"))
    actions = []
    for task in _list(macro.get("active_macro_tasks")):
        task = _dict(task)
        to_count = task.get("to_count", "?")
        current = task.get("current_count")
        if current is None:
            progress = f"?/{to_count}"
        else:
            progress = f"{current}/{to_count}"
        text = (
            f"action={task.get('action', '?')}, "
            f"progress={progress}"
        )
        text += f", status={task.get('status', 'active_unsatisfied')}"
        if task.get("disabled"):
            text += f" disabled({task.get('error', 'unknown')})"
        actions.append(text)
    return [
        "[Macro Execution]",
        f"execution_status={_value(macro.get('status'))}",
        *([f"last_tasks={_items(_list(macro.get('last_tasks')))}"] if include_last_tasks else []),
        f"active_macro_tasks={_items(actions)}",
        f"seconds_since_last_macro_update={_value(macro.get('last_update_seconds_ago'))}",
        f"last_issues={_items(_list(macro.get('last_issues')))}",
    ]


def _policy_text(policy: Dict[str, Any]) -> str:
    parts = []
    for command in _list(policy.get("commands")):
        command = _dict(command)
        parts.append(
            f"group_id={command.get('group_id', '?')}, "
            f"movement_mode={command.get('movement_mode', '?')}, "
            f"destination_zone_id={command.get('destination_zone_id', '?')}"
        )
    parts.append(
        f"scan_zone_id={policy.get('scan_zone_id') or 'none'}"
    )
    parts.append(
        f"scout_zone_id={policy.get('scout_zone_id') or 'none'}"
    )
    return " | ".join(parts)


def _combat_execution_lines(
    execution: Dict[str, Any],
    *,
    include_current_policy: bool = True,
    omit_idle: bool = False,
) -> List[str]:
    combat_exec = _dict(execution.get("combat"))
    status = str(combat_exec.get("status") or "idle")
    issues = _list(combat_exec.get("last_command_issues"))
    policy = _dict(combat_exec.get("last_policy"))
    if (
        omit_idle
        and status in {"idle", ""}
        and not issues
        and not policy
        and combat_exec.get("policy_age_seconds") is None
    ):
        # Live group commands already cover army orders; idle block is noise.
        return []
    return [
        "[Combat Execution]",
        f"execution_status={_value(combat_exec.get('status'))}",
        *([f"current_policy={_policy_text(policy) if policy else 'none'}"] if include_current_policy else []),
        f"seconds_since_current_policy_applied={_value(combat_exec.get('policy_age_seconds'))}",
        f"last_command_issues={_items(issues)}",
    ]


def _previous_decision_lines(execution: Dict[str, Any]) -> List[str]:
    """Render the last Commander tool-call decision (commands only)."""
    previous = _dict(execution.get("previous_decision"))
    if not previous:
        return []
    macro_bits = []
    for item in _list(previous.get("macro_commands")):
        item = _dict(item)
        name = item.get("name")
        if not name:
            continue
        if item.get("to_count") is None:
            macro_bits.append(str(name))
        else:
            macro_bits.append(f"{name}->{item.get('to_count')}")
    army_bits = []
    for item in _list(previous.get("army_commands")):
        item = _dict(item)
        army_bits.append(
            f"{item.get('group_id', '?')}:"
            f"{item.get('movement_mode', '?')}->"
            f"{item.get('destination_zone_id', '?')}"
        )
    wake = _dict(previous.get("wake_event"))
    wake_text = "none"
    if wake:
        conditions = []
        for cond in _list(wake.get("conditions")):
            cond = _dict(cond)
            ctype = cond.get("type") or "?"
            extras = []
            for key in ("unit", "count", "status", "zone", "seconds"):
                if cond.get(key) is not None:
                    extras.append(f"{key}={cond.get(key)}")
            conditions.append(
                ctype if not extras else f"{ctype}({', '.join(extras)})"
            )
        wake_text = (
            f"logic={wake.get('logic') or 'any'}; "
            f"conditions={_items(conditions)}"
        )
    return [
        "[Previous Decision]",
        f"game_time_seconds={_value(previous.get('game_time_seconds'))}",
        f"macro_commands={_items(macro_bits)}",
        f"army_commands={_items(army_bits)}",
        f"scan_zone_id={_value(previous.get('scan_zone_id'))}",
        f"scout_zone_id={_value(previous.get('scout_zone_id'))}",
        f"wake_event={wake_text}",
        f"issues={_items(_list(previous.get('issues')))}",
    ]


def _execution_history_lines(execution: Dict[str, Any]) -> List[str]:
    # Prefer the explicit previous Commander decision commands when present.
    previous_lines = _previous_decision_lines(execution)
    if previous_lines:
        return previous_lines
    history = _dict(execution.get("since_last_decision"))
    macro_records = [_dict(item) for item in _list(history.get("macro"))]
    combat_records = [_dict(item) for item in _list(history.get("combat"))]
    if not macro_records and not combat_records:
        return []
    lines = [
        "[Macro Decisions Since Previous Decision]",
        (
            "window_start_game_time_seconds="
            f"{_value(history.get('window_start_game_time_seconds'))}; "
            f"decision_count={len(macro_records)}"
        ),
    ]
    for index, event in enumerate(macro_records, start=1):
        lines.append(
            f"decision_{index}: game_time_seconds={_value(event.get('game_time_seconds'))}; "
            f"status={_value(event.get('status'))}; "
            f"tasks={_items(_list(event.get('tasks')))}; "
            f"issues={_items(_list(event.get('issues')))}"
        )
    if not macro_records:
        lines.append("none")

    lines.extend([
        "",
        "[Combat Decisions Since Previous Decision]",
        f"decision_count={len(combat_records)}",
    ])
    for index, event in enumerate(combat_records, start=1):
        policy = {
            "commands": _list(event.get("commands")),
            "scan_zone_id": event.get("scan_zone_id"),
            "scout_zone_id": event.get("scout_zone_id"),
        }
        lines.append(
            f"decision_{index}: game_time_seconds={_value(event.get('game_time_seconds'))}; "
            f"status={_value(event.get('status'))}; "
            f"applied={_value(event.get('applied'))}; "
            f"policy={_policy_text(policy)}; "
            f"issues={_items(_list(event.get('issues')))}"
        )
    if not combat_records:
        lines.append("none")
    return lines


def _training_unit_counts(active_queues: Any) -> Dict[str, int]:
    """Parse ``Training UNIT`` queue keys into unit-name counts (no workers)."""
    workerish = {"SCV", "PROBE", "DRONE", "MULE"}
    out: Dict[str, int] = {}
    for key, count in _dict(active_queues).items():
        text = str(key)
        if not text.startswith("Training "):
            continue
        name = text[len("Training ") :].strip()
        if not name or name in workerish:
            continue
        try:
            amount = int(count)
        except (TypeError, ValueError):
            continue
        if amount > 0:
            out[name] = out.get(name, 0) + amount
    return out


def _own_forces_summary_line(
    own: Dict[str, Any],
    army_control: Dict[str, Any],
) -> str:
    """Show living vs in-training combat units as separate fields."""
    del army_control
    living = _counts(own.get("combat_composition"))
    training = _counts(own.get("training_combat_composition"))
    return (
        f"army_supply={_value(own.get('army_supply'))}; "
        f"living_combat_unit_composition={living}; "
        f"training_combat_unit_composition={training} "
        f"(living = on-field ready combat units; training = still in queues)"
    )


def _append_section(lines: List[str], section_lines: List[str]) -> None:
    if not section_lines:
        return
    if lines and lines[-1] != "":
        lines.append("")
    lines.extend(section_lines)


def _render_full(view: Dict[str, Any]) -> str:
    time = _dict(view.get("time"))
    economy = _dict(view.get("economy"))
    own = _dict(view.get("own_forces"))
    production = _dict(view.get("production"))
    technology = _dict(view.get("technology"))
    enemy = _dict(view.get("enemy"))
    flags = _dict(view.get("threat_flags"))
    army = _dict(view.get("army_control"))
    capabilities = _dict(view.get("capabilities"))
    execution = _dict(view.get("execution"))
    map_control = _dict(view.get("map_control"))
    combat = _dict(view.get("combat"))
    lines: List[str] = [
        "[Game]",
        _game_time_line(
            time,
            extra_fields=(
                f"own_race={_value(own.get('race'))}; "
                f"enemy_race={_value(enemy.get('race'))}"
            ),
        ),
        "",
        "[Economy]",
        _economy_line(economy),
        "",
        "[Map Control]",
        f"own_base_count={_value(map_control.get('own_base_count'))}; active_mining_base_count={_value(map_control.get('active_mining_base_count'))}; known_enemy_base_count={_value(map_control.get('known_enemy_base_count'))}; neutral_expansion_count={_value(map_control.get('neutral_expansion_count'))}",
        *_base_resource_lines(map_control),
        "",
        "[Production]",
        *_production_lines(production),
        "",
        "[Technology]",
        f"completed_upgrades={_items(_list(technology.get('completed_upgrades')))}",
        f"upgrades_in_progress={_items(_list(technology.get('upgrades_in_progress')))}",
        "",
        "[Own Forces]",
        _own_forces_summary_line(own, army),
        "",
        "[Enemy Intelligence]",
        *_enemy_lines(enemy, include_base_count=False),
        "",
        "[Combat Analysis]",
        _combat_line(combat),
        "",
        "[Threats]",
        _threat_line(flags),
        "",
        "[Army Groups]",
        *_group_lines(army),
        "",
        "[Army Zones]",
        *_zone_lines(army),
    ]
    _append_section(
        lines, _scan_lines(capabilities, include_worker_count=False)
    )
    _append_section(
        lines, _macro_execution_lines(execution, include_last_tasks=False)
    )
    _append_section(
        lines,
        _combat_execution_lines(
            execution, include_current_policy=False, omit_idle=True
        ),
    )
    _append_section(lines, _execution_history_lines(execution))
    return "\n".join(lines)


def _base_resource_lines(map_control: Dict[str, Any]) -> List[str]:
    minerals = _dict(map_control.get("own_base_minerals"))
    gas = _dict(map_control.get("own_base_gas"))
    mineral_details = [_dict(item) for item in _list(minerals.get("details"))]
    gas_details = [_dict(item) for item in _list(gas.get("details"))]
    mineral_records = [
        (
            f"base_label={item.get('label', item.get('zone_index', '?'))}, "
            f"zone_index={_value(item.get('zone_index'))}, "
            f"mineral_status={item.get('resources', 'unknown')}, "
            f"minerals_remaining={item.get('minerals_left', 0)}"
        )
        for item in mineral_details
    ]
    gas_records = [
        (
            f"base_label={item.get('label', item.get('zone_index', '?'))}, "
            f"zone_index={_value(item.get('zone_index'))}, "
            f"owned_gas_structure_count={_value(item.get('geysers', 0))}, "
            f"vespene_gas_remaining={_value(item.get('gas_left', 0))}"
        )
        for item in gas_details
    ]
    return [
        "own_base_mineral_details="
        + (" | ".join(mineral_records) if mineral_records else "none"),
        "own_base_vespene_gas_details="
        + (" | ".join(gas_records) if gas_records else "none"),
    ]

def _render_macro(view: Dict[str, Any]) -> str:
    time = _dict(view.get("time"))
    economy = _dict(view.get("economy"))
    own = _dict(view.get("own_forces"))
    enemy = _dict(view.get("enemy"))
    technology = _dict(view.get("technology"))
    map_control = _dict(view.get("map_control"))
    lines = [
        "[Game]",
        _game_time_line(time),
        "",
        "[Economy]",
        _economy_line(economy),
        "",
        "[Production]",
        *_production_lines(_dict(view.get("production"))),
        "",
        "[Technology]",
        f"completed_upgrades={_items(_list(technology.get('completed_upgrades')))}",
        f"upgrades_in_progress={_items(_list(technology.get('upgrades_in_progress')))}",
        "",
        "[Strategic Situation]",
        (
            f"army_supply={_value(own.get('army_supply'))}; "
            f"living_combat_unit_composition={_counts(own.get('combat_composition'))}; "
            f"training_combat_unit_composition={_counts(own.get('training_combat_composition'))} "
            f"(living = on-field ready combat units; training = still in queues)"
        ),
        "",
        "[Map Control And Base Resources]",
        f"own_base_count={_value(map_control.get('own_base_count'))}; "
        f"active_mining_base_count={_value(map_control.get('active_mining_base_count'))}; "
        f"known_enemy_base_count={_value(map_control.get('known_enemy_base_count'))}; "
        f"neutral_expansion_count={_value(map_control.get('neutral_expansion_count'))}",
        *_base_resource_lines(map_control),
        "",
        "[Enemy Intelligence]",
        *_enemy_lines(enemy, include_base_count=False),
        "",
        *_macro_execution_lines(_dict(view.get("execution")), include_last_tasks=False),
    ]
    return "\n".join(lines)


def _render_combat(view: Dict[str, Any]) -> str:
    time = _dict(view.get("time"))
    own = _dict(view.get("own_forces"))
    readiness = _dict(view.get("military_readiness"))
    technology = _dict(readiness.get("technology"))
    army = _dict(view.get("army_control"))
    lines: List[str] = [
        "[Game]",
        _game_time_line(
            time,
            extra_fields=(
                f"own_race={_value(own.get('race'))}; "
                f"enemy_race={_value(_dict(view.get('enemy')).get('race'))}"
            ),
        ),
        "",
        "[Military]",
        (
            f"army_supply={_value(own.get('army_supply'))}; "
            f"supply_used={_value(own.get('supply_used'))}; "
            f"supply_capacity={_value(own.get('supply_cap'))}; "
            f"supply_available={_value(own.get('supply_free'))}; "
            f"living_combat_unit_composition={_counts(own.get('combat_composition'))}; "
            f"training_combat_unit_composition={_counts(own.get('training_combat_composition'))} "
            f"(living = on-field ready combat units; training = still in queues)"
        ),
        "",
        "[Enemy Intelligence]",
        *_enemy_lines(_dict(view.get("enemy"))),
        "",
        "[Combat Analysis]",
        _combat_line(_dict(view.get("combat"))),
        "",
        "[Military Readiness]",
        f"units_and_structures_under_construction={_counts(readiness.get('under_construction'))}",
        f"completed_upgrades={_items(_list(technology.get('completed_upgrades')))}; upgrades_in_progress={_items(_list(technology.get('upgrades_in_progress')))}",
        "",
        "[Threats]",
        _threat_line(_dict(view.get("threat_flags"))),
        "",
        "[Army Groups]",
        *_group_lines(army),
        "",
        "[Army Zones]",
        *_zone_lines(army),
    ]
    _append_section(
        lines,
        _scan_lines(_dict(view.get("capabilities")), include_worker_count=False),
    )
    _append_section(
        lines,
        _combat_execution_lines(
            _dict(view.get("execution")),
            include_current_policy=False,
            omit_idle=True,
        ),
    )
    return "\n".join(lines)
