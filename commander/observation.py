"""Unified structured observations and text rendering for the Commander.

Collectors that know about Sharpy remain in their modules. This module owns
the observation schema and text rendering only; it never reads live game state.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

from commander.retreat_policy import DEFAULT_RETREAT_RATIO


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


def _norm_upgrade_token(text: Any) -> str:
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def _research_action_matches_upgrade(action: Any, upgrade: Any) -> bool:
    """Best-effort match of research_* macro keys to displayed upgrade names."""
    name = str(action or "")
    if not name.startswith("research_"):
        return False
    stem = _norm_upgrade_token(name[len("research_") :])
    up = _norm_upgrade_token(upgrade)
    if not stem or not up:
        return False
    return stem == up or stem in up or up in stem


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
    if (
        not macro_commands
        and not army_commands
        and not state.get("army_intent")
        and not state.get("wake_event")
    ):
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
        "army_intent": _dict(state.get("army_intent")) or None,
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
    previous_decision: Optional[Dict[str, Any]] = None,
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
            "producer_addons": deepcopy(
                _dict(legacy_own.get("producer_addons"))
            ),
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
            "previous_decision": _normalise_previous_decision(previous_decision),
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
) -> Dict[str, Any]:
    """Return a deep-copied observation without reading live game state."""
    view = deepcopy(_dict(full_observation))
    view["view_type"] = "full"
    return view


def render_observation(view: Dict[str, Any]) -> str:
    return _render_full(view)


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


# train_* that need a Tech Lab on a specific producer. Used only to annotate
# Macro Execution blockers for the model — never auto-injects tools.
_TRAIN_TECHLAB_REQUIREMENTS = {
    "train_marauder": ("BARRACKS", "build_barracks_techlab"),
    "train_ghost": ("BARRACKS", "build_barracks_techlab"),
    "train_siege_tank": ("FACTORY", "build_factory_techlab"),
    "train_thor": ("FACTORY", "build_factory_techlab"),
    "train_raven": ("STARPORT", "build_starport_techlab"),
    "train_banshee": ("STARPORT", "build_starport_techlab"),
    "train_battlecruiser": ("STARPORT", "build_starport_techlab"),
}

# Structure aliases counted from production completed / under_construction.
_STRUCTURE_ALIASES: Dict[str, tuple] = {
    "townhall": (
        "COMMANDCENTER",
        "ORBITALCOMMAND",
        "PLANETARYFORTRESS",
        "COMMANDCENTERFLYING",
        "ORBITALCOMMANDFLYING",
    ),
    "supply": (
        "SUPPLYDEPOT",
        "SUPPLYDEPOTLOWERED",
        "SUPPLYDEPOTDROP",
    ),
    "barracks": ("BARRACKS", "BARRACKSFLYING"),
    "factory": ("FACTORY", "FACTORYFLYING"),
    "starport": ("STARPORT", "STARPORTFLYING"),
    "armory": ("ARMORY",),
    "fusioncore": ("FUSIONCORE",),
    "engineeringbay": ("ENGINEERINGBAY",),
    "ghostacademy": ("GHOSTACADEMY",),
    "barrackstechlab": ("BARRACKSTECHLAB", "TECHLAB"),
    "factorytechlab": ("FACTORYTECHLAB", "TECHLAB"),
    "starporttechlab": ("STARPORTTECHLAB", "TECHLAB"),
}

# Hard building prerequisites for common Terran macros (ready OR pending).
# Value is (alias_key, blocker_tag) — blocker_tag appears in blocked=no_<tag>.
_MACRO_STRUCTURE_PREREQS: Dict[str, tuple] = {
    "train_scv": ("townhall", "townhall"),
    "train_marine": ("barracks", "barracks"),
    "train_marauder": ("barracks", "barracks"),
    "train_reaper": ("barracks", "barracks"),
    "train_ghost": ("barracks", "barracks"),
    "train_hellion": ("factory", "factory"),
    "train_hellbat": ("factory", "factory"),
    "train_widow_mine": ("factory", "factory"),
    "train_cyclone": ("factory", "factory"),
    "train_siege_tank": ("factory", "factory"),
    "train_thor": ("factory", "factory"),
    "train_viking": ("starport", "starport"),
    "train_medivac": ("starport", "starport"),
    "train_liberator": ("starport", "starport"),
    "train_raven": ("starport", "starport"),
    "train_banshee": ("starport", "starport"),
    "train_battlecruiser": ("starport", "starport"),
    "build_barracks": ("supply", "supply_depot"),
    "build_factory": ("barracks", "barracks"),
    "build_starport": ("factory", "factory"),
    "build_engineering_bay": ("supply", "supply_depot"),
    "build_armory": ("factory", "factory"),
    "build_fusion_core": ("starport", "starport"),
    "build_ghost_academy": ("barracks", "barracks"),
    "build_gas": ("townhall", "townhall"),
    "build_barracks_techlab": ("barracks", "barracks"),
    "build_barracks_reactor": ("barracks", "barracks"),
    "build_factory_techlab": ("factory", "factory"),
    "build_factory_reactor": ("factory", "factory"),
    "build_starport_techlab": ("starport", "starport"),
    "build_starport_reactor": ("starport", "starport"),
    "morph_orbital_command": ("barracks", "barracks"),
    "research_shieldwall": ("barrackstechlab", "barracks_techlab"),
    "research_stimpack": ("barrackstechlab", "barracks_techlab"),
    "research_concussive_shells": ("barrackstechlab", "barracks_techlab"),
    "research_yamato_cannon": ("fusioncore", "fusion_core"),
}

# Extra tech-building gates beyond the producer itself.
_TRAIN_TECH_BUILDING_REQUIREMENTS = {
    "train_thor": ("armory", "armory"),
    "train_battlecruiser": ("fusioncore", "fusion_core"),
    "train_ghost": ("ghostacademy", "ghost_academy"),
}

# build_*_techlab / build_*_reactor soft-failure diagnostics.
# (producer_key, addon_kind, producer_alias, pending_addon_keys)
_ADDON_BUILD_SPECS: Dict[str, tuple] = {
    "build_barracks_techlab": (
        "BARRACKS",
        "techlab",
        "barracks",
        ("BARRACKSTECHLAB", "TECHLAB"),
    ),
    "build_barracks_reactor": (
        "BARRACKS",
        "reactor",
        "barracks",
        ("BARRACKSREACTOR", "REACTOR"),
    ),
    "build_factory_techlab": (
        "FACTORY",
        "techlab",
        "factory",
        ("FACTORYTECHLAB", "TECHLAB"),
    ),
    "build_factory_reactor": (
        "FACTORY",
        "reactor",
        "factory",
        ("FACTORYREACTOR", "REACTOR"),
    ),
    "build_starport_techlab": (
        "STARPORT",
        "techlab",
        "starport",
        ("STARPORTTECHLAB", "TECHLAB"),
    ),
    "build_starport_reactor": (
        "STARPORT",
        "reactor",
        "starport",
        ("STARPORTREACTOR", "REACTOR"),
    ),
}


def _structure_counts(
    production: Dict[str, Any], alias_key: str
) -> tuple:
    """Return (completed, pending) counts for a structure alias group."""
    names = _STRUCTURE_ALIASES.get(alias_key, ())
    completed_map = _dict(production.get("completed"))
    pending_map = _dict(production.get("under_construction"))
    # Flying/morphing producers still count as pending capacity for gates.
    en_route = _dict(production.get("workers_en_route"))
    completed = 0
    pending = 0
    for name in names:
        try:
            completed += int(completed_map.get(name) or 0)
        except (TypeError, ValueError):
            pass
        try:
            pending += int(pending_map.get(name) or 0)
        except (TypeError, ValueError):
            pass
        try:
            pending += int(en_route.get(name) or 0)
        except (TypeError, ValueError):
            pass
    return completed, pending


def _production_lines(production: Dict[str, Any]) -> List[str]:
    lines = [
        f"completed_units_and_structures={_counts(production.get('completed'))}",
        f"units_and_structures_under_construction={_counts(production.get('under_construction'))}",
        f"workers_en_route_to_build={_counts(production.get('workers_en_route'))}",
        f"active_production_and_research_queues={_counts(production.get('active_queues'))}",
    ]
    addon_bits: List[str] = []
    for producer, stats in sorted(_dict(production.get("producer_addons")).items()):
        stats = _dict(stats)
        addon_bits.append(
            f"{producer} ready={_value(stats.get('ready'))} "
            f"techlab={_value(stats.get('with_techlab'))} "
            f"reactor={_value(stats.get('with_reactor'))} "
            f"no_addon={_value(stats.get('no_addon'))}"
        )
    if addon_bits:
        lines.append(f"producer_addons={_items(addon_bits)}")
    return lines


def _train_techlab_block_reason(
    action: Any,
    production: Dict[str, Any],
) -> Optional[str]:
    """Hard miss only: no techlab and none building. Pending is waiting, not failure."""
    req = _TRAIN_TECHLAB_REQUIREMENTS.get(str(action or ""))
    if not req:
        return None
    producer, _techlab_tool = req
    stats = _dict(_dict(production.get("producer_addons")).get(producer))
    with_techlab = 0
    try:
        with_techlab = int(stats.get("with_techlab") or 0)
    except (TypeError, ValueError):
        with_techlab = 0
    if with_techlab > 0:
        return None
    pending = 0
    under = _dict(production.get("under_construction"))
    pending_keys = {
        "BARRACKS": ("BARRACKSTECHLAB", "TECHLAB"),
        "FACTORY": ("FACTORYTECHLAB", "TECHLAB"),
        "STARPORT": ("STARPORTTECHLAB", "TECHLAB"),
    }.get(producer, ())
    for key in pending_keys:
        try:
            pending += int(under.get(key) or 0)
        except (TypeError, ValueError):
            pass
    if pending > 0:
        return None
    return f"blocked=no_{producer.lower()}_techlab"


def _int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _addon_soft_block_reason(
    action: Any,
    production: Dict[str, Any],
    *,
    to_count: Any = None,
    current_count: Any = None,
) -> Optional[str]:
    """Soft fail: addon target unmet and no free producer slot (and none pending)."""
    spec = _ADDON_BUILD_SPECS.get(str(action or ""))
    if not spec:
        return None
    producer, addon_kind, producer_tag, pending_keys = spec
    stats = _dict(_dict(production.get("producer_addons")).get(producer))
    no_addon = _int_or_none(stats.get("no_addon")) or 0
    with_techlab = _int_or_none(stats.get("with_techlab")) or 0
    with_reactor = _int_or_none(stats.get("with_reactor")) or 0

    target = _int_or_none(to_count)
    current = _int_or_none(current_count)
    if current is None:
        current = with_reactor if addon_kind == "reactor" else with_techlab
    if target is None or current >= target:
        return None

    under = _dict(production.get("under_construction"))
    pending_addon = 0
    for key in pending_keys:
        pending_addon += _int_or_none(under.get(key)) or 0
    if pending_addon > 0:
        return None  # addon already building — waiting, not failure

    _completed_producers, pending_producers = _structure_counts(
        production, producer_tag
    )
    if pending_producers > 0:
        return None  # new producer incoming — may free an addon slot

    if no_addon > 0:
        return None  # free slot exists; BuildAddon can still attempt

    # All ready producers already have addons; need another producer building.
    return f"blocked=need_more_{producer_tag}"


def _macro_prereq_block_reason(
    action: Any,
    production: Dict[str, Any],
    *,
    to_count: Any = None,
    current_count: Any = None,
) -> Optional[str]:
    """Obs-only block labels. Hard miss first, then addon soft-fail."""
    name = str(action or "")
    if not name:
        return None

    prereq = _MACRO_STRUCTURE_PREREQS.get(name)
    if prereq:
        alias_key, tag = prereq
        completed, pending = _structure_counts(production, alias_key)
        if completed <= 0 and pending <= 0:
            return f"blocked=no_{tag}"
        # Producer still building / en route: waiting, not a failed execution.
        if completed <= 0:
            return None

    tech_building = _TRAIN_TECH_BUILDING_REQUIREMENTS.get(name)
    if tech_building:
        alias_key, tag = tech_building
        completed, pending = _structure_counts(production, alias_key)
        if completed <= 0 and pending <= 0:
            return f"blocked=no_{tag}"
        if completed <= 0:
            return None

    hard_techlab = _train_techlab_block_reason(name, production)
    if hard_techlab:
        return hard_techlab

    return _addon_soft_block_reason(
        name,
        production,
        to_count=to_count,
        current_count=current_count,
    )


def _planned_research_actions(execution: Dict[str, Any]) -> List[Dict[str, Any]]:
    macro = _dict(execution.get("macro"))
    planned: List[Dict[str, Any]] = []
    for task in _list(macro.get("active_macro_tasks")):
        task = _dict(task)
        action = str(task.get("action") or "")
        if not action.startswith("research_"):
            continue
        planned.append(task)
    return planned


def _technology_lines(
    technology: Dict[str, Any],
    execution: Dict[str, Any],
) -> List[str]:
    """Render tech evidence; tag in-progress upgrades vs current research_* plan."""
    completed = [str(item) for item in _list(technology.get("completed_upgrades"))]
    in_progress = [
        str(item) for item in _list(technology.get("upgrades_in_progress"))
    ]
    planned = _planned_research_actions(execution)

    progress_bits: List[str] = []
    for upgrade in in_progress:
        matches = [
            str(task.get("action"))
            for task in planned
            if _research_action_matches_upgrade(task.get("action"), upgrade)
        ]
        if matches:
            progress_bits.append(f"{upgrade}(plan={matches[0]})")
        else:
            progress_bits.append(f"{upgrade}(plan=omitted)")

    return [
        "[Technology]",
        f"completed_upgrades={_items(completed)}",
        f"upgrades_in_progress={_items(progress_bits)}",
    ]


def _macro_execution_lines(
    execution: Dict[str, Any],
    production: Optional[Dict[str, Any]] = None,
) -> List[str]:
    # last_tasks duplicate [Previous Decision] macro_commands; omit them.
    macro = _dict(execution.get("macro"))
    production = _dict(production)
    actions = []
    for task in _list(macro.get("active_macro_tasks")):
        task = _dict(task)
        to_count = task.get("to_count", "?")
        current = task.get("current_count")
        progress = f"?/{to_count}" if current is None else f"{current}/{to_count}"
        status = str(task.get("status") or "active_unsatisfied")
        text = (
            f"action={task.get('action', '?')}, "
            f"progress={progress}, "
            f"status={status}"
        )
        # Annotate only when the task is still trying and cannot execute
        # because a hard prerequisite is missing (not while waiting on pending).
        if status in {"active_unsatisfied", "failed"}:
            try:
                done = current is not None and int(current) >= int(to_count)
            except (TypeError, ValueError):
                done = False
            if not done:
                block = _macro_prereq_block_reason(
                    task.get("action"),
                    production,
                    to_count=to_count,
                    current_count=current,
                )
                if block:
                    text += f", {block}"
        if task.get("disabled"):
            text += f", disabled({task.get('error', 'unknown')})"
        elif task.get("error") and status == "failed":
            text += f", error({task.get('error')})"
        actions.append(text)
    return [
        "[Macro Execution]",
        f"execution_status={_value(macro.get('status'))}",
        f"active_macro_tasks={_items(actions)}",
        f"seconds_since_last_macro_update={_value(macro.get('last_update_seconds_ago'))}",
        f"last_issues={_items(_list(macro.get('last_issues')))}",
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

def _enemy_lines(enemy: Dict[str, Any]) -> List[str]:
    # known_enemy_base_count already appears under [Map Control].
    # Nearby enemies for decisions live under [Army Groups]; do not repeat a
    # coarse near-army composition here.
    return [
        f"known_enemy_units_and_buildings={_counts(enemy.get('known_composition'))}",
        f"known_enemy_combat_units={_counts(enemy.get('known_combat_composition'))}",
        f"enemy_macro_build={_value(enemy.get('macro_build'))}",
        (
            f"seconds_since_enemy_army_last_seen="
            f"{_value(enemy.get('seconds_since_last_seen'))}"
        ),
    ]


def _group_lines(army_control: Dict[str, Any]) -> List[str]:
    """Force state only; command progress is under [Combat Execution]."""
    groups = [_dict(group) for group in _list(army_control.get("groups"))]
    if not groups:
        return ["none"]
    lines = []
    for group in groups:
        lines.append(
            f"group_id={_value(group.get('group_id'))}; role={_value(group.get('role'))}; "
            f"unit_count={_value(group.get('unit_count', 0))}; "
            f"unit_composition={_counts(group.get('unit_type_counts'))}; "
            f"army_group_power={_value(group.get('power'))}; "
            f"nearest_zone_id={_value(group.get('nearest_zone_id'))}; "
            f"is_fragmented={_value(group.get('is_fragmented'))}; "
            f"nearby_enemy_count={_value(group.get('nearby_enemy_count', 0))}; "
            f"nearby_enemy_army_power={_value(group.get('nearby_enemy_power', 0.0))}; "
            f"nearby_enemy_composition={_counts(group.get('nearby_enemy_type_counts'))}."
        )
    return lines


def format_map_topology(topology: Dict[str, Any]) -> str:
    """Render the static map topology block for the system prompt."""
    topology = _dict(topology)
    topo_zones = [_dict(zone) for zone in _list(topology.get("zones"))]
    primary_route = _list(topology.get("primary_route"))

    lines: List[str] = ["[8] Map Topology"]
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


def _scan_lines(capabilities: Dict[str, Any]) -> List[str]:
    """Scanner + SCV scout in one recon block."""
    scan = _dict(capabilities.get("scan"))
    scout = _dict(capabilities.get("scv_scout"))
    return [
        "[Recon]",
        (
            f"scanner: orbital_energy_values={_items(_list(scan.get('orbital_energies')))}; "
            f"available_scanner_sweep_count={_value(scan.get('available_scan_count'))}; "
            f"last_scan_target_zone_id={_value(scan.get('last_target_zone_id'))}; "
            f"last_scan_result={_value(scan.get('last_result'))}; "
            f"seconds_since_last_scan_result={_value(scan.get('last_result_seconds_ago'))}"
        ),
        (
            f"scv_scout: scv_scout_active={_value(scout.get('active'))}; "
            f"active_scout_target_zone_id={_value(scout.get('active_target_zone_id'))}; "
            f"last_scout_target_zone_id={_value(scout.get('last_target_zone_id'))}; "
            f"last_scout_result={_value(scout.get('last_result'))}; "
            f"seconds_since_last_scout_result={_value(scout.get('last_result_seconds_ago'))}"
        ),
    ]


def _combat_execution_lines(army_control: Dict[str, Any]) -> List[str]:
    """Command progress for army decisions (analogous to Macro Execution)."""
    groups = [_dict(group) for group in _list(army_control.get("groups"))]
    lines: List[str] = ["[Combat Execution]"]
    if not groups:
        lines.append("active_army_commands=none")
        return lines
    for group in groups:
        command = _dict(group.get("current_command"))
        if not command:
            lines.append(f"{_value(group.get('group_id'))}: no_active_command")
            continue
        retreat = command.get("retreat_ratio")
        retreat_text = (
            _value(retreat)
            if retreat is not None
            else f"default({DEFAULT_RETREAT_RATIO})"
        )
        bits = [
            f"{_value(group.get('group_id'))}: "
            f"{_value(command.get('movement_mode'))}"
            f"->{_value(command.get('destination_zone_id'))}",
            f"retreat_ratio={retreat_text}",
            f"reached={_value(group.get('current_destination_reached'))}",
            f"objective={_value(group.get('current_objective_status'))}",
            f"age_s={_value(group.get('command_age_seconds'))}",
            f"source={_value(group.get('command_source') or 'llm')}",
        ]
        if group.get("policy_state"):
            bits.append(f"override={_value(group.get('policy_state'))}")
            if group.get("blocked_mode"):
                bits.append(f"blocked_mode={_value(group.get('blocked_mode'))}")
            if group.get("policy_detail"):
                bits.append(f"detail={_value(group.get('policy_detail'))}")
        if group.get("search_target_zone_id"):
            bits.append(
                f"search_target={_value(group.get('search_target_zone_id'))}"
            )
        searched = _list(group.get("searched_zone_ids"))
        if searched:
            bits.append(f"searched={_items(searched)}")
        lines.append("; ".join(bits))
    return lines


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
    intent = _dict(previous.get("army_intent"))
    for item in _list(previous.get("army_commands")):
        item = _dict(item)
        ratio = item.get("retreat_ratio")
        ratio_text = f"(r{ratio})" if ratio is not None else ""
        army_bits.append(
            f"{item.get('group_id', '?')}:"
            f"{item.get('movement_mode', '?')}->"
            f"{item.get('destination_zone_id', '?')}"
            f"{ratio_text}"
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
        (
            f"army_intent={_value(intent.get('mode'))}->"
            f"{_value(intent.get('zone_id'))}"
            if intent
            else "army_intent=none"
        ),
        f"scan_zone_id={_value(previous.get('scan_zone_id'))}",
        f"scout_zone_id={_value(previous.get('scout_zone_id'))}",
        f"wake_event={wake_text}",
        f"issues={_items(_list(previous.get('issues')))}",
    ]


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
        f"(living=on-field ready combat; training=still in queues)"
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
    army = _dict(view.get("army_control"))
    capabilities = _dict(view.get("capabilities"))
    execution = _dict(view.get("execution"))
    map_control = _dict(view.get("map_control"))
    combat = _dict(view.get("combat"))
    # Order: match/economy → build/macro → own army → enemy/space → recon → history.
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
    ]
    _append_section(lines, _macro_execution_lines(execution, production))
    lines.extend(
        [
            "",
            *_technology_lines(technology, execution),
        ]
    )
    lines.extend(
        [
            "",
            "[Own Forces]",
            _own_forces_summary_line(own, army),
            "",
            "[Army Groups]",
            *_group_lines(army),
        ]
    )
    _append_section(lines, _combat_execution_lines(army))
    lines.extend(
        [
            "",
            "[Enemy Intelligence]",
            *_enemy_lines(enemy),
            "",
            "[Combat Analysis]",
            _combat_line(combat),
            "",
            "[Army Zones]",
            *_zone_lines(army),
        ]
    )
    _append_section(lines, _scan_lines(capabilities))
    _append_section(lines, _previous_decision_lines(execution))
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
    available_total = 0
    gas_records = []
    for item in gas_details:
        owned = int(item.get("owned_gas_structure_count", item.get("geysers", 0)) or 0)
        slots = int(item.get("geyser_slots", owned) or 0)
        available = int(item.get("available_geyser_slots", max(0, slots - owned)) or 0)
        available_total += available
        gas_records.append(
            (
                f"base_label={item.get('label', item.get('zone_index', '?'))}, "
                f"zone_index={_value(item.get('zone_index'))}, "
                f"owned_gas_structure_count={owned}, "
                f"geyser_slots={slots}, "
                f"available_geyser_slots={available}, "
                f"vespene_gas_remaining={_value(item.get('gas_left', 0))}"
            )
        )
    return [
        "own_base_mineral_details="
        + (" | ".join(mineral_records) if mineral_records else "none"),
        f"available_geyser_slots_total={available_total}",
        "own_base_vespene_gas_details="
        + (" | ".join(gas_records) if gas_records else "none"),
    ]
