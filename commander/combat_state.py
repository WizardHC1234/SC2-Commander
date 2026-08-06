"""Army observation state + cleanup runtime hints for Commander.

Collects live combat/zone/scout state for observations and wake polling.
Emits optional [Runtime Search-And-Destroy Hint] when cleanup conditions hold.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sc2.ids.unit_typeid import UnitTypeId


logger = logging.getLogger("commander.combat_state")


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
            ""
            "Once search_and_destroy has started, keep all combat groups in "
            "search_and_destroy for the rest of the game.",
        ]
    )


def _living_combat_unit_counts(act: Any) -> Dict[str, int]:
    """Ready combat units on the field only (never buildings / never in queue).

    Morph variants are collapsed via real_types (e.g. SIEGETANKSIEGED
    counts as SIEGETANK) so composition matches how strategy gates are written.
    """
    ai = getattr(act, "ai", None)
    unit_values = getattr(act, "unit_values", None)
    if ai is None or unit_values is None:
        return {}
    try:
        units = list(getattr(ai, "all_own_units", []) or [])
    except (AttributeError, TypeError):
        return {}
    counts: Dict[str, int] = {}
    for unit in units:
        if getattr(unit, "is_structure", False):
            continue
        if not bool(getattr(unit, "is_ready", True)):
            continue
        try:
            if not unit_values.should_attack(unit):
                continue
        except Exception:
            continue
        type_id = getattr(unit, "type_id", None)
        name = _canonical_unit_type_name(type_id)
        counts[name] = counts.get(name, 0) + 1
    return counts


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
        "own_unit_type_counts": _living_combat_unit_counts(act),
        "close_enemy_type_counts": _unit_counts(close_enemies),
        "known_enemy_type_counts": _unit_counts(enemy_combat_units),
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


def _canonical_unit_type_name(type_id: Any) -> str:
    """Collapse morph variants (sieged tank, burrowed, etc.) to the base type name."""
    from sharpy.managers.core.unit_value import real_types

    if type_id in real_types:
        type_id = real_types[type_id]
    return getattr(type_id, "name", None) or str(type_id or "UNKNOWN")


def _unit_counts(units: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for unit in units:
        type_id = getattr(unit, "type_id", None)
        name = _canonical_unit_type_name(type_id)
        counts[name] = counts.get(name, 0) + 1
    return counts
