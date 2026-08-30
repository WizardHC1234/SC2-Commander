"""Bounded support-aware combat estimates for EvolAgent.

Sharpy's direct army power intentionally focuses on units that attack.  That is
useful for runtime control, but it omits the sustained-combat value of support
units such as the Terran Medivac.  This module keeps direct power unchanged and
adds a small, explicit, capped estimate of healing that can be inspected by the
analysis and optimization agents.

The result is a comparison aid, not a deterministic battle simulator.  Terrain,
focus fire, upgrades, spell use, positioning, and whether the support unit is
actually near the fight still have to be checked against match trajectories.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Iterable, Mapping

from sc2.ids.unit_typeid import UnitTypeId

from sharpy.general.unit_feature import UnitFeature
from sharpy.managers.core.unit_value import UnitValue


SUPPORT_POWER_SCHEMA = "evol.support_aware_power.v1"

# The bundled SC2 knowledge base records 12.6 HP/s for Medivac Heal.  Eight
# seconds stays inside the initial-energy budget and represents one meaningful
# exchange without pretending to simulate a complete battle.
MEDIVAC_HEAL_PER_SECOND = 12.6
DEFAULT_ENGAGEMENT_HORIZON_SECONDS = 8.0
HEALING_UTILIZATION = 0.80
MAX_BIO_POWER_BONUS_FRACTION = 0.35
MIN_SURVIVAL_FACTOR_WITH_ANTI_AIR = 0.55

# Maximum health is needed only to convert bounded healing into the same coarse
# power units used by Sharpy.  These are all Terran biological combat forms that
# Medivacs can heal in the current executor.
_HEALABLE_MAX_HEALTH = {
    "MARINE": 45.0,
    "MARAUDER": 125.0,
    "REAPER": 60.0,
    "GHOST": 100.0,
    "HELLIONTANK": 135.0,
}

_ACTION_UNIT_ALIASES = {
    "SIEGETANK": "SIEGETANK",
    "HELLBAT": "HELLIONTANK",
    "VIKING": "VIKINGFIGHTER",
    "WIDOWMINE": "WIDOWMINE",
}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _canonical_token(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _unit_name(value: Any) -> str:
    token = _canonical_token(value)
    return _ACTION_UNIT_ALIASES.get(token, token)


def _counts(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for raw_name, raw_count in value.items():
        name = _unit_name(raw_name)
        count = _number(raw_count)
        if name and count is not None and count > 0:
            result[name] = result.get(name, 0.0) + count
    return result


@lru_cache(maxsize=1)
def _sharpy_values() -> UnitValue:
    # UnitValue's static unit_data is available without starting a game or
    # assigning a BotAI instance.
    return UnitValue()


def _unit_data(name: str) -> Any | None:
    unit_type = UnitTypeId.__members__.get(_unit_name(name))
    if unit_type is None:
        return None
    return _sharpy_values().unit_data.get(unit_type)


def _combat_value(name: str) -> float:
    data = _unit_data(name)
    value = _number(getattr(data, "combat_value", None))
    return value or 0.0


def _direct_power(composition: Mapping[str, float]) -> float:
    # Medivac's Sharpy combat_value is deliberately not used as direct power:
    # its contribution is represented by the bounded healing term below.
    return sum(
        count * _combat_value(name)
        for name, count in composition.items()
        if name != "MEDIVAC"
    )


def _enemy_anti_air_power(composition: Mapping[str, float]) -> float:
    power = 0.0
    for name, count in composition.items():
        data = _unit_data(name)
        features = list(getattr(data, "features", []) or [])
        if UnitFeature.ShootsAir in features:
            power += count * _combat_value(name)
    return power


def composition_from_timing_package(package: Mapping[str, Any] | None) -> dict[str, float]:
    """Extract living gate-unit counts from a normalized timing package."""
    result: dict[str, float] = {}
    if not isinstance(package, Mapping):
        return result
    for item in package.get("gate_components") or []:
        if not isinstance(item, Mapping):
            continue
        action = str(item.get("action") or "").strip().lower()
        if not action.startswith("train_"):
            continue
        name = _unit_name(action[len("train_") :])
        count = _number(item.get("quantity"))
        if name and count is not None and count > 0:
            result[name] = result.get(name, 0.0) + count
    return result


def upgrades_from_timing_package(package: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(package, Mapping):
        return []
    upgrades: list[str] = []
    for item in package.get("setup_actions") or []:
        if not isinstance(item, Mapping):
            continue
        action = str(item.get("action") or "").strip().lower()
        if action.startswith("research_"):
            upgrades.append(action[len("research_") :])
    return upgrades


def estimate_support_aware_power(
    composition: Mapping[str, Any],
    *,
    direct_power: float | None = None,
    enemy_composition: Mapping[str, Any] | None = None,
    enemy_direct_power: float | None = None,
    completed_upgrades: Iterable[Any] | None = None,
    engagement_horizon_seconds: float = DEFAULT_ENGAGEMENT_HORIZON_SECONDS,
    direct_power_includes_fixed_support: bool = False,
) -> dict[str, Any]:
    """Estimate direct and Medivac-supported power on one common scale.

    Healing is capped by the healable biological force, by a short engagement
    horizon, and by 35% of that force's direct power.  Visible anti-air reduces
    the support term but never the underlying direct army power.
    """
    own = _counts(composition)
    enemy = _counts(enemy_composition)
    medivacs = own.get("MEDIVAC", 0.0)

    calculated_direct = _direct_power(own)
    supplied_direct = _number(direct_power)
    removed_fixed_support_power = 0.0
    if supplied_direct is not None and direct_power_includes_fixed_support:
        removed_fixed_support_power = medivacs * _combat_value("MEDIVAC")
    direct = (
        max(0.0, supplied_direct - removed_fixed_support_power)
        if supplied_direct is not None
        else calculated_direct
    )

    upgrade_tokens = {_canonical_token(item) for item in completed_upgrades or []}
    marine_bonus_hp = 10.0 if {"COMBATSHIELD", "SHIELDWALL"} & upgrade_tokens else 0.0
    bio_count = 0.0
    bio_power = 0.0
    bio_max_hp = 0.0
    for name, max_health in _HEALABLE_MAX_HEALTH.items():
        count = own.get(name, 0.0)
        if count <= 0:
            continue
        health = max_health + (marine_bonus_hp if name == "MARINE" else 0.0)
        bio_count += count
        bio_power += count * _combat_value(name)
        bio_max_hp += count * health

    horizon = max(0.0, float(engagement_horizon_seconds))
    raw_healing_hp = medivacs * MEDIVAC_HEAL_PER_SECOND * horizon
    usable_healing_hp = raw_healing_hp * HEALING_UTILIZATION
    injury_cap_hp = bio_max_hp * MAX_BIO_POWER_BONUS_FRACTION
    bounded_healing_hp = min(usable_healing_hp, injury_cap_hp)

    hp_per_power = bio_max_hp / bio_power if bio_power > 0 else None
    unrisked_bonus = (
        bounded_healing_hp / hp_per_power
        if hp_per_power and medivacs > 0
        else 0.0
    )
    unrisked_bonus = min(unrisked_bonus, bio_power * MAX_BIO_POWER_BONUS_FRACTION)

    anti_air_power = _enemy_anti_air_power(enemy)
    if anti_air_power > 0 and direct > 0:
        anti_air_ratio = anti_air_power / direct
        survival_factor = max(
            MIN_SURVIVAL_FACTOR_WITH_ANTI_AIR,
            1.0 - 0.35 * anti_air_ratio,
        )
    else:
        anti_air_ratio = 0.0
        survival_factor = 1.0
    support_bonus = unrisked_bonus * survival_factor
    adjusted = direct + support_bonus

    enemy_power = _number(enemy_direct_power)
    ratio = adjusted / enemy_power if enemy_power is not None and enemy_power > 0 else None
    confidence = "medium" if enemy else "low"
    if not medivacs or not bio_count:
        confidence = "high"

    result: dict[str, Any] = {
        "schema": SUPPORT_POWER_SCHEMA,
        "method": "bounded_medivac_sustain",
        "direct_power": round(direct, 3),
        "support_adjusted_power": round(adjusted, 3),
        "support_bonus_power": round(support_bonus, 3),
        "support_bonus_fraction": round(support_bonus / direct, 4) if direct > 0 else 0.0,
        "medivac_count": int(medivacs) if medivacs.is_integer() else round(medivacs, 3),
        "healable_biological_count": (
            int(bio_count) if bio_count.is_integer() else round(bio_count, 3)
        ),
        "engagement_horizon_seconds": round(horizon, 3),
        "effective_healing_hp": round(bounded_healing_hp * survival_factor, 3),
        "enemy_anti_air_power": round(anti_air_power, 3),
        "medivac_survival_factor": round(survival_factor, 3),
        "confidence": confidence,
    }
    if supplied_direct is not None and direct_power_includes_fixed_support:
        result["recorded_sharpy_power"] = round(supplied_direct, 3)
        result["removed_fixed_support_power"] = round(
            removed_fixed_support_power, 3
        )
    if ratio is not None:
        result["support_adjusted_to_enemy_power_ratio"] = round(ratio, 3)
    if medivacs > 0:
        result["interpretation"] = (
            "bounded sustain estimate; verify support proximity, focus fire, terrain, "
            "upgrades, and actual force retention in the trajectory"
        )
    return result


def estimate_timing_package_support_power(
    package: Mapping[str, Any] | None,
) -> dict[str, Any]:
    composition = composition_from_timing_package(package)
    return estimate_support_aware_power(
        composition,
        completed_upgrades=upgrades_from_timing_package(package),
    )


def estimate_observation_support_power(
    observation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Build an estimate from one recorded Commander observation.

    Commander combat groups omit Medivacs, so their living count is recovered
    from completed_counts/production while all attacking-unit counts still come
    from combat_composition.
    """
    if not isinstance(observation, Mapping):
        return None
    own = observation.get("own_forces")
    own = own if isinstance(own, Mapping) else {}
    production = observation.get("production")
    production = production if isinstance(production, Mapping) else {}
    combat = observation.get("combat")
    combat = combat if isinstance(combat, Mapping) else {}
    enemy = observation.get("enemy")
    enemy = enemy if isinstance(enemy, Mapping) else {}
    technology = observation.get("technology")
    technology = technology if isinstance(technology, Mapping) else {}

    composition = _counts(own.get("combat_composition"))
    completed = _counts(own.get("completed_counts")) or _counts(
        production.get("completed")
    )
    if completed.get("MEDIVAC", 0.0) > composition.get("MEDIVAC", 0.0):
        composition["MEDIVAC"] = completed["MEDIVAC"]
    if composition.get("MEDIVAC", 0.0) <= 0:
        return None

    visible_enemy = _counts(enemy.get("visible_composition"))
    direct = _number(combat.get("controlled_own_army_power"))
    visible_enemy_power = _number(combat.get("visible_enemy_army_power"))
    return estimate_support_aware_power(
        composition,
        direct_power=direct,
        enemy_composition=visible_enemy,
        enemy_direct_power=visible_enemy_power,
        completed_upgrades=technology.get("completed_upgrades") or [],
        direct_power_includes_fixed_support=True,
    )


__all__ = [
    "SUPPORT_POWER_SCHEMA",
    "composition_from_timing_package",
    "estimate_observation_support_power",
    "estimate_support_aware_power",
    "estimate_timing_package_support_power",
    "upgrades_from_timing_package",
]
