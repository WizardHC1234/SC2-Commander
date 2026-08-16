from __future__ import annotations

import importlib
from typing import Any


ARMY_CONTROLS = (
    "hold_or_gather",
    "semantic_zone_movement",
    "persistent_army_intent",
    "push_or_assault",
    "retreat_ratio",
    "main_force_reinforcement",
    "explicit_endgame_cleanup",
)
INFORMATION_CONTROLS = (
    "one_active_scv_scout",
    "scanner_sweep_on_observed_zone",
    "one_observable_wake_event",
)


def build_executor_capability_manifest(race: str) -> dict[str, Any]:
    """Describe the strategy levers that the live Commander can execute.

    The macro list comes from the same race action catalog used at runtime, so
    EvolAgent cannot silently drift away from the executable tool vocabulary.
    """
    normalized_race = str(race or "terran").strip().lower() or "terran"
    action_space: dict[str, str] = {}
    error = ""
    try:
        module = importlib.import_module(f"commander.races.{normalized_race}.actions")
        getter = getattr(module, "get_action_space", None)
        if callable(getter):
            action_space = dict(getter())
    except Exception as exc:  # keep EvolAgent usable for incomplete race adapters
        error = f"{type(exc).__name__}: {exc}"

    names = sorted(action_space)
    macro_names = [
        name
        for name in names
        if name not in {
            "army_intent",
            "scanner_sweep",
            "scout",
            "set_wake_event",
        }
    ]
    return {
        "schema": "commander_executor_capabilities.v1",
        "race": normalized_race,
        "macro_contract": {
            "target_semantics": "absolute concurrent targets replaced each decision",
            "priority_semantics": "tool-call order is resource-spend priority",
            "dependency_closure": "runtime expands structural prerequisites and gas",
            "available_actions": macro_names,
        },
        "army_controls": list(ARMY_CONTROLS),
        "information_controls": list(INFORMATION_CONTROLS),
        "runtime_owned": [
            "worker distribution",
            "pathfinding",
            "formations",
            "targeting",
            "abilities and transformations",
            "transport and unit-level micro",
        ],
        "strategy_must_not_require": [
            "map-specific zone IDs",
            "group IDs fixed before observation",
            "manual unit-level target selection",
            "unobserved opponent truth",
        ],
        "catalog_error": error,
    }


__all__ = ["ARMY_CONTROLS", "build_executor_capability_manifest"]
