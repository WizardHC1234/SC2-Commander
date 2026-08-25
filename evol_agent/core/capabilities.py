from __future__ import annotations

import importlib
from typing import Any


ARMY_CONTROLS = (
    "hold_or_gather",
    "semantic_zone_movement",
    "push_or_assault",
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
    action_specs: dict[str, Any] = {}
    error = ""
    try:
        module = importlib.import_module(f"commander.races.{normalized_race}.actions")
        getter = getattr(module, "get_action_space", None)
        if callable(getter):
            action_space = dict(getter())
        raw_specs = getattr(module, "ACTION_SPECS", None)
        if isinstance(raw_specs, dict):
            action_specs = dict(raw_specs)
    except Exception as exc:  # keep EvolAgent usable for incomplete race adapters
        error = f"{type(exc).__name__}: {exc}"

    names = sorted(action_space)
    control_names = {
        name
        for name, spec in action_specs.items()
        if str(getattr(spec, "action_type", "")) in {"army", "meta"}
    }
    if not control_names:
        control_names = {
            name
            for name in names
            if name in {"army_intent", "move_group", "scanner_sweep", "scout", "set_wake_event"}
        }
    macro_names = [name for name in names if name not in control_names]
    army_controls = list(ARMY_CONTROLS)
    if "move_group" in control_names:
        army_controls.extend(
            [
                "one_command_per_observed_army_group",
                "movement_mode_per_group",
                "retreat_ratio_per_group_command",
            ]
        )
    if "army_intent" in control_names:
        army_controls.append("persistent_whole_army_intent")
    return {
        "schema": "commander_executor_capabilities.v2",
        "race": normalized_race,
        "macro_contract": {
            "target_semantics": "absolute concurrent targets replaced each decision",
            "priority_semantics": "tool-call order is resource-spend priority",
            "dependency_closure": "runtime expands structural prerequisites and gas",
            "available_actions": macro_names,
        },
        "army_controls": army_controls,
        "control_actions": {
            name: action_space.get(name, "") for name in sorted(control_names)
        },
        "information_controls": list(INFORMATION_CONTROLS),
        "observation_contract": {
            "strategy_usable": [
                "current own economy, technology, production, and living unit counts",
                "currently visible enemy units and structures",
                "last-seen enemy contents only with explicit recency",
                "current scan readiness and current objective progress",
            ],
            "analysis_only": [
                "post-match enemy_truth extracted from Replay",
                "facts first revealed after the decision being evaluated",
            ],
            "not_persistent_strategy_state": [
                "whether a previous scan completed",
                "which branch was selected in an earlier Commander cycle",
                "arbitrary memory of a prior observation",
            ],
        },
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
            "post-match enemy_truth as a live strategy condition",
            "cross-cycle scan or branch history as strategy state",
            "transport loading or unloading",
            "manual unit transformations or transformation-readiness gates",
            "manual combat-ability use",
            "a runtime transformation state as an attack-gate prerequisite",
        ],
        "catalog_error": error,
    }


__all__ = ["ARMY_CONTROLS", "build_executor_capability_manifest"]
