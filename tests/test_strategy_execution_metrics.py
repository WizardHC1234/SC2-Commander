from __future__ import annotations

import json
import math
from pathlib import Path

from tools.strategy_execution_metrics import (
    batch_name,
    discover_record_files,
    evaluate_match,
    load_match,
)


def _spec(*, gate: int = 2, first_role: str | None = None) -> dict:
    spec = {
        "strategy_name": "test_strategy",
        "worker": {"unit": "SCV", "target": 1},
        "required_entities_by_end": {"BARRACKS": 1},
        "required_upgrades_by_end": [],
        "unit_aliases": {
            "SCV": ["SCV"],
            "BARRACKS": ["BARRACKS"],
            "MARINE": ["MARINE"],
        },
        "unit_supply": {"MARINE": 1},
        "attack_gate": {"MARINE": gate},
        "attack_modes": ["push", "assault", "search_and_destroy"],
        "attack_group_roles": ["main_force"],
    }
    if first_role is not None:
        spec["first_attack_zone_roles"] = [first_role]
    return spec


def _write_match(tmp_path: Path, interactions: list[dict]) -> object:
    record = {
        "metadata": {
            "strategy_id": "test_strategy",
            "commander_model_key": "test-model",
            "interval_seconds": 60,
            "game_duration_seconds": max(
                (item["game_time"] for item in interactions),
                default=0,
            ),
            "opponent_id": "commander.terran-ai.terran.hard.macro",
        },
        "interactions": interactions,
    }
    path = tmp_path / "match.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return load_match(path)


def _observation(*, marine_count: int, attacking: bool) -> dict:
    command = {
        "group_id": "group_0",
        "destination_zone_id": "zone_1",
        "movement_mode": "push",
        "retreat_ratio": None,
    }
    return {
        "economy": {"own_base_count": 1},
        "production": {
            "completed": {
                "SCV": 1,
                "BARRACKS": 1,
                "MARINE": marine_count,
            },
            "under_construction": {},
        },
        "own_forces": {
            "completed_counts": {
                "SCV": 1,
                "BARRACKS": 1,
                "MARINE": marine_count,
            },
            "combat_composition": {"MARINE": marine_count},
        },
        "technology": {"completed_upgrades": []},
        "army_control": {
            "groups": [
                {
                    "group_id": "group_0",
                    "role": "main_force",
                    "unit_count": marine_count,
                    "unit_type_counts": {"MARINE": marine_count},
                    "nearest_zone_id": "zone_0",
                    "current_command": command if attacking else {},
                }
            ],
            "zones": [
                {"zone_id": "zone_0", "owner": "own", "zone_role": "own_main"},
                {
                    "zone_id": "zone_1",
                    "owner": "enemy",
                    "zone_role": "enemy_main",
                },
            ],
            "current_commands": [command] if attacking else [],
            "zone_topology": {"zones": []},
        },
    }


def test_evolution_batch_directories_are_discovered(tmp_path: Path) -> None:
    evolution_batch = tmp_path / "ev_example_g000_cand"
    record_dir = evolution_batch / "match_0"
    record_dir.mkdir(parents=True)
    record = record_dir / "match.json"
    record.write_text("{}", encoding="utf-8")

    assert discover_record_files([tmp_path]) == [record]
    assert batch_name(record) == evolution_batch.name


def test_current_commander_record_uses_original_five_metrics(tmp_path: Path) -> None:
    command = {
        "group_id": "group_0",
        "destination_zone_id": "zone_1",
        "movement_mode": "push",
        "retreat_ratio": None,
    }
    match = _write_match(
        tmp_path,
        [
            {
                "agent": "commander",
                "trigger_reason": "commander_bootstrap",
                "game_time": 0,
                "strategy_id": "test_strategy",
                "observation": _observation(marine_count=1, attacking=False),
                "army_policy": {"commands": []},
                "accepted": True,
            },
            {
                "agent": "commander",
                "trigger_reason": "wake_event",
                "game_time": 60,
                "strategy_id": "test_strategy",
                "observation": _observation(marine_count=2, attacking=True),
                "army_policy": {"commands": [command]},
                "accepted": True,
            },
        ],
    )

    row, requirements = evaluate_match(match, _spec())

    assert match.strategy_name == "test_strategy"
    assert row["model"] == "test-model"
    assert row["economy_completion"] == 1.0
    assert row["technology_completion"] == 1.0
    assert row["army_completion"] == 1.0
    assert row["engagement_trigger_consistency"] == 1.0
    assert row["engagement_continuation_consistency"] == 1.0
    assert row["overall_strategy_compliance"] == 1.0
    assert {result.category for result in requirements} == {
        "economy",
        "technology",
        "army",
        "engagement",
    }


def test_no_gate_and_no_attack_makes_engagement_not_evaluable(
    tmp_path: Path,
) -> None:
    match = _write_match(
        tmp_path,
        [
            {
                "agent": "commander",
                "game_time": 0,
                "strategy_id": "test_strategy",
                "observation": _observation(marine_count=1, attacking=False),
                "army_policy": {"commands": []},
                "accepted": True,
            }
        ],
    )

    row, requirements = evaluate_match(match, _spec(gate=2))

    assert row["engagement_trigger_consistency"] is None
    assert row["engagement_continuation_consistency"] is None
    assert row["engagement_trigger_evaluable"] == 0.0
    assert math.isclose(row["overall_strategy_compliance"], 5 / 6)
    assert all(result.category != "engagement" for result in requirements)


def test_reinforcement_group_does_not_complete_main_force_attack_gate(
    tmp_path: Path,
) -> None:
    command_0 = {
        "group_id": "group_0",
        "destination_zone_id": "zone_1",
        "movement_mode": "push",
    }
    command_1 = {**command_0, "group_id": "group_1"}
    observation = _observation(marine_count=20, attacking=False)
    observation["army_control"]["groups"] = [
        {
            "group_id": "group_0",
            "role": "main_force",
            "unit_count": 10,
            "unit_type_counts": {"MARINE": 10},
            "nearest_zone_id": "zone_0",
            "current_command": command_0,
        },
        {
            "group_id": "group_1",
            "role": "reinforcement",
            "unit_count": 10,
            "unit_type_counts": {"MARINE": 10},
            "nearest_zone_id": "zone_0",
            "current_command": command_1,
        },
    ]
    observation["army_control"]["current_commands"] = [command_0, command_1]
    match = _write_match(
        tmp_path,
        [
            {
                "agent": "commander",
                "game_time": 0,
                "strategy_id": "test_strategy",
                "observation": observation,
                "army_policy": {"commands": [command_0, command_1]},
                "accepted": True,
            }
        ],
    )

    row, _requirements = evaluate_match(match, _spec(gate=20))

    assert row["army_completion"] == 1.0
    assert row["attack_readiness_source"] == "gathered_main_force"
    assert row["attack_force_progress"] == 0.5
    assert row["engagement_trigger_consistency"] == 0.5


def test_wrong_strategy_first_objective_fails_trigger(tmp_path: Path) -> None:
    command = {
        "group_id": "group_0",
        "destination_zone_id": "zone_2",
        "movement_mode": "push",
    }
    observation = _observation(marine_count=2, attacking=False)
    observation["army_control"]["zones"].append(
        {"zone_id": "zone_2", "owner": "enemy", "zone_role": "enemy_natural"}
    )
    observation["army_control"]["groups"][0]["current_command"] = command
    observation["army_control"]["current_commands"] = [command]
    match = _write_match(
        tmp_path,
        [
            {
                "agent": "commander",
                "game_time": 0,
                "strategy_id": "test_strategy",
                "observation": observation,
                "army_policy": {"commands": [command]},
                "accepted": True,
            }
        ],
    )

    row, _requirements = evaluate_match(
        match,
        _spec(gate=2, first_role="enemy_main"),
    )

    assert row["first_attack_objective_correct"] is False
    assert row["engagement_trigger_consistency"] == 0.0
    assert row["attack_evaluation_status"] == "wrong_first_objective"


def test_continuation_accepts_recovery_and_reinforcement_joining_main(
    tmp_path: Path,
) -> None:
    attack = {
        "group_id": "group_0",
        "destination_zone_id": "zone_1",
        "movement_mode": "push",
    }
    retreat = {
        "group_id": "group_0",
        "destination_zone_id": "zone_0",
        "movement_mode": "defensive_retreat",
    }
    regroup = {
        "group_id": "group_1",
        "destination_zone_id": "zone_0",
        "movement_mode": "regroup",
    }
    attack_observation = _observation(marine_count=20, attacking=True)
    recovery_observation = _observation(marine_count=10, attacking=False)
    recovery_observation["army_control"]["groups"] = [
        {
            "group_id": "group_0",
            "role": "main_force",
            "unit_count": 8,
            "unit_type_counts": {"MARINE": 8},
            "nearest_zone_id": "zone_0",
            "current_command": retreat,
            "command_source": "commander",
        },
        {
            "group_id": "group_1",
            "role": "reinforcement",
            "unit_count": 2,
            "unit_type_counts": {"MARINE": 2},
            "nearest_zone_id": "zone_1",
            "current_command": regroup,
            "command_source": "commander",
        },
    ]
    recovery_observation["army_control"]["current_commands"] = [retreat, regroup]
    match = _write_match(
        tmp_path,
        [
            {
                "agent": "commander",
                "game_time": 0,
                "strategy_id": "test_strategy",
                "observation": _observation(marine_count=20, attacking=False),
                "army_policy": {"commands": [attack]},
                "accepted": True,
            },
            {
                "agent": "commander",
                "game_time": 60,
                "strategy_id": "test_strategy",
                "observation": attack_observation,
                "army_policy": {"commands": [retreat]},
                "accepted": True,
            },
            {
                "agent": "commander",
                "game_time": 120,
                "strategy_id": "test_strategy",
                "observation": recovery_observation,
                "army_policy": {"commands": [retreat, regroup]},
                "accepted": True,
            },
        ],
    )

    row, _requirements = evaluate_match(match, _spec(gate=20))

    assert row["engagement_continuation_opportunities"] == 2
    assert row["engagement_continuation_consistency"] == 1.0
