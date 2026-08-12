from __future__ import annotations

import json
from pathlib import Path

from tools.strategy_execution_metrics import evaluate_match, load_match


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


def test_current_commander_record_uses_original_five_metrics(tmp_path: Path) -> None:
    command = {
        "group_id": "group_0",
        "destination_zone_id": "zone_1",
        "movement_mode": "push",
        "retreat_ratio": None,
    }
    record = {
        "metadata": {
            "strategy_id": "test_strategy",
            "commander_model_key": "test-model",
            "interval_seconds": 60,
            "game_duration_seconds": 60,
            "opponent_id": "commander.terran-ai.terran.hard.macro",
        },
        "interactions": [
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
    }
    path = tmp_path / "match.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    match = load_match(path)
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
        "attack_gate": {"MARINE": 2},
        "attack_modes": ["push", "assault", "search_and_destroy"],
        "attack_group_roles": ["main_force"],
    }

    row, requirements = evaluate_match(match, spec)

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
