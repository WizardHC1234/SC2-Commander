"""Unit tests for model-authored wake events."""

from __future__ import annotations

import unittest

from commander.tools import (
    apply_tool_calls,
    army_group_ids_from_observation,
    build_commander_tools,
    validate_army_tools_for_cycle,
)
from commander.combat_policy import ArmyControlPolicy, ArmyGroupCommand
from commander.wake_events import (
    build_wake_snapshot,
    evaluate_wake_event,
    fallback_wake_event,
    format_wake_reflection_feedback,
    normalize_wake_event,
    rising_edge,
    validate_wake_for_cycle,
)


class NormalizeWakeEventTests(unittest.TestCase):
    def test_valid_all_event(self):
        event, issues = normalize_wake_event(
            {
                "logic": "all",
                "conditions": [
                    {"type": "unit_count_at_least", "unit": "Marine", "count": 20},
                    {"type": "destination_reached"},
                ],
            }
        )
        self.assertIsNotNone(event)
        self.assertEqual(event["logic"], "all")
        self.assertEqual(len(event["conditions"]), 2)
        self.assertEqual(issues, [])

    def test_rejects_unknown_predicate(self):
        event, issues = normalize_wake_event(
            {
                "logic": "any",
                "conditions": [{"type": "hack_the_planet"}],
            }
        )
        self.assertIsNone(event)
        self.assertTrue(any("bad_type" in item for item in issues))

    def test_legacy_abbreviated_aliases(self):
        event, issues = normalize_wake_event(
            {
                "logic": "any",
                "conditions": [
                    {"type": "game_time_gte", "seconds": 90},
                    {"type": "unit_count_gte", "unit": "Marine", "count": 20},
                    {"type": "supply_left_lte", "count": 4},
                ],
            }
        )
        self.assertIsNotNone(event)
        types = [c["type"] for c in event["conditions"]]
        self.assertEqual(
            types,
            [
                "game_time_at_least",
                "unit_count_at_least",
                "supply_left_at_most",
            ],
        )
        self.assertEqual(issues, [])

    def test_scout_wake_disabled(self):
        event, issues = normalize_wake_event(
            {
                "logic": "any",
                "conditions": [
                    {"type": "scout_just_finished"},
                    {"type": "game_time_at_least", "seconds": 90},
                ],
            }
        )
        self.assertIsNotNone(event)
        self.assertEqual(len(event["conditions"]), 1)
        self.assertEqual(event["conditions"][0]["type"], "game_time_at_least")
        self.assertTrue(any("scout_wake_disabled" in item for item in issues))

    def test_movement_mode_wake_disabled(self):
        event, issues = normalize_wake_event(
            {
                "logic": "any",
                "conditions": [
                    {
                        "type": "movement_mode_not_in",
                        "modes": ["assault", "push", "harass"],
                    },
                    {"type": "unit_count_at_least", "unit": "Marine", "count": 20},
                ],
            }
        )
        self.assertIsNotNone(event)
        self.assertEqual(len(event["conditions"]), 1)
        self.assertEqual(event["conditions"][0]["type"], "unit_count_at_least")
        self.assertTrue(
            any("movement_mode_wake_disabled" in item for item in issues)
        )
    def test_drops_invalid_keeps_valid(self):
        event, issues = normalize_wake_event(
            {
                "logic": "any",
                "conditions": [
                    {"type": "game_time_at_least", "seconds": 120},
                    {"type": "unit_count_at_least", "count": 5},  # missing unit
                ],
            }
        )
        self.assertIsNotNone(event)
        self.assertEqual(len(event["conditions"]), 1)
        self.assertTrue(any("missing_unit" in item for item in issues))

    def test_empty_conditions(self):
        event, issues = normalize_wake_event({"logic": "all", "conditions": []})
        self.assertIsNone(event)
        self.assertTrue(any("empty_conditions" in item for item in issues))


class ValidateWakeForCycleTests(unittest.TestCase):
    def test_unit_count_without_train_is_blocking(self):
        event, _ = normalize_wake_event(
            {
                "logic": "all",
                "conditions": [
                    {"type": "unit_count_at_least", "unit": "Marine", "count": 45},
                    {"type": "unit_count_at_least", "unit": "SiegeTank", "count": 10},
                ],
            }
        )
        blocking = validate_wake_for_cycle(
            event,
            macro_actions=["train_scv", "build_barracks"],
            legal_macro_keys=[
                "train_scv",
                "train_marine",
                "train_siege_tank",
                "build_barracks",
            ],
        )
        self.assertTrue(any("Marine" in item for item in blocking))
        self.assertTrue(any("SiegeTank" in item for item in blocking))
        self.assertTrue(any("train_marine" in item for item in blocking))

    def test_unit_count_with_train_ok(self):
        event, _ = normalize_wake_event(
            {
                "logic": "any",
                "conditions": [
                    {"type": "unit_count_at_least", "unit": "Marine", "count": 20},
                ],
            }
        )
        blocking = validate_wake_for_cycle(
            event,
            macro_actions=["train_marine", "build_barracks"],
        )
        self.assertEqual(blocking, [])

    def test_missing_wake_blocks(self):
        blocking = validate_wake_for_cycle(
            None,
            macro_actions=["train_scv"],
            apply_issues=["wake_event:missing"],
        )
        self.assertTrue(any("missing" in item for item in blocking))

    def test_reflection_feedback_lists_issues(self):
        text = format_wake_reflection_feedback(
            blocking_issues=["wake_unreachable:unit_count_at_least:Marine"],
            previous_tool_calls=[
                {"name": "train_scv", "arguments": {"to_count": 44}},
            ],
        )
        self.assertIn("[Decision Validation Failed", text)
        self.assertIn("Marine", text)
        self.assertIn("train_scv", text)
        self.assertIn("move_group", text)


class EvaluateWakeEventTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = build_wake_snapshot(
            time_seconds=100.0,
            supply_used=40,
            supply_cap=46,
            own_unit_type_counts={"Marine": 18, "SiegeTank": 2},
            army_groups=[
                {
                    "group_id": "group_0",
                    "role": "main_force",
                    "nearest_zone_id": "zone_5",
                    "current_command": {
                        "destination_zone_id": "zone_5",
                        "movement_mode": "regroup",
                    },
                }
            ],
            army_summary={
                "commands": [
                    {
                        "group_id": "group_0",
                        "destination_zone_id": "zone_5",
                        "movement_mode": "regroup",
                    }
                ]
            },
            available_zones=[
                {
                    "zone_id": "zone_5",
                    "vision_state": "visible",
                    "visible_enemy_contents": {},
                    "visible_enemy_units": 0,
                }
            ],
            last_scout_result="completed",
            scan_ready=True,
            cleanup_hint_present=False,
        )

    def test_all_requires_every_predicate(self):
        event, _ = normalize_wake_event(
            {
                "logic": "all",
                "conditions": [
                    {"type": "unit_count_at_least", "unit": "Marine", "count": 20},
                    {"type": "supply_left_at_most", "count": 2},
                ],
            }
        )
        self.assertFalse(evaluate_wake_event(event, self.snapshot))
        self.snapshot["own_unit_type_counts"]["Marine"] = 20
        self.assertFalse(evaluate_wake_event(event, self.snapshot))
        self.snapshot["supply_used"] = 45
        self.assertTrue(evaluate_wake_event(event, self.snapshot))
    def test_any_fires_on_one_predicate(self):
        event, _ = normalize_wake_event(
            {
                "logic": "any",
                "conditions": [
                    {"type": "unit_count_at_least", "unit": "Marine", "count": 40},
                    {"type": "scan_ready"},
                ],
            }
        )
        self.assertTrue(evaluate_wake_event(event, self.snapshot))

    def test_objective_and_destination(self):
        event, _ = normalize_wake_event(
            {
                "logic": "all",
                "conditions": [
                    {
                        "type": "objective_status_became",
                        "status": "confirmed_clear",
                    },
                    {"type": "destination_reached"},
                ],
            }
        )
        # Status already matches baseline → became is false; destination alone
        # is not enough under logic=all.
        self.assertFalse(
            evaluate_wake_event(
                event,
                {
                    **self.snapshot,
                    "baseline_objective_status": "confirmed_clear",
                },
            )
        )
        self.assertTrue(
            evaluate_wake_event(
                event,
                {
                    **self.snapshot,
                    "baseline_objective_status": "en_route_unconfirmed",
                },
            )
        )

    def test_army_group_count_wake_disabled(self):
        event, issues = normalize_wake_event(
            {
                "logic": "any",
                "conditions": [
                    {"type": "army_group_count_at_least", "count": 1},
                    {"type": "game_time_at_least", "seconds": 90},
                ],
            }
        )
        self.assertIsNotNone(event)
        self.assertEqual(len(event["conditions"]), 1)
        self.assertEqual(event["conditions"][0]["type"], "game_time_at_least")
        self.assertTrue(
            any("army_group_count_wake_disabled" in item for item in issues)
        )

    def test_objective_status_is_wake_disabled(self):
        event, issues = normalize_wake_event(
            {
                "logic": "any",
                "conditions": [
                    {"type": "objective_status_is", "status": "confirmed_clear"},
                    {"type": "destination_reached"},
                ],
            }
        )
        self.assertIsNotNone(event)
        self.assertEqual(len(event["conditions"]), 1)
        self.assertEqual(event["conditions"][0]["type"], "destination_reached")
        self.assertTrue(
            any("objective_status_is_wake_disabled" in item for item in issues)
        )

    def test_supply_left(self):
        event, _ = normalize_wake_event(
            {
                "logic": "any",
                "conditions": [{"type": "supply_left_at_most", "count": 6}],
            }
        )
        self.assertTrue(evaluate_wake_event(event, self.snapshot))

    def test_game_time(self):
        event = fallback_wake_event(50.0, delay=60.0)
        self.assertFalse(evaluate_wake_event(event, self.snapshot))
        self.snapshot["time_seconds"] = 110.0
        self.assertTrue(evaluate_wake_event(event, self.snapshot))

    def test_scout_result_aliases_reached(self):
        # scout_* is disabled at normalize; evaluate still accepts aliases.
        event = {
            "logic": "any",
            "conditions": [
                {"type": "scout_result_is", "result": "completed"},
            ],
        }
        snap = build_wake_snapshot(
            time_seconds=10.0,
            last_scout_result="reached",
        )
        self.assertTrue(evaluate_wake_event(event, snap))

    def test_scout_just_finished_requires_result_after_arm(self):
        event = {
            "logic": "any",
            "conditions": [{"type": "scout_just_finished"}],
        }
        stale = build_wake_snapshot(
            time_seconds=100.0,
            last_scout_result="completed",
            last_scout_result_time=50.0,
            wake_armed_at=80.0,
        )
        self.assertFalse(evaluate_wake_event(event, stale))
        fresh = build_wake_snapshot(
            time_seconds=100.0,
            last_scout_result="completed",
            last_scout_result_time=90.0,
            wake_armed_at=80.0,
        )
        self.assertTrue(evaluate_wake_event(event, fresh))

    def test_objective_status_became(self):
        event, _ = normalize_wake_event(
            {
                "logic": "any",
                "conditions": [
                    {
                        "type": "objective_status_became",
                        "status": "confirmed_clear",
                    }
                ],
            }
        )
        same = build_wake_snapshot(
            time_seconds=10.0,
            army_groups=[
                {
                    "group_id": "group_0",
                    "role": "main_force",
                    "nearest_zone_id": "zone_5",
                    "current_command": {
                        "destination_zone_id": "zone_5",
                        "movement_mode": "assault",
                    },
                }
            ],
            available_zones=[
                {
                    "zone_id": "zone_5",
                    "vision_state": "visible",
                    "visible_enemy_units": 0,
                }
            ],
            baseline_objective_status="confirmed_clear",
        )
        self.assertFalse(evaluate_wake_event(event, same))
        changed = build_wake_snapshot(
            time_seconds=10.0,
            army_groups=[
                {
                    "group_id": "group_0",
                    "role": "main_force",
                    "nearest_zone_id": "zone_5",
                    "current_command": {
                        "destination_zone_id": "zone_5",
                        "movement_mode": "assault",
                    },
                }
            ],
            available_zones=[
                {
                    "zone_id": "zone_5",
                    "vision_state": "visible",
                    "visible_enemy_units": 0,
                }
            ],
            baseline_objective_status="en_route_unconfirmed",
        )
        self.assertTrue(evaluate_wake_event(event, changed))


class RisingEdgeTests(unittest.TestCase):
    def test_false_to_true(self):
        self.assertTrue(rising_edge(True, False))
        self.assertTrue(rising_edge(True, None))
        self.assertFalse(rising_edge(True, True))
        self.assertFalse(rising_edge(False, False))
        self.assertFalse(rising_edge(False, True))


class ApplyToolCallsWakeTests(unittest.TestCase):
    def test_parses_set_wake_event(self):
        tasks, policy, issues, wake = apply_tool_calls(
            [
                {
                    "name": "train_marine",
                    "arguments": {"to_count": 20},
                },
                {
                    "name": "set_wake_event",
                    "arguments": {
                        "logic": "all",
                        "conditions": [
                            {
                                "type": "unit_count_at_least",
                                "unit": "Marine",
                                "count": 20,
                            }
                        ],
                    },
                },
            ],
            legal_action_keys={"train_marine"},
        )
        self.assertEqual(tasks, [{"action": "train_marine", "to_count": 20}])
        self.assertEqual(policy.commands, [])
        self.assertIsNotNone(wake)
        self.assertEqual(wake["logic"], "all")
        self.assertNotIn("wake_event:missing", issues)

    def test_missing_wake_adds_issue(self):
        _tasks, _policy, issues, wake = apply_tool_calls(
            [{"name": "train_scv", "arguments": {"to_count": 16}}],
            legal_action_keys={"train_scv"},
        )
        self.assertIsNone(wake)
        self.assertIn("wake_event:missing", issues)

    def test_invalid_wake_yields_none(self):
        _tasks, _policy, issues, wake = apply_tool_calls(
            [
                {
                    "name": "set_wake_event",
                    "arguments": {
                        "logic": "all",
                        "conditions": [{"type": "nope"}],
                    },
                }
            ],
            legal_action_keys=set(),
        )
        self.assertIsNone(wake)
        self.assertTrue(any("bad_type" in item for item in issues))
        self.assertNotIn("wake_event:missing", issues)

    def test_last_wake_wins(self):
        _tasks, _policy, _issues, wake = apply_tool_calls(
            [
                {
                    "name": "set_wake_event",
                    "arguments": {
                        "logic": "any",
                        "conditions": [
                            {"type": "game_time_at_least", "seconds": 10}
                        ],
                    },
                },
                {
                    "name": "set_wake_event",
                    "arguments": {
                        "logic": "all",
                        "conditions": [
                            {"type": "game_time_at_least", "seconds": 99}
                        ],
                    },
                },
            ],
            legal_action_keys=set(),
        )
        self.assertEqual(wake["logic"], "all")
        self.assertEqual(wake["conditions"][0]["seconds"], 99)

    def test_build_tools_includes_set_wake_event(self):
        tools = build_commander_tools(
            {
                "train_scv": "Train SCVs",
                "move_group": "Move army",
                "set_wake_event": "Wake next",
            }
        )
        names = [t["function"]["name"] for t in tools]
        self.assertIn("set_wake_event", names)
        self.assertIn("train_scv", names)
        self.assertIn("move_group", names)


class ArmyToolValidationTests(unittest.TestCase):
    def test_missing_move_group_blocks(self):
        policy = ArmyControlPolicy(commands=[])
        blocking = validate_army_tools_for_cycle(
            policy, required_group_ids=["group_0"]
        )
        self.assertTrue(any("army_move_group:missing" in item for item in blocking))

    def test_incomplete_move_group_blocks(self):
        policy = ArmyControlPolicy(
            commands=[
                ArmyGroupCommand(
                    group_id="group_0",
                    destination_zone_id="zone_1",
                    movement_mode="regroup",
                    move_type="ReGroup",
                )
            ]
        )
        blocking = validate_army_tools_for_cycle(
            policy, required_group_ids=["group_0", "group_1"]
        )
        self.assertTrue(any("army_move_group:incomplete" in item for item in blocking))
        self.assertTrue(any("group_1" in item for item in blocking))

    def test_complete_move_groups_ok(self):
        policy = ArmyControlPolicy(
            commands=[
                ArmyGroupCommand(
                    group_id="group_0",
                    destination_zone_id="zone_1",
                    movement_mode="regroup",
                    move_type="ReGroup",
                ),
                ArmyGroupCommand(
                    group_id="group_1",
                    destination_zone_id="zone_2",
                    movement_mode="push",
                    move_type="Push",
                ),
            ]
        )
        self.assertEqual(
            validate_army_tools_for_cycle(
                policy, required_group_ids=["group_0", "group_1"]
            ),
            [],
        )

    def test_empty_groups_do_not_require_army(self):
        self.assertEqual(
            validate_army_tools_for_cycle(
                ArmyControlPolicy(commands=[]), required_group_ids=[]
            ),
            [],
        )

    def test_group_ids_from_observation(self):
        ids = army_group_ids_from_observation(
            {
                "army_control": {
                    "groups": [
                        {"group_id": "group_0"},
                        {"group_id": "group_1"},
                    ]
                }
            }
        )
        self.assertEqual(ids, ["group_0", "group_1"])


class TriggerHintTests(unittest.TestCase):
    def test_format_and_list_satisfied(self):
        from commander.wake_events import (
            build_trigger_hint,
            format_wake_condition,
            list_satisfied_wake_conditions,
        )

        self.assertEqual(
            format_wake_condition(
                {"type": "unit_count_at_least", "unit": "Marine", "count": 45}
            ),
            "unit_count_at_least(Marine>=45)",
        )
        event = {
            "logic": "any",
            "conditions": [
                {"type": "unit_count_at_least", "unit": "Marine", "count": 45},
                {"type": "unit_count_at_least", "unit": "SIEGETANK", "count": 10},
                {"type": "game_time_at_least", "seconds": 600},
            ],
        }
        snap = build_wake_snapshot(
            time_seconds=500.0,
            own_unit_type_counts={"Marine": 50, "SIEGETANK": 4},
        )
        fired = list_satisfied_wake_conditions(event, snap)
        self.assertEqual(fired, ["unit_count_at_least(Marine>=45)"])
        hint = build_trigger_hint(
            reason="wake_event",
            event=event,
            fired_conditions=fired,
        )
        self.assertIn("woken_by=unit_count_at_least(Marine>=45)", hint)
        self.assertIn("armed_logic=any", hint)
        self.assertIn("[Runtime Decision Trigger]", hint)

    def test_deadline_fuse_label(self):
        from commander.wake_events import build_trigger_hint

        hint = build_trigger_hint(
            reason="wake_fallback_timeout",
            event=fallback_wake_event(100.0),
            fired_conditions=["runtime_deadline_fuse"],
        )
        self.assertIn("woken_by=runtime_deadline_fuse", hint)


if __name__ == "__main__":
    unittest.main()
