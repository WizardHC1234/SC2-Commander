"""Tests for whole-army intent expansion and live reinforcement following."""

from types import SimpleNamespace

from sc2.position import Point2

from commander.combat_exec import CombatControlAct
from commander.combat_policy import ArmyControlPolicy, ArmyIntent
from commander.combat_state import build_cleanup_runtime_hint


class _FakeUnits(list):
    @property
    def exists(self):
        return bool(self)


def _unit(x, y):
    return SimpleNamespace(position=Point2((x, y)))


def test_attack_intent_expands_main_to_assault_and_reinforcement_to_regroup():
    act = CombatControlAct()
    act._main_group_id = act.MAIN_GROUP_ID
    act._current_groups = {
        act.MAIN_GROUP_ID: _FakeUnits([_unit(20, 30)]),
        act.REINFORCEMENT_GROUP_ID: _FakeUnits([_unit(2, 3)]),
    }

    expanded = act._materialize_army_intent(
        ArmyControlPolicy(army_intent=ArmyIntent(mode="attack", zone_id="zone_9"))
    )

    commands = {command.group_id: command for command in expanded.commands}
    assert commands[act.MAIN_GROUP_ID].movement_mode == "assault"
    assert commands[act.REINFORCEMENT_GROUP_ID].movement_mode == "regroup"
    assert commands[act.REINFORCEMENT_GROUP_ID].destination_zone_id == "zone_9"


def test_reinforcement_regroup_resolves_to_live_main_force_center():
    act = CombatControlAct()
    act._main_group_id = act.MAIN_GROUP_ID
    act._current_groups = {
        act.MAIN_GROUP_ID: _FakeUnits([_unit(20, 30), _unit(24, 34)]),
        act.REINFORCEMENT_GROUP_ID: _FakeUnits([_unit(2, 3)]),
    }

    target = act._resolve_regroup_target(
        "zone_9",
        act._current_groups[act.REINFORCEMENT_GROUP_ID],
        group_id=act.REINFORCEMENT_GROUP_ID,
    )

    assert target == Point2((22, 32))


def test_cleanup_hint_requests_whole_army_cleanup_mode():
    act = SimpleNamespace(
        _peak_known_enemy_bases=1,
        ai=SimpleNamespace(time=100, game_time_limit_seconds=3600),
    )
    state = {
        "controlled_combat_units": 10,
        "known_enemy_bases": 0,
        "visible_enemy_power": 0,
        "army_nearest_zone": "zone_15",
        "army_groups": [
            {"role": "main_force", "nearest_zone_id": "zone_15"}
        ],
        "available_zones": [],
    }

    first_hint = build_cleanup_runtime_hint(act, state)
    assert "cleanup_recommended=yes" in first_hint
    assert "mode=cleanup" in first_hint


def test_cleanup_intent_expands_all_groups_to_search_and_destroy():
    act = CombatControlAct()
    act._main_group_id = act.MAIN_GROUP_ID
    act._current_groups = {
        act.MAIN_GROUP_ID: _FakeUnits([_unit(20, 30)]),
        act.REINFORCEMENT_GROUP_ID: _FakeUnits([_unit(2, 3)]),
    }

    expanded = act._materialize_army_intent(
        ArmyControlPolicy(army_intent=ArmyIntent(mode="cleanup", zone_id="zone_15"))
    )

    assert {command.movement_mode for command in expanded.commands} == {
        "search_and_destroy"
    }
    assert {command.group_id for command in expanded.commands} == {
        act.MAIN_GROUP_ID,
        act.REINFORCEMENT_GROUP_ID,
    }
