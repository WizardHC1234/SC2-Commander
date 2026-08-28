"""Unit tests for move_group's retreat_ratio parameter (auto-retreat gate)."""

from commander.retreat_policy import (
    DEFAULT_RETREAT_RATIO,
    GroupRetreatState,
    RECOVER_MARGIN,
    RETREAT_MAX,
    RETREAT_MIN,
    STATE_ACTIVE,
    STATE_RETREATING,
    clamp_retreat_ratio,
    effective_retreat_ratio,
    preserve_blocked_offensive_command,
    retreat_confirmation_ready,
)
from commander.combat_policy import ArmyGroupCommand
from commander.combat_policy import ArmyControlPolicy
from commander.combat_exec import CombatControlAct
from commander.tools import apply_tool_calls, _parse_retreat_ratio


def test_parse_retreat_ratio_none():
    assert _parse_retreat_ratio(None) is None


def test_parse_retreat_ratio_value():
    assert _parse_retreat_ratio(0.8) == 0.8


def test_parse_retreat_ratio_clamps():
    assert _parse_retreat_ratio(9.9) == RETREAT_MAX
    assert _parse_retreat_ratio(0.01) == RETREAT_MIN


def test_parse_retreat_ratio_bad_types_fall_back_to_default():
    assert _parse_retreat_ratio("high") is None
    assert _parse_retreat_ratio(True) is None
    assert _parse_retreat_ratio({"x": 1}) is None


def test_clamp_bounds():
    assert clamp_retreat_ratio(1.2) == 1.2
    assert clamp_retreat_ratio(99) == RETREAT_MAX
    assert clamp_retreat_ratio(-1) == RETREAT_MIN


def _wake_call():
    return {
        "name": "set_wake_event",
        "arguments": {
            "logic": "any",
            "conditions": [{"type": "game_time_at_least", "seconds": 120}],
        },
    }


def test_apply_tool_calls_move_group_with_retreat_ratio():
    _tasks, policy, issues, _wake = apply_tool_calls(
        [
            {
                "name": "move_group",
                "arguments": {
                    "group_id": "group_0",
                    "destination_zone_id": "zone_15",
                    "movement_mode": "assault",
                    "retreat_ratio": 0.7,
                },
            },
            _wake_call(),
        ],
        legal_action_keys=set(),
    )
    assert issues == []
    command = policy.commands[0]
    assert command.movement_mode == "assault"
    assert command.retreat_ratio == 0.7


def test_apply_tool_calls_move_group_default_when_omitted():
    _tasks, policy, issues, _wake = apply_tool_calls(
        [
            {
                "name": "move_group",
                "arguments": {
                    "group_id": "group_0",
                    "destination_zone_id": "zone_15",
                    "movement_mode": "assault",
                },
            },
            _wake_call(),
        ],
        legal_action_keys=set(),
    )
    assert issues == []
    assert policy.commands[0].retreat_ratio is None


def test_apply_tool_calls_bad_ratio_does_not_drop_command():
    _tasks, policy, issues, _wake = apply_tool_calls(
        [
            {
                "name": "move_group",
                "arguments": {
                    "group_id": "group_0",
                    "destination_zone_id": "zone_15",
                    "movement_mode": "push",
                    "retreat_ratio": "soon",
                },
            },
            _wake_call(),
        ],
        legal_action_keys=set(),
    )
    assert issues == []
    assert len(policy.commands) == 1
    assert policy.commands[0].retreat_ratio is None


def test_recover_margin_hysteresis():
    """Recovery threshold stays meaningfully above the trigger."""
    assert DEFAULT_RETREAT_RATIO + RECOVER_MARGIN >= 1.0
    assert RECOVER_MARGIN >= 0.3


def test_nearby_support_prevents_a_false_local_retreat():
    ratio = effective_retreat_ratio(
        group_ratio=0.4,
        support_ratio=1.2,
        mission_ratio=1.3,
        group_power_share=0.5,
    )
    assert ratio == 1.2


def test_tiny_vanguard_uses_mission_ratio_but_main_force_does_not():
    fragment_ratio = effective_retreat_ratio(
        group_ratio=0.2,
        support_ratio=0.2,
        mission_ratio=2.0,
        group_power_share=0.05,
    )
    main_ratio = effective_retreat_ratio(
        group_ratio=0.2,
        support_ratio=0.2,
        mission_ratio=2.0,
        group_power_share=0.8,
    )
    assert fragment_ratio == 2.0
    assert main_ratio == 0.2


def test_non_catastrophic_low_ratio_requires_persistence():
    assert retreat_confirmation_ready(
        now=101.0,
        below_threshold_since=100.0,
        effective_ratio_value=0.5,
    ) is False
    assert retreat_confirmation_ready(
        now=102.1,
        below_threshold_since=100.0,
        effective_ratio_value=0.5,
    ) is True


def test_catastrophic_ratio_retreats_immediately():
    assert retreat_confirmation_ready(
        now=100.0,
        below_threshold_since=100.0,
        effective_ratio_value=0.1,
    ) is True


def _army_command(mode: str, zone: str = "zone_15") -> ArmyGroupCommand:
    move_type = {
        "assault": "Assault",
        "push": "Push",
        "hold": "Hold",
        "regroup": "ReGroup",
        "defensive_retreat": "DefensiveRetreat",
    }[mode]
    return ArmyGroupCommand(
        group_id="group_0",
        destination_zone_id=zone,
        movement_mode=mode,
        move_type=move_type,
    )


def test_temporary_hold_does_not_replace_blocked_offensive() -> None:
    original = _army_command("assault")
    state = GroupRetreatState(
        state=STATE_RETREATING,
        original_command=original,
    )

    assert preserve_blocked_offensive_command(
        state, _army_command("hold", "zone_1")
    ) is original


def test_cached_regroup_does_not_replace_offensive_after_recovery() -> None:
    original = _army_command("assault")
    state = GroupRetreatState(
        state=STATE_ACTIVE,
        original_command=original,
    )

    assert preserve_blocked_offensive_command(
        state, _army_command("regroup", "zone_1")
    ) is original


def test_explicit_retreat_can_cancel_blocked_offensive() -> None:
    original = _army_command("assault")
    explicit = _army_command("defensive_retreat", "zone_1")
    state = GroupRetreatState(
        state=STATE_RETREATING,
        original_command=original,
    )

    assert preserve_blocked_offensive_command(state, explicit) is explicit


def test_policy_sync_keeps_reinforcements_on_blocked_offensive() -> None:
    original = _army_command("assault")
    act = object.__new__(CombatControlAct)
    act._current_groups = {"group_0": object(), "group_1": object()}
    act._main_group_id = "group_0"
    act._retreat_states = {
        "group_0": GroupRetreatState(
            state=STATE_RETREATING,
            original_command=original,
        )
    }
    incoming = ArmyControlPolicy(
        commands=[
            _army_command("hold", "zone_1"),
            ArmyGroupCommand(
                group_id="group_1",
                destination_zone_id="zone_1",
                movement_mode="regroup",
                move_type="ReGroup",
            ),
        ]
    )

    synced = act._sync_policy_with_groups(incoming)

    assert [(item.group_id, item.movement_mode, item.destination_zone_id) for item in synced.commands] == [
        ("group_0", "assault", "zone_15"),
        ("group_1", "assault", "zone_15"),
    ]
