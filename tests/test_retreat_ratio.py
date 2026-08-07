"""Unit tests for move_group's retreat_ratio parameter (auto-retreat gate)."""

from commander.retreat_policy import (
    DEFAULT_RETREAT_RATIO,
    RECOVER_MARGIN,
    RETREAT_MAX,
    RETREAT_MIN,
    clamp_retreat_ratio,
)
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
