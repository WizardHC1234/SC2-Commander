"""Unit tests for the runtime-owned auto-retreat threshold."""

from commander.retreat_policy import (
    DEFAULT_RETREAT_RATIO,
    RECOVER_MARGIN,
    RETREAT_MAX,
    RETREAT_MIN,
    clamp_retreat_ratio,
)
def test_clamp_bounds():
    assert clamp_retreat_ratio(1.2) == 1.2
    assert clamp_retreat_ratio(99) == RETREAT_MAX
    assert clamp_retreat_ratio(-1) == RETREAT_MIN


def test_recover_margin_hysteresis():
    """Recovery threshold stays meaningfully above the trigger."""
    assert DEFAULT_RETREAT_RATIO + RECOVER_MARGIN >= 1.0
    assert RECOVER_MARGIN >= 0.3
