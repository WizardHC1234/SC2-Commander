"""Per-group auto-retreat state machine for advancing army commands.

Follows the shape of Sharpy's native ``PlanZoneAttack`` retreat logic
(front-runner local battle ratio, arrival-stop, time-based recovery). The
threshold travels with each ``move_group`` command (``retreat_ratio``);
there is no global persistent policy.
"""

from dataclasses import dataclass
from typing import Optional

from commander.combat_policy import ArmyGroupCommand

# --- retreat_ratio range and default (move_group parameter) ---
RETREAT_MIN = 0.3
RETREAT_MAX = 1.5
DEFAULT_RETREAT_RATIO = 0.6
# hysteresis: resume the original command once the local ratio climbs back
# to retreat_ratio + RECOVER_MARGIN
RECOVER_MARGIN = 0.4

# --- evaluation geometry / timing (runtime constants, not LLM-tunable) ---
# radius of the local battle circle around the group's front runner
LOCAL_BATTLE_RADIUS = 28.0
# Friendly reinforcements can support a fight before permanent group membership
# catches up, so use a wider radius for retreat evaluation.
RETREAT_SUPPORT_RADIUS = 40.0
# Mission-wide power may protect only a tiny vanguard/remnant from causing an
# operation-wide withdrawal. It must not rescue an isolated main force.
RETREAT_FRAGMENT_POWER_SHARE = 0.25
# Ignore brief pathing and targeting fluctuations; catastrophic losses remain
# immediate.
RETREAT_CONFIRM_SECONDS = 2.0
CATASTROPHIC_RETREAT_RATIO = 0.25
# time-based recovery from retreating/holding (game seconds)
RETREAT_TIME_CAP_SECONDS = 60.0
# distance to the retreat-zone center that counts as "arrived" -> holding
ARRIVAL_RADIUS = 10.0

# movement modes the survival gate watches: advancing forces plus contain
# positions (a blockade can be counter-attacked). Explicit hold/regroup/
# retreat orders are deliberate stances and are never interrupted.
RETREAT_WATCHED_MODES = frozenset({"assault", "push", "harass", "contain"})

STATE_ACTIVE = "active"
STATE_RETREATING = "retreating"
STATE_HOLDING = "holding"


@dataclass
class GroupRetreatState:
    """Mutable per-group machine state owned by the combat act."""

    state: str = STATE_ACTIVE
    original_command: Optional[ArmyGroupCommand] = None
    retreat_zone_id: Optional[str] = None
    since: float = 0.0
    detail: str = ""
    below_threshold_since: Optional[float] = None
    group_ratio: float = float("inf")
    support_ratio: float = float("inf")
    mission_ratio: float = float("inf")
    effective_ratio: float = float("inf")
    group_power_share: float = 1.0
    support_power: float = 0.0
    enemy_power: float = 0.0
    observed_command_signature: str = ""


def clamp_retreat_ratio(value: float) -> float:
    return max(RETREAT_MIN, min(RETREAT_MAX, value))


def effective_retreat_ratio(
    *,
    group_ratio: float,
    support_ratio: float,
    mission_ratio: float,
    group_power_share: float,
) -> float:
    """Choose a retreat ratio without mistaking global inventory for support."""
    ratio = max(float(group_ratio), float(support_ratio))
    if float(group_power_share) <= RETREAT_FRAGMENT_POWER_SHARE:
        ratio = max(ratio, float(mission_ratio))
    return ratio


def retreat_confirmation_ready(
    *,
    now: float,
    below_threshold_since: Optional[float],
    effective_ratio_value: float,
) -> bool:
    """Return whether a low-power observation is persistent enough to retreat."""
    if effective_ratio_value < CATASTROPHIC_RETREAT_RATIO:
        return True
    if below_threshold_since is None:
        return False
    return float(now) - float(below_threshold_since) >= RETREAT_CONFIRM_SECONDS
