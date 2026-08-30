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
# When the persistent main-group id has become a minority of the operation,
# nearby and mission-wide reinforcements are the effective fighting force. Use
# their combined power so an old main-force remnant does not trigger a false
# operation-wide withdrawal. A group that still owns most of the mission must
# stand on its own local support instead of being rescued by distant inventory.
RETREAT_FRAGMENT_POWER_SHARE = 0.45
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

# These modes are also the runtime's own temporary retreat postures. If the LLM
# repeats one of them while an automatic retreat is active, it must not silently
# replace the recoverable offensive command. An explicit defensive_retreat or
# panic_retreat still cancels the offensive normally.
TRANSIENT_OVERRIDE_MODES = frozenset({"hold", "regroup"})


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


def preserve_blocked_offensive_command(
    state: Optional[GroupRetreatState],
    incoming: ArmyGroupCommand,
) -> ArmyGroupCommand:
    """Keep a recoverable offensive through an automatic retreat.

    Commander decisions are event driven, so a model may echo the observed
    temporary hold/regroup posture after the runtime wakes it. Treating that echo
    as a new strategic command discarded ``original_command`` and converted a
    short tactical withdrawal into an indefinite rebuild loop. Preserve the
    command through recovery until an advancing command selects a new objective or
    an existing explicit retreat mode cancels the offensive intentionally.
    """

    if (
        state is not None
        and state.original_command is not None
        and incoming.movement_mode in TRANSIENT_OVERRIDE_MODES
    ):
        return state.original_command
    return incoming


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
