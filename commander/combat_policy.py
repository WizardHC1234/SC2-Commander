from dataclasses import dataclass, field
from typing import List, Optional


ALLOWED_MOVEMENT_MODES = {
    "regroup",
    "push",
    "assault",
    "harass",
    "hold",
    "contain",
    "defensive_retreat",
    "panic_retreat",
    "search_and_destroy",
}
MOVE_TYPE_BY_MOVEMENT_MODE = {
    "regroup": "ReGroup",
    "push": "Push",
    "assault": "Assault",
    "harass": "Harass",
    # hold/contain use MoveType.Hold micro: settle at the resolved point,
    # shoot enemies in range, never chase and never attack structures.
    "hold": "Hold",
    "contain": "Hold",
    "defensive_retreat": "DefensiveRetreat",
    "panic_retreat": "PanicRetreat",
    "search_and_destroy": "SearchAndDestroy",
}


@dataclass(frozen=True)
class ArmyGroupCommand:
    group_id: str
    destination_zone_id: str
    movement_mode: str
    move_type: str
    # Optional survival-gate threshold for advancing modes (assault/push/
    # harass/contain): when the local battle ratio drops below this value the
    # group auto-retreats, resuming this command once the ratio recovers.
    # None means the runtime default.
    retreat_ratio: Optional[float] = None


@dataclass(frozen=True)
class ArmyIntent:
    """Persistent high-level army intent owned by the Commander.

    The combat act expands this into per-group commands every frame.  The
    model therefore chooses only the strategic stance and zone; it does not
    micromanage group membership.
    """

    mode: str
    zone_id: str


@dataclass(frozen=True)
class ArmyControlPolicy:
    commands: List[ArmyGroupCommand] = field(default_factory=list)
    scan_zone_id: Optional[str] = None
    scout_zone_id: Optional[str] = None
    army_intent: Optional[ArmyIntent] = None


class InjectedArmyPolicyProvider:
    """Reads the latest Commander-applied policy from ``ai.commander_army_policy``."""

    def get_policy(self, act) -> ArmyControlPolicy:
        policy = getattr(getattr(act, "ai", None), "commander_army_policy", None)
        if isinstance(policy, ArmyControlPolicy):
            return policy
        return ArmyControlPolicy(
            commands=[],
            scan_zone_id=None,
            scout_zone_id=None,
        )
