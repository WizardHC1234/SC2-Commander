"""Always-on Terran automation shared by every Commander strategy."""

from sc2.ids.unit_typeid import UnitTypeId

from sharpy.plans import BuildOrder
from sharpy.plans.acts import MineOpenBlockedBase
from sharpy.plans.acts.terran import AutoDepot, MorphOrbitals
from sharpy.plans.build_step import Step
from sharpy.plans.require import UnitReady
from sharpy.plans.tactics import (
    DistributeWorkers,
    PlanCancelBuilding,
    PlanZoneDefense,
    SpeedMining,
)
from sharpy.plans.tactics.terran import (
    CallMule,
    ContinueBuilding,
    LowerDepots,
    Repair,
)

from commander.combat_exec import CombatControlAct


class TerranTactics(BuildOrder):
    """Race-level automation that is independent of the selected strategy."""

    def __init__(self) -> None:
        super().__init__(
            [
                AutoDepot(),
                Step(
                    None,
                    MorphOrbitals(),
                    skip_until=UnitReady(UnitTypeId.BARRACKS, 1),
                ),
                MineOpenBlockedBase(),
                PlanCancelBuilding(),
                LowerDepots(),
                PlanZoneDefense(),
                CallMule(),
                DistributeWorkers(),
                Step(None, SpeedMining(), lambda ai: ai.client.game_step > 5),
                Repair(),
                ContinueBuilding(),
                CombatControlAct(),
            ]
        )


def create_tactics() -> BuildOrder:
    """Create the Terran always-on tactics plan."""
    return TerranTactics()


__all__ = ["TerranTactics", "create_tactics"]
