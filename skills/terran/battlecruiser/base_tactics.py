from sc2.ids.unit_typeid import UnitTypeId

from sharpy.plans import BuildOrder
from sharpy.plans.acts import *
from sharpy.plans.acts.terran import *
from sharpy.plans.build_step import Step
from sharpy.plans.require import UnitReady
from sharpy.plans.tactics import *
from sharpy.plans.tactics.terran import *

from commander.combat_exec import CombatControlAct


class LateBattlecruiserTactics(BuildOrder):

    def __init__(self):
        super().__init__(
            [
                AutoDepot(),
                Step(None, MorphOrbitals(), skip_until=UnitReady(UnitTypeId.BARRACKS, 1)),
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
