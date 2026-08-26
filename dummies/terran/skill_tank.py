"""Deterministic bot aligned with ``skills/terran/tank/strategy.md``.

Efficiency focus: open Factory tanks as early as possible on two bases, scale
Barracks with reactors in parallel, finish Combat Shield without delaying the
45 Marine + 10 Tank gate.
"""

from __future__ import annotations

from typing import Mapping

from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sharpy.general.extended_power import ExtendedPower
from sharpy.interfaces import IZoneManager
from sharpy.knowledges import KnowledgeBot
from sharpy.plans import BuildOrder, Step, StepBuildGas, SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.terran import *
from sharpy.plans.require import *
from sharpy.plans.tactics import *
from sharpy.plans.tactics.terran import *
from sharpy.plans.tactics.zone_attack import PlanZoneAttack


def _living(cache, unit_type: UnitTypeId) -> int:
    if unit_type == UnitTypeId.SIEGETANK:
        return (
            cache.own(UnitTypeId.SIEGETANK).ready.amount
            + cache.own(UnitTypeId.SIEGETANKSIEGED).ready.amount
        )
    return cache.own(unit_type).ready.amount


class _TankGateAttack(PlanZoneAttack):
    def __init__(self, composition: Mapping[UnitTypeId, int]):
        super().__init__(start_attack_power=1)
        self.composition = dict(composition)
        self.attack_on_advantage = False

    def composition_ready(self) -> bool:
        return all(
            _living(self.cache, unit_type) >= need
            for unit_type, need in self.composition.items()
        )

    def _should_attack(self, power: ExtendedPower) -> bool:
        return self.composition_ready()


class SkillTankBot(KnowledgeBot):
    zone_manager: IZoneManager

    def __init__(self):
        super().__init__("SkillTank")
        self.attack = _TankGateAttack(
            {UnitTypeId.MARINE: 45, UnitTypeId.SIEGETANK: 10}
        )

    async def on_start(self):
        await super().on_start()
        self.zone_manager = self.knowledge.get_required_manager(IZoneManager)

    def _gate_reached(self, _knowledge) -> bool:
        return (
            _living(self.cache, UnitTypeId.MARINE) >= 45
            and _living(self.cache, UnitTypeId.SIEGETANK) >= 10
        )

    async def create_plan(self) -> BuildOrder:
        # Two-base tank timing: rax -> gas -> expand -> factory/tank ASAP,
        # then scale barracks reactors while the second factory comes up.
        scvs = [
            Step(
                None,
                ActUnit(UnitTypeId.SCV, UnitTypeId.COMMANDCENTER, 22),
                skip=UnitExists(UnitTypeId.COMMANDCENTER, 2),
            ),
            Step(None, ActUnit(UnitTypeId.SCV, UnitTypeId.COMMANDCENTER, 44)),
        ]
        buildings = [
            Step(Supply(13), GridBuilding(UnitTypeId.SUPPLYDEPOT, 1, priority=True)),
            Step(UnitReady(UnitTypeId.SUPPLYDEPOT, 0.95), GridBuilding(UnitTypeId.BARRACKS, 1, priority=True)),
            StepBuildGas(1, Supply(15)),
            Step(None, MorphOrbitals(), skip_until=UnitReady(UnitTypeId.BARRACKS, 0.1)),
            # Natural as soon as the Barracks is started / first marine is in flight.
            Step(UnitExists(UnitTypeId.BARRACKS, 1), Expand(2)),
            Step(Supply(16), GridBuilding(UnitTypeId.SUPPLYDEPOT, 2)),
            StepBuildGas(2, UnitExists(UnitTypeId.COMMANDCENTER, 2)),
            Step(None, GridBuilding(UnitTypeId.FACTORY, 1), skip_until=UnitReady(UnitTypeId.BARRACKS, 1)),
            Step(
                UnitReady(UnitTypeId.FACTORY, 1),
                BuildAddon(UnitTypeId.FACTORYTECHLAB, UnitTypeId.FACTORY, 1),
            ),
            # Scale infantry production without delaying the first tanks.
            Step(UnitReady(UnitTypeId.FACTORY, 1), GridBuilding(UnitTypeId.BARRACKS, 3)),
            Step(None, BuildAddon(UnitTypeId.BARRACKSREACTOR, UnitTypeId.BARRACKS, 2)),
            Step(None, BuildAddon(UnitTypeId.BARRACKSTECHLAB, UnitTypeId.BARRACKS, 1)),
            Step(UnitReady(UnitTypeId.BARRACKSTECHLAB, 1), Tech(UpgradeId.SHIELDWALL)),
            Step(UnitReady(UnitTypeId.SIEGETANK, 1), GridBuilding(UnitTypeId.FACTORY, 2)),
            Step(None, BuildAddon(UnitTypeId.FACTORYTECHLAB, UnitTypeId.FACTORY, 2)),
            BuildGas(4),
            Step(None, Expand(3), skip_until=RequireCustom(self._gate_reached)),
        ]
        army = [
            # Marines immediately; tanks as soon as the tech lab is ready.
            Step(UnitReady(UnitTypeId.BARRACKS, 1), ActUnit(UnitTypeId.MARINE, UnitTypeId.BARRACKS, 96)),
            Step(
                UnitReady(UnitTypeId.FACTORYTECHLAB, 1),
                ActUnit(UnitTypeId.SIEGETANK, UnitTypeId.FACTORY, 20, priority=True),
            ),
        ]
        tactics = [
            MineOpenBlockedBase(),
            PlanCancelBuilding(),
            LowerDepots(),
            PlanZoneDefense(),
            Step(None, WorkerScout(), skip_until=UnitExists(UnitTypeId.BARRACKS, 1)),
            Step(None, CallMule(50), skip=Time(5 * 60)),
            Step(None, CallMule(100), skip_until=Time(5 * 60)),
            Step(None, ScanEnemy(), skip_until=Time(5 * 60)),
            DistributeWorkers(),
            Step(None, SpeedMining(), lambda ai: ai.client.game_step > 5),
            Repair(),
            ContinueBuilding(),
            PlanZoneGatherTerran(),
            self.attack,
            PlanFinishEnemy(),
        ]
        return BuildOrder(
            AutoDepot(),
            scvs,
            buildings,
            army,
            SequentialList(tactics),
        )


class LadderBot(SkillTankBot):
    @property
    def my_race(self):
        return Race.Terran
