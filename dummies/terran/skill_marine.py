"""Deterministic bot aligned with ``skills/terran/marine/strategy.md``.

Efficiency focus: cut workers early, open the first Barracks ASAP, then dump
minerals into 6 Barracks and Marines. Attack gate remains 20 living Marines.
"""

from __future__ import annotations

from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sharpy.general.extended_power import ExtendedPower
from sharpy.interfaces import IZoneManager
from sharpy.knowledges import KnowledgeBot
from sharpy.plans import BuildOrder, Step, SequentialList
from sharpy.plans.acts import *
from sharpy.plans.acts.terran import *
from sharpy.plans.require import *
from sharpy.plans.tactics import *
from sharpy.plans.tactics.terran import *
from sharpy.plans.tactics.zone_attack import PlanZoneAttack


class _MarineGateAttack(PlanZoneAttack):
    def __init__(self, marine_count: int = 20):
        super().__init__(start_attack_power=1)
        self.marine_count = marine_count
        self.attack_on_advantage = False

    def composition_ready(self) -> bool:
        return self.cache.own(UnitTypeId.MARINE).ready.amount >= self.marine_count

    def _should_attack(self, power: ExtendedPower) -> bool:
        return self.composition_ready()


class SkillMarineBot(KnowledgeBot):
    zone_manager: IZoneManager

    def __init__(self):
        super().__init__("SkillMarine")
        self.attack = _MarineGateAttack(20)

    async def on_start(self):
        await super().on_start()
        self.zone_manager = self.knowledge.get_required_manager(IZoneManager)

    async def pre_step_execute(self):
        if self.time >= 8 * 60:
            return
        if getattr(self.attack, "cache", None) is not None and self.attack.composition_ready():
            return
        self.knowledge.gather_point = self.zone_manager.expansion_zones[-2].gather_point

    async def create_plan(self) -> BuildOrder:
        # Fast one-base all-in path (no gas): depot -> rax -> cut SCV -> mass rax.
        opening = [
            Step(Supply(13), GridBuilding(UnitTypeId.SUPPLYDEPOT, 1, priority=True)),
            Step(UnitReady(UnitTypeId.SUPPLYDEPOT, 0.95), GridBuilding(UnitTypeId.BARRACKS, 1, priority=True)),
            Step(None, MorphOrbitals(), skip_until=UnitReady(UnitTypeId.BARRACKS, 0.1)),
            Step(Supply(16), GridBuilding(UnitTypeId.SUPPLYDEPOT, 2, priority=True)),
            # Dump leftover minerals into Barracks count as soon as the first is down.
            Step(UnitReady(UnitTypeId.BARRACKS, 1), GridBuilding(UnitTypeId.BARRACKS, 6)),
        ]
        # Cap workers near the Skill target; prefer rax/marines after the cut.
        workers = [
            Step(None, ActUnit(UnitTypeId.SCV, UnitTypeId.COMMANDCENTER, 16)),
            Step(UnitReady(UnitTypeId.BARRACKS, 1), ActUnit(UnitTypeId.SCV, UnitTypeId.COMMANDCENTER, 20)),
        ]
        tactics = [
            MineOpenBlockedBase(),
            PlanCancelBuilding(),
            LowerDepots(),
            PlanZoneDefense(),
            Step(None, WorkerScout(), skip_until=UnitExists(UnitTypeId.SUPPLYDEPOT, 1)),
            Step(None, CallMule(40), skip=Time(4 * 60)),
            Step(None, CallMule(80), skip_until=Time(4 * 60)),
            Step(None, ScanEnemy(), skip_until=Time(5 * 60)),
            DistributeWorkers(max_gas=0),
            Step(None, SpeedMining(), lambda ai: ai.client.game_step > 5),
            Repair(),
            ContinueBuilding(),
            PlanZoneGatherTerran(),
            self.attack,
            PlanFinishEnemy(),
        ]
        return BuildOrder(
            AutoDepot(),
            workers,
            opening,
            # Marines from the first Barracks immediately; target Ultimate Goal count.
            Step(UnitReady(UnitTypeId.BARRACKS, 1), ActUnit(UnitTypeId.MARINE, UnitTypeId.BARRACKS, 180)),
            SequentialList(tactics),
        )


class LadderBot(SkillMarineBot):
    @property
    def my_race(self):
        return Race.Terran
