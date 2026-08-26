"""Deterministic bot aligned with ``skills/terran/battlecruiser/strategy.md``.

Efficiency focus: survive with early Factory tanks + bunker, rush the
Starport/Fusion Core path, dedicate production (tanks then thors on factories,
BCs on starports), and only attack after Yamato plus the 6/4/6 gate.
"""

from __future__ import annotations

from typing import Mapping

from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sharpy.general.extended_power import ExtendedPower
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
    if unit_type == UnitTypeId.THOR:
        return (
            cache.own(UnitTypeId.THOR).ready.amount
            + cache.own(UnitTypeId.THORAP).ready.amount
        )
    return cache.own(unit_type).ready.amount


class _BattlecruiserGateAttack(PlanZoneAttack):
    def __init__(self, composition: Mapping[UnitTypeId, int]):
        super().__init__(start_attack_power=1)
        self.composition = dict(composition)
        self.attack_on_advantage = False

    def _yamato_ready(self) -> bool:
        return UpgradeId.YAMATOCANNON in self.ai.state.upgrades

    def composition_ready(self) -> bool:
        if not self._yamato_ready():
            return False
        return all(
            _living(self.cache, unit_type) >= need
            for unit_type, need in self.composition.items()
        )

    def _should_attack(self, power: ExtendedPower) -> bool:
        return self.composition_ready()


class SkillBattlecruiserBot(KnowledgeBot):
    def __init__(self):
        super().__init__("SkillBattlecruiser")
        self.attack = _BattlecruiserGateAttack(
            {
                UnitTypeId.BATTLECRUISER: 6,
                UnitTypeId.THOR: 4,
                UnitTypeId.SIEGETANK: 6,
            }
        )

    def _bc_started(self, _knowledge) -> bool:
        return self.cache.own(UnitTypeId.BATTLECRUISER).amount >= 1

    def _tanks_ready_for_thors(self, _knowledge) -> bool:
        # Finish the tank half of the gate before spending factories on Thors.
        return _living(self.cache, UnitTypeId.SIEGETANK) >= 6

    async def create_plan(self) -> BuildOrder:
        scvs = [
            Step(
                None,
                ActUnit(UnitTypeId.SCV, UnitTypeId.COMMANDCENTER, 22),
                skip=UnitExists(UnitTypeId.COMMANDCENTER, 2),
            ),
            Step(None, ActUnit(UnitTypeId.SCV, UnitTypeId.COMMANDCENTER, 50)),
        ]
        buildings = [
            Step(Supply(13), GridBuilding(UnitTypeId.SUPPLYDEPOT, 1, priority=True)),
            Step(UnitReady(UnitTypeId.SUPPLYDEPOT, 0.95), GridBuilding(UnitTypeId.BARRACKS, 1, priority=True)),
            StepBuildGas(1, Supply(15)),
            Step(None, MorphOrbitals(), skip_until=UnitReady(UnitTypeId.BARRACKS, 0.1)),
            Step(UnitExists(UnitTypeId.BARRACKS, 1), Expand(2)),
            Step(Supply(18), GridBuilding(UnitTypeId.SUPPLYDEPOT, 2)),
            StepBuildGas(2, UnitExists(UnitTypeId.COMMANDCENTER, 2)),
            # 1-1-1 into Fusion Core: Factory -> Starport ASAP.
            Step(None, GridBuilding(UnitTypeId.FACTORY, 1), skip_until=UnitReady(UnitTypeId.BARRACKS, 1)),
            Step(
                UnitReady(UnitTypeId.FACTORY, 1),
                BuildAddon(UnitTypeId.FACTORYTECHLAB, UnitTypeId.FACTORY, 1),
            ),
            Step(
                UnitReady(UnitTypeId.FACTORY, 1),
                DefensiveBuilding(UnitTypeId.BUNKER, DefensePosition.Entrance, 1),
            ),
            Step(None, GridBuilding(UnitTypeId.STARPORT, 1), skip_until=UnitReady(UnitTypeId.FACTORY, 1)),
            # Third base only after the tech switch is online (not before Factory).
            Step(UnitReady(UnitTypeId.STARPORT, 1), Expand(3)),
            BuildGas(4),
            Step(UnitReady(UnitTypeId.STARPORT, 1), GridBuilding(UnitTypeId.FUSIONCORE, 1)),
            Step(UnitReady(UnitTypeId.STARPORT, 1), BuildAddon(UnitTypeId.STARPORTTECHLAB, UnitTypeId.STARPORT, 1)),
            Step(UnitReady(UnitTypeId.FACTORY, 1), GridBuilding(UnitTypeId.ARMORY, 1)),
            Step(UnitReady(UnitTypeId.SIEGETANK, 2), GridBuilding(UnitTypeId.FACTORY, 2)),
            Step(None, BuildAddon(UnitTypeId.FACTORYTECHLAB, UnitTypeId.FACTORY, 2)),
            Step(UnitReady(UnitTypeId.FUSIONCORE, 0.5), GridBuilding(UnitTypeId.STARPORT, 2)),
            Step(None, BuildAddon(UnitTypeId.STARPORTTECHLAB, UnitTypeId.STARPORT, 2)),
            Step(UnitReady(UnitTypeId.FUSIONCORE, 1), Tech(UpgradeId.YAMATOCANNON)),
            BuildGas(6),
            Step(None, Expand(4), skip_until=RequireCustom(self._bc_started)),
            BuildGas(8),
        ]
        army = [
            Step(UnitReady(UnitTypeId.BARRACKS, 1), ActUnit(UnitTypeId.MARINE, UnitTypeId.BARRACKS, 12)),
            # Early tanks for the defensive opening; keep tanks prioritized to the gate.
            Step(
                UnitReady(UnitTypeId.FACTORYTECHLAB, 1),
                ActUnit(UnitTypeId.SIEGETANK, UnitTypeId.FACTORY, 8, priority=True),
            ),
            # Thors only after the tank gate count is secured (avoids factory thrash).
            Step(
                RequireCustom(self._tanks_ready_for_thors),
                ActUnit(UnitTypeId.THOR, UnitTypeId.FACTORY, 4, priority=True),
                skip_until=UnitReady(UnitTypeId.ARMORY, 1),
            ),
            Step(
                UnitReady(UnitTypeId.FUSIONCORE, 1),
                ActUnit(UnitTypeId.BATTLECRUISER, UnitTypeId.STARPORT, 12, priority=True),
            ),
        ]
        tactics = [
            MineOpenBlockedBase(),
            PlanCancelBuilding(),
            LowerDepots(),
            PlanZoneDefense(),
            ManTheBunkers(),
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


class LadderBot(SkillBattlecruiserBot):
    @property
    def my_race(self):
        return Race.Terran
