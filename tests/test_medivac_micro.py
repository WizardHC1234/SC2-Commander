from types import SimpleNamespace

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from sharpy.combat import Action
from sharpy.combat.terran.micro_medivacs import MicroMedivacs


class FakeUnits(list):
    @property
    def exists(self):
        return bool(self)

    def filter(self, predicate):
        return FakeUnits(unit for unit in self if predicate(unit))

    def closest_to(self, target):
        target_position = getattr(target, "position", target)
        return min(self, key=lambda unit: unit.position.distance_to(target_position))


def fake_unit(
    tag,
    position,
    *,
    type_id=UnitTypeId.MARINE,
    health_percentage=1.0,
    biological=True,
    flying=False,
    energy=0,
):
    return SimpleNamespace(
        tag=tag,
        position=Point2(position),
        type_id=type_id,
        health_percentage=health_percentage,
        is_biological=biological,
        is_flying=flying,
        energy=energy,
    )


def make_micro(*, group_units, all_own):
    micro = MicroMedivacs()
    micro.group = SimpleNamespace(
        ground_units=FakeUnits(group_units),
        center=(
            FakeUnits(group_units)[0].position
            if group_units
            else Point2((0, 0))
        ),
    )
    micro.cache = SimpleNamespace(all_own=FakeUnits(all_own))
    micro.unit_values = SimpleNamespace(
        should_attack=lambda unit: not getattr(unit, "is_worker", False)
    )
    micro.enemies_near_by = FakeUnits()
    micro.pather = SimpleNamespace(
        find_weak_influence_air=lambda point, _radius: point
    )
    return micro


def test_medivac_heals_a_unit_target_instead_of_attacking_a_point():
    marine = fake_unit(1, (10, 10), health_percentage=0.5)
    medivac = fake_unit(
        7,
        (12, 10),
        type_id=UnitTypeId.MEDIVAC,
        flying=True,
        biological=False,
        energy=50,
    )
    micro = make_micro(group_units=[marine], all_own=[marine, medivac])

    result = micro.unit_solve_combat(
        medivac,
        Action(Point2((40, 40)), True),
    )

    assert result.target is marine
    assert result.ability == AbilityId.MEDIVACHEAL_HEAL
    assert result.is_attack is False


def test_detached_medivac_moves_back_to_ground_army_not_attack_destination():
    marine = fake_unit(1, (20, 20))
    medivac = fake_unit(
        7,
        (2, 2),
        type_id=UnitTypeId.MEDIVAC,
        flying=True,
        biological=False,
        energy=50,
    )
    micro = make_micro(group_units=[], all_own=[marine, medivac])

    result = micro.unit_solve_combat(
        medivac,
        Action(Point2((80, 80)), True),
    )

    assert result.ability is None
    assert result.is_attack is False
    assert result.target.distance_to(marine.position) < 5
    assert result.target.distance_to(Point2((80, 80))) > 50
