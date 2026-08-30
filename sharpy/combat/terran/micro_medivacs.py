from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.units import Units
from sharpy.combat import MicroStep, Action
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit


class MicroMedivacs(MicroStep):
    HEAL_ENERGY_FLOOR = 5
    ESCORT_OFFSET = 2.5

    def __init__(self):
        super().__init__()

    def group_solve_combat(self, units: Units, current_command: Action) -> Action:
        return current_command

    def unit_solve_combat(self, unit: Unit, current_command: Action) -> Action:
        healable_targets = self.group.ground_units.filter(
            self._is_healable
        )

        # A Medivac's generic ATTACK command is its heal command, which requires
        # a Unit target. Passing the army's Point2 destination therefore causes
        # repeated "Must target unit" errors and leaves support aircraft behind.
        # Heal a concrete nearby group member when possible; otherwise issue a
        # MOVE toward the escorted ground force.
        if unit.energy >= self.HEAL_ENERGY_FLOOR and healable_targets:
            return Action(
                healable_targets.closest_to(unit),
                False,
                AbilityId.MEDIVACHEAL_HEAL,
            )

        escort_point = self._escort_point(unit, current_command)
        if self.enemies_near_by:
            escort_point = self.pather.find_weak_influence_air(escort_point, 8)

        return Action(escort_point, False)

    @staticmethod
    def _is_healable(target: Unit) -> bool:
        return (
            target.health_percentage < 1
            and not target.is_flying
            and (
                target.is_biological
                or target.type_id == UnitTypeId.HELLIONTANK
            )
        )

    def _escort_point(self, unit: Unit, current_command: Action) -> Point2:
        ground_escorts = self.group.ground_units
        if ground_escorts:
            anchor = self.group.center
        else:
            # Once Medivacs are more than Sharpy's grouping radius from the
            # army, their local combat group contains only aircraft. Recover by
            # leashing them to the nearest actual ground combat unit, excluding
            # workers and structures through UnitValue.should_attack().
            ground_escorts = self.cache.all_own.filter(
                lambda candidate: (
                    not candidate.is_flying
                    and self.unit_values.should_attack(candidate)
                )
            )
            if ground_escorts:
                anchor = ground_escorts.closest_to(unit).position
            elif current_command.position is not None:
                anchor = current_command.position
            else:
                anchor = unit.position

        target = current_command.position
        if target is not None and anchor.distance_to(target) > 0.1:
            # Remain slightly behind the escorted force instead of racing ahead
            # toward the army's destination at Medivac movement speed.
            anchor = anchor.towards(target, -self.ESCORT_OFFSET)

        # Keep multiple Medivacs from collapsing onto one exact coordinate.
        x_offset = (unit.tag % 5 - 2) * 0.55
        y_offset = (unit.tag % 7 - 3) * 0.45
        return anchor + Point2((x_offset, y_offset))
