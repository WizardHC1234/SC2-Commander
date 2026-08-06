from typing import Dict, List, Optional

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.units import Units
from sharpy.combat import MicroRules
from sharpy.interfaces import IGatherPointSolver
from sharpy.interfaces.combat_manager import MoveType
from sharpy.managers.core.roles import UnitTask
from sharpy.plans.acts.act_base import ActBase
from sharpy.plans.tactics.terran.scan_enemy import ScanEnemy

from commander.combat_policy import (
    MOVE_TYPE_BY_MOVEMENT_MODE,
    ArmyControlPolicy,
    ArmyGroupCommand,
    InjectedArmyPolicyProvider,
)


MOVE_TYPE_BY_NAME = {
    "Assault": MoveType.Assault,
    "Push": MoveType.Push,
    "Harass": MoveType.Harass,
    "DefensiveRetreat": MoveType.DefensiveRetreat,
    "PanicRetreat": MoveType.PanicRetreat,
    "ReGroup": MoveType.ReGroup,
    "SearchAndDestroy": MoveType.SearchAndDestroy,
}
RETREAT_MODES = {"defensive_retreat", "panic_retreat"}
MOVING_MODES = {"regroup", *RETREAT_MODES}


class CombatControlAct(ActBase):
    """Main-force / reinforcement control driven by Commander tool_calls."""

    gather_manager: IGatherPointSolver

    MAIN_GROUP_ID = "group_0"
    REINFORCEMENT_GROUP_ID = "group_1"
    REINFORCEMENT_JOIN_DISTANCE = 14
    MIN_COHESIVE_POWER_SHARE = 0.8
    EXPANSION_CLEARANCE = 14
    SEARCH_ZONE_REVISIT_SECONDS = 60

    def __init__(self, policy_provider: Optional[object] = None):
        super().__init__()
        self.policy_provider = (
            policy_provider or InjectedArmyPolicyProvider()
        )
        self.micro_rules: Optional[MicroRules] = None
        self._current_groups: Dict[str, Units] = {}
        self._main_unit_tags: set = set()
        self._active_commands: Dict[str, dict] = {}
        self._main_group_id: Optional[str] = None
        self._search_target_zone_ids: Dict[str, str] = {}
        self._searched_zone_ids: Dict[str, set] = {}
        self._search_zone_assigned_at: Dict[str, float] = {}
        self._last_scan_policy_identity: Optional[int] = None
        self._last_scout_policy_identity: Optional[int] = None
        self._last_scan_zone_request: Optional[str] = None
        self._last_scout_zone_request: Optional[str] = None
        self._scout_requested_zone_id: Optional[str] = None
        self._scout_tag: Optional[int] = None
        self._scout_target_zone_id: Optional[str] = None
        self._last_scout_zone_id: Optional[str] = None
        self._last_scout_result: Optional[str] = None
        self._last_scout_result_time: Optional[float] = None
        self._rallied_production_tags: set = set()
        self._last_production_rally_target: Optional[Point2] = None
        self._scan_control = ScanEnemy(interval_seconds=None)
        self._last_scan_zone_id: Optional[str] = None
        self._last_scan_result: Optional[str] = None
        self._last_scan_result_time: Optional[float] = None

    async def start(self, knowledge):
        await super().start(knowledge)
        # The unified Observation builder discovers the live Army collector
        # through the bot. This avoids maintaining a second Army snapshot.
        self.ai.llm_army_control_act = self
        self.ai.llm_combat_execution_state = {}
        self.gather_manager = self.knowledge.get_required_manager(
            IGatherPointSolver
        )
        self.micro_rules = MicroRules()
        self.micro_rules.load_default_methods()
        self.micro_rules.load_default_micro()
        await self.micro_rules.start(knowledge)
        await self._scan_control.start(knowledge)

    async def execute(self) -> bool:
        self._update_production_rallies()
        # Refresh reconnaissance lifecycle before the Army Planner builds its
        # observation. This makes a death/interruption visible in the same
        # decision cycle instead of one cycle later.
        self._refresh_scout_status()
        policy = self.policy_provider.get_policy(self)
        scan_zone = policy.scan_zone_id
        scout_zone = policy.scout_zone_id
        if scan_zone != getattr(self, "_last_scan_zone_request", object()):
            self.ai.llm_scan_zone_id = scan_zone
            self._last_scan_zone_request = scan_zone
        if scout_zone != getattr(self, "_last_scout_zone_request", object()):
            self._set_scout_request(scout_zone)
            self._last_scout_zone_request = scout_zone
        requested_scan_zone = getattr(self.ai, "llm_scan_zone_id", None)
        previous_scan_time = self._scan_control.last_scan
        await self._scan_control.execute()
        if self._scan_control.last_scan > previous_scan_time:
            self._last_scan_zone_id = requested_scan_zone
            self._last_scan_result = "executed"
            self._last_scan_result_time = float(
                getattr(self.ai, "time", 0.0)
            )
        self._execute_scout_control()
        # Refresh groups every frame so newly produced units become group_1
        # before we apply (and sync) the last LLM policy.
        self._update_groups(self._select_main_army_units())
        policy = self._sync_policy_with_groups(policy)
        self._execute_policy(policy)
        return False

    def _sync_policy_with_groups(
        self, policy: ArmyControlPolicy
    ) -> ArmyControlPolicy:
        """Keep every live group commanded between LLM decisions.

        Cached LLM policies go stale when reinforcement (group_1) disappears
        and later reappears: the old group_1 order (often regroup-at-home)
        remains in the policy and would strand newly produced units away from
        an already-committed main-force offensive. Every frame, drop commands
        for groups that no longer exist and make non-main groups follow the
        main force's current objective/mode.
        """
        if not self._current_groups or not policy.commands:
            return policy

        by_id = {command.group_id: command for command in policy.commands}
        main_id = self._main_group_id or self.MAIN_GROUP_ID
        source = by_id.get(main_id)
        if source is None:
            source = next(iter(policy.commands), None)
        if source is None:
            return policy

        synced: List[ArmyGroupCommand] = []
        # Always keep the live main-force order first when present.
        if main_id in self._current_groups:
            main_command = by_id.get(main_id, source)
            if main_command.group_id != main_id:
                main_command = ArmyGroupCommand(
                    group_id=main_id,
                    destination_zone_id=source.destination_zone_id,
                    movement_mode=source.movement_mode,
                    move_type=MOVE_TYPE_BY_MOVEMENT_MODE[source.movement_mode],
                )
            synced.append(main_command)
            follow = main_command
        else:
            follow = source

        for group_id in sorted(
            self._current_groups,
            key=lambda value: int(value.removeprefix("group_")),
        ):
            if group_id == main_id:
                continue
            # Reinforcements always chase the main objective between decisions.
            synced.append(
                ArmyGroupCommand(
                    group_id=group_id,
                    destination_zone_id=follow.destination_zone_id,
                    movement_mode=follow.movement_mode,
                    move_type=MOVE_TYPE_BY_MOVEMENT_MODE[follow.movement_mode],
                )
            )

        if (
            len(synced) == len(policy.commands)
            and all(
                a.group_id == b.group_id
                and a.destination_zone_id == b.destination_zone_id
                and a.movement_mode == b.movement_mode
                for a, b in zip(synced, policy.commands)
            )
        ):
            return policy
        return ArmyControlPolicy(
            commands=synced,
            scan_zone_id=policy.scan_zone_id,
            scout_zone_id=policy.scout_zone_id,
        )
    def _update_production_rallies(self) -> None:
        target = getattr(self.gather_manager, "gather_point", None)
        if target is None:
            return
        if (
            self._last_production_rally_target is None
            or target.distance_to(self._last_production_rally_target) > 1
        ):
            self._rallied_production_tags.clear()
            self._last_production_rally_target = target

        buildings = self.cache.own(
            [UnitTypeId.BARRACKS, UnitTypeId.FACTORY]
        ).ready
        for building in buildings.tags_not_in(
            self._rallied_production_tags
        ):
            building(AbilityId.RALLY_BUILDING, target)
            self._rallied_production_tags.add(building.tag)
    def _set_scout_request(self, zone_id: Optional[str]) -> None:
        if zone_id is None:
            # Do not overwrite a just-detected death with "cancelled" when a
            # new policy clears the scout request at the same boundary.
            self._refresh_scout_status()
            if self._scout_tag is not None:
                self._record_scout_result(
                    self._scout_target_zone_id, "cancelled"
                )
            self._release_scout()
            self._scout_requested_zone_id = None
            return
        if zone_id != self._scout_target_zone_id:
            self._release_scout()
        self._scout_requested_zone_id = zone_id

    def _refresh_scout_status(self):
        """Return the active scout, or record why the tracked scout ended."""
        if self._scout_tag is None:
            return None
        scout = self.roles.get_unit_by_tag_from_task(
            self._scout_tag, UnitTask.Scouting
        )
        if scout is not None:
            return scout

        living_unit = self.cache.by_tag(self._scout_tag)
        result = "interrupted" if living_unit is not None else "killed_en_route"
        self._record_scout_result(self._scout_target_zone_id, result)
        self._scout_tag = None
        self._scout_target_zone_id = None
        self._scout_requested_zone_id = None
        return None

    def _execute_scout_control(self) -> None:
        scout = self._refresh_scout_status()
        zone_id = self._scout_requested_zone_id
        if zone_id is None:
            return
        zones = list(self.knowledge.zone_manager.expansion_zones)
        zone = self._zone_by_id(zones, zone_id)
        if zone is None:
            self._release_scout()
            self._scout_requested_zone_id = None
            return

        target = zone.center_location
        if scout is None:
            workers = self.roles.free_workers.of_type({UnitTypeId.SCV})
            if not workers.exists:
                return
            scout = min(workers, key=lambda worker: worker.distance_to(target))
            self.roles.set_task(UnitTask.Scouting, scout)
            self._scout_tag = scout.tag
            self._scout_target_zone_id = zone_id

        self.roles.refresh_task(scout)
        if scout.distance_to(target) <= 6:
            self._record_scout_result(zone_id, "completed")
            self._release_scout()
            self._scout_requested_zone_id = None
            return
        scout.move(target)

    def _record_scout_result(
        self, zone_id: Optional[str], result: str
    ) -> None:
        self._last_scout_zone_id = zone_id
        self._last_scout_result = result
        self._last_scout_result_time = float(
            getattr(self.ai, "time", 0.0)
        )

    def _release_scout(self) -> None:
        if self._scout_tag is not None:
            scout = self.roles.get_unit_by_tag_from_task(
                self._scout_tag, UnitTask.Scouting
            )
            if scout is not None:
                self.roles.clear_task(scout)
        self._scout_tag = None
        self._scout_target_zone_id = None

    def get_scv_scout_state(self) -> dict:
        scout_alive = False
        if self._scout_tag is not None:
            scout_alive = self.roles.get_unit_by_tag_from_task(
                self._scout_tag, UnitTask.Scouting
            ) is not None
        try:
            worker_count = len(self.ai.units(UnitTypeId.SCV))
        except (AttributeError, TypeError):
            worker_count = 0
        result_age = None
        if self._last_scout_result_time is not None:
            result_age = round(
                max(
                    0.0,
                    float(getattr(self.ai, "time", 0.0))
                    - self._last_scout_result_time,
                ),
                1,
            )
        return {
            "workers": worker_count,
            "active_scout": scout_alive,
            "scout_zone_id": self._scout_target_zone_id,
            "last_target_zone": self._last_scout_zone_id,
            "last_result": self._last_scout_result,
            "last_result_seconds_ago": result_age,
        }

    def get_scan_state(self) -> dict:
        result_age = None
        if self._last_scan_result_time is not None:
            result_age = round(
                max(
                    0.0,
                    float(getattr(self.ai, "time", 0.0))
                    - self._last_scan_result_time,
                ),
                1,
            )
        return {
            "last_target_zone": self._last_scan_zone_id,
            "last_result": self._last_scan_result,
            "last_result_seconds_ago": result_age,
        }
    def _execute_policy(self, policy: ArmyControlPolicy) -> None:
        search_command = next(
            (
                command
                for command in policy.commands
                if command.movement_mode == "search_and_destroy"
            ),
            None,
        )
        if search_command is not None:
            self._execute_search_and_destroy(search_command)
            return

        self._clear_all_search_state()
        for command in policy.commands:
            units = self._current_groups.get(command.group_id)
            if units is None or not units.exists:
                continue
            if command.movement_mode == "regroup":
                target = self._resolve_regroup_target(
                    command.destination_zone_id, units
                )
            elif command.movement_mode in {"push", "assault", "harass"}:
                target = self._resolve_target(
                    command.destination_zone_id,
                    enter_zone=True,
                )
            else:
                target = self._resolve_target(command.destination_zone_id)
            if target is None:
                continue

            self.roles.set_tasks(
                self._task_for_mode(command.movement_mode),
                units,
            )
            for unit in units:
                self.combat.add_unit(unit)

            rules = self._configure_rules(command)
            self.combat.execute(
                target,
                MOVE_TYPE_BY_NAME[command.move_type],
                rules,
            )
            self._active_commands[command.group_id] = {
                "destination_zone_id": command.destination_zone_id,
                "movement_mode": command.movement_mode,
                "issued_at": float(getattr(self.ai, "time", 0.0)),
            }

    def _execute_search_and_destroy(
        self,
        command: ArmyGroupCommand,
    ) -> None:
        """Keep existing search orders and dispatch only newly idle units."""
        units = self._all_current_group_units()
        if not units.exists:
            return

        self.roles.set_tasks(UnitTask.Attacking, units)
        now = float(getattr(self.ai, "time", 0.0))
        for group_id in self._current_groups:
            self._active_commands[group_id] = {
                "destination_zone_id": command.destination_zone_id,
                "movement_mode": command.movement_mode,
                "issued_at": now,
            }

        idle_units = units.filter(
            lambda unit: bool(getattr(unit, "is_idle", False))
        )
        if not idle_units.exists:
            return

        target = self._resolve_search_target(
            "search_all",
            idle_units,
            command.destination_zone_id,
        )
        if target is None:
            return

        idle_tags = {unit.tag for unit in idle_units}
        target_zone_id = self._search_target_zone_ids.get("search_all")
        for group_id, group_units in self._current_groups.items():
            if idle_tags.intersection(unit.tag for unit in group_units):
                if target_zone_id is None:
                    self._search_target_zone_ids.pop(group_id, None)
                else:
                    self._search_target_zone_ids[group_id] = target_zone_id

        for unit in idle_units:
            self.combat.add_unit(unit)
        rules = self._configure_rules(command)
        self.combat.execute(
            target,
            MOVE_TYPE_BY_NAME[command.move_type],
            rules,
        )

    def get_army_group_states(self, controlled_units: Units) -> List[dict]:
        self._update_groups(controlled_units)
        if not self._current_groups:
            return []

        powers = {
            group_id: self._power(units)
            for group_id, units in self._current_groups.items()
        }
        main_group_id = self._main_group_id
        states = []
        for group_id in sorted(
            self._current_groups,
            key=lambda value: int(value.removeprefix("group_")),
        ):
            units = self._current_groups[group_id]
            components = self._spatial_clusters(units)
            core = max(
                components,
                key=lambda component: (self._power(component), len(component)),
            )
            center = self._group_center(core)
            total_power = powers[group_id]
            core_power = self._power(core)
            if total_power > 0:
                cohesive_share = core_power / total_power
            else:
                cohesive_share = len(core) / max(len(units), 1)
            nearby_enemies = (
                self.ai.all_enemy_units.closer_than(24, center)
                if self.ai.all_enemy_units
                else []
            )
            current = self._active_commands.get(group_id)
            role = (
                "main_force"
                if group_id == main_group_id
                else "reinforcement"
            )
            state = {
                "group_id": group_id,
                "role": role,
                "unit_count": len(units),
                "power": round(powers[group_id], 2),
                "nearest_zone_id": self._nearest_zone_id(center),
                "unit_type_counts": self._unit_counts(units),
                "nearby_enemy_count": len(nearby_enemies),
                "nearby_enemy_power": round(
                    self._power(nearby_enemies), 2
                ),
                "nearby_enemy_type_counts": self._unit_counts(
                    nearby_enemies
                ),
                # A broad but connected formation is cohesive. A small
                # straggler also must not make the whole main force appear
                # fragmented or move its reported zone across a boundary.
                "is_fragmented": (
                    len(components) > 1
                    and cohesive_share < self.MIN_COHESIVE_POWER_SHARE
                ),
                "current_command": None,
            }
            if current is not None:
                state["current_command"] = {
                    key: current[key]
                    for key in (
                        "destination_zone_id",
                        "movement_mode",
                    )
                }
                if current["movement_mode"] == "search_and_destroy":
                    state["search_target_zone_id"] = (
                        self._search_target_zone_ids.get(group_id)
                    )
                    state["searched_zone_ids"] = sorted(
                        self._searched_zone_ids.get(group_id, set()),
                        key=lambda value: int(value.removeprefix("zone_")),
                    )
                state["command_age_seconds"] = round(
                    max(
                        0.0,
                        float(getattr(self.ai, "time", 0.0))
                        - current["issued_at"],
                    ),
                    1,
                )
            states.append(state)
        return states

    def _update_groups(self, units: Units) -> None:
        current_by_tag = {unit.tag: unit for unit in units}
        current_tags = set(current_by_tag)
        living_main_tags = self._living_main_tags(current_tags)

        if not living_main_tags and current_tags:
            # Establish or replace the operational main force only when the
            # previous one has no living members. Spatial clustering is used
            # only for this one-time selection, never for continuous splitting.
            clusters = self._spatial_clusters(units)
            selected = max(
                clusters,
                key=lambda cluster: (self._power(cluster), len(cluster)),
            )
            living_main_tags = {unit.tag for unit in selected}
            self._main_group_id = self.MAIN_GROUP_ID
        elif living_main_tags:
            self._main_group_id = self.MAIN_GROUP_ID
        else:
            self._main_group_id = None

        current_main = Units(
            [
                current_by_tag[tag]
                for tag in current_tags & living_main_tags
            ],
            self.ai,
        )
        reinforcement = Units(
            [
                current_by_tag[tag]
                for tag in current_tags - living_main_tags
            ],
            self.ai,
        )

        # Membership is monotonic while the main force lives. Reinforcements
        # join permanently once they physically reach any main-force member.
        while current_main.exists and reinforcement.exists:
            joining = reinforcement.filter(
                lambda unit: any(
                    unit.distance_to(main_unit)
                    <= self.REINFORCEMENT_JOIN_DISTANCE
                    for main_unit in current_main
                )
            )
            if not joining.exists:
                break
            current_main.extend(joining)
            joining_tags = {unit.tag for unit in joining}
            living_main_tags.update(joining_tags)
            reinforcement = reinforcement.tags_not_in(joining_tags)

        self._main_unit_tags = living_main_tags
        assigned: Dict[str, Units] = {}
        if current_main.exists:
            assigned[self.MAIN_GROUP_ID] = current_main
        if reinforcement.exists:
            assigned[self.REINFORCEMENT_GROUP_ID] = reinforcement
        self._current_groups = assigned

        for group_id in list(self._active_commands):
            if group_id not in assigned:
                del self._active_commands[group_id]
                self._clear_search_state(group_id)

    def _living_main_tags(self, current_tags: set) -> set:
        """Keep main membership through temporary non-Army role changes."""
        living = set(self._main_unit_tags & current_tags)
        cache = getattr(self, "cache", None)
        by_tag = getattr(cache, "by_tag", None)
        if not callable(by_tag):
            return living
        for tag in self._main_unit_tags - current_tags:
            try:
                unit = by_tag(tag)
            except Exception:
                unit = None
            if unit is not None:
                living.add(tag)
        return living

    def _spatial_clusters(self, units: Units) -> List[Units]:
        remaining = list(units)
        clusters = []
        while remaining:
            cluster = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                for unit in list(remaining):
                    if any(
                        unit.distance_to(member)
                        <= self.REINFORCEMENT_JOIN_DISTANCE
                        for member in cluster
                    ):
                        cluster.append(unit)
                        remaining.remove(unit)
                        changed = True
            clusters.append(Units(cluster, self.ai))
        return clusters

    def _nearest_zone_id(self, point: Point2) -> str:
        zones = list(self.knowledge.zone_manager.expansion_zones)
        if not zones:
            return "unknown"
        index = min(
            range(len(zones)),
            key=lambda value: point.distance_to(
                zones[value].center_location
            ),
        )
        return f"zone_{index}"

    @staticmethod
    def _group_center(units) -> Point2:
        amount = len(units)
        if amount == 0:
            return Point2((0, 0))
        return Point2(
            (
                sum(unit.position.x for unit in units) / amount,
                sum(unit.position.y for unit in units) / amount,
            )
        )
    def _power(self, units) -> float:
        if not units:
            return 0.0
        try:
            return float(
                self.unit_values.calc_total_power(units).power
            )
        except Exception:
            return float(len(units))

    @staticmethod
    def _unit_counts(units) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for unit in units:
            name = getattr(
                getattr(unit, "type_id", None),
                "name",
                "UNKNOWN",
            )
            counts[name] = counts.get(name, 0) + 1
        return counts

    def _configure_rules(
        self,
        command: ArmyGroupCommand,
    ) -> Optional[MicroRules]:
        if self.micro_rules is None:
            return None
        self.micro_rules.regroup = (
            command.movement_mode not in RETREAT_MODES
            and command.movement_mode != "search_and_destroy"
        )
        self.micro_rules.regroup_percentage = 0.75
        self.micro_rules.own_group_distance = 7
        return self.micro_rules

    def _select_main_army_units(self) -> Units:
        candidates = self._merge_units(
            self.roles.free_units,
            self.roles.attacking_units,
        )
        return candidates.filter(
            lambda unit: self.unit_values.should_attack(unit)
        )

    def _merge_units(self, first: Units, second: Units) -> Units:
        merged = Units([], self.ai)
        seen = set()
        for unit in list(first) + list(second):
            if unit.tag not in seen:
                merged.append(unit)
                seen.add(unit.tag)
        return merged

    def _all_current_group_units(self) -> Units:
        merged = Units([], self.ai)
        seen = set()
        for units in self._current_groups.values():
            for unit in units:
                if unit.tag not in seen:
                    merged.append(unit)
                    seen.add(unit.tag)
        return merged

    @staticmethod
    def _task_for_mode(movement_mode: str) -> UnitTask:
        if movement_mode in MOVING_MODES:
            return UnitTask.Moving
        return UnitTask.Attacking

    def _resolve_search_target(
        self,
        group_id: str,
        units: Units,
        requested_zone_id: str,
    ) -> Optional[Point2]:
        center = self._group_center(units)
        enemy_structures = getattr(self.ai, "enemy_structures", None)
        if enemy_structures and enemy_structures.exists:
            self._search_target_zone_ids.pop(group_id, None)
            return min(
                enemy_structures,
                key=lambda structure: structure.distance_to(center),
            ).position

        zones = list(self.knowledge.zone_manager.expansion_zones)
        if not zones:
            return None
        now = float(getattr(self.ai, "time", 0.0))
        searched = self._searched_zone_ids.setdefault(group_id, set())
        just_reached_zone_id = None
        for index, zone in enumerate(zones):
            zone_id = f"zone_{index}"
            if center.distance_to(zone.center_location) <= 8:
                just_reached_zone_id = zone_id
                searched.add(zone_id)
                self._search_zone_assigned_at[zone_id] = now
                break

        requested_zone = self._zone_by_id(zones, requested_zone_id)
        if (
            not self._search_zone_assigned_at
            and requested_zone is not None
            and requested_zone_id != just_reached_zone_id
            and not any(
                self._zone_has_enemy_presence(zone) for zone in zones
            )
            and (
                not getattr(requested_zone, "is_ours", False)
                or self._zone_has_enemy_presence(requested_zone)
            )
        ):
            self._assign_search_zone(group_id, requested_zone_id, now)
            return requested_zone.center_location

        if (
            requested_zone is not None
            and getattr(requested_zone, "is_ours", False)
            and not self._zone_has_enemy_presence(requested_zone)
        ):
            searched.add(requested_zone_id)
            self._search_zone_assigned_at[requested_zone_id] = now

        candidates = [
            (index, zone)
            for index, zone in enumerate(zones)
            if f"zone_{index}" != just_reached_zone_id
            and (
                not getattr(zone, "is_ours", False)
                or self._zone_has_enemy_presence(zone)
            )
        ]
        if not candidates:
            return None

        enemy_presence = [
            item
            for item in candidates
            if self._zone_has_enemy_presence(item[1])
        ]
        recent_cutoff = now - self.SEARCH_ZONE_REVISIT_SECONDS
        not_recently_assigned = [
            item
            for item in candidates
            if self._search_zone_assigned_at.get(
                f"zone_{item[0]}", float("-inf")
            )
            < recent_cutoff
        ]
        if enemy_presence:
            zone_index, zone = min(
                enemy_presence,
                key=lambda item: center.distance_to(
                    item[1].center_location
                ),
            )
        elif not_recently_assigned:
            zone_index, zone = min(
                not_recently_assigned,
                key=lambda item: (
                    float(
                        getattr(item[1], "last_scouted_center", -1.0)
                    ),
                    center.distance_to(item[1].center_location),
                ),
            )
        else:
            zone_index, zone = min(
                candidates,
                key=lambda item: (
                    self._search_zone_assigned_at.get(
                        f"zone_{item[0]}", float("-inf")
                    ),
                    center.distance_to(item[1].center_location),
                ),
            )
        zone_id = f"zone_{zone_index}"
        self._assign_search_zone(group_id, zone_id, now)
        return zone.center_location

    def _assign_search_zone(
        self,
        group_id: str,
        zone_id: str,
        assigned_at: float,
    ) -> None:
        self._search_target_zone_ids[group_id] = zone_id
        self._search_zone_assigned_at[zone_id] = assigned_at

    def _clear_search_state(self, group_id: str) -> None:
        self._search_target_zone_ids.pop(group_id, None)
        self._searched_zone_ids.pop(group_id, None)

    def _clear_all_search_state(self) -> None:
        self._search_target_zone_ids.clear()
        self._searched_zone_ids.clear()
        self._search_zone_assigned_at.clear()

    @staticmethod
    def _zone_by_id(zones: list, zone_id: str):
        if not str(zone_id).startswith("zone_"):
            return None
        try:
            index = int(str(zone_id).removeprefix("zone_"))
        except ValueError:
            return None
        return zones[index] if 0 <= index < len(zones) else None

    @staticmethod
    def _zone_power(zone, attribute: str) -> float:
        return float(getattr(getattr(zone, attribute, None), "power", 0.0))

    def _zone_has_enemy_presence(self, zone) -> bool:
        return (
            bool(getattr(zone, "is_enemys", False))
            or bool(getattr(zone, "known_enemy_units", []))
            or self._zone_power(zone, "known_enemy_power") > 0
            or self._zone_power(zone, "enemy_static_power") > 0
            or bool(getattr(zone, "is_under_attack", False))
        )
    def _resolve_target(
        self,
        destination_zone_id: str,
        *,
        enter_zone: bool = False,
    ) -> Optional[Point2]:
        zones = list(self.knowledge.zone_manager.expansion_zones)
        if not destination_zone_id.startswith("zone_"):
            return None
        try:
            zone_index = int(destination_zone_id.removeprefix("zone_"))
        except ValueError:
            return None
        if not 0 <= zone_index < len(zones):
            return None
        zone = zones[zone_index]
        if enter_zone:
            return zone.center_location
        if getattr(zone, "is_ours", False):
            return self._safe_own_base_target(zone)
        if not getattr(zone, "is_enemys", False):
            return self._safe_expansion_target(zone)
        return zone.center_location

    def _resolve_regroup_target(
        self,
        destination_zone_id: str,
        units: Units,
    ) -> Optional[Point2]:
        zones = list(self.knowledge.zone_manager.expansion_zones)
        zone = self._zone_by_id(zones, destination_zone_id)
        if zone is None or getattr(zone, "is_ours", False):
            return self._resolve_target(destination_zone_id)
        if getattr(zone, "is_enemys", False):
            return self._resolve_target(destination_zone_id)

        expanding_to = getattr(self.gather_manager, "expanding_to", None)
        if (
            expanding_to is None
            or expanding_to.distance_to(zone.center_location) >= 1
        ):
            return self._safe_expansion_target(zone)

        own_zones = [
            candidate
            for candidate in zones
            if getattr(candidate, "is_ours", False)
            and not getattr(candidate, "is_under_attack", False)
        ]
        if not own_zones:
            return self.ai.start_location
        position = units.center if units.exists else self.ai.start_location
        nearest = min(
            own_zones,
            key=lambda item: position.distance_to(item.center_location),
        )
        return self._safe_own_base_target(nearest)

    def _safe_expansion_target(self, zone) -> Point2:
        center = zone.center_location
        gather_point = getattr(zone, "gather_point", None)
        if gather_point is not None and gather_point.distance_to(center) > 0:
            return self._outside_expansion_footprint(center, gather_point)
        return self._outside_expansion_footprint(
            center, self.ai.start_location
        )
    def _safe_own_base_target(self, zone) -> Point2:
        center = zone.center_location
        gather_point = getattr(zone, "gather_point", None)
        if gather_point is not None and gather_point.distance_to(center) > 0:
            return self._outside_expansion_footprint(center, gather_point)
        return self._outside_expansion_footprint(
            center, self.ai.start_location
        )

    def _outside_expansion_footprint(
        self,
        center: Point2,
        preferred: Point2,
    ) -> Point2:
        if center.distance_to(preferred) > 0:
            return center.towards(preferred, self.EXPANSION_CLEARANCE)
        map_center = self.ai.game_info.map_center
        if center.distance_to(map_center) > 0:
            return center.towards(map_center, self.EXPANSION_CLEARANCE)
        return center
