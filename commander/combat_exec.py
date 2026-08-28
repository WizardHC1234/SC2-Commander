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
from commander.combat_state import _nearest_zone_name, _total_power, _unit_counts
from commander.retreat_policy import (
    ARRIVAL_RADIUS,
    DEFAULT_RETREAT_RATIO,
    LOCAL_BATTLE_RADIUS,
    RECOVER_MARGIN,
    RETREAT_CONFIRM_SECONDS,
    RETREAT_SUPPORT_RADIUS,
    RETREAT_TIME_CAP_SECONDS,
    RETREAT_WATCHED_MODES,
    STATE_ACTIVE,
    STATE_HOLDING,
    STATE_RETREATING,
    GroupRetreatState,
    effective_retreat_ratio,
    retreat_confirmation_ready,
)


MOVE_TYPE_BY_NAME = {
    "Assault": MoveType.Assault,
    "Push": MoveType.Push,
    "Harass": MoveType.Harass,
    "DefensiveRetreat": MoveType.DefensiveRetreat,
    "PanicRetreat": MoveType.PanicRetreat,
    "ReGroup": MoveType.ReGroup,
    "Hold": MoveType.Hold,
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

    def __init__(self):
        super().__init__()
        self.policy_provider = InjectedArmyPolicyProvider()
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
        # Per-group auto-retreat machine states (see retreat_policy.py).
        self._retreat_states: Dict[str, GroupRetreatState] = {}

    async def start(self, knowledge):
        await super().start(knowledge)
        # The unified Observation builder discovers the live Army collector
        # through the bot. This avoids maintaining a second Army snapshot.
        self.ai.llm_army_control_act = self
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
        # Refresh reconnaissance lifecycle before army observation / control
        # so a death/interruption is visible in the same decision cycle.
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
                    retreat_ratio=source.retreat_ratio,
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
                    retreat_ratio=follow.retreat_ratio,
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
            # Endgame sweep: auto-retreat overrides do not apply.
            self._retreat_states.clear()
            self._execute_search_and_destroy(search_command)
            return

        self._clear_all_search_state()
        for command in policy.commands:
            units = self._current_groups.get(command.group_id)
            if units is None or not units.exists:
                continue
            effective = self._advance_retreat_state(command, units)
            if effective.movement_mode == "regroup":
                target = self._resolve_regroup_target(
                    effective.destination_zone_id, units
                )
            elif effective.movement_mode in {"push", "assault", "harass"}:
                target = self._resolve_target(
                    effective.destination_zone_id,
                    enter_zone=True,
                )
            elif effective.movement_mode == "contain":
                target = self._resolve_contain_target(
                    effective.destination_zone_id, units
                )
            else:
                # hold and retreats: settle at the zone's safe point
                # (own/neutral: gather-point side; enemy: zone center).
                target = self._resolve_target(effective.destination_zone_id)
            if target is None:
                continue

            self.roles.set_tasks(
                self._task_for_mode(effective.movement_mode),
                units,
            )
            for unit in units:
                self.combat.add_unit(unit)

            rules = self._configure_rules(effective)
            self.combat.execute(
                target,
                MOVE_TYPE_BY_NAME[effective.move_type],
                rules,
            )
            now = float(getattr(self.ai, "time", 0.0))
            previous = self._active_commands.get(command.group_id) or {}
            same_command = (
                previous.get("destination_zone_id")
                == effective.destination_zone_id
                and previous.get("movement_mode") == effective.movement_mode
            )
            self._active_commands[command.group_id] = {
                "destination_zone_id": effective.destination_zone_id,
                "movement_mode": effective.movement_mode,
                "retreat_ratio": effective.retreat_ratio,
                "issued_at": (
                    float(previous["issued_at"])
                    if same_command and "issued_at" in previous
                    else now
                ),
                "source": (
                    "auto_retreat"
                    if (
                        effective.group_id in self._retreat_states
                        and self._retreat_states[effective.group_id].state
                        != STATE_ACTIVE
                    )
                    else "llm"
                ),
            }

    # ------------------------------------------------------------------
    # auto-retreat state machine (native PlanZoneAttack-style: front-runner
    # local ratio, arrival-stop, hysteresis + time-based recovery). The
    # threshold travels with each move_group command (retreat_ratio).
    # ------------------------------------------------------------------

    def _advance_retreat_state(
        self,
        command: ArmyGroupCommand,
        units: Units,
    ) -> ArmyGroupCommand:
        """Advance one group's retreat machine and return the command to run."""
        group_id = command.group_id
        if command.movement_mode not in RETREAT_WATCHED_MODES:
            self._retreat_states.pop(group_id, None)
            return command

        state = self._retreat_states.setdefault(group_id, GroupRetreatState())
        now = float(getattr(self.ai, "time", 0.0))
        command_signature = (
            f"{command.movement_mode}:{command.destination_zone_id}:"
            f"{command.retreat_ratio}"
        )
        if state.observed_command_signature != command_signature:
            state.observed_command_signature = command_signature
            state.below_threshold_since = None

        # The model issued a different command than the one we interrupted:
        # respect the new intent and re-evaluate from scratch.
        if (
            state.state != STATE_ACTIVE
            and state.original_command is not None
            and (
                command.movement_mode != state.original_command.movement_mode
                or command.destination_zone_id
                != state.original_command.destination_zone_id
            )
        ):
            state.state = STATE_ACTIVE
            state.original_command = None
            state.below_threshold_since = None

        retreat_ratio = (
            command.retreat_ratio
            if command.retreat_ratio is not None
            else DEFAULT_RETREAT_RATIO
        )
        recover_ratio = retreat_ratio + RECOVER_MARGIN
        center = self._group_center(units)
        assessment = self._battle_power_assessment(units, center)
        local_ratio = effective_retreat_ratio(
            group_ratio=assessment["group_ratio"],
            support_ratio=assessment["support_ratio"],
            mission_ratio=assessment["mission_ratio"],
            group_power_share=assessment["group_power_share"],
        )
        state.group_ratio = assessment["group_ratio"]
        state.support_ratio = assessment["support_ratio"]
        state.mission_ratio = assessment["mission_ratio"]
        state.effective_ratio = local_ratio
        state.group_power_share = assessment["group_power_share"]
        state.support_power = assessment["support_power"]
        state.enemy_power = assessment["enemy_power"]

        if state.state in (STATE_RETREATING, STATE_HOLDING):
            timed_out = now - state.since >= RETREAT_TIME_CAP_SECONDS
            if local_ratio >= recover_ratio or timed_out:
                state.state = STATE_ACTIVE
                state.original_command = None
                state.detail = ""
                state.below_threshold_since = None
            else:
                if state.state == STATE_RETREATING and self._arrived_at(
                    center, state.retreat_zone_id
                ):
                    state.state = STATE_HOLDING
                    state.detail = (
                        f"holding at {state.retreat_zone_id} after retreat "
                        f"(local_ratio {local_ratio:.2f} < recover "
                        f"{recover_ratio:.2f})"
                    )
                return self._rewritten_command(command, state)

        if local_ratio < retreat_ratio:
            if state.below_threshold_since is None:
                state.below_threshold_since = now
            if not retreat_confirmation_ready(
                now=now,
                below_threshold_since=state.below_threshold_since,
                effective_ratio_value=local_ratio,
            ):
                state.detail = (
                    f"monitoring low effective_ratio {local_ratio:.2f} < retreat "
                    f"{retreat_ratio:.2f} for {RETREAT_CONFIRM_SECONDS:.1f}s"
                )
                return command
            if state.state != STATE_RETREATING:
                retreat_zone_id = (
                    state.retreat_zone_id
                    if state.state != STATE_ACTIVE
                    else None
                ) or self._auto_retreat_zone_id(center)
                self._enter_state(
                    state,
                    STATE_RETREATING,
                    command,
                    now,
                    retreat_zone_id=retreat_zone_id,
                    detail=(
                        f"effective_ratio {local_ratio:.2f} < retreat "
                        f"{retreat_ratio:.2f}; group={assessment['group_ratio']:.2f}, "
                        f"support={assessment['support_ratio']:.2f}, "
                        f"mission={assessment['mission_ratio']:.2f}, "
                        f"share={assessment['group_power_share']:.2f}; "
                        f"retreating to {retreat_zone_id}"
                    ),
                )
            return self._rewritten_command(command, state)

        state.below_threshold_since = None
        state.detail = ""
        return command

    def _enter_state(
        self,
        state: GroupRetreatState,
        new_state: str,
        command: ArmyGroupCommand,
        now: float,
        *,
        retreat_zone_id: Optional[str] = None,
        detail: str = "",
    ) -> None:
        # Called only on an actual transition; proactively wake the Commander
        # so it can react to the override (e.g. adjust orders or thresholds).
        state.state = new_state
        state.original_command = command
        state.since = now
        state.detail = detail
        if retreat_zone_id is not None:
            state.retreat_zone_id = retreat_zone_id
        self.ai.commander_retreat_wake_pending = {
            "reason": "auto_retreat_triggered",
            "state": new_state,
            "detail": detail,
        }

    def _rewritten_command(
        self,
        command: ArmyGroupCommand,
        state: GroupRetreatState,
    ) -> ArmyGroupCommand:
        if state.state == STATE_RETREATING and state.retreat_zone_id:
            return ArmyGroupCommand(
                group_id=command.group_id,
                destination_zone_id=state.retreat_zone_id,
                movement_mode="defensive_retreat",
                move_type=MOVE_TYPE_BY_MOVEMENT_MODE["defensive_retreat"],
                retreat_ratio=command.retreat_ratio,
            )
        if state.state == STATE_HOLDING and state.retreat_zone_id:
            return ArmyGroupCommand(
                group_id=command.group_id,
                destination_zone_id=state.retreat_zone_id,
                movement_mode="hold",
                move_type=MOVE_TYPE_BY_MOVEMENT_MODE["hold"],
                retreat_ratio=command.retreat_ratio,
            )
        return command

    def _front_runner_position(self, units: Units, center: Point2) -> Point2:
        """Front runner = unit closest to the enemy nearest the group."""
        enemies = getattr(self.ai, "all_enemy_units", None)
        if not enemies:
            return center
        try:
            nearest_enemy = enemies.closest_to(center)
            return units.closest_to(nearest_enemy).position
        except (AttributeError, ValueError):
            return center

    def _battle_power_assessment(self, units: Units, center: Point2) -> dict:
        enemies = getattr(self.ai, "all_enemy_units", None)
        if not enemies:
            group_power = _total_power(self, units)
            return {
                "group_ratio": float("inf"),
                "support_ratio": float("inf"),
                "mission_ratio": float("inf"),
                "group_power_share": 1.0,
                "group_power": group_power,
                "support_power": 0.0,
                "mission_power": group_power,
                "enemy_power": 0.0,
            }
        front = self._front_runner_position(units, center)
        nearby = enemies.closer_than(LOCAL_BATTLE_RADIUS, front).filter(
            lambda unit: not getattr(unit, "is_structure", False)
            and not self.unit_values.is_worker(unit)
        )
        enemy_power = _total_power(self, nearby)
        own_tags = {unit.tag for unit in units}
        mission_units_by_tag = {
            unit.tag: unit
            for group in self._current_groups.values()
            for unit in group
        }
        support_units = Units(
            [
                unit
                for tag, unit in mission_units_by_tag.items()
                if tag not in own_tags
                and unit.distance_to(front) <= RETREAT_SUPPORT_RADIUS
            ],
            self.ai,
        )
        own_power = _total_power(self, units)
        support_power = _total_power(self, support_units)
        mission_units = Units(list(mission_units_by_tag.values()), self.ai)
        mission_power = _total_power(self, mission_units)
        group_power_share = own_power / mission_power if mission_power > 0 else 1.0
        if enemy_power <= 0:
            group_ratio = support_ratio = mission_ratio = float("inf")
        else:
            group_ratio = own_power / enemy_power if own_power > 0 else 0.0
            support_ratio = (
                (own_power + support_power) / enemy_power
                if own_power + support_power > 0
                else 0.0
            )
            mission_ratio = (
                mission_power / enemy_power if mission_power > 0 else 0.0
            )
        return {
            "group_ratio": group_ratio,
            "support_ratio": support_ratio,
            "mission_ratio": mission_ratio,
            "group_power_share": group_power_share,
            "group_power": own_power,
            "support_power": support_power,
            "mission_power": mission_power,
            "enemy_power": enemy_power,
        }

    def _local_battle_ratio(self, units: Units, center: Point2) -> float:
        """Compatibility helper returning the effective retreat-gate ratio."""
        assessment = self._battle_power_assessment(units, center)
        return effective_retreat_ratio(
            group_ratio=assessment["group_ratio"],
            support_ratio=assessment["support_ratio"],
            mission_ratio=assessment["mission_ratio"],
            group_power_share=assessment["group_power_share"],
        )

    def _arrived_at(self, center: Point2, zone_id: Optional[str]) -> bool:
        if not zone_id:
            return False
        zones = list(self.knowledge.zone_manager.expansion_zones)
        zone = self._zone_by_id(zones, zone_id)
        if zone is None:
            return False
        return center.distance_to(zone.center_location) <= ARRIVAL_RADIUS

    def _auto_retreat_zone_id(self, center: Point2) -> str:
        """Nearest safe own zone; falls back to the nearest own zone."""
        zones = list(self.knowledge.zone_manager.expansion_zones)
        if not zones:
            return "zone_0"
        own_safe = [
            (index, zone)
            for index, zone in enumerate(zones)
            if getattr(zone, "is_ours", False)
            and not getattr(zone, "is_under_attack", False)
        ]
        pool = own_safe or [
            (index, zone)
            for index, zone in enumerate(zones)
            if getattr(zone, "is_ours", False)
        ]
        if not pool:
            return "zone_0"
        index, _zone = min(
            pool,
            key=lambda item: center.distance_to(item[1].center_location),
        )
        return f"zone_{index}"

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
            previous = self._active_commands.get(group_id) or {}
            same_command = previous.get("movement_mode") == command.movement_mode
            self._active_commands[group_id] = {
                "destination_zone_id": command.destination_zone_id,
                "movement_mode": command.movement_mode,
                "issued_at": (
                    float(previous["issued_at"])
                    if same_command and "issued_at" in previous
                    else now
                ),
                "source": "llm",
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
            group_id: _total_power(self, units)
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
                key=lambda component: (_total_power(self, component), len(component)),
            )
            center = self._group_center(core)
            total_power = powers[group_id]
            core_power = _total_power(self, core)
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
                "nearest_zone_id": _nearest_zone_name(
                    center,
                    list(self.knowledge.zone_manager.expansion_zones),
                ),
                "unit_type_counts": _unit_counts(units),
                "nearby_enemy_count": len(nearby_enemies),
                "nearby_enemy_power": round(
                    _total_power(self, nearby_enemies), 2
                ),
                "nearby_enemy_type_counts": _unit_counts(
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
                    "destination_zone_id": current.get("destination_zone_id"),
                    "movement_mode": current.get("movement_mode"),
                    "retreat_ratio": current.get("retreat_ratio"),
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
                        - float(current.get("issued_at") or 0.0),
                    ),
                    1,
                )
                state["command_source"] = str(
                    current.get("source") or "llm"
                )
            policy_state = self._retreat_states.get(group_id)
            if policy_state is not None:
                def compact_ratio(value: float) -> Optional[float]:
                    return None if value == float("inf") else round(value, 3)

                state["retreat_assessment"] = {
                    "group_ratio": compact_ratio(policy_state.group_ratio),
                    "support_ratio": compact_ratio(policy_state.support_ratio),
                    "mission_ratio": compact_ratio(policy_state.mission_ratio),
                    "effective_ratio": compact_ratio(policy_state.effective_ratio),
                    "group_power_share": round(policy_state.group_power_share, 3),
                    "support_power": round(policy_state.support_power, 3),
                    "enemy_power": round(policy_state.enemy_power, 3),
                }
            if policy_state is not None and policy_state.state != STATE_ACTIVE:
                state["policy_state"] = policy_state.state
                state["policy_detail"] = policy_state.detail
                state["command_source"] = "auto_retreat"
                if policy_state.original_command is not None:
                    state["blocked_mode"] = (
                        policy_state.original_command.movement_mode
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
                key=lambda cluster: (_total_power(self, cluster), len(cluster)),
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
                self._retreat_states.pop(group_id, None)

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
    def _configure_rules(
        self,
        command: ArmyGroupCommand,
    ) -> Optional[MicroRules]:
        if self.micro_rules is None:
            return None
        # LLM regroup must keep destination priority; Sharpy's cohesion
        # regroup would divert groups toward nearby fights / other packs.
        self.micro_rules.regroup = (
            command.movement_mode not in RETREAT_MODES
            and command.movement_mode != "search_and_destroy"
            and command.movement_mode != "regroup"
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

    def _resolve_contain_target(
        self,
        destination_zone_id: str,
        units: Units,
    ) -> Optional[Point2]:
        """Contain = settle outside the target zone's entrance, never inside.

        Uses the zone's gather point (its ground entrance) when available,
        otherwise a point on the own-main side of the zone.
        """
        zones = list(self.knowledge.zone_manager.expansion_zones)
        zone = self._zone_by_id(zones, destination_zone_id)
        if zone is None:
            return None
        center = zone.center_location
        gather_point = getattr(zone, "gather_point", None)
        if gather_point is not None and gather_point.distance_to(center) > 0:
            return self._outside_expansion_footprint(center, gather_point)
        reference = units.center if units.exists else self.ai.start_location
        return self._outside_expansion_footprint(center, reference)

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
