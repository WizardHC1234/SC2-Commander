"""Sharpy acts that execute Commander macro targets and end-game helpers."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from sc2.position import Point2
from sharpy.managers.core.roles import UnitTask
from sharpy.plans import BuildOrder
from sharpy.plans.acts.act_base import ActBase

logger = logging.getLogger("commander.macro_exec")


class ActOngoingMacroTasks(ActBase):
    """Each frame: instantiate/execute Sharpy acts for active macro targets."""

    def __init__(self, active_tasks_ref: List[Dict[str, Any]]):
        super().__init__()
        self.active_tasks = active_tasks_ref

    async def execute(self) -> bool:
        for task in self.active_tasks:
            if task.get("_disabled", False):
                continue

            action_key = task.get("action")
            try:
                to_count = int(task.get("to_count", 1))
            except (TypeError, ValueError):
                logger.warning("Invalid to_count in task %s; disabling.", task)
                task["_disabled"] = True
                task["_error"] = "invalid_to_count"
                continue

            if not action_key:
                task["_disabled"] = True
                task["_error"] = "missing_action"
                continue

            act: Optional[ActBase] = task.get("_act")
            if act is None:
                get_action_fn = task.get("_get_action_fn")
                if get_action_fn is None:
                    task["_disabled"] = True
                    task["_error"] = "no_action_resolver"
                    continue
                try:
                    act = get_action_fn(action_key, to_count)
                except Exception as exc:
                    logger.warning(
                        "Failed to instantiate act for action=%s to_count=%s: %s",
                        action_key,
                        to_count,
                        exc,
                    )
                    task["_disabled"] = True
                    task["_error"] = f"instantiate_failed: {exc}"
                    continue
                task["_act"] = act
                task["_started"] = False

            if not task.get("_started", False):
                try:
                    await self.start_component(act, self.knowledge)
                except Exception as exc:
                    logger.warning("Failed to start act for action=%s: %s", action_key, exc)
                    task["_disabled"] = True
                    task["_error"] = f"start_failed: {exc}"
                    continue
                task["_started"] = True

            try:
                task["_completed"] = bool(await act.execute())
                task.pop("_execution_error", None)
            except Exception as exc:
                logger.warning("Act execute failed for action=%s: %s", action_key, exc)
                task["_completed"] = False
                task["_execution_error"] = repr(exc)

        return True


class EmptyTactics(BuildOrder):
    """Minimal race-agnostic fallback when a race has no tactics adapter."""

    def __init__(self):
        from sharpy.plans.tactics import (
            DistributeWorkers,
            PlanFinishEnemy,
            PlanZoneAttack,
            PlanZoneDefense,
        )

        super().__init__(
            [
                PlanZoneDefense(),
                DistributeWorkers(),
                PlanZoneAttack(40),
                PlanFinishEnemy(),
            ]
        )


class ForceFinishEnemyOnGG(ActBase):
    """Force combat units to finish after enemy chat surrender."""

    def __init__(
        self,
        should_force_finish: Callable[[Any], bool],
        sweep_interval_seconds: float = 8,
    ):
        super().__init__()
        self.should_force_finish = should_force_finish
        self.sweep_interval_seconds = sweep_interval_seconds
        self._sweep_targets: List[Point2] = []
        self._sweep_index: int = 0
        self._last_sweep_switch_time: Optional[float] = None

    async def execute(self) -> bool:
        if not self.should_force_finish(self.ai):
            return True

        target = self._find_attack_target()
        if target is None:
            return True

        attackers = [
            unit for unit in self.ai.units if self.unit_values.should_attack(unit)
        ]
        self._clear_pending_actions_for_units(attackers)
        for unit in attackers:
            if self.unit_values.should_attack(unit):
                unit.attack(target)
                self.roles.set_task(UnitTask.Attacking, unit)
        return True

    def _clear_pending_actions_for_units(self, units) -> None:
        tags = {unit.tag for unit in units}
        if not tags:
            return
        actions = getattr(self.ai, "actions", None)
        if actions is not None:
            self.ai.actions[:] = [
                action
                for action in actions
                if getattr(getattr(action, "unit", None), "tag", None) not in tags
            ]
        received_tags = getattr(self.ai, "unit_tags_received_action", None)
        if received_tags is not None:
            received_tags.difference_update(tags)

    def _find_attack_target(self):
        own_main = self.zone_manager.own_main_zone.center_location
        enemy_structures = self.ai.enemy_structures
        if enemy_structures.exists:
            return enemy_structures.closest_to(own_main).position
        return self._next_sweep_target(own_main)

    def _next_sweep_target(self, own_main: Point2):
        targets = self._get_sweep_targets(own_main)
        if not targets:
            return None
        now = float(getattr(self.ai, "time", 0.0) or 0.0)
        if self._last_sweep_switch_time is None:
            self._last_sweep_switch_time = now
            return targets[self._sweep_index]
        if now - self._last_sweep_switch_time >= self.sweep_interval_seconds:
            self._sweep_index = (self._sweep_index + 1) % len(targets)
            self._last_sweep_switch_time = now
        return targets[self._sweep_index]

    def _get_sweep_targets(self, own_main: Point2) -> List[Point2]:
        raw_targets = []
        enemy_start = getattr(self.zone_manager, "enemy_start_location", None)
        if enemy_start is not None:
            raw_targets.append(enemy_start)
        expansion_locations = getattr(self.ai, "expansion_locations_list", None)
        if expansion_locations:
            raw_targets.extend(expansion_locations)
        elif self.zone_manager.expansion_zones:
            raw_targets.extend(
                zone.center_location for zone in self.zone_manager.expansion_zones
            )
        unique_targets = []
        seen = set()
        for target in raw_targets:
            key = (round(target.x, 1), round(target.y, 1))
            if key in seen:
                continue
            seen.add(key)
            unique_targets.append(target)
        self._sweep_targets = unique_targets
        if self._sweep_index >= len(unique_targets):
            self._sweep_index = 0
        return unique_targets
