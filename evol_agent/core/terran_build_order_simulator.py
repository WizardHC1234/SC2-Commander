"""Deterministic Terran opening feasibility simulation for EvolAgent.

The simulator intentionally stops at the first planned commitment package.  It
does not model movement, combat, model latency, or empirical execution delay.
Its purpose is narrower: given a structured package extracted from strategy.md,
calculate when that package can first exist under SC2 resource, prerequisite,
supply, builder, add-on, and production-queue constraints.

Mining uses the same approximation as Sharpy's IncomeCalculator:

* one mineral per second for each ideally saturated mineral worker;
* half efficiency beyond 16 mineral workers per completed base;
* ``0.9433962264`` gas per second for each worker on a completed Refinery.

This makes the result a deterministic feasibility estimate aligned with the
runtime's own economy observations, rather than an LLM arithmetic estimate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import heapq
import math
import re
from typing import Any, Iterable


MINERAL_RATE_PER_IDEAL_WORKER = 1.0
MINERAL_RATE_PER_OVERSATURATED_WORKER = 0.5
GAS_RATE_PER_WORKER = 0.9433962264
MINERAL_WORKERS_PER_BASE = 16
GAS_WORKERS_PER_REFINERY = 3
# This project runs the legacy 8-worker SC2 start. Sharpy queues the first SCV
# before the first logged observation, so logs begin at 0 minerals and 8/13.
INITIAL_WORKERS = 8
INITIAL_MINERALS = 50.0
INITIAL_GAS = 0.0
INITIAL_SUPPLY_USED = 8.0
INITIAL_SUPPLY_CAP = 13.0
SUPPLY_PER_DEPOT = 8.0
SUPPLY_PER_COMMAND_CENTER = 13.0
DEFAULT_TIME_LIMIT_SECONDS = 1800.0
EPSILON = 1e-7
SCHEDULING_POLICIES: tuple[tuple[str, bool], ...] = (
    ("balanced", False),
    ("economy_first", True),
    ("infrastructure_first", True),
    ("army_first", True),
)


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


@dataclass(frozen=True)
class SimAction:
    name: str
    description: str
    action_type: str
    minerals: float
    gas: float
    supply: float
    duration: float
    production_location: str
    prerequisites: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()


@dataclass
class Goal:
    action: str
    target: int
    max_parallel: int
    gate: bool = False
    explicit: bool = False


@dataclass(order=True)
class CompletionEvent:
    finish_time: float
    serial: int
    action: str = field(compare=False)
    queue_key: str | None = field(compare=False, default=None)
    builder: bool = field(compare=False, default=False)


@dataclass
class SimState:
    time: float = 0.0
    minerals: float = INITIAL_MINERALS
    gas: float = INITIAL_GAS
    workers: int = INITIAL_WORKERS
    building_workers: int = 0
    bases: int = 1
    refineries: int = 0
    supply_used: float = INITIAL_SUPPLY_USED
    supply_cap: float = INITIAL_SUPPLY_CAP
    completed: Counter[str] = field(default_factory=Counter)
    in_progress: Counter[str] = field(default_factory=Counter)
    busy_queues: Counter[str] = field(default_factory=Counter)
    events: list[CompletionEvent] = field(default_factory=list)
    serial: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


def _catalog_from_runtime(
    knowledge_facts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, SimAction]:
    """Load authoritative action metadata without modifying Commander."""
    catalog: dict[str, SimAction] = {}
    try:
        from commander.races.terran.actions import ACTION_SPECS

        for name, spec in ACTION_SPECS.items():
            if not getattr(spec, "is_macro", False):
                continue
            catalog[name] = SimAction(
                name=name,
                description=str(spec.description or ""),
                action_type=str(spec.action_type),
                minerals=float(spec.minerals or 0),
                gas=float(spec.vespene or 0),
                supply=float(spec.supply or 0),
                duration=float(spec.base_time_seconds or 0),
                production_location=str(spec.production_location or ""),
                prerequisites=tuple(spec.prerequisites or ()),
                dependencies=tuple(spec.dependencies or ()),
            )
    except Exception:
        pass

    # Verified knowledge can supplement the runtime catalog in isolated tests or
    # old checkpoints, but the runtime catalog remains the single source for
    # action type when it is available.
    for row in (knowledge_facts or {}).values():
        name = str(row.get("action") or "").strip()
        if not name or name in catalog:
            continue
        inferred_type = "unit" if name.startswith("train_") else "tech" if name.startswith("research_") else "building"
        catalog[name] = SimAction(
            name=name,
            description=str(row.get("description") or ""),
            action_type=inferred_type,
            minerals=float(row.get("minerals") or 0),
            gas=float(row.get("gas") or 0),
            supply=float(row.get("supply") or 0),
            duration=float(row.get("base_time_seconds") or 0),
            production_location=str(row.get("production_location") or ""),
            prerequisites=tuple(row.get("prerequisites") or ()),
            dependencies=tuple(row.get("dependencies") or ()),
        )
    return catalog


def _family_from_location(location: str) -> str:
    key = _norm(location)
    aliases = {
        "commandcenter": "command_center",
        "orbitalcommand": "command_center",
        "barracks": "barracks",
        "factory": "factory",
        "starport": "starport",
        "barrackstechlab": "barracks_techlab",
        "factorytechlab": "factory_techlab",
        "starporttechlab": "starport_techlab",
        "engineeringbay": "engineering_bay",
        "armory": "armory",
        "fusioncore": "fusion_core",
        "ghostacademy": "ghost_academy",
    }
    return aliases.get(key, re.sub(r"[^a-z0-9]+", "_", location.casefold()).strip("_"))


STRUCTURE_ACTION_BY_FAMILY = {
    "barracks": "build_barracks",
    "factory": "build_factory",
    "starport": "build_starport",
    "engineering_bay": "build_engineering_bay",
    "armory": "build_armory",
    "fusion_core": "build_fusion_core",
    "ghost_academy": "build_ghost_academy",
}

ADDON_ACTIONS = {
    "build_barracks_techlab": ("barracks", "techlab"),
    "build_barracks_reactor": ("barracks", "reactor"),
    "build_factory_techlab": ("factory", "techlab"),
    "build_factory_reactor": ("factory", "reactor"),
    "build_starport_techlab": ("starport", "techlab"),
    "build_starport_reactor": ("starport", "reactor"),
}

FAMILY_BUILD_ACTION = {
    "barracks": "build_barracks",
    "factory": "build_factory",
    "starport": "build_starport",
}


def _is_builder_action(action: SimAction) -> bool:
    return _norm(action.production_location) == "scv"


def _requires_techlab(action: SimAction) -> bool:
    return any("techlab" in _norm(item) for item in action.prerequisites) or any(
        "techlab" in _norm(item) for item in action.dependencies
    )


def _structural_dependencies(action: SimAction) -> list[str]:
    deps: list[str] = []
    for dep in action.dependencies:
        # A starting Command Center and the initial SCVs already satisfy these runtime
        # convenience dependencies.  Expansion remains a dependency only when
        # the strategy explicitly requests more than one base.
        if dep in {"train_scv", "expand"}:
            continue
        deps.append(dep)
    if action.gas > 0 and action.name != "build_gas":
        deps.append("build_gas")
    return list(dict.fromkeys(deps))


class TerranBuildOrderSimulator:
    """Resource-aware first-package simulator using deterministic ASAP scheduling."""

    def __init__(
        self,
        package: dict[str, Any],
        *,
        knowledge_facts: dict[str, dict[str, Any]] | None = None,
        time_limit_seconds: float = DEFAULT_TIME_LIMIT_SECONDS,
        scheduling_policy: str = "balanced",
        reserve_for_priority: bool = False,
    ) -> None:
        self.package = package if isinstance(package, dict) else {}
        self.catalog = _catalog_from_runtime(knowledge_facts)
        self.time_limit = float(time_limit_seconds)
        self.scheduling_policy = scheduling_policy
        self.reserve_for_priority = bool(reserve_for_priority)
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.goals: dict[str, Goal] = {}
        self.gate_actions: set[str] = set()
        self._prepare_goals()

    def _resolve_action_name(self, requested: str) -> str:
        if requested in self.catalog:
            return requested
        wanted = _norm(requested)
        wanted_target = re.sub(r"^(?:train|build|research|morph)", "", wanted)
        candidates: list[str] = []
        for name, action in self.catalog.items():
            normalized = _norm(name)
            target = re.sub(r"^(?:train|build|research|morph)", "", normalized)
            description = _norm(action.description)
            if wanted == normalized or (
                wanted_target and (wanted_target == target or wanted_target in description)
            ):
                candidates.append(name)
        unique = list(dict.fromkeys(candidates))
        return unique[0] if len(unique) == 1 else requested

    def _merge_goal(
        self,
        action: str,
        target: int,
        max_parallel: int,
        *,
        gate: bool = False,
        explicit: bool = False,
    ) -> None:
        if action not in self.catalog:
            self.errors.append(f"unknown action: {action}")
            return
        previous = self.goals.get(action)
        if previous is None:
            self.goals[action] = Goal(
                action=action,
                target=max(1, target),
                max_parallel=max(1, max_parallel),
                gate=gate,
                explicit=explicit,
            )
        else:
            previous.target = max(previous.target, target)
            previous.max_parallel = max(previous.max_parallel, max_parallel)
            previous.gate = previous.gate or gate
            previous.explicit = previous.explicit or explicit
        if gate:
            self.gate_actions.add(action)

    def _prepare_goals(self) -> None:
        for index, item in enumerate(self.package.get("gate_components") or [], 1):
            if not isinstance(item, dict):
                self.errors.append(f"gate component {index} is not an object")
                continue
            action = self._resolve_action_name(str(item.get("action") or "").strip())
            quantity = _positive_int(item.get("quantity"))
            slots = _positive_int(item.get("production_slots"))
            if not action or not quantity or not slots:
                self.errors.append(
                    f"gate component {action or index} lacks action, quantity, or production_slots"
                )
                continue
            self._merge_goal(action, quantity, slots, gate=True, explicit=True)

        for index, item in enumerate(self.package.get("setup_actions") or [], 1):
            if not isinstance(item, dict):
                self.errors.append(f"setup action {index} is not an object")
                continue
            action = self._resolve_action_name(str(item.get("action") or "").strip())
            quantity = _positive_int(item.get("quantity"))
            slots = _positive_int(item.get("parallel_slots"))
            if not action or not quantity or not slots:
                self.errors.append(
                    f"setup action {action or index} lacks action, quantity, or parallel_slots"
                )
                continue
            self._merge_goal(action, quantity, slots, explicit=True)

        economy = self.package.get("economy") if isinstance(self.package.get("economy"), dict) else {}
        worker_target = _positive_int(economy.get("worker_target_before_commitment"), INITIAL_WORKERS)
        base_target = _positive_int(economy.get("base_target_before_commitment"), 1)
        self.worker_target = max(INITIAL_WORKERS, worker_target)
        self.base_target = max(1, base_target)
        if self.worker_target == INITIAL_WORKERS and not economy.get("worker_target_before_commitment"):
            self.warnings.append(
                f"worker target missing; simulation keeps the initial {INITIAL_WORKERS} SCVs"
            )
        self._merge_goal("train_scv", self.worker_target, self.base_target)
        if self.base_target > 1:
            self._merge_goal("expand", self.base_target, max(1, self.base_target - 1), explicit=True)

        # Complete the structural dependency graph using authoritative metadata.
        pending = list(self.goals)
        seen: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen or name not in self.catalog:
                continue
            seen.add(name)
            for dep in _structural_dependencies(self.catalog[name]):
                if dep not in self.catalog:
                    self.errors.append(f"{name} has unavailable dependency {dep}")
                    continue
                if dep not in self.goals:
                    self._merge_goal(dep, 1, 1)
                pending.append(dep)

        # Explicit production_slots describe queue capacity allocated before the
        # first commitment.  Ensure the corresponding structures/add-ons exist.
        for name in list(self.gate_actions):
            goal = self.goals.get(name)
            action = self.catalog.get(name)
            if goal is None or action is None or action.action_type != "unit":
                continue
            family = _family_from_location(action.production_location)
            if family not in FAMILY_BUILD_ACTION and family != "command_center":
                continue
            if family == "command_center":
                self.base_target = max(self.base_target, goal.max_parallel)
                if self.base_target > 1:
                    self._merge_goal("expand", self.base_target, self.base_target - 1)
                continue
            if _requires_techlab(action):
                addon = f"build_{family}_techlab"
                self._merge_goal(addon, goal.max_parallel, goal.max_parallel)
                self._merge_goal(FAMILY_BUILD_ACTION[family], goal.max_parallel, goal.max_parallel)
            else:
                reactor_target = self.goals.get(f"build_{family}_reactor")
                reactors = reactor_target.target if reactor_target else 0
                producer_target = max(1, goal.max_parallel - reactors)
                addon_total = reactors
                techlab_target = self.goals.get(f"build_{family}_techlab")
                if techlab_target:
                    addon_total += techlab_target.target
                producer_target = max(producer_target, addon_total)
                self._merge_goal(FAMILY_BUILD_ACTION[family], producer_target, producer_target)

        # Re-run closure for producer goals introduced above.
        pending = list(self.goals)
        seen.clear()
        while pending:
            name = pending.pop()
            if name in seen or name not in self.catalog:
                continue
            seen.add(name)
            for dep in _structural_dependencies(self.catalog[name]):
                if dep in self.catalog and dep not in self.goals:
                    self._merge_goal(dep, 1, 1)
                    pending.append(dep)

        refinery_goal = self.goals.get("build_gas")
        explicit_refineries = refinery_goal.target if refinery_goal else 0
        desired_gas_workers = economy.get("gas_workers_before_commitment")
        if desired_gas_workers is None:
            desired_gas_workers = explicit_refineries * GAS_WORKERS_PER_REFINERY
        self.gas_worker_target = max(0, _positive_int(desired_gas_workers, 0))
        if self.gas_worker_target >= self.worker_target:
            self.errors.append(
                "gas worker target must leave at least one pre-commitment worker for minerals"
            )
        required_refineries = int(
            math.ceil(self.gas_worker_target / GAS_WORKERS_PER_REFINERY)
        )
        if required_refineries:
            self._merge_goal(
                "build_gas",
                required_refineries,
                max(1, required_refineries),
            )

        # Add enough depots for the complete gate and pre-commitment economy.
        gate_supply = sum(
            self.goals[name].target * self.catalog[name].supply
            for name in self.gate_actions
            if name in self.catalog
        )
        target_supply = self.worker_target + gate_supply
        if target_supply > 200 + EPSILON:
            self.errors.append(
                f"first-commitment package requires {target_supply:g} supply, exceeding the 200-supply cap"
            )
        base_supply = INITIAL_SUPPLY_CAP + max(0, self.base_target - 1) * SUPPLY_PER_COMMAND_CENTER
        depot_target = int(math.ceil(max(0.0, target_supply - base_supply) / SUPPLY_PER_DEPOT))
        if "build_barracks" in self.goals:
            depot_target = max(1, depot_target)
        if depot_target:
            self._merge_goal("build_supply_depot", depot_target, min(depot_target, 2))

    def _completed_count(self, state: SimState, action: str) -> int:
        if action == "train_scv":
            return state.workers
        if action == "expand":
            return state.bases
        if action == "build_gas":
            return state.refineries
        return int(state.completed[action])

    def _target_met(self, state: SimState, goal: Goal) -> bool:
        return self._completed_count(state, goal.action) >= goal.target

    def _all_targets_met(self, state: SimState) -> bool:
        return bool(self.gate_actions) and all(
            self._target_met(state, goal) for goal in self.goals.values()
        )

    def _income_rates(self, state: SimState) -> tuple[float, float, int, int]:
        available = max(0, state.workers - state.building_workers)
        gas_workers = min(
            self.gas_worker_target,
            state.refineries * GAS_WORKERS_PER_REFINERY,
            max(0, available - 1),
        )
        mineral_workers = max(0, available - gas_workers)
        ideal_slots = state.bases * MINERAL_WORKERS_PER_BASE
        ideal = min(mineral_workers, ideal_slots)
        overflow = max(0, mineral_workers - ideal_slots)
        mineral_rate = (
            ideal * MINERAL_RATE_PER_IDEAL_WORKER
            + overflow * MINERAL_RATE_PER_OVERSATURATED_WORKER
        )
        gas_rate = gas_workers * GAS_RATE_PER_WORKER
        return mineral_rate, gas_rate, mineral_workers, gas_workers

    def _advance(self, state: SimState, new_time: float) -> None:
        new_time = min(float(new_time), self.time_limit)
        if new_time <= state.time + EPSILON:
            return
        mineral_rate, gas_rate, _, _ = self._income_rates(state)
        dt = new_time - state.time
        state.minerals += mineral_rate * dt
        state.gas += gas_rate * dt
        state.time = new_time

    def _process_events(self, state: SimState) -> None:
        while state.events and state.events[0].finish_time <= state.time + EPSILON:
            event = heapq.heappop(state.events)
            state.in_progress[event.action] -= 1
            if state.in_progress[event.action] <= 0:
                del state.in_progress[event.action]
            if event.queue_key:
                state.busy_queues[event.queue_key] -= 1
                if state.busy_queues[event.queue_key] <= 0:
                    del state.busy_queues[event.queue_key]
            if event.builder:
                state.building_workers = max(0, state.building_workers - 1)

            state.completed[event.action] += 1
            if event.action == "train_scv":
                state.workers += 1
            elif event.action == "expand":
                state.bases += 1
                state.supply_cap = min(200.0, state.supply_cap + SUPPLY_PER_COMMAND_CENTER)
            elif event.action == "build_gas":
                state.refineries += 1
            elif event.action == "build_supply_depot":
                state.supply_cap = min(200.0, state.supply_cap + SUPPLY_PER_DEPOT)
            state.trace.append(
                {
                    "time": round(state.time, 3),
                    "event": "complete",
                    "action": event.action,
                }
            )

    def _action_dependencies_ready(self, state: SimState, action: SimAction) -> bool:
        return all(
            self._completed_count(state, dep) >= 1
            for dep in _structural_dependencies(action)
        )

    def _family_counts(self, state: SimState, family: str) -> tuple[int, int, int]:
        buildings = self._completed_count(state, FAMILY_BUILD_ACTION[family])
        techlabs = self._completed_count(state, f"build_{family}_techlab")
        reactors = self._completed_count(state, f"build_{family}_reactor")
        return buildings, techlabs, reactors

    def _queue_capacity(self, state: SimState, key: str) -> int:
        if key == "command_center":
            return state.bases
        if key.endswith(":research"):
            location = key.removesuffix(":research")
            if location.endswith("_techlab"):
                family = location.removesuffix("_techlab")
                return self._completed_count(state, f"build_{family}_techlab")
            action = STRUCTURE_ACTION_BY_FAMILY.get(location)
            return self._completed_count(state, action) if action else 0
        match = re.fullmatch(r"(barracks|factory|starport):(plain|techlab|reactor|reactor_extra)", key)
        if match:
            family, subtype = match.groups()
            buildings, techlabs, reactors = self._family_counts(state, family)
            if subtype == "plain":
                return max(0, buildings - techlabs - reactors)
            if subtype == "techlab":
                return techlabs
            return reactors
        return 0

    def _acquire_queue(self, state: SimState, action: SimAction) -> str | None:
        name = action.name
        if name in ADDON_ACTIONS:
            family, _ = ADDON_ACTIONS[name]
            candidates = [f"{family}:plain"]
        elif action.action_type == "tech":
            candidates = [f"{_family_from_location(action.production_location)}:research"]
        elif action.action_type == "unit":
            family = _family_from_location(action.production_location)
            if family == "command_center":
                candidates = ["command_center"]
            elif family in FAMILY_BUILD_ACTION:
                if _requires_techlab(action):
                    candidates = [f"{family}:techlab"]
                else:
                    candidates = [
                        f"{family}:reactor_extra",
                        f"{family}:reactor",
                        f"{family}:plain",
                        f"{family}:techlab",
                    ]
            else:
                candidates = []
        elif name in {"morph_orbital_command", "morph_planetary_fortress"}:
            candidates = ["command_center"]
        else:
            candidates = []

        for key in candidates:
            if state.busy_queues[key] < self._queue_capacity(state, key):
                return key
        return None

    def _free_builder_available(self, state: SimState) -> bool:
        return state.workers - state.building_workers > 0

    def _supply_depot_needed_now(self, state: SimState) -> bool:
        goal = self.goals.get("build_supply_depot")
        if goal is None:
            return False
        started = self._completed_count(state, goal.action) + state.in_progress[goal.action]
        if started >= goal.target:
            return False
        # The first Depot is a structural Barracks prerequisite.
        if "build_barracks" in self.goals and self._completed_count(state, "build_supply_depot") < 1:
            return state.in_progress["build_supply_depot"] < 1

        free_supply = state.supply_cap - state.supply_used
        military_producer_ready = False
        for name in self.gate_actions:
            action = self.catalog.get(name)
            if action is None or action.action_type != "unit":
                continue
            family = _family_from_location(action.production_location)
            if family == "command_center":
                military_producer_ready = True
            elif family in FAMILY_BUILD_ACTION and self._completed_count(
                state, FAMILY_BUILD_ACTION[family]
            ) > 0:
                military_producer_ready = True
        if not military_producer_ready and free_supply > 1 + EPSILON:
            return False

        # Later Depots are built just in time for approximately one concurrent
        # production wave.  Building every eventual Depot at the opening would
        # consume minerals needed for the production structures and would no
        # longer be an earliest-feasible schedule.
        wave_supply = 0.0
        for name in self.gate_actions | {"train_scv"}:
            item = self.goals.get(name)
            action = self.catalog.get(name)
            if item is None or action is None or action.supply <= 0:
                continue
            remaining = max(
                0,
                item.target
                - self._completed_count(state, name)
                - state.in_progress[name],
            )
            wave_supply += min(remaining, item.max_parallel) * action.supply
        reserve = max(2.0, min(SUPPLY_PER_DEPOT, wave_supply))
        if state.in_progress["build_supply_depot"]:
            # One pending Depot normally covers the next production wave.  A
            # second is useful only when the current cap is already exhausted.
            return free_supply <= EPSILON and state.in_progress["build_supply_depot"] < goal.max_parallel
        return free_supply <= reserve + EPSILON

    def _can_start(self, state: SimState, goal: Goal) -> tuple[bool, str | None]:
        action = self.catalog[goal.action]
        total_started = self._completed_count(state, goal.action) + state.in_progress[goal.action]
        if total_started >= goal.target or state.in_progress[goal.action] >= goal.max_parallel:
            return False, None
        if goal.action == "build_supply_depot" and not self._supply_depot_needed_now(state):
            return False, None
        if not self._action_dependencies_ready(state, action):
            return False, None
        if state.minerals + EPSILON < action.minerals or state.gas + EPSILON < action.gas:
            return False, None
        if action.supply > 0 and state.supply_used + action.supply > state.supply_cap + EPSILON:
            return False, None
        if _is_builder_action(action):
            return self._free_builder_available(state), None
        queue_key = self._acquire_queue(state, action)
        return queue_key is not None, queue_key

    def _goal_priority(self, state: SimState, goal: Goal) -> tuple[int, float, str]:
        action = self.catalog[goal.action]
        free_supply = state.supply_cap - state.supply_used
        if goal.action == "build_supply_depot" and free_supply <= 2:
            phase = 0
        elif goal.action == "build_supply_depot" and "build_barracks" in self.goals and self._completed_count(state, "build_supply_depot") < 1:
            phase = 1
        else:
            phases = {
                "balanced": {
                    "gas": 2, "building": 3, "worker": 4,
                    "tech": 5, "gate": 6, "expand": 7,
                },
                "economy_first": {
                    "worker": 2, "expand": 3, "gas": 4,
                    "building": 5, "tech": 6, "gate": 7,
                },
                "infrastructure_first": {
                    "gas": 2, "building": 3, "tech": 4,
                    "worker": 5, "gate": 6, "expand": 7,
                },
                "army_first": {
                    "gas": 2, "building": 3, "tech": 4,
                    "gate": 5, "worker": 6, "expand": 7,
                },
            }.get(self.scheduling_policy)
            if phases is None:
                phases = {
                    "gas": 2, "building": 3, "worker": 4,
                    "tech": 5, "gate": 6, "expand": 7,
                }
            if goal.action == "build_gas":
                phase = phases["gas"]
            elif action.action_type == "building" and goal.action not in {"expand", "morph_orbital_command"}:
                phase = phases["building"]
            elif goal.action == "train_scv":
                phase = phases["worker"]
            elif action.action_type == "tech":
                phase = phases["tech"]
            elif goal.gate:
                phase = phases["gate"]
            elif goal.action == "expand":
                phase = phases["expand"]
            else:
                phase = 8
        remaining = max(0, goal.target - self._completed_count(state, goal.action))
        return phase, -(remaining * action.duration), goal.action

    def _start(self, state: SimState, goal: Goal, queue_key: str | None) -> None:
        action = self.catalog[goal.action]
        state.minerals -= action.minerals
        state.gas -= action.gas
        state.supply_used += action.supply
        builder = _is_builder_action(action)
        if builder:
            state.building_workers += 1
        if queue_key:
            state.busy_queues[queue_key] += 1
        state.in_progress[action.name] += 1
        state.serial += 1
        heapq.heappush(
            state.events,
            CompletionEvent(
                state.time + action.duration,
                state.serial,
                action.name,
                queue_key,
                builder,
            ),
        )
        state.trace.append(
            {
                "time": round(state.time, 3),
                "event": "start",
                "action": action.name,
                "minerals_after": round(state.minerals, 3),
                "gas_after": round(state.gas, 3),
            }
        )

    def _resource_ready_time(self, state: SimState, goal: Goal) -> float | None:
        action = self.catalog[goal.action]
        total_started = self._completed_count(state, goal.action) + state.in_progress[goal.action]
        if total_started >= goal.target or state.in_progress[goal.action] >= goal.max_parallel:
            return None
        if goal.action == "build_supply_depot" and not self._supply_depot_needed_now(state):
            return None
        if not self._action_dependencies_ready(state, action):
            return None
        if action.supply > 0 and state.supply_used + action.supply > state.supply_cap + EPSILON:
            return None
        if _is_builder_action(action):
            if not self._free_builder_available(state):
                return None
        elif self._acquire_queue(state, action) is None:
            return None
        mineral_rate, gas_rate, _, _ = self._income_rates(state)
        mineral_wait = 0.0
        gas_wait = 0.0
        if state.minerals + EPSILON < action.minerals:
            if mineral_rate <= EPSILON:
                return None
            mineral_wait = (action.minerals - state.minerals) / mineral_rate
        if state.gas + EPSILON < action.gas:
            if gas_rate <= EPSILON:
                return None
            gas_wait = (action.gas - state.gas) / gas_rate
        return state.time + max(mineral_wait, gas_wait, 0.0)

    def run(self) -> dict[str, Any]:
        if self.errors or not self.gate_actions:
            if not self.gate_actions:
                self.errors.append("no first-commitment gate components were supplied")
            return self._result(None, None)

        state = SimState()
        state.completed["train_scv"] = INITIAL_WORKERS
        state.completed["expand"] = 1

        iterations = 0
        while state.time < self.time_limit - EPSILON:
            iterations += 1
            if iterations > 200000:
                self.errors.append("simulation iteration limit exceeded")
                break
            self._process_events(state)
            if self._all_targets_met(state):
                return self._result(state, state.time)

            started = False
            ordered_goals = sorted(
                self.goals.values(), key=lambda item: self._goal_priority(state, item)
            )
            for goal in ordered_goals:
                can_start, queue_key = self._can_start(state, goal)
                if can_start:
                    self._start(state, goal, queue_key)
                    started = True
                elif self.reserve_for_priority:
                    # If this action is otherwise startable and only waiting
                    # for resources, reserve for it instead of spending those
                    # resources on a lower-priority goal. Running several
                    # policies avoids baking one such trade-off into the audit.
                    ready = self._resource_ready_time(state, goal)
                    if ready is not None and ready > state.time + EPSILON:
                        break
            if started:
                continue

            next_times = [event.finish_time for event in state.events]
            resource_times = [
                (self._goal_priority(state, goal), ready)
                for goal in ordered_goals
                if (ready := self._resource_ready_time(state, goal)) is not None
                and ready > state.time + EPSILON
            ]
            if self.reserve_for_priority and resource_times:
                next_times.append(min(resource_times, key=lambda item: item[0])[1])
            else:
                next_times.extend(ready for _, ready in resource_times)
            if not next_times:
                unresolved = [
                    f"{goal.action}:{self._completed_count(state, goal.action)}/{goal.target}"
                    for goal in self.goals.values()
                    if not self._target_met(state, goal)
                ]
                self.errors.append(
                    "simulation deadlocked with unresolved targets: " + ", ".join(unresolved)
                )
                break
            next_time = min(next_times)
            self._advance(state, next_time)

        if state.time >= self.time_limit - EPSILON:
            self.errors.append(
                f"first-commitment package is not feasible within {self.time_limit:.0f}s"
            )
        return self._result(state, None)

    def _result(self, state: SimState | None, earliest: float | None) -> dict[str, Any]:
        total_cost = {"minerals": 0.0, "gas": 0.0, "supply": 0.0}
        for goal in self.goals.values():
            action = self.catalog.get(goal.action)
            if action is None:
                continue
            initial = INITIAL_WORKERS if goal.action == "train_scv" else 1 if goal.action == "expand" else 0
            quantity = max(0, goal.target - initial)
            total_cost["minerals"] += quantity * action.minerals
            total_cost["gas"] += quantity * action.gas
            total_cost["supply"] += quantity * action.supply

        bottlenecks: list[str] = []
        if state is not None and state.trace:
            if self.gas_worker_target:
                bottlenecks.append("gas collection and gas-dependent technology")
            if any(name in ADDON_ACTIONS for name in self.goals):
                bottlenecks.append("producer add-on availability")
            if self.goals.get("build_supply_depot"):
                bottlenecks.append("supply availability")

        return {
            "complete": earliest is not None and not self.errors,
            "earliest_feasible_time_seconds": round(earliest, 3) if earliest is not None else None,
            "selected_schedule_policy": self.scheduling_policy,
            "economy": {
                "worker_target_before_commitment": self.worker_target,
                "base_target_before_commitment": self.base_target,
                "gas_workers_before_commitment": self.gas_worker_target,
                "mineral_workers_per_base_at_full_efficiency": MINERAL_WORKERS_PER_BASE,
                "gas_workers_per_refinery": GAS_WORKERS_PER_REFINERY,
                "mineral_rate_per_ideal_worker_per_second": MINERAL_RATE_PER_IDEAL_WORKER,
                "gas_rate_per_worker_per_second": round(GAS_RATE_PER_WORKER, 10),
                "oversaturated_mineral_worker_efficiency": MINERAL_RATE_PER_OVERSATURATED_WORKER,
            },
            "targets": {
                name: {"count": goal.target, "max_parallel": goal.max_parallel, "gate": goal.gate}
                for name, goal in sorted(self.goals.items())
            },
            "total_cost": {key: round(value, 3) for key, value in total_cost.items()},
            "bottlenecks": bottlenecks,
            "assumptions": [
                "project runtime start: 8 SCVs, one Command Center, 50 minerals, and 13 supply",
                "mining follows Sharpy IncomeCalculator saturation rates",
                "gas workers transfer immediately when a Refinery completes",
                "the reported estimate is the fastest feasible result found across deterministic scheduling policies",
                "MULE income, movement, combat, model latency, and army travel are excluded",
            ],
            "warnings": list(dict.fromkeys(self.warnings)),
            "errors": list(dict.fromkeys(self.errors)),
            "trace": state.trace if state is not None else [],
        }


def simulate_terran_first_commitment(
    package: Any,
    *,
    knowledge_facts: dict[str, dict[str, Any]] | None = None,
    time_limit_seconds: float = DEFAULT_TIME_LIMIT_SECONDS,
) -> dict[str, Any]:
    """Calculate a robust resource-feasible first-package timing estimate.

    Build-order scheduling contains genuine economy-versus-tech trade-offs. We
    therefore simulate several deterministic legal priorities and retain the
    fastest completed schedule rather than relying on one fragile greedy order.
    """
    results: list[dict[str, Any]] = []
    normalized_package = package if isinstance(package, dict) else {}
    for policy, reserve in SCHEDULING_POLICIES:
        simulator = TerranBuildOrderSimulator(
            normalized_package,
            knowledge_facts=knowledge_facts,
            time_limit_seconds=time_limit_seconds,
            scheduling_policy=policy,
            reserve_for_priority=reserve,
        )
        results.append(simulator.run())

    completed = [result for result in results if result.get("complete")]
    if completed:
        selected = min(
            completed,
            key=lambda result: float(result["earliest_feasible_time_seconds"]),
        )
    else:
        selected = results[0]
    selected["scheduling_policies_evaluated"] = [
        {
            "policy": result.get("selected_schedule_policy"),
            "complete": bool(result.get("complete")),
            "time_seconds": result.get("earliest_feasible_time_seconds"),
        }
        for result in results
    ]
    return selected


__all__ = [
    "GAS_RATE_PER_WORKER",
    "MINERAL_RATE_PER_IDEAL_WORKER",
    "TerranBuildOrderSimulator",
    "simulate_terran_first_commitment",
]
