"""LLM Observation Recorder.

This manager periodically captures the structured game state (every
``interval_seconds`` of in-game time) and produces a paired English text
summary that is suitable for feeding into an LLM. Snapshots are buffered in memory and written to JSON on game end, process
exit, interrupt signals, and periodic autosaves so force-quit / pause-kill
still leave a usable record.

The recorder is built around four concerns kept strictly decoupled:

1. Timing control - ``last_recorded_time`` plus a step-size check.
2. Modular extractors - one method per data domain returning a plain ``Dict``.
3. Dual-state formatting - a master snapshot dict, plus a text observation
   derived only from that dict.
4. Persistence - buffered writes with best-effort flush on any exit path.
"""

import atexit
import json
import logging
import os
import threading
import weakref
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from sc2.data import Race, Result
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

from sharpy.managers.core.manager_base import ManagerBase

# v3 (2026-05-15): a small denylist of upgrades that have negligible
# tactical/macro impact and therefore add only noise to the LLM prompt. The
# default ``state.upgrades`` set on a real game rarely contains anything
# outside of combat/macro upgrades, but we keep this list as the single
# tuning point for "ignore basic gathering / cosmetic upgrades, if any".
_UPGRADE_BLOCKLIST: Set[UpgradeId] = set()

if TYPE_CHECKING:
    from sharpy.knowledges import Knowledge


DEFAULT_INTERVAL_SECONDS: float = 12.0
DEFAULT_OUTPUT_FOLDER: str = "games"
DEFAULT_FILENAME_PREFIX: str = "Replay"

# Rewrite JSON this often so a hard kill still leaves a recent file.
DEFAULT_AUTOSAVE_EVERY_INTERACTIONS: int = 1
DEFAULT_AUTOSAVE_EVERY_SNAPSHOTS: int = 3

_ACTIVE_RECORDERS = weakref.WeakSet()
_ATEXIT_REGISTERED = False
_FLUSH_LOCK = threading.Lock()


def _flush_all_active_recorders(reason: str = "atexit") -> None:
    """Best-effort flush for every live recorder (process exit / interrupt)."""
    for recorder in list(_ACTIVE_RECORDERS):
        try:
            recorder.flush_to_disk(game_result=None, reason=reason)
        except Exception:
            pass


def _ensure_atexit_hook() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    atexit.register(_flush_all_active_recorders, "atexit")
    _ATEXIT_REGISTERED = True

# Short race tags used inside the auto-generated file name (PvT, ZvP, ...).
_RACE_SHORT: Dict[Race, str] = {
    Race.Protoss: "P",
    Race.Terran: "T",
    Race.Zerg: "Z",
    Race.Random: "R",
}

# Worker types per race, used to scope "workers en route" detection.
_WORKER_TYPES: Dict[Race, UnitTypeId] = {
    Race.Protoss: UnitTypeId.PROBE,
    Race.Terran: UnitTypeId.SCV,
    Race.Zerg: UnitTypeId.DRONE,
}

_NON_ARMY_MOBILE_TYPES: Set[UnitTypeId] = set(_WORKER_TYPES.values()) | {
    UnitTypeId.MULE,
    UnitTypeId.DRONEBURROWED,
}

# Vespene geyser type ids - if a worker is ordered to build on top of one of
# these, the structure foundation does NOT yet exist on the map, so the worker
# is still "en route" rather than actively constructing.
_VESPENE_GEYSER_TYPES: Set[UnitTypeId] = {
    UnitTypeId.VESPENEGEYSER,
    UnitTypeId.RICHVESPENEGEYSER,
    UnitTypeId.PROTOSSVESPENEGEYSER,
    UnitTypeId.PURIFIERVESPENEGEYSER,
    UnitTypeId.SHAKURASVESPENEGEYSER,
}


class LLMObservationRecorder(ManagerBase):
    """Capture, format and persist LLM-friendly game observations.

    The recorder hooks into the regular manager update cycle but only does
    real work every ``interval_seconds`` of in-game time. Each trigger builds
    a master snapshot via the modular ``_extract_*`` methods and derives an
    English text observation from it. Both representations are appended to
    ``record_history`` and flushed to disk via :py:meth:`flush_to_disk`.
    """

    def __init__(
        self,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        output_folder: str = DEFAULT_OUTPUT_FOLDER,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.interval_seconds: float = interval_seconds
        self.output_folder: str = output_folder
        self.enabled: bool = enabled
        # Strategy Coordinator can record LLM calls before manager ``start()``.
        # Keep these present so early autosave does not raise AttributeError.
        self.knowledge = None
        self.ai = None

        # Initialised so that the first capture happens close to t=0 once the
        # game timer crosses ``interval_seconds``.
        self.last_recorded_time: float = -interval_seconds

        # In-memory buffer of all captured snapshots. Flushed to JSON once on
        # ``on_end`` to avoid I/O stalls during the game.
        self.record_history: List[Dict] = []
        self.llm_interactions: List[Dict] = []

        # Optional override: when set, the JSON file is written next to the
        # SC2Replay using exactly the same prefix (replay path with ``.json``
        # extension instead of ``.SC2Replay``). The bot loader fills this in
        # automatically; setting it manually is also supported.
        self.replay_save_path: Optional[str] = None

        # Optional override: an explicit absolute output path for the JSON.
        # Takes precedence over both ``replay_save_path`` and auto-naming.
        self.output_path: Optional[str] = None

        self._autosave_every_interactions = DEFAULT_AUTOSAVE_EVERY_INTERACTIONS
        self._autosave_every_snapshots = DEFAULT_AUTOSAVE_EVERY_SNAPSHOTS
        self._interactions_since_autosave = 0
        self._snapshots_since_autosave = 0
        self._last_flush_reason: Optional[str] = None
        # Set by ``on_end`` (or any flush that passes a real Result). Later
        # best-effort flushes with ``game_result=None`` must not wipe this,
        # otherwise Victory/Defeat becomes Interrupted/Defeat incorrectly.
        self._known_game_result: Optional[Result] = None

        # Cached references resolved during ``start``. They are intentionally
        # optional so the recorder degrades gracefully when a bot does not
        # register every helper manager.
        self._income_calculator = None
        self._enemy_units_manager = None
        self._lost_units_manager = None
        self._game_analyzer = None
        self._memory_manager = None
        self._build_detector = None

        # Lookups populated lazily in ``start`` from python-sc2's auto-generated
        # ability dictionaries. Used to translate worker / building orders into
        # human-readable "what is being built / trained / researched" strings.
        self._build_ability_to_structure: Dict[AbilityId, UnitTypeId] = {}
        self._train_ability_to_unit: Dict[AbilityId, UnitTypeId] = {}
        self._research_ability_to_upgrade: Dict[AbilityId, UpgradeId] = {}

        # Timestamp of the most recent step in which a non-worker enemy army
        # unit was visible. Structures and workers do not refresh army intel.
        self.last_enemy_seen_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Manager lifecycle
    # ------------------------------------------------------------------

    async def start(self, knowledge: "Knowledge"):
        await super().start(knowledge)
        _ACTIVE_RECORDERS.add(self)
        _ensure_atexit_hook()

        from sharpy.interfaces import (
            IIncomeCalculator,
            IEnemyUnitsManager,
            ILostUnitsManager,
            IGameAnalyzer,
            IMemoryManager,
        )
        from sharpy.managers.extensions.build_detector import BuildDetector

        self._income_calculator = knowledge.get_manager(IIncomeCalculator)
        self._enemy_units_manager = knowledge.get_manager(IEnemyUnitsManager)
        self._lost_units_manager = knowledge.get_manager(ILostUnitsManager)
        self._game_analyzer = knowledge.get_manager(IGameAnalyzer)
        self._memory_manager = knowledge.get_manager(IMemoryManager)
        self._build_detector = knowledge.get_manager(BuildDetector)

        self._build_ability_lookups()

    def _build_ability_lookups(self) -> None:
        """Populate the build / train / research ability dictionaries.

        We dynamically derive these from python-sc2's auto-generated
        ``TRAIN_INFO`` and ``RESEARCH_INFO`` so we do not need to hard-code a
        long list (and stay correct as the SC2 data files are regenerated).
        """
        try:
            from sc2.dicts.unit_train_build_abilities import TRAIN_INFO
            from sc2.dicts.unit_research_abilities import RESEARCH_INFO
        except Exception as exc:
            self.print(
                f"LLMObservationRecorder could not load ability dicts: {exc}",
                stats=False,
                log_level=logging.WARNING,
            )
            return

        worker_types = set(_WORKER_TYPES.values())

        for trainer, produced in TRAIN_INFO.items():
            for produced_type, info in produced.items():
                ability = info.get("ability")
                if ability is None:
                    continue
                # Worker entries (SCV/Probe/Drone) produce structures; everyone
                # else produces non-structure units.
                if trainer in worker_types:
                    self._build_ability_to_structure[ability] = produced_type
                else:
                    self._train_ability_to_unit[ability] = produced_type

        for _building, upgrades in RESEARCH_INFO.items():
            for upgrade_id, info in upgrades.items():
                ability = info.get("ability")
                if ability is not None:
                    self._research_ability_to_upgrade[ability] = upgrade_id

    async def update(self):
        if not self.enabled:
            return

        # v3: refresh enemy-sighting freshness on every tick (not just on the
        # ``interval_seconds`` cadence) so that a 12s snapshot interval cannot
        # quantise the "last seen X seconds ago" reading to a multiple of 12.
        self._refresh_enemy_seen_timestamp()

        if self.ai.time - self.last_recorded_time < self.interval_seconds:
            return

        try:
            snapshot = self.build_full_observation()
            view = self.mask_observation(snapshot, "full")
            text_obs = self.render_observation(view, "full")
            self.record_history.append(
                {
                    "game_time_seconds": round(self.ai.time, 2),
                    "observation_full": snapshot,
                    "observation_view": view,
                    "observation_view_type": "full",
                    "text_observation": text_obs,
                }
            )
            self._snapshots_since_autosave += 1
            if (
                self._autosave_every_snapshots > 0
                and self._snapshots_since_autosave >= self._autosave_every_snapshots
            ):
                self.flush_to_disk(game_result=None, reason="autosave_snapshot")
                self._snapshots_since_autosave = 0
        except Exception as exc:
            self.print(
                f"LLMObservationRecorder failed to capture snapshot: {exc}",
                stats=False,
                log_level=logging.WARNING,
            )
        finally:
            self.last_recorded_time = self.ai.time

    async def post_update(self):
        # Nothing to render in-game.
        pass

    def record_llm_interaction(self, record: Dict) -> None:
        """Append one LLM response and the action history at response time."""
        if not self.enabled:
            return
        self.llm_interactions.append(record)
        self._interactions_since_autosave += 1
        if (
            self._autosave_every_interactions > 0
            and self._interactions_since_autosave >= self._autosave_every_interactions
        ):
            self.flush_to_disk(game_result=None, reason="autosave_interaction")
            self._interactions_since_autosave = 0

    @staticmethod
    def _ensure_parent_dir(path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    @staticmethod
    def _open_text_write(path: str):
        """Open a UTF-8 text file for writing, tolerating Windows long paths."""
        LLMObservationRecorder._ensure_parent_dir(path)
        try:
            return open(path, "w", encoding="utf-8")
        except OSError:
            # Windows MAX_PATH workaround for deeply nested record folders.
            if os.name == "nt" and not str(path).startswith("\\\\?\\"):
                extended = "\\\\?\\" + os.path.abspath(path)
                LLMObservationRecorder._ensure_parent_dir(os.path.abspath(path))
                return open(extended, "w", encoding="utf-8")
            raise

    def flush_to_disk(
        self,
        game_result: Optional[Result] = None,
        reason: str = "manual",
    ) -> Optional[str]:
        """Write the current buffer to JSON.

        Safe to call repeatedly (overwrites the same path). Used by ``on_end``,
        autosave, atexit, and interrupt handlers so force-quit still leaves a
        record.

        Once ``on_end`` has persisted a real Result, later best-effort flushes
        that pass ``game_result=None`` are skipped so they cannot overwrite
        Victory/Defeat. A true interrupt with no known result is recorded as
        Defeat (treated as failure).
        """
        if not self.enabled or (not self.record_history and not self.llm_interactions):
            return None

        with _FLUSH_LOCK:
            try:
                if game_result is not None:
                    self._known_game_result = game_result

                # ``match_runner_finally`` / atexit / signals often run after a
                # successful ``on_end`` save; rewriting would wipe the result.
                if (
                    reason != "on_end"
                    and self._last_flush_reason == "on_end"
                    and self._known_game_result is not None
                    and game_result is None
                ):
                    return self.output_path or self._resolve_output_path()

                effective_result = (
                    game_result
                    if game_result is not None
                    else self._known_game_result
                )
                output_path = self._resolve_output_path()
                metadata = self._build_metadata(effective_result)
                metadata["save_reason"] = reason
                payload: Dict[str, Any] = {"metadata": metadata}
                if self.record_history:
                    payload["records"] = self.record_history
                if self.llm_interactions:
                    payload["interactions"] = self.llm_interactions

                with self._open_text_write(output_path) as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)

                self._last_flush_reason = reason
                # Autosave is frequent during play; keep the console quiet.
                if not str(reason).startswith("autosave"):
                    self._emit(
                        f"LLM observations saved to {output_path} "
                        f"(reason={reason}, {len(self.record_history)} snapshots, "
                        f"{len(self.llm_interactions)} LLM interactions)."
                    )
                return output_path
            except Exception as exc:
                self._emit(
                    f"LLMObservationRecorder failed to save ({reason}): {exc}",
                    log_level=logging.WARNING,
                )
                return None

    def _emit(self, msg: str, *, log_level: int = logging.INFO) -> None:
        """Log without requiring manager ``start()`` / ``knowledge``."""
        knowledge = getattr(self, "knowledge", None)
        if knowledge is not None:
            try:
                self.print(msg, stats=False, log_level=log_level)
                return
            except Exception:
                pass
        logging.getLogger(__name__).log(
            log_level, "[LLMObservationRecorder] %s", msg
        )

    async def on_end(self, game_result: Result):
        self.flush_to_disk(game_result=game_result, reason="on_end")
        try:
            _ACTIVE_RECORDERS.discard(self)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Snapshot pipeline (Step 1: Build Structured Data)
    # ------------------------------------------------------------------

    def _build_snapshot(self) -> Dict:
        """Aggregate all extractors into a single master snapshot dict."""
        return {
            "time": round(self.ai.time, 2),
            "time_formatted": self.ai.time_formatted,
            "economy": self._extract_economy_state(),
            "own_forces": self._extract_own_forces_infrastructure(),
            "enemy": self._extract_enemy_intelligence(),
            "map_control": self._extract_map_control(),
            "combat": self._extract_combat_analysis(),
            "memory_flags": self._extract_memory_flags(),
            # v3: top-level list of completed upgrades, so the LLM can spot
            # "qualitative" power spikes (Warp Gate, Stim, +1 attack, ...).
            "upgrades": self._extract_upgrades(),
        }

    def build_full_observation(
        self,
        *,
        army_state: Optional[Dict] = None,
    ) -> Dict:
        """Build the single canonical snapshot used by every live Agent.

        Existing extractors remain the low-level source of truth. Army state
        is collected through the registered ``LLMArmyControlAct`` when it is
        available; callers may also pass the state they already collected so
        one Army poll never samples it twice.
        """
        from commander.observation import build_full_observation

        legacy_snapshot = self._build_snapshot()
        if army_state is None:
            army_state = self._collect_registered_army_state()

        macro_execution = self._collect_macro_execution_state()
        combat_execution = getattr(
            self.ai, "llm_combat_execution_state", None
        )
        previous_decision = getattr(self.ai, "llm_previous_decision", None)
        return build_full_observation(
            legacy_snapshot,
            army_state=army_state,
            macro_execution=macro_execution,
            combat_execution=combat_execution,
            previous_decision=previous_decision,
            game_loop=self._current_game_loop(),
            game_time_limit_seconds=getattr(
                self.ai, "game_time_limit_seconds", None
            ),
        )

    @staticmethod
    def mask_observation(
        full_observation: Dict,
        view_type: str,
        **kwargs,
    ) -> Dict:
        from commander.observation import mask_observation

        return mask_observation(
            full_observation, view_type, **kwargs
        )

    @staticmethod
    def render_observation(view: Dict, view_type: str) -> str:
        from commander.observation import render_observation

        return render_observation(view, view_type)

    def capture_observation_bundle(
        self,
        view_type: str,
        *,
        army_state: Optional[Dict] = None,
        **mask_kwargs,
    ) -> tuple:
        """Return ``(text, full, masked_view)`` from one immutable snapshot."""
        full = self.build_full_observation(army_state=army_state)
        view = self.mask_observation(full, view_type, **mask_kwargs)
        text = self.render_observation(view, view_type)
        return text, full, view

    def _collect_registered_army_state(self) -> Dict:
        act = getattr(self.ai, "llm_army_control_act", None)
        if act is None:
            return {}
        try:
            from commander.combat_state import (
                collect_army_control_state,
            )

            return collect_army_control_state(act)
        except Exception as exc:
            self.print(
                f"Unable to collect unified Army observation: {exc}",
                stats=False,
                log_level=logging.WARNING,
            )
            return {}

    def _collect_macro_execution_state(self) -> Dict:
        state = dict(
            getattr(self.ai, "llm_macro_execution_state", {}) or {}
        )
        serialise = getattr(self.ai, "_serialise_active_tasks", None)
        if callable(serialise):
            try:
                state["active_macro_tasks"] = serialise()
            except Exception:
                state.setdefault("active_macro_tasks", [])
        else:
            state.setdefault("active_macro_tasks", [])
        return state

    def _current_game_loop(self) -> Optional[int]:
        try:
            return int(self.ai.state.game_loop)
        except Exception:
            return None

    def _extract_economy_state(self) -> Dict:
        """Current resources, supply usage, per-minute incomes and saturation."""
        mineral_per_min = 0.0
        gas_per_min = 0.0
        if self._income_calculator is not None:
            # IncomeCalculator stores per-second mining estimates; LLM prompts
            # are easier to reason about with per-minute numbers.
            mineral_per_min = round(self._income_calculator.mineral_income * 60, 1)
            gas_per_min = round(self._income_calculator.gas_income * 60, 1)

        ideal_worker_count = self._calculate_ideal_worker_count()

        return {
            "minerals": int(self.ai.minerals),
            "vespene": int(self.ai.vespene),
            "supply_used": int(self.ai.supply_used),
            "supply_cap": int(self.ai.supply_cap),
            "supply_left": int(self.ai.supply_left),
            "supply_workers": int(self.ai.supply_workers),
            "supply_army": int(self.ai.supply_army),
            "minerals_per_min": mineral_per_min,
            "vespene_per_min": gas_per_min,
            # v3: economic saturation. Pairs with ``supply_workers`` so the
            # LLM can tell "I'm at 22/30 ideal -> keep producing SCVs" from
            # "I'm at 22/22 ideal -> stop, build more bases / cut workers".
            "ideal_worker_count": ideal_worker_count,
        }

    def _calculate_ideal_worker_count(self) -> int:
        """Sum ``ideal_harvesters`` across all ready townhalls and gas buildings.

        A finished Command Center / Nexus / Hatchery reports ``2 * <mineral
        patches in that base>`` (so 16 for a fresh main with 8 patches), and
        a finished Refinery / Assimilator / Extractor reports ``3``. Adding
        these gives the total "ideal worker slot count" the bot should aim
        for. We deliberately skip *under-construction* bases / gas because
        their ideal count is reported as 0 until the structure completes,
        which would over-state how saturated the economy already is.
        """
        total = 0
        try:
            for townhall in self.ai.townhalls.ready:
                total += max(0, int(getattr(townhall, "ideal_harvesters", 0) or 0))
            for gas in self.ai.gas_buildings.ready:
                total += max(0, int(getattr(gas, "ideal_harvesters", 0) or 0))
        except Exception:
            # ``townhalls`` / ``gas_buildings`` may not exist in some test
            # stubs; bail out with whatever we accumulated so the snapshot
            # still renders.
            return total
        return total

    def _extract_own_forces_infrastructure(self) -> Dict[str, Dict[str, int]]:
        """Classify own units / structures into four tiers.

        Returned shape::

            {
                "completed":         {<UnitTypeId.name>: count, ...},
                "under_construction":{<UnitTypeId.name>: count, ...},  # buildings 0<bp<1
                "workers_en_route":  {<UnitTypeId.name>: count, ...},  # SCV/Drone/Probe traveling
                "active_queues":     {<"Training X" | "Researching Y">: count, ...},
            }

        The four-tier breakdown lets the LLM (or downstream consumer) tell
        apart "the building is finished" / "the foundation is laid and being
        hammered" / "a worker is on its way to lay the foundation" / "a queue
        is producing units or upgrades inside an existing building". This is
        what stops the LLM from re-issuing the same construction order while
        the previous one is still in flight.
        """
        result: Dict[str, Dict[str, int]] = {
            "completed": {},
            "under_construction": {},
            "workers_en_route": {},
            "active_queues": {},
        }

        if self.cache is None:
            return result

        completed = result["completed"]
        under_construction = result["under_construction"]
        active_queues = result["active_queues"]

        # Count from BotAI's current frame. The manager cache can lag behind
        # the economy counters used elsewhere in the same observation.
        try:
            current_own_units = list(self.ai.all_own_units)
        except (AttributeError, TypeError):
            current_own_units = [
                unit
                for units in self.cache.own_unit_cache.values()
                for unit in units
            ]

        # Per-type list of partial structures' centre points, used below to
        # decide whether a worker's BUILD_X target points at an already-laid
        # foundation (i.e. the worker is hammering / redundantly queued, not
        # really "en route").
        partial_positions: Dict[UnitTypeId, List] = {}

        for unit in current_own_units:
            unit_type = unit.type_id
            if unit.is_structure:
                if unit.is_ready:
                    completed[unit_type.name] = completed.get(unit_type.name, 0) + 1
                else:
                    # 0 < build_progress < 1: the foundation has been laid.
                    under_construction[unit_type.name] = (
                        under_construction.get(unit_type.name, 0) + 1
                    )
                    partial_positions.setdefault(unit_type, []).append(unit.position)
            elif unit.is_ready:
                completed[unit_type.name] = completed.get(unit_type.name, 0) + 1
            else:
                # Non-structure with build_progress < 1: a Zerg unit
                # currently morphing inside an egg / cocoon.
                key = f"Training {unit_type.name}"
                active_queues[key] = active_queues.get(key, 0) + 1

        # Workers en route: a worker counts as "en route" only if its BUILD_X
        # target is NOT on top of an already-laid foundation of the same type.
        #
        # The position-based dedup handles three subtly-different cases that
        # plain subtraction (worker_count - partial_count) gets wrong:
        #
        #   1. Terran SCV hammering an in-progress foundation keeps its
        #      ``TERRANBUILD_X`` order, with target == the foundation's
        #      Point2. So its target lands on the partial structure's
        #      position and we correctly skip it.
        #   2. Refinery / Extractor / Assimilator: the worker's target is a
        #      vespene geyser tag whose ``position`` matches the gas
        #      building's centre, so the same position-overlap rule fires
        #      cleanly without a special tag-based code path.
        #   3. Multiple workers redundantly dispatched to the SAME placement
        #      (sharpy's GridBuilding can do this when a previous order has
        #      not yet started construction): subtraction would report
        #      ``2 - 1 = 1`` extra en-route worker even though no second
        #      structure is coming. Position-overlap correctly skips both
        #      workers because they target the same Point2.
        my_race = self.knowledge.my_race if self.knowledge is not None else None
        worker_type = _WORKER_TYPES.get(my_race) if my_race else None
        if worker_type is not None and self._build_ability_to_structure:
            workers = self.cache.own(worker_type)
            workers_en_route = result["workers_en_route"]
            for worker in workers:
                if not worker.orders:
                    continue
                first_order = worker.orders[0]
                ability_id = self._safe_ability_id(first_order.ability)
                if ability_id is None:
                    continue
                struct_type = self._build_ability_to_structure.get(ability_id)
                if struct_type is None:
                    continue

                # Resolve the order's target into a Point2-like position.
                target = first_order.target
                target_pos = None
                if isinstance(target, int):
                    target_unit = self.cache.by_tag(target)
                    if target_unit is None:
                        # Unknown target tag - safest to skip rather than
                        # over-report.
                        continue
                    target_pos = target_unit.position
                elif target is not None:
                    target_pos = target  # Point2 placement spot

                if target_pos is None:
                    continue

                # If any partial structure of the matching type sits on this
                # exact spot, the worker is hammering it (or a redundant
                # duplicate) - do not count as en route.
                candidates = partial_positions.get(struct_type, [])
                if any(self._positions_match(target_pos, p) for p in candidates):
                    continue

                workers_en_route[struct_type.name] = (
                    workers_en_route.get(struct_type.name, 0) + 1
                )

        # Active queues from completed production / research buildings. Each
        # queued action is one entry in ``unit.orders``.
        for unit in current_own_units:
            if not unit.is_structure or not unit.is_ready or not unit.orders:
                continue
            for order in unit.orders:
                ability_id = self._safe_ability_id(order.ability)
                if ability_id is None:
                    continue

                produced_unit = self._train_ability_to_unit.get(ability_id)
                if produced_unit is not None:
                    key = f"Training {produced_unit.name}"
                    active_queues[key] = active_queues.get(key, 0) + 1
                    continue

                upgrade = self._research_ability_to_upgrade.get(ability_id)
                if upgrade is not None:
                    key = f"Researching {upgrade.name}"
                    active_queues[key] = active_queues.get(key, 0) + 1

        return result

    @staticmethod
    def _positions_match(p1, p2, tol: float = 1.0) -> bool:
        """Return True if two Point2-like values lie within ``tol`` of each other.

        Hand-rolled distance check so we don't have to care which Point2
        helper the underlying lib exposes - we only need ``.x`` / ``.y``.
        """
        if p1 is None or p2 is None:
            return False
        try:
            dx = float(p1.x) - float(p2.x)
            dy = float(p1.y) - float(p2.y)
        except Exception:
            return False
        return (dx * dx + dy * dy) < tol * tol

    @staticmethod
    def _safe_ability_id(ability) -> Optional[AbilityId]:
        """Defensively pull an ``AbilityId`` from a UnitOrder's ability field."""
        if ability is None:
            return None
        # ``UnitOrder.ability`` is normally an ``AbilityData``; ``exact_id``
        # gives us the un-remapped AbilityId. Fall back to ``id`` and finally
        # to the raw value if either path raises.
        for attr in ("exact_id", "id"):
            try:
                value = getattr(ability, attr, None)
                if isinstance(value, AbilityId):
                    return value
            except Exception:
                continue
        try:
            return AbilityId(ability)  # type: ignore[arg-type]
        except Exception:
            return None

    def _refresh_enemy_seen_timestamp(self) -> None:
        """Refresh army-intel age only when a non-worker army unit is visible."""
        try:
            visible_army = self.ai.enemy_units.filter(
                lambda unit: unit.type_id not in _NON_ARMY_MOBILE_TYPES
            )
        except Exception:
            return
        if visible_army:
            self.last_enemy_seen_at = float(self.ai.time)

    def _extract_enemy_intelligence(self) -> Dict:
        """Aggregated counts of enemy units/buildings ever observed.

        v3 shape::

            {
                "composition": {<UnitTypeId.name>: count, ...},
                "last_observation_time": <float seconds | None>,
                "seconds_since_last_seen": <float seconds | None>,
            }

        ``last_observation_time`` / ``seconds_since_last_seen`` are ``None``
        when we have never had any mobile enemy unit in vision. The LLM
        prompt formatter is responsible for rendering the ``None`` case
        gracefully (no "Last seen: ..." suffix).
        """
        composition: Dict[str, int] = {}
        if self._enemy_units_manager is not None:
            # ``unit_types`` is a KeysView, materialise to avoid mutation
            # issues if the manager updates while we iterate.
            for unit_type in list(self._enemy_units_manager.unit_types):
                count = self._enemy_units_manager.unit_count(unit_type)
                if count > 0:
                    composition[unit_type.name] = count

        last_seen = self.last_enemy_seen_at
        if last_seen is None:
            seconds_since = None
        else:
            seconds_since = round(max(0.0, float(self.ai.time) - last_seen), 1)
            last_seen = round(last_seen, 2)

        return {
            "composition": composition,
            "last_observation_time": last_seen,
            "seconds_since_last_seen": seconds_since,
        }

    def _extract_upgrades(self) -> List[str]:
        """Return a sorted list of completed-upgrade names for the LLM.

        Reads :pyattr:`BotAI.state.upgrades` (a ``set[UpgradeId]`` of
        upgrades that finished researching this game) and translates each
        entry to its uppercase :pyattr:`UpgradeId.name`. The blocklist is
        applied here so the resulting list is "things the LLM should
        actually consider when deciding the next action".

        Output is sorted alphabetically for stable text formatting and
        cheap diff-ability across consecutive snapshots.
        """
        names: List[str] = []
        try:
            upgrades = self.ai.state.upgrades
        except Exception:
            return names

        for upgrade in upgrades:
            if upgrade in _UPGRADE_BLOCKLIST:
                continue
            try:
                names.append(upgrade.name)
            except Exception:
                # Defensive: an unknown numeric id snuck in. Render the int
                # so the LLM at least sees *something*, rather than dropping.
                names.append(str(upgrade))

        names.sort()
        return names

    def _extract_map_control(self) -> Dict:
        """Counts of own/enemy/neutral expansion zones."""
        own_bases = 0
        known_enemy_bases = 0
        neutral_zones = 0

        own_base_minerals = self._empty_own_base_minerals()
        own_base_gas = self._empty_own_base_gas()

        if self.zone_manager is not None and self.zone_manager.expansion_zones:
            for index, zone in enumerate(self.zone_manager.expansion_zones):
                if zone.is_ours:
                    own_bases += 1
                    self._add_own_base_mineral_status(own_base_minerals, zone, index, own_bases)
                    self._add_own_base_gas_status(own_base_gas, zone, index, own_bases)
                elif zone.is_enemys:
                    known_enemy_bases += 1
                else:
                    neutral_zones += 1

        return {
            "own_bases": own_bases,
            "known_enemy_bases": known_enemy_bases,
            "neutral_expansions": neutral_zones,
            "own_base_minerals": own_base_minerals,
            "own_base_gas": own_base_gas,
        }

    @staticmethod
    def _empty_own_base_minerals() -> Dict:
        return {
            "Full": 0,
            "Plenty": 0,
            "Limited": 0,
            "NearEmpty": 0,
            "Empty": 0,
            "details": [],
        }

    def _add_own_base_mineral_status(self, own_base_minerals: Dict, zone, index: int, own_base_number: int) -> None:
        resources = getattr(zone, "resources", None)
        resource_name = getattr(resources, "name", str(resources) if resources is not None else "Unknown")
        if resource_name in own_base_minerals:
            own_base_minerals[resource_name] += 1

        minerals_left = getattr(zone, "last_minerals", 0)
        try:
            minerals_left = int(minerals_left or 0)
        except (TypeError, ValueError):
            minerals_left = 0

        own_base_minerals["details"].append(
            {
                "label": self._own_base_label(own_base_number),
                "zone_index": int(getattr(zone, "zone_index", index)),
                "resources": resource_name,
                "minerals_left": minerals_left,
            }
        )

    @staticmethod
    def _empty_own_base_gas() -> Dict:
        return {"details": []}

    def _add_own_base_gas_status(self, own_base_gas: Dict, zone, index: int, own_base_number: int) -> None:
        gas_buildings = getattr(zone, "gas_buildings", []) or []
        gas_left = 0
        geysers = 0
        for gas in gas_buildings:
            geysers += 1
            try:
                gas_left += int(getattr(gas, "vespene_contents", 0) or 0)
            except (TypeError, ValueError):
                pass

        own_base_gas["details"].append(
            {
                "label": self._own_base_label(own_base_number),
                "zone_index": int(getattr(zone, "zone_index", index)),
                "geysers": geysers,
                "gas_left": gas_left,
            }
        )

    @staticmethod
    def _own_base_label(own_base_number: int) -> str:
        labels = {
            1: "main",
            2: "natural",
            3: "third",
            4: "fourth",
            5: "fifth",
            6: "sixth",
        }
        return labels.get(own_base_number, f"base{own_base_number}")

    def _extract_combat_analysis(self) -> Dict:
        """Framework-level advantage estimates plus accumulated losses."""
        result: Dict = {
            "advantage_predicted": "Even",
            "army_advantage": "Even",
            "income_advantage": "Even",
            "our_army_power": 0.0,
            "enemy_army_power": 0.0,
            "enemy_air": "NoAir",
            "own_lost_minerals": 0,
            "own_lost_gas": 0,
            "enemy_lost_minerals": 0,
            "enemy_lost_gas": 0,
        }

        if self._game_analyzer is not None:
            try:
                result["advantage_predicted"] = self._game_analyzer.our_army_predict.name
                result["army_advantage"] = self._game_analyzer.our_army_advantage.name
                result["income_advantage"] = self._game_analyzer.our_income_advantage.name
                if self._game_analyzer.our_power is not None:
                    result["our_army_power"] = round(self._game_analyzer.our_power.power, 1)
                if self._game_analyzer.enemy_power is not None:
                    result["enemy_army_power"] = round(self._game_analyzer.enemy_power.power, 1)
                result["enemy_air"] = self._game_analyzer.enemy_air.name
            except Exception:
                # Game analyzer reads heavy state; missing data should not
                # break the whole recorder, so we silently skip.
                pass

        if self._lost_units_manager is not None:
            try:
                own_min, own_gas = self._lost_units_manager.calculate_own_lost_resources()
                enemy_min, enemy_gas = self._lost_units_manager.calculate_enemy_lost_resources()
                result["own_lost_minerals"] = int(own_min)
                result["own_lost_gas"] = int(own_gas)
                result["enemy_lost_minerals"] = int(enemy_min)
                result["enemy_lost_gas"] = int(enemy_gas)
            except Exception:
                pass

        return result

    def _extract_memory_flags(self) -> Dict:
        """Boolean flags summarising tactical signals from extension managers."""
        flags: Dict = {
            "is_rushing": False,
            "rush_build": "Macro",
            "macro_build": "StandardMacro",
            "enemy_cloak_threat": False,
            "has_proxy_buildings": False,
            "remembered_enemy_units": 0,
        }

        if self._build_detector is not None:
            try:
                flags["is_rushing"] = bool(self._build_detector.rush_detected)
                flags["rush_build"] = self._build_detector.rush_build.name
                flags["macro_build"] = self._build_detector.macro_build.name
            except Exception:
                pass

        if self._enemy_units_manager is not None:
            try:
                flags["enemy_cloak_threat"] = bool(self._enemy_units_manager.enemy_cloak_trigger)
            except Exception:
                pass

        if self._memory_manager is not None:
            try:
                flags["remembered_enemy_units"] = len(self._memory_manager.ghost_units)
            except Exception:
                pass

        flags["has_proxy_buildings"] = self._detect_proxy_buildings()
        return flags

    def _detect_proxy_buildings(self) -> bool:
        """Heuristic: any enemy structure within ~60 tiles of our main base."""
        if self.zone_manager is None:
            return False

        own_main = self.zone_manager.zones.get(self.ai.start_location)
        if own_main is None:
            return False

        center = own_main.center_location
        for structure in self.ai.enemy_structures:
            if structure.distance_to(center) < 60:
                return True
        return False

    # ------------------------------------------------------------------
    # Text generation (Step 2: Generate English Text Observation)
    # ------------------------------------------------------------------

    def _generate_english_text_obs(self, snapshot: Dict) -> str:
        """Render the structured snapshot into an LLM-friendly English prompt.

        Output is divided into ``[Tag]``-labelled sections so an LLM can
        attend to the relevant block when answering queries like "what
        forces do I have" or "what is the enemy doing". This method
        intentionally only reads from ``snapshot`` so the prompt template can
        be modified without touching extraction logic.
        """
        eco = snapshot["economy"]
        own = snapshot["own_forces"]
        enemy = snapshot["enemy"]
        mc = snapshot["map_control"]
        combat = snapshot["combat"]
        flags = snapshot["memory_flags"]
        upgrades = snapshot.get("upgrades") or []

        # [Time]
        time_section = (
            f"[Time] {snapshot['time_formatted']} ({snapshot['time']:.1f}s)."
        )

        # [Economy] - v3: now includes worker saturation (X / ideal).
        #
        # The ``workers {current}/{ideal}`` notation is non-standard (SC2's
        # built-in UI only ever shows ``used/cap``), so we annotate it
        # inline with ``current/ideal`` to make the prompt self-explanatory
        # for the downstream LLM. ``ideal`` is the sum of
        # ``ideal_harvesters`` across all ready townhalls (2 * mineral
        # patches per base, ~16 on a fresh main) and gas buildings (3 per
        # refinery / assimilator / extractor), i.e. how many SCV/Drone/Probe
        # the current economy can productively employ.
        ideal_workers = eco.get("ideal_worker_count", 0)
        economy_section = (
            "[Economy] "
            f"{eco['minerals']} minerals, {eco['vespene']} vespene; "
            f"income {eco['minerals_per_min']:.0f} mins/min, "
            f"{eco['vespene_per_min']:.0f} gas/min. "
            f"Supply: {eco['supply_used']}/{eco['supply_cap']} "
            f"(workers {eco['supply_workers']}/{ideal_workers} current/ideal, "
            f"army {eco['supply_army']})."
        )

        # [Own Forces & Infrastructure] - the four-tier breakdown.
        own_lines: List[str] = ["[Own Forces & Infrastructure]"]
        own_lines.append(
            f"  Completed: {self._format_count_dict(own.get('completed'), empty='nothing built yet')}."
        )
        own_lines.append(
            f"  Under Construction: "
            f"{self._format_count_dict(own.get('under_construction'), empty='none')}."
        )
        own_lines.append(
            f"  Workers En Route: "
            f"{self._format_count_dict(own.get('workers_en_route'), empty='none')}."
        )
        own_lines.append(
            f"  Active Queues: "
            f"{self._format_active_queues(own.get('active_queues'))}."
        )
        own_section = "\n".join(own_lines)

        # [Enemy Intelligence] - v3: composition + freshness suffix.
        enemy_section = self._format_enemy_section(enemy)

        # [Map Control]
        map_lines = [
            "[Map Control] "
            f"{mc['own_bases']} own bases, "
            f"{mc['known_enemy_bases']} known enemy bases, "
            f"{mc['neutral_expansions']} neutral expansions remaining"
            f"{self._format_own_base_mineral_summary(mc.get('own_base_minerals'))}."
        ]
        details = self._format_own_base_mineral_details(mc.get("own_base_minerals"))
        if details:
            map_lines.append(f"  Own base details: {details}.")
        gas_details = self._format_own_base_gas_details(mc.get("own_base_gas"))
        if gas_details:
            map_lines.append(f"  Own base gas details: {gas_details}.")
        map_section = "\n".join(map_lines)

        # [Combat Analysis]
        combat_section = (
            "[Combat Analysis] "
            f"army advantage = {combat['army_advantage']}, "
            f"income advantage = {combat['income_advantage']}, "
            f"predicted = {combat['advantage_predicted']}. "
            f"Power: {combat['our_army_power']:.0f} vs "
            f"{combat['enemy_army_power']:.0f}. "
            f"Losses: own {combat['own_lost_minerals']} minerals/"
            f"{combat['own_lost_gas']} gas, "
            f"enemy {combat['enemy_lost_minerals']} minerals/"
            f"{combat['enemy_lost_gas']} gas."
        )

        # [Research & Technology] - v3.
        if upgrades:
            tech_section = "[Research & Technology] " + ", ".join(upgrades) + "."
        else:
            tech_section = "[Research & Technology] none."

        # [Threat Flags]
        flag_parts: List[str] = []
        if flags["is_rushing"]:
            flag_parts.append(f"enemy rush detected ({flags['rush_build']})")
        if flags["macro_build"] != "StandardMacro":
            flag_parts.append(f"enemy macro build = {flags['macro_build']}")
        if flags["has_proxy_buildings"]:
            flag_parts.append("proxy buildings spotted near base")
        if flags["enemy_cloak_threat"]:
            flag_parts.append("enemy cloak/burrow threat detected")
        if flag_parts:
            threat_section = "[Threat Flags] " + "; ".join(flag_parts) + "."
        else:
            threat_section = "[Threat Flags] none."

        return "\n".join(
            [
                time_section,
                economy_section,
                own_section,
                enemy_section,
                map_section,
                combat_section,
                tech_section,
                threat_section,
            ]
        )

    def _format_enemy_section(self, enemy: Dict) -> str:
        """Render the ``[Enemy Intelligence]`` line with optional freshness.

        ``enemy`` has the v3 shape produced by
        :py:meth:`_extract_enemy_intelligence`. The structured snapshot still
        carries ``last_observation_time`` / ``seconds_since_last_seen`` for
        offline analysis and possible RL features, but the LLM-facing text
        deliberately *omits* the ``(Last seen: Ns ago)`` suffix - in practice
        the value mostly hovers at a small multiple of ``interval_seconds``
        and rarely changes LLM decisions, so it just adds prompt noise.
        """
        composition = enemy.get("composition") or {}
        units_text = self._format_count_dict(
            composition, empty="nothing scouted yet"
        )
        return f"[Enemy Intelligence] {units_text}."

        # --- v3 freshness suffix (disabled 2026-05-15) -----------------
        # The structured fields are still populated; re-enable the lines
        # below if you want the LLM to see staleness in the prompt.
        # seconds_since = enemy.get("seconds_since_last_seen")
        # if seconds_since is None:
        #     return f"[Enemy Intelligence] {units_text}."
        # return (
        #     f"[Enemy Intelligence] {units_text}. "
        #     f"(Last seen: {int(round(seconds_since))}s ago)."
        # )

    @staticmethod
    def _format_count_dict(
        data: Optional[Dict[str, int]], empty: str = "none"
    ) -> str:
        """Render ``{name: count}`` as ``count1 name1, count2 name2`` sorted."""
        if not data:
            return empty
        ordered = sorted(data.items(), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(f"{count} {name}" for name, count in ordered)

    @staticmethod
    def _format_active_queues(
        data: Optional[Dict[str, int]], empty: str = "none"
    ) -> str:
        """Render queue keys like ``"Training MARINE"`` / ``"Researching X"``.

        Training items get ``Training N <name>`` (count makes sense - several
        marines may sit in queue). Research items get ``Researching <name>``
        because each research action is a single 0/1 toggle on the building.
        """
        if not data:
            return empty
        ordered = sorted(data.items(), key=lambda kv: (-kv[1], kv[0]))
        parts: List[str] = []
        for key, count in ordered:
            verb, _, name = key.partition(" ")
            if verb == "Researching" and name:
                parts.append(f"Researching {name}")
            elif verb == "Training" and name:
                parts.append(f"Training {count} {name}")
            else:
                parts.append(f"{count} {key}")
        return ", ".join(parts)

    @staticmethod
    def _format_own_base_mineral_summary(data: Optional[Dict]) -> str:
        if not data:
            return ""
        keys = ("Full", "Plenty", "Limited", "NearEmpty", "Empty")
        parts = [f"{key}={int(data.get(key, 0) or 0)}" for key in keys]
        return ". Own base minerals: " + ", ".join(parts)

    @staticmethod
    def _format_own_base_mineral_details(data: Optional[Dict]) -> str:
        if not data:
            return ""
        details = data.get("details") or []
        if not details:
            return ""
        parts = []
        for item in details:
            label = item.get("label", "?")
            resources = item.get("resources", "Unknown")
            minerals_left = item.get("minerals_left", "?")
            parts.append(f"{label}={resources}({minerals_left} minerals)")
        return ", ".join(parts)

    @staticmethod
    def _format_own_base_gas_details(data: Optional[Dict]) -> str:
        if not data:
            return ""
        details = data.get("details") or []
        if not details:
            return ""
        parts = []
        for item in details:
            label = item.get("label", "?")
            geysers = int(item.get("geysers", 0) or 0)
            gas_left = item.get("gas_left", "?")
            geyser_word = "geyser" if geysers == 1 else "geysers"
            parts.append(f"{label}={geysers} {geyser_word}({gas_left} gas)")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _build_metadata(self, game_result: Optional[Result]) -> Dict:
        knowledge = getattr(self, "knowledge", None)
        ai = getattr(self, "ai", None)
        my_race = getattr(knowledge, "my_race", None) if knowledge is not None else None
        enemy_race = (
            getattr(knowledge, "enemy_race", None) if knowledge is not None else None
        )
        game_info = getattr(ai, "game_info", None) if ai is not None else None
        time_value = getattr(ai, "time", 0.0) if ai is not None else 0.0
        try:
            time_seconds = round(float(time_value), 2)
        except (TypeError, ValueError):
            time_seconds = 0.0
        return {
            "map_name": getattr(game_info, "map_name", "Unknown"),
            "my_race": my_race.name if my_race else "Unknown",
            "enemy_race": enemy_race.name if enemy_race else "Unknown",
            "matchup": (
                f"{_RACE_SHORT.get(my_race, '?')}v"
                f"{_RACE_SHORT.get(enemy_race, '?')}"
            ),
            "opponent_id": getattr(ai, "opponent_id", None) if ai is not None else None,
            "bot_name": getattr(ai, "name", None) if ai is not None else None,
            "game_duration_seconds": time_seconds,
            "game_duration_formatted": (
                getattr(ai, "time_formatted", "00:00") if ai is not None else "00:00"
            ),
            # No known SC2 Result means the match was cut short; treat as loss.
            "result": game_result.name if game_result is not None else "Defeat",
            "interval_seconds": self.interval_seconds,
            "record_count": len(self.record_history),
            "llm_interaction_count": len(self.llm_interactions),
        }

    def _resolve_output_path(self) -> str:
        """Resolve the JSON output path with three-tier precedence.

        1. Explicit ``output_path`` if provided.
        2. ``replay_save_path`` (mirrors the SC2Replay prefix) if provided.
        3. Auto-generated ``Replay_<timestamp>_<matchup>_<map>.json`` in
           ``output_folder`` so files remain easy to correlate even when no
           replay path was injected.
        """
        if self.output_path:
            return self.output_path

        if self.replay_save_path:
            base, _ext = os.path.splitext(self.replay_save_path)
            return base + ".json"

        knowledge = getattr(self, "knowledge", None)
        ai = getattr(self, "ai", None)
        my_race = getattr(knowledge, "my_race", None) if knowledge is not None else None
        enemy_race = (
            getattr(knowledge, "enemy_race", None) if knowledge is not None else None
        )
        matchup = (
            f"{_RACE_SHORT.get(my_race, '?')}v"
            f"{_RACE_SHORT.get(enemy_race, '?')}"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_info = getattr(ai, "game_info", None) if ai is not None else None
        map_name = getattr(game_info, "map_name", "Unknown").replace(" ", "")
        filename = f"{DEFAULT_FILENAME_PREFIX}_{timestamp}_{matchup}_{map_name}.json"
        return os.path.join(self.output_folder, filename)
