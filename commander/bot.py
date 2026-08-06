"""Single-agent Commander bot: strategy.md + model-authored wake events."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from sc2.data import Race
from sharpy.interfaces import IZoneManager
from sharpy.knowledges import KnowledgeBot
from sharpy.plans import BuildOrder

from commander.agent import run_commander_decision
from commander.macro_exec import (
    ActOngoingMacroTasks,
    EmptyTactics,
    ForceFinishEnemyOnGG,
)
from commander.strategy import parse_strategy_document
from commander.wake_events import (
    FALLBACK_DELAY_SECONDS,
    build_trigger_hint,
    build_wake_snapshot,
    evaluate_wake_event,
    fallback_wake_event,
    rising_edge,
)

logger = logging.getLogger("commander.bot")

_RACE_MAP = {
    "terran": Race.Terran,
    "zerg": Race.Zerg,
    "protoss": Race.Protoss,
    "random": Race.Random,
}


def _is_enemy_gg_message(message: str) -> bool:
    text = (message or "").strip().lower()
    return re.fullmatch(r"(gg|ggwp|good game)[.!?]*", text) is not None


class CommanderBot(KnowledgeBot):
    """Execute one strategy.md via a single LLM with OpenAI tool_calls."""

    # Observation recorder only (does not trigger LLM decisions).
    OBS_RECORD_INTERVAL: float = 60.0
    WAKE_COOLDOWN: float = 2.0
    zone_manager: IZoneManager

    def __init__(
        self,
        race_name: str = "terran",
        instruct: str = "",
        commander_model_key: str = "",
        record_dir: str = "",
        *,
        force_strategy: Optional[str] = None,
    ):
        super().__init__("SC2 Commander")
        self.race_name = race_name.strip().lower()
        self.instruct = (instruct or "").strip()
        self.commander_model_key = (commander_model_key or "").strip()
        self.record_dir = (record_dir or "").strip()
        force = (force_strategy or "").strip()
        self.force_strategy: Optional[str] = (
            force if force and force.lower() != "none" else None
        )

        self.selected_strategy: Optional[str] = None
        self.strategy_description: str = ""
        self.active_tasks: List[Dict[str, Any]] = []
        self.commander_army_policy = None
        self._last_army_summary: Dict[str, Any] = {}
        # Snapshot of the last Commander tool decision for the next observation.
        self.llm_previous_decision: Optional[Dict[str, Any]] = None
        self._last_decision_time: float = -self.WAKE_COOLDOWN
        self._decision_count: int = 0
        self._wake_event: Optional[Dict[str, Any]] = None
        self._wake_prev_satisfied: Optional[bool] = None
        self._wake_is_fallback: bool = False
        self._wake_armed_at: Optional[float] = None
        self._wake_baseline_objective_status: str = ""
        # Always-on max sleep after each decision (independent of model event).
        self._wake_deadline: Optional[float] = None
        self.llm_macro_execution_state: Dict[str, Any] = {
            "last_tasks": [],
            "last_update_game_time": None,
            "last_issues": [],
        }

        self._get_action_fn: Optional[Callable] = None
        self._get_action_space_fn: Optional[Callable] = None
        self._action_space_cache: Optional[Dict[str, str]] = None

        self.enemy_said_gg: bool = False
        self.enemy_gg_message: str = ""
        self._seen_chat_messages: Set[Tuple[Optional[int], str]] = set()

        if self.record_dir:
            self.llm_observation_recorder.output_folder = self.record_dir

    # ------------------------------------------------------------------
    # skills/
    # ------------------------------------------------------------------

    @property
    def _skills_root(self) -> str:
        return os.path.normpath(
            os.path.join(os.path.dirname(
                os.path.abspath(__file__)), os.pardir, "skills")
        )

    @property
    def _skills_race_dir(self) -> str:
        return os.path.normpath(os.path.join(self._skills_root, self.race_name))

    def _load_race_action_module(self) -> None:
        module_path = f"skills.{self.race_name}.Action"
        try:
            mod = importlib.import_module(module_path)
            self._get_action_fn = getattr(mod, "get_action", None)
            self._get_action_space_fn = getattr(mod, "get_action_space", None)
        except ImportError:
            logger.warning("Action module %s not found.", module_path)
            self._get_action_fn = None
            self._get_action_space_fn = None
        self._action_space_cache = (
            self._get_action_space_fn() if self._get_action_space_fn else {}
        )

    def _apply_forced_strategy(self, name: str) -> None:
        target_dir = os.path.join(self._skills_race_dir, name)
        md_path = os.path.join(target_dir, "strategy.md")
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"strategy folder not found: {target_dir}")

        detail = ""
        if os.path.isfile(md_path):
            with open(md_path, "r", encoding="utf-8") as handle:
                raw = handle.read()
            parsed = parse_strategy_document(raw)
            detail = parsed.get("detail") or raw.strip()

        self.selected_strategy = name
        self.strategy_description = detail
        self._emit(
            "force_strategy=%s description_chars=%d",
            name,
            len(detail),
        )
        self._record_llm_interaction(
            {
                "game_time": 0.0,
                "trigger_reason": "strategy_forced",
                "forced_strategy": name,
                "strategy_description": detail,
            }
        )

    def _load_dynamic_tactics(self) -> BuildOrder:
        if not self.selected_strategy:
            return EmptyTactics()
        module_path = f"skills.{self.race_name}.{self.selected_strategy}.base_tactics"
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            logger.warning(
                "Cannot import %s; using EmptyTactics.", module_path)
            return EmptyTactics()
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BuildOrder)
                and attr is not BuildOrder
            ):
                try:
                    return attr()
                except TypeError:
                    try:
                        return attr(20)
                    except Exception as exc:
                        logger.warning(
                            "Failed to instantiate %s: %s", attr_name, exc)
        return EmptyTactics()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def on_start(self):
        self._load_race_action_module()
        strategy = self.force_strategy or "mid_tank"
        self._apply_forced_strategy(strategy)
        await super().on_start()
        self.zone_manager = self.knowledge.get_required_manager(IZoneManager)
        self.llm_observation_recorder.interval_seconds = self.OBS_RECORD_INTERVAL
        if self.record_dir:
            self.llm_observation_recorder.output_folder = self.record_dir

    async def pre_step_execute(self):
        self._process_current_chat_messages()
        if self._decision_count <= 0:
            self._run_decision(trigger_reason="commander_bootstrap")
            return
        if self.time - self._last_decision_time < self.WAKE_COOLDOWN:
            return
        reason = self._poll_wake_trigger()
        if reason:
            self._run_decision(trigger_reason=reason)

    async def create_plan(self) -> BuildOrder:
        return BuildOrder(
            [
                ActOngoingMacroTasks(active_tasks_ref=self.active_tasks),
                self._load_dynamic_tactics(),
                ForceFinishEnemyOnGG(lambda ai: getattr(
                    ai, "enemy_said_gg", False)),
            ]
        )

    # ------------------------------------------------------------------
    # decision
    # ------------------------------------------------------------------

    def _resolved_model_key(self) -> str:
        return (self.commander_model_key or "").strip()

    def _poll_wake_trigger(self) -> Optional[str]:
        """Return trigger reason on rising-edge wake or deadline fuse."""
        snapshot = self._build_wake_snapshot()
        event_ok = (
            evaluate_wake_event(self._wake_event, snapshot)
            if self._wake_event
            else False
        )
        event_edge = rising_edge(event_ok, self._wake_prev_satisfied)
        self._wake_prev_satisfied = event_ok
        if event_edge:
            if self._wake_is_fallback:
                return "wake_fallback_timeout"
            return "wake_event"
        # Always-on now+N fuse: fires once even if the model event stays false
        # (or stays sticky-true without a new rising edge).
        if (
            self._wake_deadline is not None
            and float(self.time) >= float(self._wake_deadline)
        ):
            self._wake_deadline = None
            return "wake_fallback_timeout"
        return None

    def _build_wake_snapshot(self) -> Dict[str, Any]:
        cleanup_hint = ""
        army_state: Dict[str, Any] = {}
        last_scout_result_time: Optional[float] = None
        act = getattr(self, "llm_army_control_act", None)
        if act is not None:
            try:
                from commander.combat_state import (
                    build_cleanup_runtime_hint,
                    collect_army_control_state,
                )

                army_state = collect_army_control_state(act)
                cleanup_hint = (
                    build_cleanup_runtime_hint(act, army_state) or ""
                ).strip()
                raw_time = getattr(act, "_last_scout_result_time", None)
                if raw_time is not None:
                    last_scout_result_time = float(raw_time)
            except Exception as exc:
                logger.warning("wake snapshot army state failed: %s", exc)
                army_state = {}
        return build_wake_snapshot(
            time_seconds=float(self.time),
            supply_used=int(getattr(self, "supply_used", 0) or 0),
            supply_cap=int(getattr(self, "supply_cap", 0) or 0),
            own_unit_type_counts=army_state.get("own_unit_type_counts") or {},
            army_groups=army_state.get("army_groups") or [],
            army_summary=self._last_army_summary,
            available_zones=army_state.get("available_zones") or [],
            last_scout_result=str(
                army_state.get("last_scv_scout_result") or ""
            ),
            last_scout_result_time=last_scout_result_time,
            scan_ready=int(army_state.get("scan_ready") or 0) > 0,
            cleanup_hint_present="[Runtime Search-And-Destroy Hint]"
            in cleanup_hint,
            wake_armed_at=self._wake_armed_at,
            baseline_objective_status=self._wake_baseline_objective_status,
        )

    def _run_decision(self, *, trigger_reason: str) -> None:
        model_key = self._resolved_model_key()
        if not model_key:
            logger.warning("commander model key empty; skip")
            return
        self.commander_model_key = model_key
        action_space = self._action_space_cache or {}
        if not action_space:
            logger.warning("action space empty; skip")
            return

        self._last_decision_time = float(self.time)
        obs_text, full_obs, _view = self._capture_observation_bundle("full")
        cleanup_hint = self._cleanup_runtime_hint()
        trigger_hint = ""
        if trigger_reason != "commander_bootstrap":
            trigger_hint = build_trigger_hint(
                reason=trigger_reason,
                event=self._wake_event,
            )
        runtime_hint = "\n\n".join(
            part for part in (cleanup_hint, trigger_hint) if part
        )
        previous_macro = [
            {"action": t.get("action"), "to_count": t.get("to_count")}
            for t in self.active_tasks
        ]
        outcome = run_commander_decision(
            race=self.race_name,
            strategy_description=self.strategy_description,
            observation_text=obs_text,
            previous_macro_tasks=previous_macro,
            previous_army_summary=self._last_army_summary,
            runtime_hint=runtime_hint,
            action_space=action_space,
            model_key=model_key,
            full_observation=full_obs,
            ensure_addon_parents=self._ensure_addon_parent_tasks,
        )
        tasks = outcome["tasks"]
        policy = outcome["policy"]
        self._replace_active_tasks(tasks)
        self.commander_army_policy = policy
        self._last_army_summary = outcome["army_summary"]
        wake_event = outcome.get("wake_event")
        wake_fallback = False
        if not wake_event:
            wake_event = fallback_wake_event(
                float(self.time), delay=FALLBACK_DELAY_SECONDS
            )
            wake_fallback = True
            issues = list(outcome.get("issues") or [])
            if "wake_event:missing" not in issues:
                issues.append("wake_event:fallback_now_plus_60")
            outcome["issues"] = issues
        self._wake_event = wake_event
        self._wake_is_fallback = wake_fallback
        # Arm baselines before resampling so scout_just_finished /
        # objective_status_became ignore already-true stale states.
        self._wake_armed_at = float(self.time)
        self._wake_deadline = float(self.time) + float(FALLBACK_DELAY_SECONDS)
        arm_snapshot = self._build_wake_snapshot()
        self._wake_baseline_objective_status = str(
            arm_snapshot.get("objective_status") or ""
        )
        # Rebuild once baselines are latched, then sample rising-edge prev.
        self._wake_prev_satisfied = evaluate_wake_event(
            wake_event, self._build_wake_snapshot()
        )
        self._decision_count += 1
        self.llm_macro_execution_state = {
            "last_tasks": [
                f"{t['action']} to {t['to_count']}" for t in tasks
            ],
            "last_update_game_time": round(float(self.time), 1),
            "last_issues": list(outcome["issues"]),
            "wake_event": wake_event,
            "wake_fallback": wake_fallback,
            "wake_armed_at": self._wake_armed_at,
            "wake_deadline": self._wake_deadline,
            "wake_baseline_objective_status": self._wake_baseline_objective_status,
        }
        # Available to the next observation cycle as [Previous Decision].
        army_summary = outcome.get("army_summary") or {}
        self.llm_previous_decision = {
            "game_time_seconds": round(float(self.time), 1),
            "macro_commands": [
                {
                    "name": str(t.get("action")),
                    "to_count": t.get("to_count"),
                }
                for t in tasks
                if isinstance(t, dict) and t.get("action")
            ],
            "army_commands": [
                {
                    "group_id": c.get("group_id"),
                    "destination_zone_id": c.get("destination_zone_id"),
                    "movement_mode": c.get("movement_mode"),
                }
                for c in list(army_summary.get("commands") or [])
                if isinstance(c, dict)
            ],
            "scan_zone_id": army_summary.get("scan_zone_id"),
            "scout_zone_id": army_summary.get("scout_zone_id"),
            "wake_event": wake_event,
            "issues": list(outcome.get("issues") or []),
        }
        self._emit(
            "t=%.1f reason=%s mode=%s tools=%d macro=%d groups=%d "
            "wake_fallback=%s deadline=%.1f reflect=%s issues=%s err=%s",
            float(self.time),
            trigger_reason,
            outcome.get("tool_mode") or "?",
            len(outcome["tool_calls"]),
            len(tasks),
            len(policy.commands),
            wake_fallback,
            float(self._wake_deadline),
            outcome.get("reflection_retries") or 0,
            outcome["issues"][:5],
            outcome["error"] or "ok",
        )
        for call in outcome["tool_calls"]:
            self._emit(
                "  tool %s %s",
                call.get("name"),
                call.get("arguments") or {},
            )
        if tasks:
            self._emit(
                "  macro=%s",
                ", ".join(f"{t['action']}->{t['to_count']}" for t in tasks),
            )
        else:
            self._emit("  macro=(none)")
        army = self._last_army_summary
        if army.get("commands"):
            self._emit(
                "  army=%s",
                "; ".join(
                    f"{c['group_id']}:{c['movement_mode']}->{c['destination_zone_id']}"
                    for c in army["commands"]
                ),
            )
        else:
            self._emit("  army=(no groups)")
        if army.get("scan_zone_id"):
            self._emit("  scan=%s", army["scan_zone_id"])
        if army.get("scout_zone_id"):
            self._emit("  scout=%s", army["scout_zone_id"])
        self._emit("  wake=%s deadline=%.1f", wake_event, float(self._wake_deadline))
        content = (outcome.get("content") or "").strip()
        if content:
            preview = content.replace("\n", " ")
            self._emit("  content=%s", preview)
        self._record_llm_interaction(
            {
                "trigger_reason": trigger_reason,
                "agent": "commander",
                "game_time": round(float(self.time), 1),
                "model_key": model_key,
                "text_observation": obs_text,
                "tool_calls": outcome["tool_calls"],
                "macro_tasks": tasks,
                "army_policy": self._last_army_summary,
                "wake_event": wake_event,
                "wake_fallback": wake_fallback,
                "wake_armed_at": self._wake_armed_at,
                "wake_deadline": self._wake_deadline,
                "reflection_retries": outcome.get("reflection_retries") or 0,
                "reflection_issues": outcome.get("reflection_issues") or [],
                "issues": outcome["issues"],
                "error": outcome["error"],
                "latency_seconds": outcome.get("latency_seconds"),
                "finish_reason": outcome.get("finish_reason"),
                "content": outcome.get("content") or "",
            }
        )

    # ------------------------------------------------------------------
    # observation / tasks
    # ------------------------------------------------------------------

    def _cleanup_runtime_hint(self) -> str:
        """Append map-clear cue when program-side cleanup conditions hold."""
        act = getattr(self, "llm_army_control_act", None)
        if act is None:
            return ""
        try:
            from commander.combat_state import (
                build_cleanup_runtime_hint,
                collect_army_control_state,
            )

            state = collect_army_control_state(act)
            return (build_cleanup_runtime_hint(act, state) or "").strip()
        except Exception as exc:
            logger.warning("cleanup runtime hint failed: %s", exc)
            return ""

    def _capture_observation_bundle(self, view_type: str = "full"):
        recorder = getattr(self, "llm_observation_recorder", None)
        if recorder is None:
            return "(observation unavailable)", None, None
        try:
            return recorder.capture_observation_bundle(view_type)
        except Exception as exc:
            logger.warning("observation failed: %s", exc)
            return "(observation unavailable)", None, None

    def _replace_active_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        self.active_tasks.clear()
        for idx, task in enumerate(tasks, start=1):
            self.active_tasks.append(
                {
                    "sequence": idx,
                    "action": task.get("action"),
                    "to_count": task.get("to_count"),
                    "_get_action_fn": self._get_action_fn,
                }
            )

    @staticmethod
    def _task_current_count(task: Dict[str, Any]) -> Optional[int]:
        """Best-effort progress count using the live Sharpy act when available."""
        act = task.get("_act")
        if act is None:
            if task.get("_completed", False):
                try:
                    return int(task.get("to_count") or 1)
                except (TypeError, ValueError):
                    return 1
            return None
        try:
            if hasattr(act, "current_active_base_count"):
                return int(act.current_active_base_count)
            get_unit_count = getattr(act, "get_unit_count", None)
            if callable(get_unit_count):
                return int(get_unit_count())
            unit_type = getattr(act, "unit_type", None)
            get_count = getattr(act, "get_count", None)
            if unit_type is not None and callable(get_count):
                # Match typical build/addon completion: ready + pending.
                return int(get_count(unit_type))
            if task.get("_completed", False):
                return int(task.get("to_count") or 1)
            # Tech / binary goals without a countable unit.
            return 0
        except Exception:
            return None

    def _serialise_active_tasks(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, task in enumerate(self.active_tasks, start=1):
            if task.get("_disabled", False):
                status = "failed"
            elif task.get("_execution_error"):
                status = "failed"
            elif task.get("_completed", False):
                status = "target_satisfied"
            else:
                status = "active_unsatisfied"
            try:
                to_count = int(task.get("to_count") or 0)
            except (TypeError, ValueError):
                to_count = 0
            current_count = self._task_current_count(task)
            item: Dict[str, Any] = {
                "sequence": task.get("sequence", idx),
                "action": task.get("action"),
                "to_count": to_count,
                "status": status,
            }
            if current_count is not None:
                item["current_count"] = int(current_count)
            if task.get("_error"):
                item["error"] = task.get("_error")
            elif task.get("_execution_error"):
                item["error"] = task.get("_execution_error")
            out.append(item)
        return out

    @staticmethod
    def _obs_unit_count(obs_snapshot: Optional[Dict[str, Any]], unit_name: str) -> int:
        if not obs_snapshot:
            return 0
        production = obs_snapshot.get("production", {}) or {}
        if production:
            completed = production.get("completed", {}) or {}
            pending = production.get("under_construction", {}) or {}
        else:
            own_forces = obs_snapshot.get("own_forces", {}) or {}
            completed = own_forces.get("completed", {}) or {}
            pending = own_forces.get("under_construction", {}) or {}
        target = unit_name.upper()

        def _count(source: Dict[str, Any]) -> int:
            total = 0
            for key, value in source.items():
                if str(key).upper() == target:
                    try:
                        total += int(value)
                    except (TypeError, ValueError):
                        pass
            return total

        return _count(completed) + _count(pending)

    def _ensure_addon_parent_tasks(
        self,
        tasks: List[Dict[str, Any]],
        obs_snapshot: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        addon_requirements = {
            "build_barracks_techlab": ("build_barracks", "BARRACKS"),
            "build_barracks_reactor": ("build_barracks", "BARRACKS"),
            "build_factory_techlab": ("build_factory", "FACTORY"),
            "build_factory_reactor": ("build_factory", "FACTORY"),
            "build_starport_techlab": ("build_starport", "STARPORT"),
            "build_starport_reactor": ("build_starport", "STARPORT"),
        }
        if not tasks:
            return tasks
        planned_parent_counts: Dict[str, int] = {}
        for task in tasks:
            action = task.get("action")
            try:
                to_count = int(task.get("to_count") or 0)
            except (TypeError, ValueError):
                to_count = 0
            if isinstance(action, str):
                planned_parent_counts[action] = max(
                    planned_parent_counts.get(action, 0), to_count
                )
        parent_inserts: Dict[str, Dict[str, Any]] = {}
        for task in tasks:
            action = task.get("action")
            req = addon_requirements.get(
                action) if isinstance(action, str) else None
            if not req:
                continue
            parent_action, unit_name = req
            have = self._obs_unit_count(obs_snapshot, unit_name)
            planned = planned_parent_counts.get(parent_action, 0)
            need = max(have, planned, 1)
            if have < need and planned < need:
                parent_inserts[parent_action] = {
                    "action": parent_action,
                    "to_count": need,
                }
        if not parent_inserts:
            return tasks
        return list(parent_inserts.values()) + list(tasks)

    # ------------------------------------------------------------------
    # chat / logging
    # ------------------------------------------------------------------

    async def on_chat(self, *args):
        if len(args) == 1:
            player_id, message = None, args[0]
        elif len(args) >= 2:
            player_id, message = args[0], args[1]
        else:
            return
        self._handle_chat_message(player_id=player_id, message=message)

    def _process_current_chat_messages(self) -> None:
        chat_messages = getattr(getattr(self, "state", None), "chat", None)
        if not chat_messages:
            return
        for item in chat_messages:
            player_id = getattr(item, "player_id", None)
            message = getattr(item, "message", "")
            key = (player_id, message)
            if key in self._seen_chat_messages:
                continue
            self._seen_chat_messages.add(key)
            self._handle_chat_message(player_id=player_id, message=message)

    def _handle_chat_message(self, player_id: Optional[int], message: str) -> None:
        if player_id is not None and player_id == getattr(self, "player_id", None):
            return
        if not _is_enemy_gg_message(message):
            return
        self.enemy_said_gg = True
        self.enemy_gg_message = message

    def _record_llm_interaction(self, record: Dict[str, Any]) -> None:
        recorder = getattr(self, "llm_observation_recorder", None)
        if recorder is None:
            return
        append_func = getattr(recorder, "record_llm_interaction", None)
        if append_func is None:
            return
        try:
            append_func(record)
        except Exception as exc:
            logger.warning("failed to record interaction: %s", exc)

    def _emit(self, message: str, *args: Any) -> None:
        line = "[Commander] " + (message % args if args else message)
        try:
            self.knowledge.print(line, stats=False)
            return
        except Exception:
            pass
        logger.info("%s", line)
