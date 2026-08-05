"""Single-agent Commander bot: strategy.md + tool_calls every 20s."""

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

    DECISION_INTERVAL: float = 20.0
    zone_manager: IZoneManager

    def __init__(
        self,
        race_name: str = "terran",
        instruct: str = "",
        commander_model_key: str = "",
        record_dir: str = "",
        *,
        force_strategy: Optional[str] = None,
        **legacy_kwargs: Any,
    ):
        super().__init__("SC2 Commander")
        # GameStarter may still pass coordinator/macro/translator keys.
        legacy_model = (
            legacy_kwargs.pop("macro_model_key", "")
            or legacy_kwargs.pop("coordinator_model_key", "")
            or legacy_kwargs.pop("translator_model_key", "")
            or ""
        )
        legacy_kwargs.pop("coordinator_model_key", None)
        legacy_kwargs.pop("macro_model_key", None)
        legacy_kwargs.pop("translator_model_key", None)
        if legacy_kwargs:
            logger.warning("Ignoring unexpected kwargs: %s", sorted(legacy_kwargs))

        self.race_name = race_name.strip().lower()
        self.instruct = (instruct or "").strip()
        self.commander_model_key = (
            commander_model_key or legacy_model or ""
        ).strip()
        # Aliases so GameStarter can still assign macro_model_key etc.
        self.macro_model_key = self.commander_model_key
        self.coordinator_model_key = self.commander_model_key
        self.translator_model_key = self.commander_model_key
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
        self._last_decision_time: float = -self.DECISION_INTERVAL
        self.llm_mid_execution_state: Dict[str, Any] = {
            "last_tasks": [],
            "last_update_game_time": None,
            "last_translation_issues": [],
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
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "skills")
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
            logger.warning("Cannot import %s; using EmptyTactics.", module_path)
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
                        logger.warning("Failed to instantiate %s: %s", attr_name, exc)
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
        self.llm_observation_recorder.interval_seconds = self.DECISION_INTERVAL
        if self.record_dir:
            self.llm_observation_recorder.output_folder = self.record_dir

    async def pre_step_execute(self):
        self._process_current_chat_messages()
        if self.time - self._last_decision_time >= self.DECISION_INTERVAL:
            self._last_decision_time = self.time
            self._run_decision(trigger_reason="commander_poll")

    async def create_plan(self) -> BuildOrder:
        return BuildOrder(
            [
                ActOngoingMacroTasks(active_tasks_ref=self.active_tasks),
                self._load_dynamic_tactics(),
                ForceFinishEnemyOnGG(lambda ai: getattr(ai, "enemy_said_gg", False)),
            ]
        )

    # ------------------------------------------------------------------
    # decision
    # ------------------------------------------------------------------

    def _resolved_model_key(self) -> str:
        return (
            (self.commander_model_key or "").strip()
            or (getattr(self, "macro_model_key", "") or "").strip()
            or (getattr(self, "coordinator_model_key", "") or "").strip()
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

        obs_text, full_obs, _view = self._capture_observation_bundle("top")
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
        self.llm_mid_execution_state = {
            "last_tasks": [
                f"{t['action']} to {t['to_count']}" for t in tasks
            ],
            "last_update_game_time": round(float(self.time), 1),
            "last_translation_issues": list(outcome["issues"]),
        }
        self._emit(
            "t=%.1f reason=%s mode=%s tools=%d macro=%d groups=%d issues=%s err=%s",
            float(self.time),
            trigger_reason,
            outcome.get("tool_mode") or "?",
            len(outcome["tool_calls"]),
            len(tasks),
            len(policy.commands),
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
        content = (outcome.get("content") or "").strip()
        if content:
            preview = content.replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:240] + "…"
            self._emit("  content=%s", preview)
        self._record_llm_interaction(
            {
                "trigger_reason": trigger_reason,
                "agent": "commander",
                "game_time": round(float(self.time), 1),
                "model_key": model_key,
                "tool_calls": outcome["tool_calls"],
                "macro_tasks": tasks,
                "army_policy": self._last_army_summary,
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

    def _capture_observation_bundle(self, view_type: str = "top"):
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
            item: Dict[str, Any] = {
                "sequence": task.get("sequence", idx),
                "action": task.get("action"),
                "to_count": task.get("to_count"),
                "status": status,
            }
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
            req = addon_requirements.get(action) if isinstance(action, str) else None
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
