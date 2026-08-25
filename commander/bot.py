"""Single-agent Commander bot: strategy.md + model-authored wake events."""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import re
import shutil
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
from commander.tool_selection import select_tools_for_strategy
from commander.tools import NON_MACRO_TOOL_NAMES
from commander.retreat_policy import DEFAULT_RETREAT_RATIO
from commander.wake_events import (
    FALLBACK_DELAY_SECONDS,
    build_trigger_hint,
    build_wake_snapshot,
    evaluate_wake_event,
    fallback_wake_event,
    list_satisfied_wake_conditions,
    rising_edge,
)

logger = logging.getLogger("commander.bot")

_RACE_MAP = {
    "terran": Race.Terran,
    "zerg": Race.Zerg,
    "protoss": Race.Protoss,
    "random": Race.Random,
}

# Overlay used by evolution so candidates are not stored under skills/.
_STRATEGY_ROOT_ENV = "SC2_STRATEGY_ROOT"

# Legacy multi-agent folder names → current skills/<race>/<name> directories.
_STRATEGY_FOLDER_ALIASES = {
    "early_marine": "marine",
    "mid_tank": "tank",
    "late_battlecruiser": "battlecruiser",
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
        self.strategy_hash: str = ""
        self.active_tasks: List[Dict[str, Any]] = []
        self.commander_army_policy = None
        self._last_army_summary: Dict[str, Any] = {}
        # Set by the combat act when the auto-retreat machine intercepts a
        # command; consumed by _poll_wake_trigger for a proactive wake.
        self.commander_retreat_wake_pending: Optional[Dict[str, Any]] = None
        # Snapshot of the last Commander tool decision for the next observation.
        self.llm_previous_decision: Optional[Dict[str, Any]] = None
        self._last_decision_time: float = -self.WAKE_COOLDOWN
        self._decision_count: int = 0
        self._map_topology_text: Optional[str] = None
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
        self._expand_action_dependencies_fn: Optional[Callable] = None
        self._full_action_space: Optional[Dict[str, str]] = None
        self._action_space_cache: Optional[Dict[str, str]] = None
        self._strategy_raw: str = ""
        self._tool_selection: Optional[Dict[str, Any]] = None

        self.enemy_said_gg: bool = False
        self.enemy_gg_message: str = ""
        self._seen_chat_messages: Set[Tuple[Optional[int], str]] = set()

        if self.record_dir:
            self.llm_observation_recorder.output_folder = self.record_dir

    # ------------------------------------------------------------------
    # strategy documents and race adapters
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

    def _resolve_strategy_dir(self, folder: str) -> str:
        roots: list[str] = []
        overlay = str(os.environ.get(_STRATEGY_ROOT_ENV) or "").strip()
        if overlay:
            roots.append(os.path.normpath(overlay))
        roots.append(self._skills_race_dir)
        for root in roots:
            candidate = os.path.join(root, folder)
            if os.path.isfile(os.path.join(candidate, "strategy.md")):
                return candidate
        raise FileNotFoundError(
            f"strategy folder not found for {folder!r} (searched {roots})"
        )

    def _load_race_action_module(self) -> None:
        module_path = f"commander.races.{self.race_name}.actions"
        try:
            mod = importlib.import_module(module_path)
            self._get_action_fn = getattr(mod, "get_action", None)
            self._get_action_space_fn = getattr(mod, "get_action_space", None)
            self._expand_action_dependencies_fn = getattr(
                mod, "expand_action_dependencies", None
            )
        except ImportError:
            logger.warning("Race action module %s not found.", module_path)
            self._get_action_fn = None
            self._get_action_space_fn = None
            self._expand_action_dependencies_fn = None
        full = self._get_action_space_fn() if self._get_action_space_fn else {}
        self._full_action_space = dict(full)
        # Until strategy tool selection runs, expose nothing risky: keep full
        # only as a temporary fallback; selection replaces this in on_start.
        self._action_space_cache = dict(full)

    def _apply_forced_strategy(self, name: str) -> None:
        key = str(name or "").strip()
        folder = _STRATEGY_FOLDER_ALIASES.get(key.lower(), key)
        target_dir = self._resolve_strategy_dir(folder)
        md_path = os.path.join(target_dir, "strategy.md")
        if not os.path.isfile(md_path):
            raise FileNotFoundError(f"strategy.md not found: {md_path}")

        raw = ""
        detail = ""
        if os.path.isfile(md_path):
            with open(md_path, "r", encoding="utf-8") as handle:
                raw = handle.read()
            parsed = parse_strategy_document(raw)
            detail = parsed.get("detail") or raw.strip()

        self.selected_strategy = folder
        self.strategy_description = detail
        self._strategy_raw = raw.strip() or detail
        self.strategy_hash = hashlib.sha256(
            detail.encode("utf-8")
        ).hexdigest()[:16]
        self._emit(
            "force_strategy=%s description_chars=%d hash=%s",
            folder,
            len(detail),
            self.strategy_hash,
        )
        self._record_llm_interaction(
            {
                "game_time": 0.0,
                "trigger_reason": "strategy_forced",
                "forced_strategy": name,
                "strategy_id": name,
                "strategy_hash": self.strategy_hash,
                "strategy_description": detail,
            }
        )
        self._copy_strategy_to_record_dir(md_path)

    def _apply_no_skill_strategy(self) -> None:
        """No Skill baseline: empty strategy.md text; full race tool catalog later."""
        self.selected_strategy = "none"
        self.strategy_description = ""
        self._strategy_raw = ""
        self.strategy_hash = "none"
        self._emit("force_strategy=none (no strategy.md; full tool catalog)")
        self._record_llm_interaction(
            {
                "game_time": 0.0,
                "trigger_reason": "strategy_none",
                "forced_strategy": "none",
                "strategy_id": "none",
                "strategy_hash": self.strategy_hash,
                "strategy_description": "",
            }
        )

    def _copy_strategy_to_record_dir(self, md_path: str) -> None:
        if not self.record_dir or not os.path.isfile(md_path):
            return
        os.makedirs(self.record_dir, exist_ok=True)
        shutil.copy2(md_path, os.path.join(self.record_dir, "strategy.md"))

    def _select_strategy_tools(self) -> None:
        """Once per match: semantic selection plus dependency expansion.

        No Skill (``selected_strategy == \"none\"``) skips the selection LLM and
        exposes the full race action catalog so the action interface matches
        With-Skill games aside from the missing strategy text.
        """
        full = dict(self._full_action_space or self._action_space_cache or {})
        if not full:
            logger.warning("tool selection skipped: empty action space")
            return
        no_skill = (self.selected_strategy or "").strip().lower() == "none"
        strategy_text = self._strategy_raw or self.strategy_description
        outcome = select_tools_for_strategy(
            strategy_text=strategy_text,
            full_action_space=full,
            model_key=self._resolved_model_key(),
            use_llm=not no_skill,
            dependency_resolver=self._expand_action_dependencies_fn,
        )
        if no_skill:
            # select_tools_for_strategy already falls back to full when use_llm
            # is False; stamp an explicit reason for match_info / JSON.
            outcome = dict(outcome)
            outcome["fallback_used"] = True
            outcome["fallback_reason"] = "no_skill_full_catalog"
            outcome["action_space"] = dict(full)
            outcome["selected_tools"] = sorted(full)
            outcome["selected_tool_count"] = len(full)
            outcome["semantic_tools"] = []
            outcome["dependency_tools"] = []
        self._tool_selection = outcome
        selected = outcome.get("action_space") or {}
        if selected:
            self._action_space_cache = dict(selected)
        selected_names = list(outcome.get("selected_tools") or sorted(selected))
        summary = (
            "tool_selection selected=%d/%d semantic=%d dependencies=%d "
            "fallback=%s err=%s"
            % (
                int(outcome.get("selected_tool_count") or 0),
                int(outcome.get("full_tool_count") or 0),
                len(outcome.get("semantic_tools") or []),
                len(outcome.get("dependency_tools") or []),
                outcome.get("fallback_reason") or "no",
                outcome.get("llm_error") or "ok",
            )
        )
        # Always log: early on_start knowledge.print is easy to miss in the UI.
        logger.info("[Commander] %s", summary)
        logger.info("[Commander]   selected_tools=%s", ",".join(selected_names))
        self._emit("%s", summary)
        self._emit("  selected_tools=%s", ",".join(selected_names))
        if outcome.get("dependency_tools"):
            self._emit(
                "  dependency_tools=%s",
                ",".join(outcome.get("dependency_tools") or []),
            )
        self._append_tool_selection_match_info(outcome, selected_names)
        self._record_llm_interaction(
            {
                "game_time": 0.0,
                "trigger_reason": "strategy_tool_selection",
                "strategy_id": self.selected_strategy,
                "strategy_hash": self.strategy_hash,
                "selected_tools": selected_names,
                "baseline_tools": list(outcome.get("baseline_tools") or []),
                "semantic_tools": list(outcome.get("semantic_tools") or []),
                "dependency_tools": list(outcome.get("dependency_tools") or []),
                "fallback_used": bool(outcome.get("fallback_used")),
                "fallback_reason": outcome.get("fallback_reason") or "",
                "dependency_error": outcome.get("dependency_error") or "",
                "full_tool_count": outcome.get("full_tool_count"),
                "selected_tool_count": outcome.get("selected_tool_count"),
                "llm_error": outcome.get("llm_error") or "",
                "llm_content": outcome.get("llm_content") or "",
                "latency_seconds": outcome.get("llm_latency_seconds"),
                "messages": outcome.get("messages") or [],
            }
        )

    def _append_tool_selection_match_info(
        self, outcome: Dict[str, Any], selected_names: List[str]
    ) -> None:
        if not self.record_dir:
            return
        path = os.path.join(self.record_dir, "match_info.txt")
        army = [n for n in selected_names if n in NON_MACRO_TOOL_NAMES]
        macro = [n for n in selected_names if n not in NON_MACRO_TOOL_NAMES]
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("\n")
                handle.write(
                    f"tool_selection:     {outcome.get('selected_tool_count')}/"
                    f"{outcome.get('full_tool_count')} "
                    f"(semantic={len(outcome.get('semantic_tools') or [])}, "
                    f"dependencies={len(outcome.get('dependency_tools') or [])}, "
                    f"fallback={outcome.get('fallback_reason') or 'no'}, "
                    f"err={outcome.get('llm_error') or 'ok'})\n"
                )
                handle.write("army_tools:         " + ", ".join(army) + "\n")
                handle.write("macro_tools:        " + ", ".join(macro) + "\n")
                semantic = list(outcome.get("semantic_tools") or [])
                if semantic:
                    handle.write(
                        "semantic_tools:     " + ", ".join(semantic) + "\n"
                    )
                dependencies = list(outcome.get("dependency_tools") or [])
                if dependencies:
                    handle.write(
                        "dependency_tools:   " + ", ".join(dependencies) + "\n"
                    )
        except Exception as exc:
            logger.warning("failed to append tool selection to match_info: %s", exc)

    def _load_race_tactics(self) -> BuildOrder:
        module_path = f"commander.races.{self.race_name}.tactics"
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            logger.warning(
                "Cannot import %s; using EmptyTactics.", module_path)
            return EmptyTactics()
        factory = getattr(mod, "create_tactics", None)
        if callable(factory):
            try:
                tactics = factory()
                if isinstance(tactics, BuildOrder):
                    return tactics
                logger.warning(
                    "%s.create_tactics() returned %s; using EmptyTactics.",
                    module_path,
                    type(tactics).__name__,
                )
            except Exception as exc:
                logger.warning("Failed to create tactics from %s: %s", module_path, exc)
        return EmptyTactics()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def on_start(self):
        self._load_race_action_module()
        # ``force_strategy is None`` means explicit No Skill (CLI ``none``),
        # not "pick a default folder". With-Skill callers pass marine/tank/…
        if self.force_strategy:
            self._apply_forced_strategy(self.force_strategy)
        else:
            self._apply_no_skill_strategy()
        await super().on_start()
        self.zone_manager = self.knowledge.get_required_manager(IZoneManager)
        self.llm_observation_recorder.interval_seconds = self.OBS_RECORD_INTERVAL
        if self.record_dir:
            self.llm_observation_recorder.output_folder = self.record_dir
        # After Sharpy start so [Commander] lines show in the match log/UI.
        self._select_strategy_tools()

    async def pre_step_execute(self):
        self._process_current_chat_messages()
        if self._decision_count <= 0:
            self._run_decision(trigger_reason="commander_bootstrap")
            return
        if self.time - self._last_decision_time < self.WAKE_COOLDOWN:
            return
        reason = self._poll_wake_trigger()
        if reason:
            self._run_decision(
                trigger_reason=reason["reason"],
                fired_conditions=reason.get("fired_conditions") or [],
            )

    async def create_plan(self) -> BuildOrder:
        return BuildOrder(
            [
                ActOngoingMacroTasks(active_tasks_ref=self.active_tasks),
                self._load_race_tactics(),
                ForceFinishEnemyOnGG(lambda ai: getattr(
                    ai, "enemy_said_gg", False)),
            ]
        )

    # ------------------------------------------------------------------
    # decision
    # ------------------------------------------------------------------

    def _resolved_model_key(self) -> str:
        return (self.commander_model_key or "").strip()

    def _poll_wake_trigger(self) -> Optional[Dict[str, Any]]:
        """Return trigger payload on rising-edge wake or deadline fuse."""
        pending = self.commander_retreat_wake_pending
        if pending is not None:
            self.commander_retreat_wake_pending = None
            return {
                "reason": "auto_retreat_triggered",
                "fired_conditions": [
                    f"{pending.get('state')}:{pending.get('detail')}"
                ],
            }
        # The wake snapshot runs the full army-state collection, which is the
        # single most expensive per-frame operation. Evaluating conditions at
        # ~3Hz (every 8 loops) costs nothing decision-wise: a wake fires at
        # most ~0.4s (game time) later.
        game_loop = int(getattr(getattr(self, "state", None), "game_loop", 0) or 0)
        if game_loop % 8 == 0:
            snapshot = self._build_wake_snapshot()
            event_ok = (
                evaluate_wake_event(self._wake_event, snapshot)
                if self._wake_event
                else False
            )
            event_edge = rising_edge(event_ok, self._wake_prev_satisfied)
            self._wake_prev_satisfied = event_ok
            if event_edge:
                fired = list_satisfied_wake_conditions(self._wake_event, snapshot)
                if self._wake_is_fallback:
                    return {
                        "reason": "wake_fallback_timeout",
                        "fired_conditions": fired or ["runtime_deadline_fuse"],
                    }
                return {
                    "reason": "wake_event",
                    "fired_conditions": fired,
                }
        # Always-on now+N fuse: fires once even if the model event stays false
        # (or stays sticky-true without a new rising edge).
        if (
            self._wake_deadline is not None
            and float(self.time) >= float(self._wake_deadline)
        ):
            self._wake_deadline = None
            return {
                "reason": "wake_fallback_timeout",
                "fired_conditions": ["runtime_deadline_fuse"],
            }
        return None

    def _build_wake_snapshot(self) -> Dict[str, Any]:
        cleanup_hint = ""
        army_state: Dict[str, Any] = {}
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
            except Exception as exc:
                logger.warning("wake snapshot army state failed: %s", exc)
                army_state = {}
        return build_wake_snapshot(
            time_seconds=float(self.time),
            supply_used=int(getattr(self, "supply_used", 0) or 0),
            supply_cap=int(getattr(self, "supply_cap", 0) or 0),
            own_unit_type_counts=army_state.get("own_unit_type_counts") or {},
            own_structure_counts=self._wake_structure_counts(),
            completed_upgrades=self._wake_completed_upgrades(),
            army_groups=army_state.get("army_groups") or [],
            available_zones=army_state.get("available_zones") or [],
            scan_ready=int(army_state.get("scan_ready") or 0) > 0,
            cleanup_hint_present="[Runtime Search-And-Destroy Hint]"
            in cleanup_hint,
            baseline_objective_status=self._wake_baseline_objective_status,
        )

    def _wake_structure_counts(self) -> Dict[str, int]:
        """Ready own structures only (cheap wake sampling)."""
        counts: Dict[str, int] = {}
        try:
            structures = list(getattr(self, "structures", None) or [])
        except Exception:
            return counts
        for unit in structures:
            try:
                if not getattr(unit, "is_ready", False):
                    continue
                name = getattr(getattr(unit, "type_id", None), "name", None)
                if not name:
                    continue
                counts[str(name)] = counts.get(str(name), 0) + 1
            except Exception:
                continue
        return counts

    def _wake_completed_upgrades(self) -> List[str]:
        """Completed upgrade names for wake sampling."""
        out: List[str] = []
        try:
            upgrades = getattr(getattr(self, "state", None), "upgrades", None) or []
        except Exception:
            return out
        for upgrade in upgrades:
            try:
                name = getattr(upgrade, "name", None) or str(upgrade)
            except Exception:
                continue
            if name:
                out.append(str(name))
        return out

    def _get_map_topology_text(self) -> str:
        """Static map topology for the system prompt; computed once per match."""
        if self._map_topology_text is not None:
            return self._map_topology_text
        text = ""
        try:
            from commander.combat_state import _zone_topology
            from commander.observation import format_map_topology

            zones = list(getattr(self.zone_manager, "expansion_zones", []) or [])
            if zones:
                topology = _zone_topology(
                    zones,
                    set(getattr(self.zone_manager, "gather_points", []) or []),
                )
                text = format_map_topology(topology)
        except Exception as exc:
            logger.warning("map topology build failed: %s", exc)
            text = ""
        self._map_topology_text = text
        return text

    def _run_decision(
        self,
        *,
        trigger_reason: str,
        fired_conditions: Optional[List[str]] = None,
    ) -> None:
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
        obs_text, full_obs, _view = self._capture_observation_bundle()
        cleanup_hint = self._cleanup_runtime_hint()
        trigger_hint = ""
        if trigger_reason != "commander_bootstrap":
            trigger_hint = build_trigger_hint(
                reason=trigger_reason,
                event=self._wake_event,
                fired_conditions=fired_conditions,
            )
        runtime_hint = "\n\n".join(
            part for part in (cleanup_hint, trigger_hint) if part
        )
        outcome = run_commander_decision(
            race=self.race_name,
            strategy_description=self.strategy_description,
            observation_text=obs_text,
            runtime_hint=runtime_hint,
            map_topology_text=self._get_map_topology_text(),
            action_space=action_space,
            model_key=model_key,
            full_observation=full_obs,
            ensure_addon_parents=self._ensure_addon_parent_tasks,
        )
        tasks = outcome["tasks"]
        policy = outcome["policy"]
        accepted = bool(outcome.get("accepted", True))
        if accepted:
            self._replace_active_tasks(tasks)
            self.commander_army_policy = policy
            self._last_army_summary = outcome["army_summary"]
        else:
            # Validation failed even after reflection: inherit the previous
            # decision rather than wiping macro tasks / army policy with the
            # empty parses from the rejected response.
            issues = list(outcome.get("issues") or [])
            if "decision_inherited_from_previous" not in issues:
                issues.append("decision_inherited_from_previous")
            outcome["issues"] = issues
            prev_decision = self.llm_previous_decision or {}
            tasks = [
                {"action": c.get("name"), "to_count": c.get("to_count")}
                for c in (prev_decision.get("macro_commands") or [])
                if isinstance(c, dict) and c.get("name")
            ]
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
        # Arm baselines before resampling so objective_status_became ignores
        # already-true stale states.
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
        # Reuse _last_army_summary so a rejected cycle reports the inherited
        # (still active) army commands instead of the rejected empty parse.
        army_summary = self._last_army_summary or {}
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
                    "retreat_ratio": c.get("retreat_ratio"),
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
            "t=%.1f reason=%s woken_by=%s mode=%s tools=%d macro=%d groups=%d "
            "wake_fallback=%s deadline=%.1f reflect=%s issues=%s err=%s",
            float(self.time),
            trigger_reason,
            "; ".join(fired_conditions or []) or "-",
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
                    f"(r{c.get('retreat_ratio') if c.get('retreat_ratio') is not None else f'{DEFAULT_RETREAT_RATIO}d'})"
                    for c in army["commands"]
                ),
            )
        else:
            # Applied commands this cycle (not observation army_groups).
            self._emit("  army=(no commands applied)")
        if army.get("scan_zone_id"):
            self._emit("  scan=%s", army["scan_zone_id"])
        if army.get("scout_zone_id"):
            self._emit("  scout=%s", army["scout_zone_id"])
        self._emit("  wake=%s deadline=%.1f", wake_event, float(self._wake_deadline))
        content = (outcome.get("content") or "").strip()
        if content:
            preview = content.replace("\n", " ")
            self._emit("  content=%s", preview)
        record: Dict[str, Any] = {
            "trigger_reason": trigger_reason,
            "woken_by": list(fired_conditions or []),
            "agent": "commander",
            "game_time": round(float(self.time), 1),
            "model_key": model_key,
            "tool_mode": outcome.get("tool_mode") or "json",
            "strategy_id": self.selected_strategy,
            "strategy_hash": self.strategy_hash,
            "text_observation": obs_text,
            "observation": full_obs,
            "runtime_hint": runtime_hint,
            "messages": outcome.get("messages") or [],
            "messages_transcript": outcome.get("messages_transcript") or [],
            "assistant_content": outcome.get("assistant_content")
            or outcome.get("content")
            or "",
            "content": outcome.get("content") or "",
            "tool_calls": outcome["tool_calls"],
            "macro_tasks": tasks,
            "army_policy": self._last_army_summary,
            "wake_event": wake_event,
            "wake_fallback": wake_fallback,
            "wake_armed_at": self._wake_armed_at,
            "wake_deadline": self._wake_deadline,
            "reflection_retries": outcome.get("reflection_retries") or 0,
            "reflection_issues": outcome.get("reflection_issues") or [],
            "reflection_rounds": outcome.get("reflection_rounds") or [],
            "accepted": bool(outcome.get("accepted", True)),
            "issues": outcome["issues"],
            "error": outcome["error"],
            "latency_seconds": outcome.get("latency_seconds"),
            "finish_reason": outcome.get("finish_reason"),
            "usage": outcome.get("usage") or {},
            "usage_total": outcome.get("usage_total") or {},
        }
        if outcome.get("raw_message") is not None:
            record["raw_message"] = outcome.get("raw_message")
        self._record_llm_interaction(record)

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

    def _capture_observation_bundle(self):
        recorder = getattr(self, "llm_observation_recorder", None)
        if recorder is None:
            return "(observation unavailable)", None, None
        try:
            return recorder.capture_observation_bundle()
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
