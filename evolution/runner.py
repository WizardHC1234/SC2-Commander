from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from evol_agent import EvolAgent
from evol_agent.analysis.record_reader import find_record_jsons, is_completed_match_record
from evol_agent.core.checkpoint import PIPELINE_VERSION, load_checkpoint, stage_reached
from evol_agent.core.config import STRATEGY_ROOT_ENV, canonical_strategy_folder, resolve_skill_dir
from evol_agent.core.experiment_audit import audit_experiment
from evol_agent.core.types import EvolRunRequest, EvolRunResult
from evol_agent.optimization.snapshot import output_dir_for_strategy
from .feedback import (
    combine_batch_evidence,
    compare_batch_evidence,
    summarize_batch_evidence,
)
from .outcomes import (
    aggregate_outcomes,
    decide_candidate,
    posterior_probability_better,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIFFICULTIES = (
    "harder",
    "veryhard",
    "cheatvision",
    "cheatmoney",
    "cheatinsane",
)
HISTORY_FIELDS = (
    "strategy_style",
    "generation",
    "strategy",
    "parent",
    "difficulty",
    "wins",
    "draws",
    "losses",
    "games",
    "score",
    "win_rate",
    "mastered_levels",
    "curriculum_progress_score",
    "accepted",
    "batch",
)

_MECHANISM_RENAME_NOISE = {
    "a",
    "air",
    "and",
    "against",
    "balance",
    "count",
    "denial",
    "earlier",
    "early",
    "for",
    "from",
    "improve",
    "improved",
    "improvement",
    "increase",
    "increased",
    "late",
    "later",
    "matchup",
    "mechanism",
    "of",
    "package",
    "response",
    "strategy",
    "support",
    "the",
    "timing",
    "to",
    "unit",
    "units",
    "versus",
    "vs",
    "with",
}


def canonical_mechanism_signature(value: Any) -> str:
    """Collapse cosmetic family renames while retaining causal anchor terms."""
    tokens = re.findall(r"[a-z0-9]+", str(value or "").casefold())
    anchors = sorted(
        {
            token
            for token in tokens
            if len(token) > 1 and token not in _MECHANISM_RENAME_NOISE
        }
    )
    if anchors:
        return "_".join(anchors)
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")


@dataclass(frozen=True)
class EvolutionConfig:
    strategy: str
    commander_model: str
    evolution_model: str = ""
    race: str = "terran"
    enemy_race: str = "terran"
    enemy_build: str = "macro"
    map_name: str = "KairosJunctionLE"
    difficulties: tuple[str, ...] = DEFAULT_DIFFICULTIES
    matches_per_batch: int = 10
    candidate_matches: int = 10
    candidate_generation_retries: int = 3
    confirmation_matches: int = 0
    concurrency: int = 5
    mastery_score_threshold: float = 0.90
    analysis_batch_games: int = 10
    max_analysis_games_per_generation: int = 20
    max_generations_per_difficulty: int = 10
    max_total_generations: int = 50
    knowledge_mode: str = "enabled"
    bot_name: str = "commander"
    bot_instruct: str = ""
    real_time: bool = False
    baseline_batch_dir: str = ""

    def validate(self) -> None:
        if not self.strategy.strip():
            raise ValueError("strategy cannot be empty")
        if not self.commander_model.strip():
            raise ValueError("commander_model cannot be empty")
        if not self.difficulties:
            raise ValueError("at least one difficulty is required")
        if self.matches_per_batch <= 0 or self.concurrency <= 0:
            raise ValueError("matches_per_batch and concurrency must be positive")
        if self.candidate_matches <= 0:
            raise ValueError("candidate_matches must be positive")
        if self.candidate_generation_retries < 0:
            raise ValueError("candidate_generation_retries cannot be negative")
        if self.confirmation_matches < 0:
            raise ValueError("confirmation_matches cannot be negative")
        if not 0.0 <= self.mastery_score_threshold <= 1.0:
            raise ValueError("mastery_score_threshold must be between 0 and 1")
        if self.analysis_batch_games <= 0:
            raise ValueError("analysis_batch_games must be positive")
        if self.max_analysis_games_per_generation < self.analysis_batch_games:
            raise ValueError(
                "max_analysis_games_per_generation must be at least analysis_batch_games"
            )
        if self.max_generations_per_difficulty < 0:
            raise ValueError("max_generations_per_difficulty cannot be negative")
        if self.max_total_generations < 0:
            raise ValueError("max_total_generations cannot be negative")


@dataclass(frozen=True)
class BatchResult:
    name: str
    path: Path
    strategy: str
    difficulty: str
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def outcome_points(self) -> float:
        return self.wins + 0.5 * self.draws

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["score"] = self.score
        data["win_rate"] = self.win_rate
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchResult":
        return cls(
            name=str(data["name"]),
            path=Path(data["path"]),
            strategy=str(data["strategy"]),
            difficulty=str(data["difficulty"]),
            wins=int(data["wins"]),
            draws=int(data["draws"]),
            losses=int(data["losses"]),
        )


def combine_batch_results(first: BatchResult, second: BatchResult) -> BatchResult:
    if first.strategy != second.strategy or first.difficulty != second.difficulty:
        raise ValueError("cannot combine batches from different strategies or difficulties")
    return BatchResult(
        name=f"{first.name}+{second.name}",
        path=first.path,
        strategy=first.strategy,
        difficulty=first.difficulty,
        wins=first.wins + second.wins,
        draws=first.draws + second.draws,
        losses=first.losses + second.losses,
    )


def close_batch_results(champion: BatchResult, candidate: BatchResult) -> bool:
    """Return whether observed rates are within one result at the smaller sample."""
    smaller_sample = min(champion.games, candidate.games)
    if smaller_sample <= 0:
        return False
    return abs(candidate.score - champion.score) <= 1.0 / smaller_sample


def curriculum_progress_score(
    mastered_levels: int,
    current_win_rate: float,
    mastery_threshold: float,
) -> float:
    """Return verified progress through an ordered difficulty curriculum."""
    if mastery_threshold <= 0.0:
        partial_level = 1.0
    else:
        partial_level = min(
            max(float(current_win_rate), 0.0) / mastery_threshold,
            1.0,
        )
    return max(int(mastered_levels), 0) + partial_level


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := str(item).strip())]


def _safe_name(value: str, limit: int = 24) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "run"
    return text[:limit]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read_batch_result(
    batch_dir: Path,
    *,
    name: str,
    strategy: str,
    difficulty: str,
    expected_games: int,
) -> BatchResult:
    counts = {"victory": 0, "tie": 0, "defeat": 0}
    completed = 0
    for path in find_record_jsons(batch_dir):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not is_completed_match_record(data):
            continue
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        recorded_strategy = str(metadata.get("strategy_id") or "")
        if recorded_strategy and recorded_strategy != strategy:
            continue
        result = str(metadata.get("result") or "Defeat").strip().lower()
        if result == "draw":
            result = "tie"
        if result not in counts:
            result = "defeat"
        counts[result] += 1
        completed += 1
    if completed != expected_games:
        raise RuntimeError(
            f"batch {name} has {completed}/{expected_games} completed records; "
            "it cannot be scored"
        )
    return BatchResult(
        name=name,
        path=batch_dir,
        strategy=strategy,
        difficulty=difficulty,
        wins=counts["victory"],
        draws=counts["tie"],
        losses=counts["defeat"],
    )


def read_partial_batch_result(
    batch_dir: Path,
    *,
    name: str,
    strategy: str,
    difficulty: str,
) -> BatchResult | None:
    completed = completed_record_count(batch_dir, strategy=strategy)
    if not completed:
        return None
    return read_batch_result(
        batch_dir,
        name=name,
        strategy=strategy,
        difficulty=difficulty,
        expected_games=completed,
    )


def completed_record_count(batch_dir: Path, *, strategy: str) -> int:
    if not batch_dir.exists():
        return 0
    completed = 0
    for path in find_record_jsons(batch_dir):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not is_completed_match_record(data):
            continue
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        recorded_strategy = str(metadata.get("strategy_id") or "")
        if not recorded_strategy or recorded_strategy == strategy:
            completed += 1
    return completed


class EvolutionRunner:
    def __init__(
        self,
        config: EvolutionConfig,
        *,
        run_dir: Path | None = None,
        project_root: Path = PROJECT_ROOT,
        batch_executor: Callable[..., BatchResult] | None = None,
        candidate_generator: Callable[[str, BatchResult, list[Any]], EvolRunResult] | None = None,
        experiment_auditor: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.project_root = project_root.resolve()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = (run_dir or self.project_root / "evolution_runs" / config.strategy / stamp).resolve()
        self.run_id = _safe_name(self.run_dir.name, 15)
        self.state_path = self.run_dir / "state.json"
        self.history_path = self.run_dir / "history.csv"
        self._batch_executor = batch_executor
        self._candidate_generator = candidate_generator
        self._experiment_auditor = experiment_auditor
        self.strategies_dir = self.run_dir / "strategies"
        self.strategies_dir.mkdir(parents=True, exist_ok=True)

    def _new_state(self) -> dict[str, Any]:
        selection_protocol = (
            "confirmed_score_only_v2"
            if self.config.confirmation_matches
            else "score_only_v2"
        )
        return {
            "schema": "sc2_evolution.v3",
            "status": "running",
            "config": {**asdict(self.config), "difficulties": list(self.config.difficulties)},
            "selection_protocol": selection_protocol,
            "mastery_protocol": {
                "metric": "win_rate",
                "operator": ">=",
                "threshold": self.config.mastery_score_threshold,
            },
            "style": self.config.strategy,
            "champion": self.config.strategy,
            "search_parent": self.config.strategy,
            "search_parent_batch": None,
            "inconclusive_streak": 0,
            "difficulty_index": 0,
            "difficulty": self.config.difficulties[0],
            "mastered_difficulties": [],
            "generation": 0,
            "difficulty_generation": 0,
            "games_used": 0,
            "champion_batch": None,
            "champion_baseline": None,
            "pending_candidate": None,
            "experiment_history": [],
            "mechanism_ledger": [],
            "candidate_generation_failures": [],
            "mechanism_policy_rejections": [],
            "candidate_resume_dir": None,
            "analysis_checkpoints": {},
            "evidence_pool": {},
            "updated_at": datetime.now().isoformat(),
        }

    def load_or_create_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            saved = state.get("config") or {}
            current = {**asdict(self.config), "difficulties": list(self.config.difficulties)}
            config_changed = False
            legacy_candidate_matches = saved.get("candidate_max_matches")
            for obsolete in (
                "candidate_initial_matches",
                "candidate_max_matches",
                "candidate_step_matches",
                "promotion_probability_threshold",
            ):
                if obsolete in saved:
                    saved.pop(obsolete)
                    config_changed = True
            if "candidate_matches" not in saved:
                saved["candidate_matches"] = int(
                    legacy_candidate_matches or current["candidate_matches"]
                )
                config_changed = True
            if "candidate_generation_retries" not in saved:
                saved["candidate_generation_retries"] = current[
                    "candidate_generation_retries"
                ]
                config_changed = True
            if "max_total_generations" not in saved:
                saved["max_total_generations"] = current["max_total_generations"]
                config_changed = True
            if "mastery_score_threshold" not in saved:
                saved["mastery_score_threshold"] = current["mastery_score_threshold"]
                config_changed = True
            for obsolete in (
                "pass_score",
                "max_generations",
                "candidate_accept_probability",
                "candidate_reject_probability",
            ):
                if obsolete in saved:
                    saved.pop(obsolete)
                    config_changed = True
            for key in (
                "baseline_batch_dir",
                "analysis_batch_games",
                "max_analysis_games_per_generation",
                "max_generations_per_difficulty",
                "confirmation_matches",
            ):
                if key not in saved:
                    saved[key] = current[key]
                    config_changed = True
            if saved != current:
                raise ValueError("resume configuration does not match state.json")
            state["config"] = saved
            changed = config_changed
            changed = self._migrate_experiment_history(state) or changed
            changed = self._backfill_experiment_evidence(state) or changed
            changed = self._migrate_evidence_pool(state) or changed
            changed = self._migrate_lifecycle_state(state) or changed
            if changed:
                self._save_state(state)
            return state
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state = self._new_state()
        if self.config.baseline_batch_dir:
            baseline_path = Path(self.config.baseline_batch_dir).expanduser().resolve()
            baseline = read_batch_result(
                baseline_path,
                name=baseline_path.name,
                strategy=self.config.strategy,
                difficulty=self.config.difficulties[0],
                expected_games=self.config.matches_per_batch,
            )
            state["champion_batch"] = baseline.to_dict()
            self._sync_champion_baseline(state, baseline)
            self._sync_search_parent(state, self.config.strategy, baseline)
            self._register_evidence(state, baseline)
            self._sync_games_used(state)
            self._append_history(state=state, batch=baseline, parent="", accepted=True)
        self._save_state(state)
        return state

    def _pool_entries(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
        strategy: str,
    ) -> list[dict[str, Any]]:
        pool = state.setdefault("evidence_pool", {})
        by_difficulty = pool.setdefault(difficulty, {})
        entries = by_difficulty.setdefault(strategy, [])
        return entries if isinstance(entries, list) else []

    def _register_evidence(self, state: dict[str, Any], batch: BatchResult) -> bool:
        entries = self._pool_entries(
            state,
            difficulty=batch.difficulty,
            strategy=batch.strategy,
        )
        value = batch.to_dict()
        key = str(batch.path.resolve())
        for index, existing in enumerate(entries):
            if str(Path(str(existing.get("path") or "")).resolve()) == key:
                if existing != value:
                    entries[index] = value
                    return True
                return False
        entries.append(value)
        return True

    def _evidence_batches(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
        strategy: str,
    ) -> list[BatchResult]:
        return [
            BatchResult.from_dict(item)
            for item in self._pool_entries(
                state,
                difficulty=difficulty,
                strategy=strategy,
            )
            if isinstance(item, dict)
        ]

    def _aggregate_evidence(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
        strategy: str,
    ) -> BatchResult:
        batches = self._evidence_batches(
            state,
            difficulty=difficulty,
            strategy=strategy,
        )
        if not batches:
            raise RuntimeError(f"no outcome evidence for {strategy} at {difficulty}")
        total = batches[0]
        for batch in batches[1:]:
            total = combine_batch_results(total, batch)
        return total

    def _sync_games_used(self, state: dict[str, Any]) -> None:
        seen: set[str] = set()
        games = 0
        pool = state.get("evidence_pool") or {}
        for by_strategy in pool.values():
            if not isinstance(by_strategy, dict):
                continue
            for entries in by_strategy.values():
                for item in entries if isinstance(entries, list) else []:
                    if not isinstance(item, dict):
                        continue
                    key = str(Path(str(item.get("path") or "")).resolve())
                    if key in seen:
                        continue
                    seen.add(key)
                    games += int(item.get("wins") or 0)
                    games += int(item.get("draws") or 0)
                    games += int(item.get("losses") or 0)
        state["games_used"] = games

    def _migrate_evidence_pool(self, state: dict[str, Any]) -> bool:
        """Recover every completed outcome from legacy state and decision files."""
        changed = "evidence_pool" not in state
        state.setdefault("evidence_pool", {})

        def add_path(path_value: Any, strategy: str, difficulty: str) -> None:
            nonlocal changed
            path = Path(str(path_value or ""))
            if not path.is_dir():
                return
            batch = read_partial_batch_result(
                path,
                name=path.name,
                strategy=strategy,
                difficulty=difficulty,
            )
            if batch is not None:
                changed = self._register_evidence(state, batch) or changed

        champion_batch = state.get("champion_batch")
        if isinstance(champion_batch, dict):
            changed = self._register_evidence(
                state, BatchResult.from_dict(champion_batch)
            ) or changed
        pending = state.get("pending_candidate")
        if isinstance(pending, dict):
            for key in ("candidate_batch", "champion_confirmation", "candidate_confirmation"):
                item = pending.get(key)
                if isinstance(item, dict):
                    changed = self._register_evidence(
                        state, BatchResult.from_dict(item)
                    ) or changed

        for decision_path in sorted(self.run_dir.glob("generation_*/decision.json")):
            try:
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            difficulty = str(decision.get("difficulty") or "")
            add_path(decision.get("parent_batch"), str(decision.get("parent") or ""), difficulty)
            add_path(
                decision.get("candidate_batch"),
                str(decision.get("candidate") or ""),
                difficulty,
            )
            confirmation = decision.get("confirmation")
            if isinstance(confirmation, dict):
                for key, strategy_key in (
                    ("champion_batch", "parent"),
                    ("candidate_batch", "candidate"),
                ):
                    item = confirmation.get(key)
                    if isinstance(item, dict):
                        add_path(item.get("path"), str(decision.get(strategy_key) or ""), difficulty)

        # A Ctrl-C may leave a completed legacy confirmation batch on disk
        # before pending_candidate was updated. Recover that partial batch too.
        if isinstance(pending, dict):
            generation = int(state.get("generation") or 0)
            difficulty_index = int(state.get("difficulty_index") or 0)
            if difficulty_index < len(self.config.difficulties):
                difficulty = self.config.difficulties[difficulty_index]
                for role, strategy in (
                    ("champ_confirm", str(state.get("champion") or "")),
                    ("cand_confirm", str(pending.get("strategy") or "")),
                ):
                    name = self._batch_name(generation, role, difficulty)
                    add_path(self.project_root / "game_records" / name, strategy, difficulty)

        previous_games = int(state.get("games_used") or 0)
        self._sync_games_used(state)
        return changed or int(state.get("games_used") or 0) != previous_games

    def _migrate_experiment_history(self, state: dict[str, Any]) -> bool:
        changed = False
        if state.get("schema") != "sc2_evolution.v3":
            state["schema"] = "sc2_evolution.v3"
            changed = True
        history = state.get("experiment_history")
        if not isinstance(history, list):
            history = []
            changed = True
        legacy = state.get("failed_experiences")
        if isinstance(legacy, list) and legacy:
            existing = {
                (
                    int(item.get("generation") or -1),
                    str(item.get("candidate") or ""),
                )
                for item in history
                if isinstance(item, dict)
            }
            for item in legacy:
                if not isinstance(item, dict):
                    continue
                key = (
                    int(item.get("generation") or -1),
                    str(item.get("candidate") or ""),
                )
                if key in existing:
                    continue
                migrated = dict(item)
                migrated.setdefault("decision", "rejected")
                migrated["legacy"] = True
                history.append(migrated)
                existing.add(key)
                changed = True
        if "failed_experiences" in state:
            del state["failed_experiences"]
            changed = True
        for item in history:
            if not isinstance(item, dict):
                continue
            defaults = {
                "mechanism_prediction": {},
                "mechanism_evidence": [],
                "implementation_verdict": "unknown",
                "hypothesis_verdict": (
                    "supported"
                    if str(item.get("decision") or "") == "accepted"
                    else "inconclusive"
                ),
            }
            for key, value in defaults.items():
                if key not in item:
                    item[key] = value
                    changed = True
        state["experiment_history"] = history
        state.setdefault("candidate_generation_failures", [])
        state.setdefault("candidate_resume_dir", None)
        ledger = state.get("mechanism_ledger")
        if not isinstance(ledger, list):
            ledger = []
            changed = True
        existing_ledger = {
            (
                int(item.get("generation") or -1),
                str(item.get("candidate") or ""),
            )
            for item in ledger
            if isinstance(item, dict)
        }
        for item in history:
            if not isinstance(item, dict):
                continue
            key = (
                int(item.get("generation") or -1),
                str(item.get("candidate") or ""),
            )
            if key in existing_ledger:
                continue
            decision = str(item.get("decision") or "")
            implementation = str(item.get("implementation_verdict") or "unknown")
            valid_inconclusive = (
                decision == "inconclusive" and implementation != "execution_invalid"
            )
            ledger.append(
                {
                    "experiment_id": str(item.get("experiment_id") or ""),
                    "generation": key[0],
                    "difficulty": str(item.get("difficulty") or ""),
                    "mutation_parent": str(
                        item.get("mutation_parent") or item.get("parent") or ""
                    ),
                    "comparison_champion": str(
                        item.get("comparison_champion")
                        or item.get("champion")
                        or item.get("parent")
                        or ""
                    ),
                    "candidate": key[1],
                    "mechanism_family": str(
                        item.get("mechanism_family") or ""
                    ),
                    "inheritance": (
                        dict(item.get("inheritance"))
                        if isinstance(item.get("inheritance"), dict)
                        else {}
                    ),
                    "base_decision": str(item.get("base_decision") or decision),
                    "decision": decision,
                    "implementation_verdict": implementation,
                    "search_parent_before": str(
                        item.get("search_parent_before")
                        or item.get("mutation_parent")
                        or item.get("parent")
                        or ""
                    ),
                    "search_parent_after": str(
                        item.get("search_parent_after")
                        or (key[1] if decision == "accepted" or valid_inconclusive else "")
                    ),
                    "inconclusive_streak_before": int(
                        item.get("inconclusive_streak_before") or 0
                    ),
                    "inconclusive_streak_after": int(
                        item.get("inconclusive_streak_after")
                        or (1 if valid_inconclusive else 0)
                    ),
                }
            )
            existing_ledger.add(key)
            changed = True
        state["mechanism_ledger"] = ledger
        return changed

    def _append_experiment_history(
        self,
        state: dict[str, Any],
        record: dict[str, Any],
    ) -> bool:
        history = state.setdefault("experiment_history", [])
        generation = int(record.get("generation") or -1)
        candidate = str(record.get("candidate") or "")
        for item in history:
            if not isinstance(item, dict):
                continue
            if str(item.get("experiment_id") or "") and str(
                item.get("experiment_id") or ""
            ) == str(record.get("experiment_id") or ""):
                return False
            if (
                int(item.get("generation") or -1) == generation
                and str(item.get("candidate") or "") == candidate
            ):
                return False
        history.append(record)
        return True

    def _migrate_lifecycle_state(self, state: dict[str, Any]) -> bool:
        changed = False
        if not isinstance(state.get("mastered_difficulties"), list):
            backfill = list(self.config.difficulties[: int(state.get("difficulty_index") or 0)])
            state["mastered_difficulties"] = backfill
            changed = True
        if "difficulty_generation" not in state:
            state["difficulty_generation"] = 0
            changed = True
        index = int(state.get("difficulty_index") or 0)
        if index < len(self.config.difficulties):
            difficulty = self.config.difficulties[index]
            if state.get("difficulty") != difficulty:
                state["difficulty"] = difficulty
                changed = True
        champion_batch = state.get("champion_batch")
        if isinstance(champion_batch, dict) and not isinstance(
            state.get("champion_baseline"), dict
        ):
            self._sync_champion_baseline(state, BatchResult.from_dict(champion_batch))
            changed = True
        elif champion_batch is None and state.get("champion_baseline") is not None:
            state["champion_baseline"] = None
            changed = True
        selection_protocol = (
            "confirmed_score_only_v2"
            if self.config.confirmation_matches
            else "score_only_v2"
        )
        if state.get("selection_protocol") != selection_protocol:
            state["selection_protocol"] = selection_protocol
            changed = True
        mastery_protocol = {
            "metric": "win_rate",
            "operator": ">=",
            "threshold": self.config.mastery_score_threshold,
        }
        if state.get("mastery_protocol") != mastery_protocol:
            state["mastery_protocol"] = mastery_protocol
            changed = True
        if "search_parent" not in state:
            search_parent = str(state.get("champion") or self.config.strategy)
            streak = 0
            difficulty = str(state.get("difficulty") or "")
            history = [
                item
                for item in (state.get("experiment_history") or [])
                if isinstance(item, dict)
                and str(item.get("difficulty") or "") in {"", difficulty}
            ]
            if history:
                latest = history[-1]
                if (
                    str(latest.get("decision") or "") == "inconclusive"
                    and str(latest.get("implementation_verdict") or "")
                    != "execution_invalid"
                    and str(latest.get("candidate") or "").strip()
                ):
                    search_parent = str(latest["candidate"])
                    streak = 1
            state["search_parent"] = search_parent
            state["inconclusive_streak"] = streak
            changed = True
            pending = state.get("pending_candidate")
            if isinstance(pending, dict) and str(
                pending.get("mutation_parent") or ""
            ) != search_parent:
                # Legacy pending candidates were always generated from Champion.
                # Once an inconclusive child becomes the search parent, that old
                # candidate no longer has the required inheritance lineage.
                state["pending_candidate"] = None
                changed = True
        if "inconclusive_streak" not in state:
            state["inconclusive_streak"] = 0
            changed = True
        resume_dir = str(state.get("candidate_resume_dir") or "").strip()
        if (
            state.get("status") == "evol_agent_failed"
            and state.get("pending_candidate") is None
            and resume_dir
            and Path(resume_dir).is_dir()
        ):
            # Candidate generation failures preserve an analysis-complete
            # checkpoint. A later invocation should retry only optimization
            # instead of remaining permanently stopped in the failed status.
            state["status"] = "running"
            changed = True
        search_parent = str(state.get("search_parent") or state.get("champion") or "")
        search_parent_batch = state.get("search_parent_batch")
        if search_parent and not isinstance(search_parent_batch, dict):
            try:
                aggregate = self._aggregate_evidence(
                    state,
                    difficulty=str(state.get("difficulty") or ""),
                    strategy=search_parent,
                )
            except RuntimeError:
                aggregate = None
            if aggregate is not None:
                state["search_parent_batch"] = aggregate.to_dict()
                changed = True
            elif search_parent == str(state.get("champion") or "") and isinstance(
                state.get("champion_batch"), dict
            ):
                state["search_parent_batch"] = dict(state["champion_batch"])
                changed = True
        if not isinstance(state.get("mechanism_ledger"), list):
            state["mechanism_ledger"] = []
            changed = True
        if not isinstance(state.get("mechanism_policy_rejections"), list):
            state["mechanism_policy_rejections"] = []
            changed = True
        if not isinstance(state.get("analysis_checkpoints"), dict):
            state["analysis_checkpoints"] = {}
            changed = True
        return changed

    def _remember_analysis_checkpoint(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
        strategy: str,
        checkpoint_dir: Path | None,
    ) -> None:
        if checkpoint_dir is None:
            return
        path = checkpoint_dir.resolve()
        if not (path / "analysis_checkpoint.json").is_file():
            return
        checkpoints = state.setdefault("analysis_checkpoints", {})
        by_difficulty = checkpoints.setdefault(difficulty, {})
        by_difficulty[strategy] = str(path)

    def _find_analysis_seed_checkpoint(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
        strategy: str,
        record_paths: list[Path],
    ) -> Path | None:
        current_records = {str(path.resolve()) for path in record_paths}
        if not current_records:
            return None
        candidates: list[Path] = []
        remembered = state.get("analysis_checkpoints")
        if isinstance(remembered, dict):
            by_difficulty = remembered.get(difficulty)
            if isinstance(by_difficulty, dict):
                value = str(by_difficulty.get(strategy) or "").strip()
                if value:
                    candidates.append(Path(value))
        log_root = self.project_root / "evol_agent" / "logs" / strategy
        if log_root.is_dir():
            candidates.extend(path for path in log_root.iterdir() if path.is_dir())

        compatible: list[tuple[int, int, Path]] = []
        seen: set[str] = set()
        for path in candidates:
            resolved = path.resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            try:
                checkpoint = load_checkpoint(resolved)
            except (OSError, ValueError):
                continue
            meta = checkpoint.meta
            if str(meta.get("pipeline_version") or "") != PIPELINE_VERSION:
                continue
            if not stage_reached(checkpoint.stage, "analysis_complete"):
                continue
            if str(meta.get("strategy_name") or "") != strategy:
                continue
            if str(meta.get("race") or "").lower() != self.config.race.lower():
                continue
            if str(meta.get("knowledge_mode") or "") != self.config.knowledge_mode:
                continue
            seed_records = {
                str(Path(item).resolve())
                for item in (meta.get("record_files") or [])
                if str(item).strip()
            }
            if not seed_records or not seed_records.issubset(current_records):
                continue
            try:
                modified = (resolved / "checkpoint.json").stat().st_mtime_ns
            except OSError:
                modified = 0
            compatible.append((len(seed_records), modified, resolved))
        if not compatible:
            return None
        compatible.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return compatible[0][2]

    def _find_resumable_analysis_checkpoint(
        self,
        *,
        strategy: str,
        record_paths: list[Path],
    ) -> Path | None:
        current_records = {str(path.resolve()) for path in record_paths}
        log_root = self.project_root / "evol_agent" / "logs" / strategy
        if not current_records or not log_root.is_dir():
            return None
        resumable: list[tuple[int, Path]] = []
        for path in log_root.iterdir():
            if not path.is_dir():
                continue
            try:
                checkpoint = load_checkpoint(path)
            except (OSError, ValueError):
                continue
            meta = checkpoint.meta
            if str(meta.get("pipeline_version") or "") != PIPELINE_VERSION:
                continue
            if not stage_reached(checkpoint.stage, "match_summaries"):
                continue
            if stage_reached(checkpoint.stage, "candidate"):
                continue
            if stage_reached(checkpoint.stage, "analysis_complete"):
                try:
                    completed_analysis = json.loads(
                        (path / "analysis.json").read_text(encoding="utf-8-sig")
                    )
                except (OSError, ValueError):
                    completed_analysis = {}
                completed_action = str(
                    completed_analysis.get("next_action")
                    if isinstance(completed_analysis, dict)
                    else ""
                ).strip()
                # Only a completed proposal has unfinished optimization work.
                # Terminal analysis decisions must be reconsidered on a later
                # invocation so prompt/validator/runtime repairs can take effect.
                if completed_action and completed_action != "propose_strategy_patch":
                    continue
            if str(meta.get("strategy_name") or "") != strategy:
                continue
            if str(meta.get("race") or "").lower() != self.config.race.lower():
                continue
            if str(meta.get("knowledge_mode") or "") != self.config.knowledge_mode:
                continue
            checkpoint_records = {
                str(Path(item).resolve())
                for item in (meta.get("record_files") or [])
                if str(item).strip()
            }
            if checkpoint_records != current_records:
                continue
            try:
                modified = (path / "checkpoint.json").stat().st_mtime_ns
            except OSError:
                modified = 0
            resumable.append((modified, path.resolve()))
        if not resumable:
            return None
        resumable.sort(key=lambda item: item[0], reverse=True)
        return resumable[0][1]

    def _compact_champion_baseline(self, batch: BatchResult) -> dict[str, Any]:
        return {
            "strategy": batch.strategy,
            "difficulty": batch.difficulty,
            "wins": batch.wins,
            "draws": batch.draws,
            "losses": batch.losses,
            "games": batch.games,
            "score": batch.score,
        }

    def _sync_champion_baseline(
        self,
        state: dict[str, Any],
        batch: BatchResult | None,
    ) -> None:
        if batch is None:
            state["champion_batch"] = None
            state["champion_baseline"] = None
            return
        state["champion_batch"] = batch.to_dict()
        state["champion_baseline"] = self._compact_champion_baseline(batch)

    def _sync_search_parent(
        self,
        state: dict[str, Any],
        strategy: str,
        batch: BatchResult | None,
    ) -> None:
        state["search_parent"] = strategy
        state["search_parent_batch"] = batch.to_dict() if batch is not None else None

    def _current_difficulty(self, state: dict[str, Any]) -> str | None:
        index = int(state.get("difficulty_index") or 0)
        if index < 0 or index >= len(self.config.difficulties):
            return None
        return self.config.difficulties[index]

    def _is_mastered(self, batch: BatchResult) -> bool:
        return batch.win_rate >= self.config.mastery_score_threshold

    def _analysis_games(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
        strategy: str,
    ) -> int:
        return sum(
            batch.games
            for batch in self._evidence_batches(
                state, difficulty=difficulty, strategy=strategy
            )
        )

    def _experiment_id(
        self,
        *,
        style: str,
        generation: int,
        difficulty: str,
        candidate: str,
    ) -> str:
        return f"{style}:g{generation:03d}:{difficulty}:{candidate}"

    def _complete_curriculum(self, state: dict[str, Any]) -> None:
        state["status"] = "completed"
        state["completion_reason"] = "curriculum_mastered"

    def _reset_generation_local_analysis_state(self, state: dict[str, Any]) -> None:
        state["pending_candidate"] = None
        state.pop("last_agent_decision", None)

    def _prior_experiences(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
    ) -> list[dict[str, Any]]:
        history = [
            item
            for item in (state.get("experiment_history") or [])
            if isinstance(item, dict)
        ]
        related = [
            item
            for item in history
            if str(item.get("difficulty") or "") in {"", difficulty}
        ]
        blocked = self._blocked_mechanism_families(state, difficulty=difficulty)
        policy_rejections = [
            item
            for item in (state.get("mechanism_policy_rejections") or [])
            if isinstance(item, dict)
            and str(item.get("difficulty") or "") in {"", difficulty}
        ]
        policy = []
        if blocked:
            policy.append(
                {
                    "kind": "mechanism_search_policy",
                    "blocked_mechanism_families": blocked,
                    "rule": (
                        "Do not propose a blocked family or a materially equivalent "
                        "renaming. Choose a different causal mechanism."
                    ),
                }
            )
        return related + policy_rejections + policy

    def _blocked_mechanism_families(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
    ) -> dict[str, str]:
        attempts: dict[str, int] = {}
        representative: dict[str, str] = {}
        blocked: dict[str, str] = {}
        for item in state.get("experiment_history") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("difficulty") or "") not in {"", difficulty}:
                continue
            family = str(item.get("mechanism_family") or "").strip().lower()
            if not family:
                continue
            signature = canonical_mechanism_signature(family)
            representative.setdefault(signature, family)
            decision = str(item.get("decision") or "")
            implementation = str(item.get("implementation_verdict") or "")
            hypothesis = str(item.get("hypothesis_verdict") or "")
            if decision != "accepted":
                attempts[signature] = attempts.get(signature, 0) + 1
            if implementation == "execution_invalid":
                blocked[representative[signature]] = (
                    "depends on an unsupported execution capability"
                )
            elif implementation == "implemented" and hypothesis == "contradicted":
                blocked[representative[signature]] = (
                    "implemented experiment contradicted the hypothesis"
                )
        for signature, count in attempts.items():
            family = representative[signature]
            if count >= 2:
                blocked.setdefault(family, f"already has {count} non-accepted attempts")
        return blocked

    def _blocked_mechanism_reason(
        self,
        blocked: dict[str, str],
        candidate_family: str,
    ) -> tuple[str, str]:
        candidate_signature = canonical_mechanism_signature(candidate_family)
        for blocked_family, reason in blocked.items():
            if canonical_mechanism_signature(blocked_family) == candidate_signature:
                return blocked_family, reason
        return "", ""

    def _previous_result_was_statistically_inconclusive(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
    ) -> bool:
        history = [
            item
            for item in (state.get("experiment_history") or [])
            if isinstance(item, dict)
            and str(item.get("difficulty") or "") == difficulty
        ]
        if not history:
            return False
        previous = history[-1]
        return bool(
            str(previous.get("decision") or "") == "inconclusive"
            and str(previous.get("implementation_verdict") or "")
            != "execution_invalid"
        )

    def _experiment_spec_from_rationale(self, rationale: dict[str, Any]) -> dict[str, Any]:
        existing = rationale.get("experiment_spec")
        spec = dict(existing) if isinstance(existing, dict) else {}
        priority = rationale.get("priority_problem")
        if "priority_problem" not in spec:
            if isinstance(priority, dict):
                spec["priority_problem"] = dict(priority)
            elif str(priority or "").strip():
                spec["priority_problem"] = {"problem": str(priority).strip()}
            else:
                spec["priority_problem"] = {}
        spec.setdefault("hypothesis", str(rationale.get("hypothesis") or ""))
        spec.setdefault(
            "mechanism_family", str(rationale.get("mechanism_family") or "").strip()
        )
        spec.setdefault(
            "mechanism_signature",
            canonical_mechanism_signature(spec.get("mechanism_family")),
        )
        if "mechanism_prediction" not in spec:
            mechanism_prediction = rationale.get("mechanism_prediction")
            spec["mechanism_prediction"] = (
                dict(mechanism_prediction)
                if isinstance(mechanism_prediction, dict)
                else {}
            )
        if "failure_mode_analysis" not in spec:
            failure_mode_analysis = rationale.get("failure_mode_analysis")
            spec["failure_mode_analysis"] = (
                dict(failure_mode_analysis)
                if isinstance(failure_mode_analysis, dict)
                else {}
            )
        if "priority_alignment" not in spec:
            priority_alignment = rationale.get("priority_alignment")
            spec["priority_alignment"] = (
                dict(priority_alignment)
                if isinstance(priority_alignment, dict)
                else {}
            )
        if "retrieval_assessment" not in spec:
            retrieval_assessment = rationale.get("retrieval_assessment")
            spec["retrieval_assessment"] = (
                dict(retrieval_assessment)
                if isinstance(retrieval_assessment, dict)
                else {}
            )
        if "intervention_package" not in spec:
            intervention_package = rationale.get("intervention_package")
            spec["intervention_package"] = (
                dict(intervention_package)
                if isinstance(intervention_package, dict)
                else {}
            )
        spec.setdefault(
            "plan_direction",
            str(
                rationale.get("plan_direction")
                or rationale.get("overall_assessment")
                or rationale.get("primary_change")
                or ""
            ),
        )
        if "strengths_to_preserve" not in spec:
            strengths = rationale.get("strengths_to_preserve")
            spec["strengths_to_preserve"] = (
                list(strengths) if isinstance(strengths, list) else []
            )
        if "patches" not in spec:
            patches = rationale.get("patches")
            if not isinstance(patches, list) or not patches:
                patches = rationale.get("selected_changes")
            spec["patches"] = _dict_list(patches)
        spec.setdefault("expected_effect", str(rationale.get("expected_effect") or ""))
        spec.setdefault("main_risk", str(rationale.get("main_risk") or ""))
        return spec

    def _evaluation_baseline(self, state: dict[str, Any]) -> BatchResult:
        champion_batch = state.get("champion_batch")
        if not isinstance(champion_batch, dict):
            raise RuntimeError("champion evaluation baseline is missing")
        batch = BatchResult.from_dict(champion_batch)
        current = self._current_difficulty(state)
        if current is not None and batch.difficulty != current:
            raise RuntimeError(
                "champion evaluation baseline is for "
                f"{batch.difficulty}, but the current difficulty is {current}"
            )
        return batch

    def _backfill_experiment_evidence(self, state: dict[str, Any]) -> bool:
        """Enrich pre-v3 rejection memory from immutable decision/batch artifacts."""
        changed = False
        experiences = state.get("experiment_history")
        if not isinstance(experiences, list):
            return False
        for experience in experiences:
            if not isinstance(experience, dict) or isinstance(
                experience.get("experiment_evidence"), dict
            ):
                continue
            try:
                generation = int(experience.get("generation"))
            except (TypeError, ValueError):
                continue
            decision_path = self.run_dir / f"generation_{generation:03d}" / "decision.json"
            if not decision_path.is_file():
                continue
            try:
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
                parent = summarize_batch_evidence(Path(decision["parent_batch"]))
                candidate = summarize_batch_evidence(Path(decision["candidate_batch"]))
                confirmation = decision.get("confirmation")
                confirmation_evidence: dict[str, Any] | None = None
                if isinstance(confirmation, dict):
                    confirmed_parent = summarize_batch_evidence(
                        Path(confirmation["champion_batch"]["path"])
                    )
                    confirmed_candidate = summarize_batch_evidence(
                        Path(confirmation["candidate_batch"]["path"])
                    )
                    parent = combine_batch_evidence(parent, confirmed_parent)
                    candidate = combine_batch_evidence(candidate, confirmed_candidate)
                    confirmation_evidence = {
                        "parent_batch": confirmed_parent,
                        "candidate_batch": confirmed_candidate,
                    }
            except (KeyError, OSError, TypeError, ValueError):
                continue
            experience["experiment_evidence"] = {
                "parent_batch": parent,
                "candidate_batch": candidate,
                "candidate_minus_parent": compare_batch_evidence(parent, candidate),
                "comparison_used_confirmation": confirmation_evidence is not None,
                "confirmation_batches": confirmation_evidence,
            }
            for key in (
                "hypothesis",
                "primary_lever",
                "predictions",
                "disproof_conditions",
                "capability_mapping",
            ):
                if key not in experience and key in decision:
                    experience[key] = decision[key]
            changed = True
        return changed

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = datetime.now().isoformat()
        _write_json(self.state_path, state)

    def _batch_name(self, generation: int, role: str, difficulty: str = "") -> str:
        # Include difficulty so a mastered champion baseline is not reused as
        # the next curriculum level (same generation/role, different AI).
        parts = [f"ev_{self.run_id}", f"g{generation:03d}"]
        if difficulty:
            parts.append(difficulty)
        parts.append(role)
        return _safe_name("_".join(parts), 56)

    def _strategy_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env[STRATEGY_ROOT_ENV] = str(self.strategies_dir)
        return env

    def _stage_strategy(self, name: str) -> Path:
        dest = self.strategies_dir / canonical_strategy_folder(name)
        dest.mkdir(parents=True, exist_ok=True)
        dest_md = dest / "strategy.md"
        if dest_md.is_file():
            return dest
        src = resolve_skill_dir(
            name,
            self.config.race,
            overlay_root=self.strategies_dir,
            skill_root=self.project_root / "skills",
        )
        src_md = src / "strategy.md"
        if src_md.is_file() and src.resolve() != dest.resolve():
            shutil.copy2(src_md, dest_md)
        return dest

    def _copy_strategy_sidecar(self, name: str, dest_dir: Path) -> None:
        src = self._stage_strategy(name) / "strategy.md"
        if not src.is_file():
            return
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / "strategy.md")

    def _read_strategy_text(
        self,
        name: str,
        *,
        batch_dirs: list[Path] | None = None,
    ) -> str:
        for batch_dir in batch_dirs or []:
            sidecar = Path(batch_dir) / "strategy.md"
            if sidecar.is_file():
                return sidecar.read_text(encoding="utf-8-sig")
        path = (
            resolve_skill_dir(
                name,
                self.config.race,
                overlay_root=self.strategies_dir,
                skill_root=self.project_root / "skills",
            )
            / "strategy.md"
        )
        return path.read_text(encoding="utf-8-sig")

    def run_batch(
        self,
        strategy: str,
        difficulty: str,
        *,
        generation: int,
        role: str,
        target_games: int | None = None,
    ) -> BatchResult:
        if self._batch_executor is not None:
            expected_games = int(target_games or self.config.matches_per_batch)
            try:
                return self._batch_executor(strategy, difficulty, expected_games)
            except TypeError:
                return self._batch_executor(strategy, difficulty)
        expected_games = int(target_games or self.config.matches_per_batch)
        if expected_games <= 0:
            raise ValueError("target_games must be positive")
        batch_name = self._batch_name(generation, role, difficulty)
        batch_dir = self.project_root / "game_records" / batch_name
        completed = completed_record_count(batch_dir, strategy=strategy)
        if completed == expected_games:
            return read_batch_result(
                batch_dir,
                name=batch_name,
                strategy=strategy,
                difficulty=difficulty,
                expected_games=expected_games,
            )
        if completed > expected_games:
            raise RuntimeError(
                f"batch {batch_name} has {completed} completed records, more than the "
                f"requested {expected_games}"
            )
        remaining = expected_games - completed
        self._copy_strategy_sidecar(strategy, batch_dir)
        if os.name == "nt":
            command = [
                "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(self.project_root / "scripts" / "start_batch.ps1"),
                "-MY_BOT_NAME", self.config.bot_name,
                "-MAP_NAME", self.config.map_name,
                "-REAL_TIME", "1" if self.config.real_time else "0",
                "-ENEMY_RACE", self.config.enemy_race,
                "-ENEMY_DIFFICULTY", difficulty,
                "-ENEMY_BUILD", self.config.enemy_build,
                "-BOT_RACE", self.config.race,
                "-FORCE_STRATEGY", strategy,
                "-COMMANDER_MODEL", self.config.commander_model,
                "-TOTAL_MATCHES", str(remaining),
                "-CONCURRENCY", str(self.config.concurrency),
                "-START_INDEX", str(completed),
                "-BATCH_NAME", batch_name,
            ]
            if self.config.bot_instruct:
                command.extend(["-BOT_INSTRUCT", self.config.bot_instruct])
        else:
            command = [
                "bash", str(self.project_root / "scripts" / "start_batch.sh"),
                "--my-bot-name", self.config.bot_name,
                "--map-name", self.config.map_name,
                "--real-time", "1" if self.config.real_time else "0",
                "--enemy-race", self.config.enemy_race,
                "--enemy-difficulty", difficulty,
                "--enemy-build", self.config.enemy_build,
                "--bot-race", self.config.race,
                "--force-strategy", strategy,
                "--commander-model", self.config.commander_model,
                "--total-matches", str(remaining),
                "--concurrency", str(self.config.concurrency),
                "--start-index", str(completed),
                "--batch-name", batch_name,
            ]
            if self.config.bot_instruct:
                command.extend(["--bot-instruct", self.config.bot_instruct])
        subprocess.run(
            command,
            cwd=self.project_root,
            check=True,
            env=self._strategy_env(),
        )
        self._copy_strategy_sidecar(strategy, batch_dir)
        return read_batch_result(
            batch_dir,
            name=batch_name,
            strategy=strategy,
            difficulty=difficulty,
            expected_games=expected_games,
        )

    def generate_candidate(
        self,
        champion: str,
        champion_batch: BatchResult,
        prior_experiences: list[Any],
        *,
        evidence_batches: list[BatchResult] | None = None,
        resume_dir: Path | None = None,
        analysis_seed_dir: Path | None = None,
        retry_feedback: list[str] | None = None,
    ) -> EvolRunResult:
        if self._candidate_generator is not None:
            return self._candidate_generator(champion, champion_batch, prior_experiences)
        record_paths: list[Path] = []
        for batch in evidence_batches or [champion_batch]:
            record_paths.extend(find_record_jsons(batch.path))
        skill_dir = resolve_skill_dir(
            champion,
            self.config.race,
            overlay_root=self.strategies_dir,
            skill_root=self.project_root / "skills",
        )
        output_dir = output_dir_for_strategy(
            champion,
            self.config.race,
            overlay_root=self.strategies_dir,
        )
        return EvolAgent(model=self.config.evolution_model).run(
            EvolRunRequest(
                record_paths=list(dict.fromkeys(record_paths)),
                strategy_name=champion,
                race=self.config.race,
                skill_dir=skill_dir,
                output_dir=output_dir,
                model=self.config.evolution_model,
                knowledge_mode=self.config.knowledge_mode,
                prior_experiences=prior_experiences,
                resume_dir=resume_dir,
                analysis_seed_dir=analysis_seed_dir,
                match_summary_cache_path=(
                    self.run_dir / "experiment_match_summary_cache.json"
                ),
                retry_feedback=list(retry_feedback or []),
            )
        )

    def _ensure_history_schema(self) -> None:
        if not self.history_path.is_file():
            return
        with self.history_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            source_fields = tuple(reader.fieldnames or ())
            if source_fields == HISTORY_FIELDS:
                return
        legacy_generation_numbering = bool(
            "evolution_score" in source_fields or "games_used" in source_fields
        )
        migrated: list[dict[str, Any]] = []
        for old in rows:
            wins = int(old.get("wins") or 0)
            draws = int(old.get("draws") or 0)
            losses = int(old.get("losses") or 0)
            games = wins + draws + losses
            score = float(old.get("score") or 0.0)
            win_rate = wins / games if games else 0.0
            mastered = int(old.get("mastered_levels") or 0)
            parent = str(old.get("parent") or "")
            generation = int(old.get("generation") or 0) + int(
                bool(parent) and legacy_generation_numbering
            )
            progress_score = curriculum_progress_score(
                mastered,
                win_rate,
                self.config.mastery_score_threshold,
            )
            migrated.append(
                {
                    "strategy_style": str(old.get("strategy_style") or ""),
                    "generation": generation,
                    "strategy": str(old.get("strategy") or ""),
                    "parent": parent,
                    "difficulty": str(old.get("difficulty") or ""),
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "games": games,
                    "score": f"{score:.4f}",
                    "win_rate": f"{win_rate:.4f}",
                    "mastered_levels": mastered,
                    "curriculum_progress_score": f"{progress_score:.4f}",
                    "accepted": str(old.get("accepted") or "false"),
                    "batch": str(old.get("batch") or ""),
                }
            )
        temp_path = self.history_path.with_suffix(".csv.tmp")
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(migrated)
        temp_path.replace(self.history_path)

    def _append_history(
        self,
        *,
        state: dict[str, Any],
        batch: BatchResult,
        parent: str,
        accepted: bool,
        generation: int | None = None,
    ) -> None:
        self._ensure_history_schema()
        if self.history_path.is_file():
            with self.history_path.open(encoding="utf-8", newline="") as handle:
                if any(
                    row.get("batch") == batch.name and row.get("strategy") == batch.strategy
                    for row in csv.DictReader(handle)
                ):
                    return
        mastered = int(state["difficulty_index"])
        progress_score = curriculum_progress_score(
            mastered,
            batch.win_rate,
            self.config.mastery_score_threshold,
        )
        row = {
            "strategy_style": state["style"],
            "generation": int(state["generation"]) if generation is None else generation,
            "strategy": batch.strategy,
            "parent": parent,
            "difficulty": batch.difficulty,
            "wins": batch.wins,
            "draws": batch.draws,
            "losses": batch.losses,
            "games": batch.games,
            "score": f"{batch.score:.4f}",
            "win_rate": f"{batch.win_rate:.4f}",
            "mastered_levels": mastered,
            "curriculum_progress_score": f"{progress_score:.4f}",
            "accepted": str(accepted).lower(),
            "batch": batch.name,
        }
        new_file = not self.history_path.exists()
        with self.history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def _update_history_strategy_evidence(self, batch: BatchResult) -> None:
        """Refresh the latest accepted row with all evidence used for comparison."""
        self._ensure_history_schema()
        if not self.history_path.is_file():
            return
        with self.history_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        target_index: int | None = None
        for index, row in enumerate(rows):
            if (
                row.get("strategy") == batch.strategy
                and row.get("difficulty") == batch.difficulty
                and str(row.get("accepted") or "").lower() == "true"
            ):
                target_index = index
        if target_index is None:
            return
        row = rows[target_index]
        if batch.games < int(row.get("games") or 0):
            return
        mastered = int(row.get("mastered_levels") or 0)
        progress_score = curriculum_progress_score(
            mastered,
            batch.win_rate,
            self.config.mastery_score_threshold,
        )
        row.update(
            {
                "wins": str(batch.wins),
                "draws": str(batch.draws),
                "losses": str(batch.losses),
                "games": str(batch.games),
                "score": f"{batch.score:.4f}",
                "win_rate": f"{batch.win_rate:.4f}",
                "curriculum_progress_score": f"{progress_score:.4f}",
                "batch": batch.name,
            }
        )
        temp_path = self.history_path.with_suffix(".csv.tmp")
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(self.history_path)

    def _advance_difficulty(self, state: dict[str, Any]) -> None:
        difficulty = self._current_difficulty(state)
        if difficulty:
            mastered = state.setdefault("mastered_difficulties", [])
            if difficulty not in mastered:
                mastered.append(difficulty)
        state["difficulty_index"] = int(state.get("difficulty_index") or 0) + 1
        self._sync_champion_baseline(state, None)
        self._sync_search_parent(state, str(state.get("champion") or ""), None)
        state["inconclusive_streak"] = 0
        self._reset_generation_local_analysis_state(state)
        state["difficulty_generation"] = 0
        if state["difficulty_index"] >= len(self.config.difficulties):
            self._complete_curriculum(state)
        else:
            state["difficulty"] = self.config.difficulties[state["difficulty_index"]]

    def _record_agent_decision(
        self,
        state: dict[str, Any],
        result: EvolRunResult,
        *,
        difficulty: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "action": result.decision_action,
            "reason": result.action_reason,
            "message": result.message,
            "generation": int(state["generation"]),
            "difficulty": difficulty,
        }
        if extra:
            payload.update(extra)
        state["last_agent_decision"] = payload

    def _pause(
        self,
        state: dict[str, Any],
        *,
        status: str,
        result: EvolRunResult,
        difficulty: str,
    ) -> dict[str, Any]:
        state["status"] = status
        state["pending_candidate"] = None
        self._record_agent_decision(state, result, difficulty=difficulty)
        self._save_state(state)
        return state

    def run(self) -> dict[str, Any]:
        state = self.load_or_create_state()
        while state["status"] == "running":
            difficulty = self._current_difficulty(state)
            if difficulty is None:
                self._complete_curriculum(state)
                self._save_state(state)
                break
            state["difficulty"] = difficulty
            champion = str(state["champion"])

            if state.get("champion_batch") is None:
                baseline = self.run_batch(
                    champion,
                    difficulty,
                    generation=int(state["generation"]),
                    role="champ",
                    target_games=self.config.matches_per_batch,
                )
                self._register_evidence(state, baseline)
                self._sync_games_used(state)
                self._sync_champion_baseline(state, baseline)
                if str(state.get("search_parent") or champion) == champion:
                    self._sync_search_parent(state, champion, baseline)
                self._append_history(state=state, batch=baseline, parent="", accepted=True)
                self._save_state(state)

            champion_batch = self._evaluation_baseline(state)
            if self._is_mastered(champion_batch):
                self._advance_difficulty(state)
                self._save_state(state)
                continue

            if int(state.get("generation") or 0) >= self.config.max_total_generations:
                state["status"] = "total_budget_exhausted"
                self._save_state(state)
                break
            if (
                int(state.get("difficulty_generation") or 0)
                >= self.config.max_generations_per_difficulty
            ):
                state["status"] = "difficulty_budget_exhausted"
                state["failed_difficulty"] = difficulty
                state["champion_score"] = champion_batch.score
                self._save_state(state)
                break

            pending = state.get("pending_candidate")
            if not isinstance(pending, dict):
                if not self._analyze_champion(state, difficulty, champion, champion_batch):
                    break
                continue

            self._evaluate_and_commit_experiment(
                state,
                difficulty=difficulty,
                champion=champion,
                champion_batch=champion_batch,
                pending=pending,
            )
            self._save_state(state)
        return state

    def _analyze_champion(
        self,
        state: dict[str, Any],
        difficulty: str,
        champion: str,
        champion_batch: BatchResult,
    ) -> bool:
        search_parent = str(state.get("search_parent") or champion)
        parent_evidence_batches = self._evidence_batches(
            state,
            difficulty=difficulty,
            strategy=search_parent,
        )
        if not parent_evidence_batches and search_parent == champion:
            self._register_evidence(state, champion_batch)
            parent_evidence_batches = [champion_batch]
        if not parent_evidence_batches:
            raise RuntimeError(
                f"search parent {search_parent} has no outcome evidence at {difficulty}"
            )
        search_parent_batch = parent_evidence_batches[0]
        for batch in parent_evidence_batches[1:]:
            search_parent_batch = combine_batch_results(search_parent_batch, batch)
        self._sync_search_parent(state, search_parent, search_parent_batch)
        resume_value = str(state.get("candidate_resume_dir") or "").strip()
        record_paths = [
            path
            for batch in parent_evidence_batches
            for path in find_record_jsons(batch.path)
        ]
        if not resume_value:
            recovered = self._find_resumable_analysis_checkpoint(
                strategy=search_parent,
                record_paths=list(dict.fromkeys(record_paths)),
            )
            if recovered is not None:
                resume_value = str(recovered)
                state["candidate_resume_dir"] = resume_value
                self._save_state(state)
                print(
                    "EvolAgent resuming the latest compatible unfinished analysis: "
                    f"{recovered}",
                    flush=True,
                )
        analysis_seed_dir = self._find_analysis_seed_checkpoint(
            state,
            difficulty=difficulty,
            strategy=search_parent,
            record_paths=list(dict.fromkeys(record_paths)),
        )
        retry_feedback: list[str] = []
        if resume_value:
            for failure in reversed(state.get("candidate_generation_failures") or []):
                if not isinstance(failure, dict):
                    continue
                if str(failure.get("parent") or "") != search_parent:
                    continue
                failure_checkpoint = str(failure.get("checkpoint_dir") or "").strip()
                if failure_checkpoint and failure_checkpoint != resume_value:
                    continue
                message = str(failure.get("message") or "").strip()
                if message:
                    retry_feedback.append(
                        "Previous candidate-generation failure from the resumed "
                        f"checkpoint: {message}"
                    )
                break
        candidate_result: EvolRunResult | None = None
        total_attempts = self.config.candidate_generation_retries + 1
        for attempt in range(1, total_attempts + 1):
            candidate_result = self.generate_candidate(
                search_parent,
                search_parent_batch,
                self._prior_experiences(state, difficulty=difficulty),
                evidence_batches=parent_evidence_batches,
                resume_dir=Path(resume_value) if resume_value else None,
                analysis_seed_dir=analysis_seed_dir,
                retry_feedback=retry_feedback,
            )
            self._remember_analysis_checkpoint(
                state,
                difficulty=difficulty,
                strategy=search_parent,
                checkpoint_dir=candidate_result.checkpoint_dir,
            )
            if candidate_result.ok and candidate_result.output_dir is not None:
                state["candidate_resume_dir"] = None
                break
            if (
                candidate_result.ok
                and candidate_result.decision_action != "propose_strategy_patch"
            ):
                return self._handle_analysis_decision(
                    state,
                    candidate_result,
                    difficulty=difficulty,
                    champion=search_parent,
                )

            message = str(candidate_result.message or "candidate generation failed")
            failure = {
                "kind": "candidate_generation_failure",
                "generation": int(state["generation"]),
                "attempt": attempt,
                "max_attempts": total_attempts,
                "difficulty": difficulty,
                "parent": search_parent,
                "comparison_champion": champion,
                "message": message,
                "checkpoint_dir": str(candidate_result.checkpoint_dir or ""),
                "created_at": datetime.now().isoformat(),
            }
            state.setdefault("candidate_generation_failures", []).append(failure)
            checkpoint_value = str(candidate_result.checkpoint_dir or resume_value)
            state["candidate_resume_dir"] = checkpoint_value
            if attempt >= total_attempts:
                break

            retry_feedback.append(
                f"Candidate-generation attempt {attempt}/{total_attempts} failed: {message}"
            )
            # A broken resume checkpoint cannot repair itself; restart from the
            # compatible analysis seed while retaining the explicit feedback.
            if message.startswith("failed to load checkpoint:") or message.startswith(
                "strategy mismatch with checkpoint:"
            ):
                resume_value = ""
                state["candidate_resume_dir"] = ""
            else:
                resume_value = checkpoint_value
            self._save_state(state)
            print(
                f"EvolAgent candidate generation failed; retrying "
                f"({attempt + 1}/{total_attempts}) with feedback: {message}",
                flush=True,
            )

        if candidate_result is None:
            raise RuntimeError("candidate generation produced no result")
        if (
            not candidate_result.ok
            or candidate_result.output_dir is None
        ):
            state["status"] = "evol_agent_failed"
            self._save_state(state)
            print(
                "EvolAgent could not produce a usable candidate; state and the "
                f"analysis checkpoint are saved at {self.run_dir}",
                flush=True,
            )
            return False
        rationale = (
            candidate_result.improvement.analysis
            if candidate_result.improvement is not None
            else {}
        )
        experiment_spec = self._experiment_spec_from_rationale(rationale)
        mechanism_family = str(
            experiment_spec.get("mechanism_family") or ""
        ).strip().lower()
        blocked_families = self._blocked_mechanism_families(
            state, difficulty=difficulty
        )
        blocked_family, blocked_reason = self._blocked_mechanism_reason(
            blocked_families,
            mechanism_family,
        )
        if mechanism_family and blocked_reason:
            rejection = {
                "kind": "mechanism_policy_rejection",
                "generation": int(state["generation"]),
                "difficulty": difficulty,
                "candidate": candidate_result.output_dir.name,
                "mechanism_family": mechanism_family,
                "mechanism_signature": canonical_mechanism_signature(
                    mechanism_family
                ),
                "equivalent_blocked_family": blocked_family,
                "reason": blocked_reason,
                "decision": "policy_rejected_before_matches",
                "created_at": datetime.now().isoformat(),
            }
            state.setdefault("mechanism_policy_rejections", []).append(rejection)
            retries = sum(
                1
                for item in state["mechanism_policy_rejections"]
                if isinstance(item, dict)
                and int(item.get("generation") or -1) == int(state["generation"])
            )
            if retries >= 2:
                state["status"] = "mechanism_policy_attention_required"
            self._save_state(state)
            print(
                "EvolAgent candidate blocked before matches by mechanism policy: "
                f"{mechanism_family} is equivalent to {blocked_family} "
                f"({blocked_reason})",
                flush=True,
            )
            return retries < 2
        state["pending_candidate"] = {
            "strategy": candidate_result.output_dir.name,
            "strategy_dir": str(candidate_result.output_dir),
            "mutation_parent": search_parent,
            "comparison_champion": champion,
            "mutation_parent_batch": search_parent_batch.to_dict(),
            "mutation_parent_batch_paths": [
                str(batch.path) for batch in parent_evidence_batches
            ],
            "candidate_hash": candidate_result.candidate_hash,
            "analysis_checkpoint_dir": str(candidate_result.checkpoint_dir or ""),
            "experiment_spec": experiment_spec,
            "candidate_batch": None,
            "evaluation_complete": False,
            "experiment_committed": False,
            "primary_change": str(rationale.get("primary_change") or ""),
            "selected_plan_ids": _string_list(rationale.get("selected_plan_ids")),
            "overall_assessment": str(rationale.get("overall_assessment") or ""),
            "selected_changes": _dict_list(rationale.get("selected_changes")),
            "expected_effect": str(rationale.get("expected_effect") or ""),
            "main_risk": str(rationale.get("main_risk") or ""),
            "hypothesis": str(rationale.get("hypothesis") or ""),
            "mechanism_family": str(rationale.get("mechanism_family") or "").strip(),
            "mechanism_signature": canonical_mechanism_signature(
                rationale.get("mechanism_family")
            ),
            "mechanism_prediction": (
                dict(rationale.get("mechanism_prediction"))
                if isinstance(rationale.get("mechanism_prediction"), dict)
                else {}
            ),
            "failure_mode_analysis": (
                dict(rationale.get("failure_mode_analysis"))
                if isinstance(rationale.get("failure_mode_analysis"), dict)
                else {}
            ),
            "priority_alignment": (
                dict(rationale.get("priority_alignment"))
                if isinstance(rationale.get("priority_alignment"), dict)
                else {}
            ),
            "retrieval_assessment": (
                dict(rationale.get("retrieval_assessment"))
                if isinstance(rationale.get("retrieval_assessment"), dict)
                else {}
            ),
            "intervention_package": (
                dict(rationale.get("intervention_package"))
                if isinstance(rationale.get("intervention_package"), dict)
                else {}
            ),
            "primary_lever": str(rationale.get("primary_lever") or ""),
            "predictions": _string_list(rationale.get("predictions")),
            "disproof_conditions": _string_list(rationale.get("disproof_conditions")),
            "capability_mapping": (
                dict(rationale.get("capability_mapping"))
                if isinstance(rationale.get("capability_mapping"), dict)
                else {}
            ),
            "inheritance": (
                dict(rationale.get("inheritance"))
                if isinstance(rationale.get("inheritance"), dict)
                else {}
            ),
            "semantic_validation": (
                dict(rationale.get("semantic_validation"))
                if isinstance(rationale.get("semantic_validation"), dict)
                else {"status": "passed", "errors": []}
            ),
        }
        self._save_state(state)
        return True

    def _handle_analysis_decision(
        self,
        state: dict[str, Any],
        result: EvolRunResult,
        *,
        difficulty: str,
        champion: str,
    ) -> bool:
        action = result.decision_action
        if action == "request_more_matches":
            analysis_games = self._analysis_games(
                state, difficulty=difficulty, strategy=champion
            )
            if analysis_games >= self.config.max_analysis_games_per_generation:
                self._pause(
                    state,
                    status="insufficient_evidence",
                    result=result,
                    difficulty=difficulty,
                )
                return False
            more_batch = self.run_batch(
                champion,
                difficulty,
                generation=int(state["generation"]),
                role="champ_more",
                target_games=self.config.analysis_batch_games,
            )
            self._register_evidence(state, more_batch)
            self._sync_games_used(state)
            self._record_agent_decision(
                state,
                result,
                difficulty=difficulty,
                extra={"additional_batch": more_batch.to_dict()},
            )
            self._save_state(state)
            return True
        if action == "inspect_runtime":
            self._pause(
                state,
                status="runtime_attention_required",
                result=result,
                difficulty=difficulty,
            )
            return False
        if action == "stop":
            self._pause(
                state,
                status="stopped_no_actionable_improvement",
                result=result,
                difficulty=difficulty,
            )
            return False
        self._pause(
            state,
            status="agent_paused",
            result=result,
            difficulty=difficulty,
        )
        return False

    def _evaluate_and_commit_experiment(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
        champion: str,
        champion_batch: BatchResult,
        pending: dict[str, Any],
    ) -> None:
        candidate = str(pending["strategy"])
        mutation_parent = str(pending.get("mutation_parent") or champion)
        comparison_champion_name = str(
            pending.get("comparison_champion") or champion
        )
        if comparison_champion_name != champion:
            raise RuntimeError(
                "pending candidate comparison champion no longer matches state"
            )
        evaluation_games = self.config.candidate_matches
        pending_batch = pending.get("candidate_batch")
        candidate_games = 0
        if isinstance(pending_batch, dict):
            candidate_games = BatchResult.from_dict(pending_batch).games
        if candidate_games < evaluation_games:
            candidate_games = sum(
                batch.games
                for batch in self._evidence_batches(
                    state,
                    difficulty=difficulty,
                    strategy=candidate,
                )
            )
        if candidate_games < evaluation_games:
            candidate_batch = self.run_batch(
                candidate,
                difficulty,
                generation=int(state["generation"]),
                role="cand",
                target_games=evaluation_games,
            )
            self._register_evidence(state, candidate_batch)
            self._sync_games_used(state)
            pending["candidate_batch"] = candidate_batch.to_dict()
            pending["evaluation_complete"] = True
            state["pending_candidate"] = pending
            self._save_state(state)
        comparison_champion = self._evaluation_baseline(state)
        if isinstance(pending.get("candidate_batch"), dict):
            comparison_candidate = BatchResult.from_dict(pending["candidate_batch"])
        else:
            comparison_candidate = self._evidence_batches(
                state,
                difficulty=difficulty,
                strategy=candidate,
            )[-1]
        if comparison_candidate.games < evaluation_games:
            raise RuntimeError(
                f"candidate {candidate} has {comparison_candidate.games}/"
                f"{evaluation_games} evaluation games; it cannot be decided"
            )
        pending["evaluation_complete"] = True
        initial_champion = comparison_champion
        initial_candidate = comparison_candidate
        candidate_mastered = self._is_mastered(initial_candidate)
        confirmation: dict[str, Any] | None = None
        if (
            self.config.confirmation_matches > 0
            and not candidate_mastered
            and close_batch_results(initial_champion, initial_candidate)
        ):
            if initial_champion.games == initial_candidate.games:
                champion_confirmation_games = self.config.confirmation_matches
                candidate_confirmation_games = self.config.confirmation_matches
            elif initial_champion.games > initial_candidate.games:
                champion_confirmation_games = 0
                candidate_confirmation_games = (
                    initial_champion.games - initial_candidate.games
                )
            else:
                champion_confirmation_games = (
                    initial_candidate.games - initial_champion.games
                )
                candidate_confirmation_games = 0
            champion_confirmation_data = pending.get("champion_confirmation")
            candidate_confirmation_data = pending.get("candidate_confirmation")
            if isinstance(champion_confirmation_data, dict):
                champion_confirmation = BatchResult.from_dict(
                    champion_confirmation_data
                )
            elif champion_confirmation_games > 0:
                champion_confirmation = self.run_batch(
                    champion,
                    difficulty,
                    generation=int(state["generation"]),
                    role="champ_confirm",
                    target_games=champion_confirmation_games,
                )
                self._register_evidence(state, champion_confirmation)
                pending["champion_confirmation"] = champion_confirmation.to_dict()
                self._sync_games_used(state)
                state["pending_candidate"] = pending
                self._save_state(state)
            else:
                champion_confirmation = None
            if isinstance(candidate_confirmation_data, dict):
                candidate_confirmation = BatchResult.from_dict(
                    candidate_confirmation_data
                )
            elif candidate_confirmation_games > 0:
                candidate_confirmation = self.run_batch(
                    candidate,
                    difficulty,
                    generation=int(state["generation"]),
                    role="cand_confirm",
                    target_games=candidate_confirmation_games,
                )
                self._register_evidence(state, candidate_confirmation)
                pending["candidate_confirmation"] = candidate_confirmation.to_dict()
                self._sync_games_used(state)
                state["pending_candidate"] = pending
                self._save_state(state)
            else:
                candidate_confirmation = None
            comparison_champion = (
                combine_batch_results(initial_champion, champion_confirmation)
                if champion_confirmation is not None
                else initial_champion
            )
            comparison_candidate = (
                combine_batch_results(initial_candidate, candidate_confirmation)
                if candidate_confirmation is not None
                else initial_candidate
            )
            confirmation = {
                "champion_batch": (
                    champion_confirmation.to_dict()
                    if champion_confirmation is not None
                    else None
                ),
                "candidate_batch": (
                    candidate_confirmation.to_dict()
                    if candidate_confirmation is not None
                    else None
                ),
            }
        self._sync_champion_baseline(state, comparison_champion)
        if str(state.get("search_parent") or champion) == champion:
            self._sync_search_parent(state, champion, comparison_champion)
        probability = posterior_probability_better(
            comparison_candidate.to_dict(),
            comparison_champion.to_dict(),
        )
        score_outcome = decide_candidate(
            comparison_candidate.score,
            comparison_champion.score,
        )
        outcome = "accepted" if candidate_mastered else score_outcome
        accepted = outcome == "accepted"
        score_delta = comparison_candidate.score - comparison_champion.score
        experiment_spec = (
            dict(pending["experiment_spec"])
            if isinstance(pending.get("experiment_spec"), dict)
            else self._experiment_spec_from_rationale(pending)
        )
        mutation_parent_batches = self._evidence_batches(
            state,
            difficulty=difficulty,
            strategy=mutation_parent,
        )
        if mutation_parent == champion:
            mutation_parent_result = comparison_champion
            parent_batch_dirs = [initial_champion.path]
        else:
            if not mutation_parent_batches:
                raise RuntimeError(
                    f"mutation parent {mutation_parent} has no evidence at {difficulty}"
                )
            mutation_parent_result = mutation_parent_batches[0]
            for batch in mutation_parent_batches[1:]:
                mutation_parent_result = combine_batch_results(
                    mutation_parent_result, batch
                )
            parent_batch_dirs = [batch.path for batch in mutation_parent_batches]
        candidate_batch_dirs = [initial_candidate.path]
        if confirmation is not None:
            champion_confirmation_info = confirmation.get("champion_batch")
            candidate_confirmation_info = confirmation.get("candidate_batch")
            if mutation_parent == champion and isinstance(
                champion_confirmation_info, dict
            ):
                parent_batch_dirs.append(
                    Path(champion_confirmation_info["path"])
                )
            if isinstance(candidate_confirmation_info, dict):
                candidate_batch_dirs.append(
                    Path(candidate_confirmation_info["path"])
                )
        fallback_audit = {
            "implementation_verdict": "unknown",
            "hypothesis_verdict": "inconclusive",
            "mechanism_evidence": [],
            "combat_evidence": [],
            "runtime_findings": [],
            "evidence_limits": [
                (
                    "mechanism audit skipped because the candidate reached the "
                    "difficulty mastery threshold"
                    if candidate_mastered
                    else "post-experiment mechanism audit was unavailable"
                )
            ],
            "lesson": (
                "The candidate reached the difficulty mastery threshold, so it was "
                "accepted and advanced without a post-experiment mechanism audit."
                if candidate_mastered
                else ""
            ),
        }
        auditor = None if candidate_mastered else self._experiment_auditor
        if not candidate_mastered and auditor is None and self._batch_executor is None:
            auditor = audit_experiment
        if auditor is not None:
            try:
                parent_strategy_text = self._read_strategy_text(
                    mutation_parent,
                    batch_dirs=parent_batch_dirs,
                )
                candidate_strategy_text = self._read_strategy_text(
                    candidate,
                    batch_dirs=candidate_batch_dirs,
                )
                audit_kwargs: dict[str, Any] = {
                    "race": self.config.race,
                    "parent_strategy_name": mutation_parent,
                    "candidate_strategy_name": candidate,
                    "parent_strategy": parent_strategy_text,
                    "candidate_strategy": candidate_strategy_text,
                    "parent_batch_dirs": parent_batch_dirs,
                    "candidate_batch_dirs": candidate_batch_dirs,
                    "experiment_spec": experiment_spec,
                    "outcome_comparison": {
                        "mutation_parent_score": mutation_parent_result.score,
                        "comparison_champion": champion,
                        "comparison_champion_score": comparison_champion.score,
                        "candidate_score": comparison_candidate.score,
                        "score_delta": score_delta,
                        "posterior_probability_better": probability,
                        "posterior_used_for_selection": False,
                        "provisionally_selected_by_outcomes": accepted,
                    },
                    "model": self.config.evolution_model or self.config.commander_model,
                }
                if auditor is audit_experiment:
                    checkpoint_dir = str(
                        pending.get("analysis_checkpoint_dir") or ""
                    ).strip()
                    parent_analysis_files: list[Path] = []
                    if checkpoint_dir:
                        analysis_file = Path(checkpoint_dir) / "analysis.json"
                        if analysis_file.is_file():
                            parent_analysis_files.append(analysis_file)
                    if not parent_analysis_files:
                        log_root = (
                            self.project_root
                            / "evol_agent"
                            / "logs"
                            / mutation_parent
                        )
                        if log_root.is_dir():
                            parent_analysis_files = sorted(
                                log_root.glob("*/analysis.json"),
                                key=lambda path: path.stat().st_mtime_ns,
                                reverse=True,
                            )
                    parent_analysis: dict[str, Any] = {}
                    parent_analysis_record_paths: list[str] = []
                    for analysis_file in parent_analysis_files:
                        try:
                            loaded_analysis = json.loads(
                                analysis_file.read_text(encoding="utf-8-sig")
                            )
                        except (OSError, ValueError):
                            continue
                        if not isinstance(loaded_analysis, dict):
                            continue
                        recorded_strategy = str(
                            loaded_analysis.get("strategy_name") or ""
                        ).strip()
                        if recorded_strategy and recorded_strategy != mutation_parent:
                            continue
                        parent_analysis = loaded_analysis
                        checkpoint_file = analysis_file.parent / "checkpoint.json"
                        try:
                            checkpoint_meta = json.loads(
                                checkpoint_file.read_text(encoding="utf-8-sig")
                            )
                        except (OSError, ValueError):
                            checkpoint_meta = {}
                        if isinstance(checkpoint_meta, dict):
                            parent_analysis_record_paths = [
                                str(path)
                                for path in (checkpoint_meta.get("record_files") or [])
                                if str(path).strip()
                            ]
                        break
                    audit_kwargs.update(
                        summary_cache_path=(
                            self.run_dir / "experiment_match_summary_cache.json"
                        ),
                        parent_analysis=parent_analysis,
                        parent_analysis_record_paths=parent_analysis_record_paths,
                    )
                mechanism_audit = auditor(**audit_kwargs)
                if not isinstance(mechanism_audit, dict):
                    mechanism_audit = fallback_audit
            except Exception as exc:  # mechanism audit must not lose match results
                mechanism_audit = {
                    **fallback_audit,
                    "evidence_limits": [
                        f"post-experiment audit failed: {type(exc).__name__}: {exc}"
                    ],
                }
        else:
            mechanism_audit = fallback_audit
        base_outcome = outcome
        implementation_verdict = str(
            mechanism_audit.get("implementation_verdict") or "unknown"
        )
        hypothesis_verdict = str(
            mechanism_audit.get("hypothesis_verdict") or "inconclusive"
        )
        if accepted and implementation_verdict == "implemented":
            performance_gain_cause = (
                "supported_mechanism"
                if hypothesis_verdict == "supported"
                else "unknown"
            )
        elif accepted:
            performance_gain_cause = "unverified"
        else:
            performance_gain_cause = "not_applicable"
        mechanism_family = str(
            experiment_spec.get("mechanism_family")
            or pending.get("mechanism_family")
            or ""
        ).strip()
        prior_family_attempts = sum(
            1
            for item in (state.get("mechanism_ledger") or [])
            if isinstance(item, dict)
            and str(item.get("difficulty") or "") == difficulty
            and str(item.get("mechanism_family") or "").strip().casefold()
            == mechanism_family.casefold()
            and str(item.get("decision") or "") != "accepted"
        )
        repairable_underpowered = bool(
            not accepted
            and mechanism_family
            and implementation_verdict == "underpowered"
            and hypothesis_verdict in {"inconclusive", "not_tested"}
            and prior_family_attempts == 0
        )
        underpowered_retry_exhausted = bool(
            not accepted
            and mechanism_family
            and implementation_verdict == "underpowered"
            and hypothesis_verdict in {"inconclusive", "not_tested"}
            and prior_family_attempts >= 1
        )
        streak_before = int(state.get("inconclusive_streak") or 0)
        valid_inconclusive = bool(
            base_outcome == "inconclusive"
            and implementation_verdict != "execution_invalid"
        )
        forced_promotion_after_inconclusive = bool(
            valid_inconclusive and streak_before >= 1
        )
        if forced_promotion_after_inconclusive:
            accepted = True
            outcome = "accepted"
        promotion_blocked_by_audit = bool(
            accepted
            and implementation_verdict == "execution_invalid"
        )
        if promotion_blocked_by_audit:
            accepted = False
            outcome = "inconclusive"
        self._update_history_strategy_evidence(comparison_champion)
        self._append_history(
            state=state,
            batch=comparison_candidate,
            parent=mutation_parent,
            accepted=accepted,
            generation=int(state["generation"]) + 1,
        )
        if accepted:
            search_parent_after = candidate
            streak_after = 0
        elif valid_inconclusive:
            search_parent_after = candidate
            streak_after = streak_before + 1
        elif repairable_underpowered:
            # Keep Champion selection unchanged, but allow one implementation
            # repair to inherit the concrete candidate instead of rebuilding the
            # same mechanism from Champion. A second failed family attempt returns
            # to Champion through the normal rejected path.
            search_parent_after = candidate
            streak_after = 0
        elif underpowered_retry_exhausted:
            search_parent_after = champion
            streak_after = 0
        else:
            search_parent_after = str(state.get("search_parent") or mutation_parent)
            streak_after = 0 if base_outcome == "rejected" else streak_before
        decision = {
            "generation": state["generation"],
            "difficulty": difficulty,
            "parent": mutation_parent,
            "mutation_parent": mutation_parent,
            "comparison_champion": champion,
            "candidate": candidate,
            "parent_score": mutation_parent_result.score,
            "mutation_parent_score": mutation_parent_result.score,
            "candidate_score": comparison_candidate.score,
            "candidate_win_rate": comparison_candidate.win_rate,
            "champion_score": comparison_champion.score,
            "comparison_champion_score": comparison_champion.score,
            "champion_win_rate": comparison_champion.win_rate,
            "score_delta": score_delta,
            "delta": score_delta,
            "decision": outcome,
            "accepted": accepted,
            "mechanism_evidence": list(
                mechanism_audit.get("mechanism_evidence") or []
            ),
            "combat_evidence": list(
                mechanism_audit.get("combat_evidence") or []
            ),
            "runtime_findings": list(
                mechanism_audit.get("runtime_findings") or []
            ),
            "audit_evidence_limits": list(
                mechanism_audit.get("evidence_limits") or []
            ),
            "implementation_verdict": str(
                mechanism_audit.get("implementation_verdict") or "unknown"
            ),
            "hypothesis_verdict": str(
                mechanism_audit.get("hypothesis_verdict") or "inconclusive"
            ),
            "performance_result": outcome,
            "causal_result": hypothesis_verdict,
            "performance_gain_cause": performance_gain_cause,
            "posterior_probability_better": probability,
            "selection_rule": (
                "force_latest_candidate_after_two_consecutive_inconclusive"
                if forced_promotion_after_inconclusive
                else (
                    "candidate_win_rate_meets_mastery_threshold"
                    if candidate_mastered
                    else "candidate_score_strictly_greater"
                )
            ),
            "base_decision": base_outcome,
            "forced_promotion_after_inconclusive": (
                forced_promotion_after_inconclusive
            ),
            "search_parent_before": str(
                state.get("search_parent") or mutation_parent
            ),
            "search_parent_after": search_parent_after,
            "inconclusive_streak_before": streak_before,
            "inconclusive_streak_after": streak_after,
            "candidate_mastered": candidate_mastered,
            "mastery_win_rate_threshold": self.config.mastery_score_threshold,
            "posterior_used_for_selection": False,
            "promotion_blocked_by_audit": promotion_blocked_by_audit,
            "repairable_underpowered_retry": repairable_underpowered,
            "underpowered_retry_exhausted": underpowered_retry_exhausted,
            "champion_evidence_games": comparison_champion.games,
            "candidate_evidence_games": comparison_candidate.games,
            "evaluation_rounds": [
                {
                    "candidate_games": comparison_candidate.games,
                    "candidate_score": comparison_candidate.score,
                    "champion_games": comparison_champion.games,
                    "champion_score": comparison_champion.score,
                    "posterior_probability_better": probability,
                }
            ],
            "candidate_hash": str(pending.get("candidate_hash") or ""),
            "parent_batch": str(mutation_parent_result.path),
            "comparison_champion_batch": str(champion_batch.path),
            "candidate_batch": str(comparison_candidate.path),
            "comparison_games_per_strategy": comparison_candidate.games,
            "confirmation": confirmation,
            "experiment_spec": experiment_spec,
            "candidate_strategy_dir": str(pending.get("strategy_dir") or ""),
            "primary_change": str(pending.get("primary_change") or ""),
            "selected_plan_ids": _string_list(pending.get("selected_plan_ids")),
            "overall_assessment": str(pending.get("overall_assessment") or ""),
            "selected_changes": _dict_list(pending.get("selected_changes")),
            "expected_effect": str(
                experiment_spec.get("expected_effect")
                or pending.get("expected_effect")
                or ""
            ),
            "main_risk": str(
                experiment_spec.get("main_risk") or pending.get("main_risk") or ""
            ),
            "hypothesis": str(
                experiment_spec.get("hypothesis") or pending.get("hypothesis") or ""
            ),
            "mechanism_family": str(
                experiment_spec.get("mechanism_family")
                or pending.get("mechanism_family")
                or ""
            ),
            "mechanism_signature": canonical_mechanism_signature(
                experiment_spec.get("mechanism_family")
                or pending.get("mechanism_family")
            ),
            "mechanism_prediction": (
                dict(experiment_spec.get("mechanism_prediction"))
                if isinstance(experiment_spec.get("mechanism_prediction"), dict)
                else {}
            ),
            "failure_mode_analysis": (
                dict(experiment_spec.get("failure_mode_analysis"))
                if isinstance(experiment_spec.get("failure_mode_analysis"), dict)
                else {}
            ),
            "priority_alignment": (
                dict(experiment_spec.get("priority_alignment"))
                if isinstance(experiment_spec.get("priority_alignment"), dict)
                else {}
            ),
            "retrieval_assessment": (
                dict(experiment_spec.get("retrieval_assessment"))
                if isinstance(experiment_spec.get("retrieval_assessment"), dict)
                else {}
            ),
            "intervention_package": (
                dict(experiment_spec.get("intervention_package"))
                if isinstance(experiment_spec.get("intervention_package"), dict)
                else {}
            ),
            "plan_direction": str(experiment_spec.get("plan_direction") or ""),
            "patches": _dict_list(experiment_spec.get("patches")),
            "primary_lever": str(pending.get("primary_lever") or ""),
            "predictions": _string_list(pending.get("predictions")),
            "disproof_conditions": _string_list(pending.get("disproof_conditions")),
            "capability_mapping": (
                dict(pending.get("capability_mapping"))
                if isinstance(pending.get("capability_mapping"), dict)
                else {}
            ),
            "inheritance": (
                dict(pending.get("inheritance"))
                if isinstance(pending.get("inheritance"), dict)
                else {}
            ),
            "semantic_validation": (
                dict(pending.get("semantic_validation"))
                if isinstance(pending.get("semantic_validation"), dict)
                else {"status": "unknown", "errors": []}
            ),
        }
        generation_dir = self.run_dir / f"generation_{int(state['generation']):03d}"
        _write_json(generation_dir / "decision.json", decision)
        if accepted:
            state["champion"] = candidate
            self._sync_champion_baseline(state, comparison_candidate)
            self._sync_search_parent(state, candidate, comparison_candidate)
        elif valid_inconclusive or repairable_underpowered:
            self._sync_search_parent(state, candidate, comparison_candidate)
        elif underpowered_retry_exhausted:
            self._sync_search_parent(state, champion, comparison_champion)
        state["inconclusive_streak"] = streak_after
        parent_evidence = {
            **aggregate_outcomes([mutation_parent_result.to_dict()]),
            "strategy": mutation_parent,
            "difficulty": difficulty,
        }
        comparison_champion_evidence = {
            **aggregate_outcomes([comparison_champion.to_dict()]),
            "strategy": champion,
            "difficulty": difficulty,
        }
        candidate_evidence = {
            **aggregate_outcomes([comparison_candidate.to_dict()]),
            "strategy": candidate,
            "difficulty": difficulty,
        }
        if forced_promotion_after_inconclusive:
            lesson = (
                "Two consecutive valid candidates were statistically inconclusive. "
                "The latest candidate was promoted by the stagnation rule so the next "
                "generation continues from the newer strategy instead of remaining "
                "locked to the same Champion."
            )
        elif outcome == "accepted":
            lesson = (
                "The candidate was accepted, so the match outcome supports this "
                "hypothesis at this difficulty. Realized mechanism evidence should "
                "still be checked when available."
            )
        elif outcome == "rejected":
            lesson = (
                "This candidate combination was rejected, but candidate rejection "
                "alone does not contradict the causal hypothesis. The realized "
                "mechanism was not audited, so the hypothesis remains inconclusive. "
                "A stronger retry must explain the substantive intervention difference."
            )
        elif promotion_blocked_by_audit:
            lesson = (
                "The score improved, but the post-experiment audit found that "
                "the candidate depends on unavailable runtime behavior; it was "
                "not promoted and the hypothesis was not tested."
            )
        else:
            lesson = (
                "The evaluation scores were equal; this is not proof for "
                "or against the hypothesis, and the Champion is unchanged."
            )
        audit_lesson = str(mechanism_audit.get("lesson") or "").strip()
        if audit_lesson:
            lesson = audit_lesson
        experience = {
            "experiment_id": self._experiment_id(
                style=str(state.get("style") or self.config.strategy),
                generation=int(state["generation"]),
                difficulty=difficulty,
                candidate=candidate,
            ),
            "generation": int(state["generation"]),
            "difficulty": difficulty,
            "parent": mutation_parent,
            "mutation_parent": mutation_parent,
            "champion": champion,
            "comparison_champion": champion,
            "candidate": candidate,
            "hypothesis": str(decision.get("hypothesis") or ""),
            "mechanism_family": str(decision.get("mechanism_family") or ""),
            "mechanism_signature": str(
                decision.get("mechanism_signature")
                or canonical_mechanism_signature(decision.get("mechanism_family"))
            ),
            "mechanism_prediction": (
                dict(decision.get("mechanism_prediction"))
                if isinstance(decision.get("mechanism_prediction"), dict)
                else {}
            ),
            "failure_mode_analysis": (
                dict(decision.get("failure_mode_analysis"))
                if isinstance(decision.get("failure_mode_analysis"), dict)
                else {}
            ),
            "priority_alignment": (
                dict(decision.get("priority_alignment"))
                if isinstance(decision.get("priority_alignment"), dict)
                else {}
            ),
            "retrieval_assessment": (
                dict(decision.get("retrieval_assessment"))
                if isinstance(decision.get("retrieval_assessment"), dict)
                else {}
            ),
            "intervention_package": (
                dict(decision.get("intervention_package"))
                if isinstance(decision.get("intervention_package"), dict)
                else {}
            ),
            "mechanism_evidence": list(decision.get("mechanism_evidence") or []),
            "combat_evidence": list(decision.get("combat_evidence") or []),
            "runtime_findings": list(decision.get("runtime_findings") or []),
            "audit_evidence_limits": list(
                decision.get("audit_evidence_limits") or []
            ),
            "implementation_verdict": str(
                decision.get("implementation_verdict") or "unknown"
            ),
            "hypothesis_verdict": str(
                decision.get("hypothesis_verdict") or "inconclusive"
            ),
            "performance_result": outcome,
            "causal_result": hypothesis_verdict,
            "performance_gain_cause": performance_gain_cause,
            "plan_direction": str(decision.get("plan_direction") or ""),
            "patches": _dict_list(decision.get("patches")),
            "decision": outcome,
            "base_decision": base_outcome,
            "forced_promotion_after_inconclusive": (
                forced_promotion_after_inconclusive
            ),
            "repairable_underpowered_retry": repairable_underpowered,
            "underpowered_retry_exhausted": underpowered_retry_exhausted,
            "search_parent_before": str(decision.get("search_parent_before") or ""),
            "search_parent_after": search_parent_after,
            "inconclusive_streak_before": streak_before,
            "inconclusive_streak_after": streak_after,
            "candidate_hash": str(decision.get("candidate_hash") or ""),
            "primary_change": str(
                decision.get("primary_change") or "the candidate change"
            ),
            "selected_plan_ids": _string_list(decision.get("selected_plan_ids")),
            "overall_assessment": str(decision.get("overall_assessment") or ""),
            "selected_changes": _dict_list(decision.get("selected_changes")),
            "inheritance": (
                dict(decision.get("inheritance"))
                if isinstance(decision.get("inheritance"), dict)
                else {}
            ),
            "semantic_validation": (
                dict(decision.get("semantic_validation"))
                if isinstance(decision.get("semantic_validation"), dict)
                else {"status": "unknown", "errors": []}
            ),
            "expected_effect": str(decision.get("expected_effect") or ""),
            "main_risk": str(decision.get("main_risk") or ""),
            "parent_score": mutation_parent_result.score,
            "mutation_parent_score": mutation_parent_result.score,
            "comparison_champion_score": comparison_champion.score,
            "candidate_score": comparison_candidate.score,
            "score_delta": score_delta,
            "delta": score_delta,
            "champion_games": comparison_champion.games,
            "candidate_games": comparison_candidate.games,
            "posterior_probability_better": probability,
            "evaluation": {
                "champion": {
                    "wins": comparison_champion.wins,
                    "draws": comparison_champion.draws,
                    "losses": comparison_champion.losses,
                    "games": comparison_champion.games,
                    "score": comparison_champion.score,
                },
                "candidate": {
                    "wins": comparison_candidate.wins,
                    "draws": comparison_candidate.draws,
                    "losses": comparison_candidate.losses,
                    "games": comparison_candidate.games,
                    "score": comparison_candidate.score,
                },
                "score_delta": score_delta,
                "posterior": probability,
                "decision": outcome,
            },
            "experiment_evidence": {
                "parent_batch": parent_evidence,
                "comparison_champion_batch": comparison_champion_evidence,
                "candidate_batch": candidate_evidence,
                "candidate_minus_parent": {
                    "score_delta": (
                        comparison_candidate.score - mutation_parent_result.score
                    )
                },
                "candidate_minus_comparison_champion": {
                    "score_delta": score_delta
                },
                "comparison_used_confirmation": confirmation is not None,
                "confirmation_batches": confirmation,
            },
            "lesson": lesson,
        }
        appended = self._append_experiment_history(state, experience)
        if appended:
            state.setdefault("mechanism_ledger", []).append(
                {
                    "experiment_id": str(experience.get("experiment_id") or ""),
                    "generation": int(state["generation"]),
                    "difficulty": difficulty,
                    "mutation_parent": mutation_parent,
                    "comparison_champion": champion,
                    "candidate": candidate,
                    "mechanism_family": str(
                        decision.get("mechanism_family") or ""
                    ),
                    "mechanism_signature": str(
                        decision.get("mechanism_signature")
                        or canonical_mechanism_signature(
                            decision.get("mechanism_family")
                        )
                    ),
                    "inheritance": dict(experience.get("inheritance") or {}),
                    "base_decision": base_outcome,
                    "decision": outcome,
                    "implementation_verdict": implementation_verdict,
                    "hypothesis_verdict": hypothesis_verdict,
                    "performance_gain_cause": performance_gain_cause,
                    "repairable_underpowered_retry": repairable_underpowered,
                    "underpowered_retry_exhausted": underpowered_retry_exhausted,
                    "search_parent_before": str(
                        decision.get("search_parent_before") or ""
                    ),
                    "search_parent_after": search_parent_after,
                    "inconclusive_streak_before": streak_before,
                    "inconclusive_streak_after": streak_after,
                }
            )
        pending["experiment_committed"] = True
        state["pending_candidate"] = None
        if appended:
            state["generation"] = int(state.get("generation") or 0) + 1
            state["difficulty_generation"] = (
                int(state.get("difficulty_generation") or 0) + 1
            )
        if accepted and self._is_mastered(comparison_candidate):
            self._advance_difficulty(state)


__all__ = [
    "BatchResult",
    "DEFAULT_DIFFICULTIES",
    "EvolutionConfig",
    "EvolutionRunner",
    "close_batch_results",
    "combine_batch_results",
    "completed_record_count",
    "read_batch_result",
]
