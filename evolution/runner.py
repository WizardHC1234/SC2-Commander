from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from evol_agent.optimization.snapshot import (
    output_dir_for_strategy,
    strategy_content_hash,
)
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
ANALYSIS_EXPERIENCE_MODES = ("multi_match", "single_failure")
MAX_SEARCH_RESTARTS_PER_GENERATION = 3
MAX_CONSECUTIVE_SEARCH_RESTARTS = 6
MAX_BATCH_EXECUTION_ATTEMPTS = 3
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

def canonical_mechanism_signature(value: Any) -> str:
    """Normalize a model-provided family id without guessing semantic equivalence.

    Whether two differently named mechanisms express the same causal intervention
    is a semantic judgement made from experiment history by the analysis model.
    The runner only normalizes spelling for stable storage and exact-id lookup.
    """
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")


def canonical_material_change_signature(value: Any) -> str:
    """Normalize an explicitly pre-registered material behavior change.

    Semantic equivalence remains the analysis model's responsibility.  This exact
    normalized fallback prevents a model from replaying the same intervention while
    changing only its mechanism-family label (for example ``_v2``/``_v3``).
    """

    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _match_result(record_path: Path) -> str:
    """Read the final outcome used only for reproducible ablation sampling."""
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return "unknown"
    metadata = payload.get("metadata") if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        return "unknown"
    return str(metadata.get("result") or "unknown").strip().casefold()


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
    analysis_experience_mode: str = "multi_match"
    analysis_sample_seed: int = 0
    max_generations_per_difficulty: int = 10
    max_total_generations: int = 50
    require_full_generation_budget: bool = False
    knowledge_mode: str = "enabled"
    bot_name: str = "commander"
    bot_instruct: str = ""
    real_time: bool = False
    baseline_batch_dir: str = ""
    records_dir: str = "game_records"

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
        if self.analysis_experience_mode not in ANALYSIS_EXPERIENCE_MODES:
            raise ValueError(
                "analysis_experience_mode must be one of: "
                + ", ".join(ANALYSIS_EXPERIENCE_MODES)
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
    while "__" in text:
        text = text.replace("__", "_")
    trimmed = text[:limit].rstrip("_")
    return trimmed or "run"


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


def completed_record_indices(batch_dir: Path, *, strategy: str) -> set[int]:
    """Return batch run indices that have a completed record.

    Current match directories end in ``_rN``.  Older/imported records may not
    expose a run index; callers must therefore compare the returned set with the
    completed-record count before relying on it to identify holes.
    """
    indices: set[int] = set()
    if not batch_dir.exists():
        return indices
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
        match = re.search(r"_r(\d+)$", path.parent.name)
        if match:
            indices.add(int(match.group(1)))
    return indices


def contiguous_index_ranges(indices: list[int] | set[int]) -> list[tuple[int, int]]:
    """Collapse sorted indices into ``(start_index, total_matches)`` ranges."""
    ordered = sorted({int(index) for index in indices})
    if not ordered:
        return []
    ranges: list[tuple[int, int]] = []
    start = ordered[0]
    count = 1
    for index in ordered[1:]:
        if index == start + count:
            count += 1
            continue
        ranges.append((start, count))
        start = index
        count = 1
    ranges.append((start, count))
    return ranges


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
        records_path = Path(config.records_dir).expanduser()
        self.records_root = (
            records_path if records_path.is_absolute() else self.project_root / records_path
        ).resolve()
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

    def _mechanism_audit_enabled(self) -> bool:
        # Normal CLI runs use the built-in auditor. Unit/integration harnesses that
        # inject a batch executor may explicitly omit it when testing score mechanics.
        return self._experiment_auditor is not None or self._batch_executor is None

    def _new_state(self) -> dict[str, Any]:
        if self._mechanism_audit_enabled():
            selection_protocol = (
                "confirmed_score_and_realized_mechanism_v3"
                if self.config.confirmation_matches
                else "score_and_realized_mechanism_v3"
            )
        else:
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
            "generation_semantics": "completed_candidate_evaluations_v1",
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
            "generation_search_restarts": [],
            "consecutive_search_restarts": 0,
            "exhausted_search_cycles": [],
            "skipped_generations": [],
            "abandoned_analysis_checkpoints": [],
            "candidate_resume_dir": None,
            "analysis_checkpoints": {},
            "analysis_input_history": [],
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
                "records_dir",
                "analysis_batch_games",
                "max_analysis_games_per_generation",
                "max_generations_per_difficulty",
                "confirmation_matches",
                "require_full_generation_budget",
                "analysis_experience_mode",
                "analysis_sample_seed",
            ):
                if key not in saved:
                    saved[key] = current[key]
                    config_changed = True
            # These are run-control budgets rather than experiment identity.
            # Allow a stopped run to resume with safer concurrency or a revised
            # generation budget while keeping strategy/model/map settings fixed.
            for key in (
                "concurrency",
                "candidate_generation_retries",
                "max_total_generations",
                "max_generations_per_difficulty",
                "require_full_generation_budget",
            ):
                if saved.get(key) != current[key]:
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
                    add_path(self._batch_dir_for(name), strategy, difficulty)

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
                        or (
                            key[1]
                            if decision == "accepted"
                            else item.get("comparison_champion")
                            or item.get("champion")
                            or item.get("parent")
                            or ""
                        )
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
        if state.get("generation_semantics") != "completed_candidate_evaluations_v1":
            experiments = [
                item
                for item in (state.get("experiment_history") or [])
                if isinstance(item, dict) and str(item.get("candidate") or "").strip()
            ]
            completed_total = len(experiments)
            current_difficulty = str(state.get("difficulty") or "")
            completed_here = sum(
                1
                for item in experiments
                if str(item.get("difficulty") or "") == current_difficulty
            )
            state["generation"] = completed_total
            state["difficulty_generation"] = completed_here
            state["generation_semantics"] = "completed_candidate_evaluations_v1"
            if (
                state.get("status") == "completed"
                and state.get("completion_reason") == "generation_budget_reached"
                and completed_total < self.config.max_total_generations
            ):
                state["status"] = "running"
                state.pop("completion_reason", None)
            changed = True
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
        if self._mechanism_audit_enabled():
            selection_protocol = (
                "confirmed_score_and_realized_mechanism_v3"
                if self.config.confirmation_matches
                else "score_and_realized_mechanism_v3"
            )
        else:
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
        # There is exactly one textual parent: the official Champion. Rejected or
        # equal-score candidates remain experiment evidence, but their unverified
        # text is never inherited by the next generation.
        champion_name = str(state.get("champion") or self.config.strategy)
        if str(state.get("search_parent") or "") != champion_name:
            state["search_parent"] = champion_name
            state["search_parent_batch"] = None
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
        if not isinstance(state.get("analysis_input_history"), list):
            state["analysis_input_history"] = []
            changed = True
        if not isinstance(state.get("analysis_checkpoints"), dict):
            state["analysis_checkpoints"] = {}
            changed = True
        if not isinstance(state.get("consecutive_search_restarts"), int):
            state["consecutive_search_restarts"] = 0
            changed = True
        for key in (
            "generation_search_restarts",
            "skipped_generations",
            "abandoned_analysis_checkpoints",
        ):
            if not isinstance(state.get(key), list):
                state[key] = []
                changed = True
        pending = state.get("pending_candidate")
        if isinstance(pending, dict):
            strategy_dir = Path(str(pending.get("strategy_dir") or ""))
            if not strategy_dir.is_dir():
                checkpoint = str(
                    pending.get("analysis_checkpoint_dir") or ""
                ).strip()
                if checkpoint:
                    abandoned = state.setdefault(
                        "abandoned_analysis_checkpoints", []
                    )
                    resolved_checkpoint = str(Path(checkpoint).resolve())
                    if resolved_checkpoint not in abandoned:
                        abandoned.append(resolved_checkpoint)
                state["pending_candidate"] = None
                state["candidate_resume_dir"] = None
                state["status"] = "running"
                changed = True
        resumable_statuses = {
            "evol_agent_failed",
            "mechanism_policy_attention_required",
            "runtime_attention_required",
            "stopped_no_actionable_improvement",
            "insufficient_evidence",
            "agent_paused",
        }
        if (
            str(state.get("status") or "") in resumable_statuses
            and state.get("pending_candidate") is None
            and int(state.get("generation") or 0) < self.config.max_total_generations
        ):
            state["status"] = "running"
            state["completion_reason"] = ""
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
        abandoned = {
            str(Path(str(path)).resolve())
            for path in (state.get("abandoned_analysis_checkpoints") or [])
            if str(path).strip()
        }
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
            if key in abandoned:
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
        state: dict[str, Any] | None = None,
        strategy: str,
        record_paths: list[Path],
    ) -> Path | None:
        current_records = {str(path.resolve()) for path in record_paths}
        abandoned = {
            str(Path(str(path)).resolve())
            for path in ((state or {}).get("abandoned_analysis_checkpoints") or [])
            if str(path).strip()
        }
        log_root = self.project_root / "evol_agent" / "logs" / strategy
        if not current_records or not log_root.is_dir():
            return None
        resumable: list[tuple[int, Path]] = []
        for path in log_root.iterdir():
            if not path.is_dir():
                continue
            if str(path.resolve()) in abandoned:
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
        durable = [
            item
            for item in history
            if str(item.get("decision") or "").strip().lower() == "accepted"
            or (
                str(item.get("difficulty") or "") in {"", difficulty}
                and (
                    str(item.get("hypothesis_verdict") or "").strip().lower()
                    == "contradicted"
                    or str(item.get("implementation_verdict") or "").strip().lower()
                    == "execution_invalid"
                    or item.get("underpowered_retry_exhausted") is True
                )
            )
        ]
        # Preserve all accepted Champion improvements and terminal failed causal
        # directions, then add recent same-difficulty experiments for local context.
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        for item in [*durable, *related[-8:]]:
            key = str(item.get("experiment_id") or "").strip() or str(id(item))
            if key not in selected_ids:
                selected.append(item)
                selected_ids.add(key)
        latest_restart = [
            item
            for item in (state.get("generation_search_restarts") or [])
            if isinstance(item, dict)
            and str(item.get("difficulty") or "") in {"", difficulty}
        ][-1:]
        visible = selected + latest_restart
        if self.config.analysis_experience_mode != "single_failure":
            return visible

        # Keep outcome-level evolution history, but do not leak the other nine
        # trajectories from the post-experiment audit into the next analysis.
        sanitized: list[dict[str, Any]] = []
        hidden_fields = (
            "mechanism_evidence",
            "combat_evidence",
            "runtime_findings",
            "salvageable_changes",
            "failed_dependencies",
            "audit_evidence_limits",
        )
        for item in visible:
            copied = json.loads(json.dumps(item, ensure_ascii=False))
            for field in hidden_fields:
                copied[field] = []
            copied["gate_execution_audit"] = {}
            copied["lesson"] = (
                "Match-level post-experiment evidence is withheld by the "
                "single_failure ablation; use only the intervention and aggregate "
                "outcome scores."
            )
            inheritance = copied.get("inheritance")
            if isinstance(inheritance, dict):
                for change in inheritance.get("verified_changes") or []:
                    if isinstance(change, dict):
                        change.pop("evidence", None)
            copied["analysis_visibility"] = "aggregate_outcomes_only"
            sanitized.append(copied)
        return sanitized

    def _select_analysis_record_paths(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
        strategy: str,
        record_paths: list[Path],
    ) -> list[Path]:
        """Select trajectory evidence visible to EvolAgent for this generation."""
        unique_paths = list(
            dict.fromkeys(path.resolve() for path in record_paths if path.is_file())
        )
        if self.config.analysis_experience_mode == "multi_match":
            return unique_paths
        if not unique_paths:
            return []

        generation = int(state.get("generation") or 0)
        history = state.setdefault("analysis_input_history", [])
        for item in reversed(history):
            if not isinstance(item, dict):
                continue
            if (
                int(item.get("generation") or -1) != generation
                or str(item.get("difficulty") or "") != difficulty
                or str(item.get("strategy") or "") != strategy
                or str(item.get("mode") or "") != "single_failure"
            ):
                continue
            selected = Path(str(item.get("selected_record") or ""))
            if selected.is_file() and selected.resolve() in unique_paths:
                return [selected.resolve()]

        outcomes = [(path, _match_result(path)) for path in sorted(unique_paths)]
        defeats = [path for path, result in outcomes if result in {"defeat", "loss"}]
        non_wins = [
            path
            for path, result in outcomes
            if result in {"defeat", "loss", "tie", "draw"}
        ]
        if defeats:
            pool = defeats
            selection_reason = "fixed_seed_failure_sample"
        elif non_wins:
            pool = non_wins
            selection_reason = "fixed_seed_non_win_fallback"
        else:
            # A perfect batch normally advances before analysis. This keeps runs
            # reproducible at a mastered final difficulty with a forced full budget.
            pool = [path for path, _result in outcomes]
            selection_reason = "no_failure_available_fallback"
        token = (
            f"{self.config.analysis_sample_seed}|{generation}|"
            f"{difficulty}|{strategy}"
        )
        index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % len(pool)
        selected = pool[index]
        selected_result = next(
            (result for path, result in outcomes if path == selected), "unknown"
        )
        history.append(
            {
                "generation": generation,
                "difficulty": difficulty,
                "strategy": strategy,
                "mode": "single_failure",
                "sample_seed": int(self.config.analysis_sample_seed),
                "available_records": len(unique_paths),
                "available_defeats": len(defeats),
                "selected_record": str(selected),
                "selected_result": selected_result,
                "selection_reason": selection_reason,
                "created_at": datetime.now().isoformat(),
            }
        )
        print(
            "Ablation analysis input: single_failure selected "
            f"{selected.name} ({selected_result}) from {len(unique_paths)} records",
            flush=True,
        )
        return [selected]

    def _blocked_mechanism_families(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
    ) -> dict[str, str]:
        inconclusive_attempts: dict[str, int] = {}
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
            if (
                decision != "accepted"
                and implementation != "execution_invalid"
                and hypothesis in {"inconclusive", "not_tested"}
            ):
                inconclusive_attempts[signature] = (
                    inconclusive_attempts.get(signature, 0) + 1
                )
            if (
                decision != "accepted"
                and implementation == "implemented"
                and hypothesis == "contradicted"
            ):
                blocked[representative[signature]] = (
                    "implemented experiment contradicted the hypothesis"
                )
            if bool(item.get("underpowered_retry_exhausted")):
                blocked.setdefault(
                    representative[signature],
                    "underpowered retry budget was exhausted",
                )
        for signature, count in inconclusive_attempts.items():
            family = representative[signature]
            if count >= 2:
                blocked.setdefault(
                    family,
                    f"already has {count} underpowered tests without realizing the mechanism",
                )
        return blocked

    def _blocked_material_changes(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
    ) -> dict[str, str]:
        """Return exact material interventions that must not be replayed.

        The semantic validator still judges paraphrases and genuine repairs.  This
        deterministic guard handles the common failure mode where the analysis model
        emits the identical minimum material change under a renamed family.
        """

        blocked: dict[str, str] = {}
        for item in state.get("experiment_history") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("difficulty") or "") not in {"", difficulty}:
                continue
            prediction = item.get("mechanism_prediction")
            if not isinstance(prediction, dict):
                continue
            material_change = str(
                prediction.get("minimum_material_change") or ""
            ).strip()
            signature = canonical_material_change_signature(material_change)
            if not signature:
                continue
            implementation = str(item.get("implementation_verdict") or "unknown")
            hypothesis = str(item.get("hypothesis_verdict") or "inconclusive")
            if implementation != "implemented":
                blocked.setdefault(
                    signature,
                    "the same material change was not realized in prior matches",
                )
            elif hypothesis == "contradicted":
                blocked.setdefault(
                    signature,
                    "the same implemented material change was contradicted",
                )
        return blocked

    def _restart_generation_search(
        self,
        state: dict[str, Any],
        *,
        difficulty: str,
        champion: str,
        reason: str,
        source_action: str,
        checkpoint_dir: Path | str | None = None,
        mechanism_family: str = "",
    ) -> bool:
        """Retry search without consuming a completed-candidate generation."""
        checkpoint_value = str(checkpoint_dir or state.get("candidate_resume_dir") or "").strip()
        if checkpoint_value:
            resolved = str(Path(checkpoint_value).resolve())
            abandoned = state.setdefault("abandoned_analysis_checkpoints", [])
            if resolved not in abandoned:
                abandoned.append(resolved)

        generation = int(state.get("generation") or 0)
        restart = {
            "kind": "generation_search_restart",
            "generation": generation,
            "difficulty": difficulty,
            "champion": champion,
            "source_action": source_action,
            "reason": str(reason or "search attempt did not produce an evaluable candidate"),
            "mechanism_family": str(mechanism_family or "").strip(),
            "instruction": (
                "Keep the current Champion and select a materially different, "
                "evidence-supported strategy-fixable mechanism. Do not repeat the "
                "failed concrete package or its unavailable dependency."
            ),
            "created_at": datetime.now().isoformat(),
        }
        restarts = state.setdefault("generation_search_restarts", [])
        restarts.append(restart)
        consecutive_restarts = int(state.get("consecutive_search_restarts") or 0) + 1
        state["consecutive_search_restarts"] = consecutive_restarts
        restart_count = sum(
            1
            for item in restarts
            if isinstance(item, dict)
            and int(item.get("generation", -1)) == generation
            and str(item.get("difficulty") or "") == difficulty
        )

        if consecutive_restarts >= MAX_CONSECUTIVE_SEARCH_RESTARTS:
            state["status"] = "candidate_search_blocked"
            state["candidate_search_blocked_reason"] = restart["reason"]
            state["pending_candidate"] = None
            state["candidate_resume_dir"] = None
            print(
                "EvolAgent candidate search reached the bounded retry limit without "
                "a basic-valid candidate; stopping this run instead of repeating "
                "the same analysis indefinitely.",
                flush=True,
            )
            self._save_state(state)
            return False

        state["status"] = "running"
        state["pending_candidate"] = None
        state["candidate_resume_dir"] = None
        state["inconclusive_streak"] = 0
        try:
            champion_evidence = self._aggregate_evidence(
                state, difficulty=difficulty, strategy=champion
            )
        except RuntimeError:
            champion_evidence = (
                BatchResult.from_dict(state["champion_batch"])
                if isinstance(state.get("champion_batch"), dict)
                else None
            )
        self._sync_search_parent(state, champion, champion_evidence)

        restart_in_cycle = (
            (restart_count - 1) % MAX_SEARCH_RESTARTS_PER_GENERATION
        ) + 1
        if restart_in_cycle >= MAX_SEARCH_RESTARTS_PER_GENERATION:
            state.setdefault("exhausted_search_cycles", []).append(
                {
                    "generation": generation,
                    "difficulty": difficulty,
                    "champion": champion,
                    "reason": "search_cycle_exhausted_without_evaluable_candidate",
                    "search_cycle": (
                        (restart_count - 1) // MAX_SEARCH_RESTARTS_PER_GENERATION
                    ) + 1,
                    "restart_count_total": restart_count,
                    "last_failure": restart["reason"],
                    "created_at": datetime.now().isoformat(),
                }
            )
            print(
                "EvolAgent exhausted one candidate-search cycle; the Champion is "
                "retained and a fresh search cycle starts for the same unevaluated "
                "generation.",
                flush=True,
            )
        else:
            print(
                "EvolAgent search attempt was not evaluable; retaining the "
                f"Champion and trying a different mechanism ({restart_in_cycle}/"
                f"{MAX_SEARCH_RESTARTS_PER_GENERATION}).",
                flush=True,
            )
        self._save_state(state)
        return True

    def _blocked_mechanism_reason(
        self,
        blocked: dict[str, str],
        candidate_family: str,
    ) -> tuple[str, str]:
        candidate_signature = canonical_mechanism_signature(candidate_family)
        for family, reason in blocked.items():
            if canonical_mechanism_signature(family) == candidate_signature:
                return family, reason
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
        if "intervention_package" not in spec:
            intervention_package = rationale.get("intervention_package")
            spec["intervention_package"] = (
                dict(intervention_package)
                if isinstance(intervention_package, dict)
                else {}
            )
        spec.setdefault(
            "selected_package_id", str(rationale.get("selected_package_id") or "")
        )
        if "selected_timing_budget" not in spec:
            selected_timing_budget = rationale.get("selected_timing_budget")
            spec["selected_timing_budget"] = (
                dict(selected_timing_budget)
                if isinstance(selected_timing_budget, dict)
                else {}
            )
        if "selected_package_budget" not in spec:
            selected_package_budget = rationale.get("selected_package_budget")
            spec["selected_package_budget"] = (
                dict(selected_package_budget)
                if isinstance(selected_package_budget, dict)
                else {}
            )
        if "candidate_package_evaluations" not in spec:
            evaluations = rationale.get("candidate_package_evaluations")
            spec["candidate_package_evaluations"] = _dict_list(evaluations)
        if "selected_history_assessment" not in spec:
            selected_history = rationale.get("selected_history_assessment")
            spec["selected_history_assessment"] = (
                dict(selected_history)
                if isinstance(selected_history, dict)
                else {}
            )
        if "first_commitment_timing" not in spec:
            feasibility = rationale.get("deterministic_feasibility_audit")
            timing = (
                feasibility.get("contact_timing_report")
                if isinstance(feasibility, dict)
                and isinstance(feasibility.get("contact_timing_report"), dict)
                else {}
            )
            spec["first_commitment_timing"] = dict(timing)
        if "mechanism_equivalence_audit" not in spec:
            feasibility = rationale.get("deterministic_feasibility_audit")
            audit = (
                feasibility.get("mechanism_equivalence_audit")
                if isinstance(feasibility, dict)
                and isinstance(feasibility.get("mechanism_equivalence_audit"), dict)
                else {}
            )
            spec["mechanism_equivalence_audit"] = dict(audit)
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
        if "document_changes" not in spec:
            changes = rationale.get("document_changes")
            if not isinstance(changes, list) or not changes:
                changes = rationale.get("patches")
            if not isinstance(changes, list) or not changes:
                changes = rationale.get("selected_changes")
            spec["document_changes"] = _dict_list(changes)
        spec.setdefault("patches", list(spec.get("document_changes") or []))
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

    def _discard_policy_rejected_candidate(self, candidate_dir: Path) -> bool:
        """Remove an unplayed candidate rejected before match evaluation."""
        candidate_dir = Path(candidate_dir).resolve()
        strategies_dir = self.strategies_dir.resolve()
        if candidate_dir.parent != strategies_dir:
            print(
                "EvolAgent did not remove a policy-rejected candidate outside "
                f"the run strategy directory: {candidate_dir}",
                flush=True,
            )
            return False
        if not candidate_dir.exists():
            return True
        if candidate_dir.is_symlink():
            candidate_dir.unlink()
        elif candidate_dir.is_dir():
            shutil.rmtree(candidate_dir)
        else:
            candidate_dir.unlink()
        return True

    def _historical_strategy_duplicate(
        self,
        state: dict[str, Any],
        *,
        candidate_dir: Path,
        current_difficulty: str,
    ) -> dict[str, Any] | None:
        """Find an evaluated historical strategy with identical normalized text."""

        candidate_path = Path(candidate_dir) / "strategy.md"
        if not candidate_path.is_file():
            return None
        candidate_hash = strategy_content_hash(
            candidate_path.read_text(encoding="utf-8-sig")
        )
        matches: list[dict[str, Any]] = []
        pool = state.get("evidence_pool") or {}
        for difficulty, by_strategy in pool.items():
            if not isinstance(by_strategy, dict):
                continue
            for strategy, raw_entries in by_strategy.items():
                entries = [item for item in raw_entries if isinstance(item, dict)]
                if not entries:
                    continue
                batch_dirs = [Path(str(item.get("path") or "")) for item in entries]
                try:
                    historical_text = self._read_strategy_text(
                        str(strategy), batch_dirs=batch_dirs
                    )
                except (OSError, ValueError):
                    continue
                if strategy_content_hash(historical_text) != candidate_hash:
                    continue
                wins = sum(int(item.get("wins") or 0) for item in entries)
                draws = sum(int(item.get("draws") or 0) for item in entries)
                losses = sum(int(item.get("losses") or 0) for item in entries)
                games = wins + draws + losses
                matches.append(
                    {
                        "strategy": str(strategy),
                        "difficulty": str(difficulty),
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                        "games": games,
                        "win_rate": wins / games if games else 0.0,
                    }
                )
        if not matches:
            return None
        matches.sort(
            key=lambda item: (
                str(item.get("difficulty") or "") != current_difficulty,
                str(item.get("strategy") or ""),
            )
        )
        return {"candidate_hash": candidate_hash, "matches": matches}

    @staticmethod
    def _historical_duplicate_feedback(duplicate: dict[str, Any]) -> str:
        closest = duplicate["matches"][0]
        return (
            "analysis_replan_required: generated strategy duplicates the "
            f"evaluated historical strategy {closest['strategy']} "
            f"(difficulty={closest['difficulty']}, result="
            f"{closest['wins']}-{closest['draws']}-{closest['losses']}, "
            f"win_rate={closest['win_rate']:.3f}). Select a genuinely different "
            "evidence-supported intervention; do not restore an earlier complete "
            "strategy document."
        )

    def _reject_unplayed_pending_historical_duplicate(
        self,
        state: dict[str, Any],
        *,
        pending: dict[str, Any],
        difficulty: str,
        champion: str,
    ) -> bool:
        """Reject a duplicate restored from state only when no match has begun."""

        candidate = str(pending.get("strategy") or "")
        if not candidate or self._analysis_games(
            state, difficulty=difficulty, strategy=candidate
        ):
            return False
        candidate_batch_dir = self._batch_dir_for(
            self._batch_name(int(state.get("generation") or 0), "cand", difficulty)
        )
        if completed_record_count(candidate_batch_dir, strategy=candidate):
            return False
        candidate_dir = Path(str(pending.get("strategy_dir") or ""))
        duplicate = self._historical_strategy_duplicate(
            state,
            candidate_dir=candidate_dir,
            current_difficulty=difficulty,
        )
        if duplicate is None:
            return False
        message = self._historical_duplicate_feedback(duplicate)
        removed = self._discard_policy_rejected_candidate(candidate_dir)
        state.setdefault("candidate_generation_failures", []).append(
            {
                "kind": "candidate_generation_failure",
                "failure_reason": "historical_strategy_duplicate",
                "generation": int(state.get("generation") or 0),
                "attempt": 0,
                "max_attempts": self.config.candidate_generation_retries + 1,
                "difficulty": difficulty,
                "parent": str(pending.get("mutation_parent") or champion),
                "comparison_champion": champion,
                "message": message,
                "candidate_hash": duplicate["candidate_hash"],
                "duplicate_matches": duplicate["matches"],
                "candidate_removed": removed,
                "checkpoint_dir": str(
                    pending.get("analysis_checkpoint_dir") or ""
                ),
                "created_at": datetime.now().isoformat(),
            }
        )
        return self._restart_generation_search(
            state,
            difficulty=difficulty,
            champion=champion,
            reason=message,
            source_action="historical_strategy_duplicate",
            checkpoint_dir=pending.get("analysis_checkpoint_dir"),
        )

    def _batch_name(self, generation: int, role: str, difficulty: str = "") -> str:
        # Include difficulty so a mastered champion baseline is not reused as
        # the next curriculum level (same generation/role, different AI).
        parts = [f"ev_{self.run_id}", f"g{generation:03d}"]
        if difficulty:
            parts.append(difficulty)
        parts.append(role)
        return _safe_name("_".join(parts), 56)

    def _batch_dir_for(self, batch_name: str) -> Path:
        """Resolve the on-disk batch folder, collapsing duplicate underscores.

        ``run_vs_ai`` writes into a collapsed slug, so a truncated run id that
        ends with ``_`` must not look for a sibling ``__`` directory.
        """
        collapsed = self.records_root / _safe_name(batch_name, 56)
        if collapsed.is_dir():
            return collapsed
        return self.records_root / batch_name

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
        batch_dir = self._batch_dir_for(batch_name)
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
                "-OUTPUT_BASE_DIR", str(self.records_root),
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
                "--output-base-dir", str(self.records_root),
            ]
            if self.config.bot_instruct:
                command.extend(["--bot-instruct", self.config.bot_instruct])
        total_flag = "-TOTAL_MATCHES" if os.name == "nt" else "--total-matches"
        start_flag = "-START_INDEX" if os.name == "nt" else "--start-index"
        concurrency_flag = "-CONCURRENCY" if os.name == "nt" else "--concurrency"
        last_return_code = 0
        for batch_attempt in range(1, MAX_BATCH_EXECUTION_ATTEMPTS + 1):
            completed_indices = completed_record_indices(
                batch_dir, strategy=strategy
            )
            indices_are_complete = len(completed_indices) == completed
            if completed > 0 and indices_are_complete:
                missing_indices = [
                    index
                    for index in range(expected_games)
                    if index not in completed_indices
                ]
                # Fill contiguous holes in one start_batch call so concurrency
                # applies; run disjoint hole ranges in parallel.
                invocations = contiguous_index_ranges(missing_indices)
            else:
                # Compatibility with imported records that predate run-indexed
                # match directories. Such records are assumed to form a prefix.
                invocations = [(completed, expected_games - completed)]

            def _execute_invocation(
                start_index: int,
                total_matches: int,
                *,
                force_serial_matches: bool,
            ) -> int:
                cmd = list(command)
                cmd[cmd.index(total_flag) + 1] = str(total_matches)
                cmd[cmd.index(start_flag) + 1] = str(start_index)
                if force_serial_matches:
                    cmd[cmd.index(concurrency_flag) + 1] = "1"
                process = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    check=False,
                    env=self._strategy_env(),
                )
                return int(getattr(process, "returncode", 0) or 0)

            if len(invocations) <= 1:
                for start_index, total_matches in invocations:
                    last_return_code = _execute_invocation(
                        start_index,
                        total_matches,
                        force_serial_matches=False,
                    )
                    self._copy_strategy_sidecar(strategy, batch_dir)
            else:
                workers = min(len(invocations), max(1, int(self.config.concurrency)))
                print(
                    f"Batch {batch_name}: filling {len(invocations)} hole ranges "
                    f"in parallel (workers={workers}).",
                    flush=True,
                )
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(
                            _execute_invocation,
                            start_index,
                            total_matches,
                            force_serial_matches=True,
                        )
                        for start_index, total_matches in invocations
                    ]
                    for future in as_completed(futures):
                        code = int(future.result() or 0)
                        if code:
                            last_return_code = code
                self._copy_strategy_sidecar(strategy, batch_dir)
            observed = completed_record_count(batch_dir, strategy=strategy)
            if observed == expected_games:
                completed = observed
                break
            if observed > expected_games:
                raise RuntimeError(
                    f"batch {batch_name} has {observed} completed records, more than "
                    f"the requested {expected_games}"
                )
            if observed <= completed and last_return_code == 0:
                raise RuntimeError(
                    f"batch {batch_name} reported success but produced no new "
                    "completed match record"
                )
            completed = observed
            if batch_attempt < MAX_BATCH_EXECUTION_ATTEMPTS:
                print(
                    f"Batch {batch_name} completed {completed}/{expected_games} "
                    "matches; retrying only the missing matches "
                    f"({batch_attempt + 1}/{MAX_BATCH_EXECUTION_ATTEMPTS}).",
                    flush=True,
                )
        if completed != expected_games:
            raise subprocess.CalledProcessError(last_return_code or 1, command)
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
        analysis_record_paths: list[Path] | None = None,
    ) -> EvolRunResult:
        if self._candidate_generator is not None:
            return self._candidate_generator(champion, champion_batch, prior_experiences)
        record_paths: list[Path] = []
        if analysis_record_paths is not None:
            record_paths.extend(analysis_record_paths)
        else:
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

    def _advance_difficulty(
        self,
        state: dict[str, Any],
        *,
        mastered: bool = True,
    ) -> None:
        difficulty = self._current_difficulty(state)
        if difficulty and mastered:
            mastered_difficulties = state.setdefault("mastered_difficulties", [])
            if difficulty not in mastered_difficulties:
                mastered_difficulties.append(difficulty)
        elif difficulty:
            exhausted = state.setdefault("exhausted_difficulties", [])
            if difficulty not in exhausted:
                exhausted.append(difficulty)

        current_index = int(state.get("difficulty_index") or 0)
        next_index = current_index + 1
        if next_index >= len(self.config.difficulties):
            if not self.config.require_full_generation_budget:
                state["difficulty_index"] = next_index
                self._sync_champion_baseline(state, None)
                self._sync_search_parent(
                    state, str(state.get("champion") or ""), None
                )
                state["inconclusive_streak"] = 0
                self._reset_generation_local_analysis_state(state)
                state["difficulty_generation"] = 0
                self._complete_curriculum(state)
                return
            # The configured experiment requires all evolution rounds.  Once
            # the curriculum ends, keep evaluating on its strongest level
            # until the total generation budget is genuinely exhausted.
            state["curriculum_completed"] = bool(mastered)
            state["difficulty_index"] = current_index
            state["difficulty"] = self.config.difficulties[current_index]
            state["difficulty_generation"] = 0
            state["inconclusive_streak"] = 0
            self._reset_generation_local_analysis_state(state)
            return

        state["difficulty_index"] = next_index
        self._sync_champion_baseline(state, None)
        self._sync_search_parent(state, str(state.get("champion") or ""), None)
        state["inconclusive_streak"] = 0
        self._reset_generation_local_analysis_state(state)
        state["difficulty_generation"] = 0
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
            already_mastered_final = bool(
                self.config.require_full_generation_budget
                and int(state.get("difficulty_index") or 0)
                == len(self.config.difficulties) - 1
                and difficulty in (state.get("mastered_difficulties") or [])
            )
            if self._is_mastered(champion_batch) and not already_mastered_final:
                self._advance_difficulty(state)
                self._save_state(state)
                continue

            if int(state.get("generation") or 0) >= self.config.max_total_generations:
                state["status"] = "completed"
                state["completion_reason"] = "generation_budget_reached"
                self._save_state(state)
                break
            if (
                int(state.get("difficulty_generation") or 0)
                >= self.config.max_generations_per_difficulty
            ):
                if not self.config.require_full_generation_budget:
                    state["status"] = "difficulty_budget_exhausted"
                    state["failed_difficulty"] = difficulty
                    state["champion_score"] = champion_batch.score
                    self._save_state(state)
                    break
                state.setdefault("difficulty_budget_events", []).append(
                    {
                        "difficulty": difficulty,
                        "generation": int(state.get("generation") or 0),
                        "champion": champion,
                        "champion_score": champion_batch.score,
                        "action": (
                            "advance_without_mastery"
                            if int(state.get("difficulty_index") or 0)
                            < len(self.config.difficulties) - 1
                            else "continue_final_difficulty"
                        ),
                        "created_at": datetime.now().isoformat(),
                    }
                )
                self._advance_difficulty(state, mastered=False)
                self._save_state(state)
                continue

            pending = state.get("pending_candidate")
            if not isinstance(pending, dict):
                if not self._analyze_champion(state, difficulty, champion, champion_batch):
                    break
                continue
            if self._reject_unplayed_pending_historical_duplicate(
                state,
                pending=pending,
                difficulty=difficulty,
                champion=champion,
            ):
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
        # Always mutate the same verified Champion that will be used for scoring.
        # Experiment history may suggest a direction, but a non-winning candidate
        # cannot become the strategy text inherited by that direction.
        search_parent = champion
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
        record_paths = self._select_analysis_record_paths(
            state,
            difficulty=difficulty,
            strategy=search_parent,
            record_paths=record_paths,
        )
        self._save_state(state)
        if not resume_value:
            recovered = self._find_resumable_analysis_checkpoint(
                state=state,
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
                analysis_record_paths=record_paths,
            )
            self._remember_analysis_checkpoint(
                state,
                difficulty=difficulty,
                strategy=search_parent,
                checkpoint_dir=candidate_result.checkpoint_dir,
            )
            if candidate_result.ok and candidate_result.output_dir is not None:
                duplicate = self._historical_strategy_duplicate(
                    state,
                    candidate_dir=candidate_result.output_dir,
                    current_difficulty=difficulty,
                )
                if duplicate is not None:
                    closest = duplicate["matches"][0]
                    message = self._historical_duplicate_feedback(duplicate)
                    removed = self._discard_policy_rejected_candidate(
                        candidate_result.output_dir
                    )
                    failure = {
                        "kind": "candidate_generation_failure",
                        "failure_reason": "historical_strategy_duplicate",
                        "generation": int(state["generation"]),
                        "attempt": attempt,
                        "max_attempts": total_attempts,
                        "difficulty": difficulty,
                        "parent": search_parent,
                        "comparison_champion": champion,
                        "message": message,
                        "candidate_hash": duplicate["candidate_hash"],
                        "duplicate_matches": duplicate["matches"],
                        "candidate_removed": removed,
                        "checkpoint_dir": str(candidate_result.checkpoint_dir or ""),
                        "created_at": datetime.now().isoformat(),
                    }
                    state.setdefault("candidate_generation_failures", []).append(
                        failure
                    )
                    checkpoint_value = str(candidate_result.checkpoint_dir or resume_value)
                    retry_feedback.append(message)
                    print(
                        "EvolAgent rejected an unplayed historical-strategy duplicate; "
                        f"regenerating ({attempt}/{total_attempts}): "
                        f"{closest['strategy']} at {closest['difficulty']}",
                        flush=True,
                    )
                    if attempt < total_attempts:
                        # A candidate-stage checkpoint cannot emit a different
                        # strategy.md. Abandon it and regenerate with feedback.
                        if checkpoint_value:
                            resolved = str(Path(checkpoint_value).resolve())
                            abandoned = state.setdefault(
                                "abandoned_analysis_checkpoints", []
                            )
                            if resolved not in abandoned:
                                abandoned.append(resolved)
                        resume_value = ""
                        state["candidate_resume_dir"] = None
                        self._save_state(state)
                        continue
                    state["candidate_resume_dir"] = checkpoint_value
                    self._save_state(state)
                    candidate_result = EvolRunResult(
                        ok=False,
                        message=message,
                        checkpoint_dir=candidate_result.checkpoint_dir,
                        strategy_name=search_parent,
                        race=self.config.race,
                    )
                    break
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
            if "mechanism history" in message.casefold():
                self._save_state(state)
                print(
                    "EvolAgent rejected a semantically repeated failed mechanism; "
                    "restarting cross-match package selection without playing matches.",
                    flush=True,
                )
                break
            if attempt >= total_attempts:
                break

            retry_feedback.append(
                f"Candidate-generation attempt {attempt}/{total_attempts} failed: {message}"
            )
            # A broken or already-consumed resume checkpoint cannot repair
            # itself; restart from a compatible analysis seed with feedback.
            if (
                message.startswith("failed to load checkpoint:")
                or message.startswith("strategy mismatch with checkpoint:")
                or message.startswith("checkpoint already produced a candidate")
            ):
                if checkpoint_value:
                    resolved = str(Path(checkpoint_value).resolve())
                    abandoned = state.setdefault(
                        "abandoned_analysis_checkpoints", []
                    )
                    if resolved not in abandoned:
                        abandoned.append(resolved)
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
            return self._restart_generation_search(
                state,
                difficulty=difficulty,
                champion=champion,
                reason=str(
                    candidate_result.message
                    or "candidate generation retries were exhausted"
                ),
                source_action="candidate_generation_failed",
                checkpoint_dir=candidate_result.checkpoint_dir,
            )
        rationale = (
            candidate_result.improvement.analysis
            if candidate_result.improvement is not None
            else {}
        )
        experiment_spec = self._experiment_spec_from_rationale(rationale)
        # Historical experiments are supplied as causal evidence. A separate
        # semantic judge may reject an unchanged failed direction before matches;
        # mechanism labels and exact wording never decide equivalence by themselves.
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
            "intervention_package": (
                dict(rationale.get("intervention_package"))
                if isinstance(rationale.get("intervention_package"), dict)
                else {}
            ),
            "inheritance": (
                dict(rationale.get("inheritance"))
                if isinstance(rationale.get("inheritance"), dict)
                else {}
            ),
            "selected_history_assessment": (
                dict(rationale.get("selected_history_assessment"))
                if isinstance(rationale.get("selected_history_assessment"), dict)
                else {}
            ),
            "document_changes": _dict_list(rationale.get("document_changes")),
            "semantic_validation": (
                dict(rationale.get("semantic_validation"))
                if isinstance(rationale.get("semantic_validation"), dict)
                else {"status": "passed", "errors": []}
            ),
        }
        state["consecutive_search_restarts"] = 0
        state.pop("candidate_search_blocked_reason", None)
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
            if self.config.analysis_experience_mode == "single_failure":
                self._record_agent_decision(
                    state,
                    result,
                    difficulty=difficulty,
                )
                return self._restart_generation_search(
                    state,
                    difficulty=difficulty,
                    champion=champion,
                    reason=(
                        "single_failure ablation fixes the visible trajectory "
                        "budget at one match; additional match analysis is disabled"
                    ),
                    source_action="request_more_matches_disabled_by_ablation",
                    checkpoint_dir=result.checkpoint_dir,
                )
            analysis_games = self._analysis_games(
                state, difficulty=difficulty, strategy=champion
            )
            if analysis_games >= self.config.max_analysis_games_per_generation:
                self._record_agent_decision(
                    state,
                    result,
                    difficulty=difficulty,
                )
                return self._restart_generation_search(
                    state,
                    difficulty=difficulty,
                    champion=champion,
                    reason=(
                        result.action_reason
                        or "the analysis evidence limit was reached without a candidate"
                    ),
                    source_action="request_more_matches_exhausted",
                    checkpoint_dir=result.checkpoint_dir,
                )
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
            self._record_agent_decision(
                state,
                result,
                difficulty=difficulty,
            )
            return self._restart_generation_search(
                state,
                difficulty=difficulty,
                champion=champion,
                reason=(
                    result.action_reason
                    or "the selected explanation belongs to runtime execution"
                ),
                source_action="inspect_runtime",
                checkpoint_dir=result.checkpoint_dir,
            )
        if action == "stop":
            self._record_agent_decision(
                state,
                result,
                difficulty=difficulty,
            )
            return self._restart_generation_search(
                state,
                difficulty=difficulty,
                champion=champion,
                reason=(
                    result.action_reason
                    or "the current analysis found no actionable strategy change"
                ),
                source_action="stop",
                checkpoint_dir=result.checkpoint_dir,
            )
        self._record_agent_decision(
            state,
            result,
            difficulty=difficulty,
        )
        return self._restart_generation_search(
            state,
            difficulty=difficulty,
            champion=champion,
            reason=result.action_reason or result.message or f"unknown action: {action}",
            source_action=action or "unknown",
            checkpoint_dir=result.checkpoint_dir,
        )

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
        # Mastery advances the curriculum only after a candidate has beaten the
        # official Champion.  Reaching an absolute threshold must never replace a
        # stronger Champion with a lower-scoring candidate.
        outcome = score_outcome
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
            "salvageable_changes": [],
            "failed_dependencies": [],
            "evidence_limits": [
                "post-experiment mechanism audit was unavailable"
            ],
            "lesson": "",
            "gate_execution_audit": {},
        }
        auditor = self._experiment_auditor
        if auditor is None and self._batch_executor is None:
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
        # Equal-score candidates remain useful evidence, but they never become the
        # official Champion or the textual parent of a later candidate. After two
        # such trials, history asks analysis to test a different causal direction.
        forced_mechanism_change_after_inconclusive = bool(
            valid_inconclusive and streak_before >= 1
        )
        forced_promotion_after_inconclusive = False  # checkpoint compatibility
        mastery_overrode_audit_uncertainty = bool(
            self._mechanism_audit_enabled()
            and accepted
            and candidate_mastered
            and implementation_verdict in {"underpowered", "unknown"}
        )
        promotion_blocked_by_audit = bool(
            self._mechanism_audit_enabled()
            and accepted
            and implementation_verdict != "implemented"
            and not mastery_overrode_audit_uncertainty
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
            search_parent_after = champion
            streak_after = streak_before + 1
        else:
            # A lower-scoring candidate remains useful as experiment evidence, but
            # it must not become the textual parent of the next mutation. Rebuild
            # any corrected retry from the accepted Champion so failed edits do
            # not accumulate across generations.
            search_parent_after = champion
            streak_after = 0 if base_outcome == "rejected" else streak_before
        experiment_id = self._experiment_id(
            style=str(state.get("style") or self.config.strategy),
            generation=int(state["generation"]),
            difficulty=difficulty,
            candidate=candidate,
        )
        decision_inheritance = (
            dict(pending.get("inheritance"))
            if isinstance(pending.get("inheritance"), dict)
            else {}
        )
        if accepted and implementation_verdict == "implemented":
            verified_changes = [
                dict(item)
                for item in (decision_inheritance.get("verified_changes") or [])
                if isinstance(item, dict)
            ]
            if not any(
                str(item.get("experiment_id") or "") == experiment_id
                for item in verified_changes
            ):
                verified_changes.append(
                    {
                        "experiment_id": experiment_id,
                        "difficulty": difficulty,
                        "mechanism_family": mechanism_family,
                        "change": str(
                            pending.get("primary_change")
                            or experiment_spec.get("plan_direction")
                            or experiment_spec.get("hypothesis")
                            or ""
                        ),
                        "evidence": str(mechanism_audit.get("lesson") or ""),
                        "score_delta": score_delta,
                    }
                )
            decision_inheritance = {
                "verified_changes": verified_changes[-12:],
                "preservation_rule": (
                    "Preserve each trajectory-realized Champion improvement unless "
                    "current cross-match evidence directly supports revising it."
                ),
            }
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
            "salvageable_changes": list(
                mechanism_audit.get("salvageable_changes") or []
            ),
            "failed_dependencies": list(
                mechanism_audit.get("failed_dependencies") or []
            ),
            "audit_evidence_limits": list(
                mechanism_audit.get("evidence_limits") or []
            ),
            "gate_execution_audit": (
                dict(mechanism_audit.get("gate_execution_audit"))
                if isinstance(mechanism_audit.get("gate_execution_audit"), dict)
                else {}
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
                "candidate_mastery_threshold_and_score_gain"
                if mastery_overrode_audit_uncertainty
                else (
                    "candidate_score_strictly_greater_and_mechanism_implemented"
                    if self._mechanism_audit_enabled()
                    else "candidate_score_strictly_greater"
                )
            ),
            "base_decision": base_outcome,
            "forced_promotion_after_inconclusive": (
                forced_promotion_after_inconclusive
            ),
            "forced_mechanism_change_after_inconclusive": (
                forced_mechanism_change_after_inconclusive
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
            "mastery_overrode_audit_uncertainty": mastery_overrode_audit_uncertainty,
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
            "selected_history_assessment": (
                dict(experiment_spec.get("selected_history_assessment"))
                if isinstance(
                    experiment_spec.get("selected_history_assessment"), dict
                )
                else {}
            ),
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
                decision_inheritance
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
        else:
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
        if forced_mechanism_change_after_inconclusive:
            lesson = (
                "Two consecutive valid candidates were statistically inconclusive. "
                "The official Champion remains the only mutation parent, and the "
                "next generation must test a different "
                "causal mechanism instead of relabeling the same direction."
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
                "The score improved, but the post-experiment audit did not confirm "
                "that the candidate's material mechanism was implemented in the "
                "matches. It was not promoted; a score change without a realized "
                "intervention is not strategy-evolution evidence."
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
            "experiment_id": experiment_id,
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
            "salvageable_changes": list(
                decision.get("salvageable_changes") or []
            ),
            "failed_dependencies": list(
                decision.get("failed_dependencies") or []
            ),
            "audit_evidence_limits": list(
                decision.get("audit_evidence_limits") or []
            ),
            "gate_execution_audit": (
                dict(decision.get("gate_execution_audit"))
                if isinstance(decision.get("gate_execution_audit"), dict)
                else {}
            ),
            "first_commitment_timing": (
                dict(experiment_spec.get("first_commitment_timing"))
                if isinstance(experiment_spec.get("first_commitment_timing"), dict)
                else {}
            ),
            "mechanism_equivalence_audit": (
                dict(experiment_spec.get("mechanism_equivalence_audit"))
                if isinstance(
                    experiment_spec.get("mechanism_equivalence_audit"), dict
                )
                else {}
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
            "selected_history_assessment": (
                dict(decision.get("selected_history_assessment"))
                if isinstance(decision.get("selected_history_assessment"), dict)
                else {}
            ),
            "patches": _dict_list(decision.get("patches")),
            "decision": outcome,
            "base_decision": base_outcome,
            "forced_promotion_after_inconclusive": (
                forced_promotion_after_inconclusive
            ),
            "forced_mechanism_change_after_inconclusive": (
                forced_mechanism_change_after_inconclusive
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
                    "selected_history_assessment": dict(
                        experience.get("selected_history_assessment") or {}
                    ),
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
    "completed_record_indices",
    "read_batch_result",
]
