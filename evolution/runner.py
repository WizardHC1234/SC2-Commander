from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from evol_agent import EvolAgent
from evol_agent.analysis.record_reader import find_record_jsons, is_completed_match_record
from evol_agent.core.types import EvolRunRequest, EvolRunResult
from .feedback import (
    combine_batch_evidence,
    compare_batch_evidence,
    summarize_batch_evidence,
)
from .outcomes import aggregate_outcomes, posterior_probability_better


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
    "score",
    "mastered_levels",
    "evolution_score",
    "accepted",
    "games_used",
    "batch",
)
MAX_CANDIDATE_GENERATION_ATTEMPTS = 3


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
    candidate_initial_matches: int = 6
    candidate_max_matches: int = 10
    candidate_step_matches: int = 2
    candidate_accept_probability: float = 0.80
    candidate_reject_probability: float = 0.20
    concurrency: int = 5
    pass_score: float = 0.8
    max_generations: int = 10
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
        if not 0 < self.candidate_initial_matches <= self.candidate_max_matches:
            raise ValueError(
                "candidate_initial_matches must be positive and no greater than "
                "candidate_max_matches"
            )
        if self.candidate_step_matches <= 0:
            raise ValueError("candidate_step_matches must be positive")
        if not 0.5 < self.candidate_accept_probability <= 1.0:
            raise ValueError("candidate_accept_probability must be in (0.5, 1.0]")
        if not 0.0 <= self.candidate_reject_probability < 0.5:
            raise ValueError("candidate_reject_probability must be in [0.0, 0.5)")
        if not 0.0 <= self.pass_score <= 1.0:
            raise ValueError("pass_score must be between 0 and 1")
        if self.max_generations < 0:
            raise ValueError("max_generations cannot be negative")


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
    def outcome_points(self) -> float:
        return self.wins + 0.5 * self.draws

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["score"] = self.score
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
    """Return whether one more equal-size batch is needed for both strategies."""
    if champion.games != candidate.games:
        return False
    return abs(candidate.outcome_points - champion.outcome_points) <= 1.0


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
        batch_executor: Callable[[str, str], BatchResult] | None = None,
        candidate_generator: Callable[[str, BatchResult, list[Any]], EvolRunResult] | None = None,
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

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema": "sc2_evolution.v2",
            "status": "running",
            "config": {**asdict(self.config), "difficulties": list(self.config.difficulties)},
            "style": self.config.strategy,
            "champion": self.config.strategy,
            "difficulty_index": 0,
            "generation": 0,
            "games_used": 0,
            "champion_batch": None,
            "pending_candidate": None,
            "failed_experiences": [],
            "candidate_generation_failures": [],
            "evidence_pool": {},
            "updated_at": datetime.now().isoformat(),
        }

    def load_or_create_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            saved = state.get("config") or {}
            current = {**asdict(self.config), "difficulties": list(self.config.difficulties)}
            # Outcome-pool settings were added after v2 runs already existed.
            # Missing new keys inherit the explicit resume configuration.
            for key in (
                "candidate_initial_matches",
                "candidate_max_matches",
                "candidate_step_matches",
                "candidate_accept_probability",
                "candidate_reject_probability",
                "baseline_batch_dir",
            ):
                saved.setdefault(key, current[key])
            if saved != current:
                raise ValueError("resume configuration does not match state.json")
            state["config"] = saved
            changed = self._backfill_experiment_evidence(state)
            changed = self._migrate_evidence_pool(state) or changed
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
                    name = self._batch_name(generation, role)
                    add_path(self.project_root / "game_records" / name, strategy, difficulty)

        previous_games = int(state.get("games_used") or 0)
        self._sync_games_used(state)
        return changed or int(state.get("games_used") or 0) != previous_games

    def _backfill_experiment_evidence(self, state: dict[str, Any]) -> bool:
        """Enrich pre-v3 rejection memory from immutable decision/batch artifacts."""
        changed = False
        experiences = state.get("failed_experiences")
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

    def _batch_name(self, generation: int, role: str) -> str:
        return _safe_name(f"ev_{self.run_id}_g{generation:03d}_{role}", 40)

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
            return self._batch_executor(strategy, difficulty)
        expected_games = int(target_games or self.config.matches_per_batch)
        if expected_games <= 0:
            raise ValueError("target_games must be positive")
        batch_name = self._batch_name(generation, role)
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
        subprocess.run(command, cwd=self.project_root, check=True)
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
        failed_experiences: list[Any],
        *,
        evidence_batches: list[BatchResult] | None = None,
    ) -> EvolRunResult:
        if self._candidate_generator is not None:
            return self._candidate_generator(champion, champion_batch, failed_experiences)
        record_paths: list[Path] = []
        for batch in evidence_batches or [champion_batch]:
            record_paths.extend(find_record_jsons(batch.path))
        return EvolAgent(model=self.config.evolution_model).run(
            EvolRunRequest(
                record_paths=list(dict.fromkeys(record_paths)),
                strategy_name=champion,
                race=self.config.race,
                model=self.config.evolution_model,
                knowledge_mode=self.config.knowledge_mode,
                prior_experiences=failed_experiences,
            )
        )

    def _append_history(
        self,
        *,
        state: dict[str, Any],
        batch: BatchResult,
        parent: str,
        accepted: bool,
    ) -> None:
        if self.history_path.is_file():
            with self.history_path.open(encoding="utf-8", newline="") as handle:
                if any(
                    row.get("batch") == batch.name and row.get("strategy") == batch.strategy
                    for row in csv.DictReader(handle)
                ):
                    return
        mastered = int(state["difficulty_index"])
        evolution_score = (mastered + batch.score) / len(self.config.difficulties)
        row = {
            "strategy_style": state["style"],
            "generation": state["generation"],
            "strategy": batch.strategy,
            "parent": parent,
            "difficulty": batch.difficulty,
            "wins": batch.wins,
            "draws": batch.draws,
            "losses": batch.losses,
            "score": f"{batch.score:.4f}",
            "mastered_levels": mastered,
            "evolution_score": f"{evolution_score:.4f}",
            "accepted": str(accepted).lower(),
            "games_used": state["games_used"],
            "batch": batch.name,
        }
        new_file = not self.history_path.exists()
        with self.history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def _advance_difficulty(self, state: dict[str, Any]) -> None:
        state["difficulty_index"] += 1
        state["champion_batch"] = None
        state["pending_candidate"] = None
        if state["difficulty_index"] >= len(self.config.difficulties):
            state["status"] = "completed"

    def run(self) -> dict[str, Any]:
        state = self.load_or_create_state()
        while state["status"] == "running":
            if state["difficulty_index"] >= len(self.config.difficulties):
                state["status"] = "completed"
                self._save_state(state)
                break
            if state["generation"] >= self.config.max_generations:
                state["status"] = "budget_exhausted"
                self._save_state(state)
                break

            difficulty = self.config.difficulties[state["difficulty_index"]]
            champion = str(state["champion"])
            if state.get("champion_batch") is None:
                baseline = self.run_batch(
                    champion,
                    difficulty,
                    generation=int(state["generation"]),
                    role="champ",
                )
                self._register_evidence(state, baseline)
                self._sync_games_used(state)
                state["champion_batch"] = baseline.to_dict()
                self._append_history(state=state, batch=baseline, parent="", accepted=True)
                self._save_state(state)
                if baseline.score >= self.config.pass_score:
                    self._advance_difficulty(state)
                    self._save_state(state)
                    continue

            champion_batch = BatchResult.from_dict(state["champion_batch"])
            champion_evidence_batches = self._evidence_batches(
                state,
                difficulty=difficulty,
                strategy=champion,
            )
            if not champion_evidence_batches:
                self._register_evidence(state, champion_batch)
                champion_evidence_batches = [champion_batch]
            comparison_champion = self._aggregate_evidence(
                state,
                difficulty=difficulty,
                strategy=champion,
            )
            if comparison_champion.score >= self.config.pass_score:
                self._advance_difficulty(state)
                self._save_state(state)
                continue
            pending = state.get("pending_candidate")
            if not isinstance(pending, dict):
                candidate_result: EvolRunResult | None = None
                for candidate_attempt in range(1, MAX_CANDIDATE_GENERATION_ATTEMPTS + 1):
                    candidate_result = self.generate_candidate(
                        champion,
                        champion_batch,
                        [
                            *list(state.get("failed_experiences") or []),
                            *list(state.get("candidate_generation_failures") or []),
                        ],
                        evidence_batches=champion_evidence_batches,
                    )
                    if candidate_result.ok and (
                        candidate_result.output_dir is not None
                        or candidate_result.decision_action
                        != "propose_strategy_patch"
                    ):
                        break
                    failure = {
                        "kind": "candidate_generation_failure",
                        "generation": int(state["generation"]),
                        "attempt": candidate_attempt,
                        "difficulty": difficulty,
                        "parent": champion,
                        "message": str(candidate_result.message),
                        "created_at": datetime.now().isoformat(),
                    }
                    state.setdefault("candidate_generation_failures", []).append(failure)
                    self._save_state(state)
                    print(
                        "EvolAgent candidate generation failed; "
                        f"retrying ({candidate_attempt}/{MAX_CANDIDATE_GENERATION_ATTEMPTS}): "
                        f"{candidate_result.message}",
                        flush=True,
                    )
                if (
                    candidate_result is None
                    or not candidate_result.ok
                    or candidate_result.output_dir is None
                ):
                    if (
                        candidate_result is not None
                        and candidate_result.ok
                        and candidate_result.decision_action
                        != "propose_strategy_patch"
                    ):
                        if candidate_result.decision_action == "request_more_matches":
                            comparison_champion = self._aggregate_evidence(
                                state,
                                difficulty=difficulty,
                                strategy=champion,
                            )
                            max_champion_evidence = self.config.matches_per_batch * 2
                            more_batch_name = self._batch_name(
                                int(state["generation"]), "champ_more"
                            )
                            more_batch_dir = self.project_root / "game_records" / more_batch_name
                            completed_more = completed_record_count(
                                more_batch_dir, strategy=champion
                            )
                            if comparison_champion.games < max_champion_evidence:
                                next_more_target = min(
                                    self.config.matches_per_batch,
                                    completed_more + self.config.candidate_step_matches,
                                )
                                more_batch = self.run_batch(
                                    champion,
                                    difficulty,
                                    generation=int(state["generation"]),
                                    role="champ_more",
                                    target_games=next_more_target,
                                )
                                evidence_changed = self._register_evidence(state, more_batch)
                                self._sync_games_used(state)
                                state["last_agent_decision"] = {
                                    "action": candidate_result.decision_action,
                                    "reason": candidate_result.action_reason,
                                    "message": candidate_result.message,
                                    "generation": int(state["generation"]),
                                    "difficulty": difficulty,
                                    "additional_batch": more_batch.to_dict(),
                                }
                                self._save_state(state)
                                refreshed_champion = self._aggregate_evidence(
                                    state,
                                    difficulty=difficulty,
                                    strategy=champion,
                                )
                                if (
                                    evidence_changed
                                    and refreshed_champion.games
                                    > comparison_champion.games
                                ):
                                    continue
                        state["status"] = {
                            "inspect_runtime": "runtime_attention_required",
                            "request_more_matches": "more_evidence_requested",
                            "stop": "stopped_no_actionable_improvement",
                        }.get(candidate_result.decision_action, "agent_paused")
                        state["last_agent_decision"] = {
                            "action": candidate_result.decision_action,
                            "reason": candidate_result.action_reason,
                            "message": candidate_result.message,
                            "generation": int(state["generation"]),
                            "difficulty": difficulty,
                        }
                        self._save_state(state)
                        return state
                    state["status"] = "evol_agent_failed"
                    self._save_state(state)
                    print(
                        "EvolAgent exhausted candidate-generation retries; state is "
                        f"saved and resumable at {self.run_dir}",
                        flush=True,
                    )
                    break
                rationale = (
                    candidate_result.improvement.analysis
                    if candidate_result.improvement is not None
                    else {}
                )
                pending = {
                    "strategy": candidate_result.output_dir.name,
                    "strategy_dir": str(candidate_result.output_dir),
                    "candidate_hash": candidate_result.candidate_hash,
                    "primary_change": str(rationale.get("primary_change") or ""),
                    "selected_plan_ids": _string_list(
                        rationale.get("selected_plan_ids")
                    ),
                    "overall_assessment": str(
                        rationale.get("overall_assessment") or ""
                    ),
                    "selected_changes": _dict_list(rationale.get("selected_changes")),
                    "expected_effect": str(rationale.get("expected_effect") or ""),
                    "main_risk": str(rationale.get("main_risk") or ""),
                    "hypothesis": str(rationale.get("hypothesis") or ""),
                    "primary_lever": str(rationale.get("primary_lever") or ""),
                    "predictions": _string_list(rationale.get("predictions")),
                    "disproof_conditions": _string_list(
                        rationale.get("disproof_conditions")
                    ),
                    "capability_mapping": (
                        dict(rationale.get("capability_mapping"))
                        if isinstance(rationale.get("capability_mapping"), dict)
                        else {}
                    ),
                }
                state["pending_candidate"] = pending
                self._save_state(state)
            candidate = str(pending["strategy"])
            candidate_batches = self._evidence_batches(
                state,
                difficulty=difficulty,
                strategy=candidate,
            )
            candidate_games = sum(batch.games for batch in candidate_batches)
            if candidate_games < self.config.candidate_initial_matches:
                candidate_batch = self.run_batch(
                    candidate,
                    difficulty,
                    generation=int(state["generation"]),
                    role="cand",
                    target_games=self.config.candidate_initial_matches,
                )
                self._register_evidence(state, candidate_batch)
                self._sync_games_used(state)
                pending["candidate_batch"] = candidate_batch.to_dict()
                self._save_state(state)

            evaluation_rounds: list[dict[str, Any]] = []
            while True:
                comparison_champion = self._aggregate_evidence(
                    state,
                    difficulty=difficulty,
                    strategy=champion,
                )
                comparison_candidate = self._aggregate_evidence(
                    state,
                    difficulty=difficulty,
                    strategy=candidate,
                )
                probability = posterior_probability_better(
                    comparison_candidate.to_dict(),
                    comparison_champion.to_dict(),
                )
                evaluation_rounds.append(
                    {
                        "candidate_games": comparison_candidate.games,
                        "candidate_score": comparison_candidate.score,
                        "champion_games": comparison_champion.games,
                        "champion_score": comparison_champion.score,
                        "posterior_probability_better": probability,
                    }
                )
                enough_to_accept = (
                    probability >= self.config.candidate_accept_probability
                    and comparison_candidate.score > comparison_champion.score
                )
                enough_to_reject = (
                    probability <= self.config.candidate_reject_probability
                )
                at_limit = (
                    comparison_candidate.games >= self.config.candidate_max_matches
                )
                if enough_to_accept or enough_to_reject or at_limit:
                    break
                next_target = min(
                    self.config.candidate_max_matches,
                    comparison_candidate.games + self.config.candidate_step_matches,
                )
                previous_games = comparison_candidate.games
                candidate_batch = self.run_batch(
                    candidate,
                    difficulty,
                    generation=int(state["generation"]),
                    role="cand",
                    target_games=next_target,
                )
                self._register_evidence(state, candidate_batch)
                self._sync_games_used(state)
                pending["candidate_batch"] = candidate_batch.to_dict()
                self._save_state(state)
                if candidate_batch.games <= previous_games:
                    # Test executors may return a fixed synthetic batch. Avoid
                    # an infinite loop while preserving a deterministic reject.
                    break

            accepted = (
                probability >= self.config.candidate_accept_probability
                and comparison_candidate.score > comparison_champion.score
            )
            candidate_batch = self._evidence_batches(
                state,
                difficulty=difficulty,
                strategy=candidate,
            )[-1]
            self._append_history(
                state=state,
                batch=comparison_candidate,
                parent=champion,
                accepted=accepted,
            )
            decision = {
                "generation": state["generation"],
                "difficulty": difficulty,
                "parent": champion,
                "candidate": candidate,
                "parent_score": comparison_champion.score,
                "candidate_score": comparison_candidate.score,
                "delta": comparison_candidate.score - comparison_champion.score,
                "accepted": accepted,
                "posterior_probability_better": probability,
                "accept_probability": self.config.candidate_accept_probability,
                "reject_probability": self.config.candidate_reject_probability,
                "champion_evidence_games": comparison_champion.games,
                "candidate_evidence_games": comparison_candidate.games,
                "evaluation_rounds": evaluation_rounds,
                "candidate_hash": str(pending.get("candidate_hash") or ""),
                "parent_batch": str(champion_batch.path),
                "candidate_batch": str(candidate_batch.path),
                "comparison_games_per_strategy": None,
                "confirmation": None,
                "candidate_strategy_dir": str(pending.get("strategy_dir") or ""),
                "primary_change": str(pending.get("primary_change") or ""),
                "selected_plan_ids": _string_list(pending.get("selected_plan_ids")),
                "overall_assessment": str(
                    pending.get("overall_assessment") or ""
                ),
                "selected_changes": _dict_list(pending.get("selected_changes")),
                "expected_effect": str(pending.get("expected_effect") or ""),
                "main_risk": str(pending.get("main_risk") or ""),
                "hypothesis": str(pending.get("hypothesis") or ""),
                "primary_lever": str(pending.get("primary_lever") or ""),
                "predictions": _string_list(pending.get("predictions")),
                "disproof_conditions": _string_list(
                    pending.get("disproof_conditions")
                ),
                "capability_mapping": (
                    dict(pending.get("capability_mapping"))
                    if isinstance(pending.get("capability_mapping"), dict)
                    else {}
                ),
            }
            generation_dir = self.run_dir / f"generation_{int(state['generation']):03d}"
            _write_json(generation_dir / "decision.json", decision)
            if accepted:
                state["champion"] = candidate
                state["champion_batch"] = candidate_batch.to_dict()
            else:
                parent_evidence = {
                    **aggregate_outcomes(
                        item.to_dict()
                        for item in self._evidence_batches(
                            state, difficulty=difficulty, strategy=champion
                        )
                    ),
                    "strategy": champion,
                    "difficulty": difficulty,
                }
                candidate_evidence = {
                    **aggregate_outcomes(
                        item.to_dict()
                        for item in self._evidence_batches(
                            state, difficulty=difficulty, strategy=candidate
                        )
                    ),
                    "strategy": candidate,
                    "difficulty": difficulty,
                }
                experience = {
                    "generation": int(state["generation"]),
                    "difficulty": difficulty,
                    "parent": champion,
                    "candidate": candidate,
                    "candidate_hash": str(decision.get("candidate_hash") or ""),
                    "primary_change": str(
                        decision.get("primary_change") or "the candidate change"
                    ),
                    "selected_plan_ids": _string_list(
                        decision.get("selected_plan_ids")
                    ),
                    "overall_assessment": str(
                        decision.get("overall_assessment") or ""
                    ),
                    "selected_changes": _dict_list(decision.get("selected_changes")),
                    "expected_effect": str(decision.get("expected_effect") or ""),
                    "main_risk": str(decision.get("main_risk") or ""),
                    "hypothesis": str(decision.get("hypothesis") or ""),
                    "primary_lever": str(decision.get("primary_lever") or ""),
                    "predictions": _string_list(decision.get("predictions")),
                    "disproof_conditions": _string_list(
                        decision.get("disproof_conditions")
                    ),
                    "capability_mapping": (
                        dict(decision.get("capability_mapping"))
                        if isinstance(decision.get("capability_mapping"), dict)
                        else {}
                    ),
                    "parent_score": comparison_champion.score,
                    "candidate_score": comparison_candidate.score,
                    "delta": comparison_candidate.score - comparison_champion.score,
                    "champion_games": comparison_champion.games,
                    "candidate_games": comparison_candidate.games,
                    "posterior_probability_better": probability,
                    "experiment_evidence": {
                        "parent_batch": parent_evidence,
                        "candidate_batch": candidate_evidence,
                        "candidate_minus_parent": {
                            "score_delta": (
                                comparison_candidate.score - comparison_champion.score
                            )
                        },
                        "comparison_used_confirmation": False,
                        "confirmation_batches": None,
                    },
                    "lesson": (
                        "This change combination was rejected. Do not repeat a materially "
                        "equivalent combination unless new match evidence supports it and "
                        "the new plan explains the substantive difference."
                    ),
                }
                state.setdefault("failed_experiences", []).append(experience)
            state["generation"] += 1
            state["pending_candidate"] = None
            if accepted and comparison_candidate.score >= self.config.pass_score:
                self._advance_difficulty(state)
            self._save_state(state)
        return state


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
