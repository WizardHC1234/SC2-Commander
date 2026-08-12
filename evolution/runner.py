from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from evol_agent import EvolAgent
from evol_agent.analysis.record_reader import find_record_jsons, is_completed_match_record
from evol_agent.core.types import EvolRunRequest, EvolRunResult


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
    concurrency: int = 5
    pass_score: float = 0.8
    max_generations: int = 10
    knowledge_mode: str = "enabled"
    bot_name: str = "commander"
    bot_instruct: str = ""
    real_time: bool = False

    def validate(self) -> None:
        if not self.strategy.strip():
            raise ValueError("strategy cannot be empty")
        if not self.commander_model.strip():
            raise ValueError("commander_model cannot be empty")
        if not self.difficulties:
            raise ValueError("at least one difficulty is required")
        if self.matches_per_batch <= 0 or self.concurrency <= 0:
            raise ValueError("matches_per_batch and concurrency must be positive")
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
        candidate_generator: Callable[[str, BatchResult, list[str]], EvolRunResult] | None = None,
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
            "schema": "sc2_evolution.v1",
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
            "updated_at": datetime.now().isoformat(),
        }

    def load_or_create_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            saved = state.get("config") or {}
            current = {**asdict(self.config), "difficulties": list(self.config.difficulties)}
            if saved != current:
                raise ValueError("resume configuration does not match state.json")
            return state
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state = self._new_state()
        self._save_state(state)
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = datetime.now().isoformat()
        _write_json(self.state_path, state)

    def _batch_name(self, generation: int, role: str) -> str:
        return _safe_name(f"ev_{self.run_id}_g{generation:03d}_{role}", 40)

    def run_batch(self, strategy: str, difficulty: str, *, generation: int, role: str) -> BatchResult:
        if self._batch_executor is not None:
            return self._batch_executor(strategy, difficulty)
        batch_name = self._batch_name(generation, role)
        batch_dir = self.project_root / "game_records" / batch_name
        completed = completed_record_count(batch_dir, strategy=strategy)
        if completed == self.config.matches_per_batch:
            return read_batch_result(
                batch_dir,
                name=batch_name,
                strategy=strategy,
                difficulty=difficulty,
                expected_games=self.config.matches_per_batch,
            )
        if completed > self.config.matches_per_batch:
            raise RuntimeError(
                f"batch {batch_name} has {completed} completed records, more than the "
                f"configured {self.config.matches_per_batch}"
            )
        remaining = self.config.matches_per_batch - completed
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.project_root / "scripts" / "start_batch.ps1"),
            "-MY_BOT_NAME",
            self.config.bot_name,
            "-MAP_NAME",
            self.config.map_name,
            "-REAL_TIME",
            "1" if self.config.real_time else "0",
            "-ENEMY_RACE",
            self.config.enemy_race,
            "-ENEMY_DIFFICULTY",
            difficulty,
            "-ENEMY_BUILD",
            self.config.enemy_build,
            "-BOT_RACE",
            self.config.race,
            "-FORCE_STRATEGY",
            strategy,
            "-COMMANDER_MODEL",
            self.config.commander_model,
            "-TOTAL_MATCHES",
            str(remaining),
            "-CONCURRENCY",
            str(self.config.concurrency),
            "-START_INDEX",
            str(completed),
            "-BATCH_NAME",
            batch_name,
        ]
        if self.config.bot_instruct:
            command.extend(["-BOT_INSTRUCT", self.config.bot_instruct])
        subprocess.run(command, cwd=self.project_root, check=True)
        return read_batch_result(
            batch_dir,
            name=batch_name,
            strategy=strategy,
            difficulty=difficulty,
            expected_games=self.config.matches_per_batch,
        )

    def generate_candidate(
        self,
        champion: str,
        champion_batch: BatchResult,
        failed_experiences: list[str],
    ) -> EvolRunResult:
        if self._candidate_generator is not None:
            return self._candidate_generator(champion, champion_batch, failed_experiences)
        return EvolAgent(model=self.config.evolution_model).run(
            EvolRunRequest(
                batch_dir=champion_batch.path,
                strategy_name=champion,
                race=self.config.race,
                model=self.config.evolution_model,
                knowledge_mode=self.config.knowledge_mode,
                prior_experiences=failed_experiences[-3:],
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
                state["games_used"] += baseline.games
                state["champion_batch"] = baseline.to_dict()
                self._append_history(state=state, batch=baseline, parent="", accepted=True)
                self._save_state(state)
                if baseline.score >= self.config.pass_score:
                    self._advance_difficulty(state)
                    self._save_state(state)
                    continue

            champion_batch = BatchResult.from_dict(state["champion_batch"])
            pending = state.get("pending_candidate")
            if not isinstance(pending, dict):
                candidate_result = self.generate_candidate(
                    champion,
                    champion_batch,
                    list(state.get("failed_experiences") or []),
                )
                if not candidate_result.ok or candidate_result.output_dir is None:
                    raise RuntimeError(f"EvolAgent failed: {candidate_result.message}")
                pending = {
                    "strategy": candidate_result.output_dir.name,
                    "strategy_dir": str(candidate_result.output_dir),
                    "candidate_hash": candidate_result.candidate_hash,
                    "primary_change": (
                        candidate_result.improvement.analysis.get("primary_change", "")
                        if candidate_result.improvement is not None
                        else ""
                    ),
                }
                state["pending_candidate"] = pending
                self._save_state(state)
            candidate = str(pending["strategy"])
            candidate_batch = self.run_batch(
                candidate,
                difficulty,
                generation=int(state["generation"]),
                role="cand",
            )
            state["games_used"] += candidate_batch.games
            accepted = candidate_batch.score > champion_batch.score
            self._append_history(
                state=state,
                batch=candidate_batch,
                parent=champion,
                accepted=accepted,
            )
            decision = {
                "generation": state["generation"],
                "difficulty": difficulty,
                "parent": champion,
                "candidate": candidate,
                "parent_score": champion_batch.score,
                "candidate_score": candidate_batch.score,
                "delta": candidate_batch.score - champion_batch.score,
                "accepted": accepted,
                "candidate_hash": str(pending.get("candidate_hash") or ""),
                "parent_batch": str(champion_batch.path),
                "candidate_batch": str(candidate_batch.path),
                "candidate_strategy_dir": str(pending.get("strategy_dir") or ""),
                "primary_change": str(pending.get("primary_change") or ""),
            }
            generation_dir = self.run_dir / f"generation_{int(state['generation']):03d}"
            _write_json(generation_dir / "decision.json", decision)
            if accepted:
                state["champion"] = candidate
                state["champion_batch"] = candidate_batch.to_dict()
            else:
                primary_change = str(decision.get("primary_change") or "the candidate change")
                experience = (
                    f"At {difficulty}, {candidate} applied {primary_change!r} and changed the score from "
                    f"{champion_batch.score:.2f} to {candidate_batch.score:.2f}; "
                    "treat this rejected change as evidence, not a permanent prohibition."
                )
                state.setdefault("failed_experiences", []).append(experience)
                state["failed_experiences"] = state["failed_experiences"][-10:]
            state["generation"] += 1
            state["pending_candidate"] = None
            if accepted and candidate_batch.score >= self.config.pass_score:
                self._advance_difficulty(state)
            self._save_state(state)
        return state


__all__ = [
    "BatchResult",
    "DEFAULT_DIFFICULTIES",
    "EvolutionConfig",
    "EvolutionRunner",
    "completed_record_count",
    "read_batch_result",
]
