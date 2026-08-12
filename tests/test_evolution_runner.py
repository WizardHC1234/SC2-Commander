from __future__ import annotations

import csv
from pathlib import Path

from evolution.runner import BatchResult, EvolutionConfig, EvolutionRunner, completed_record_count
from evol_agent.core.types import EvolImprovement, EvolRunResult


def _batch(strategy: str, difficulty: str, wins: int, root: Path) -> BatchResult:
    return BatchResult(
        name=f"{strategy}_{difficulty}_{wins}",
        path=root / f"{strategy}_{difficulty}_{wins}",
        strategy=strategy,
        difficulty=difficulty,
        wins=wins,
        draws=0,
        losses=10 - wins,
    )


def test_evolution_accepts_only_strict_improvement_and_advances(tmp_path: Path) -> None:
    scores = {"tank": 5, "tank_opt1": 8}

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, scores[strategy], tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[str]) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate, candidate_hash="abc")

    config = EvolutionConfig(
        strategy="tank",
        commander_model="model",
        difficulties=("harder",),
        max_generations=2,
    )
    runner = EvolutionRunner(
        config,
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    )
    state = runner.run()

    assert state["status"] == "completed"
    assert state["champion"] == "tank_opt1"
    assert state["games_used"] == 20
    rows = list(csv.DictReader(runner.history_path.open(encoding="utf-8")))
    assert [row["accepted"] for row in rows] == ["true", "true"]
    assert [row["games_used"] for row in rows] == ["10", "20"]


def test_rejected_candidate_is_saved_as_experience(tmp_path: Path) -> None:
    calls = 0

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 4, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[str]) -> EvolRunResult:
        nonlocal calls
        calls += 1
        candidate = tmp_path / "skills" / "terran" / f"tank_opt{calls}"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(
            ok=True,
            message="OK",
            output_dir=candidate,
            improvement=EvolImprovement(
                analysis={"primary_change": "lower the attack threshold"},
                files={"strategy.md": "strategy"},
            ),
        )

    config = EvolutionConfig(
        strategy="tank",
        commander_model="model",
        difficulties=("harder",),
        max_generations=1,
    )
    runner = EvolutionRunner(
        config,
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    )
    state = runner.run()

    assert state["status"] == "budget_exhausted"
    assert state["champion"] == "tank"
    assert len(state["failed_experiences"]) == 1
    assert "lower the attack threshold" in state["failed_experiences"][0]
    assert "0.50 to 0.40" in state["failed_experiences"][0]


def test_completed_record_count_ignores_autosaves(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    match = batch / "match"
    match.mkdir(parents=True)
    (match / "final.json").write_text(
        '{"metadata":{"strategy_id":"tank","save_reason":"match_runner_finally","result":"Victory"}}',
        encoding="utf-8",
    )
    (match / "autosave.json").write_text(
        '{"metadata":{"strategy_id":"tank","save_reason":"autosave_snapshot","result":"Defeat"}}',
        encoding="utf-8",
    )

    assert completed_record_count(batch, strategy="tank") == 1
