from __future__ import annotations

import csv
from pathlib import Path

from evolution.runner import (
    BatchResult,
    EvolutionConfig,
    EvolutionRunner,
    close_batch_results,
    completed_record_count,
)
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
                analysis={
                    "primary_change": "lower the attack threshold",
                    "selected_plan_ids": ["D1"],
                    "overall_assessment": "the timing needs a smaller first force",
                    "selected_changes": [
                        {
                            "source_plan_id": "D1",
                            "problem_id": "P1",
                            "change": "attack with 40 instead of 45 Marines",
                            "why": "reach the timing sooner",
                        }
                    ],
                    "expected_effect": "attack sooner",
                    "main_risk": "smaller first force",
                },
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
    assert state["games_used"] == 40
    assert len(state["failed_experiences"]) == 1
    experience = state["failed_experiences"][0]
    assert experience["primary_change"] == "lower the attack threshold"
    assert experience["selected_plan_ids"] == ["D1"]
    assert experience["overall_assessment"] == "the timing needs a smaller first force"
    assert experience["selected_changes"][0]["change"] == "attack with 40 instead of 45 Marines"
    assert experience["parent_score"] == 0.5
    assert experience["candidate_score"] == 0.4
    assert experience["games_per_strategy"] == 20


def test_close_result_runs_confirmation_for_both_and_uses_cumulative_score(
    tmp_path: Path,
) -> None:
    results = {"tank": [5, 8], "tank_opt1": [6, 4]}
    calls = {"tank": 0, "tank_opt1": 0}

    def play(strategy: str, difficulty: str) -> BatchResult:
        index = calls[strategy]
        calls[strategy] += 1
        return _batch(strategy, difficulty, results[strategy][index], tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_generations=1,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    )
    state = runner.run()

    assert calls == {"tank": 2, "tank_opt1": 2}
    assert state["games_used"] == 40
    assert state["champion"] == "tank"
    decision = __import__("json").loads(
        (tmp_path / "run" / "generation_000" / "decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["comparison_games_per_strategy"] == 20
    assert decision["parent_score"] == 0.65
    assert decision["candidate_score"] == 0.5
    assert decision["confirmation"] is not None


def test_close_batch_results_uses_one_outcome_point_threshold(tmp_path: Path) -> None:
    champion = _batch("tank", "harder", 5, tmp_path)
    assert close_batch_results(champion, _batch("tank_opt1", "harder", 6, tmp_path))
    assert not close_batch_results(
        champion, _batch("tank_opt1", "harder", 7, tmp_path)
    )


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
