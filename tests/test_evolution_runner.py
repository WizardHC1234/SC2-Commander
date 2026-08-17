from __future__ import annotations

import csv
import json
from pathlib import Path

from evolution.outcomes import posterior_probability_better
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
    scores = {"tank": 5, "tank_opt1": 10}

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
        max_total_generations=2,
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
    assert state["completion_reason"] == "curriculum_mastered"
    assert state["champion"] == "tank_opt1"
    assert state["mastered_difficulties"] == ["harder"]
    assert state["schema"] == "sc2_evolution.v3"
    assert state["selection_protocol"] == "score_only_v1"
    assert state["games_used"] == 20
    assert state["experiment_history"][0]["decision"] == "accepted"
    assert "failed_experiences" not in state
    rows = list(csv.DictReader(runner.history_path.open(encoding="utf-8")))
    assert [row["accepted"] for row in rows] == ["true", "true"]
    assert [row["games_used"] for row in rows] == ["10", "20"]


def test_rejected_candidate_is_saved_as_experience(tmp_path: Path) -> None:
    calls = 0

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 2, tmp_path)

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
                    "hypothesis": "an earlier commitment reaches a stronger relative window",
                    "mechanism_prediction": {
                        "expected_change": "the first commitment occurs earlier",
                        "minimum_material_change": "commitment timing must materially precede the parent",
                        "outcome_prediction": "the first engagement becomes more competitive",
                        "disproof_condition": "timing improves materially but the same failure persists",
                    },
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
        max_total_generations=1,
    )
    runner = EvolutionRunner(
        config,
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    )
    state = runner.run()

    assert state["status"] == "total_budget_exhausted"
    assert state["champion"] == "tank"
    assert state["games_used"] == 20
    assert len(state["experiment_history"]) == 1
    experience = state["experiment_history"][0]
    assert experience["decision"] == "rejected"
    assert experience["implementation_verdict"] == "unknown"
    assert experience["hypothesis_verdict"] == "inconclusive"
    assert experience["mechanism_evidence"] == []
    assert experience["mechanism_prediction"]["expected_change"].startswith(
        "the first commitment"
    )
    assert "does not contradict" in experience["lesson"]
    decision_artifact = json.loads(
        (tmp_path / "run" / "generation_000" / "decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision_artifact["implementation_verdict"] == "unknown"
    assert decision_artifact["hypothesis_verdict"] == "inconclusive"
    assert decision_artifact["mechanism_evidence"] == []
    assert experience["primary_change"] == "lower the attack threshold"
    assert experience["selected_plan_ids"] == ["D1"]
    assert experience["overall_assessment"] == "the timing needs a smaller first force"
    assert experience["selected_changes"][0]["change"] == "attack with 40 instead of 45 Marines"
    assert experience["parent_score"] == 0.5
    assert experience["candidate_score"] == 0.2
    assert experience["score_delta"] == -0.3
    assert experience["evaluation"]["decision"] == "rejected"
    assert experience["evaluation"]["score_delta"] == -0.3
    assert "posterior" in experience["evaluation"]
    assert experience["champion_games"] == 10
    assert experience["candidate_games"] == 10
    assert round(
        experience["experiment_evidence"]["candidate_minus_parent"]["score_delta"], 4
    ) == -0.3


def test_candidate_evaluation_never_replays_the_champion(
    tmp_path: Path,
) -> None:
    results = {"tank": [5], "tank_opt1": [6]}
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
            max_total_generations=1,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    )
    state = runner.run()

    assert calls == {"tank": 1, "tank_opt1": 1}
    assert state["games_used"] == 20
    assert state["champion"] == "tank_opt1"
    decision = json.loads(
        (tmp_path / "run" / "generation_000" / "decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["decision"] == "accepted"
    assert decision["selection_rule"] == "candidate_score_strictly_greater"
    assert round(decision["score_delta"], 4) == 0.1
    assert decision["champion_evidence_games"] == 10
    assert decision["candidate_evidence_games"] == 10
    assert decision["parent_score"] == 0.5
    assert decision["candidate_score"] == 0.6
    assert decision["confirmation"] is None


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


def test_candidate_generation_failure_retries_without_crashing(tmp_path: Path) -> None:
    attempts = 0

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 10, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return EvolRunResult(ok=False, message="candidate contract rejected")
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=2,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    )

    state = runner.run()

    assert state["status"] == "completed"
    assert attempts == 2
    assert state["candidate_generation_failures"][0]["message"] == (
        "candidate contract rejected"
    )
    assert all(
        item.get("kind") != "candidate_generation_failure"
        for item in state.get("experiment_history") or []
    )


def test_runtime_action_pauses_without_candidate_retries(tmp_path: Path) -> None:
    attempts = 0

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        nonlocal attempts
        attempts += 1
        return EvolRunResult(
            ok=True,
            message="runtime commands are unstable",
            decision_action="inspect_runtime",
            action_reason="repeated rejected group commands",
        )

    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=2,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    )

    state = runner.run()

    assert attempts == 1
    assert state["status"] == "runtime_attention_required"
    assert state["last_agent_decision"]["action"] == "inspect_runtime"
    assert state.get("candidate_generation_failures") == []
    assert state.get("experiment_history") == []


def test_candidate_evaluation_requests_exactly_ten_games(tmp_path: Path) -> None:
    requested: list[tuple[str, int]] = []

    def play(strategy: str, difficulty: str, target_games: int = 10) -> BatchResult:
        requested.append((strategy, target_games))
        wins = 8 if strategy != "tank" else 5
        return BatchResult(
            name=f"{strategy}_{target_games}",
            path=tmp_path / f"{strategy}_{target_games}",
            strategy=strategy,
            difficulty=difficulty,
            wins=wins,
            draws=0,
            losses=target_games - wins,
        )

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=1,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert requested == [("tank", 10), ("tank_opt1", 10)]


def test_equal_five_five_is_inconclusive(tmp_path: Path) -> None:
    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=1,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert state["champion"] == "tank"
    assert state["experiment_history"][0]["decision"] == "inconclusive"
    assert state["experiment_history"][0]["score_delta"] == 0.0
    assert state["experiment_history"][0]["evaluation"]["decision"] == "inconclusive"
    posterior = state["experiment_history"][0]["posterior_probability_better"]
    assert abs(posterior - 0.5) < 0.02
    assert abs(posterior_probability_better(
        {"wins": 5, "draws": 0, "losses": 5},
        {"wins": 5, "draws": 0, "losses": 5},
    ) - 0.5) < 0.02


def test_champion_baseline_ignores_historical_pool_games(tmp_path: Path) -> None:
    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 2, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=1,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    )
    state = runner.load_or_create_state()
    historical = BatchResult(
        name="historical_champ",
        path=tmp_path / "historical_champ",
        strategy="tank",
        difficulty="harder",
        wins=9,
        draws=0,
        losses=1,
    )
    runner._register_evidence(state, historical)
    runner._save_state(state)

    state = runner.run()
    decision = json.loads(
        (tmp_path / "run" / "generation_000" / "decision.json").read_text(encoding="utf-8")
    )
    assert decision["champion_evidence_games"] == 10
    assert decision["parent_score"] == 0.5
    assert decision["decision"] == "rejected"
    assert state["champion"] == "tank"


def test_generation_failures_are_not_prior_experiences(tmp_path: Path) -> None:
    seen: list[list[object]] = []

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 10, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        seen.append(list(experiences))
        if len(seen) == 1:
            return EvolRunResult(ok=False, message="optimizer json invalid")
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=2,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert seen[0] == []
    assert seen[1] == []
    assert state["candidate_generation_failures"][0]["message"] == "optimizer json invalid"
    assert state["experiment_history"][0]["decision"] == "accepted"


def test_resume_pending_candidate_does_not_regenerate(tmp_path: Path) -> None:
    evolve_calls = 0
    requested: list[tuple[str, int]] = []

    def play(strategy: str, difficulty: str, target_games: int = 10) -> BatchResult:
        requested.append((strategy, target_games))
        wins = 2 if strategy != "tank" else 5
        return BatchResult(
            name=f"{strategy}_{len(requested)}",
            path=tmp_path / f"{strategy}_{len(requested)}",
            strategy=strategy,
            difficulty=difficulty,
            wins=wins,
            draws=0,
            losses=target_games - wins,
        )

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        nonlocal evolve_calls
        evolve_calls += 1
        raise AssertionError("pending candidate must not be regenerated")

    candidate_dir = tmp_path / "skills" / "terran" / "tank_opt1"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=1,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    )
    champion_batch = _batch("tank", "harder", 5, tmp_path)
    partial = BatchResult(
        name="partial_cand",
        path=tmp_path / "partial_cand",
        strategy="tank_opt1",
        difficulty="harder",
        wins=3,
        draws=0,
        losses=4,
    )
    state = runner.load_or_create_state()
    state["champion_batch"] = champion_batch.to_dict()
    runner._register_evidence(state, champion_batch)
    runner._register_evidence(state, partial)
    state["pending_candidate"] = {
        "strategy": "tank_opt1",
        "strategy_dir": str(candidate_dir),
        "candidate_hash": "abc",
        "experiment_spec": {
            "hypothesis": "attack earlier",
            "plan_direction": "lower the gather gate",
            "patches": [{"target": "Main Attack Gate", "why_required": "timing"}],
            "expected_effect": "hit sooner",
            "main_risk": "thin army",
        },
        "candidate_batch": partial.to_dict(),
        "hypothesis": "attack earlier",
    }
    runner._sync_games_used(state)
    runner._save_state(state)

    state = runner.run()
    assert evolve_calls == 0
    assert requested == [("tank_opt1", 10)]
    assert len(state["experiment_history"]) == 1
    assert state["experiment_history"][0]["decision"] == "rejected"
    assert state["pending_candidate"] is None


def test_run_batch_requests_only_remaining_games(tmp_path: Path, monkeypatch) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    batch_name = runner._batch_name(0, "cand")
    batch_dir = tmp_path / "game_records" / batch_name
    for index in range(7):
        match = batch_dir / f"match_{index:03d}"
        match.mkdir(parents=True)
        (match / "final.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "strategy_id": "tank_opt1",
                        "save_reason": "match_runner_finally",
                        "result": "Victory",
                    }
                }
            ),
            encoding="utf-8",
        )

    captured: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        captured["command"] = [str(item) for item in command]
        for index in range(7, 10):
            match = batch_dir / f"match_{index:03d}"
            match.mkdir(parents=True)
            (match / "final.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "strategy_id": "tank_opt1",
                            "save_reason": "match_runner_finally",
                            "result": "Defeat",
                        }
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr("evolution.runner.subprocess.run", fake_run)
    result = runner.run_batch(
        "tank_opt1",
        "harder",
        generation=0,
        role="cand",
        target_games=10,
    )
    command = captured["command"]
    if "-TOTAL_MATCHES" in command:
        assert command[command.index("-TOTAL_MATCHES") + 1] == "3"
        assert command[command.index("-START_INDEX") + 1] == "7"
    else:
        assert command[command.index("--total-matches") + 1] == "3"
        assert command[command.index("--start-index") + 1] == "7"
    assert result.games == 10
    assert result.wins == 7
    assert result.losses == 3


def test_six_patches_produce_one_experiment_record(tmp_path: Path) -> None:
    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 2, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        patches = [
            {"target": f"Detail {index}", "why_required": f"needed {index}"}
            for index in range(1, 7)
        ]
        return EvolRunResult(
            ok=True,
            message="OK",
            output_dir=candidate,
            improvement=EvolImprovement(
                analysis={
                    "hypothesis": "one timing hypothesis",
                    "plan_direction": "adjust several supporting details",
                    "patches": patches,
                    "expected_effect": "earlier attack",
                    "main_risk": "supply block",
                },
                files={"strategy.md": "strategy"},
            ),
        )

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=1,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert len(state["experiment_history"]) == 1
    record = state["experiment_history"][0]
    assert record["hypothesis"] == "one timing hypothesis"
    assert len(record["patches"]) == 6
    pending_was_cleared = state["pending_candidate"] is None
    assert pending_was_cleared


def test_migrates_failed_experiences_into_experiment_history(tmp_path: Path) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    state = runner.load_or_create_state()
    state["schema"] = "sc2_evolution.v2"
    state["config"] = {
        **state["config"],
        "candidate_initial_matches": 6,
        "candidate_max_matches": 10,
        "candidate_step_matches": 2,
    }
    del state["config"]["candidate_matches"]
    state["failed_experiences"] = [
        {
            "generation": 0,
            "candidate": "tank_opt1",
            "hypothesis": "legacy hypothesis",
        }
    ]
    del state["experiment_history"]
    runner._save_state(state)

    loaded = runner.load_or_create_state()
    assert loaded["schema"] == "sc2_evolution.v3"
    assert "failed_experiences" not in loaded
    assert loaded["experiment_history"][0]["decision"] == "rejected"
    assert loaded["experiment_history"][0]["legacy"] is True
    assert loaded["experiment_history"][0]["hypothesis"] == "legacy hypothesis"
    assert loaded["experiment_history"][0]["implementation_verdict"] == "unknown"
    assert loaded["experiment_history"][0]["hypothesis_verdict"] == "inconclusive"
    assert loaded["experiment_history"][0]["mechanism_prediction"] == {}
    assert loaded["experiment_history"][0]["mechanism_evidence"] == []
    assert "candidate_matches" in loaded["config"]


def test_nine_wins_one_loss_does_not_master_difficulty(tmp_path: Path) -> None:
    evolve_calls = 0

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 9, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        nonlocal evolve_calls
        evolve_calls += 1
        return EvolRunResult(
            ok=True,
            message="stop",
            decision_action="stop",
            action_reason="not enough improvement",
        )

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder", "veryhard"),
            max_total_generations=10,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert evolve_calls == 1
    assert state["status"] == "stopped_no_actionable_improvement"
    assert state["champion"] == "tank"
    assert state["difficulty_index"] == 0
    assert state["mastered_difficulties"] == []
    assert state["champion_baseline"]["score"] == 0.9
    assert state.get("experiment_history") == []
    assert state["difficulty_generation"] == 0


def test_nine_wins_one_draw_masters_difficulty(tmp_path: Path) -> None:
    def play(strategy: str, difficulty: str) -> BatchResult:
        return BatchResult(
            name=f"{strategy}_{difficulty}",
            path=tmp_path / f"{strategy}_{difficulty}",
            strategy=strategy,
            difficulty=difficulty,
            wins=9,
            draws=1,
            losses=0,
        )

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        raise AssertionError("0.95 champion must master without a candidate")

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert state["status"] == "completed"
    assert state["mastered_difficulties"] == ["harder"]
    assert state["champion_baseline"] is None


def test_ten_wins_masters_each_difficulty_without_candidates(tmp_path: Path) -> None:
    def play(strategy: str, difficulty: str) -> BatchResult:
        return BatchResult(
            name=f"{strategy}_{difficulty}",
            path=tmp_path / f"{strategy}_{difficulty}",
            strategy=strategy,
            difficulty=difficulty,
            wins=10,
            draws=0,
            losses=0,
        )

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        raise AssertionError("mastered champion must not generate a candidate")

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder", "veryhard"),
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert state["status"] == "completed"
    assert state["completion_reason"] == "curriculum_mastered"
    assert state["champion"] == "tank"
    assert state["mastered_difficulties"] == ["harder", "veryhard"]
    assert state["games_used"] == 20
    assert state["experiment_history"] == []


def test_accepted_below_mastery_keeps_same_difficulty(tmp_path: Path) -> None:
    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 8, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder", "veryhard"),
            max_total_generations=1,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert state["status"] == "total_budget_exhausted"
    assert state["champion"] == "tank_opt1"
    assert state["difficulty_index"] == 0
    assert state["mastered_difficulties"] == []
    assert state["champion_baseline"]["score"] == 0.8
    assert state["experiment_history"][0]["decision"] == "accepted"


def test_request_more_matches_adds_analysis_games_then_reruns(tmp_path: Path) -> None:
    requested: list[tuple[str, int]] = []
    evolve_calls = 0

    def play(strategy: str, difficulty: str, target_games: int = 10) -> BatchResult:
        requested.append((strategy, target_games))
        index = len(requested)
        wins = 5 if strategy == "tank" else 2
        return BatchResult(
            name=f"{strategy}_{index}",
            path=tmp_path / f"{strategy}_{index}",
            strategy=strategy,
            difficulty=difficulty,
            wins=wins,
            draws=0,
            losses=target_games - wins,
        )

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        nonlocal evolve_calls
        evolve_calls += 1
        if evolve_calls == 1:
            return EvolRunResult(
                ok=True,
                message="need more matches",
                decision_action="request_more_matches",
                action_reason="sparse fights",
            )
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=1,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert evolve_calls == 2
    assert requested == [("tank", 10), ("tank", 10), ("tank_opt1", 10)]
    assert state["champion_baseline"]["games"] == 10
    assert state["champion_baseline"]["score"] == 0.5
    assert len(state["experiment_history"]) == 1
    assert state["difficulty_generation"] == 1


def test_request_more_matches_stops_after_analysis_cap(tmp_path: Path) -> None:
    requested: list[tuple[str, int]] = []

    def play(strategy: str, difficulty: str, target_games: int = 10) -> BatchResult:
        requested.append((strategy, target_games))
        index = len(requested)
        return BatchResult(
            name=f"tank_{index}",
            path=tmp_path / f"tank_{index}",
            strategy=strategy,
            difficulty=difficulty,
            wins=5,
            draws=0,
            losses=target_games - 5,
        )

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        return EvolRunResult(
            ok=True,
            message="still not enough",
            decision_action="request_more_matches",
        )

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=5,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert state["status"] == "insufficient_evidence"
    assert requested == [("tank", 10), ("tank", 10)]
    assert state.get("experiment_history") == []
    assert state["difficulty_generation"] == 0
    assert state["generation"] == 0


def test_difficulty_budget_exhausted_does_not_skip_ahead(tmp_path: Path) -> None:
    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 2, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder", "veryhard"),
            max_generations_per_difficulty=1,
            max_total_generations=10,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert state["status"] == "difficulty_budget_exhausted"
    assert state["failed_difficulty"] == "harder"
    assert state["difficulty_index"] == 0
    assert state["champion"] == "tank"
    assert state["difficulty_generation"] == 1
    assert len(state["experiment_history"]) == 1


def test_experiment_id_uses_style_generation_difficulty_candidate(tmp_path: Path) -> None:
    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 2, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=1,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert state["experiment_history"][0]["experiment_id"] == "tank:g000:harder:tank_opt1"


def test_nine_vs_ten_is_accepted_and_masters(tmp_path: Path) -> None:
    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 9 if strategy == "tank" else 10, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=5,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    record = state["experiment_history"][0]
    assert record["decision"] == "accepted"
    assert record["implementation_verdict"] == "unknown"
    assert record["hypothesis_verdict"] == "supported"
    assert round(record["score_delta"], 4) == 0.1
    assert record["evaluation"]["champion"]["score"] == 0.9
    assert record["evaluation"]["candidate"]["score"] == 1.0
    assert "posterior" in record["evaluation"]
    assert state["selection_protocol"] == "score_only_v1"
    assert state["status"] == "completed"
    assert state["champion"] == "tank_opt1"
    assert state["mastered_difficulties"] == ["harder"]

