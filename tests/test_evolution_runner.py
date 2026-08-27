from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evolution.outcomes import posterior_probability_better
from evolution.runner import (
    BatchResult,
    EvolutionConfig,
    EvolutionRunner,
    close_batch_results,
    completed_record_count,
    curriculum_progress_score,
)
from evol_agent.core.types import EvolImprovement, EvolRunResult
from evol_agent.core.checkpoint import CHECKPOINT_SCHEMA, PIPELINE_VERSION, EvolCheckpoint
from evol_agent.core.types import BattleAnalysis, GameDigest


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


def test_generate_candidate_passes_shared_match_summary_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    class FakeEvolAgent:
        def __init__(self, *, model: str = "") -> None:
            captured["model"] = model

        def run(self, request):
            captured["request"] = request
            return EvolRunResult(ok=False, message="captured")

    monkeypatch.setattr("evolution.runner.EvolAgent", FakeEvolAgent)
    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="commander-model",
            evolution_model="analysis-model",
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    batch = _batch("tank", "harder", 5, tmp_path)

    runner.generate_candidate(
        "tank",
        batch,
        [],
        evidence_batches=[batch],
        retry_feedback=["candidate contract rejected"],
    )

    assert captured["request"].match_summary_cache_path == (
        tmp_path / "run" / "experiment_match_summary_cache.json"
    )
    assert captured["request"].retry_feedback == ["candidate contract rejected"]
    assert captured["request"].output_dir == (
        tmp_path / "run" / "strategies" / "tank_opt1"
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
    assert state["selection_protocol"] == "score_only_v2"
    assert state["games_used"] == 20
    assert state["experiment_history"][0]["decision"] == "accepted"
    assert "failed_experiences" not in state
    rows = list(csv.DictReader(runner.history_path.open(encoding="utf-8")))
    assert [row["accepted"] for row in rows] == ["true", "true"]
    assert [row["generation"] for row in rows] == ["0", "1"]
    assert [row["games"] for row in rows] == ["10", "10"]
    assert [row["win_rate"] for row in rows] == ["0.5000", "1.0000"]
    assert rows[1]["curriculum_progress_score"] == "1.0000"


def test_curriculum_progress_score_uses_mastery_threshold() -> None:
    assert curriculum_progress_score(0, 0.5, 0.9) == pytest.approx(0.5555556)
    assert curriculum_progress_score(0, 0.9, 0.9) == 1.0
    assert curriculum_progress_score(1, 0.45, 0.9) == 1.5
    assert curriculum_progress_score(0, 0.8, 0.9) == pytest.approx(0.8888889)


def test_runner_finds_latest_completed_analysis_seed_for_record_subset(
    tmp_path: Path,
) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(strategy="tank", commander_model="test-model"),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    records = [tmp_path / f"match_{index}.json" for index in range(3)]
    for path in records:
        path.write_text("{}", encoding="utf-8")
    seed_dir = tmp_path / "evol_agent" / "logs" / "tank" / "seed"
    seed_dir.mkdir(parents=True)
    seed = EvolCheckpoint(
        seed_dir,
        {
            "schema": CHECKPOINT_SCHEMA,
            "pipeline_version": PIPELINE_VERSION,
            "stage": "created",
            "strategy_name": "tank",
            "race": "terran",
            "knowledge_mode": "enabled",
            "models": {"analysis": "test-model"},
            "record_files": [str(path.resolve()) for path in records[:2]],
        },
    )
    seed.save_match_summaries(
        game_digests=[
            GameDigest(
                record_path=str(path.resolve()),
                result="Defeat",
                duration="10:00",
                summary="cached",
            )
            for path in records[:2]
        ],
        single_game_analyses=[
            BattleAnalysis(
                strategy_name="tank",
                race="terran",
                sample_size=1,
                record_mix="0W/1L",
                raw={"summary": "cached"},
            )
            for _path in records[:2]
        ],
        completed_matches=2,
        events=[
            {"record_path": str(path.resolve()), "completed": True}
            for path in records[:2]
        ],
    )
    seed.save_analysis_complete(
        battle_analysis=BattleAnalysis(
            strategy_name="tank",
            race="terran",
            sample_size=2,
            record_mix="0W/2L",
            raw={"record_mix": "0W/2L"},
        ),
        tool_observations=[],
        knowledge_trace={},
    )

    found = runner._find_analysis_seed_checkpoint(
        runner._new_state(),
        difficulty="harder",
        strategy="tank",
        record_paths=records,
    )

    assert found == seed_dir.resolve()


def test_runner_recovers_latest_unfinished_checkpoint_for_exact_records(
    tmp_path: Path,
) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(strategy="tank", commander_model="test-model"),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    records = [tmp_path / f"match_{index}.json" for index in range(2)]
    for path in records:
        path.write_text("{}", encoding="utf-8")
    checkpoint_dir = tmp_path / "evol_agent" / "logs" / "tank" / "unfinished"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = EvolCheckpoint(
        checkpoint_dir,
        {
            "schema": CHECKPOINT_SCHEMA,
            "pipeline_version": PIPELINE_VERSION,
            "stage": "created",
            "strategy_name": "tank",
            "race": "terran",
            "knowledge_mode": "enabled",
            "models": {"analysis": "test-model"},
            "record_files": [str(path.resolve()) for path in records],
        },
    )
    checkpoint.save_match_summaries(
        game_digests=[
            GameDigest(
                record_path=str(path.resolve()),
                result="Defeat",
                duration="10:00",
                summary="cached",
            )
            for path in records
        ],
        single_game_analyses=[
            BattleAnalysis(
                strategy_name="tank",
                race="terran",
                sample_size=1,
                record_mix="0W/1L",
                raw={"summary": "cached"},
            )
            for _path in records
        ],
        completed_matches=2,
        events=[
            {"record_path": str(path.resolve()), "completed": True}
            for path in records
        ],
    )

    recovered = runner._find_resumable_analysis_checkpoint(
        strategy="tank",
        record_paths=records,
    )

    assert recovered == checkpoint_dir.resolve()


def test_runner_does_not_resume_completed_terminal_analysis(
    tmp_path: Path,
) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(strategy="tank", commander_model="test-model"),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    record = tmp_path / "match.json"
    record.write_text("{}", encoding="utf-8")
    checkpoint_dir = tmp_path / "evol_agent" / "logs" / "tank" / "terminal"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema": CHECKPOINT_SCHEMA,
                "pipeline_version": PIPELINE_VERSION,
                "stage": "analysis_complete",
                "strategy_name": "tank",
                "race": "terran",
                "knowledge_mode": "enabled",
                "models": {"analysis": "test-model"},
                "record_files": [str(record.resolve())],
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "analysis.json").write_text(
        json.dumps({"next_action": "inspect_runtime"}),
        encoding="utf-8",
    )

    recovered = runner._find_resumable_analysis_checkpoint(
        strategy="tank",
        record_paths=[record],
    )

    assert recovered is None


def test_legacy_history_migrates_to_per_strategy_games(tmp_path: Path) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(strategy="tank", commander_model="model"),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    runner.run_dir.mkdir(parents=True, exist_ok=True)
    runner.history_path.write_text(
        "strategy_style,generation,strategy,parent,difficulty,wins,draws,losses,"
        "score,mastered_levels,evolution_score,accepted,games_used,batch\n"
        "tank,0,tank,,harder,5,0,5,0.5000,0,0.1000,true,10,baseline\n"
        "tank,0,tank_opt1,tank,harder,10,0,4,0.7143,0,0.1429,true,28,candidate\n",
        encoding="utf-8",
    )

    runner._ensure_history_schema()

    rows = list(csv.DictReader(runner.history_path.open(encoding="utf-8")))
    assert [row["generation"] for row in rows] == ["0", "1"]
    assert [row["games"] for row in rows] == ["10", "14"]
    assert [row["win_rate"] for row in rows] == ["0.5000", "0.7143"]
    assert rows[0]["curriculum_progress_score"] == "0.5556"
    assert rows[1]["curriculum_progress_score"] == "0.7937"
    assert "games_used" not in rows[1]


def test_intermediate_history_migration_keeps_generation_number(
    tmp_path: Path,
) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(strategy="tank", commander_model="model"),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    runner.run_dir.mkdir(parents=True, exist_ok=True)
    runner.history_path.write_text(
        "strategy_style,generation,strategy,parent,difficulty,wins,draws,losses,"
        "games,score,comparison_strategy,mastered_levels,"
        "curriculum_progress_score,accepted,batch\n"
        "tank,1,tank_opt1,tank,harder,10,0,4,14,0.7143,tank,0,0.7937,true,candidate\n",
        encoding="utf-8",
    )

    runner._ensure_history_schema()

    rows = list(csv.DictReader(runner.history_path.open(encoding="utf-8")))
    assert rows[0]["generation"] == "1"
    assert rows[0]["games"] == "14"
    assert rows[0]["win_rate"] == "0.7143"
    assert "comparison_strategy" not in rows[0]


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

    assert state["status"] == "completed"
    assert state["completion_reason"] == "generation_budget_reached"
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


def test_close_result_runs_confirmation_and_keeps_champion_when_evidence_fades(
    tmp_path: Path,
) -> None:
    results = {"tank": [6, 6], "tank_opt1": [7, 5]}
    calls = {"tank": 0, "tank_opt1": 0}

    def play(
        strategy: str, difficulty: str, expected_games: int = 10
    ) -> BatchResult:
        assert expected_games == 10
        index = calls[strategy]
        calls[strategy] += 1
        wins = results[strategy][index]
        return BatchResult(
            name=f"{strategy}_{difficulty}_{index}",
            path=tmp_path / f"{strategy}_{difficulty}_{index}",
            strategy=strategy,
            difficulty=difficulty,
            wins=wins,
            draws=0,
            losses=10 - wins,
        )

    def evolve(
        champion: str, batch: BatchResult, experiences: list[object]
    ) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=1,
            confirmation_matches=10,
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
    assert state["selection_protocol"] == "confirmed_score_only_v2"
    decision = json.loads(
        (tmp_path / "run" / "generation_000" / "decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["decision"] == "inconclusive"
    assert decision["confirmation"] is not None
    assert decision["champion_evidence_games"] == 20
    assert decision["candidate_evidence_games"] == 20
    history_rows = list(csv.DictReader(runner.history_path.open(encoding="utf-8")))
    champion_row = history_rows[0]
    candidate_row = history_rows[-1]
    assert champion_row["games"] == "20"
    assert champion_row["wins"] == "12"
    assert champion_row["losses"] == "8"
    assert champion_row["win_rate"] == "0.6000"
    assert champion_row["batch"] == "tank_harder_0+tank_harder_1"
    assert candidate_row["generation"] == "1"
    assert candidate_row["games"] == "20"
    assert candidate_row["wins"] == "12"
    assert candidate_row["losses"] == "8"
    assert candidate_row["win_rate"] == "0.6000"
    assert "total_games_used" not in candidate_row


def test_later_close_candidate_is_topped_up_without_replaying_champion(
    tmp_path: Path,
) -> None:
    results = {
        "tank": [(5, 10), (0, 4)],
        "tank_opt1": [(6, 10), (4, 4)],
        "tank_opt2": [(7, 10), (3, 4)],
    }
    calls = {strategy: 0 for strategy in results}

    def play(
        strategy: str, difficulty: str, expected_games: int = 10
    ) -> BatchResult:
        index = calls[strategy]
        calls[strategy] += 1
        wins, games = results[strategy][index]
        assert expected_games == games
        return BatchResult(
            name=f"{strategy}_{index}",
            path=tmp_path / f"{strategy}_{index}",
            strategy=strategy,
            difficulty=difficulty,
            wins=wins,
            draws=0,
            losses=games - wins,
        )

    def evolve(
        champion: str, batch: BatchResult, experiences: list[object]
    ) -> EvolRunResult:
        candidate_name = "tank_opt1" if champion == "tank" else "tank_opt2"
        candidate = tmp_path / "skills" / "terran" / candidate_name
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=2,
            confirmation_matches=4,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    )

    state = runner.run()

    assert calls == {"tank": 2, "tank_opt1": 2, "tank_opt2": 2}
    assert state["games_used"] == 42
    decision = json.loads(
        (tmp_path / "run" / "generation_001" / "decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["confirmation"]["champion_batch"] is None
    candidate_confirmation = decision["confirmation"]["candidate_batch"]
    assert (
        candidate_confirmation["wins"]
        + candidate_confirmation["draws"]
        + candidate_confirmation["losses"]
        == 4
    )
    assert decision["champion_evidence_games"] == 14
    assert decision["candidate_evidence_games"] == 14


def test_later_generation_does_not_overwrite_confirmed_champion_with_smaller_sample(
    tmp_path: Path,
) -> None:
    results = {
        "tank": [(5, 10), (3, 4)],
        "tank_opt1": [(5, 10), (2, 4)],
        "tank_opt2": [(9, 10)],
    }
    calls = {strategy: 0 for strategy in results}
    generation = 0

    def play(
        strategy: str, difficulty: str, expected_games: int = 10
    ) -> BatchResult:
        index = calls[strategy]
        calls[strategy] += 1
        wins, games = results[strategy][index]
        assert expected_games == games
        return BatchResult(
            name=f"{strategy}_{index}",
            path=tmp_path / f"{strategy}_{index}",
            strategy=strategy,
            difficulty=difficulty,
            wins=wins,
            draws=0,
            losses=games - wins,
        )

    def evolve(
        champion: str, batch: BatchResult, experiences: list[object]
    ) -> EvolRunResult:
        nonlocal generation
        generation += 1
        candidate = tmp_path / "skills" / "terran" / f"tank_opt{generation}"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=2,
            confirmation_matches=4,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    )

    state = runner.run()

    rows = list(csv.DictReader(runner.history_path.open(encoding="utf-8")))
    tank_row = next(row for row in rows if row["strategy"] == "tank")
    assert tank_row["wins"] == "8"
    assert tank_row["losses"] == "6"
    assert tank_row["games"] == "14"
    assert tank_row["batch"] == "tank_0+tank_1"
    assert state["champion"] == "tank_opt2"


def test_candidate_with_ninety_percent_win_rate_skips_confirmation_and_advances(
    tmp_path: Path,
) -> None:
    parent_dir = tmp_path / "skills" / "terran" / "tank"
    parent_dir.mkdir(parents=True)
    (parent_dir / "strategy.md").write_text("parent strategy", encoding="utf-8")
    results = {"tank": 8, "tank_opt1": 9}
    calls = {"tank": 0, "tank_opt1": 0}

    def play(
        strategy: str, difficulty: str, expected_games: int = 10
    ) -> BatchResult:
        assert expected_games == 10
        calls[strategy] += 1
        return _batch(strategy, difficulty, results[strategy], tmp_path)

    def evolve(
        champion: str, batch: BatchResult, experiences: list[object]
    ) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        (candidate / "strategy.md").write_text(
            "candidate strategy", encoding="utf-8"
        )
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    def audit(**_kwargs: object) -> dict[str, object]:
        return {
            "implementation_verdict": "implemented",
            "hypothesis_verdict": "supported",
            "mechanism_evidence": ["the candidate realized the planned change"],
            "combat_evidence": ["the decisive engagement improved"],
            "runtime_findings": [],
            "evidence_limits": [],
            "lesson": "The mastered candidate also passed the mechanism audit.",
        }

    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=1,
            confirmation_matches=4,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
        experiment_auditor=audit,
    )
    state = runner.run()

    assert calls == {"tank": 1, "tank_opt1": 1}
    assert state["status"] == "completed"
    assert state["champion"] == "tank_opt1"
    assert state["mastered_difficulties"] == ["harder"]
    decision = json.loads(
        (tmp_path / "run" / "generation_000" / "decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["decision"] == "accepted"
    assert decision["candidate_mastered"] is True
    assert decision["candidate_win_rate"] == 0.9
    assert decision["confirmation"] is None
    assert decision["selection_rule"] == "candidate_score_strictly_greater"
    assert decision["implementation_verdict"] == "implemented"
    assert decision["audit_evidence_limits"] == []


def test_execution_invalid_audit_blocks_candidate_promotion(tmp_path: Path) -> None:
    parent = tmp_path / "skills" / "terran" / "tank"
    parent.mkdir(parents=True)
    (parent / "strategy.md").write_text("parent", encoding="utf-8")

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 8, tmp_path)

    def evolve(
        champion: str, batch: BatchResult, experiences: list[object]
    ) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        (candidate / "strategy.md").write_text("candidate", encoding="utf-8")
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    def audit(**kwargs: object) -> dict[str, object]:
        return {
            "implementation_verdict": "execution_invalid",
            "hypothesis_verdict": "not_tested",
            "mechanism_evidence": [],
            "combat_evidence": [],
            "runtime_findings": ["the strategy requires runtime-owned micro"],
            "evidence_limits": [],
            "lesson": "The proposed mechanism is not strategy-controllable.",
        }

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
        experiment_auditor=audit,
    )
    state = runner.run()

    assert state["champion"] == "tank"
    record = state["experiment_history"][0]
    assert record["decision"] == "inconclusive"
    assert record["implementation_verdict"] == "execution_invalid"
    decision = json.loads(
        (tmp_path / "run" / "generation_000" / "decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["promotion_blocked_by_audit"] is True


def test_accepted_candidate_keeps_score_gain_separate_from_contradicted_cause(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "skills" / "terran" / "tank"
    parent.mkdir(parents=True)
    (parent / "strategy.md").write_text("parent", encoding="utf-8")

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 6, tmp_path)

    def evolve(
        champion: str, batch: BatchResult, experiences: list[object]
    ) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        (candidate / "strategy.md").write_text("candidate", encoding="utf-8")
        improvement = EvolImprovement(
            analysis={
                "mechanism_family": "viking_air_denial_vs_liberator",
                "hypothesis": "the air response explains repeated losses",
            },
            files={"strategy.md": "candidate"},
        )
        return EvolRunResult(
            ok=True,
            message="OK",
            output_dir=candidate,
            improvement=improvement,
        )

    def audit(**_kwargs: object) -> dict[str, object]:
        return {
            "implementation_verdict": "implemented",
            "hypothesis_verdict": "contradicted",
            "mechanism_evidence": ["the requested unit was produced"],
            "combat_evidence": ["losses persisted without the predicted cause"],
            "runtime_findings": [],
            "evidence_limits": [],
            "lesson": "The score improved, but the selected mechanism was contradicted.",
        }

    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=1,
            confirmation_matches=0,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
        experiment_auditor=audit,
    )
    state = runner.run()

    record = state["experiment_history"][0]
    assert state["champion"] == "tank_opt1"
    assert record["decision"] == "accepted"
    assert record["performance_result"] == "accepted"
    assert record["causal_result"] == "contradicted"
    assert record["performance_gain_cause"] == "unknown"
    assert record["mechanism_signature"] == "viking_air_denial_vs_liberator"
    blocked = runner._blocked_mechanism_families(state, difficulty="harder")
    assert blocked == {}


def test_close_batch_results_uses_one_outcome_point_threshold(tmp_path: Path) -> None:
    champion = _batch("tank", "harder", 5, tmp_path)
    assert close_batch_results(champion, _batch("tank_opt1", "harder", 6, tmp_path))
    assert not close_batch_results(
        champion, _batch("tank_opt1", "harder", 7, tmp_path)
    )


def test_close_batch_results_compares_rates_when_game_counts_differ(
    tmp_path: Path,
) -> None:
    champion = BatchResult(
        name="champion",
        path=tmp_path / "champion",
        strategy="tank",
        difficulty="harder",
        wins=10,
        draws=0,
        losses=4,
    )
    candidate = _batch("tank_opt1", "harder", 7, tmp_path)

    assert close_batch_results(champion, candidate)


def test_mechanism_family_is_blocked_after_two_inconclusive_attempts(
    tmp_path: Path,
) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(strategy="tank", commander_model="model"),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    state = runner._new_state()
    state["experiment_history"] = [
        {
            "difficulty": "harder",
            "mechanism_family": "anti_air_support",
            "decision": "inconclusive",
            "implementation_verdict": "implemented",
            "hypothesis_verdict": "inconclusive",
        },
        {
            "difficulty": "harder",
            "mechanism_family": "anti_air_support",
            "decision": "inconclusive",
            "implementation_verdict": "underpowered",
            "hypothesis_verdict": "not_tested",
        },
    ]

    blocked = runner._blocked_mechanism_families(state, difficulty="harder")

    assert "anti_air_support" in blocked


def test_execution_invalid_does_not_block_entire_mechanism_family_immediately(
    tmp_path: Path,
) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(strategy="tank", commander_model="model"),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    state = runner._new_state()
    state["experiment_history"] = [
        {
            "difficulty": "harder",
            "mechanism_family": "runtime_transformation_gate",
            "decision": "rejected",
            "implementation_verdict": "execution_invalid",
            "hypothesis_verdict": "not_tested",
        }
    ]

    blocked = runner._blocked_mechanism_families(state, difficulty="harder")

    assert "runtime_transformation_gate" not in blocked


def test_mechanism_policy_leaves_semantic_rename_judgement_to_analysis(
    tmp_path: Path,
) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(strategy="tank", commander_model="model"),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    state = runner._new_state()
    state["experiment_history"] = [
        {
            "difficulty": "harder",
            "mechanism_family": "viking_air_denial_vs_liberator",
            "decision": "inconclusive",
            "implementation_verdict": "implemented",
            "hypothesis_verdict": "contradicted",
        }
    ]

    blocked = runner._blocked_mechanism_families(state, difficulty="harder")
    blocked_family, reason = runner._blocked_mechanism_reason(
        blocked,
        "viking_count_and_timing_vs_liberator",
    )

    assert blocked_family == ""
    assert reason == ""

    blocked_family, reason = runner._blocked_mechanism_reason(
        blocked,
        "viking_air_denial_vs_liberator",
    )
    assert blocked_family == "viking_air_denial_vs_liberator"
    assert "contradicted" in reason


def test_stale_policy_rejection_is_not_returned_after_block_is_relaxed(
    tmp_path: Path,
) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(strategy="tank", commander_model="model"),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    state = runner._new_state()
    state["experiment_history"] = [
        {
            "difficulty": "harder",
            "mechanism_family": "earlier_commitment_window",
            "decision": "accepted",
            "implementation_verdict": "implemented",
            "hypothesis_verdict": "contradicted",
        }
    ]
    state["mechanism_policy_rejections"] = [
        {
            "kind": "mechanism_policy_rejection",
            "difficulty": "harder",
            "mechanism_family": "earlier_commitment_window",
        }
    ]

    experiences = runner._prior_experiences(state, difficulty="harder")

    assert len(experiences) == 1
    assert experiences[0]["decision"] == "accepted"


def test_policy_rejected_candidate_is_removed_only_from_run_strategies(
    tmp_path: Path,
) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(strategy="tank", commander_model="model"),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    candidate = runner.strategies_dir / "tank_opt1"
    candidate.mkdir()
    (candidate / "strategy.md").write_text("candidate", encoding="utf-8")

    assert runner._discard_policy_rejected_candidate(candidate)
    assert not candidate.exists()

    outside = tmp_path / "tank_opt2"
    outside.mkdir()
    assert not runner._discard_policy_rejected_candidate(outside)
    assert outside.exists()


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


def test_candidate_generation_failure_retries_within_the_same_run(tmp_path: Path) -> None:
    attempts = 0

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 10, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        nonlocal attempts
        attempts += 1
        if attempts > 1:
            candidate = tmp_path / "skills" / "terran" / "tank_opt1"
            candidate.mkdir(parents=True, exist_ok=True)
            return EvolRunResult(ok=True, message="OK", output_dir=candidate)
        checkpoint = tmp_path / "analysis_checkpoint"
        checkpoint.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(
            ok=False,
            message="candidate contract rejected",
            checkpoint_dir=checkpoint,
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

    assert state["status"] == "completed"
    assert attempts == 2
    assert state["candidate_resume_dir"] is None
    assert state["candidate_generation_failures"][0]["message"] == (
        "candidate contract rejected"
    )
    assert state["candidate_generation_failures"][0]["attempt"] == 1
    assert state["candidate_generation_failures"][0]["max_attempts"] == 4
    assert all(
        item.get("kind") != "candidate_generation_failure"
        for item in state.get("experiment_history") or []
    )

    assert state["champion"] == "tank_opt1"


def test_runtime_action_changes_search_direction_until_budget_completes(
    tmp_path: Path,
) -> None:
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

    assert attempts == 6
    assert state["status"] == "completed"
    assert state["completion_reason"] == "generation_budget_reached"
    assert state["last_agent_decision"]["action"] == "inspect_runtime"
    assert state.get("candidate_generation_failures") == []
    assert state.get("experiment_history") == []
    assert len(state["skipped_generations"]) == 2


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


def test_second_consecutive_inconclusive_preserves_single_champion_parent(
    tmp_path: Path,
) -> None:
    generation_calls = 0
    mutation_parents: list[str] = []

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5, tmp_path)

    def evolve(
        champion: str, batch: BatchResult, experiences: list[object]
    ) -> EvolRunResult:
        nonlocal generation_calls
        generation_calls += 1
        mutation_parents.append(champion)
        candidate = (
            tmp_path / "skills" / "terran" / f"tank_opt{generation_calls}"
        )
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=2,
            confirmation_matches=0,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert state["games_used"] == 30
    assert state["champion"] == "tank"
    assert state["search_parent"] == "tank"
    assert state["inconclusive_streak"] == 2
    assert mutation_parents == ["tank", "tank"]
    assert [item["decision"] for item in state["experiment_history"]] == [
        "inconclusive",
        "inconclusive",
    ]
    forced = state["experiment_history"][1]
    assert forced["base_decision"] == "inconclusive"
    assert forced["forced_promotion_after_inconclusive"] is False
    assert forced["forced_mechanism_change_after_inconclusive"] is True
    decision = json.loads(
        (tmp_path / "run" / "generation_001" / "decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["selection_rule"] == "candidate_score_strictly_greater"
    assert decision["mutation_parent"] == "tank"
    assert decision["comparison_champion"] == "tank"


def test_rejection_returns_search_parent_to_champion_and_resets_streak(
    tmp_path: Path,
) -> None:
    generation_calls = 0
    mutation_parents: list[str] = []
    wins = {"tank": 5, "tank_opt1": 5, "tank_opt2": 2}

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, wins[strategy], tmp_path)

    def evolve(
        champion: str, batch: BatchResult, experiences: list[object]
    ) -> EvolRunResult:
        nonlocal generation_calls
        generation_calls += 1
        mutation_parents.append(champion)
        candidate = (
            tmp_path / "skills" / "terran" / f"tank_opt{generation_calls}"
        )
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=2,
            confirmation_matches=0,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert mutation_parents == ["tank", "tank"]
    assert state["champion"] == "tank"
    assert state["search_parent"] == "tank"
    assert state["inconclusive_streak"] == 0
    assert [item["decision"] for item in state["experiment_history"]] == [
        "inconclusive",
        "rejected",
    ]


def test_underpowered_retry_rebuilds_from_champion_without_failed_edit_inheritance(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "skills" / "terran" / "tank"
    parent.mkdir(parents=True)
    (parent / "strategy.md").write_text("parent", encoding="utf-8")
    mutation_parents: list[str] = []

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 4, tmp_path)

    def evolve(
        champion: str, batch: BatchResult, experiences: list[object]
    ) -> EvolRunResult:
        mutation_parents.append(champion)
        candidate_name = f"tank_opt{len(mutation_parents)}"
        candidate = tmp_path / "skills" / "terran" / candidate_name
        candidate.mkdir(parents=True, exist_ok=True)
        (candidate / "strategy.md").write_text(
            f"candidate inherited from {champion}", encoding="utf-8"
        )
        improvement = EvolImprovement(
            analysis={
                "mechanism_family": "mobile_army_readiness",
                "hypothesis": "a stronger implementation improves the first fight",
                "expected_effect": "more combat power survives first contact",
                "main_risk": "the timing may become slower",
            },
            files={"strategy.md": f"candidate inherited from {champion}"},
        )
        return EvolRunResult(
            ok=True,
            message="OK",
            output_dir=candidate,
            improvement=improvement,
        )

    def audit(**_kwargs: object) -> dict[str, object]:
        return {
            "implementation_verdict": "underpowered",
            "hypothesis_verdict": "not_tested",
            "mechanism_evidence": [],
            "combat_evidence": [],
            "runtime_findings": [],
            "evidence_limits": [],
            "lesson": "The mechanism is executable but the implementation is too weak.",
        }

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=2,
            confirmation_matches=0,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
        experiment_auditor=audit,
    ).run()

    assert mutation_parents == ["tank", "tank"]
    assert state["champion"] == "tank"
    assert state["search_parent"] == "tank"
    first, second = state["experiment_history"]
    assert first["repairable_underpowered_retry"] is True
    assert first["search_parent_after"] == "tank"
    assert second["repairable_underpowered_retry"] is False
    assert second["underpowered_retry_exhausted"] is True
    assert second["search_parent_after"] == "tank"


def test_higher_score_is_accepted_without_a_posterior_gate(
    tmp_path: Path,
) -> None:
    wins = {"tank": 5, "tank_opt1": 6}

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, wins[strategy], tmp_path)

    def evolve(
        champion: str, batch: BatchResult, experiences: list[object]
    ) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=1,
            confirmation_matches=0,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    decision = state["experiment_history"][0]
    assert decision["decision"] == "accepted"
    assert decision["candidate_score"] == 0.6
    assert decision["comparison_champion_score"] == 0.5
    assert decision["posterior_probability_better"] < 0.8
    assert decision["evaluation"]["decision"] == "accepted"


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


def test_generation_failures_are_bounded_and_become_search_feedback(
    tmp_path: Path,
) -> None:
    seen: list[list[object]] = []

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 5 if strategy == "tank" else 10, tmp_path)

    def evolve(champion: str, batch: BatchResult, experiences: list[object]) -> EvolRunResult:
        seen.append(list(experiences))
        return EvolRunResult(ok=False, message="optimizer json invalid")

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
    assert len(seen) == 8
    assert all(experiences == [] for experiences in seen[:4])
    assert any(
        isinstance(item, dict) and item.get("kind") == "generation_search_restart"
        for item in seen[4]
    )
    assert state["candidate_generation_failures"][0]["message"] == "optimizer json invalid"
    assert [item["attempt"] for item in state["candidate_generation_failures"]] == [
        1, 2, 3, 4, 1, 2, 3, 4
    ]
    assert state["experiment_history"] == []
    assert len(state["skipped_generations"]) == 2


def test_legacy_attention_status_resumes_under_continuous_search_protocol(
    tmp_path: Path,
) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="marine",
            commander_model="model",
            max_total_generations=10,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    state = runner._new_state()
    state["status"] = "mechanism_policy_attention_required"
    state["generation"] = 3

    changed = runner._migrate_lifecycle_state(state)

    assert changed
    assert state["status"] == "running"
    assert state["generation_search_restarts"] == []


def test_resume_allows_run_control_budget_changes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    original = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            concurrency=3,
            max_total_generations=10,
        ),
        run_dir=run_dir,
        project_root=tmp_path,
    )
    original._save_state(original._new_state())

    resumed = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            concurrency=2,
            max_total_generations=12,
        ),
        run_dir=run_dir,
        project_root=tmp_path,
    ).load_or_create_state()

    assert resumed["config"]["concurrency"] == 2
    assert resumed["config"]["max_total_generations"] == 12


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
    batch_name = runner._batch_name(0, "cand", "harder")
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


def test_run_batch_uses_configured_records_directory(
    tmp_path: Path, monkeypatch
) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            records_dir="game_records_evol/tank",
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    batch_name = runner._batch_name(0, "cand", "harder")
    batch_dir = tmp_path / "game_records_evol" / "tank" / batch_name
    captured: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        captured["command"] = [str(item) for item in command]
        _write_completed_matches(batch_dir, strategy="tank_opt1", count=10)

    monkeypatch.setattr("evolution.runner.subprocess.run", fake_run)
    result = runner.run_batch(
        "tank_opt1",
        "harder",
        generation=0,
        role="cand",
        target_games=10,
    )

    command = captured["command"]
    output_flag = "-OUTPUT_BASE_DIR" if "-OUTPUT_BASE_DIR" in command else "--output-base-dir"
    assert Path(command[command.index(output_flag) + 1]) == batch_dir.parent
    assert result.path == batch_dir


def _write_completed_matches(
    batch_dir: Path, *, strategy: str, count: int, result: str = "Victory"
) -> None:
    for index in range(count):
        match = batch_dir / f"match_{index:03d}"
        match.mkdir(parents=True)
        (match / "final.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "strategy_id": strategy,
                        "save_reason": "match_runner_finally",
                        "result": result,
                    }
                }
            ),
            encoding="utf-8",
        )


def test_champion_baseline_does_not_reuse_batch_across_difficulties(
    tmp_path: Path, monkeypatch
) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(
            strategy="marine",
            commander_model="model",
            difficulties=("hard", "harder"),
            matches_per_batch=10,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    hard_dir = tmp_path / "game_records" / runner._batch_name(0, "champ", "hard")
    _write_completed_matches(hard_dir, strategy="marine", count=10)
    launched: list[str] = []

    def fake_run(command, **kwargs):
        launched.append(" ".join(str(item) for item in command))
        harder_dir = tmp_path / "game_records" / runner._batch_name(0, "champ", "harder")
        _write_completed_matches(harder_dir, strategy="marine", count=10)

    monkeypatch.setattr("evolution.runner.subprocess.run", fake_run)
    hard = runner.run_batch(
        "marine", "hard", generation=0, role="champ", target_games=10
    )
    harder = runner.run_batch(
        "marine", "harder", generation=0, role="champ", target_games=10
    )

    assert hard.path != harder.path
    assert hard.difficulty == "hard"
    assert harder.difficulty == "harder"
    assert launched, "harder must launch a new batch instead of reusing hard's records"
    assert runner._batch_name(0, "champ", "hard") != runner._batch_name(
        0, "champ", "harder"
    )


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


def test_missing_pending_candidate_is_cleared_for_safe_resume(tmp_path: Path) -> None:
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
    missing_candidate = tmp_path / "run" / "strategies" / "tank_opt1"
    checkpoint = tmp_path / "logs" / "stale_analysis"
    state["pending_candidate"] = {
        "strategy": "tank_opt1",
        "strategy_dir": str(missing_candidate),
        "analysis_checkpoint_dir": str(checkpoint),
    }
    state["candidate_resume_dir"] = str(checkpoint)
    state["status"] = "running"
    runner._save_state(state)

    loaded = runner.load_or_create_state()

    assert loaded["pending_candidate"] is None
    assert loaded["candidate_resume_dir"] is None
    assert loaded["status"] == "running"
    assert str(checkpoint.resolve()) in loaded["abandoned_analysis_checkpoints"]


def test_nine_wins_one_loss_masters_difficulty(tmp_path: Path) -> None:
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

    assert evolve_calls == 0
    assert state["status"] == "completed"
    assert state["champion"] == "tank"
    assert state["difficulty_index"] == 2
    assert state["mastered_difficulties"] == ["harder", "veryhard"]
    assert state["champion_baseline"] is None
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
        raise AssertionError("90% win-rate champion must master without a candidate")

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

    assert state["status"] == "completed"
    assert state["completion_reason"] == "generation_budget_reached"
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


def test_request_more_matches_continues_after_analysis_cap(tmp_path: Path) -> None:
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

    assert state["status"] == "completed"
    assert state["completion_reason"] == "generation_budget_reached"
    assert requested == [("tank", 10), ("tank", 10)]
    assert state.get("experiment_history") == []
    assert state["difficulty_generation"] == 5
    assert state["generation"] == 5
    assert len(state["skipped_generations"]) == 5


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


def test_full_budget_mode_continues_after_final_difficulty_mastery(
    tmp_path: Path,
) -> None:
    evolve_calls = 0

    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 10, tmp_path)

    def evolve(
        champion: str,
        batch: BatchResult,
        experiences: list[object],
    ) -> EvolRunResult:
        nonlocal evolve_calls
        evolve_calls += 1
        return EvolRunResult(
            ok=True,
            message="no safe patch in this search",
            decision_action="stop",
            action_reason="try another mechanism",
        )

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder",),
            max_total_generations=2,
            require_full_generation_budget=True,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert state["status"] == "completed"
    assert state["completion_reason"] == "generation_budget_reached"
    assert state["generation"] == 2
    assert state["mastered_difficulties"] == ["harder"]
    assert state["curriculum_completed"] is True
    assert evolve_calls == 6


def test_full_budget_mode_advances_after_local_budget_without_stopping(
    tmp_path: Path,
) -> None:
    def play(strategy: str, difficulty: str) -> BatchResult:
        wins = 5 if strategy == "tank" else 2
        return _batch(strategy, difficulty, wins, tmp_path)

    def evolve(
        champion: str,
        batch: BatchResult,
        experiences: list[object],
    ) -> EvolRunResult:
        candidate = tmp_path / "skills" / "terran" / "tank_opt1"
        candidate.mkdir(parents=True, exist_ok=True)
        return EvolRunResult(ok=True, message="OK", output_dir=candidate)

    state = EvolutionRunner(
        EvolutionConfig(
            strategy="tank",
            commander_model="model",
            difficulties=("harder", "veryhard"),
            max_generations_per_difficulty=1,
            max_total_generations=2,
            require_full_generation_budget=True,
        ),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
        batch_executor=play,
        candidate_generator=evolve,
    ).run()

    assert state["status"] == "completed"
    assert state["completion_reason"] == "generation_budget_reached"
    assert state["generation"] == 2
    assert state["difficulty_index"] == 1
    assert state["exhausted_difficulties"] == ["harder"]
    assert state["difficulty_budget_events"][0]["action"] == "advance_without_mastery"


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


def test_eight_vs_nine_is_accepted_and_masters(tmp_path: Path) -> None:
    def play(strategy: str, difficulty: str) -> BatchResult:
        return _batch(strategy, difficulty, 8 if strategy == "tank" else 9, tmp_path)

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
    assert record["hypothesis_verdict"] == "inconclusive"
    assert round(record["score_delta"], 4) == 0.1
    assert record["evaluation"]["champion"]["score"] == 0.8
    assert record["evaluation"]["candidate"]["score"] == 0.9
    assert "posterior" in record["evaluation"]
    assert state["selection_protocol"] == "score_only_v2"
    assert state["status"] == "completed"
    assert state["champion"] == "tank_opt1"
    assert state["mastered_difficulties"] == ["harder"]


def test_rejected_underpowered_attempts_exhaust_mechanism_retry(tmp_path: Path) -> None:
    runner = EvolutionRunner(
        EvolutionConfig(strategy="marine", commander_model="model"),
        run_dir=tmp_path / "run",
        project_root=tmp_path,
    )
    state = runner._new_state()
    state["experiment_history"] = [
        {
            "difficulty": "harder",
            "mechanism_family": "support_unit_break_siege",
            "decision": "rejected",
            "implementation_verdict": "underpowered",
            "hypothesis_verdict": "inconclusive",
        },
        {
            "difficulty": "harder",
            "mechanism_family": "support_unit_break_siege",
            "decision": "rejected",
            "implementation_verdict": "underpowered",
            "hypothesis_verdict": "not_tested",
            "underpowered_retry_exhausted": True,
        },
    ]

    blocked = runner._blocked_mechanism_families(state, difficulty="harder")

    assert blocked["support_unit_break_siege"] == (
        "underpowered retry budget was exhausted"
    )
