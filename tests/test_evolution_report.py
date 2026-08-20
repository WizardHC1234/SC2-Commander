from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from tools.evolution_report import (
    load_evolution_history,
    main,
    summarize_consistency,
    summarize_resources,
    wilson_interval,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _history_rows() -> list[dict[str, object]]:
    return [
        {
            "strategy_style": "tank",
            "generation": 0,
            "strategy": "tank",
            "parent": "",
            "difficulty": "harder",
            "wins": 5,
            "draws": 0,
            "losses": 5,
            "games": 10,
            "score": 0.5,
            "win_rate": 0.5,
            "mastered_levels": 0,
            "curriculum_progress_score": 0.5556,
            "accepted": "true",
            "batch": "ev_run_g000_champ",
        },
        {
            "strategy_style": "tank",
            "generation": 1,
            "strategy": "tank_opt1",
            "parent": "tank",
            "difficulty": "harder",
            "wins": 9,
            "draws": 0,
            "losses": 1,
            "games": 10,
            "score": 0.9,
            "win_rate": 0.9,
            "mastered_levels": 0,
            "curriculum_progress_score": 1.0,
            "accepted": "true",
            "batch": "ev_run_g000_cand",
        },
    ]


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(9, 10)

    assert low is not None and high is not None
    assert low < 0.9 < high
    assert wilson_interval(0, 0) == (None, None)


def test_history_keeps_win_rate_and_budget_separate(tmp_path: Path) -> None:
    history_path = tmp_path / "history.csv"
    _write_csv(history_path, _history_rows())

    rows = load_evolution_history(history_path)

    assert [row["win_rate"] for row in rows] == [0.5, 0.9]
    assert [row["cumulative_evaluation_games"] for row in rows] == [10, 20]
    assert rows[1]["curriculum_progress_score"] == 1.0
    assert rows[1]["accepted"] is True


def test_consistency_is_not_folded_into_outcome_score() -> None:
    rows = [
        {
            "strategy": "tank",
            "difficulty": "harder",
            "result": "Victory",
            "economy_completion": "1.0",
            "technology_completion": "0.8",
            "army_completion": "0.9",
            "engagement_trigger_consistency": "0.7",
            "engagement_continuation_consistency": "0.6",
            "overall_strategy_compliance": "0.8",
        },
        {
            "strategy": "tank",
            "difficulty": "harder",
            "result": "Defeat",
            "economy_completion": "0.8",
            "technology_completion": "0.6",
            "army_completion": "0.7",
            "engagement_trigger_consistency": "0.5",
            "engagement_continuation_consistency": "",
            "overall_strategy_compliance": "0.65",
        },
    ]

    metrics, relation, outcomes = summarize_consistency(rows)

    assert len(metrics) == 6
    assert relation[0]["games"] == 2
    assert relation[0]["win_rate"] == 0.5
    assert math.isclose(relation[0]["overall_strategy_compliance"], 0.725)
    assert {row["result"] for row in outcomes} == {"victory", "defeat"}


def test_resource_usage_aggregates_only_available_fields() -> None:
    rows = [
        {
            "strategy": "tank_opt1",
            "difficulty": "harder",
            "duration_s": "600",
            "llm_interaction_count": "12",
            "record_count": "10",
            "commander_decision_count": "9",
        },
        {
            "strategy": "tank_opt1",
            "difficulty": "harder",
            "duration_s": "900",
            "llm_interaction_count": "18",
            "record_count": "15",
            "commander_decision_count": "14",
        },
    ]

    summary = summarize_resources(rows)[0]

    assert summary["games"] == 2
    assert summary["total_game_duration_s"] == 1500
    assert summary["total_llm_interactions"] == 30
    assert summary["mean_records_per_game"] == 12.5
    assert summary["total_commander_decisions"] == 23


def test_no_plot_report_writes_tidy_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "evolution_run"
    run_dir.mkdir()
    _write_csv(run_dir / "history.csv", _history_rows())
    decision_dir = run_dir / "generation_000"
    decision_dir.mkdir()
    (decision_dir / "decision.json").write_text(
        json.dumps(
            {
                "generation": 0,
                "difficulty": "harder",
                "parent": "tank",
                "candidate": "tank_opt1",
                "parent_score": 0.5,
                "candidate_score": 0.9,
                "score_delta": 0.4,
                "decision": "accepted",
                "accepted": True,
                "candidate_evidence_games": 10,
                "champion_evidence_games": 10,
            }
        ),
        encoding="utf-8",
    )
    consistency_path = tmp_path / "per_game.csv"
    _write_csv(
        consistency_path,
        [
            {
                "strategy": "tank_opt1",
                "difficulty": "harder",
                "result": "Victory",
                "economy_completion": 1.0,
                "technology_completion": 0.9,
                "army_completion": 0.9,
                "engagement_trigger_consistency": 0.8,
                "engagement_continuation_consistency": 0.8,
                "overall_strategy_compliance": 0.88,
            }
        ],
    )
    out_dir = tmp_path / "report"

    exit_code = main(
        [
            "--run-dir",
            str(run_dir),
            "--consistency-csv",
            str(consistency_path),
            "--out-dir",
            str(out_dir),
            "--no-plots",
        ]
    )

    assert exit_code == 0
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["total_evaluation_games"] == 20
    assert report["accepted_candidates"] == 1
    assert (out_dir / "data" / "evolution_history.csv").is_file()
    assert (out_dir / "data" / "candidate_decisions.csv").is_file()
    assert (out_dir / "data" / "strategy_consistency.csv").is_file()
    assert (out_dir / "data" / "resource_usage.csv").is_file()
