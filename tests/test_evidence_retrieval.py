from __future__ import annotations

from types import SimpleNamespace

from evol_agent.core.evidence_retrieval import (
    parse_evidence_reference,
    query_experiment_history,
    query_match_records,
)
from evol_agent.core.llm import normalize_prompt_layout
from evol_agent.sc2_data_agent.strategy_knowledge import build_strategy_knowledge
from evol_agent.sc2_data_agent.bridge import (
    find_knowledge_run_error,
    run_knowledge_query,
)


def test_parse_evidence_reference_supports_ranges() -> None:
    parsed = parse_evidence_reference("Game 3 @ 592-712s: army collapses")
    assert parsed == {
        "game_index": 3,
        "start_s": 592.0,
        "end_s": 712.0,
        "reference": "Game 3 @ 592-712s: army collapses",
    }


def test_prompt_layout_removes_only_soft_wrapped_prose() -> None:
    prompt = """4. Opponent pressure patterns
Compare when and how opponents pressure owned bases, economy, production, or the
defending army. Record the observable cues, the own defensive package available at
contact, whether the strategy survived until its intended power spike.

- Keep exact evidence.
- Preserve list structure.

Return JSON:
{
  "ok": true
}"""
    normalized = normalize_prompt_layout(prompt)
    assert "or the\ndefending army" not in normalized
    assert "or the defending army" in normalized
    assert "4. Opponent pressure patterns\nCompare" in normalized
    assert "- Keep exact evidence.\n- Preserve list structure." in normalized
    assert 'Return JSON:\n{\n  "ok": true\n}' in normalized


def test_match_record_query_classifies_enemy_pressure_from_recorded_rows(
    monkeypatch,
) -> None:
    timeline = "\n".join(
        [
            'SCHEMA {"columns":["chunk","time_s","trigger","phase","production","technology","army","enemy","opponent_truth_after_match","combat","threat","macro_targets","macro_progress_before_decision","groups","zones","orders","accepted_issues","fallback_state"]}',
            'R [1,500,"wake","",[],[],[],[],[],[],[],[],[],[],[["zone_0","own","own_main",false,0,0,0,0]],[],[],{}]',
            'R [2,540,"under_attack","",[],[],[],[],[],[],[],[],[],[],[["zone_0","own","own_main",true,12,8,6,6]],[],[],{}]',
        ]
    )
    monkeypatch.setattr(
        "evol_agent.core.evidence_retrieval.MatchRecordReader.fixed_timeline",
        lambda self: timeline,
    )
    packet = query_match_records(
        [SimpleNamespace(file="match.json")],
        [
            {
                "id": "M1",
                "query_reason": "verify the engagement attribution",
                "evidence_refs": ["Game 1 @ 521s"],
            }
        ],
    )
    result = packet["queries"][0]["results"][0]
    assert result["interaction_check"]["classification"] == "enemy_pressure"
    assert result["interaction_check"]["enemy_in_owned_zone_observed"] is True
    assert result["interaction_check"]["own_assault_command_observed"] is False


def test_history_query_returns_failed_interventions_as_negative_evidence() -> None:
    packet = query_experiment_history(
        [
            {
                "experiment_id": "e1",
                "hypothesis": "improve first engagement survival under early pressure",
                "decision": "rejected",
                "hypothesis_verdict": "inconclusive",
                "lesson": "the concrete production-only package did not improve score",
            },
            {
                "experiment_id": "e2",
                "hypothesis": "late economic recovery",
                "decision": "accepted",
            },
        ],
        {
            "query_reason": "find interventions for first engagement survival",
            "failure_signature": ["early pressure", "army broken at first engagement"],
        },
    )
    assert packet["results"][0]["experience"]["experiment_id"] == "e1"
    assert packet["results"][0]["experience"]["decision"] == "rejected"


def test_history_query_compacts_strategy_text_and_timing_snapshots() -> None:
    packet = query_experiment_history(
        [
            {
                "experiment_id": "tank:g005:harder:tank_opt6",
                "hypothesis": "add Marauders to break heavy armor",
                "decision": "rejected",
                "score_delta": -0.2,
                "patches": [{"target": "production", "replacement": "very long strategy text"}],
                "first_commitment_timing": {
                    "earliest_feasible_timing_delta_seconds": 34.0,
                    "empirical_opponent_windows": [{"large": "trajectory payload"}],
                },
            }
        ],
        {
            "query_reason": "heavy armor engagement",
            "failure_signature": ["Marauders did not improve outcomes"],
        },
    )
    experience = packet["results"][0]["experience"]
    assert "patches" not in experience
    assert experience["first_commitment_timing"] == {
        "earliest_feasible_timing_delta_seconds": 34.0
    }


def test_parallel_production_and_resource_demand_are_deterministic() -> None:
    packet = build_strategy_knowledge(
        {
            "question": "Calculate production throughput and gas demand.",
            "entities": ["Siege Tank"],
            "needs": ["requirements"],
            "calculations": [
                {
                    "type": "parallel_production",
                    "action": "train_siege_tank",
                    "quantity": 10,
                    "production_slots": 2,
                },
                {
                    "type": "parallel_production",
                    "action": "train_siege_tank",
                    "quantity": 10,
                    "production_slots": 3,
                },
                {
                    "type": "resource_demand_per_minute",
                    "action": "train_siege_tank",
                    "production_slots": 3,
                },
            ],
        },
        race="terran",
    )
    two_slots, three_slots, demand = packet["calculations"]
    assert two_slots["cycles"] == 5
    assert three_slots["cycles"] == 4
    assert two_slots["minimum_continuous_time_seconds"] < 200
    assert three_slots["minimum_continuous_time_seconds"] < two_slots[
        "minimum_continuous_time_seconds"
    ]
    assert demand["demand_per_minute"]["gas"] > 690


def test_calculation_action_resolves_without_duplicate_entity_request() -> None:
    run = run_knowledge_query(
        {
            "id": "Q1",
            "question": "Calculate the requested production throughput.",
            "entities": [],
            "needs": ["requirements"],
            "calculations": [
                {
                    "type": "parallel_production",
                    "action": "train_siege_tank",
                    "quantity": 10,
                    "production_slots": 2,
                }
            ],
        },
        race="terran",
    )

    assert run["ok"] is True
    assert find_knowledge_run_error(run) == ""
    packet = run["dataset_evidence"][0]["result"]
    assert packet["requested_calculation_count"] == 1
    assert packet["calculations"][0]["minimum_continuous_time_seconds"] == 160.5


def test_failed_requested_calculation_is_not_verified() -> None:
    run = run_knowledge_query(
        {
            "id": "Q1",
            "question": "Calculate an unsupported action.",
            "entities": [],
            "calculations": [
                {
                    "type": "parallel_production",
                    "action": "train_nonexistent_unit",
                    "quantity": 10,
                    "production_slots": 2,
                }
            ],
        },
        race="terran",
    )

    assert "knowledge calculation failed" in find_knowledge_run_error(run)
