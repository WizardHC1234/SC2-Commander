import json
import shutil
import uuid
from pathlib import Path

from evol_agent.analysis.get_chunk import extract_chunks
from evol_agent.analysis.record_reader import (
    IncompleteMatchRecordError,
    build_record_evidence_baseline,
    extract_strategy_from_record,
    find_record_jsons,
    group_records_by_strategy,
    is_completed_match_record,
)
from evol_agent.core import config
from evol_agent.core.analysis_agent_loop import _normalize_batch_analysis
from evol_agent.core.checkpoint import EvolCheckpoint
from evol_agent.core.candidate_critic import critique_candidate_contract
from evol_agent.core.capabilities import build_executor_capability_manifest
from evol_agent.core.loop_helpers import normalize_strategy_contract
from evol_agent.core.optimization_agent_loop import run_optimization_agent_loop
from evol_agent.core.prompts import (
    build_batch_analysis_prompt,
    build_candidate_prompt,
)
from evol_agent.core.types import BattleAnalysis
from evol_agent.optimization.snapshot import save_snapshot
from evol_agent.optimization.strategy_document import (
    StrategyDocument,
    paragraph_hash,
)
from evol_agent.sc2_data_agent.bridge import (
    KNOWLEDGE_VERIFICATION_SCHEMA,
    run_knowledge_query,
)
from evol_agent.sc2_data_agent.strategy_knowledge import (
    infer_knowledge_needs,
    resolve_knowledge_entities,
)
from evol_agent.sc2_data_agent.sc2_data_store import get_dataset_store
from evol_agent.analysis.match_record import MatchRecordReader
from evol_agent.analysis.replay_truth import commander_game_loops, enemy_truth_path
from evol_agent.validation import (
    validate_improvement,
    validate_strategy_markdown,
)


VALID_STRATEGY = """# Summary
A compact Terran plan built around a gathered Marine and Tank force.

# Details
* Opening: Build workers, supply, production, gas, and technology toward fixed absolute targets.
* Main Attack Gate: Gather the persistent main force before attacking and send reinforcements toward the same objective.
"""


def _current_record() -> dict:
    observation = {
        "schema_version": "2.0",
        "economy": {
            "minerals": 250,
            "vespene": 100,
            "mineral_income": 720.0,
            "vespene_income": 240.0,
            "supply_used": 42,
            "supply_cap": 54,
            "supply_free": 12,
            "workers": 24,
            "ideal_workers": 32,
            "own_base_count": 2,
        },
        "own_forces": {
            "army_supply": 18,
            "combat_composition": {"MARINE": 12, "SIEGETANK": 2},
            "training_combat_composition": {"MARINE": 2},
        },
        "production": {
            "completed": {"COMMANDCENTER": 2, "BARRACKS": 3, "FACTORY": 1},
            "under_construction": {},
            "active_queues": {"Training Marine": 2},
        },
        "technology": {"completed_upgrades": ["STIMPACK"]},
        "enemy": {
            "visible_composition": {"MARINE": 8},
            "known_combat_composition": {"MARINE": 8},
            "seconds_since_last_seen": 4.0,
        },
        "map_control": {"own_base_count": 2, "known_enemy_base_count": 2},
        "combat": {"advantage_predicted": "Even", "our_army_power": 20, "enemy_army_power": 18},
        "threat_flags": {},
        "execution": {
            "macro": {
                "status": "active",
                "last_tasks": ["train_marine to 30"],
                "active_macro_tasks": [{"action": "train_marine", "to_count": 30}],
                "last_issues": ["waiting_for_factory"],
            }
        },
        "army_control": {
            "groups": [
                {
                    "group_id": "group_0",
                    "role": "main_force",
                    "unit_count": 14,
                    "power": 20,
                    "nearest_zone_id": "zone_2",
                    "unit_type_counts": {"MARINE": 12, "SIEGETANK": 2},
                    "is_fragmented": False,
                },
                {
                    "group_id": "group_1",
                    "role": "reinforcement",
                    "unit_count": 2,
                    "power": 2,
                    "nearest_zone_id": "zone_0",
                    "unit_type_counts": {"MARINE": 2},
                    "is_fragmented": False,
                },
            ],
            "zones": [{"zone_id": "zone_2", "owner": "own"}],
        },
    }
    return {
        "metadata": {
            "strategy_id": "tank",
            "strategy_hash": "test-hash",
            "commander_model_key": "test-model",
            "save_reason": "on_end",
            "result": "Victory",
            "matchup": "TvT",
            "my_race": "Terran",
        },
        "interactions": [
            {
                "game_time": 0.0,
                "trigger_reason": "strategy_forced",
                "forced_strategy": "tank",
                "strategy_id": "tank",
            },
            {
                "game_time": 0.0,
                "trigger_reason": "strategy_tool_selection",
                "strategy_id": "tank",
                "selected_tools": ["train_marine", "train_siege_tank", "build_gas"],
                "semantic_tools": ["train_marine", "train_siege_tank"],
                "dependency_tools": ["build_gas"],
                "baseline_tools": ["set_wake_event"],
                "selected_tool_count": 3,
                "full_tool_count": 74,
                "fallback_used": False,
            },
            {
                "agent": "commander",
                "game_time": 300.0,
                "trigger_reason": "wake_event",
                "strategy_id": "tank",
                "observation": observation,
                "text_observation": "[Game]\nrecorded_exact_observation=yes",
                "assistant_content": "Keep the full macro plan active and gather reinforcements.",
                "macro_tasks": [{"action": "train_marine", "to_count": 30}],
                "army_policy": {
                    "commands": [
                        {"group_id": "group_0", "movement_mode": "hold", "destination_zone_id": "zone_2"},
                        {"group_id": "group_1", "movement_mode": "regroup", "destination_zone_id": "zone_2"},
                    ],
                    "scan_zone_id": None,
                    "scout_zone_id": None,
                },
                "wake_event": {"logic": "any", "conditions": [{"type": "game_time_at_least", "seconds": 330}]},
                "accepted": True,
            },
        ],
    }


def test_current_commander_record_is_extracted() -> None:
    record = _current_record()
    assert extract_strategy_from_record(record) == "tank"

    extracted = extract_chunks(record)
    assert [chunk["agent_role"] for chunk in extracted["chunks"]] == ["init", "selector", "commander"]
    commander = extracted["chunks"][-1]
    assert commander["trigger"] == "wake_event"
    assert "recorded_exact_observation=yes" in commander["text"]
    assert "[Commander Macro Targets] train_marine->30" in commander["text"]
    assert "group_1:regroup->zone_2" in commander["text"]
    assert commander["decision"]["macro_tasks"] == [
        {"action": "train_marine", "to_count": 30}
    ]
    selector = extracted["chunks"][1]
    assert selector["decision"]["tool_selection"]["dependency_tools"] == ["build_gas"]
    assert "dependencies=build_gas" in selector["text"]

    record_reader = MatchRecordReader(Path("synthetic-record.json"))
    record_reader._data = json.loads(json.dumps(record))
    manifest = record_reader.manifest("synthetic")
    assert manifest["chunk_count"] == 3
    exact_selection = manifest["action_space_selection"]
    assert exact_selection["semantic_tools"] == ["train_marine", "train_siege_tank"]
    assert exact_selection["dependency_tools"] == ["build_gas"]
    assert exact_selection["baseline_tools"] == ["set_wake_event"]
    timeline = record_reader.fixed_timeline()
    assert sum(line.startswith("R ") for line in timeline.splitlines()) == 1
    assert "fixed arrays" in timeline
    assert '"dependency_tools":["build_gas"]' in timeline
    assert "group_1" in timeline
    assert "train_marine" in timeline
    assert "recorded_exact_observation=yes" not in timeline
    assert "previous_decision duplicate" in timeline


def test_post_match_enemy_truth_is_joined_by_commander_game_loop(tmp_path: Path) -> None:
    record = _current_record()
    observation = record["interactions"][-1]["observation"]
    observation["snapshot_id"] = "game_loop:6720"
    observation["time"] = {
        "game_loop": 6720,
        "seconds": 300.0,
        "formatted": "05:00",
    }
    record_path = tmp_path / "match.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    truth_path = enemy_truth_path(record_path)
    truth_path.write_text(
        json.dumps(
            {
                "schema": "sc2_opponent_truth.v1",
                "source": "post_match_replay_observed_opponent",
                "snapshot_count": 1,
                "snapshots": [
                    {
                        "requested_game_loop": 6720,
                        "game_loop": 6720,
                        "resources": {"minerals": 400, "vespene": 200},
                        "supply": {"used": 60, "cap": 78, "army": 30, "workers": 30},
                        "workers": 30,
                        "army_units": {"MARINE": 20, "SIEGETANK": 3},
                        "structures_completed": {"COMMANDCENTER": 2, "FACTORY": 1},
                        "structures_in_progress": {},
                        "upgrades": ["STIMPACK"],
                        "active_orders": {"TRAIN_MARINE": 2},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert commander_game_loops(record) == [6720]
    assert find_record_jsons(tmp_path) == [record_path]
    reader = MatchRecordReader(record_path)
    assert reader.manifest("match")["opponent_truth"]["available"] is True
    timeline = reader.fixed_timeline()
    assert "opponent_truth_after_match" in timeline
    assert "post_match_replay_observed_opponent" in timeline
    assert '"SIEGETANK":3' in timeline

def test_prompts_offer_multiple_evidenced_plans_for_one_candidate() -> None:
    analyses = [
        BattleAnalysis(
            strategy_name="tank",
            race="terran",
            sample_size=1,
            record_mix="1W/0L",
            raw={"outcome_summary": "MATCH_RAW_WIN_MARKER"},
        ),
        BattleAnalysis(
            strategy_name="tank",
            race="terran",
            sample_size=1,
            record_mix="0W/1L",
            raw={"outcome_summary": "MATCH_RAW_LOSS_MARKER"},
        ),
    ]
    batch_prompt = build_batch_analysis_prompt(
        strategy_name="tank",
        race="terran",
        single_game_analyses=analyses,
        skill_texts={"strategy.md": VALID_STRATEGY},
        validation_errors=[],
        knowledge_mode="enabled",
    )
    battle_analysis = BattleAnalysis(
        strategy_name="tank",
        race="terran",
        sample_size=2,
        record_mix="1W/1L",
        raw={
            "winning_mechanism": "gathered push",
            "problems": [{"problem_id": "P1", "problem": "late regroup"}],
            "candidate_plans": [
                {
                    "id": "D1",
                    "name": "improve regroup timing",
                    "addresses_problem_ids": ["P1"],
                    "changes": [],
                }
            ],
        },
    )
    candidate_prompt = build_candidate_prompt(
        strategy_name="tank",
        race="terran",
        battle_analysis=battle_analysis,
        skill_texts={"strategy.md": VALID_STRATEGY},
        tool_observations=[],
        validation_errors=[],
        candidate=None,
        knowledge_mode="enabled",
    )

    assert "MATCH_RAW_WIN_MARKER" in batch_prompt
    assert "MATCH_RAW_LOSS_MARKER" in batch_prompt
    assert '"action": "analyze_batch"' in batch_prompt
    assert '"problems"' in batch_prompt
    assert '"candidate_plans"' in batch_prompt
    assert "zero to four focused, non-overlapping knowledge_questions" in batch_prompt
    assert "requested command is not proof" in batch_prompt.lower()
    assert "likely effect on match outcomes" in batch_prompt
    assert "strengthening the existing core through relevant upgrades" in batch_prompt
    assert "Static defenses are not a default" in batch_prompt
    assert "force-readiness curve against actual enemy pressure" in batch_prompt
    assert "commander_execution, runtime_execution, or observation_limited" in batch_prompt
    assert "Unit composition and upgrade changes are first-class plans" in batch_prompt
    assert "experiment_evidence when present" in batch_prompt
    assert "unable to survive until that power stage" in batch_prompt
    assert "full dependency chain, not from final costs alone" in batch_prompt
    assert "cannot plausibly complete before" in batch_prompt
    assert '"evidence_motivation"' in batch_prompt
    assert '"decision_use"' in batch_prompt
    assert "Stimpack" not in batch_prompt
    assert '"action": "draft_candidate"' in candidate_prompt
    assert "gathered push" in candidate_prompt
    assert "Select exactly one self-contained candidate plan" in candidate_prompt
    assert "Multiple dependent deterministic changes are allowed" in candidate_prompt
    assert "credible path through economy, production capacity" in candidate_prompt
    assert "bounded information-conditioned branch" in candidate_prompt
    assert "sole optimization objective is higher expected match win rate" in candidate_prompt
    assert "Add static defense only when" in candidate_prompt
    assert "survival path before its main power stage" in candidate_prompt
    assert "candidate's critical path" in candidate_prompt
    assert '"hypothesis"' in candidate_prompt
    assert '"capability_mapping"' in candidate_prompt


def test_match_summary_prompt_keeps_batch_stable_prefix_before_record_data() -> None:
    from evol_agent.core.prompts import build_fixed_match_summary_prompt

    shared = {
        "strategy_name": "tank",
        "race": "terran",
    }
    first = build_fixed_match_summary_prompt(
        **shared,
        record_manifest={"record_id": "match_001", "result": "Victory"},
        match_timeline="TIMELINE_ONE",
    )
    second = build_fixed_match_summary_prompt(
        **shared,
        record_manifest={"record_id": "match_002", "result": "Defeat"},
        match_timeline="TIMELINE_TWO",
    )
    first_dynamic = first.index("Match-specific metadata:")
    second_dynamic = second.index("Match-specific metadata:")
    assert first[:first_dynamic] == second[:second_dynamic]
    assert "Current strategy.md:" not in first


def test_strategy_contract_normalizes_legacy_checkpoints() -> None:
    contract = normalize_strategy_contract(
        {
            "intended_plan": "Legacy concentrated push",
            "must_preserve": ["gather before attacking"],
            "must_not_break": ["keep the core win condition"],
        },
        strategy_name="legacy",
    )

    assert contract == {
        "identity": "Legacy concentrated push",
        "core_commitments": ["gather before attacking"],
        "optimization_boundary": "keep the core win condition",
        "direction": "adjust",
    }


def test_batch_analysis_keeps_one_plan_and_only_linked_knowledge_questions() -> None:
    payload, error = _normalize_batch_analysis(
        {
            "strategy_contract": {"identity": "two-base tank timing"},
            "winning_mechanism": "a concentrated timing push",
            "wins_to_preserve": [{"pattern": "gather first", "evidence": ["Match 1"]}],
            "problems": [
                {
                    "problem": "the main force repeatedly attacks without enough preparation",
                    "evidence": ["Match 2 before the first attack"],
                    "strategy_fixable": True,
                }
            ],
            "candidate_plans": [
                {
                    "name": f"fixed correction {index}",
                    "addresses_problem_ids": ["P1"],
                    "changes": [
                        {
                            "baseline_rule": f"old fixed rule {index}",
                            "candidate_rule": f"new fixed rule {index}",
                            "why_required": "supports the complete force timing",
                        }
                    ],
                    "risk_to_winning_mechanism": "could delay the timing",
                }
                for index in range(3)
            ],
            "knowledge_questions": [
                {
                    "question": f"What effects distinguish unit {index}?",
                    "evidence_motivation": f"Match {index + 1} exposed a missing fact",
                    "decision_use": f"select or size plan {(index % 3) + 1}",
                    "entities": [
                        "Marine",
                        "Siege Tank",
                        "Barracks",
                        "Factory",
                        "Combat Shield",
                    ][index:index + 1],
                    "needs": ["effects"],
                    "plan_ids": [f"D{(index % 3) + 1}"],
                }
                for index in range(5)
            ],
        },
        strategy_name="tank",
        knowledge_mode="enabled",
    )

    assert error == ""
    assert payload is not None
    assert payload["primary_problem"]["problem_id"] == "P1"
    assert len(payload["candidate_plans"]) == 1
    assert len(payload["optimization_targets"]) == 1
    assert len(payload["knowledge_questions"]) == 2
    assert all(question["problem_ids"] == ["P1"] for question in payload["knowledge_questions"])
    assert all(question["plan_ids"] for question in payload["knowledge_questions"])


def test_candidate_semantics_are_not_hard_coded_in_basic_validation() -> None:
    fixed_candidate = VALID_STRATEGY.replace(
        "Gather the persistent main force before attacking",
        "Gather at least 40 Marines and 10 Siege Tanks before attacking",
    )
    assert validate_improvement(
        files={"strategy.md": fixed_candidate},
        race="terran",
    ).ok

    conditional_candidate = VALID_STRATEGY.replace(
        "Gather the persistent main force before attacking and send reinforcements toward the same objective.",
        "If a scan confirms weak enemy defenses, attack earlier; otherwise gather the persistent main force before attacking and send reinforcements toward the same objective.",
    )
    assert validate_improvement(
        files={"strategy.md": conditional_candidate},
        race="terran",
    ).ok

    summary_conditional = fixed_candidate.replace(
        "A compact Terran plan built around a gathered Marine and Tank force.",
        "If scouting reveals an exposed enemy base, change the attack objective.",
    )
    assert validate_improvement(
        files={"strategy.md": summary_conditional},
        race="terran",
    ).ok


def test_strategy_document_applies_only_selected_paragraphs() -> None:
    document = StrategyDocument.parse(VALID_STRATEGY)
    gate = next(item for item in document.details if item.id == "main_attack_gate")
    candidate, changes = document.apply_patch(
        [
            {
                "op": "replace_detail",
                "target": gate.id,
                "expected_old_hash": paragraph_hash(gate.value),
                "value": "Gather 40 Marines and 8 Siege Tanks before attacking.",
            }
        ]
    )

    assert changes == [{"op": "replace_detail", "target": "main_attack_gate"}]
    assert "Gather 40 Marines and 8 Siege Tanks before attacking." in candidate
    assert "Build workers, supply, production, gas" in candidate
    assert validate_strategy_markdown(candidate, race="terran") is None


def test_deterministic_match_features_expose_runtime_classification(tmp_path: Path) -> None:
    record = _current_record()
    record["interactions"][-1]["accepted"] = False
    record["interactions"][-1]["issues"] = ["army command was rejected"]
    record_path = tmp_path / "match.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    features = MatchRecordReader(record_path).deterministic_features("match_001")

    assert features["summary_quality"] == "deterministic"
    assert features["decision_metrics"]["commander_rows"] == 1
    assert features["runtime_assessment"]["contaminated"] is True
    assert features["runtime_assessment"]["classification"] == "runtime_contaminated"


def test_batch_analysis_can_route_runtime_work_without_a_candidate() -> None:
    payload, error = _normalize_batch_analysis(
        {
            "next_action": "inspect_runtime",
            "action_reason": "most losses contain rejected army commands",
            "problems": [
                {
                    "problem": "Sharpy repeatedly rejects group movement",
                    "control_class": "runtime_execution",
                    "strategy_fixable": False,
                    "evidence": ["Match 1", "Match 2"],
                }
            ],
            "candidate_plans": [],
        },
        strategy_name="tank",
        knowledge_mode="enabled",
    )

    assert error == ""
    assert payload is not None
    assert payload["next_action"] == "inspect_runtime"
    assert payload["candidate_plans"] == []


def test_complete_analysis_plan_is_applied_without_second_llm_call(monkeypatch) -> None:
    manifest = build_executor_capability_manifest("terran")
    analysis = BattleAnalysis(
        strategy_name="tank",
        race="terran",
        sample_size=2,
        record_mix="1W/1L",
        raw={
            "next_action": "propose_strategy_patch",
            "action_reason": "the attack gate is repeatedly late",
            "winning_mechanism": "one gathered timing attack",
            "candidate_plans": [
                {
                    "id": "D1",
                    "name": "lower the gathered attack threshold",
                    "hypothesis": "an earlier gathered force converts before scaling",
                    "primary_lever": "attack_timing",
                    "addresses_problem_ids": ["P1"],
                    "changes": [
                        {
                            "target_paragraph_id": "main_attack_gate",
                            "candidate_rule": "Gather 40 Marines and 8 Siege Tanks before attacking.",
                            "why_required": "the old force arrives after enemy scaling",
                        }
                    ],
                    "predictions": ["the first attack command occurs earlier"],
                    "disproof_conditions": ["attack timing does not improve"],
                    "capability_mapping": {
                        "macro_actions": ["train_marine", "train_siege_tank"],
                        "army_controls": ["hold_or_gather", "push_or_assault"],
                        "unsupported_dependencies": [],
                    },
                    "expected_benefit": "an earlier first engagement",
                    "risk_to_winning_mechanism": "the force may be too small",
                }
            ],
        },
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("a complete analyzed plan must not call the optimizer LLM")

    monkeypatch.setattr(
        "evol_agent.core.optimization_agent_loop.call_json_llm", fail_if_called
    )
    result, improvement, _observations, errors, events = run_optimization_agent_loop(
        strategy_name="tank",
        race="terran",
        battle_analysis=analysis,
        skill_texts={"strategy.md": VALID_STRATEGY},
        initial_tool_observations=[],
        capability_manifest=manifest,
    )

    assert result.ok
    assert errors == []
    assert improvement is not None
    assert "Gather 40 Marines and 8 Siege Tanks" in improvement.files["strategy.md"]
    assert events[0]["llm_calls"] == 0

def test_executor_manifest_and_candidate_critic_share_action_vocabulary() -> None:
    manifest = build_executor_capability_manifest("terran")
    assert "train_siege_tank" in manifest["macro_contract"]["available_actions"]
    valid = {
        "hypothesis": "more completed Tanks improve the first engagement",
        "primary_lever": "composition",
        "predictions": ["first contact has more living Siege Tanks"],
        "disproof_conditions": ["Tank count does not improve"],
        "capability_mapping": {
            "macro_actions": ["train_siege_tank"],
            "army_controls": ["main_force_reinforcement"],
            "unsupported_dependencies": [],
        },
    }
    assert critique_candidate_contract(valid, capability_manifest=manifest) == []
    valid["capability_mapping"]["macro_actions"] = ["micro_every_tank"]
    assert "unknown macro actions" in critique_candidate_contract(
        valid, capability_manifest=manifest
    )[0]


def test_candidate_critic_compares_only_changed_actions_with_plan_delta() -> None:
    manifest = build_executor_capability_manifest("terran")
    rationale = {
        "hypothesis": "Stimpack improves the first engagement",
        "primary_lever": "upgrade",
        "predictions": ["the first force retains more Marines"],
        "disproof_conditions": ["the upgraded force still collapses"],
        "selected_plan_ids": ["D1"],
        "selected_changes": [{"source_plan_id": "D1"}],
        "capability_mapping": {
            "macro_actions": [
                "research_stimpack",
                "train_scv",
                "build_gas",
                "expand",
            ],
            "changed_macro_actions": ["research_stimpack"],
            "unsupported_dependencies": [],
        },
    }
    plan = {
        "id": "D1",
        "primary_lever": "upgrade",
        "capability_mapping": {"macro_actions": ["research_stimpack"]},
    }

    assert critique_candidate_contract(
        rationale,
        capability_manifest=manifest,
        selected_plan=plan,
    ) == []


def test_strategy_supply_budget_rejects_end_state_over_200() -> None:
    invalid = VALID_STRATEGY + (
        "* Ultimate Goal: Continue toward 96 Marines, 20 Siege Tanks, "
        "8 Vikings, and 48 SCVs to fill the 200-supply limit.\n"
    )

    error = validate_strategy_markdown(invalid, race="terran")

    assert error is not None
    assert "requires 220 supply" in error
    assert "exceeding the hard 200 supply cap" in error


def test_strategy_supply_budget_accepts_end_state_at_200() -> None:
    valid = VALID_STRATEGY + (
        "* Ultimate Goal: Continue toward 76 Marines, 20 Siege Tanks, "
        "8 Vikings, and 48 SCVs to fill the 200-supply limit.\n"
    )

    assert validate_strategy_markdown(valid, race="terran") is None


def test_only_current_deterministic_knowledge_is_treated_as_verified(tmp_path: Path) -> None:
    checkpoint = EvolCheckpoint(tmp_path, {"stage": "created"})
    checkpoint.save_knowledge_result(
        {"question_id": "Q1", "ok": False, "answer": "", "error": "failed"}
    )
    assert checkpoint.completed_knowledge_ids() == set()
    checkpoint.save_knowledge_result({"question_id": "Q1", "ok": True, "answer": "legacy prose only"})
    assert checkpoint.completed_knowledge_ids() == set()
    checkpoint.save_knowledge_result({"question_id": "Q1", "ok": True, "answer": "verified", "verification_schema": KNOWLEDGE_VERIFICATION_SCHEMA, "dataset_evidence": [{"tool": "get_strategy_knowledge", "result": {"entities": ["Marine"]}}]})
    assert checkpoint.completed_knowledge_ids() == {"Q1"}


def test_deterministic_strategy_knowledge_is_complete_and_action_grounded() -> None:
    item = {
        "id": "Q1",
        "problem_ids": ["P1"],
        "question": (
            "Which known effects and synergies support adding Stimpack and "
            "Medivac to the Marine core?"
        ),
        "entities": ["Stimpack", "Medivac", "Marine"],
        "needs": ["effects", "synergy", "requirements"],
    }
    run = run_knowledge_query(item, race="terran")

    assert run["ok"] is True
    assert run["verification_schema"] == KNOWLEDGE_VERIFICATION_SCHEMA
    assert "research_stimpack: 100M/100G; 100.0s" in run["answer"]
    assert "train_medivac: 100M/100G/2S; 30.0s" in run["answer"]
    assert "Medivac synergizes_with Marine" in run["answer"]
    assert "BanelingBurrowed" not in run["answer"]
    assert "Medivac AI handles attack commands differently" in run["answer"]


def test_knowledge_question_fields_and_text_fallback_resolve_the_same_entities() -> None:
    question = "Compare Stimpack and Medivac support for Marine."
    explicit = resolve_knowledge_entities(
        question,
        ["Stimpack", "Medivac", "Marine"],
    )
    inferred = resolve_knowledge_entities(question)

    assert [row["name"] for row in explicit] == ["Stimpack", "Medivac", "Marine"]
    assert {row["name"] for row in inferred} == {"Stimpack", "Medivac", "Marine"}
    assert infer_knowledge_needs(question, ["synergy", "requirements"]) == [
        "synergy",
        "requirements",
    ]


def test_deterministic_knowledge_returns_only_requested_categories_with_descriptions() -> None:
    requirements = run_knowledge_query(
        {
            "id": "Q1",
            "question": "What does Marine production require?",
            "entities": ["Marine"],
            "needs": ["requirements"],
        },
        race="terran",
    )
    assert "train_marine: 50M/0G/1S" in requirements["answer"]
    assert "Marine stats:" not in requirements["answer"]
    assert "Marine:" in requirements["answer"]

    effects = run_knowledge_query(
        {
            "id": "Q2",
            "question": "What are the Marine's effects and combat properties?",
            "entities": ["Marine"],
            "needs": ["effects"],
        },
        race="terran",
    )
    assert "Marine stats:" in effects["answer"]
    assert "Marine weapon:" in effects["answer"]
    assert "train_marine:" not in effects["answer"]
    assert "Marine:" in effects["answer"]


def test_explicit_effects_need_does_not_expand_to_requirements_and_uses_scan_metadata() -> None:
    run = run_knowledge_query(
        {
            "id": "Q3",
            "question": "What is the SCV speed and Scanner Sweep energy cost and cooldown?",
            "entities": ["SCV", "Scanner Sweep"],
            "needs": ["effects"],
        },
        race="terran",
    )
    assert "SCV stats:" in run["answer"]
    assert "train_scv:" not in run["answer"]
    assert "Scanner Sweep: energy_cost=50" in run["answer"]
    assert "limit=energy_limited" in run["answer"]


def test_current_fallback_renderer_includes_full_army_group_state() -> None:
    record = _current_record()
    commander_interaction = record["interactions"][-1]
    commander_interaction.pop("text_observation")
    group_0 = commander_interaction["observation"]["army_control"]["groups"][0]
    group_0.update(
        {
            "current_command": {
                "movement_mode": "hold",
                "destination_zone_id": "zone_2",
            },
            "command_age_seconds": 8.0,
            "current_objective_status": "holding_destination",
        }
    )
    text = extract_chunks(record)["chunks"][-1]["text"]
    assert "[Army Groups]" in text
    assert "group_0: hold->zone_2" in text
    assert "objective=holding_destination" in text


def test_autosave_record_is_not_admitted_as_completed_match(tmp_path: Path) -> None:
    final_record = _current_record()
    autosave_record = json.loads(json.dumps(final_record))
    autosave_record["metadata"]["save_reason"] = "autosave_interaction"
    autosave_record["metadata"]["result"] = "Defeat"

    final_path = tmp_path / "final.json"
    autosave_path = tmp_path / "autosave.json"
    final_path.write_text(json.dumps(final_record), encoding="utf-8")
    autosave_path.write_text(json.dumps(autosave_record), encoding="utf-8")

    assert is_completed_match_record(final_record) is True
    assert is_completed_match_record(autosave_record) is False
    try:
        build_record_evidence_baseline(autosave_path)
    except IncompleteMatchRecordError:
        pass
    else:
        raise AssertionError("autosave record should be rejected")

    grouped = group_records_by_strategy([autosave_path, final_path])
    assert len(grouped[("terran", "tank")]["records"]) == 1


def test_strategy_format_uses_only_summary_and_details() -> None:
    assert validate_strategy_markdown(VALID_STRATEGY) is None
    with_costs = VALID_STRATEGY + "\n# Resource Costs\n* Marine: 50 minerals.\n"
    assert "exactly" in validate_strategy_markdown(with_costs)


def test_snapshot_writes_only_strategy_markdown() -> None:
    root = Path(__file__).resolve().parents[1] / "tmp" / f"evol-snapshot-{uuid.uuid4().hex}"
    source = root / "tank"
    output = root / "tank_opt1"
    try:
        source.mkdir(parents=True)
        changes = save_snapshot(
            source_dir=source,
            files={"strategy.md": VALID_STRATEGY},
            output_dir=output,
            source_info={},
            race="terran",
        )

        assert changes == [{"file": "strategy.md", "applied": True}]
        assert [path.name for path in output.iterdir()] == ["strategy.md"]
        assert not (output.parent / "registry.json").exists()

        try:
            save_snapshot(
                source_dir=source,
                files={"strategy.md": VALID_STRATEGY},
                output_dir=output,
                source_info={},
                race="terran",
            )
        except FileExistsError as exc:
            assert "immutable" in str(exc)
        else:
            raise AssertionError("an evaluated candidate directory must not be overwritten")
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_evol_agent_paths_and_vendored_dataset_are_local() -> None:
    project_root = Path(__file__).resolve().parents[1]
    assert config.PROJECT_ROOT == project_root
    assert config.SKILL_ROOT == project_root / "skills"

    fusion_core = get_dataset_store().get_entity("Unit", "Fusion Core")
    assert fusion_core is not None
    assert fusion_core.get("name") == "FusionCore"
