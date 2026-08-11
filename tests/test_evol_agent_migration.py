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
from evol_agent.core.checkpoint import EvolCheckpoint
from evol_agent.core.loop_helpers import normalize_strategy_contract
from evol_agent.core.prompts import build_analysis_agent_prompt
from evol_agent.core.types import BattleAnalysis
from evol_agent.optimization.snapshot import save_snapshot
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
    find_out_of_scope_knowledge_question_error,
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

def test_finish_prompt_uses_validated_diagnosis_without_repeating_match_summaries() -> None:
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
    diagnosis = {
        "strategy_contract": {
            "identity": "Two-base gathered mid-game push",
            "core_commitments": ["gather main force"],
            "optimization_boundary": "Keep the concentrated mid-game push",
            "direction": "adjust",
        },
        "problems": [{"problem_id": "P1", "problem": "loss", "evidence": ["Match 2"], "strategy_fixable": True}],
        "wins_to_preserve": [{"win_id": "W1", "pattern": "gather", "evidence": ["Match 1"], "why": "won"}],
        "cross_outcome_comparison": ["Match 1 gathered; Match 2 did not"],
        "knowledge_questions": [],
        "evidence_limits": [],
    }
    runs = [
        {"question_id": "Q1", "problem_ids": ["P1"], "query": "q", "answer": "VERIFIED_KNOWLEDGE_MARKER", "ok": True, "verification_schema": KNOWLEDGE_VERIFICATION_SCHEMA, "dataset_evidence": [{"tool": "get_strategy_knowledge", "result": {"entities": ["Marine"]}}]},
        {"question_id": "Q2", "problem_ids": ["P1"], "query": "q2", "answer": "FAILED_FALLBACK_DUMP", "error": "context overflow", "ok": False},
    ]
    finish = build_analysis_agent_prompt(
        strategy_name="tank",
        race="terran",
        single_game_analyses=analyses,
        skill_texts={"strategy.md": VALID_STRATEGY},
        tool_observations=[],
        validation_errors=[],
        phase="finish",
        diagnosis=diagnosis,
        knowledge_mode="enabled",
        knowledge_runs=runs,
    )
    diagnose = build_analysis_agent_prompt(
        strategy_name="tank",
        race="terran",
        single_game_analyses=analyses,
        skill_texts={"strategy.md": VALID_STRATEGY},
        tool_observations=[],
        validation_errors=[],
        phase="diagnose",
        diagnosis=None,
        knowledge_mode="enabled",
    )
    assert "MATCH_RAW_WIN_MARKER" not in finish
    assert "MATCH_RAW_LOSS_MARKER" not in finish
    assert finish.count("VERIFIED_KNOWLEDGE_MARKER") == 1
    assert "FAILED_FALLBACK_DUMP" not in finish
    assert "context overflow" in finish
    assert "MATCH_RAW_WIN_MARKER" in diagnose
    assert '"match_coverage"' not in diagnose
    assert '"decisive_evidence"' not in diagnose
    assert "Stimpack + Medivac for Marine sustain" not in diagnose
    assert "Which known effects and synergies support adding Stimpack" not in diagnose
    assert "Normally return one knowledge question" not in diagnose
    assert "for each independent strategy-fixable decision" in diagnose
    assert '"direction": "preserve|adjust|replace"' in diagnose
    assert "Never ask the dataset which option is best" in diagnose
    assert "only purpose is to retrieve costs" in diagnose
    assert "Link a problem_id only when the returned facts" in diagnose


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


def test_deterministic_strategy_knowledge_is_compact_and_action_grounded() -> None:
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
    assert len(run["answer"]) < 4_000


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


def test_match_dependent_optimal_timing_is_outside_static_knowledge_scope() -> None:
    error = find_out_of_scope_knowledge_question_error(
        "Using the bundled SC2 dataset tools as the source of truth, what is the optimal third-base timing?"
    )
    assert error is not None
    assert "match-dependent strategy timing" in error


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
