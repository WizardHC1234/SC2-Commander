from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from evol_agent.analysis.match_record import MatchRecordReader
from evol_agent.core.analysis_agent_loop import _summarize_matches, run_analysis_agent_loop
from evol_agent.core.checkpoint import (
    PIPELINE_VERSION,
    EvolCheckpoint,
    validate_checkpoint_fingerprint,
)
from evol_agent.core.context import render_single_game_analyses
from evol_agent.core.match_summary import run_fixed_match_summary
from evol_agent.core.prompts import (
    build_cross_match_decision_prompt,
    build_cross_match_discovery_prompt,
    build_fixed_match_summary_prompt,
)
from evol_agent.sc2_data_agent.bridge import KNOWLEDGE_VERIFICATION_SCHEMA
from evol_agent.core.types import BattleAnalysis, GameDigest


VALID_STRATEGY = """# Summary
A compact Terran plan built around a gathered Marine and Tank force.

# Details
* Opening: Build workers, supply, production, gas, and technology toward fixed absolute targets.
* Main Attack Gate: Gather the persistent main force before attacking and send reinforcements toward the same objective.
"""


def _observation(*, include_buildings: bool = True, include_technology: bool = True) -> dict:
    observation = {
        "schema_version": "2.0",
        "economy": {
            "minerals": 250,
            "vespene": 100,
            "workers": 24,
            "own_base_count": 2,
        },
        "own_forces": {
            "army_supply": 18,
            "combat_composition": {"MARINE": 12, "SIEGETANK": 2},
        },
        "enemy": {
            "visible_composition": {"MARINE": 8},
            "known_combat_composition": {"MARINE": 8},
        },
        "army_control": {
            "groups": [
                {
                    "group_id": "group_0",
                    "role": "main_force",
                    "unit_count": 14,
                    "nearest_zone_id": "zone_2",
                }
            ],
            "zones": [{"zone_id": "zone_2", "owner": "own"}],
        },
    }
    if include_buildings:
        observation["production"] = {
            "completed": {"COMMANDCENTER": 2, "BARRACKS": 3, "FACTORY": 1},
            "under_construction": {},
        }
    if include_technology:
        observation["technology"] = {"completed_upgrades": ["STIMPACK"]}
    return observation


def _commander_row(game_time: float, observation: dict, *, trigger: str = "wake_event") -> dict:
    return {
        "agent": "commander",
        "game_time": game_time,
        "trigger_reason": trigger,
        "strategy_id": "tank",
        "observation": observation,
        "text_observation": "[Game]",
        "assistant_content": "Keep the plan active.",
        "macro_tasks": [{"action": "train_marine", "to_count": 30}],
        "army_policy": {
            "commands": [
                {
                    "group_id": "group_0",
                    "movement_mode": "hold",
                    "destination_zone_id": "zone_2",
                }
            ]
        },
        "wake_event": {
            "logic": "any",
            "conditions": [{"type": "game_time_at_least", "seconds": game_time + 30}],
        },
        "accepted": True,
    }


def _match_record(*rows: dict) -> dict:
    return {
        "metadata": {
            "strategy_id": "tank",
            "result": "Defeat",
            "matchup": "TvT",
            "my_race": "Terran",
            "game_duration_seconds": 742,
            "game_duration_formatted": "12:22",
            "save_reason": "on_end",
        },
        "interactions": [
            {
                "game_time": 0.0,
                "trigger_reason": "strategy_forced",
                "forced_strategy": "tank",
                "strategy_id": "tank",
            },
            *rows,
        ],
    }


def _write_record(path: Path, record: dict) -> Path:
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _record_ns(path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        file=str(path),
        result="Defeat",
        duration="12:22",
        timeline="",
        meta={},
    )


def _multi_row_record() -> dict:
    early = _observation()
    late = copy.deepcopy(early)
    late["economy"]["workers"] = 40
    late["own_forces"]["combat_composition"] = {"MARINE": 28, "SIEGETANK": 6}
    return _match_record(
        _commander_row(120.0, early, trigger="wake_event"),
        _commander_row(248.0, late, trigger="wake_event"),
        _commander_row(700.0, late, trigger="game_ended"),
    )


def test_full_timeline_enters_single_match_llm(tmp_path: Path, monkeypatch) -> None:
    record_path = _write_record(tmp_path / "match.json", _multi_row_record())
    captured: list[str] = []

    def fake_llm(prompt: str, **kwargs):
        captured.append(prompt)
        return {"result": "Defeat", "duration_s": 742, "events": []}

    monkeypatch.setattr("evol_agent.core.match_summary.call_json_llm", fake_llm)
    run_fixed_match_summary(
        strategy_name="tank",
        race="terran",
        record=_record_ns(record_path),
        game_index=1,
        model="test-model",
        prefix="",
    )

    assert len(captured) == 1
    prompt = captured[0]
    timeline = MatchRecordReader(record_path).fixed_timeline()
    assert sum(line.startswith("R ") for line in timeline.splitlines()) >= 2
    assert "120" in prompt or "248" in prompt
    assert "700" in prompt or "248" in prompt
    assert "Complete fixed match timeline:" in prompt
    assert "deterministic_match_features.v1" not in prompt
    assert "deterministic_match_features_v1" not in prompt


def test_one_match_makes_one_llm_call(tmp_path: Path, monkeypatch) -> None:
    record_path = _write_record(tmp_path / "match.json", _multi_row_record())
    calls: list[str] = []

    def fake_llm(prompt: str, **kwargs):
        calls.append(prompt)
        return {"result": "Defeat", "duration_s": 742, "events": [{"time_s": 248}]}

    monkeypatch.setattr("evol_agent.core.match_summary.call_json_llm", fake_llm)
    run_fixed_match_summary(
        strategy_name="tank",
        race="terran",
        record=_record_ns(record_path),
        game_index=1,
        model="test-model",
        prefix="",
    )
    assert len(calls) == 1


def test_summary_output_is_factual_event_timeline(tmp_path: Path, monkeypatch) -> None:
    record_path = _write_record(tmp_path / "match.json", _multi_row_record())

    def fake_llm(prompt: str, **kwargs):
        return {
            "result": "Defeat",
            "duration_s": 742,
            "events": [
                {
                    "time_s": 248,
                    "trigger": "wake_event",
                    "own_state": {"army": "28 Marines, 6 Tanks"},
                    "commands": ["train_marine -> 30"],
                }
            ],
            "opening_and_economy": ["should not be required"],
        }

    monkeypatch.setattr("evol_agent.core.match_summary.call_json_llm", fake_llm)
    _digest, analysis, ok, errors, _events = run_fixed_match_summary(
        strategy_name="tank",
        race="terran",
        record=_record_ns(record_path),
        game_index=1,
        model="test-model",
        prefix="",
    )

    assert ok is True
    assert errors == []
    assert analysis.raw["result"] == "Defeat"
    assert analysis.raw["duration_s"] == 742
    assert isinstance(analysis.raw["events"], list)
    assert analysis.raw["events"][0]["time_s"] == 248
    assert "opening_and_economy" not in analysis.raw


def test_missing_state_fields_are_omitted(tmp_path: Path, monkeypatch) -> None:
    record = _match_record(
        _commander_row(
            200.0,
            _observation(include_buildings=False, include_technology=False),
        )
    )
    record_path = _write_record(tmp_path / "match.json", record)

    def fake_llm(prompt: str, **kwargs):
        return {
            "result": "Defeat",
            "duration_s": 742,
            "events": [
                {
                    "time_s": 200,
                    "trigger": "wake_event",
                    "own_state": {"economy": "24 workers", "army": "12 Marines"},
                    "enemy_observed": {"army": "8 Marines"},
                }
            ],
        }

    monkeypatch.setattr("evol_agent.core.match_summary.call_json_llm", fake_llm)
    _digest, analysis, ok, _errors, _events = run_fixed_match_summary(
        strategy_name="tank",
        race="terran",
        record=_record_ns(record_path),
        game_index=1,
        model="test-model",
        prefix="",
    )

    assert ok is True
    own_state = analysis.raw["events"][0]["own_state"]
    assert "economy" in own_state
    assert "army" in own_state
    assert "technology" not in own_state
    assert "buildings" not in own_state


def test_enemy_observed_and_enemy_truth_stay_separated(tmp_path: Path, monkeypatch) -> None:
    prompt = build_fixed_match_summary_prompt(
        strategy_name="tank",
        race="terran",
        record_manifest={"result": "Defeat"},
        match_timeline="R observed=A truth=B",
    )
    assert "enemy_observed" in prompt
    assert "enemy_truth" in prompt
    assert "Keep two enemy sources strictly separate" in prompt

    record_path = _write_record(tmp_path / "match.json", _multi_row_record())

    def fake_llm(prompt_text: str, **kwargs):
        return {
            "result": "Defeat",
            "duration_s": 742,
            "events": [
                {
                    "time_s": 248,
                    "enemy_observed": {"army": "A"},
                    "enemy_truth": {"army": "B"},
                    "commands": ["army_intent: regroup -> zone_2"],
                }
            ],
        }

    monkeypatch.setattr("evol_agent.core.match_summary.call_json_llm", fake_llm)
    _digest, analysis, ok, _errors, _events = run_fixed_match_summary(
        strategy_name="tank",
        race="terran",
        record=_record_ns(record_path),
        game_index=1,
        model="test-model",
        prefix="",
    )
    assert ok is True
    event = analysis.raw["events"][0]
    assert event["enemy_observed"] == {"army": "A"}
    assert event["enemy_truth"] == {"army": "B"}


def test_single_match_prompt_forbids_analysis() -> None:
    prompt = build_fixed_match_summary_prompt(
        strategy_name="tank",
        race="terran",
        record_manifest={"result": "Defeat"},
        match_timeline="R 1",
    )
    lowered = prompt.lower()
    assert "do not diagnose" in lowered
    assert "root cause" in lowered
    assert "do not recommend strategy changes" in lowered
    assert "good or bad" in lowered
    assert "opening_and_economy" not in prompt
    assert "commander_decision_summary" not in prompt


def test_cross_match_prompt_keeps_complete_events() -> None:
    analyses = [
        BattleAnalysis(
            strategy_name="tank",
            race="terran",
            sample_size=1,
            record_mix="0W/1L",
            raw={
                "result": "Defeat",
                "duration_s": 431,
                "events": [
                    {
                        "time_s": 431,
                        "enemy_observed": "visible marines",
                        "enemy_truth": "hidden tanks",
                        "commands": ["army_intent: attack -> zone_9"],
                    }
                ],
            },
        )
    ]
    rendered = render_single_game_analyses(analyses)
    assert '"time_s":431' in rendered
    assert '"enemy_observed":"visible marines"' in rendered
    assert '"enemy_truth":"hidden tanks"' in rendered
    assert '"commands":["army_intent: attack -> zone_9"]' in rendered

    prompt = build_cross_match_discovery_prompt(
        strategy_name="tank",
        race="terran",
        single_game_analyses=analyses,
        skill_texts={"strategy.md": VALID_STRATEGY},
        validation_errors=[],
        knowledge_mode="enabled",
    )
    assert "Independent factual match summaries" in prompt
    assert "deterministic match evidence" not in prompt
    assert "Cross-Match Discovery Agent" in prompt
    assert "Do not return next_action, candidate_plans, candidate_rule, or target_paragraph_id" in prompt
    assert "query_knowledge" not in prompt
    assert '"time_s":431' in prompt
    assert '"enemy_observed":"visible marines"' in prompt
    assert '"enemy_truth":"hidden tanks"' in prompt


def test_one_failed_match_does_not_stop_the_batch(tmp_path: Path, monkeypatch) -> None:
    records = []
    for index in range(1, 4):
        path = _write_record(tmp_path / f"match_{index}.json", _multi_row_record())
        records.append(_record_ns(path))

    def fake_llm(prompt: str, **kwargs):
        if "match_002" in prompt:
            return None
        return {
            "result": "Defeat",
            "duration_s": 742,
            "events": [{"time_s": 248, "trigger": "wake_event"}],
        }

    monkeypatch.setattr("evol_agent.core.match_summary.call_json_llm", fake_llm)
    digests, analyses, completed, _events, errors = _summarize_matches(
        strategy_name="tank",
        race="terran",
        records=records,
        skill_texts={"strategy.md": VALID_STRATEGY},
        model="test-model",
        prefix="",
        checkpoint=None,
    )

    assert len(digests) == 3
    assert len(analyses) == 3
    assert completed == 2
    assert analyses[0].raw["events"]
    assert analyses[1].raw["summary_quality"] == "degraded"
    assert analyses[2].raw["events"]
    assert any("match_002" in error for error in errors)


def test_old_checkpoint_pipeline_version_is_rejected(tmp_path: Path) -> None:
    old = EvolCheckpoint(
        tmp_path,
        {
            "pipeline_version": "deterministic_features_v1_paragraph_patch_v1",
            "strategy_name": "tank",
            "race": "terran",
            "knowledge_mode": "enabled",
            "record_files": [],
        },
    )
    try:
        validate_checkpoint_fingerprint(
            old,
            strategy_name="tank",
            race="terran",
            knowledge_mode="enabled",
            record_files=[],
        )
        raise AssertionError("old checkpoint should be rejected")
    except ValueError as exc:
        assert "pipeline_version mismatch" in str(exc)

    current = EvolCheckpoint(
        tmp_path,
        {
            "pipeline_version": PIPELINE_VERSION,
            "strategy_name": "tank",
            "race": "terran",
            "knowledge_mode": "enabled",
            "record_files": [],
        },
    )
    validate_checkpoint_fingerprint(
        current,
        strategy_name="tank",
        race="terran",
        knowledge_mode="enabled",
        record_files=[],
    )
    assert PIPELINE_VERSION == "full_timeline_summary_v1_cross_match_discovery_v1"


def _stub_summaries():
    digest = GameDigest(
        record_path="match.json",
        result="Defeat",
        duration="12:22",
        summary="Defeat",
    )
    analysis = BattleAnalysis(
        strategy_name="tank",
        race="terran",
        sample_size=1,
        record_mix="0W/1L",
        raw={"result": "Defeat", "duration_s": 742, "events": [{"time_s": 430}]},
    )
    return [digest], [analysis], 1, [], []


def _empty_discovery() -> dict:
    return {
        "strengths": [{"pattern": "two-base opener", "evidence": ["Game 1 @ 90s"]}],
        "weaknesses": [{"pattern": "Tank production is too late", "evidence": ["Game 2 @ 430s"]}],
        "unknowns": [
            {
                "unknown": "whether more tanks were feasible at that producer count",
                "why_it_matters": "it changes whether strategy.md is at fault",
                "evidence": ["Game 2 @ 430s"],
            }
        ],
        "knowledge_questions": [],
    }


def _propose_decision(**overrides) -> dict:
    payload = {
        "strengths_to_preserve": [{"pattern": "two-base opener", "evidence": ["Game 1 @ 90s"]}],
        "priority_problem": {
            "problem": "first fight is too weak",
            "evidence": ["Game 2 @ 430s"],
            "control_class": "strategy_fixable",
        },
        "hypothesis": "a second factory completes more tanks before contact",
        "next_action": "propose_strategy_patch",
        "action_reason": "the first fight is repeatedly too weak",
        "plan": {
            "direction": "Build a second Factory before marine scaling.",
            "preserve": ["two-base opener"],
        },
        "evidence_limits": [],
    }
    payload.update(overrides)
    return payload


def _verified_knowledge_run() -> dict:
    return {
        "question_id": "Q1",
        "question": "What are Siege Tank production requirements?",
        "ok": True,
        "query": (
            "What are Siege Tank production requirements? | "
            "entities=Siege Tank | needs=requirements | race=terran"
        ),
        "answer": "Earlier completion was infeasible at that producer count.",
        "error": "",
        "verification_schema": KNOWLEDGE_VERIFICATION_SCHEMA,
        "dataset_evidence": [{"tool": "get_strategy_knowledge", "result": {}}],
    }


def test_round2_prompt_reuses_discovery_findings() -> None:
    prompt = build_cross_match_decision_prompt(
        strategy_name="tank",
        race="terran",
        single_game_analyses=[
            BattleAnalysis(
                strategy_name="tank",
                race="terran",
                sample_size=1,
                record_mix="0W/1L",
                raw={"result": "Defeat", "events": [{"time_s": 430}]},
            )
        ],
        skill_texts={"strategy.md": VALID_STRATEGY},
        validation_errors=[],
        knowledge_mode="enabled",
        discovery={
            "strengths": [{"pattern": "two-base opener", "evidence": ["Game 1 @ 90s"]}],
            "weaknesses": [{"pattern": "first fight loses more army", "evidence": ["Game 2 @ 430s"]}],
            "unknowns": [
                {
                    "unknown": "tank production cap",
                    "why_it_matters": "feasibility",
                    "evidence": ["Game 2 @ 430s"],
                }
            ],
        },
        knowledge_runs=[],
    )
    assert "Cross-Match Decision Agent" in prompt
    assert "Knowledge may invalidate an earlier interpretation" in prompt
    assert "Do not query the knowledge database again" in prompt
    assert "tank production cap" in prompt
    assert "target_paragraph_id" not in prompt.split("Return one JSON object only:")[1]


def test_cross_match_always_makes_two_llm_calls_without_knowledge(monkeypatch) -> None:
    prompts: list[str] = []

    def fake_llm(prompt: str, **kwargs):
        prompts.append(prompt)
        if "Cross-Match Discovery Agent" in prompt:
            return _empty_discovery()
        assert "Cross-Match Decision Agent" in prompt
        return _propose_decision()

    monkeypatch.setattr("evol_agent.core.analysis_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.analysis_agent_loop._summarize_matches",
        lambda **kwargs: _stub_summaries(),
    )
    result = run_analysis_agent_loop(
        strategy_name="tank",
        race="terran",
        records=[_record_ns(Path("match.json"))],
        skill_texts={"strategy.md": VALID_STRATEGY},
        knowledge_mode="enabled",
    )
    complete = next(item for item in result.events if item.get("action") == "analysis_complete")
    assert result.completed is True
    assert len(prompts) == 2
    assert complete["llm_cross_match_calls"] == 2
    assert result.battle_analysis.raw["plan"]["direction"].startswith("Build a second Factory")
    assert any(item.get("action") == "cross_match_discovery" for item in result.events)
    assert any(item.get("action") == "cross_match_decision" for item in result.events)


def test_cross_match_queries_knowledge_once_then_decides(monkeypatch) -> None:
    prompts: list[str] = []
    knowledge_calls: list[list] = []

    def fake_llm(prompt: str, **kwargs):
        prompts.append(prompt)
        if "Cross-Match Discovery Agent" in prompt:
            discovery = _empty_discovery()
            discovery["knowledge_questions"] = [
                {
                    "question": "What are Siege Tank production requirements?",
                    "entities": ["Siege Tank"],
                    "needs": ["requirements"],
                }
            ]
            return discovery
        assert "Cross-Match Decision Agent" in prompt
        assert "Earlier completion was infeasible" in prompt
        return _propose_decision()

    def fake_knowledge(questions, **kwargs):
        knowledge_calls.append(questions)
        return [_verified_knowledge_run()]

    monkeypatch.setattr("evol_agent.core.analysis_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.analysis_agent_loop._summarize_matches",
        lambda **kwargs: _stub_summaries(),
    )
    monkeypatch.setattr(
        "evol_agent.core.analysis_agent_loop._run_knowledge_queries",
        fake_knowledge,
    )
    result = run_analysis_agent_loop(
        strategy_name="tank",
        race="terran",
        records=[_record_ns(Path("match.json"))],
        skill_texts={"strategy.md": VALID_STRATEGY},
        knowledge_mode="enabled",
    )
    complete = next(item for item in result.events if item.get("action") == "analysis_complete")
    assert result.completed is True
    assert len(prompts) == 2
    assert len(knowledge_calls) == 1
    assert complete["llm_cross_match_calls"] == 2
    assert result.battle_analysis.raw["next_action"] == "propose_strategy_patch"
    assert result.knowledge_trace["discovery"]["knowledge_questions"][0]["id"] == "Q1"


def test_round2_can_reject_round1_weakness(monkeypatch) -> None:
    def fake_llm(prompt: str, **kwargs):
        if "Cross-Match Discovery Agent" in prompt:
            discovery = _empty_discovery()
            discovery["knowledge_questions"] = [
                {
                    "question": "What are Siege Tank production requirements?",
                    "entities": ["Siege Tank"],
                    "needs": ["requirements"],
                }
            ]
            return discovery
        assert "Earlier completion was infeasible" in prompt
        return {
            "strengths_to_preserve": [{"pattern": "two-base opener", "evidence": ["Game 1 @ 90s"]}],
            "priority_problem": {
                "problem": "Tank production is too late",
                "evidence": ["Game 2 @ 430s"],
                "control_class": "observation_limited",
            },
            "hypothesis": "",
            "next_action": "request_more_matches",
            "action_reason": "knowledge shows earlier tanks were infeasible; more games are needed",
            "plan": None,
        }

    monkeypatch.setattr("evol_agent.core.analysis_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.analysis_agent_loop._summarize_matches",
        lambda **kwargs: _stub_summaries(),
    )
    monkeypatch.setattr(
        "evol_agent.core.analysis_agent_loop._run_knowledge_queries",
        lambda questions, **kwargs: [_verified_knowledge_run()],
    )
    result = run_analysis_agent_loop(
        strategy_name="tank",
        race="terran",
        records=[_record_ns(Path("match.json"))],
        skill_texts={"strategy.md": VALID_STRATEGY},
        knowledge_mode="enabled",
    )
    assert result.completed is True
    assert result.battle_analysis.raw["next_action"] == "request_more_matches"
    assert result.battle_analysis.raw["plan"] is None


def test_inspect_runtime_does_not_need_a_plan(monkeypatch) -> None:
    def fake_llm(prompt: str, **kwargs):
        if "Cross-Match Discovery Agent" in prompt:
            return _empty_discovery()
        return {
            "next_action": "inspect_runtime",
            "action_reason": "Commander already issued the gather command but movement failed",
            "priority_problem": {
                "problem": "group movement is rejected",
                "evidence": ["Game 1 @ 200s"],
                "control_class": "runtime_execution",
            },
            "plan": None,
        }

    monkeypatch.setattr("evol_agent.core.analysis_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.analysis_agent_loop._summarize_matches",
        lambda **kwargs: _stub_summaries(),
    )
    result = run_analysis_agent_loop(
        strategy_name="tank",
        race="terran",
        records=[_record_ns(Path("match.json"))],
        skill_texts={"strategy.md": VALID_STRATEGY},
        knowledge_mode="enabled",
    )
    assert result.completed is True
    assert result.battle_analysis.raw["next_action"] == "inspect_runtime"
    assert result.battle_analysis.raw["plan"] is None
    assert result.battle_analysis.raw["candidate_plans"] == []


def test_resume_skips_discovery_and_reuses_knowledge_cache(tmp_path: Path, monkeypatch) -> None:
    checkpoint = EvolCheckpoint(tmp_path, {"stage": "created"})
    discovery = _empty_discovery()
    discovery["knowledge_questions"] = [
        {
            "id": "Q1",
            "question": "What are Siege Tank production requirements?",
            "entities": ["Siege Tank"],
            "needs": ["requirements"],
        }
    ]
    checkpoint.save_cross_match_discovery(discovery)
    checkpoint.save_knowledge_result(_verified_knowledge_run())
    prompts: list[str] = []
    knowledge_llm_calls: list[int] = []

    def fake_llm(prompt: str, **kwargs):
        prompts.append(prompt)
        assert "Cross-Match Discovery Agent" not in prompt
        assert "Cross-Match Decision Agent" in prompt
        assert "Earlier completion was infeasible" in prompt
        return _propose_decision()

    def boom_knowledge(*args, **kwargs):
        knowledge_llm_calls.append(1)
        raise AssertionError("cached knowledge should be reused")

    monkeypatch.setattr("evol_agent.core.analysis_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.analysis_agent_loop._summarize_matches",
        lambda **kwargs: _stub_summaries(),
    )
    monkeypatch.setattr(
        "evol_agent.core.analysis_agent_loop.run_knowledge_query",
        boom_knowledge,
    )
    result = run_analysis_agent_loop(
        strategy_name="tank",
        race="terran",
        records=[_record_ns(Path("match.json"))],
        skill_texts={"strategy.md": VALID_STRATEGY},
        knowledge_mode="enabled",
        checkpoint=checkpoint,
    )
    assert result.completed is True
    assert len(prompts) == 1
    assert knowledge_llm_calls == []
    assert result.battle_analysis.raw["next_action"] == "propose_strategy_patch"


def test_rejected_experiments_are_visible_in_round2(monkeypatch) -> None:
    prompts: list[str] = []
    prior = [
        {
            "primary_lever": "attack_timing",
            "hypothesis": "attack earlier with the same composition",
            "delta": -0.42,
            "lesson": "same primary lever produced significant score regression",
            "experiment_evidence": {
                "candidate_minus_parent": {"score": -0.42},
                "parent_batch": {"wins": 3, "losses": 1, "games": 4, "score": 0.7},
                "candidate_batch": {"wins": 1, "losses": 3, "games": 4, "score": 0.28},
            },
        }
    ]

    def fake_llm(prompt: str, **kwargs):
        prompts.append(prompt)
        if "Cross-Match Discovery Agent" in prompt:
            return _empty_discovery()
        return _propose_decision()

    monkeypatch.setattr("evol_agent.core.analysis_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.analysis_agent_loop._summarize_matches",
        lambda **kwargs: _stub_summaries(),
    )
    result = run_analysis_agent_loop(
        strategy_name="tank",
        race="terran",
        records=[_record_ns(Path("match.json"))],
        skill_texts={"strategy.md": VALID_STRATEGY},
        knowledge_mode="enabled",
        prior_experiences=prior,
    )
    assert result.completed is True
    assert any("Cross-Match Decision Agent" in prompt for prompt in prompts)
    decision_prompt = next(prompt for prompt in prompts if "Cross-Match Decision Agent" in prompt)
    assert "attack_timing" in decision_prompt
    assert "same primary lever produced significant score regression" in decision_prompt
    assert "candidate_minus_parent" in decision_prompt

