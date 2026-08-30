from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evol_agent.analysis.match_record import MatchRecordReader
from evol_agent.core.analysis_agent_loop import (
    _annotate_package_knowledge,
    _candidate_package_knowledge_questions,
    _evaluate_candidate_package_budgets,
    _normalize_candidate_package_proposal,
    _normalize_cross_match_decision,
    _normalize_parent_timing_package_extraction,
    _summarize_matches,
    run_analysis_agent_loop,
)
from evol_agent.core.checkpoint import (
    PIPELINE_VERSION,
    EvolCheckpoint,
    validate_analysis_seed_checkpoint,
    validate_checkpoint_fingerprint,
)
from evol_agent.core.context import (
    render_engagement_transition_digest,
    render_single_game_analyses,
)
from evol_agent.core.experiment_audit import _normalize_audit
from evol_agent.core.match_summary import _normalize_summary_payload, run_fixed_match_summary
from evol_agent.core.match_summary_cache import MATCH_SUMMARY_FORMAT, MatchSummaryCache
from evol_agent.core.optimization_agent_loop import extract_final_cross_match_decision
from evol_agent.core.prompts import (
    build_cross_match_decision_prompt,
    build_cross_match_discovery_prompt,
    build_fixed_match_summary_prompt,
    build_optimization_package_prompt,
    build_parent_timing_package_prompt,
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


def test_fixed_timeline_adds_support_aware_power_for_medivacs(tmp_path: Path) -> None:
    observation = _observation()
    observation["production"]["completed"]["MEDIVAC"] = 2
    observation["combat"] = {
        "controlled_own_army_power": 18.0,
        "visible_enemy_army_power": 8.0,
    }
    record_path = _write_record(
        tmp_path / "support.json",
        _match_record(_commander_row(300.0, observation)),
    )

    timeline = MatchRecordReader(record_path).fixed_timeline()

    assert "support_aware_power" in timeline
    assert "bounded_medivac_sustain" in timeline
    assert "support_adjusted_power" in timeline


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
            "enemy_pressure_events": [
                {
                    "time_s": 248,
                    "observed_cue": "enemy force reached the natural",
                    "own_defense": "28 Marines, 6 Tanks",
                    "enemy_observed": "visible ground army",
                    "enemy_truth": "larger mixed army",
                    "outcome": "army_broken",
                }
            ],
            "major_engagements": [
                {
                    "time_s": 248,
                    "initiator": "enemy",
                    "contact_zone": "zone_1",
                    "zone_owner": "own",
                    "terrain_context": "own ramp with prepared Siege Tanks",
                    "own_force_before": "28 Marines, 6 Tanks",
                    "enemy_observed": "visible ground army",
                    "enemy_truth": "larger mixed army",
                    "own_force_after": "small remnant",
                    "enemy_force_after": "enemy withdrew with two Tanks",
                    "own_reinforcement_after": "three Marines arrive after withdrawal",
                    "production_context_before": "six Barracks, two idle queues",
                    "runtime_override": "auto-retreat fired after the army fell to a small remnant",
                    "retreat_policy": "retreat_ratio=0.6; local power ratio=0.31; no re-engagement",
                    "loss_timing": "losses_before_override",
                    "outcome": "army_broken",
                }
            ],
            "defense_to_counterattack_windows": [
                {
                    "pressure_time_s": 248,
                    "pressure_cleared_time_s": 270,
                    "contact_zone": "zone_1",
                    "own_force_after_defense": "18 Marines, 4 Tanks",
                    "enemy_force_after_defense": "two retreating Tanks",
                    "next_offensive_time_s": 290,
                    "counterattack_delay_seconds": 20,
                    "next_offensive_command": "assault enemy natural",
                    "counterattack_outcome": "breakthrough",
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
    assert analysis.raw["enemy_pressure_events"][0]["outcome"] == "army_broken"
    assert analysis.raw["major_engagements"][0]["initiator"] == "enemy"
    assert analysis.raw["major_engagements"][0]["contact_zone"] == "zone_1"
    assert analysis.raw["major_engagements"][0]["zone_owner"] == "own"
    assert "prepared Siege Tanks" in analysis.raw["major_engagements"][0][
        "terrain_context"
    ]
    assert analysis.raw["major_engagements"][0]["loss_timing"] == (
        "losses_before_override"
    )
    assert "auto-retreat" in analysis.raw["major_engagements"][0][
        "runtime_override"
    ]
    assert analysis.raw["major_engagements"][0]["retreat_policy"].startswith(
        "retreat_ratio=0.6"
    )
    assert "three Marines" in analysis.raw["major_engagements"][0][
        "own_reinforcement_after"
    ]
    assert analysis.raw["defense_to_counterattack_windows"][0][
        "counterattack_delay_seconds"
    ] == "20"
    assert analysis.raw["defense_to_counterattack_windows"][0][
        "counterattack_outcome"
    ] == "breakthrough"
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
    assert "defense_to_counterattack_windows" in prompt
    assert "contact_zone" in prompt
    assert "terrain_context" in prompt
    assert "does not make the decisive engagement enemy-initiated" in prompt


def test_audit_focused_summary_requests_and_normalizes_mechanism_probe() -> None:
    focus = {
        "minimum_material_change": "at least two Vikings before first contact",
        "expected_change": "air support is present before the decisive fight",
    }
    prompt = build_fixed_match_summary_prompt(
        strategy_name="tank_opt7",
        race="terran",
        record_manifest={"result": "Victory"},
        match_timeline="R 1",
        audit_focus=focus,
    )

    assert "post-experiment mechanism audit" in prompt
    assert focus["minimum_material_change"] in prompt
    assert '"mechanism_probe"' in prompt
    payload = _normalize_summary_payload(
        {
            "result": "Victory",
            "duration_s": 600,
            "events": [],
            "enemy_pressure_events": [],
            "major_engagements": [],
            "mechanism_probe": {
                "status": "observed",
                "observations": [
                    {"time_s": 540, "fact": "2 Vikings present before contact"}
                ],
                "evidence_limit": "",
            },
        },
        manifest={"result": "Victory"},
        duration_s=600,
    )

    assert payload is not None
    assert payload["mechanism_probe"] == {
        "status": "observed",
        "observations": [
            {"time_s": 540, "fact": "2 Vikings present before contact"}
        ],
        "evidence_limit": "",
    }


def test_audit_probe_without_timestamp_is_unknown() -> None:
    payload = _normalize_summary_payload(
        {
            "result": "Victory",
            "duration_s": 600,
            "events": [],
            "enemy_pressure_events": [],
            "major_engagements": [],
            "mechanism_probe": {
                "status": "observed",
                "observations": [{"fact": "Vikings were present"}],
                "evidence_limit": "",
            },
        },
        manifest={"result": "Victory"},
        duration_s=600,
    )

    assert payload is not None
    assert payload["mechanism_probe"]["status"] == "unknown"
    assert "no recorded timestamp" in payload["mechanism_probe"]["evidence_limit"]


def test_audit_requires_two_observed_candidate_probes_for_implemented() -> None:
    raw = {
        "implementation_verdict": "implemented",
        "hypothesis_verdict": "supported",
        "evidence_limits": [],
    }
    one_observed = [
        {
            "summary": {
                "mechanism_probe": {
                    "status": "observed",
                    "observations": [{"time_s": 540, "fact": "4 Vikings present"}],
                }
            }
        }
    ]

    thin = _normalize_audit(
        raw,
        candidate_summaries=one_observed,
        require_observed_probes=True,
    )
    implemented = _normalize_audit(
        raw,
        candidate_summaries=[
            *one_observed,
            {
                "summary": {
                    "mechanism_probe": {
                        "status": "observed",
                        "observations": [
                            {"time_s": 565, "fact": "4 Vikings present"}
                        ],
                    }
                }
            },
        ],
        require_observed_probes=True,
    )

    assert thin["implementation_verdict"] == "underpowered"
    assert thin["hypothesis_verdict"] == "not_tested"
    assert implemented["implementation_verdict"] == "implemented"
    assert implemented["hypothesis_verdict"] == "supported"


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
    assert "depends on costs, prerequisites, production time" in prompt
    assert "query_knowledge" not in prompt
    assert "Treat the defining combat style and win mechanism as strategy identity" in prompt
    assert "not automatic opportunities for expansion" in prompt
    assert "who initiated the decisive engagement" in prompt
    assert "trace every successful defense into its next counterattack window" in prompt
    assert "engagement_initiative_patterns" in prompt
    assert "defense_counterattack_patterns" in prompt
    assert '"time_s":431' in prompt
    assert '"enemy_observed":"visible marines"' in prompt
    assert '"enemy_truth":"hidden tanks"' in prompt


def test_engagement_transition_digest_keeps_initiative_and_counterattack() -> None:
    analysis = BattleAnalysis(
        strategy_name="tank",
        race="terran",
        sample_size=1,
        record_mix="1W/0L",
        raw={
            "result": "Victory",
            "major_engagements": [
                {
                    "time_s": 420,
                    "initiator": "enemy",
                    "contact_zone": "zone_1",
                    "zone_owner": "own",
                    "terrain_context": "own ramp",
                    "outcome": "held",
                }
            ],
            "defense_to_counterattack_windows": [
                {
                    "pressure_time_s": 420,
                    "counterattack_delay_seconds": 18,
                    "counterattack_outcome": "breakthrough",
                }
            ],
        },
    )

    digest = render_engagement_transition_digest([analysis])

    assert '"enemy:own:held":1' in digest
    assert '"contact_zone":"zone_1"' in digest
    assert '"counterattack_delay_seconds":18' in digest
    assert '"counterattack_outcome":"breakthrough"' in digest


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


def test_match_summary_seed_reuses_old_records_and_summarizes_only_new_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    records = [
        _record_ns(_write_record(tmp_path / f"match_{index}.json", _multi_row_record()))
        for index in range(1, 4)
    ]
    seed = EvolCheckpoint(tmp_path / "seed", {"stage": "created"})
    seed.run_dir.mkdir(parents=True)
    seed_digests = [
        GameDigest(
            record_path=record.file,
            result="Defeat",
            duration="12:22",
            summary=f"cached-{index}",
            raw={"record_path": record.file, "summary": f"cached-{index}"},
        )
        for index, record in enumerate(records[:2], 1)
    ]
    seed_analyses = [
        BattleAnalysis(
            strategy_name="tank",
            race="terran",
            sample_size=1,
            record_mix="0W/1L",
                raw={
                    "summary": f"cached-{index}",
                    "summary_format": MATCH_SUMMARY_FORMAT,
                },
        )
        for index in range(1, 3)
    ]
    seed.save_match_summaries(
        game_digests=seed_digests,
        single_game_analyses=seed_analyses,
        completed_matches=2,
        events=[
            {"record_path": record.file, "completed": True}
            for record in records[:2]
        ],
    )
    target = EvolCheckpoint(tmp_path / "target", {"stage": "created"})
    target.run_dir.mkdir(parents=True)
    summarized: list[str] = []

    def fake_summary(*, record, game_index: int, **_kwargs):
        summarized.append(record.file)
        digest = GameDigest(
            record_path=record.file,
            result="Defeat",
            duration="12:22",
            summary="new",
            raw={"record_path": record.file, "summary": "new"},
        )
        analysis = BattleAnalysis(
            strategy_name="tank",
            race="terran",
            sample_size=1,
            record_mix="0W/1L",
            raw={"summary": "new", "game_index": game_index},
        )
        return digest, analysis, True, [], []

    monkeypatch.setattr(
        "evol_agent.core.analysis_agent_loop.run_fixed_match_summary",
        fake_summary,
    )
    digests, analyses, completed, events, errors = _summarize_matches(
        strategy_name="tank",
        race="terran",
        records=records,
        skill_texts={"strategy.md": VALID_STRATEGY},
        model="test-model",
        prefix="",
        checkpoint=target,
        summary_seed_checkpoint=seed,
    )

    assert summarized == [records[2].file]
    assert [digest.summary for digest in digests] == ["cached-1", "cached-2", "new"]
    assert len(analyses) == 3
    assert completed == 3
    assert errors == []
    assert [event["reused"] for event in events] == [True, True, False]
    assert target.load_match_summaries()[2] == 3


def test_match_summary_combines_checkpoint_seed_with_persistent_audit_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    records = [
        _record_ns(_write_record(tmp_path / f"match_{index}.json", _multi_row_record()))
        for index in range(1, 4)
    ]
    seed = EvolCheckpoint(tmp_path / "seed", {"stage": "created"})
    seed.run_dir.mkdir(parents=True)
    seed.save_match_summaries(
        game_digests=[
            GameDigest(
                record_path=record.file,
                result="Defeat",
                duration="12:22",
                summary=f"checkpoint-{index}",
                raw={
                    "record_path": record.file,
                    "result": "Defeat",
                    "duration": "12:22",
                    "summary": f"checkpoint-{index}",
                },
            )
            for index, record in enumerate(records[:2], 1)
        ],
        single_game_analyses=[
            BattleAnalysis(
                strategy_name="tank",
                race="terran",
                sample_size=1,
                record_mix="0W/1L",
                    raw={
                        "strategy_name": "tank",
                        "race": "terran",
                        "sample_size": 1,
                        "record_mix": "0W/1L",
                        "summary_format": MATCH_SUMMARY_FORMAT,
                        "result": "Defeat",
                        "events": [],
                    },
            )
            for _record in records[:2]
        ],
        completed_matches=2,
        events=[
            {"record_path": record.file, "completed": True}
            for record in records[:2]
        ],
    )
    cache_path = tmp_path / "summary_cache.json"
    cache = MatchSummaryCache(cache_path)
    cache.put(
        records[2],
        strategy_name="tank",
        race="terran",
        model="test-model",
        summary={
            "strategy_name": "tank",
            "race": "terran",
            "sample_size": 1,
            "record_mix": "0W/1L",
            "result": "Defeat",
            "duration_s": 742,
            "events": [],
        },
        errors=[],
        source="experiment_audit",
        digest={
            "record_path": records[2].file,
            "result": "Defeat",
            "duration": "12:22",
            "summary": "audit-cache",
        },
    )
    target = EvolCheckpoint(tmp_path / "target", {"stage": "created"})
    target.run_dir.mkdir(parents=True)

    def unexpected_summary(**_kwargs):
        raise AssertionError("all summaries should be reused")

    monkeypatch.setattr(
        "evol_agent.core.analysis_agent_loop.run_fixed_match_summary",
        unexpected_summary,
    )
    digests, analyses, completed, events, errors = _summarize_matches(
        strategy_name="tank",
        race="terran",
        records=records,
        skill_texts={"strategy.md": VALID_STRATEGY},
        model="test-model",
        prefix="",
        checkpoint=target,
        summary_seed_checkpoint=seed,
        match_summary_cache_path=cache_path,
    )

    assert [digest.summary for digest in digests] == [
        "checkpoint-1",
        "checkpoint-2",
        "audit-cache",
    ]
    assert len(analyses) == 3
    assert completed == 3
    assert errors == []
    assert [event["reused"] for event in events] == [True, True, True]
    assert events[2]["events"][0]["cached_by"] == "experiment_audit"
    assert digests[2].raw["game_index"] == 3


def test_analysis_seed_accepts_a_completed_subset(tmp_path: Path) -> None:
    current = [str((tmp_path / f"match_{index}.json").resolve()) for index in range(3)]
    seed = EvolCheckpoint(
        tmp_path / "seed",
        {
            "pipeline_version": PIPELINE_VERSION,
            "stage": "analysis_complete",
            "strategy_name": "tank",
            "race": "terran",
            "knowledge_mode": "enabled",
            "models": {"analysis": "test-model"},
            "record_files": current[:2],
        },
    )

    validate_analysis_seed_checkpoint(
        seed,
        strategy_name="tank",
        race="terran",
        knowledge_mode="enabled",
        record_files=current,
        analysis_model="test-model",
    )

    with pytest.raises(ValueError, match="subset"):
        validate_analysis_seed_checkpoint(
            seed,
            strategy_name="tank",
            race="terran",
            knowledge_mode="enabled",
            record_files=current[1:],
            analysis_model="test-model",
        )


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
    assert PIPELINE_VERSION == "full_timeline_summary_v1_intent_contrast_v2"


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
        "strategy_contract": {
            "identity": "two-base gathered push",
            "style": "concentrated timing attack",
            "core_win_mechanism": "assemble the intended package before committing",
            "critical_power_window": "first completed gathered push",
            "core_commitments": ["two-base opener", "gather before attacking"],
            "protected_invariants": ["two-base opener", "gather before attacking"],
        },
        "outcome_contrast": {
            "winning_pattern": "completed gathered push survives first contact",
            "winning_evidence": ["Game 1 @ 620s"],
            "loss_shortfall": "the core force is incomplete at contact",
            "loss_evidence": ["Game 2 @ 430s"],
            "loss_relationship_to_wins": "winning_mechanism_reproduced_but_failed",
            "causal_difference": "wins retain more of the intended core force",
            "preservation_rule": "retain the two-base gathered-push structure",
        },
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
        "outcome_contrast": {
            "winning_pattern": "completed gathered push survives first contact",
            "winning_evidence": ["Game 1 @ 620s"],
            "loss_shortfall": "the core force is incomplete at contact",
            "loss_evidence": ["Game 2 @ 430s", "Game 5 @ 510s"],
            "loss_relationship_to_wins": "winning_mechanism_reproduced_but_failed",
            "causal_difference": "wins retain more of the intended core force",
            "preservation_rule": "retain the two-base gathered-push structure",
        },
        "priority_problem": {
            "problem": "first fight is too weak",
            "evidence": ["Game 2 @ 430s"],
            "control_class": "strategy_fixable",
        },
        "hypothesis": "a second factory completes more tanks before contact",
        "failure_mode_analysis": {
            "failure_stage": "during_commitment_or_engagement",
            "gate_attainment_and_launch": "the intended gate is reached and commitment follows without a recorded delay",
            "earliest_strategy_fixable_link": "the completed package is insufficient at first contact",
            "why_later_levers_do_not_outrank_it": "post-contact recovery cannot repair the initial package loss",
            "commitment_and_contact_timing": "the force commits in the intended window but meets stronger pressure",
            "own_package_at_contact": "the core force is incomplete at decisive contact",
            "opponent_package_and_growth": "the opponent has a stronger completed package at contact",
            "post_contact_continuity": "continuity fails after the first force is broken",
            "production_feasibility": "the current producer allocation leaves the core force incomplete",
            "optimization_implication": "repair pre-contact production before changing later behavior",
            "failure_mode": "the army breaks in the first decisive engagement",
            "survival_prerequisite": "the opener usually survives until the planned change is active",
            "opponent_pressure_pattern": "pressure repeatedly arrives before the intended push",
            "matchup_assessment": "the assembled force lacks enough durable combat power at contact",
            "counterexample_check": "wins retain more force through the first contact",
            "covered_failures": [
                "Game 2 @ 430s: the army breaks at first contact",
                "Game 5 @ 510s: the incomplete army breaks under repeated pressure",
            ],
            "unexplained_failures": [],
            "counterexamples": [
                "Game 1 @ 620s: the completed force survives first contact"
            ],
        },
        "priority_alignment": {
            "selected_priority": "decisive combat viability",
            "higher_priority_assessment": "no higher-priority combat issue is better supported",
            "downstream_combat_effect": "the completed package improves first-engagement survival",
        },
        "retrieval_assessment": {
            "query_summary": "record, history, and static facts support the selected diagnosis",
            "match_evidence_used": [],
            "historical_experience_used": [],
            "knowledge_used": [],
            "conflicting_evidence": [],
            "confidence": "medium",
        },
        "mechanism_prediction": {
            "expected_change": "the selected army package is more complete before contact",
            "minimum_material_change": "the candidate must materially improve pre-contact completion",
            "outcome_prediction": "the first engagement becomes more competitive",
            "combat_success_measure": "first-engagement force retention improves",
            "disproof_condition": "completion materially improves but the same first-engagement failure persists",
        },
        "next_action": "propose_strategy_patch",
        "action_reason": "the first fight is repeatedly too weak",
        "plan": {
            "direction": "Build a second Factory before marine scaling.",
            "material_behavior_change": "field a materially more complete fighting package at first contact",
            "coordinated_changes": [
                {
                    "change": "shift production capacity toward the delayed core force",
                    "why_required": "the core force otherwise remains incomplete at contact",
                },
                {
                    "change": "retain enough early defense while production shifts",
                    "why_required": "the strategy must survive until the package is active",
                },
            ],
            "preserve": ["two-base opener"],
            "contact_window_effect": "similar",
            "new_hard_prerequisites": [],
            "production_tradeoffs": ["more Factory capacity before marine scaling"],
            "window_tradeoff_evidence": [],
            "why_window_remains_favorable": "the same window contains more of the intended core force",
            "preservation_checks": [
                {
                    "invariant": "two-base gathered-push structure",
                    "effect": "preserve",
                    "reason": "the economy and commitment style remain unchanged",
                    "evidence": ["Game 1 @ 620s"],
                }
            ],
            "stage_scope_evidence": [],
            "stage_scope_reason": (
                "The failure is at first contact, but the selected mechanism changes "
                "production completion rather than composition or retreat behavior."
            ),
            "strategy_area_audit": [
                {
                    "area": area,
                    "decision": "revise" if area == "production_order_capacity" else "preserve",
                    "finding": "the area was checked against the selected production mechanism",
                    "required_change": "increase staged core production throughput" if area == "production_order_capacity" else "",
                    "evidence": ["Game 2 @ 430s: incomplete package at contact"],
                }
                for area in (
                    "goal_identity",
                    "economy_expansion",
                    "production_order_capacity",
                    "technology_composition",
                    "attack_timing_objective",
                    "reinforcement_retreat_cleanup",
                )
            ],
        },
        "evidence_limits": [],
    }
    payload.update(overrides)
    return payload


def _package_proposal(**overrides) -> dict:
    base_plan = copy.deepcopy(_propose_decision()["plan"])
    second_plan = copy.deepcopy(base_plan)
    second_plan["direction"] = "Add one support unit without delaying the core push"
    second_plan["material_behavior_change"] = "The first push keeps core mass and adds one support unit"
    payload = {
        "strengths_to_preserve": [
            {"pattern": "two-base opener", "evidence": ["Game 1 @ 90s"]}
        ],
        "priority_problem": copy.deepcopy(_propose_decision()["priority_problem"]),
        "failure_mode_analysis": copy.deepcopy(
            _propose_decision()["failure_mode_analysis"]
        ),
        "parent_timing_package": {
            "economy": {
                "worker_target_before_commitment": 24,
                "base_target_before_commitment": 1,
                "gas_workers_before_commitment": 3,
            },
            "gate_components": [
                {"action": "train_marine", "quantity": 12, "production_slots": 2},
                {"action": "train_siege_tank", "quantity": 3, "production_slots": 1},
            ],
            "setup_actions": [
                {"action": "build_barracks", "quantity": 2, "parallel_slots": 1},
                {"action": "build_factory", "quantity": 1, "parallel_slots": 1},
            ],
        },
        "candidate_packages": [
            {
                "id": "P1",
                "hypothesis": "More Factory throughput reaches the same combat package sooner",
                "plan": base_plan,
                "timing_budget": {
                    "target_latest_first_commitment_seconds": 560,
                    "maximum_added_feasibility_seconds": 20,
                    "budget_basis": ["Game 2 @ 430s: late incomplete contact"],
                    "package": {
                        "economy": {
                            "worker_target_before_commitment": 24,
                            "base_target_before_commitment": 1,
                            "gas_workers_before_commitment": 3,
                        },
                        "gate_components": [
                            {"action": "train_marine", "quantity": 12, "production_slots": 2},
                            {"action": "train_siege_tank", "quantity": 3, "production_slots": 2},
                        ],
                        "setup_actions": [
                            {"action": "build_barracks", "quantity": 2, "parallel_slots": 1},
                            {"action": "build_factory", "quantity": 2, "parallel_slots": 2},
                        ],
                    },
                },
                "expected_effect": "Earlier complete contact",
                "main_risk": "Factory cost may reduce Marine throughput",
            },
            {
                "id": "P2",
                "hypothesis": "One support unit improves the decisive engagement",
                "plan": second_plan,
                "timing_budget": {
                    "target_latest_first_commitment_seconds": 620,
                    "maximum_added_feasibility_seconds": 80,
                    "budget_basis": ["Game 2 @ 430s: unsupported army loses contact"],
                    "package": {
                        "economy": {
                            "worker_target_before_commitment": 26,
                            "base_target_before_commitment": 1,
                            "gas_workers_before_commitment": 6,
                        },
                        "gate_components": [
                            {"action": "train_marine", "quantity": 12, "production_slots": 2},
                            {"action": "train_siege_tank", "quantity": 3, "production_slots": 1},
                            {"action": "train_medivac", "quantity": 1, "production_slots": 1},
                        ],
                        "setup_actions": [
                            {"action": "build_barracks", "quantity": 2, "parallel_slots": 1},
                            {"action": "build_factory", "quantity": 1, "parallel_slots": 1},
                            {"action": "build_starport", "quantity": 1, "parallel_slots": 1},
                        ],
                    },
                },
                "expected_effect": "Stronger first engagement",
                "main_risk": "Support tech may miss the contact window",
            },
        ],
        "next_action": "evaluate_candidate_packages",
        "action_reason": "Compare throughput and support hypotheses",
        "evidence_limits": [],
    }
    payload.update(overrides)
    return payload


def _parent_timing_extraction() -> dict:
    return {
        "parent_timing_package": {
            "economy": {
                "worker_target_before_commitment": None,
                "base_target_before_commitment": None,
                "gas_workers_before_commitment": None,
                "evidence": {},
            },
            "gate_components": [
                {
                    "action": "train_marine",
                    "quantity": 1,
                    "production_slots": 1,
                    "strategy_excerpt": "Marine and Tank force",
                }
            ],
            "setup_actions": [],
        },
        "requirement_coverage": [
            {
                "strategy_excerpt": "* Opening: Build workers, supply, production, gas, and technology toward fixed absolute targets.",
                "classification": "behavioral_pre_commitment",
                "mapped_to": [],
                "reason": "This test strategy does not provide numeric targets.",
            },
            {
                "strategy_excerpt": "* Main Attack Gate: Gather the persistent main force before attacking and send reinforcements toward the same objective.",
                "classification": "mixed",
                "mapped_to": ["gate_components.train_marine"],
                "reason": "The gate is pre-commitment and reinforcement is post-commitment.",
            },
        ],
    }


def _package_selection(package_id: str = "P1") -> dict:
    return {
        "selected_package_id": package_id,
        "candidate_diversity_assessment": {
            "is_diverse": True,
            "duplicate_groups": [],
            "reason": "The packages change different causal levers.",
        },
        "selected_history_assessment": {
            "semantic_relation": "new",
            "related_experiment_ids": [],
            "preserved_gain_ids": [],
            "repaired_dependencies": [],
            "reason": "No semantically equivalent historical intervention.",
            "confidence": "high",
        },
        "data_agent_assessment": {
            "considered_query_ids": [
                f"PKG_{package_id}_REQ",
                f"PKG_{package_id}_MATCHUP",
                f"PKG_{package_id}_UPGRADE",
            ],
            "supporting_findings": ["deterministic requirements were checked"],
            "contradicted_claims": [],
            "rejected_package_ids": [],
            "limitations": [],
        },
        "mechanism_prediction": copy.deepcopy(
            _propose_decision()["mechanism_prediction"]
        ),
        "next_action": "propose_strategy_patch",
        "action_reason": "best evidence-to-budget tradeoff",
        "evidence_limits": [],
    }


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
        "dataset_evidence": [
            {
                "tool": "get_strategy_knowledge",
                "result": {
                    "schema": KNOWLEDGE_VERIFICATION_SCHEMA,
                    "coverage": {
                        "unresolved_entities": [],
                        "unresolved_actions": [],
                        "unsupported_claims": [],
                        "complete": True,
                    },
                    "requested_calculation_count": 0,
                    "calculations": [],
                    "calculation_errors": [],
                    "missing": [],
                },
            }
        ],
    }


def test_intervention_scope_does_not_require_boolean_permission_or_two_references() -> None:
    decision = _propose_decision()
    decision["plan"]["stage_scope_evidence"] = [
        "Game 2 @ 430s: retreat fires after the main force is already broken"
    ]

    payload, error = _normalize_cross_match_decision(
        decision,
        strategy_name="tank",
        require_outcome_contract=True,
    )

    assert error == ""
    assert payload is not None
    assert "retreat_change_allowed" not in payload["plan"]
    assert "composition_change_allowed" not in payload["plan"]


def test_round2_prompt_reuses_discovery_findings() -> None:
    proposal_prompt = build_optimization_package_prompt(
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
        prior_experiences=[],
        retrieval_evidence={},
    )
    proposal, error = _normalize_candidate_package_proposal(_package_proposal())
    assert error == ""
    assert proposal is not None
    reports = _evaluate_candidate_package_budgets(proposal, race="terran")
    prompt = build_cross_match_decision_prompt(
        strategy_name="tank",
        race="terran",
        single_game_analyses=[],
        skill_texts={"strategy.md": VALID_STRATEGY},
        validation_errors=[],
        knowledge_mode="enabled",
        discovery={"unknowns": [{"unknown": "tank production cap"}]},
        knowledge_runs=[],
        candidate_package_payload=proposal,
        package_budget_reports=reports,
    )
    assert "Optimization-Package Planner" in proposal_prompt
    assert "two or three semantically distinct" in proposal_prompt
    assert "target_latest_first_commitment_seconds" in proposal_prompt
    assert "tank production cap" in proposal_prompt
    assert "fixed custom squads" in proposal_prompt
    assert "Composition, production, economy, technology" in proposal_prompt
    assert "post-contact continuation" in proposal_prompt
    assert "do not replay a failed direction" in proposal_prompt
    assert "earliest_feasible_time" in proposal_prompt
    assert "Optimization-Package Selector" in prompt
    assert "independent Optimization-Package Selector and semantic judge" in prompt
    assert "Program-calculated package budgets" in prompt
    assert "candidate_earliest_feasible_time_seconds" in prompt
    assert "selected_package_id" in prompt
    assert "regenerate_candidate_packages" in prompt
    assert "A `timing_risk` package requires direct trajectory evidence" in prompt
    assert "at most one material repair" in prompt


def test_package_preflight_marks_a_missed_time_budget() -> None:
    raw = _package_proposal()
    raw["candidate_packages"][0]["timing_budget"][
        "target_latest_first_commitment_seconds"
    ] = 1
    proposal, error = _normalize_candidate_package_proposal(raw)
    assert error == ""
    assert proposal is not None
    reports = _evaluate_candidate_package_budgets(proposal, race="terran")
    first = next(item for item in reports if item["id"] == "P1")
    assert first["target_latest_satisfied"] is False
    assert first["status"] == "timing_risk"


def test_candidate_packages_create_requirements_and_matchup_queries() -> None:
    proposal, error = _normalize_candidate_package_proposal(_package_proposal())
    assert error == ""
    assert proposal is not None

    questions = _candidate_package_knowledge_questions(proposal)

    p1_questions = [
        item for item in questions if "P1" in (item.get("plan_ids") or [])
    ]
    requirement = next(
        item
        for item in p1_questions
        if item["hypothesis_scope"] == "candidate_package_requirements"
    )
    assert "train_marine" in requirement["actions"]
    assert requirement["needs"] == ["requirements"]
    assert any(
        item.get("type") == "resource_demand_per_minute"
        for item in requirement["calculations"]
    )
    assert any(
        item["hypothesis_scope"] == "candidate_package_matchup"
        for item in p1_questions
    )
    p2_questions = [
        item for item in questions if "P2" in (item.get("plan_ids") or [])
    ]
    support = next(
        item
        for item in p2_questions
        if item["hypothesis_scope"] == "candidate_package_support"
    )
    assert support["needs"] == ["effects", "synergy"]


def test_package_preflight_reports_medivac_support_power() -> None:
    proposal, error = _normalize_candidate_package_proposal(_package_proposal())
    assert error == ""
    assert proposal is not None

    reports = _evaluate_candidate_package_budgets(proposal, race="terran")
    p1 = next(item for item in reports if item["id"] == "P1")
    p2 = next(item for item in reports if item["id"] == "P2")

    assert p1["candidate_support_aware_combat_estimate"]["support_bonus_power"] == 0
    assert p2["candidate_support_aware_combat_estimate"]["support_bonus_power"] > 0
    assert p2["support_aware_combat_delta"]["support_bonus_power"] > 0


def test_failed_package_requirement_query_marks_only_that_package_unresolved() -> None:
    reports = [
        {"id": "P1", "status": "feasible"},
        {"id": "P2", "status": "feasible"},
    ]
    failed_run = {
        "question_id": "PKG_P1_REQ",
        "plan_ids": ["P1"],
        "hypothesis_scope": "candidate_package_requirements",
        "ok": False,
        "error": "missing prerequisite action",
    }

    _annotate_package_knowledge(reports, [failed_run])

    assert reports[0]["status"] == "unresolved"
    assert reports[0]["knowledge_status"] == "unresolved"
    assert reports[1]["status"] == "feasible"


def test_package_preflight_joins_empirical_enemy_windows() -> None:
    proposal, error = _normalize_candidate_package_proposal(_package_proposal())
    assert error == ""
    assert proposal is not None
    summary = BattleAnalysis(
        strategy_name="tank",
        race="terran",
        sample_size=1,
        record_mix="0W/1L",
        raw={
            "result": "Defeat",
            "events": [
                {
                    "time_s": 430.0,
                    "own_state": {"army": "12 Marines, 3 Siege Tanks"},
                    "enemy_observed": {"army": "8 Marines"},
                    "enemy_truth": {
                        "army": "8 Marines, 4 Siege Tanks",
                        "technology": "Infantry Weapons 1",
                    },
                },
                {
                    "time_s": 560.0,
                    "own_state": {"army": "18 Marines, 5 Siege Tanks"},
                    "enemy_truth": {"army": "12 Marines, 7 Siege Tanks"},
                },
            ],
            "major_engagements": [
                {
                    "time_s": 570.0,
                    "own_force_before": "18 Marines, 5 Siege Tanks",
                    "enemy_truth": "12 Marines, 7 Siege Tanks",
                    "outcome": "army_broken",
                }
            ],
        },
    )

    reports = _evaluate_candidate_package_budgets(
        proposal,
        race="terran",
        summaries=[summary],
    )

    first = next(item for item in reports if item["id"] == "P1")
    window = first["empirical_opponent_windows"][0]
    assert window["result"] == "Defeat"
    assert window["candidate_window"]["enemy_truth"]["army"]
    assert window["first_engagement_at_or_after_candidate_window"]["outcome"] == (
        "army_broken"
    )


def test_shared_prompt_instructions_do_not_embed_a_specific_strategy() -> None:
    from evol_agent.core.prompts import SC2_STRATEGIC_PRIORITY

    instruction_text = SC2_STRATEGIC_PRIORITY.lower()
    for strategy_specific_term in (
        "marine",
        "siege tank",
        "heavy mech",
        "factory",
        "barracks",
        "engineering bay",
        "command center",
        "thor",
        "hellbat",
    ):
        assert strategy_specific_term not in instruction_text

    source = Path("evol_agent/core/prompts.py").read_text(encoding="utf-8").lower()
    assert "smallest coherent area" not in source
    assert '"commands": ["build_factory' not in source
    assert '"entities":["siege tank"]' not in source
    assert "marine/tank push" not in source


def test_parent_timing_extraction_requires_current_strategy_evidence() -> None:
    payload = _parent_timing_extraction()
    normalized, error = _normalize_parent_timing_package_extraction(
        payload,
        strategy_text=VALID_STRATEGY,
    )
    assert error == ""
    assert normalized is not None
    assert normalized["gate_components"][0]["action"] == "train_marine"

    bad = copy.deepcopy(payload)
    bad["parent_timing_package"]["gate_components"][0][
        "strategy_excerpt"
    ] = "20 Marines from four Barracks"
    normalized, error = _normalize_parent_timing_package_extraction(
        bad,
        strategy_text=VALID_STRATEGY,
    )
    assert normalized is None
    assert "gate component" in error


def test_parent_timing_extraction_rejects_omitted_strategy_bullets() -> None:
    payload = _parent_timing_extraction()
    payload["requirement_coverage"] = payload["requirement_coverage"][:1]

    normalized, error = _normalize_parent_timing_package_extraction(
        payload,
        strategy_text=VALID_STRATEGY,
    )

    assert normalized is None
    assert "omitted strategy bullets" in error


def test_parent_timing_extraction_coerces_zero_and_dict_slots() -> None:
    payload = _parent_timing_extraction()
    payload["parent_timing_package"]["gate_components"] = [
        {
            "action": "train_marine",
            "quantity": 45,
            "production_slots": {
                "build_barracks": 3,
                "build_barracks_reactor": 2,
                "build_barracks_techlab": 1,
            },
            "strategy_excerpt": "Marine and Tank force",
        }
    ]
    payload["parent_timing_package"]["setup_actions"] = [
        {
            "action": "build_gas",
            "quantity": 4,
            "parallel_slots": 0,
            "strategy_excerpt": "Opening: Build workers, supply, production, gas, and technology toward fixed absolute targets.",
        }
    ]
    payload["requirement_coverage"] = [
        {
            "strategy_excerpt": "* Opening: Build workers, supply, production, gas, and technology toward fixed absolute targets.",
            "classification": "mapped_pre_commitment",
            "mapped_to": ["setup_actions.build_gas"],
            "reason": "Refinery count is pre-commitment.",
        },
        {
            "strategy_excerpt": "* Main Attack Gate: Gather the persistent main force before attacking and send reinforcements toward the same objective.",
            "classification": "mixed",
            "mapped_to": ["gate_components.train_marine"],
            "reason": "The gate is pre-commitment and reinforcement is post-commitment.",
        },
    ]

    normalized, error = _normalize_parent_timing_package_extraction(
        payload,
        strategy_text=VALID_STRATEGY,
    )

    assert error == ""
    assert normalized is not None
    assert normalized["gate_components"][0]["production_slots"] == 6
    assert normalized["setup_actions"][0] == {
        "action": "build_gas",
        "quantity": 4,
        "parallel_slots": 1,
    }


def test_parent_timing_extraction_downgrades_empty_mixed_coverage() -> None:
    payload = _parent_timing_extraction()
    payload["requirement_coverage"] = [
        {
            "strategy_excerpt": "* Opening: Build workers, supply, production, gas, and technology toward fixed absolute targets.",
            "classification": "mixed",
            "mapped_to": [],
            "reason": "",
        },
        {
            "strategy_excerpt": "* Main Attack Gate: Gather the persistent main force before attacking and send reinforcements toward the same objective.",
            "classification": "mixed",
            "mapped_to": ["gate_components.train_marine"],
            "reason": "The gate is pre-commitment and reinforcement is post-commitment.",
        },
    ]

    normalized, error = _normalize_parent_timing_package_extraction(
        payload,
        strategy_text=VALID_STRATEGY,
    )

    assert error == ""
    assert normalized is not None


def test_parent_timing_extraction_matches_excerpts_without_bullet_marker() -> None:
    payload = _parent_timing_extraction()
    payload["requirement_coverage"] = [
        {
            "strategy_excerpt": "Opening: Build workers, supply, production, gas, and technology toward fixed absolute targets.",
            "classification": "behavioral_pre_commitment",
            "mapped_to": [],
            "reason": "This test strategy does not provide numeric targets.",
        },
        {
            "strategy_excerpt": "Main Attack Gate: Gather the persistent main force before attacking and send reinforcements toward the same objective.",
            "classification": "mixed",
            "mapped_to": ["gate_components.train_marine"],
            "reason": "The gate is pre-commitment and reinforcement is post-commitment.",
        },
    ]

    normalized, error = _normalize_parent_timing_package_extraction(
        payload,
        strategy_text=VALID_STRATEGY,
    )

    assert error == ""
    assert normalized is not None


def test_parent_timing_extraction_drops_absent_gas_mapping_for_mineral_only() -> None:
    payload = _parent_timing_extraction()
    payload["requirement_coverage"] = [
        {
            "strategy_excerpt": "* Opening: Build workers, supply, production, gas, and technology toward fixed absolute targets.",
            "classification": "mapped_pre_commitment",
            "mapped_to": ["setup_actions.build_gas"],
            "reason": "Mentions gas wording.",
        },
        {
            "strategy_excerpt": "* Main Attack Gate: Gather the persistent main force before attacking and send reinforcements toward the same objective.",
            "classification": "mixed",
            "mapped_to": ["gate_components.train_marine"],
            "reason": "The gate is pre-commitment and reinforcement is post-commitment.",
        },
    ]

    normalized, error = _normalize_parent_timing_package_extraction(
        payload,
        strategy_text=VALID_STRATEGY,
    )

    assert error == ""
    assert normalized is not None
    assert normalized["setup_actions"] == []


def test_parent_timing_extraction_skips_zero_qty_and_hallucinated_setup() -> None:
    payload = _parent_timing_extraction()
    payload["parent_timing_package"]["setup_actions"] = [
        {
            "action": "build_gas",
            "quantity": 0,
            "parallel_slots": 0,
            "strategy_excerpt": "build 0 Refineries",
        },
        {
            "action": "build_supply_depot",
            "quantity": 0,
            "parallel_slots": 0,
            "strategy_excerpt": "No explicit supply depot quantity is declared.",
        },
        {
            "action": "build_factory",
            "quantity": 2,
            "parallel_slots": 1,
            "strategy_excerpt": "invented factory line absent from strategy",
        },
    ]

    normalized, error = _normalize_parent_timing_package_extraction(
        payload,
        strategy_text=VALID_STRATEGY,
    )

    assert error == ""
    assert normalized is not None
    assert normalized["setup_actions"] == []


def test_parent_timing_package_cache_is_bound_to_strategy_hash(tmp_path: Path) -> None:
    checkpoint = EvolCheckpoint(tmp_path, {"stage": "created"})
    package = {
        "economy": {
            "worker_target_before_commitment": 44,
            "base_target_before_commitment": 2,
            "gas_workers_before_commitment": None,
        },
        "gate_components": [
            {"action": "train_siege_tank", "quantity": 10, "production_slots": 2}
        ],
        "setup_actions": [],
    }
    checkpoint.save_parent_timing_package(strategy_hash="strategy-a", package=package)

    assert checkpoint.load_parent_timing_package(strategy_hash="strategy-a") == package
    assert checkpoint.load_parent_timing_package(strategy_hash="strategy-b") is None


def test_candidate_planner_uses_canonical_parent_package() -> None:
    canonical = {
        "economy": {
            "worker_target_before_commitment": 44,
            "base_target_before_commitment": 2,
            "gas_workers_before_commitment": None,
        },
        "gate_components": [
            {"action": "train_siege_tank", "quantity": 10, "production_slots": 2}
        ],
        "setup_actions": [
            {"action": "build_factory", "quantity": 2, "parallel_slots": 2}
        ],
    }
    proposal = _package_proposal()
    proposal["parent_timing_package"] = {
        "economy": {
            "worker_target_before_commitment": 24,
            "base_target_before_commitment": 1,
            "gas_workers_before_commitment": 0,
        },
        "gate_components": [
            {"action": "train_marine", "quantity": 20, "production_slots": 4}
        ],
        "setup_actions": [],
    }
    normalized, error = _normalize_candidate_package_proposal(
        proposal,
        parent_timing_package=canonical,
    )
    assert error == ""
    assert normalized is not None
    assert normalized["parent_timing_package"] == canonical

    prompt = build_parent_timing_package_prompt(
        strategy_name="tank",
        race="terran",
        strategy_text=VALID_STRATEGY,
        validation_errors=[],
        capability_manifest={},
    )
    assert '"worker_target_before_commitment":24' not in prompt
    assert '"quantity":20' not in prompt


def test_cross_match_generates_preflights_and_selects_packages(monkeypatch) -> None:
    prompts: list[str] = []

    def fake_llm(prompt: str, **kwargs):
        prompts.append(prompt)
        if "Cross-Match Discovery Agent" in prompt:
            return _empty_discovery()
        if "Parent Strategy Package Extractor" in prompt:
            return _parent_timing_extraction()
        if "Optimization-Package Planner" in prompt:
            return _package_proposal()
        assert "Optimization-Package Selector" in prompt
        assert "candidate_earliest_feasible_time_seconds" in prompt
        return _package_selection()

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
    assert len(prompts) == 4
    assert complete["llm_cross_match_calls"] == 4
    assert result.battle_analysis.raw["plan"]["direction"].startswith("Build a second Factory")
    assert result.battle_analysis.raw["selected_package_id"] == "P1"
    assert len(result.battle_analysis.raw["package_budget_reports"]) == 2
    assert result.battle_analysis.raw["selected_package_budget"][
        "candidate_earliest_feasible_time_seconds"
    ]
    extracted = extract_final_cross_match_decision(result.battle_analysis)
    assert extracted["selected_package_id"] == "P1"
    assert extracted["selected_timing_budget"][
        "target_latest_first_commitment_seconds"
    ] == 560
    assert any(item.get("action") == "cross_match_discovery" for item in result.events)
    assert any(
        item.get("action") == "optimization_package_preflight"
        for item in result.events
    )
    assert any(item.get("action") == "cross_match_decision" for item in result.events)


def test_selector_rejection_regenerates_packages_before_strategy_edit(monkeypatch) -> None:
    planner_calls = 0
    selector_calls = 0

    def fake_llm(prompt: str, **kwargs):
        nonlocal planner_calls, selector_calls
        if "Cross-Match Discovery Agent" in prompt:
            return _empty_discovery()
        if "Parent Strategy Package Extractor" in prompt:
            return _parent_timing_extraction()
        if "Optimization-Package Planner" in prompt:
            planner_calls += 1
            return _package_proposal()
        assert "Optimization-Package Selector" in prompt
        selector_calls += 1
        if selector_calls == 1:
            return {
                "selected_package_id": "",
                "candidate_diversity_assessment": {
                    "is_diverse": False,
                    "duplicate_groups": [["P1", "P2"]],
                    "reason": "Both packages change the same attack gate.",
                },
                "next_action": "regenerate_candidate_packages",
                "action_reason": "Generate a different causal lever.",
                "evidence_limits": [],
            }
        return _package_selection()

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
        knowledge_mode="disabled",
    )
    assert result.completed is True
    assert planner_calls == 2
    assert selector_calls == 2
    assert any(
        item.get("action") == "regenerate_candidate_packages"
        for item in result.events
    )


def test_cross_match_queries_discovery_and_candidate_packages_then_decides(monkeypatch) -> None:
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
        if "Parent Strategy Package Extractor" in prompt:
            return _parent_timing_extraction()
        assert KNOWLEDGE_VERIFICATION_SCHEMA in prompt
        if "Optimization-Package Planner" in prompt:
            return _package_proposal()
        assert "Optimization-Package Selector" in prompt
        return _package_selection()

    def fake_knowledge(questions, **kwargs):
        knowledge_calls.append(questions)
        runs = []
        for question in questions:
            run = copy.deepcopy(_verified_knowledge_run())
            run["question_id"] = question.get("id")
            run["question"] = question.get("question")
            run["query_reason"] = question.get("query_reason")
            run["evidence_refs"] = question.get("evidence_refs") or []
            run["hypothesis_scope"] = question.get("hypothesis_scope")
            run["plan_ids"] = question.get("plan_ids") or []
            runs.append(run)
        return runs

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
    assert len(prompts) == 4
    assert len(knowledge_calls) == 2
    assert any(
        question.get("hypothesis_scope") == "candidate_package_requirements"
        for question in knowledge_calls[1]
    )
    assert complete["llm_cross_match_calls"] == 4
    assert result.battle_analysis.raw["next_action"] == "propose_strategy_patch"
    assert result.knowledge_trace["discovery"]["knowledge_questions"][0]["id"] == "Q1"


def test_package_planner_can_attribute_problem_to_runtime(monkeypatch) -> None:
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
        if "Parent Strategy Package Extractor" in prompt:
            return _parent_timing_extraction()
        assert KNOWLEDGE_VERIFICATION_SCHEMA in prompt
        return {
            "strengths_to_preserve": [{"pattern": "two-base opener", "evidence": ["Game 1 @ 90s"]}],
            "priority_problem": {
                "problem": "Tank production is too late",
                "evidence": ["Game 2 @ 430s"],
                "control_class": "observation_limited",
            },
            "failure_mode_analysis": {},
            "parent_timing_package": {},
            "candidate_packages": [],
            "next_action": "inspect_runtime",
            "action_reason": "knowledge shows earlier tanks were infeasible; more games are needed",
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
    assert result.battle_analysis.raw["next_action"] == "inspect_runtime"
    assert result.battle_analysis.raw["plan"] is None


def test_inspect_runtime_does_not_need_a_plan(monkeypatch) -> None:
    def fake_llm(prompt: str, **kwargs):
        if "Cross-Match Discovery Agent" in prompt:
            return _empty_discovery()
        if "Parent Strategy Package Extractor" in prompt:
            return _parent_timing_extraction()
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
    assert "candidate_plans" not in result.battle_analysis.raw


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
        if "Parent Strategy Package Extractor" in prompt:
            return _parent_timing_extraction()
        assert KNOWLEDGE_VERIFICATION_SCHEMA in prompt
        if "Optimization-Package Planner" in prompt:
            return _package_proposal()
        assert "Optimization-Package Selector" in prompt
        return _package_selection()

    def package_knowledge(question, **kwargs):
        question_id = str(question.get("id") or "")
        assert question_id != "Q1", "cached discovery knowledge should be reused"
        knowledge_llm_calls.append(question_id)
        run = copy.deepcopy(_verified_knowledge_run())
        run["question_id"] = question_id
        run["question"] = question.get("question")
        run["query_reason"] = question.get("query_reason")
        run["evidence_refs"] = question.get("evidence_refs") or []
        run["hypothesis_scope"] = question.get("hypothesis_scope")
        run["plan_ids"] = question.get("plan_ids") or []
        run["query"] = "package-specific deterministic query"
        return run

    monkeypatch.setattr("evol_agent.core.analysis_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.analysis_agent_loop._summarize_matches",
        lambda **kwargs: _stub_summaries(),
    )
    monkeypatch.setattr(
        "evol_agent.core.analysis_agent_loop.run_knowledge_query",
        package_knowledge,
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
    assert len(prompts) == 3
    assert knowledge_llm_calls
    assert result.battle_analysis.raw["next_action"] == "propose_strategy_patch"


def test_rejected_experiments_are_visible_in_round2(monkeypatch) -> None:
    prompts: list[str] = []
    prior = [
        {
            "primary_lever": "attack_timing",
            "hypothesis": "attack earlier with the same composition",
            "delta": -0.42,
            "lesson": "same primary lever produced significant score regression",
            "implementation_verdict": "unknown",
            "hypothesis_verdict": "inconclusive",
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
        if "Parent Strategy Package Extractor" in prompt:
            return _parent_timing_extraction()
        if "Optimization-Package Planner" in prompt:
            return _package_proposal()
        return _package_selection()

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
    planner_prompt = next(
        prompt for prompt in prompts if "Optimization-Package Planner" in prompt
    )
    assert "attack_timing" in planner_prompt
    assert "same primary lever produced significant score regression" in planner_prompt
    assert "candidate_minus_parent" in planner_prompt
    assert '"hypothesis_verdict":"inconclusive"' in planner_prompt
