from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

from evol_agent.core import experiment_audit
from evol_agent.core.match_summary_cache import MatchSummaryCache
from evol_agent.core.types import GameEvidence


def _record(path: Path) -> GameEvidence:
    path.write_text('{"metadata": {"result": "Victory"}}', encoding="utf-8")
    return GameEvidence(
        file=str(path),
        result="Victory",
        duration="01:00",
        timeline="",
        meta={},
    )


def test_experiment_audit_reuses_persistent_match_summary_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    record = _record(tmp_path / "match.json")
    cache_path = tmp_path / "summary_cache.json"
    calls: list[str] = []

    def summarize(**kwargs):
        calls.append(kwargs["record"].file)
        return (
            None,
            SimpleNamespace(raw={"result": "cached-summary"}),
            True,
            [],
            [],
        )

    monkeypatch.setattr(experiment_audit, "run_fixed_match_summary", summarize)
    first_cache = MatchSummaryCache(cache_path)
    first = experiment_audit._summarize_records(
        [record],
        strategy_name="tank",
        race="terran",
        model="test-model",
        label="parent",
        cache=first_cache,
    )
    assert (
        MatchSummaryCache(cache_path).get(
            record,
            strategy_name="tank",
            race="terran",
            model="different-model",
        )
        is None
    )
    second_cache = MatchSummaryCache(cache_path)
    second = experiment_audit._summarize_records(
        [record],
        strategy_name="tank",
        race="terran",
        model="test-model",
        label="parent",
        cache=second_cache,
    )

    assert calls == [record.file]
    assert first == second
    assert second[0]["summary"] == {"result": "cached-summary"}


def test_focused_audit_summaries_are_reused_by_later_generic_analysis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    record = _record(tmp_path / "match.json")
    cache_path = tmp_path / "summary_cache.json"
    calls: list[str] = []
    focus = {"expected_change": "add four Vikings to the attack gate"}

    def summarize(**kwargs):
        calls.append(kwargs["record"].file)
        return (
            SimpleNamespace(raw={"summary": "digest"}),
            SimpleNamespace(raw={"result": "focused-summary"}),
            True,
            [],
            [],
        )

    monkeypatch.setattr(experiment_audit, "run_fixed_match_summary", summarize)
    cache = MatchSummaryCache(cache_path)
    focused = experiment_audit._summarize_records(
        [record],
        strategy_name="tank_opt3",
        race="terran",
        model="test-model",
        label="candidate",
        cache=cache,
        audit_focus=focus,
    )
    generic_cache = MatchSummaryCache(cache_path)
    reused = generic_cache.get(
        record,
        strategy_name="tank_opt3",
        race="terran",
        model="test-model",
    )
    later_generic = experiment_audit._summarize_records(
        [record],
        strategy_name="tank_opt3",
        race="terran",
        model="test-model",
        label="analysis",
        cache=generic_cache,
    )
    later_same_focus = experiment_audit._summarize_records(
        [record],
        strategy_name="tank_opt3",
        race="terran",
        model="test-model",
        label="candidate",
        cache=MatchSummaryCache(cache_path),
        audit_focus=focus,
    )

    assert calls == [record.file]
    assert focused[0]["summary"] == {"result": "focused-summary"}
    assert reused is not None
    assert reused["summary"] == {"result": "focused-summary"}
    assert later_generic == focused
    assert later_same_focus == focused


def test_focused_audit_does_not_reuse_a_generic_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    record = _record(tmp_path / "match.json")
    cache_path = tmp_path / "summary_cache.json"
    calls: list[str] = []

    def summarize(**kwargs):
        calls.append(str(kwargs.get("audit_focus") or {}))
        payload = "focused" if kwargs.get("audit_focus") else "generic"
        return (
            None,
            SimpleNamespace(raw={"result": payload}),
            True,
            [],
            [],
        )

    monkeypatch.setattr(experiment_audit, "run_fixed_match_summary", summarize)
    cache = MatchSummaryCache(cache_path)
    generic = experiment_audit._summarize_records(
        [record],
        strategy_name="tank_opt3",
        race="terran",
        model="test-model",
        label="parent",
        cache=cache,
    )
    focused = experiment_audit._summarize_records(
        [record],
        strategy_name="tank_opt3",
        race="terran",
        model="test-model",
        label="candidate",
        cache=MatchSummaryCache(cache_path),
        audit_focus={"material_behavior_change": "research Stimpack"},
    )

    assert calls == ["{}", "{'material_behavior_change': 'research Stimpack'}"]
    assert generic[0]["summary"] == {"result": "generic"}
    assert focused[0]["summary"] == {"result": "focused"}


def test_experiment_audit_compacts_prior_parent_analysis() -> None:
    compact = experiment_audit._compact_parent_analysis(
        {
            "strategy_name": "tank",
            "record_mix": "5W/5L",
            "priority_problem": {"problem": "tank timing"},
            "hypothesis": "reach the power spike",
            "candidate_plans": ["large duplicated plan"],
            "retrieval_evidence": ["large duplicated evidence"],
        }
    )

    assert compact == {
        "strategy_name": "tank",
        "record_mix": "5W/5L",
        "priority_problem": {"problem": "tank timing"},
        "hypothesis": "reach the power spike",
    }


def test_experiment_audit_summarizes_only_parent_records_added_after_analysis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parent_dir = tmp_path / "parent"
    candidate_dir = tmp_path / "candidate"
    parent_dir.mkdir()
    candidate_dir.mkdir()

    def write_record(path: Path, *, strategy: str, result: str) -> Path:
        path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "strategy_id": strategy,
                        "result": result,
                        "game_duration_formatted": "10:00",
                        "save_reason": "match_runner_finally",
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    old_parent = write_record(
        parent_dir / "old_parent.json", strategy="tank_opt2", result="Defeat"
    )
    new_parent = write_record(
        parent_dir / "new_parent.json", strategy="tank_opt2", result="Victory"
    )
    candidate = write_record(
        candidate_dir / "candidate.json", strategy="tank_opt3", result="Defeat"
    )
    summarized: list[str] = []
    prompts: list[str] = []

    def summarize(**kwargs):
        summarized.append(kwargs["record"].file)
        return (
            None,
            SimpleNamespace(raw={"source": Path(kwargs["record"].file).name}),
            True,
            [],
            [],
        )

    def audit_llm(prompt, **_kwargs):
        prompts.append(prompt)
        return {
            "implementation_verdict": "implemented",
            "hypothesis_verdict": "inconclusive",
            "mechanism_evidence": [],
            "combat_evidence": [],
            "runtime_findings": [],
            "evidence_limits": [],
            "lesson": "retain verified improvements",
        }

    monkeypatch.setattr(experiment_audit, "run_fixed_match_summary", summarize)
    monkeypatch.setattr(experiment_audit, "call_json_llm", audit_llm)
    kwargs = {
        "race": "terran",
        "parent_strategy_name": "tank_opt2",
        "candidate_strategy_name": "tank_opt3",
        "parent_strategy": "parent strategy",
        "candidate_strategy": "candidate strategy",
        "parent_batch_dirs": [parent_dir],
        "candidate_batch_dirs": [candidate_dir],
        "experiment_spec": {"hypothesis": "test"},
        "outcome_comparison": {"outcome": "reject"},
        "model": "test-model",
        "summary_cache_path": tmp_path / "summary_cache.json",
        "parent_analysis": {
            "strategy_name": "tank_opt2",
            "record_mix": "2W/8L",
            "hypothesis": "prior parent analysis",
        },
        "parent_analysis_record_paths": [str(old_parent)],
    }

    experiment_audit.audit_experiment(**kwargs)

    assert summarized == [str(new_parent), str(candidate)]
    assert "new_parent.json" in prompts[0]
    assert "old_parent.json" not in prompts[0]

    experiment_audit.audit_experiment(**kwargs)

    assert summarized == [str(new_parent), str(candidate)]


def test_gate_execution_audit_detects_repeated_gate_met_without_attack(
    tmp_path: Path,
) -> None:
    records: list[GameEvidence] = []
    for game in range(2):
        path = tmp_path / f"candidate_{game}.json"
        path.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "game_time_seconds": 100.0,
                            "observation_full": {
                                "economy": {"workers": 20, "own_base_count": 1},
                                "own_forces": {"completed_counts": {"MARINE": 20}},
                                "production": {"completed": {"MARINE": 20}},
                                "technology": {"completed_upgrades": []},
                                "army_control": {"current_commands": []},
                            },
                        },
                        {
                            "game_time_seconds": 160.0,
                            "observation_full": {
                                "economy": {"workers": 20, "own_base_count": 1},
                                "own_forces": {"completed_counts": {"MARINE": 24}},
                                "production": {"completed": {"MARINE": 24}},
                                "technology": {"completed_upgrades": []},
                                "army_control": {"current_commands": []},
                            },
                        },
                    ],
                    "interactions": [
                        {"agent": "commander", "accepted": True, "game_time": 100.0, "army_policy": {"commands": [{"movement_mode": "hold"}]}},
                        {"agent": "commander", "accepted": True, "game_time": 160.0, "army_policy": {"commands": [{"movement_mode": "hold"}]}},
                    ],
                }
            ),
            encoding="utf-8",
        )
        records.append(
            GameEvidence(
                file=str(path),
                result="Defeat",
                duration="05:00",
                timeline="",
                meta={},
            )
        )
    audit = experiment_audit._audit_gate_execution(
        records,
        experiment_spec={
            "first_commitment_timing": {
                "candidate_earliest_feasible_time_seconds": 90.0,
                "declared_packages": {
                    "candidate": {
                        "economy": {
                            "worker_target_before_commitment": 20,
                            "base_target_before_commitment": 1,
                        },
                        "gate_components": [
                            {"action": "train_marine", "quantity": 20}
                        ],
                    }
                },
            }
        },
    )

    assert audit["status"] == "execution_issue"
    assert audit["execution_issue_matches"] == 2
    assert all(
        item["verdict"] == "gate_met_no_commitment" for item in audit["matches"]
    )
    normalized = experiment_audit._normalize_audit(
        {
            "implementation_verdict": "implemented",
            "hypothesis_verdict": "supported",
        },
        gate_execution_audit=audit,
    )
    assert normalized["implementation_verdict"] == "execution_invalid"
    assert normalized["hypothesis_verdict"] == "not_tested"


def test_upgrade_gate_matches_runtime_upgrade_ids() -> None:
    snapshot = {
        "observation_full": {
            "technology": {
                "completed_upgrades": [
                    "TERRANINFANTRYWEAPONSLEVEL1",
                    "PUNISHERGRENADES",
                    "HIGHCAPACITYBARRELS",
                    "LIBERATORAGRANGEUPGRADE",
                ]
            }
        }
    }

    assert experiment_audit._upgrade_completed(
        snapshot, "research_infantry_weapons_1"
    )
    assert experiment_audit._upgrade_completed(
        snapshot, "research_concussive_shells"
    )
    assert experiment_audit._upgrade_completed(
        snapshot, "research_infernal_preigniter"
    )
    assert experiment_audit._upgrade_completed(
        snapshot, "research_liberator_range"
    )
