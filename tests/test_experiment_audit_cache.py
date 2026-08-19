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
