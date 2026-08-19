from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from evol_agent.core import experiment_audit
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
    first_cache = experiment_audit._MatchSummaryCache(cache_path)
    first = experiment_audit._summarize_records(
        [record],
        strategy_name="tank",
        race="terran",
        model="test-model",
        label="parent",
        cache=first_cache,
    )
    second_cache = experiment_audit._MatchSummaryCache(cache_path)
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
