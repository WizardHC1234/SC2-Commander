from __future__ import annotations

import json
from pathlib import Path

import pytest

from evol_agent.analysis.optimization_direction_audit import (
    audit_direction,
    build_audit_context,
    build_direction_audit_prompt,
    normalize_audit_result,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _valid_result() -> dict:
    return {
        "verdict": "reject_repeated_direction",
        "root_cause": {
            "primary": "production_readiness",
            "reason": "The force was absent before contact.",
            "evidence_refs": ["Game 5 @ 535s"],
        },
        "behavioral_change": {
            "pre_contact": {
                "changes_observable_trajectory": False,
                "description": "Only changes the holding location.",
                "deadline_supported": False,
            },
            "engagement": {
                "changes_observable_trajectory": False,
                "description": "No matchup or target change.",
            },
            "post_contact": {
                "changes_observable_trajectory": False,
                "description": "Still waits for the full gate.",
            },
        },
        "history_comparison": {
            "semantic_family": "passive pre-gate defense",
            "equivalent_rejected_experiment_ids": ["tank:g006:cheatvision:tank_opt7"],
            "why_new_or_repeated": "The observable behavior is unchanged.",
        },
        "preserved_gain_experiment_ids": ["tank:g005:cheatvision:tank_opt6"],
        "blocking_reasons": ["No earlier force formation."],
        "recommended_revision": {
            "change": "Change production realization before the contact window.",
            "preserve": ["Medivac support"],
            "avoid": ["another passive holding requirement"],
        },
        "confidence": "high",
    }


def test_context_is_read_only_and_resolves_latest_child(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    analysis_dir = tmp_path / "analysis"
    _write_json(
        run_dir / "state.json",
        {
            "status": "completed",
            "style": "tank",
            "champion": "tank_opt6",
            "difficulty": "cheatvision",
            "generation": 10,
            "experiment_history": [
                {
                    "experiment_id": "tank:g009:cheatvision:tank_opt10",
                    "mutation_parent": "tank_opt6",
                    "candidate": "tank_opt10",
                    "decision": "rejected",
                    "score_delta": -0.1,
                    "primary_change": "Add a defensive core.",
                }
            ],
        },
    )
    _write_json(
        analysis_dir / "analysis.json",
        {
            "strategy_name": "tank_opt6",
            "selected_package_id": "P1",
            "candidate_packages": [{"id": "P1"}],
        },
    )
    _write_json(analysis_dir / "cross_match_discovery.json", {"weaknesses": [{"pattern": "early pressure"}]})
    for name in ("tank_opt6", "tank_opt10"):
        path = run_dir / "strategies" / name / "strategy.md"
        path.parent.mkdir(parents=True)
        path.write_text(f"# {name}", encoding="utf-8")

    before = (run_dir / "state.json").read_bytes()
    context = build_audit_context(run_dir=run_dir, analysis_dir=analysis_dir)

    assert context["parent_strategy_name"] == "tank_opt6"
    assert context["candidate_strategy_name"] == "tank_opt10"
    assert context["candidate_strategy_md"] == "# tank_opt10"
    assert context["experiment_history"] == []
    assert (run_dir / "state.json").read_bytes() == before


def test_prompt_requires_semantic_history_and_no_wake_events() -> None:
    prompt = build_direction_audit_prompt({"analysis": {"selected_package_id": "P1"}})
    assert "Judge semantic equivalence with reasoning, not character matching" in prompt
    assert "pre_contact" in prompt and "post_contact" in prompt
    assert "do not require new wake events" in prompt.lower()
    assert "must not edit any file or evolution state" in prompt


def test_normalize_audit_result_requires_behavior_booleans() -> None:
    payload = normalize_audit_result(_valid_result())
    assert payload["verdict"] == "reject_repeated_direction"
    assert payload["root_cause"]["primary"] == "production_readiness"
    assert payload["behavioral_change"]["pre_contact"]["deadline_supported"] is False

    invalid = _valid_result()
    del invalid["behavioral_change"]["engagement"]["changes_observable_trajectory"]
    with pytest.raises(ValueError, match="engagement"):
        normalize_audit_result(invalid)


def test_audit_direction_uses_reasoning_model_without_writing() -> None:
    calls: list[dict] = []

    def fake_llm(prompt: str, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return _valid_result()

    result = audit_direction(
        context={"analysis": {"selected_package_id": "P1"}},
        model="test-model",
        llm_call=fake_llm,
    )

    assert result["verdict"] == "reject_repeated_direction"
    assert calls[0]["model"] == "test-model"
    assert calls[0]["is_reasoning"] is True
