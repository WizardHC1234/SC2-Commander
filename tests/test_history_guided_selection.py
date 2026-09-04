from __future__ import annotations

from evol_agent.core.analysis_agent_loop import _normalize_package_selection
from evol_agent.core.prompts import _experiment_history_scorecard


def _proposal() -> dict:
    return {
        "candidate_packages": [
            {
                "id": "P1",
                "hypothesis": "Add Marauders to the Marine-Tank attack.",
                "plan": {"direction": "Add Marauders."},
                "timing_budget": {},
                "engagement_assessment": {},
            }
        ]
    }


def _direction_audit(verdict: str) -> dict:
    return {
        "verdict": verdict,
        "root_cause": {
            "primary": "attack_gate_timing",
            "reason": "The rejected direction keeps delaying contact.",
            "evidence_refs": ["Game 3 @ 600s"],
        },
        "behavioral_change": {
            "pre_contact": {
                "changes_observable_trajectory": True,
                "description": "Contact is delayed for a larger package.",
                "deadline_supported": False,
            },
            "engagement": {
                "changes_observable_trajectory": False,
                "description": "The same engagement remains.",
            },
            "post_contact": {
                "changes_observable_trajectory": False,
                "description": "Continuation remains unchanged.",
            },
        },
        "history_comparison": {
            "semantic_family": "delay_for_larger_pre_attack_package",
            "equivalent_rejected_experiment_ids": [],
            "why_new_or_repeated": "The package repeats the same trajectory.",
        },
        "preserved_gain_experiment_ids": [],
        "blocking_reasons": ["The later window is not supported."],
        "recommended_revision": {
            "change": "Use a different causal lever.",
            "preserve": [],
            "avoid": ["another larger opening gate"],
        },
        "confidence": "high",
    }


def test_scorecard_separates_proven_gain_from_regression() -> None:
    rows = _experiment_history_scorecard(
        [
            {
                "experiment_id": "tank:g000:harder:tank_opt1",
                "decision": "accepted",
                "implementation_verdict": "implemented",
                "score_delta": 0.2,
            },
            {
                "experiment_id": "tank:g005:harder:tank_opt6",
                "decision": "rejected",
                "implementation_verdict": "implemented",
                "score_delta": -0.2,
            },
        ]
    )
    assert [row["outcome"] for row in rows] == ["proven_gain", "regression"]


def test_scorecard_does_not_treat_unrealized_score_gain_as_proven() -> None:
    rows = _experiment_history_scorecard(
        [
            {
                "experiment_id": "marine:g004:cheatvision:marine_opt5",
                "decision": "accepted",
                "implementation_verdict": "underpowered",
                "hypothesis_verdict": "not_tested",
                "score_delta": 0.4,
            }
        ]
    )
    assert rows[0]["outcome"] == "performance_gain_unverified"


def test_selector_cannot_choose_semantic_replay_of_failed_package() -> None:
    prior = [
        {
            "experiment_id": "tank:g005:harder:tank_opt6",
            "decision": "rejected",
            "implementation_verdict": "implemented",
            "hypothesis_verdict": "contradicted",
            "score_delta": -0.2,
        }
    ]
    raw = {
        "next_action": "propose_strategy_patch",
        "selected_package_id": "P1",
        "candidate_diversity_assessment": {
            "is_diverse": True,
            "duplicate_groups": [],
            "reason": "The proposed packages use different causal levers.",
        },
        "selected_history_assessment": {
            "semantic_relation": "repeats_failed",
            "related_experiment_ids": ["tank:g005:harder:tank_opt6"],
            "preserved_gain_ids": [],
            "repaired_dependencies": [],
            "reason": "Same Marauder gate, timing, and causal prediction.",
            "confidence": "high",
        },
    }
    payload, error = _normalize_package_selection(
        raw,
        proposal=_proposal(),
        package_budget_reports=[{"id": "P1", "status": "feasible"}],
        strategy_name="tank",
        fallback_strategy_contract={},
        fallback_outcome_contrast={},
        prior_experiences=prior,
    )
    assert payload is None
    assert "cannot repeat a failed historical intervention" in error


def test_selector_can_reject_a_semantically_duplicate_package_set() -> None:
    payload, error = _normalize_package_selection(
        {
            "next_action": "regenerate_candidate_packages",
            "selected_package_id": "",
            "candidate_diversity_assessment": {
                "is_diverse": False,
                "duplicate_groups": [["P1", "P2"]],
                "reason": "Both packages delay the same attack for the same support unit.",
            },
            "action_reason": "No distinct candidate is selectable.",
        },
        proposal={
            "candidate_packages": [
                {"id": "P1", "hypothesis": "A", "plan": {}},
                {"id": "P2", "hypothesis": "B", "plan": {}},
            ]
        },
        package_budget_reports=[],
        strategy_name="tank",
        fallback_strategy_contract={},
        fallback_outcome_contrast={},
        prior_experiences=[],
    )
    assert error == ""
    assert payload is not None
    assert payload["next_action"] == "regenerate_candidate_packages"
    assert payload["candidate_diversity_assessment"]["is_diverse"] is False


def test_direction_audit_revision_regenerates_before_trial() -> None:
    raw = {
        "next_action": "propose_strategy_patch",
        "selected_package_id": "P1",
        "candidate_diversity_assessment": {
            "is_diverse": True,
            "duplicate_groups": [],
            "reason": "The packages use different causal levers.",
        },
        "selected_history_assessment": {
            "semantic_relation": "new",
            "related_experiment_ids": [],
            "preserved_gain_ids": [],
            "repaired_dependencies": [],
            "reason": "No matching prior experiment.",
            "confidence": "high",
        },
        "direction_audit": _direction_audit("revise_before_trial"),
    }

    payload, error = _normalize_package_selection(
        raw,
        proposal=_proposal(),
        package_budget_reports=[{"id": "P1", "status": "feasible"}],
        strategy_name="tank",
        fallback_strategy_contract={},
        fallback_outcome_contrast={},
        prior_experiences=[],
    )

    assert error == ""
    assert payload is not None
    assert payload["next_action"] == "regenerate_candidate_packages"
    assert "later window" in payload["action_reason"]


def test_selector_allows_only_one_material_repair_for_same_failed_direction() -> None:
    prior = [
        {
            "experiment_id": "marine:g001:harder:marine_opt1",
            "decision": "rejected",
            "score_delta": -0.2,
        },
        {
            "experiment_id": "marine:g002:harder:marine_opt2",
            "decision": "rejected",
            "score_delta": -0.1,
            "selected_history_assessment": {
                "semantic_relation": "material_repair",
                "related_experiment_ids": ["marine:g001:harder:marine_opt1"],
            },
        },
    ]
    raw = {
        "next_action": "propose_strategy_patch",
        "selected_package_id": "P1",
        "candidate_diversity_assessment": {
            "is_diverse": True,
            "duplicate_groups": [],
            "reason": "Distinct from the other proposal.",
        },
        "selected_history_assessment": {
            "semantic_relation": "material_repair",
            "related_experiment_ids": ["marine:g001:harder:marine_opt1"],
            "preserved_gain_ids": [],
            "repaired_dependencies": ["producer capacity"],
            "reason": "Repair the prior dependency again.",
            "confidence": "high",
        },
    }
    payload, error = _normalize_package_selection(
        raw,
        proposal=_proposal(),
        package_budget_reports=[{"id": "P1", "status": "feasible"}],
        strategy_name="marine",
        fallback_strategy_contract={},
        fallback_outcome_contrast={},
        prior_experiences=prior,
    )
    assert payload is None
    assert "already used its one material-repair attempt" in error
