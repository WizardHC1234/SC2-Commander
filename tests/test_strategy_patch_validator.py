from __future__ import annotations

from pathlib import Path

from evol_agent.core.capabilities import build_executor_capability_manifest
from evol_agent.core.optimization_agent_loop import (
    _candidate_knowledge_run,
    _patches_to_operations,
    run_optimization_agent_loop,
)
from evol_agent.core.optimizer_prompt import build_candidate_prompt
from evol_agent.core.strategy_patch_validator import (
    _build_contact_timing_report,
    _blocking_semantic_errors,
    _normalize_mechanism_equivalence_audit,
    build_strategy_patch_validation_prompt,
    validate_strategy_patch_semantics,
    validate_strategy_patch_structure,
)
from evol_agent.sc2_data_agent.bridge import run_knowledge_query
from evol_agent.core.types import BattleAnalysis
from evol_agent.optimization.strategy_document import StrategyDocument, paragraph_hash


TANK_STRATEGY = Path("skills/terran/tank/strategy.md").read_text(encoding="utf-8")


def _semantic_payload(*, valid: bool = True, errors: list | None = None) -> dict:
    return {
        "valid": valid,
        "production_target_audit": [
            {
                "unit": "Marine",
                "instruction": "continue Marine production",
                "stage_target": "40 Marines",
                "ultimate_goal_target": "75 Marines",
                "temporary_stop_rule": "",
                "verdict": "bounded",
            }
        ],
        "new_dependency_audit": [],
        "final_supply": {
            "total": 119,
            "calculation": "75 Marines plus 44 SCVs equals 119 supply",
            "verdict": "valid",
        },
        "style_and_window_audit": {
            "parent_combat_style": "concentrated timing attack",
            "candidate_combat_style": "concentrated timing attack",
            "style_preserved": True,
            "contact_window_effect": "earlier",
            "window_change_justified": True,
            "new_hard_prerequisites": [],
            "shared_production_tradeoffs": [],
            "hidden_attack_gate": False,
            "verdict": "evidence_supported_shift",
        },
        "mechanism_history_audit": {
            "semantic_relation": "no_prior",
            "related_experiment_ids": [],
            "repaired_dependencies": [],
            "verdict": "allowed",
        },
        "errors": list(errors or []),
    }


def _decision(**overrides) -> dict:
    payload = {
        "hypothesis": "The first attack readiness requirement is too demanding.",
        "mechanism_prediction": {
            "expected_change": "the intended commitment occurs earlier",
            "minimum_material_change": "the readiness rule must materially change commitment timing",
            "outcome_prediction": "the useful pressure window is reached more often",
            "disproof_condition": "commitment occurs materially earlier but the failure persists",
        },
        "priority_problem": {
            "problem": "The first planned push starts after the useful window.",
            "evidence": ["Game 2 @ 430s"],
            "control_class": "strategy_fixable",
        },
        "plan": {
            "direction": "Reduce the first-attack readiness requirement consistently."
        },
        "strengths_to_preserve": [
            {
                "pattern": "Stable two-base Marine-Tank production shell",
                "evidence": ["Game 1 @ 180s"],
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_optimizer_prompt_uses_compact_evidence_driven_policy() -> None:
    prompt = build_candidate_prompt(
        strategy_name="tank",
        race="terran",
        battle_analysis=BattleAnalysis(
            strategy_name="tank",
            race="terran",
            sample_size=2,
            record_mix="1W/1L",
            raw=_decision(),
        ),
        skill_texts={"strategy.md": TANK_STRATEGY},
        tool_observations=[],
        validation_errors=[],
        candidate=None,
        knowledge_mode="enabled",
        decision=_decision(),
    )

    assert "analysis and package-selection stages are complete" in prompt
    assert "implementing only the selected optimization brief" in prompt
    assert "champion_lineage" in prompt
    assert "fill audit forms" in prompt


def test_semantic_validation_allows_missing_winning_mechanism_audit() -> None:
    decision = _decision(
        plan={
            "direction": "repair the loss shortfall",
            "preservation_checks": [
                {"invariant": "early commitment", "effect": "preserve"}
            ],
        }
    )

    errors = _blocking_semantic_errors(_semantic_payload(), decision=decision)

    assert errors == []


def test_failure_stage_scope_allows_causally_selected_composition_change() -> None:
    payload = _semantic_payload()
    payload["failure_stage_scope_audit"] = {
        "failure_stage": "before_core_mechanism",
        "composition_changed": True,
        "composition_change_relation": "necessary_dependency",
        "retreat_policy_changed": False,
        "retreat_change_relation": "none",
        "stage_scope_aligned": True,
        "reason": "The candidate adds a new unit before the selected failure is repaired.",
    }

    errors = _blocking_semantic_errors(
        payload,
        decision=_decision(
            plan={
                "direction": "repair production timing",
            }
        ),
    )

    assert not any("composition scope" in error for error in errors)


def test_failure_stage_scope_audit_is_advisory_for_new_scoped_decisions() -> None:
    errors = _blocking_semantic_errors(
        _semantic_payload(),
        decision=_decision(
            failure_mode_analysis={
                "failure_stage": "during_commitment_or_engagement"
            },
            plan={
                "direction": "repair production timing",
                "stage_scope_reason": "The selected mechanism does not change either lever.",
            },
        ),
    )

    assert errors == []


def test_failure_stage_scope_blocks_unrelated_retreat_change() -> None:
    payload = _semantic_payload()
    payload["failure_stage_scope_audit"] = {
        "failure_stage": "during_commitment_or_engagement",
        "composition_changed": False,
        "composition_change_relation": "none",
        "retreat_policy_changed": True,
        "retreat_change_relation": "unrelated",
        "stage_scope_aligned": True,
        "reason": "The selected mechanism concerns production, not force preservation.",
    }

    errors = _blocking_semantic_errors(
        payload,
        decision=_decision(
            plan={
                "direction": "repair production timing",
            }
        ),
    )

    assert any("without a causal role" in error for error in errors)


def test_failure_stage_scope_allows_evidence_selected_composition_and_retreat_change() -> None:
    payload = _semantic_payload()
    payload["failure_stage_scope_audit"] = {
        "failure_stage": "during_commitment_or_engagement",
        "composition_changed": True,
        "composition_change_relation": "implements_selected_hypothesis",
        "retreat_policy_changed": True,
        "retreat_change_relation": "necessary_dependency",
        "stage_scope_aligned": True,
        "reason": "Repeated contact evidence selects both package and force-retention changes.",
    }

    errors = _blocking_semantic_errors(
        payload,
        decision=_decision(
            plan={
                "direction": "repair the decisive engagement package",
            }
        ),
    )

    assert not any("composition scope" in error for error in errors)
    assert not any("retreat scope" in error for error in errors)


def test_semantic_validation_accepts_preserved_winning_chain() -> None:
    decision = _decision(
        plan={
            "direction": "repair the loss shortfall",
            "preservation_checks": [
                {"invariant": "early commitment", "effect": "preserve"}
            ],
        }
    )
    payload = _semantic_payload()
    payload["winning_mechanism_audit"] = {
        "parent_winning_chain": "mass, commit, reinforce",
        "candidate_winning_chain": "mass, commit, reinforce",
        "reviewed_invariants": [
            {
                "invariant": "early commitment",
                "candidate_effect": "preserved",
                "reason": "the launch gate is unchanged",
            }
        ],
        "earliest_broken_link": "",
        "verdict": "preserved",
    }

    assert _blocking_semantic_errors(payload, decision=decision) == []


def _analysis() -> BattleAnalysis:
    return BattleAnalysis(
        strategy_name="tank",
        race="terran",
        sample_size=2,
        record_mix="1W/1L",
        raw=_decision(),
    )


def _patch(document: StrategyDocument, detail_id: str, replacement: str, why: str) -> dict:
    current = next(item for item in document.details if item.id == detail_id)
    return {
        "target": detail_id,
        "expected_old_hash": paragraph_hash(current.value),
        "replacement": replacement,
        "why_required": why,
    }


def test_candidate_critic_module_is_removed() -> None:
    assert not Path("evol_agent/core/candidate_critic.py").exists()


def test_validator_prompt_is_limited_to_hard_execution_errors() -> None:
    prompt = build_strategy_patch_validation_prompt(
        decision=_decision(),
        parent_text=TANK_STRATEGY,
        candidate_text=TANK_STRATEGY,
        patches=[],
        capability_manifest=build_executor_capability_manifest("terran"),
    )
    assert "direct internal contradiction" in prompt
    assert "unsupported runtime action" in prompt
    assert "missing mandatory" in prompt
    assert "above 200 supply" in prompt
    assert "similarity to history is non-blocking" in prompt
    assert "custom detachment" not in prompt
    assert "fixed-composition" not in prompt
    assert "production_target_audit" not in prompt


def test_empty_patches_are_rejected() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    errors = validate_strategy_patch_structure(
        decision=_decision(
            plan={
                "direction": "Test a coordinated retreat revision.",
            }
        ),
        patches=[],
        parent_document=document,
    )
    assert errors == ["strategy patch must contain at least one paragraph change"]


def test_unknown_paragraph_is_rejected() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    errors = validate_strategy_patch_structure(
        decision=_decision(),
        patches=[
            {
                "target": "fake_section",
                "replacement": "Do something unsupported.",
                "why_required": "This paragraph defines the readiness threshold being tested.",
            }
        ],
        parent_document=document,
    )
    assert any("unknown strategy paragraph: fake_section" in item for item in errors)


def test_goal_wording_is_not_structurally_locked() -> None:
    document = StrategyDocument.parse(
        "# Summary\nA concentrated timing attack.\n\n"
        "# Details\n- Strategy Style: Use a concentrated timing attack.\n"
        "- Main Attack Gate: Attack with a gathered force.\n"
    )
    errors = validate_strategy_patch_structure(
        decision=_decision(),
        patches=[
            _patch(
                document,
                "strategy_style",
                "Use a concentrated Marine-Tank timing attack with sustained reinforcements.",
                "The selected hypothesis clarifies how the existing combat style continues after contact.",
            )
        ],
        parent_document=document,
    )

    assert errors == []


def test_duplicate_target_is_rejected() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patch = _patch(
        document,
        "main_attack_gate",
        "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
        "This paragraph defines the readiness threshold being tested.",
    )
    errors = validate_strategy_patch_structure(
        decision=_decision(),
        patches=[patch, dict(patch)],
        parent_document=document,
    )
    assert "duplicate patch target: main_attack_gate" in errors


def test_unchanged_replacement_is_rejected() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    posture = next(item for item in document.details if item.id == "pre_attack_army_posture")
    errors = validate_strategy_patch_structure(
        decision=_decision(),
        patches=[
            _patch(
                document,
                "pre_attack_army_posture",
                posture.value,
                "This paragraph defines the readiness threshold being tested.",
            )
        ],
        parent_document=document,
    )
    assert "pre_attack_army_posture: replacement is unchanged" in errors


def test_summary_modification_is_rejected() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    errors = validate_strategy_patch_structure(
        decision=_decision(),
        patches=[
            {
                "target": "summary",
                "replacement": "A new identity.",
                "why_required": "identity change",
            }
        ],
        parent_document=document,
    )
    assert "summary modification is not allowed" in errors


def test_missing_why_required_is_rejected() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    gate = next(item for item in document.details if item.id == "main_attack_gate")
    errors = validate_strategy_patch_structure(
        decision=_decision(),
        patches=[
            {
                "target": "main_attack_gate",
                "expected_old_hash": paragraph_hash(gate.value),
                "replacement": "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
                "why_required": "",
            }
        ],
        parent_document=document,
    )
    assert any("why_required is required" in item for item in errors)


def test_five_or_more_necessary_patches_are_allowed() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    targets = [
        "main_attack_gate",
        "recovery_and_cleanup",
                "production",
        "pre_attack_army_posture",
        "ultimate_goal",
        "production",
    ]
    patches = [
        _patch(
            document,
            item.id,
            f"Keep the Marine-Tank plan while updating {item.title} for the new readiness rule.",
            "This paragraph repeats or defines the readiness rule being tested.",
        )
        for item in document.details
        if item.id in targets
    ]
    assert len(patches) >= 5
    errors = validate_strategy_patch_structure(
        decision=_decision(
            plan={
                "direction": "Test a coordinated retreat revision.",
            }
        ),
        patches=patches,
        parent_document=document,
    )
    assert errors == []


def test_unrelated_scouting_patch_is_rejected(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(
            document,
            "main_attack_gate",
            "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
            "This paragraph defines the readiness threshold being tested.",
        ),
        {
            "target": "scouting",
            "expected_old_hash": "missing",
            "replacement": "Send extra SCV scouts across the map before every attack.",
            "why_required": "Better scouting is useful.",
        },
    ]
    errors = validate_strategy_patch_structure(
        decision=_decision(),
        patches=patches,
        parent_document=document,
    )
    assert any("scouting and scanning behavior is Commander-owned" in item for item in errors)


def test_first_attack_gate_cannot_be_copied_into_recovery_without_evidence() -> None:
    payload = _semantic_payload()
    payload["failure_stage_scope_audit"] = {
        "failure_stage": "before_core_mechanism",
        "composition_changed": False,
        "composition_change_relation": "none",
        "retreat_policy_changed": True,
        "retreat_change_relation": "unrelated",
        "opening_gate_reused_as_recovery_gate": True,
        "opening_gate_reuse_supported": False,
        "stage_scope_aligned": True,
        "reason": "The candidate copied a larger opening count into recovery only for consistency.",
    }

    errors = _blocking_semantic_errors(payload, decision=_decision())

    assert any("reuses the first-attack gate" in item for item in errors)


def test_conflicting_readiness_thresholds_are_rejected(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(
            document,
            "main_attack_gate",
            "Begin the planned attack with 40 Marines and 8 Siege Tanks.",
            "This paragraph defines the readiness threshold being tested.",
        ),
        _patch(
            document,
            "recovery_and_cleanup",
            "If progress stalls, withdraw and rebuild to 45 Marines and 10 Siege Tanks.",
            "This paragraph repeats the readiness threshold.",
        ),
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: {
            "valid": False,
            "errors": [
                "Recovery and Cleanup still uses the old readiness threshold."
            ],
        },
    )
    errors = validate_strategy_patch_semantics(
        decision=_decision(),
        parent_text=TANK_STRATEGY,
        candidate_text=patched,
        patches=patches,
        capability_manifest=build_executor_capability_manifest("terran"),
    )
    assert any("old readiness threshold" in item for item in errors)


def test_destroyed_strength_is_rejected(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(
            document,
            "expansion",
            "Stay on one base and all-in from the natural without a second Command Center.",
            "This paragraph defines the readiness threshold being tested.",
        )
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: {
            "valid": False,
            "errors": [
                "The patch abandons the preserved two-base Marine-Tank production shell."
            ],
        },
    )
    errors = validate_strategy_patch_semantics(
        decision=_decision(),
        parent_text=TANK_STRATEGY,
        candidate_text=patched,
        patches=patches,
        capability_manifest=build_executor_capability_manifest("terran"),
    )
    assert any("two-base" in item for item in errors)


def test_runtime_micro_requirements_are_rejected(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(
            document,
            "engagement_and_reinforcement",
            "Use per-unit kiting and exact focus-fire micro during the attack.",
            "This paragraph defines how the revised gate should fight.",
        )
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: _semantic_payload(
            valid=False,
            errors=[{
                "type": "runtime_boundary",
                "location": "engagement_and_reinforcement",
                "description": "requires unavailable runtime behavior: per-unit micro",
                "severity": "blocking",
            }],
        ),
    )
    errors = validate_strategy_patch_semantics(
        decision=_decision(),
        parent_text=TANK_STRATEGY,
        candidate_text=patched,
        patches=patches,
        capability_manifest=build_executor_capability_manifest("terran"),
    )
    assert any("unavailable runtime behavior" in item for item in errors)


def test_scan_safety_requirement_is_rejected(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(
            document,
            "main_attack_gate",
            "Begin the planned attack with 45 Marines and 10 Siege Tanks unless scanning is unsafe.",
            "This paragraph defines the readiness threshold being tested.",
        )
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: _semantic_payload(
            valid=False,
            errors=[{
                "type": "runtime_boundary",
                "location": "main_attack_gate",
                "description": "requires unavailable runtime behavior: scan safety",
                "severity": "blocking",
            }],
        ),
    )
    errors = validate_strategy_patch_semantics(
        decision=_decision(),
        parent_text=TANK_STRATEGY,
        candidate_text=patched,
        patches=patches,
        capability_manifest=build_executor_capability_manifest("terran"),
    )
    assert any("unavailable runtime behavior" in item for item in errors)


def test_unsupported_wake_condition_is_rejected(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(
            document,
            "main_attack_gate",
            (
                "Hold the force and set a wake event for scan_ready, "
                "enemy_visible_in_target_zone, or game_time_at_least."
            ),
            "The revised gate requires a reachable redecision event.",
        )
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: _semantic_payload(
            valid=False,
            errors=[{
                "type": "runtime_boundary",
                "location": "main_attack_gate",
                "description": "unsupported wake condition: enemy_visible_in_target_zone",
                "severity": "blocking",
            }],
        ),
    )
    errors = validate_strategy_patch_semantics(
        decision=_decision(),
        parent_text=TANK_STRATEGY,
        candidate_text=patched,
        patches=patches,
        capability_manifest=build_executor_capability_manifest("terran"),
    )
    assert any("unsupported wake condition: enemy_visible_in_target_zone" in item for item in errors)


def test_macro_action_in_wake_clause_is_not_treated_as_wake_condition(
    monkeypatch,
) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(
            document,
            "pre_attack_army_posture",
            (
                "At the next wake, if enemy air is observed, add train_viking "
                "to the macro targets and set a wake event for game_time_at_least."
            ),
            "The strategy rechecks a supported observation before changing production.",
        )
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: _semantic_payload(),
    )

    errors = validate_strategy_patch_semantics(
        decision=_decision(),
        parent_text=TANK_STRATEGY,
        candidate_text=patched,
        patches=patches,
        capability_manifest=build_executor_capability_manifest("terran"),
    )

    assert "unsupported wake condition in strategy: train_viking" not in errors


def test_precise_maximum_range_requirement_is_rejected(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(
            document,
            "engagement_and_reinforcement",
            "Place the force at maximum range before beginning the engagement.",
            "The revised package requires exact positioning.",
        )
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: _semantic_payload(
            valid=False,
            errors=[{
                "type": "runtime_boundary",
                "location": "engagement_and_reinforcement",
                "description": "at maximum range requires unavailable exact positioning",
                "severity": "blocking",
            }],
        ),
    )
    errors = validate_strategy_patch_semantics(
        decision=_decision(),
        parent_text=TANK_STRATEGY,
        candidate_text=patched,
        patches=patches,
        capability_manifest=build_executor_capability_manifest("terran"),
    )
    assert any("at maximum range" in item for item in errors)


def test_non_blocking_semantic_notes_do_not_fail(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(
            document,
            "main_attack_gate",
            "Begin the planned attack with 44 Marines and 10 Siege Tanks.",
            "This paragraph defines the readiness threshold being tested.",
        ),
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: _semantic_payload(
            valid=False,
            errors=[
                {
                    "type": "scope",
                    "location": "Pre-Attack Army Posture",
                    "description": "The information gate is restated here.",
                    "severity": "non-blocking",
                },
                {
                    "type": "preserved_strengths",
                    "location": "Ultimate Goal",
                    "description": "The higher threshold is a weak solution.",
                    "severity": "non-blocking",
                },
            ],
        ),
    )
    assert (
        validate_strategy_patch_semantics(
            decision=_decision(),
            parent_text=TANK_STRATEGY,
            candidate_text=patched,
            patches=patches,
            capability_manifest=build_executor_capability_manifest("terran"),
        )
        == []
    )


def test_resumed_unit_with_stage_target_does_not_require_duplicate_final_target() -> None:
    payload = _semantic_payload()
    payload["production_target_audit"] = [
        {
            "unit": "Marauder",
            "instruction": "resume train_marauder after eight Siege Tanks",
            "stage_target": "4 Marauders",
            "ultimate_goal_target": "",
            "temporary_stop_rule": "no explicit cap in the strategy",
            "verdict": "bounded",
        }
    ]

    errors = _blocking_semantic_errors(payload)

    assert errors == []


def test_complete_resumed_unit_and_supply_audit_can_pass() -> None:
    payload = _semantic_payload()
    payload["production_target_audit"] = [
        {
            "unit": "Marauder",
            "instruction": "resume train_marauder after eight Siege Tanks",
            "stage_target": "4 Marauders",
            "ultimate_goal_target": "4 Marauders",
            "temporary_stop_rule": "",
            "verdict": "bounded",
        }
    ]
    payload["final_supply"] = {
        "total": 185,
        "calculation": "75 + 42 + 8 + 8 + 8 + 44 = 185 supply",
        "verdict": "valid",
    }

    assert _blocking_semantic_errors(payload) == []


def test_missing_production_and_supply_audits_are_advisory() -> None:
    errors = _blocking_semantic_errors({"valid": True, "errors": []})

    assert errors == []


def test_blocking_dict_errors_are_formatted(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(
            document,
            "main_attack_gate",
            "Begin the planned attack with 40 Marines and 8 Siege Tanks.",
            "This paragraph defines the readiness threshold being tested.",
        )
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: {
            "valid": False,
            "errors": [
                {
                    "type": "internal_inconsistency",
                    "location": "Main Attack Gate",
                    "description": "Recovery still uses the old readiness threshold.",
                    "severity": "blocking",
                }
            ],
        },
    )
    errors = validate_strategy_patch_semantics(
        decision=_decision(),
        parent_text=TANK_STRATEGY,
        candidate_text=patched,
        patches=patches,
        capability_manifest=build_executor_capability_manifest("terran"),
    )
    assert any("old readiness threshold" in item for item in errors)
    assert not any("{'type'" in item for item in errors)


def test_validator_does_not_reject_a_legal_but_weak_patch(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(
            document,
            "main_attack_gate",
            "Begin the planned attack with 44 Marines and 10 Siege Tanks.",
            "This paragraph defines the readiness threshold being tested.",
        ),
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: _semantic_payload(),
    )
    assert (
        validate_strategy_patch_structure(
            decision=_decision(),
            patches=patches,
            parent_document=document,
        )
        == []
    )
    assert (
        validate_strategy_patch_semantics(
            decision=_decision(),
            parent_text=TANK_STRATEGY,
            candidate_text=patched,
            patches=patches,
            capability_manifest=build_executor_capability_manifest("terran"),
        )
        == []
    )


def test_validator_errors_drive_optimizer_retry(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    calls: list[str] = []
    optimizer_calls = 0

    def fake_llm(prompt: str, **kwargs):
        nonlocal optimizer_calls
        calls.append(prompt)
        if "You are validating a strategy patch" in prompt:
            if optimizer_calls == 1:
                return {
                    "valid": False,
                    "errors": [
                        {
                            "type": "internal_inconsistency",
                            "location": "Recovery and Cleanup",
                            "description": "candidate reuses the first-attack gate as a recovery gate without post-contact evidence",
                            "severity": "blocking",
                        }
                    ],
                }
            return {"valid": True, "errors": []}
        optimizer_calls += 1
        patches = [
            _patch(
                document,
                "main_attack_gate",
                "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
                "This paragraph defines the readiness threshold being tested.",
            )
        ]
        if optimizer_calls == 1:
            patches.append(
                _patch(
                    document,
                    "recovery_and_cleanup",
                    "If progress stalls, withdraw and rebuild to 36 Marines and 8 Siege Tanks.",
                    "This incorrectly synchronizes recovery with the opening gate.",
                )
            )
        else:
            assert "reuses the first-attack gate" in prompt
        return {
            "action": "draft_candidate" if optimizer_calls == 1 else "revise_candidate",
            "patches": patches,
            "expected_effect": "earlier first attack",
            "main_risk": "smaller force",
        }

    monkeypatch.setattr("evol_agent.core.optimization_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr("evol_agent.core.strategy_patch_validator.call_json_llm", fake_llm)
    result, improvement, _obs, _errors, events = run_optimization_agent_loop(
        strategy_name="tank",
        race="terran",
        battle_analysis=_analysis(),
        skill_texts={"strategy.md": TANK_STRATEGY},
        initial_tool_observations=[],
        capability_manifest=build_executor_capability_manifest("terran"),
    )
    assert result.ok
    assert improvement is not None
    assert any(item.get("action") == "strategy_patch_semantics" for item in events)
    assert "36 Marines and 8 Siege Tanks" in next(
        item.value
        for item in StrategyDocument.parse(improvement.files["strategy.md"]).details
        if item.id == "main_attack_gate"
    )
    assert next(
        item.value
        for item in StrategyDocument.parse(improvement.files["strategy.md"]).details
        if item.id == "recovery_and_cleanup"
    ) == next(
        item.value for item in document.details if item.id == "recovery_and_cleanup"
    )
    assert any("Regenerate the whole strategy.md" in prompt for prompt in calls)


def test_style_audit_allows_unit_change_when_combat_style_is_preserved() -> None:
    payload = _semantic_payload()
    payload["style_and_window_audit"].update(
        {
            "parent_combat_style": "early one-base pressure",
            "candidate_combat_style": "early one-base pressure with optional support",
            "contact_window_effect": "similar",
            "new_hard_prerequisites": [],
            "shared_production_tradeoffs": ["one Tech Lab reduces one Marine queue"],
            "verdict": "preserved",
        }
    )

    assert _blocking_semantic_errors(payload) == []


def test_new_gas_unit_requires_compatible_gas_plan() -> None:
    payload = _semantic_payload()
    payload["new_dependency_audit"] = [
        {
            "target": "Marauder after the first attack",
            "stage": "post-contact reinforcement",
            "gas_required": True,
            "gas_plan": "none",
            "required_prerequisites": ["Barracks Tech Lab", "Refinery"],
            "declared_prerequisites": ["Barracks Tech Lab"],
            "missing_dependencies": ["compatible gas economy"],
            "shared_production_tradeoff": "uses a Barracks that produced Marines",
            "verdict": "resource_conflict",
        }
    ]

    errors = _blocking_semantic_errors(payload)

    assert any("gas economy" in error for error in errors)
    assert any("compatible gas economy" in error for error in errors)


def test_style_audit_blocks_support_unit_as_hidden_attack_gate() -> None:
    payload = _semantic_payload()
    payload["style_and_window_audit"].update(
        {
            "contact_window_effect": "later",
            "window_change_justified": False,
            "new_hard_prerequisites": ["4 Marauders before first attack"],
            "hidden_attack_gate": True,
        }
    )

    errors = _blocking_semantic_errors(payload)

    assert any("hidden attack gate" in item for item in errors)
    assert any("critical power window" in item for item in errors)


def _marine_timing_knowledge() -> dict:
    return run_knowledge_query(
        {
            "id": "QTIMING",
            "question": "Verify the Marine-Marauder first-contact package.",
            "actions": [
                "train_marine",
                "train_marauder",
                "build_barracks",
                "build_barracks_techlab",
                "build_gas",
            ],
            "needs": ["requirements"],
        },
        race="terran",
    )


def test_contact_timing_report_calculates_candidate_delay_and_cost() -> None:
    timing_model = {
        "parent": {
            "economy": {
                "worker_target_before_commitment": 20,
                "base_target_before_commitment": 1,
                "gas_workers_before_commitment": 0,
            },
            "gate_components": [
                {"action": "train_marine", "quantity": 20, "production_slots": 6}
            ],
            "setup_actions": [
                {"action": "build_barracks", "quantity": 6, "parallel_slots": 6}
            ],
        },
        "candidate": {
            "economy": {
                "worker_target_before_commitment": 20,
                "base_target_before_commitment": 1,
                "gas_workers_before_commitment": 6,
            },
            "gate_components": [
                {"action": "train_marine", "quantity": 20, "production_slots": 4},
                {"action": "train_marauder", "quantity": 4, "production_slots": 2},
            ],
            "setup_actions": [
                {"action": "build_barracks", "quantity": 6, "parallel_slots": 6},
                {
                    "action": "build_barracks_techlab",
                    "quantity": 2,
                    "parallel_slots": 2,
                },
                {"action": "build_gas", "quantity": 2, "parallel_slots": 2},
            ],
        },
        "new_hard_gate_components": ["4 Marauders"],
        "fallback_preserves_parent_window": False,
    }

    report = _build_contact_timing_report(
        timing_model,
        [_marine_timing_knowledge()],
    )

    assert report["complete"] is True
    assert report["parent_earliest_feasible_time_seconds"] == 239.591
    assert report["candidate_earliest_feasible_time_seconds"] == 374.122
    assert report["earliest_feasible_timing_delta_seconds"] == 134.531
    assert report["gate_cost_delta"] == {
        "minerals": 750.0,
        "gas": 150.0,
        "supply": 8.0,
    }


def test_contact_timing_report_ignores_unquantified_implicit_supply_rows() -> None:
    package = {
        "economy": {
            "worker_target_before_commitment": 20,
            "base_target_before_commitment": 1,
            "gas_workers_before_commitment": 0,
        },
        "gate_components": [
            {"action": "train_marine", "quantity": 20, "production_slots": 6}
        ],
        "setup_actions": [
            {"action": "build_barracks", "quantity": 6, "parallel_slots": 6},
            {
                "action": "build_supply_depot",
                "quantity": None,
                "parallel_slots": None,
            },
        ],
    }

    report = _build_contact_timing_report(
        {"parent": package, "candidate": package},
        [_marine_timing_knowledge()],
    )

    assert report["complete"] is True
    assert report["errors"] == []
    assert all(
        item.get("action") != "build_supply_depot"
        for item in report["declared_packages"]["candidate"]["setup_actions"]
    )


def test_semantic_validation_exports_deterministic_feasibility_audit(monkeypatch) -> None:
    timing_model = {
        "parent": {
            "economy": {
                "worker_target_before_commitment": 20,
                "base_target_before_commitment": 1,
                "gas_workers_before_commitment": 0,
            },
            "gate_components": [
                {"action": "train_marine", "quantity": 20, "production_slots": 6}
            ],
            "setup_actions": [
                {"action": "build_barracks", "quantity": 6, "parallel_slots": 3}
            ],
        },
        "candidate": {
            "economy": {
                "worker_target_before_commitment": 20,
                "base_target_before_commitment": 1,
                "gas_workers_before_commitment": 0,
            },
            "gate_components": [
                {"action": "train_marine", "quantity": 18, "production_slots": 6}
            ],
            "setup_actions": [
                {"action": "build_barracks", "quantity": 6, "parallel_slots": 3}
            ],
        },
        "new_hard_gate_components": [],
        "fallback_preserves_parent_window": True,
    }

    def fake_llm(prompt: str, **kwargs):
        if prompt.startswith("Extract the production package"):
            return {"timing_model": timing_model}
        payload = _semantic_payload()
        payload["contact_window_audit"] = {
            "parent_earliest_feasible_time_seconds": 205.101,
            "candidate_earliest_feasible_time_seconds": 198.0,
            "own_package_at_candidate_contact": "18 Marines",
            "opponent_package_at_candidate_contact": "recorded early defense",
            "opponent_growth_during_wait": "none; candidate is earlier",
            "matchup_and_counter_assessment": "same matchup at an earlier window",
            "reinforcement_and_continuity": "six Barracks remain available",
            "relative_advantage": "preserves",
            "evidence": ["Game 1 @ 205s"],
            "verdict": "favorable",
        }
        return payload

    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        fake_llm,
    )
    audit: dict = {}
    errors = validate_strategy_patch_semantics(
        decision=_decision(
            plan={
                "direction": "test an earlier Marine commitment",
                "contact_window_effect": "earlier",
            }
        ),
        parent_text="parent",
        candidate_text="candidate",
        patches=[],
        knowledge_runs=[_marine_timing_knowledge()],
        audit_output=audit,
    )

    assert errors == []
    report = audit["contact_timing_report"]
    assert report["complete"] is True
    assert report["candidate_earliest_feasible_time_seconds"] < report["parent_earliest_feasible_time_seconds"]


def test_candidate_knowledge_resolves_irregular_plurals_aliases_and_dependencies() -> None:
    strategy = (
        "Build two Factories with Factory Tech Labs. Research Combat Shield, "
        "then produce Marines and Siege Tanks."
    )

    run = _candidate_knowledge_run(
        candidate_text=strategy,
        parent_text=strategy,
        race="terran",
        capability_manifest=build_executor_capability_manifest("terran"),
    )

    assert run is not None and run["ok"] is True
    packet = run["dataset_evidence"][0]["result"]
    actions = {row["action"] for row in packet["action_facts"]}
    assert "build_factory" in actions
    assert "build_factory_techlab" in actions
    assert "research_shieldwall" in actions
    assert "train_siege_tank" in actions


def test_contact_timing_missing_worker_target_is_reported_without_invalidating() -> None:
    knowledge = _candidate_knowledge_run(
        candidate_text=(
            "Build two Factories with Factory Tech Labs. Research Combat Shield, "
            "then gather ten Siege Tanks."
        ),
        parent_text="Build two Factories with Factory Tech Labs and gather ten Siege Tanks.",
        race="terran",
        capability_manifest=build_executor_capability_manifest("terran"),
    )
    timing_model = {
        "parent": {
            "gate_components": [
                {"action": "train_siege_tank", "quantity": 10, "production_slots": 2}
            ],
            "setup_actions": [
                {"action": "build_factory", "quantity": 2, "parallel_slots": 2},
                {"action": "build_factory_techlab", "quantity": 2, "parallel_slots": 2},
            ],
        },
        "candidate": {
            "gate_components": [
                {"action": "train_siege_tank", "quantity": 10, "production_slots": 2}
            ],
            "setup_actions": [
                {"action": "build_factory", "quantity": 2, "parallel_slots": 2},
                {"action": "build_factory_techlab", "quantity": 2, "parallel_slots": 2},
                {"action": "research_combat_shield", "quantity": 1, "parallel_slots": 1},
            ],
        },
    }

    report = _build_contact_timing_report(timing_model, [knowledge])

    assert report["complete"] is True
    assert "worker target missing; simulation keeps the initial 8 SCVs" in report["evidence_warnings"]


def test_contact_window_audit_blocks_later_package_with_worse_relative_position() -> None:
    decision = _decision(
        plan={
            "direction": "add a support package",
            "contact_window_effect": "later",
            "new_hard_prerequisites": ["4 support units"],
            "production_tradeoffs": ["two core production slots become support slots"],
        }
    )
    payload = _semantic_payload()
    payload["contact_window_audit"] = {
        "parent_earliest_feasible_time_seconds": 205.1,
        "candidate_earliest_feasible_time_seconds": 327.0,
        "own_package_at_candidate_contact": "20 Marines and 4 Marauders",
        "opponent_package_at_candidate_contact": "enemy tech and army have grown",
        "opponent_growth_during_wait": "Factory and support tech complete",
        "matchup_and_counter_assessment": "the added package does not offset growth",
        "reinforcement_and_continuity": "core production falls from six to four slots",
        "relative_advantage": "worsens",
        "evidence": ["Game 10 @ 323s: opponent technology and army increase"],
        "verdict": "unfavorable",
    }
    report = {
        "complete": True,
        "parent_earliest_feasible_time_seconds": 205.1,
        "candidate_earliest_feasible_time_seconds": 327.0,
        "earliest_feasible_timing_delta_seconds": 121.9,
        "gate_cost_delta": {"minerals": 750.0, "gas": 150.0, "supply": 8.0},
    }

    errors = _blocking_semantic_errors(
        payload,
        decision=decision,
        contact_timing_report=report,
    )

    assert any("relative power at contact" in item for item in errors)
    assert any("contact-window verdict" in item for item in errors)


def test_contact_window_uncertainty_does_not_reject_an_executable_candidate() -> None:
    decision = _decision(
        plan={
            "direction": "change the fighting package",
            "contact_window_effect": "unknown",
            "new_hard_prerequisites": [],
            "production_tradeoffs": ["resource allocation changes"],
        }
    )
    payload = _semantic_payload()
    payload["contact_window_audit"] = {
        "parent_earliest_feasible_time_seconds": 205.1,
        "candidate_earliest_feasible_time_seconds": 213.1,
        "own_package_at_candidate_contact": "calculated package",
        "opponent_package_at_candidate_contact": "not recorded near this time",
        "opponent_growth_during_wait": "unknown",
        "matchup_and_counter_assessment": "insufficient trajectory coverage",
        "reinforcement_and_continuity": "must be measured in candidate matches",
        "relative_advantage": "unknown",
        "evidence": [],
        "verdict": "unsupported",
    }
    report = {
        "complete": True,
        "parent_earliest_feasible_time_seconds": 205.1,
        "candidate_earliest_feasible_time_seconds": 213.1,
        "earliest_feasible_timing_delta_seconds": 8.0,
        "evidence_warnings": [],
    }

    errors = _blocking_semantic_errors(
        payload,
        decision=decision,
        contact_timing_report=report,
    )

    assert not any("contact-window" in item for item in errors)
    assert not any("relative power at contact" in item for item in errors)


def test_mechanism_history_audit_blocks_semantic_rename() -> None:
    experiment_id = "battlecruiser:g001:harder:battlecruiser_opt2"
    audit, errors = _normalize_mechanism_equivalence_audit(
        {
            "mechanism_equivalence_audit": {
                "semantic_relation": "equivalent_to_prior",
                "related_experiment_ids": [experiment_id],
                "repaired_dependencies": [],
                "reason": "both changes hard-gate the same upgrade package",
                "confidence": "high",
            }
        },
        prior_experiences=[
            {
                "experiment_id": experiment_id,
                "implementation_verdict": "implemented",
                "hypothesis_verdict": "contradicted",
                "decision": "rejected",
            }
        ],
    )

    assert any("mechanism history" in item for item in errors)
    assert audit["verdict"] == "blocked"


def test_semantic_validation_blocks_failed_semantically_equivalent_history(monkeypatch) -> None:
    experiment_id = "marine:g001:harder:marine_opt1"
    prompts: list[str] = []

    def fake_llm(prompt: str, **kwargs):
        prompts.append(prompt)
        if prompt.startswith("You are an independent semantic experiment-history judge"):
            return {
                "mechanism_equivalence_audit": {
                    "semantic_relation": "equivalent_to_prior",
                    "related_experiment_ids": [experiment_id],
                    "repaired_dependencies": [],
                    "reason": "both interventions raise the same first-attack gate",
                    "confidence": "high",
                }
            }
        return _semantic_payload()

    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        fake_llm,
    )
    audit: dict = {}
    errors = validate_strategy_patch_semantics(
        decision=_decision(),
        parent_text="Attack with 20 Marines.",
        candidate_text="Attack with 50 Marines.",
        patches=[{"target": "main_attack_gate", "replacement": "Attack with 50 Marines."}],
        prior_experiences=[
            {
                "experiment_id": experiment_id,
                "mechanism_prediction": {
                    "minimum_material_change": "Raise the gate from 20 to 30 Marines."
                },
                "implementation_verdict": "implemented",
                "hypothesis_verdict": "contradicted",
                "decision": "rejected",
            }
        ],
        audit_output=audit,
    )

    assert any("mechanism history" in error for error in errors)
    assert audit["mechanism_equivalence_audit"]["verdict"] == "blocked"
    assert any("independent semantic experiment-history judge" in prompt for prompt in prompts)


def test_prior_gate_execution_issue_blocks_another_attack_gate_edit(monkeypatch) -> None:
    def fake_llm(prompt: str, **kwargs):
        if prompt.startswith("You are an independent semantic experiment-history judge"):
            return {
                "mechanism_equivalence_audit": {
                    "semantic_relation": "new",
                    "related_experiment_ids": [],
                    "repaired_dependencies": [],
                    "reason": "the proposed label is different",
                    "confidence": "medium",
                }
            }
        return _semantic_payload()

    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        fake_llm,
    )
    errors = validate_strategy_patch_semantics(
        decision=_decision(),
        parent_text="Attack with 20 Marines.",
        candidate_text="Attack with 18 Marines.",
        patches=[{"target": "main_attack_gate", "replacement": "Attack with 18 Marines."}],
        prior_experiences=[
            {
                "experiment_id": "marine:g002:harder:marine_opt2",
                "gate_execution_audit": {
                    "status": "execution_issue",
                    "execution_issue_matches": 3,
                },
                "implementation_verdict": "execution_invalid",
                "hypothesis_verdict": "not_tested",
            }
        ],
    )

    assert any(
        "keep the strategy gate unchanged until runtime execution is repaired" in error
        for error in errors
    )
