from __future__ import annotations

from pathlib import Path

from evol_agent.core.capabilities import build_executor_capability_manifest
from evol_agent.core.optimization_agent_loop import (
    _patches_to_operations,
    run_optimization_agent_loop,
)
from evol_agent.core.strategy_patch_validator import (
    build_strategy_patch_validation_prompt,
    validate_strategy_patch_semantics,
    validate_strategy_patch_structure,
)
from evol_agent.core.types import BattleAnalysis
from evol_agent.optimization.strategy_document import StrategyDocument, paragraph_hash


TANK_STRATEGY = Path("skills/terran/tank/strategy.md").read_text(encoding="utf-8")


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


def test_validator_prompt_checks_coherent_package_and_identity() -> None:
    prompt = build_strategy_patch_validation_prompt(
        decision=_decision(),
        parent_text=TANK_STRATEGY,
        candidate_text=TANK_STRATEGY,
        patches=[],
        capability_manifest=build_executor_capability_manifest("terran"),
    )
    assert "unrelated second" in prompt
    assert "strategic objective" in prompt
    assert "If this patch were removed" in prompt
    assert "required prerequisite, resource/production dependency" in prompt
    assert "global target changes" in prompt
    assert "already satisfies" in prompt
    assert "3. Test strength" in prompt
    assert "minimum_material_change" in prompt
    assert "never from patch count" in prompt
    assert "defining army concept and win plan" in prompt
    assert "NOT judging whether another causal hypothesis would have been better" in prompt


def test_empty_patches_are_rejected() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    errors = validate_strategy_patch_structure(
        decision=_decision(),
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
        decision=_decision(),
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
        _patch(
            document,
            "scouting",
            "Send extra SCV scouts across the map before every attack.",
            "Better scouting is useful.",
        ),
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: {
            "valid": False,
            "errors": [
                "Scouting was modified but is not required by the supplied hypothesis."
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
    assert any("Scouting" in item for item in errors)


def test_consistent_readiness_dependency_patches_pass(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(
            document,
            "main_attack_gate",
            "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
            "This paragraph defines the readiness threshold being tested.",
        ),
        _patch(
            document,
            "recovery_and_cleanup",
            "If progress stalls, withdraw and rebuild to 36 Marines and 8 Siege Tanks.",
            "This paragraph repeats the old readiness threshold and must stay consistent.",
        ),
        _patch(
            document,
            "production",
            "Before the first attack, prioritize Factories and Siege Tanks needed for the revised gate.",
            "Production priority must support the new readiness rule.",
        ),
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: {"valid": True, "errors": []},
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
        lambda prompt, **kwargs: {"valid": True, "errors": []},
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
        lambda prompt, **kwargs: {"valid": True, "errors": []},
    )
    errors = validate_strategy_patch_semantics(
        decision=_decision(),
        parent_text=TANK_STRATEGY,
        candidate_text=patched,
        patches=patches,
        capability_manifest=build_executor_capability_manifest("terran"),
    )
    assert any("unavailable runtime behavior" in item for item in errors)


def test_non_blocking_semantic_notes_do_not_fail(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(
            document,
            "main_attack_gate",
            "Begin the planned attack with 44 Marines and 10 Siege Tanks.",
            "This paragraph defines the readiness threshold being tested.",
        ),
        _patch(
            document,
            "recovery_and_cleanup",
            "If progress stalls, withdraw and rebuild to 44 Marines and 10 Siege Tanks.",
            "This paragraph repeats the same old threshold and must stay consistent.",
        ),
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: {
            "valid": False,
            "errors": [
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
        },
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
        _patch(
            document,
            "recovery_and_cleanup",
            "If progress stalls, withdraw and rebuild to 44 Marines and 10 Siege Tanks.",
            "This paragraph repeats the same old threshold and must stay consistent.",
        ),
    ]
    patched, _changes = document.apply_patch(_patches_to_operations(patches))
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm",
        lambda prompt, **kwargs: {"valid": True, "errors": []},
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

    def fake_llm(prompt: str, **kwargs):
        calls.append(prompt)
        if "You are validating a strategy patch" in prompt:
            if "rebuild to 36 Marines and 8 Siege Tanks" in prompt:
                return {"valid": True, "errors": []}
            return {
                "valid": False,
                "errors": [
                    "Recovery and Cleanup still uses the old readiness threshold."
                ],
            }
        if "Recovery and Cleanup still uses the old readiness threshold." in prompt:
            return {
                "action": "revise_candidate",
                "patches": [
                    _patch(
                        document,
                        "main_attack_gate",
                        "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
                        "This paragraph defines the readiness threshold being tested.",
                    ),
                    _patch(
                        document,
                        "recovery_and_cleanup",
                        "If progress stalls, withdraw and rebuild to 36 Marines and 8 Siege Tanks.",
                        "This paragraph repeats the old readiness threshold and must stay consistent.",
                    ),
                ],
                "expected_effect": "earlier first attack",
                "main_risk": "smaller force",
            }
        return {
            "action": "draft_candidate",
            "patches": [
                _patch(
                    document,
                    "main_attack_gate",
                    "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
                    "This paragraph defines the readiness threshold being tested.",
                )
            ],
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
        if item.id == "recovery_and_cleanup"
    )
    assert any(
        "Fix only the reported patch validation errors" in prompt for prompt in calls
    )
