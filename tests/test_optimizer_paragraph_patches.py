from __future__ import annotations

from pathlib import Path

from evol_agent.core.optimization_agent_loop import (
    _candidate_knowledge_run,
    _knowledge_runs_for_optimizer,
    _normalize_optimizer_candidate,
    _patches_to_operations,
    run_optimization_agent_loop,
)
from evol_agent.core.capabilities import build_executor_capability_manifest
from evol_agent.core.prompts import build_candidate_prompt
from evol_agent.core.types import BattleAnalysis, ToolObservation
from evol_agent.optimization.strategy_document import StrategyDocument, paragraph_hash
from evol_agent.validation import validate_strategy_markdown


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


def _decision_analysis(**overrides) -> BattleAnalysis:
    raw = {
        "next_action": "propose_strategy_patch",
        "hypothesis": "The current readiness design requires too much completed army before the first attack.",
        "mechanism_prediction": {
            "expected_change": "the committed army reaches its intended objective earlier",
            "minimum_material_change": "the readiness rule must materially lower the delayed commitment",
            "outcome_prediction": "the first pressure window is reached more reliably",
            "disproof_condition": "commitment occurs materially earlier but the same failure persists",
        },
        "priority_problem": {
            "problem": "The first planned push repeatedly starts after the useful pressure window.",
            "evidence": ["Game 2 @ 430s", "Game 5 @ 510s"],
            "control_class": "strategy_fixable",
        },
        "plan": {
            "direction": (
                "Reduce the first-attack readiness requirement while keeping "
                "production and recovery rules internally consistent."
            )
        },
        "strengths_to_preserve": [
            {
                "pattern": "The two-base Marine-Tank production shell is stable.",
                "evidence": ["Game 1 @ 180s"],
            }
        ],
    }
    raw.update(overrides)
    return BattleAnalysis(
        strategy_name="tank",
        race="terran",
        sample_size=2,
        record_mix="1W/1L",
        raw=raw,
    )


def _patch(document: StrategyDocument, detail_id: str, replacement: str, why: str) -> dict:
    current = next(item for item in document.details if item.id == detail_id)
    return {
        "target": detail_id,
        "expected_old_hash": paragraph_hash(current.value),
        "replacement": replacement,
        "why_required": why,
    }


def test_candidate_knowledge_covers_actions_named_by_complete_strategy() -> None:
    run = _candidate_knowledge_run(
        candidate_text=(
            TANK_STRATEGY
            + "\nUse train_marine and train_siege_tank as explicit production actions."
        ),
        race="terran",
        capability_manifest=build_executor_capability_manifest("terran"),
    )

    assert run is not None
    assert run["ok"] is True
    packet = run["dataset_evidence"][0]["result"]
    actions = {row["action"] for row in packet["action_facts"]}
    assert "train_marine" in actions
    assert "train_siege_tank" in actions
    assert packet["coverage"]["complete"] is True


def test_optimizer_preserves_structured_knowledge_from_analysis_observation() -> None:
    structured = {
        "question_id": "Q1",
        "question": "What does train_siege_tank require?",
        "answer": "verified answer",
        "ok": True,
        "verification_schema": "strategy_knowledge.v3",
        "dataset_evidence": [
            {
                "tool": "get_strategy_knowledge",
                "result": {
                    "schema": "strategy_knowledge.v3",
                    "coverage": {"complete": True},
                    "action_facts": [{"action": "train_siege_tank"}],
                },
            }
        ],
    }
    observation = ToolObservation(
        tool="sc2_knowledge",
        args={"question_id": "Q1"},
        result={"knowledge_run": structured},
        ok=True,
    )

    runs = _knowledge_runs_for_optimizer(
        {"knowledge_used": [{"question_id": "Q1", "finding": "summary"}]},
        [observation],
    )

    assert runs == [structured]
    assert runs[0]["dataset_evidence"][0]["result"]["action_facts"] == [
        {"action": "train_siege_tank"}
    ]


def test_optimizer_withholds_prose_knowledge_without_deterministic_packet() -> None:
    runs = _knowledge_runs_for_optimizer(
        {"knowledge_used": [{"question_id": "Q1", "finding": "LLM-only claim"}]},
        [],
    )

    assert runs == [
        {
            "question_id": "Q1",
            "question": "",
            "answer": "",
            "ok": False,
            "error": (
                "verified deterministic packet unavailable; "
                "the prose finding was withheld"
            ),
        }
    ]


def test_optimizer_includes_outer_generation_retry_feedback(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    prompts: list[str] = []

    def fake_llm(prompt: str, **kwargs):
        if "You are validating a strategy patch" in prompt:
            return _semantic_payload()
        prompts.append(prompt)
        return {
            "action": "draft_candidate",
            "patches": [
                _patch(
                    document,
                    "main_attack_gate",
                    "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
                    "This changes the readiness rule selected by the hypothesis.",
                )
            ],
            "expected_effect": "earlier first attack",
            "main_risk": "smaller force",
        }

    monkeypatch.setattr("evol_agent.core.optimization_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm", fake_llm
    )
    result, improvement, _obs, _errors, _events = run_optimization_agent_loop(
        strategy_name="tank",
        race="terran",
        battle_analysis=_decision_analysis(),
        skill_texts={"strategy.md": TANK_STRATEGY},
        initial_tool_observations=[],
        retry_feedback=["Candidate-generation attempt 1/4 failed: missing dependency"],
    )

    assert result.ok is True
    assert improvement is not None
    assert "Candidate-generation attempt 1/4 failed" in prompts[0]


def test_apply_patch_allows_five_detail_replacements() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    targets = [
        "main_attack_gate",
        "recovery_and_cleanup",
        "production",
        "pre_attack_army_posture",
        "ultimate_goal",
    ]
    operations = [
        {
            "op": "replace_detail",
            "target": item.id,
            "expected_old_hash": paragraph_hash(item.value),
            "value": f"Updated instruction for {item.title} while keeping the Marine-Tank plan.",
        }
        for item in document.details
        if item.id in targets
    ]
    assert len(operations) == 5
    patched, changes = document.apply_patch(operations)
    assert len(changes) == 5
    StrategyDocument.parse(patched)


def test_apply_patch_rejects_duplicate_targets() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    gate = next(item for item in document.details if item.id == "main_attack_gate")
    operation = {
        "op": "replace_detail",
        "target": gate.id,
        "expected_old_hash": paragraph_hash(gate.value),
        "value": "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
    }
    try:
        document.apply_patch([operation, dict(operation)])
        raise AssertionError("duplicate targets should be rejected")
    except ValueError as exc:
        assert "more than once" in str(exc)


def test_apply_patch_rejects_wrong_parent_hash() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    try:
        document.apply_patch(
            [
                {
                    "op": "replace_detail",
                    "target": "main_attack_gate",
                    "expected_old_hash": "deadbeefdead",
                    "value": "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
                }
            ]
        )
        raise AssertionError("wrong hash should be rejected")
    except ValueError as exc:
        assert "precondition hash" in str(exc)


def test_optimizer_prompt_does_not_select_a_plan() -> None:
    prompt = build_candidate_prompt(
        strategy_name="tank",
        race="terran",
        battle_analysis=_decision_analysis(),
        skill_texts={"strategy.md": TANK_STRATEGY},
        tool_observations=[],
        validation_errors=[],
        candidate=None,
        knowledge_mode="enabled",
    )
    assert "Select one candidate plan" not in prompt
    assert "Select exactly one self-contained candidate plan" not in prompt
    assert "selected_plan_ids" not in prompt
    assert "Do not select among candidate plans" in prompt
    assert "Generate the entire replacement strategy.md" in prompt
    assert '"strategy_md"' in prompt
    assert '"patches"' not in prompt.split("Return one JSON object only:")[1]
    assert "coherent intervention package" in prompt
    assert "minimum_material_change" in prompt
    assert "cosmetic" in prompt
    assert "globally rewritten document must still make one causal change" in prompt
    assert "all technology and unit prerequisites are feasible" in prompt
    assert "complete relationship\namong strategy identity, development" in prompt
    assert "Executor capability manifest" in prompt
    assert "production_target_audit" not in prompt.split("Return one JSON object only:")[1]
    assert "Every unit that the complete candidate says to continue or resume" in prompt
    assert "numerical final count or cap" in prompt
    assert "no more than 200 supply" in prompt
    assert "workers plus every final combat/support unit" in prompt
    assert "strategy_contract.style" in prompt
    assert "contact" in prompt and "timing, fighting package" in prompt
    assert "fighting package" in prompt
    assert "fixed category ranking" in prompt
    assert "official Champion and the only" in prompt
    assert "Independent factual match summaries" not in prompt


def test_normalize_allows_five_hypothesis_patches() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(document, "main_attack_gate", "Begin with 36 Marines and 8 Siege Tanks.", "gate"),
        _patch(document, "recovery_and_cleanup", "Rebuild to 36 Marines and 8 Siege Tanks.", "recovery"),
        _patch(
            document,
            "production",
            "Prioritize Factories and Tanks before extra Marines.",
            "priority",
        ),
        _patch(
            document,
            "pre_attack_army_posture",
            "Keep the force gathered until the revised gate is met.",
            "posture",
        ),
        _patch(
            document,
            "ultimate_goal",
            "Continue toward 96 Marines, 20 Siege Tanks, and 44 SCVs after the revised gate.",
            "goal",
        ),
    ]
    payload, error = _normalize_optimizer_candidate(
        {"action": "draft_candidate", "patches": patches},
        parent_document=document,
    )
    assert error == ""
    assert payload is not None
    patched, changes = document.apply_patch(_patches_to_operations(payload["patches"]))
    assert len(changes) == 5
    StrategyDocument.parse(patched)


def test_normalize_rejects_summary_target() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    payload, error = _normalize_optimizer_candidate(
        {
            "action": "draft_candidate",
            "patches": [
                {
                    "target": "summary",
                    "expected_old_hash": paragraph_hash(document.summary),
                    "replacement": "A completely new identity.",
                    "why_required": "identity changed",
                }
            ],
        },
        parent_document=document,
    )
    assert payload is None
    assert "Summary" in error


def test_normalize_rejects_empty_or_multiline_replacement() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    gate = next(item for item in document.details if item.id == "main_attack_gate")
    empty, empty_error = _normalize_optimizer_candidate(
        {
            "patches": [
                {
                    "target": "main_attack_gate",
                    "expected_old_hash": paragraph_hash(gate.value),
                    "replacement": "",
                    "why_required": "needed",
                }
            ]
        },
        parent_document=document,
    )
    assert empty is None
    assert "non-empty" in empty_error

    multiline, multiline_error = _normalize_optimizer_candidate(
        {
            "patches": [
                {
                    "target": "main_attack_gate",
                    "expected_old_hash": paragraph_hash(gate.value),
                    "replacement": "First line.\nSecond line.",
                    "why_required": "needed",
                }
            ]
        },
        parent_document=document,
    )
    assert multiline is None
    assert "one non-empty line" in multiline_error


def test_normalize_requires_why_required() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    gate = next(item for item in document.details if item.id == "main_attack_gate")
    payload, error = _normalize_optimizer_candidate(
        {
            "patches": [
                {
                    "target": "main_attack_gate",
                    "expected_old_hash": paragraph_hash(gate.value),
                    "replacement": "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
                    "why_required": "",
                }
            ]
        },
        parent_document=document,
    )
    assert payload is None
    assert "why_required" in error


def test_retry_can_add_a_dependency_patch(monkeypatch) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    gate = next(item for item in document.details if item.id == "main_attack_gate")
    recovery = next(item for item in document.details if item.id == "recovery_and_cleanup")
    calls: list[str] = []
    optimizer_calls = 0

    def fake_llm(prompt: str, **kwargs):
        nonlocal optimizer_calls
        calls.append(prompt)
        if "You are validating a strategy patch" in prompt:
            return _semantic_payload()
        optimizer_calls += 1
        patches = [
            _patch(
                document,
                "main_attack_gate",
                "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
                "This paragraph defines the readiness rule.",
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
            assert "post-engagement behavior may change only" in prompt
            assert "Regenerate the whole strategy.md" in prompt
        return {
            "action": "draft_candidate" if optimizer_calls == 1 else "revise_candidate",
            "patches": patches,
            "expected_effect": "earlier first attack",
            "main_risk": "smaller force",
        }

    monkeypatch.setattr("evol_agent.core.optimization_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm", fake_llm
    )
    result, improvement, _obs, _errors, events = run_optimization_agent_loop(
        strategy_name="tank",
        race="terran",
        battle_analysis=_decision_analysis(),
        skill_texts={"strategy.md": TANK_STRATEGY},
        initial_tool_observations=[],
    )
    assert result.ok
    assert improvement is not None
    assert len([item for item in calls if "You are validating a strategy patch" not in item]) == 2
    assert events[-1]["llm_calls"] == 2
    assert paragraph_hash(gate.value) != paragraph_hash(
        next(item for item in StrategyDocument.parse(improvement.files["strategy.md"]).details if item.id == "main_attack_gate").value
    )
    assert next(
        item.value
        for item in StrategyDocument.parse(improvement.files["strategy.md"]).details
        if item.id == "recovery_and_cleanup"
    ) == recovery.value
    del recovery


def test_blocking_semantic_retry_exhaustion_uses_latest_candidate(
    monkeypatch,
) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    optimizer_calls = 0

    def fake_llm(prompt: str, **kwargs):
        nonlocal optimizer_calls
        if "You are validating a strategy patch" in prompt:
            return _semantic_payload(
                valid=False,
                errors=[
                    {
                        "type": "missing_dependency",
                        "location": "Recovery and Cleanup",
                        "description": "the retry still leaves a dependency warning",
                        "severity": "blocking",
                    }
                ],
            )
        optimizer_calls += 1
        return {
            "action": "draft_candidate" if optimizer_calls == 1 else "revise_candidate",
            "patches": [
                _patch(
                    document,
                    "main_attack_gate",
                    f"Begin the planned attack with {36 - optimizer_calls} Marines and 8 Siege Tanks.",
                    "This changes the readiness rule selected by the hypothesis.",
                )
            ],
            "expected_effect": "earlier first attack",
            "main_risk": "smaller force",
        }

    monkeypatch.setattr("evol_agent.core.optimization_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm", fake_llm
    )
    result, improvement, _obs, errors, events = run_optimization_agent_loop(
        strategy_name="tank",
        race="terran",
        battle_analysis=_decision_analysis(),
        skill_texts={"strategy.md": TANK_STRATEGY},
        initial_tool_observations=[],
    )

    assert result.ok is False
    assert improvement is None
    assert optimizer_calls == 5
    assert errors
    assert events[-1]["valid"] is False


def test_underpowered_semantic_retry_exhaustion_uses_latest_candidate(
    monkeypatch,
) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    optimizer_calls = 0

    def fake_llm(prompt: str, **kwargs):
        nonlocal optimizer_calls
        if "You are validating a strategy patch" in prompt:
            return _semantic_payload(
                valid=False,
                errors=[
                    {
                        "type": "underpowered_implementation",
                        "location": "Main Attack Gate",
                        "description": "the candidate is executable but may be too weak",
                        "severity": "blocking",
                    }
                ],
            )
        optimizer_calls += 1
        return {
            "action": "draft_candidate" if optimizer_calls == 1 else "revise_candidate",
            "patches": [
                _patch(
                    document,
                    "main_attack_gate",
                    f"Begin the planned attack with {36 - optimizer_calls} Marines and 8 Siege Tanks.",
                    "This changes the readiness rule selected by the hypothesis.",
                )
            ],
            "expected_effect": "earlier first attack",
            "main_risk": "smaller force",
        }

    monkeypatch.setattr("evol_agent.core.optimization_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm", fake_llm
    )
    result, improvement, _obs, errors, events = run_optimization_agent_loop(
        strategy_name="tank",
        race="terran",
        battle_analysis=_decision_analysis(),
        skill_texts={"strategy.md": TANK_STRATEGY},
        initial_tool_observations=[],
    )

    assert result.ok is False
    assert improvement is None
    assert optimizer_calls == 5
    assert errors
    assert events[-1]["valid"] is False


def test_mixed_underpowered_and_blocking_semantic_errors_use_latest_candidate(
    monkeypatch,
) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    optimizer_calls = 0

    def fake_llm(prompt: str, **kwargs):
        nonlocal optimizer_calls
        if "You are validating a strategy patch" in prompt:
            return _semantic_payload(
                valid=False,
                errors=[
                    {
                        "type": "underpowered_implementation",
                        "location": "Main Attack Gate",
                        "description": "the intended change is too weak",
                        "severity": "blocking",
                    },
                    {
                        "type": "missing_dependency",
                        "location": "Recovery and Cleanup",
                        "description": "the follow-up transition is missing",
                        "severity": "blocking",
                    },
                ],
            )
        optimizer_calls += 1
        return {
            "action": "draft_candidate" if optimizer_calls == 1 else "revise_candidate",
            "patches": [
                _patch(
                    document,
                    "main_attack_gate",
                    f"Begin the planned attack with {36 - optimizer_calls} Marines and 8 Siege Tanks.",
                    "This changes the readiness rule selected by the hypothesis.",
                )
            ],
            "expected_effect": "earlier first attack",
            "main_risk": "smaller force",
        }

    monkeypatch.setattr("evol_agent.core.optimization_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm", fake_llm
    )
    result, improvement, _obs, errors, events = run_optimization_agent_loop(
        strategy_name="tank",
        race="terran",
        battle_analysis=_decision_analysis(),
        skill_texts={"strategy.md": TANK_STRATEGY},
        initial_tool_observations=[],
    )

    assert result.ok is False
    assert improvement is None
    assert optimizer_calls == 5
    assert any("underpowered_implementation" in error for error in errors)
    assert any("missing_dependency" in error for error in errors)
    assert events[-1]["valid"] is False


def test_basic_retry_exhaustion_uses_latest_generated_candidate(
    monkeypatch,
) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    optimizer_calls = 0

    def fake_llm(prompt: str, **kwargs):
        nonlocal optimizer_calls
        optimizer_calls += 1
        return {
            "action": "draft_candidate" if optimizer_calls == 1 else "revise_candidate",
            "patches": [
                _patch(
                    document,
                    "main_attack_gate",
                    (
                        f"At revision {optimizer_calls}, address the attacking army "
                        "by unit tags before issuing the attack."
                    ),
                    "This changes the readiness rule selected by the hypothesis.",
                )
            ],
            "expected_effect": "earlier first attack",
            "main_risk": "smaller force",
        }

    monkeypatch.setattr("evol_agent.core.optimization_agent_loop.call_json_llm", fake_llm)
    result, improvement, _obs, errors, events = run_optimization_agent_loop(
        strategy_name="tank",
        race="terran",
        battle_analysis=_decision_analysis(),
        skill_texts={"strategy.md": TANK_STRATEGY},
        initial_tool_observations=[],
    )

    assert result.ok is False
    assert improvement is None
    assert optimizer_calls == 5
    assert errors
    assert events[-1]["valid"] is False


def test_optimizer_ignores_one_unchanged_patch_and_uses_other_generated_changes(
    monkeypatch,
) -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    ultimate_goal = next(
        item for item in document.details if item.id == "ultimate_goal"
    )
    optimizer_calls = 0

    def fake_llm(prompt: str, **kwargs):
        nonlocal optimizer_calls
        if "You are validating a strategy patch" in prompt:
            return _semantic_payload()
        optimizer_calls += 1
        return {
            "action": "draft_candidate",
            "patches": [
                _patch(
                    document,
                    "main_attack_gate",
                    "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
                    "This changes the readiness rule selected by the hypothesis.",
                ),
                _patch(
                    document,
                    "ultimate_goal",
                    ultimate_goal.value,
                    "The existing end state already supports the revised gate.",
                ),
            ],
            "expected_effect": "earlier first attack",
            "main_risk": "smaller force",
        }

    monkeypatch.setattr("evol_agent.core.optimization_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm", fake_llm
    )
    result, improvement, _obs, errors, events = run_optimization_agent_loop(
        strategy_name="tank",
        race="terran",
        battle_analysis=_decision_analysis(),
        skill_texts={"strategy.md": TANK_STRATEGY},
        initial_tool_observations=[],
    )

    assert result.ok is True
    assert improvement is not None
    assert optimizer_calls == 1
    assert errors == []
    assert any(
        event.get("action") == "ignore_unchanged_patches"
        and event.get("ignored_targets") == ["ultimate_goal"]
        for event in events
    )
    assert "36 Marines and 8 Siege Tanks" in improvement.files["strategy.md"]
    assert ultimate_goal.value in improvement.files["strategy.md"]


def test_optimizer_accepts_complete_strategy_document_and_derives_changes(
    monkeypatch,
) -> None:
    complete_candidate = TANK_STRATEGY.replace(
        "45 completed and living Marines and 10 completed and living Siege Tanks",
        "36 completed and living Marines and 8 completed and living Siege Tanks",
    ).replace(
        "rebuild to 45 Marines and 10 Siege Tanks",
        "rebuild to 36 Marines and 8 Siege Tanks",
    )

    def fake_llm(prompt: str, **kwargs):
        if "You are validating a strategy patch" in prompt:
            return _semantic_payload()
        return {
            "action": "draft_candidate",
            "strategy_md": complete_candidate,
            "inheritance": {
                "keep": [
                    {
                        "item": "Marine-Siege Tank gathered push",
                        "reason": "combat style is preserved",
                    }
                ],
                "revise": [
                    {
                        "item": "attack readiness",
                        "reason": "the selected hypothesis requires earlier contact",
                    }
                ],
                "remove": [],
            },
            "expected_effect": "earlier first attack",
            "main_risk": "smaller first force",
        }

    monkeypatch.setattr("evol_agent.core.optimization_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.strategy_patch_validator.call_json_llm", fake_llm
    )

    result, improvement, _obs, errors, _events = run_optimization_agent_loop(
        strategy_name="tank",
        race="terran",
        battle_analysis=_decision_analysis(),
        skill_texts={"strategy.md": TANK_STRATEGY},
        initial_tool_observations=[],
    )

    assert result.ok is True
    assert errors == []
    assert improvement is not None
    assert improvement.raw["operations"] == []
    assert "36 completed and living Marines" in improvement.files["strategy.md"]
    changed_targets = {
        item["target"] for item in improvement.raw["document_changes"]
    }
    assert "main_attack_gate" in changed_targets
    assert "recovery_and_cleanup" not in changed_targets


def test_unrelated_paragraphs_and_titles_remain() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    operations = [
        {
            "op": "replace_detail",
            "target": "main_attack_gate",
            "expected_old_hash": paragraph_hash(
                next(item for item in document.details if item.id == "main_attack_gate").value
            ),
            "value": "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
        },
        {
            "op": "replace_detail",
            "target": "recovery_and_cleanup",
            "expected_old_hash": paragraph_hash(
                next(item for item in document.details if item.id == "recovery_and_cleanup").value
            ),
            "value": "If progress stalls, withdraw and rebuild to 36 Marines and 8 Siege Tanks.",
        },
    ]
    patched, _changes = document.apply_patch(operations)
    candidate = StrategyDocument.parse(patched)
    assert [item.title for item in candidate.details] == [item.title for item in document.details]
    assert candidate.summary == document.summary
    changed = {"main_attack_gate", "recovery_and_cleanup"}
    for parent, child in zip(document.details, candidate.details):
        assert parent.id == child.id
        if parent.id not in changed:
            assert child.value == parent.value
    assert validate_strategy_markdown(patched, race="terran") is None
