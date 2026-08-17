from __future__ import annotations

from pathlib import Path

from evol_agent.core.optimization_agent_loop import (
    _normalize_optimizer_candidate,
    _patches_to_operations,
    run_optimization_agent_loop,
)
from evol_agent.core.prompts import build_candidate_prompt
from evol_agent.core.types import BattleAnalysis, ValidationResult
from evol_agent.optimization.strategy_document import StrategyDocument, paragraph_hash
from evol_agent.validation import validate_strategy_markdown


TANK_STRATEGY = Path("skills/terran/tank/strategy.md").read_text(encoding="utf-8")


def _decision_analysis(**overrides) -> BattleAnalysis:
    raw = {
        "next_action": "propose_strategy_patch",
        "hypothesis": "The current readiness design requires too much completed army before the first attack.",
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


def test_apply_patch_allows_five_detail_replacements() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    targets = [
        "main_attack_gate",
        "recovery_and_cleanup",
        "pre_attack_production_priority",
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
    assert "Independent factual match summaries" not in prompt


def test_normalize_allows_five_hypothesis_patches() -> None:
    document = StrategyDocument.parse(TANK_STRATEGY)
    patches = [
        _patch(document, "main_attack_gate", "Begin with 36 Marines and 8 Siege Tanks.", "gate"),
        _patch(document, "recovery_and_cleanup", "Rebuild to 36 Marines and 8 Siege Tanks.", "recovery"),
        _patch(
            document,
            "pre_attack_production_priority",
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

    def fake_llm(prompt: str, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return {
                "action": "draft_candidate",
                "patches": [
                    _patch(
                        document,
                        "main_attack_gate",
                        "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
                        "This paragraph defines the readiness rule.",
                    )
                ],
                "expected_effect": "earlier first attack",
                "main_risk": "smaller force",
            }
        assert "Recovery remains inconsistent" in prompt
        return {
            "action": "revise_candidate",
            "patches": [
                _patch(
                    document,
                    "main_attack_gate",
                    "Begin the planned attack with 36 Marines and 8 Siege Tanks.",
                    "This paragraph defines the readiness rule.",
                ),
                _patch(
                    document,
                    "recovery_and_cleanup",
                    "If progress stalls, withdraw and rebuild to 36 Marines and 8 Siege Tanks.",
                    "Recovery still used the old readiness threshold.",
                ),
            ],
            "expected_effect": "earlier first attack",
            "main_risk": "smaller force",
        }

    validate_calls = {"n": 0}

    def fake_validate(*, files, race):
        validate_calls["n"] += 1
        if validate_calls["n"] == 1:
            return ValidationResult(ok=False, error="Recovery remains inconsistent")
        return ValidationResult(ok=True, files=files)

    monkeypatch.setattr("evol_agent.core.optimization_agent_loop.call_json_llm", fake_llm)
    monkeypatch.setattr(
        "evol_agent.core.optimization_agent_loop.validate_improvement", fake_validate
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
    assert len(calls) == 2
    assert events[-1]["llm_calls"] == 2
    assert paragraph_hash(gate.value) != paragraph_hash(
        next(item for item in StrategyDocument.parse(improvement.files["strategy.md"]).details if item.id == "main_attack_gate").value
    )
    assert "36 Marines and 8 Siege Tanks" in next(
        item.value
        for item in StrategyDocument.parse(improvement.files["strategy.md"]).details
        if item.id == "recovery_and_cleanup"
    )
    del recovery


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
