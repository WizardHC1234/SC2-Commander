from __future__ import annotations

from typing import Any

from .config import (
    DEFAULT_OPTIMIZATION_MODEL,
    MAX_VALIDATION_RETRIES,
    OPTIMIZATION_ENABLE_REASONING,
)
from .llm import call_json_llm
from .prompts import build_candidate_prompt
from .candidate_critic import critique_candidate_contract
from .types import BattleAnalysis, EvolImprovement, ToolObservation, ValidationResult
from ..optimization.strategy_document import StrategyDocument, paragraph_hash
from ..validation import validate_improvement


def run_optimization_agent_loop(
    *,
    strategy_name: str,
    race: str,
    battle_analysis: BattleAnalysis,
    skill_texts: dict[str, str],
    initial_tool_observations: list[ToolObservation],
    knowledge_mode: str = "enabled",
    model: str = "",
    prefix: str = "  ",
    capability_manifest: dict[str, Any] | None = None,
) -> tuple[
    ValidationResult,
    EvolImprovement | None,
    list[ToolObservation],
    list[str],
    list[dict[str, Any]],
]:
    """Generate one strategy candidate and retry deterministic validation.

    Strategic quality is intentionally not validated here. It is measured by
    playing the candidate; this loop checks only that Commander can load and
    execute the strategy document.
    """
    model = str(model or "").strip() or DEFAULT_OPTIMIZATION_MODEL
    capability_manifest = capability_manifest or {}
    observations = list(initial_tool_observations)
    validation_errors: list[str] = []
    events: list[dict[str, Any]] = []
    candidate: dict[str, Any] | None = None
    last_improvement: EvolImprovement | None = None
    valid_plan_ids = {
        str(item.get("id") or "").strip()
        for item in (battle_analysis.raw.get("candidate_plans") or [])
        if isinstance(item, dict)
    }
    plans_by_id = {
        str(item.get("id") or "").strip(): item
        for item in (battle_analysis.raw.get("candidate_plans") or [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    parent_text = str(skill_texts.get("strategy.md") or "")
    try:
        parent_document = StrategyDocument.parse(parent_text)
    except ValueError as exc:
        error = f"parent strategy.md cannot be patched: {exc}"
        return ValidationResult(ok=False, error=error), None, observations, [error], events

    # Normal path: the single cross-match reasoning call already selected one
    # plan and supplied complete paragraph replacements. Apply it locally and
    # avoid a second LLM call. Knowledge-dependent plans still use the fallback
    # call below so verified facts can be incorporated.
    plans = [
        item
        for item in (battle_analysis.raw.get("candidate_plans") or [])
        if isinstance(item, dict)
    ]
    if len(plans) == 1 and not observations:
        plan = plans[0]
        details_by_id = {item.id: item for item in parent_document.details}
        operations: list[dict[str, str]] = []
        selected_changes: list[dict[str, str]] = []
        for change in plan.get("changes") or []:
            if not isinstance(change, dict):
                continue
            target = str(change.get("target_paragraph_id") or "").strip()
            value = str(change.get("candidate_rule") or "").strip()
            current = details_by_id.get(target)
            if not target or not value or current is None:
                operations = []
                break
            operations.append(
                {
                    "op": "replace_detail",
                    "target": target,
                    "expected_old_hash": paragraph_hash(current.value),
                    "value": value,
                }
            )
            selected_changes.append(
                {
                    "source_plan_id": str(plan.get("id") or "D1"),
                    "problem_id": str(
                        (plan.get("addresses_problem_ids") or ["P1"])[0]
                    ),
                    "change": value,
                    "why": str(change.get("why_required") or ""),
                }
            )
        if operations:
            rationale = {
                "hypothesis": str(plan.get("hypothesis") or plan.get("name") or ""),
                "primary_lever": str(plan.get("primary_lever") or "other"),
                "predictions": list(plan.get("predictions") or []),
                "disproof_conditions": list(plan.get("disproof_conditions") or []),
                "capability_mapping": dict(plan.get("capability_mapping") or {}),
                "preserved_strength": str(battle_analysis.raw.get("winning_mechanism") or ""),
                "selected_plan_ids": [str(plan.get("id") or "D1")],
                "overall_assessment": str(battle_analysis.raw.get("action_reason") or ""),
                "selected_changes": selected_changes,
                "primary_change": str(plan.get("name") or "paragraph patch"),
                "expected_effect": str(plan.get("expected_benefit") or ""),
                "main_risk": str(plan.get("risk_to_winning_mechanism") or ""),
            }
            direct_candidate = {
                "action": "apply_analyzed_plan",
                "rationale": rationale,
                "operations": operations,
            }
            direct_errors = critique_candidate_contract(
                rationale,
                capability_manifest=capability_manifest,
                selected_plan=plan,
            )
            try:
                patched_text, paragraph_changes = parent_document.apply_patch(operations)
            except ValueError as exc:
                direct_errors.append(str(exc))
                patched_text, paragraph_changes = "", []
            if not direct_errors:
                direct_result = validate_improvement(
                    files={"strategy.md": patched_text}, race=race
                )
                if direct_result.ok:
                    direct_candidate["paragraph_changes"] = paragraph_changes
                    direct_candidate["files"] = dict(direct_result.files or {})
                    improvement = EvolImprovement(
                        analysis=rationale,
                        files=dict(direct_result.files or {}),
                        raw=direct_candidate,
                    )
                    events.append(
                        {
                            "attempt": 0,
                            "action": "apply_analyzed_plan",
                            "valid": True,
                            "llm_calls": 0,
                            "paragraph_changes": paragraph_changes,
                        }
                    )
                    print(
                        f"{prefix}OptimizationAgent: applied analyzed paragraph patch locally",
                        flush=True,
                    )
                    return direct_result, improvement, observations, [], events
                direct_errors.append(direct_result.error)
            candidate = direct_candidate
            validation_errors.extend(direct_errors)

    print(
        f"{prefix}OptimizationAgent: generating one candidate for "
        f"{race}/{strategy_name}",
        flush=True,
    )
    for attempt in range(1, MAX_VALIDATION_RETRIES + 2):
        action = call_json_llm(
            build_candidate_prompt(
                strategy_name=strategy_name,
                race=race,
                battle_analysis=battle_analysis,
                skill_texts=skill_texts,
                tool_observations=observations,
                validation_errors=validation_errors,
                candidate=candidate,
                knowledge_mode=knowledge_mode,
                capability_manifest=capability_manifest,
            ),
            model=model,
            is_reasoning=OPTIMIZATION_ENABLE_REASONING,
        )
        if not isinstance(action, dict):
            error = "OptimizationAgent returned no JSON object"
            validation_errors.append(error)
            events.append({"attempt": attempt, "action": "invalid", "error": error})
            continue

        name = str(action.get("action") or "").strip()
        operations = action.get("operations")
        rationale = (
            action.get("rationale")
            if isinstance(action.get("rationale"), dict)
            else action.get("analysis")
            if isinstance(action.get("analysis"), dict)
            else {}
        )
        normalized = {
            "action": name or "draft_candidate",
            "rationale": dict(rationale),
            "operations": operations if isinstance(operations, list) else [],
        }
        candidate = normalized

        selected_plan_ids = [
            str(value).strip()
            for value in normalized["rationale"].get("selected_plan_ids") or []
            if str(value).strip()
        ]
        if (
            len(selected_plan_ids) != 1
            or selected_plan_ids[0] not in valid_plan_ids
        ):
            error = (
                "candidate rationale.selected_plan_ids must contain exactly one of: "
                f"{', '.join(sorted(valid_plan_ids)) or '(none)'}"
            )
            validation_errors.append(error)
            events.append(
                {"attempt": attempt, "action": name or "draft_candidate", "valid": False, "error": error}
            )
            continue

        critic_errors = critique_candidate_contract(
            normalized["rationale"],
            capability_manifest=capability_manifest,
            selected_plan=plans_by_id.get(selected_plan_ids[0]),
        )
        selected_plan = plans_by_id.get(selected_plan_ids[0]) or {}
        planned_targets = {
            str(item.get("target_paragraph_id") or "").strip()
            for item in selected_plan.get("changes") or []
            if isinstance(item, dict)
            and str(item.get("target_paragraph_id") or "").strip()
        }
        operation_targets = {
            str(item.get("target") or "").strip()
            for item in normalized["operations"]
            if isinstance(item, dict) and str(item.get("target") or "").strip()
        }
        if planned_targets and operation_targets != planned_targets:
            critic_errors.append(
                "candidate operations must modify exactly the selected plan paragraphs: "
                + ", ".join(sorted(planned_targets))
            )
        if critic_errors:
            error = "; ".join(critic_errors)
            validation_errors.append(error)
            events.append(
                {
                    "attempt": attempt,
                    "action": "candidate_critic",
                    "valid": False,
                    "error": error,
                }
            )
            if attempt <= MAX_VALIDATION_RETRIES:
                print(
                    f"{prefix}OptimizationAgent: contract validation failed; "
                    f"retrying ({attempt}/{MAX_VALIDATION_RETRIES}): {error}",
                    flush=True,
                )
            continue

        try:
            patched_text, paragraph_changes = parent_document.apply_patch(
                normalized["operations"],
            )
        except ValueError as exc:
            error = str(exc)
            validation_errors.append(error)
            events.append(
                {
                    "attempt": attempt,
                    "action": "apply_strategy_patch",
                    "valid": False,
                    "error": error,
                }
            )
            continue

        normalized["paragraph_changes"] = paragraph_changes
        normalized["files"] = {"strategy.md": patched_text}
        last_improvement = EvolImprovement(
            analysis=normalized["rationale"],
            files=normalized["files"],
            raw=normalized,
        )

        result = validate_improvement(
            files=last_improvement.files,
            race=race,
        )
        events.append(
            {
                "attempt": attempt,
                "action": name or "draft_candidate",
                "valid": result.ok,
                "error": result.error,
            }
        )
        if result.ok:
            last_improvement.files = result.files or last_improvement.files
            last_improvement.raw["files"] = dict(last_improvement.files)
            print(
                f"{prefix}OptimizationAgent: candidate passed basic validation",
                flush=True,
            )
            return result, last_improvement, observations, validation_errors, events

        validation_errors.append(result.error)
        if attempt <= MAX_VALIDATION_RETRIES:
            print(
                f"{prefix}OptimizationAgent: basic validation failed; "
                f"retrying ({attempt}/{MAX_VALIDATION_RETRIES}): {result.error}",
                flush=True,
            )

    error = validation_errors[-1] if validation_errors else "OptimizationAgent exhausted"
    return (
        ValidationResult(ok=False, error=error),
        last_improvement,
        observations,
        validation_errors,
        events,
    )
