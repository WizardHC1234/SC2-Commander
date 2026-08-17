from __future__ import annotations

from typing import Any

from .config import (
    DEFAULT_OPTIMIZATION_MODEL,
    MAX_VALIDATION_RETRIES,
    OPTIMIZATION_ENABLE_REASONING,
)
from .llm import call_json_llm
from .prompts import build_candidate_prompt
from .strategy_patch_validator import (
    validate_strategy_patch_semantics,
    validate_strategy_patch_structure,
)
from .types import BattleAnalysis, EvolImprovement, ToolObservation, ValidationResult
from ..optimization.strategy_document import StrategyDocument
from ..validation import validate_improvement


def _unwrap_candidate(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    raw = result.get("candidate") if isinstance(result.get("candidate"), dict) else result
    return raw if isinstance(raw, dict) else None


def extract_final_cross_match_decision(battle_analysis: BattleAnalysis) -> dict[str, Any]:
    raw = dict(battle_analysis.raw or {})
    plans = [item for item in (raw.get("candidate_plans") or []) if isinstance(item, dict)]
    first_plan = plans[0] if plans else {}
    priority = raw.get("priority_problem")
    if isinstance(priority, str) and priority.strip():
        priority = {"problem": priority.strip(), "evidence": []}
    if not isinstance(priority, dict) or not str(priority.get("problem") or "").strip():
        problems = raw.get("problems") or []
        if problems and isinstance(problems[0], dict):
            priority = problems[0]
        else:
            priority = {}
    plan = raw.get("plan")
    if isinstance(plan, str) and plan.strip():
        plan = {"direction": plan.strip()}
    if not isinstance(plan, dict):
        plan = {}
    direction = str(plan.get("direction") or first_plan.get("name") or "").strip()
    plan = {**plan, "direction": direction}
    hypothesis = str(raw.get("hypothesis") or first_plan.get("hypothesis") or "").strip()
    strengths = raw.get("strengths_to_preserve")
    if not isinstance(strengths, list):
        strengths = raw.get("wins_to_preserve") if isinstance(raw.get("wins_to_preserve"), list) else []
    knowledge_used = raw.get("knowledge_used") if isinstance(raw.get("knowledge_used"), list) else []
    return {
        "strengths_to_preserve": strengths,
        "priority_problem": priority,
        "hypothesis": hypothesis,
        "plan": plan,
        "next_action": str(raw.get("next_action") or ""),
        "knowledge_used": knowledge_used,
    }


def _knowledge_runs_for_optimizer(
    decision: dict[str, Any],
    observations: list[ToolObservation],
) -> list[dict[str, Any]]:
    used = [item for item in (decision.get("knowledge_used") or []) if isinstance(item, dict)]
    if used:
        return [
            {
                "question_id": str(item.get("question_id") or ""),
                "question": str(item.get("question") or ""),
                "answer": str(item.get("finding") or item.get("answer") or ""),
                "ok": True,
            }
            for item in used
            if str(item.get("finding") or item.get("answer") or "").strip()
        ]
    runs: list[dict[str, Any]] = []
    for observation in observations:
        if not observation.ok:
            continue
        args = observation.args if isinstance(observation.args, dict) else {}
        runs.append(
            {
                "question_id": str(args.get("question_id") or ""),
                "question": str(args.get("question") or ""),
                "answer": str(observation.summary or ""),
                "ok": True,
            }
        )
    return runs


def _normalize_optimizer_candidate(
    raw: dict[str, Any],
    *,
    parent_document: StrategyDocument,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "optimizer returned no JSON object"
    action = str(raw.get("action") or "draft_candidate").strip() or "draft_candidate"
    if action not in {"draft_candidate", "revise_candidate"}:
        return None, "action must be draft_candidate or revise_candidate"

    patches = raw.get("patches")
    if not isinstance(patches, list) or not patches:
        return None, "optimizer patches must be a non-empty list"

    detail_ids = {item.id for item in parent_document.details}
    seen: set[str] = set()
    normalized_patches: list[dict[str, str]] = []
    for item in patches:
        if not isinstance(item, dict):
            return None, "each patch must be an object"
        target = str(item.get("target") or "").strip()
        replacement = str(item.get("replacement") or item.get("value") or "").strip()
        why_required = str(item.get("why_required") or "").strip()
        expected_old_hash = str(item.get("expected_old_hash") or "").strip()
        if target in {"", "summary"} or str(item.get("op") or "") == "replace_summary":
            return None, "Optimizer may not modify # Summary"
        if not target:
            return None, "each patch requires target"
        if target in seen:
            return None, f"candidate modifies paragraph {target!r} more than once"
        seen.add(target)
        if target not in detail_ids:
            allowed = ", ".join(sorted(detail_ids))
            return None, f"unknown strategy detail {target!r}; allowed targets: {allowed}"
        if not expected_old_hash:
            return None, f"patch {target!r} requires expected_old_hash"
        if not replacement:
            return None, f"patch {target!r} replacement must be a non-empty line"
        if "\n" in replacement:
            return None, f"candidate paragraph {target!r} must be one non-empty line"
        if not why_required:
            return None, f"patch {target!r} requires why_required"
        normalized_patches.append(
            {
                "target": target,
                "expected_old_hash": expected_old_hash,
                "replacement": replacement,
                "why_required": why_required,
            }
        )

    preserved = [
        str(item).strip()
        for item in (raw.get("preserved_strengths") or [])
        if str(item).strip()
    ]
    return (
        {
            "action": action,
            "patches": normalized_patches,
            "expected_effect": str(raw.get("expected_effect") or "").strip(),
            "main_risk": str(raw.get("main_risk") or "").strip(),
            "preserved_strengths": preserved,
        },
        "",
    )


def _patches_to_operations(patches: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "op": "replace_detail",
            "target": item["target"],
            "expected_old_hash": item["expected_old_hash"],
            "value": item["replacement"],
        }
        for item in patches
    ]


def _candidate_rationale(
    *,
    decision: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    priority = decision.get("priority_problem") or {}
    problem = (
        str(priority.get("problem") or "").strip()
        if isinstance(priority, dict)
        else str(priority).strip()
    )
    plan = decision.get("plan") if isinstance(decision.get("plan"), dict) else {}
    direction = str(plan.get("direction") or "").strip()
    hypothesis = str(decision.get("hypothesis") or "").strip()
    expected_effect = str(candidate.get("expected_effect") or "").strip()
    main_risk = str(candidate.get("main_risk") or "").strip()
    strengths = []
    for item in decision.get("strengths_to_preserve") or []:
        if isinstance(item, dict) and str(item.get("pattern") or "").strip():
            strengths.append(str(item.get("pattern") or "").strip())
        elif str(item).strip():
            strengths.append(str(item).strip())
    patches = list(candidate.get("patches") or [])
    return {
        "hypothesis": hypothesis,
        "priority_problem": problem,
        "plan_direction": direction,
        "primary_lever": "other",
        "predictions": [expected_effect] if expected_effect else [
            "candidate matches test the supplied hypothesis"
        ],
        "disproof_conditions": [main_risk] if main_risk else [
            "match outcomes do not improve"
        ],
        "capability_mapping": {
            "macro_actions": [],
            "changed_macro_actions": [],
            "army_controls": [],
            "information_controls": [],
            "runtime_dependencies": [],
            "unsupported_dependencies": [],
        },
        "preserved_strength": strengths[0] if strengths else "",
        "strengths_to_preserve": list(decision.get("strengths_to_preserve") or []),
        "preserved_strengths": list(candidate.get("preserved_strengths") or strengths),
        "selected_plan_ids": ["D1"],
        "overall_assessment": direction,
        "selected_changes": [
            {
                "source_plan_id": "D1",
                "problem_id": "P1",
                "target": item.get("target"),
                "change": item.get("replacement"),
                "why": item.get("why_required"),
            }
            for item in patches
            if isinstance(item, dict)
        ],
        "primary_change": direction,
        "expected_effect": expected_effect,
        "main_risk": main_risk,
        "patches": [
            {"target": item.get("target"), "why_required": item.get("why_required")}
            for item in patches
            if isinstance(item, dict)
        ],
    }


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
    """Implement one Cross-match hypothesis as strategy.md paragraph patches."""
    model = str(model or "").strip() or DEFAULT_OPTIMIZATION_MODEL
    capability_manifest = capability_manifest or {}
    observations = list(initial_tool_observations)
    validation_errors: list[str] = []
    events: list[dict[str, Any]] = []
    candidate: dict[str, Any] | None = None
    last_improvement: EvolImprovement | None = None
    parent_text = str(skill_texts.get("strategy.md") or "")
    try:
        parent_document = StrategyDocument.parse(parent_text)
    except ValueError as exc:
        error = f"parent strategy.md cannot be patched: {exc}"
        return ValidationResult(ok=False, error=error), None, observations, [error], events

    decision = extract_final_cross_match_decision(battle_analysis)
    if not decision["hypothesis"] or not str((decision.get("plan") or {}).get("direction") or "").strip():
        error = "Optimizer requires a Cross-match hypothesis and plan.direction"
        return ValidationResult(ok=False, error=error), None, observations, [error], events

    knowledge_runs = _knowledge_runs_for_optimizer(decision, observations)
    print(
        f"{prefix}OptimizationAgent: generating paragraph patches for "
        f"{race}/{strategy_name}",
        flush=True,
    )
    llm_calls = 0
    for attempt in range(1, MAX_VALIDATION_RETRIES + 2):
        llm_calls += 1
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
                decision=decision,
                knowledge_runs=knowledge_runs,
            ),
            model=model,
            is_reasoning=OPTIMIZATION_ENABLE_REASONING,
        )
        raw = _unwrap_candidate(action)
        if raw is None:
            error = "OptimizationAgent returned no JSON object"
            validation_errors.append(error)
            events.append({"attempt": attempt, "action": "invalid", "error": error, "llm_calls": llm_calls})
            continue

        normalized, error = _normalize_optimizer_candidate(
            raw, parent_document=parent_document
        )
        if normalized is None:
            candidate = raw
            validation_errors.append(error)
            events.append(
                {
                    "attempt": attempt,
                    "action": str(raw.get("action") or "draft_candidate"),
                    "valid": False,
                    "error": error,
                    "llm_calls": llm_calls,
                }
            )
            continue

        operations = _patches_to_operations(normalized["patches"])
        rationale = _candidate_rationale(decision=decision, candidate=normalized)
        structure_errors = validate_strategy_patch_structure(
            decision=decision,
            patches=normalized["patches"],
            parent_document=parent_document,
        )
        if structure_errors:
            error = "; ".join(structure_errors)
            candidate = normalized
            validation_errors.append(error)
            events.append(
                {
                    "attempt": attempt,
                    "action": "strategy_patch_structure",
                    "valid": False,
                    "error": error,
                    "llm_calls": llm_calls,
                }
            )
            if attempt <= MAX_VALIDATION_RETRIES:
                print(
                    f"{prefix}OptimizationAgent: patch structure failed; "
                    f"retrying ({attempt}/{MAX_VALIDATION_RETRIES}): {error}",
                    flush=True,
                )
            continue

        try:
            patched_text, paragraph_changes = parent_document.apply_patch(operations)
        except ValueError as exc:
            error = str(exc)
            candidate = normalized
            validation_errors.append(error)
            events.append(
                {
                    "attempt": attempt,
                    "action": "apply_strategy_patch",
                    "valid": False,
                    "error": error,
                    "llm_calls": llm_calls,
                }
            )
            continue

        semantic_errors = validate_strategy_patch_semantics(
            decision=decision,
            parent_text=parent_text,
            candidate_text=patched_text,
            patches=normalized["patches"],
            capability_manifest=capability_manifest,
            model=model,
        )
        if semantic_errors:
            error = "; ".join(semantic_errors)
            candidate = normalized
            validation_errors.append(error)
            events.append(
                {
                    "attempt": attempt,
                    "action": "strategy_patch_semantics",
                    "valid": False,
                    "error": error,
                    "llm_calls": llm_calls,
                }
            )
            if attempt <= MAX_VALIDATION_RETRIES:
                print(
                    f"{prefix}OptimizationAgent: patch semantics failed; "
                    f"retrying ({attempt}/{MAX_VALIDATION_RETRIES}): {error}",
                    flush=True,
                )
            continue

        payload = {
            **normalized,
            "rationale": rationale,
            "operations": operations,
            "paragraph_changes": paragraph_changes,
            "files": {"strategy.md": patched_text},
        }
        candidate = payload
        last_improvement = EvolImprovement(
            analysis=rationale,
            files=payload["files"],
            raw=payload,
        )
        result = validate_improvement(
            files=last_improvement.files,
            race=race,
        )
        events.append(
            {
                "attempt": attempt,
                "action": normalized["action"],
                "valid": result.ok,
                "error": result.error,
                "llm_calls": llm_calls,
                "paragraph_changes": paragraph_changes,
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
