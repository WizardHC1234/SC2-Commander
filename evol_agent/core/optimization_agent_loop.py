from __future__ import annotations

import re
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
from ..sc2_data_agent.bridge import find_knowledge_run_error, run_knowledge_query
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
        "mechanism_family": str(raw.get("mechanism_family") or "").strip(),
        "failure_mode_analysis": (
            dict(raw.get("failure_mode_analysis"))
            if isinstance(raw.get("failure_mode_analysis"), dict)
            else {}
        ),
        "priority_alignment": (
            dict(raw.get("priority_alignment"))
            if isinstance(raw.get("priority_alignment"), dict)
            else {}
        ),
        "mechanism_prediction": (
            dict(raw.get("mechanism_prediction"))
            if isinstance(raw.get("mechanism_prediction"), dict)
            else {}
        ),
        "retrieval_assessment": (
            dict(raw.get("retrieval_assessment"))
            if isinstance(raw.get("retrieval_assessment"), dict)
            else {}
        ),
        "plan": plan,
        "next_action": str(raw.get("next_action") or ""),
        "knowledge_used": knowledge_used,
    }


def _knowledge_runs_for_optimizer(
    decision: dict[str, Any],
    observations: list[ToolObservation],
) -> list[dict[str, Any]]:
    used = [item for item in (decision.get("knowledge_used") or []) if isinstance(item, dict)]
    observed_runs: dict[str, dict[str, Any]] = {}
    for observation in observations:
        if not observation.ok:
            continue
        result = observation.result if isinstance(observation.result, dict) else {}
        structured = result.get("knowledge_run")
        if not isinstance(structured, dict) or find_knowledge_run_error(structured):
            continue
        question_id = str(
            structured.get("question_id")
            or (observation.args or {}).get("question_id")
            or ""
        ).strip()
        if question_id:
            observed_runs[question_id] = dict(structured)
    if used:
        runs: list[dict[str, Any]] = []
        for item in used:
            question_id = str(item.get("question_id") or "").strip()
            if question_id in observed_runs:
                runs.append(observed_runs[question_id])
                continue
            runs.append(
                {
                    "question_id": question_id,
                    "question": str(item.get("question") or ""),
                    "answer": "",
                    "ok": False,
                    "error": (
                        "verified deterministic packet unavailable; "
                        "the prose finding was withheld"
                    ),
                }
            )
        return runs
    runs: list[dict[str, Any]] = []
    for observation in observations:
        if not observation.ok:
            continue
        result = observation.result if isinstance(observation.result, dict) else {}
        structured = result.get("knowledge_run")
        if isinstance(structured, dict) and not find_knowledge_run_error(structured):
            runs.append(dict(structured))
            continue
    return runs


def _candidate_knowledge_run(
    *,
    candidate_text: str,
    race: str,
    capability_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    """Build deterministic facts for every executable action named by a candidate."""
    macro_contract = capability_manifest.get("macro_contract")
    available = (
        macro_contract.get("available_actions")
        if isinstance(macro_contract, dict)
        else []
    )
    actions = [
        str(action)
        for action in (available or [])
        if str(action).strip()
        and re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(str(action))}(?![A-Za-z0-9_])",
            candidate_text,
            re.IGNORECASE,
        )
    ]
    if not actions:
        return None
    question = "Verify requirements for every executable action named by the complete candidate."
    run = run_knowledge_query(
        {
            "id": "QCANDIDATE",
            "question": question,
            "actions": actions,
            "needs": ["requirements"],
            "hypothesis_scope": "candidate_execution_feasibility",
        },
        race=race,
    )
    run["question"] = question
    return run


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
    raw_inheritance = raw.get("inheritance")
    inheritance = dict(raw_inheritance) if isinstance(raw_inheritance, dict) else {}

    def inheritance_items(key: str) -> list[dict[str, str]]:
        values = inheritance.get(key)
        if not isinstance(values, list):
            return []
        normalized: list[dict[str, str]] = []
        for value in values:
            if isinstance(value, dict):
                item = {
                    "item": str(
                        value.get("item")
                        or value.get("mechanism")
                        or value.get("change")
                        or ""
                    ).strip(),
                    "reason": str(
                        value.get("reason")
                        or value.get("evidence")
                        or value.get("why")
                        or ""
                    ).strip(),
                }
            else:
                item = {"item": str(value).strip(), "reason": ""}
            if item["item"]:
                normalized.append(item)
        return normalized

    normalized_inheritance = {
        "keep": inheritance_items("keep"),
        "revise": inheritance_items("revise"),
        "remove": inheritance_items("remove"),
    }
    if not any(normalized_inheritance.values()):
        # Compatibility fallback for older optimizer responses. Live prompts now
        # require an explicit ledger, but old checkpoints can still be resumed.
        normalized_inheritance = {
            "keep": [{"item": item, "reason": "preserved strength"} for item in preserved],
            "revise": [
                {
                    "item": item["target"],
                    "reason": item["why_required"],
                }
                for item in normalized_patches
            ],
            "remove": [],
        }
    return (
        {
            "action": action,
            "patches": normalized_patches,
            "expected_effect": str(raw.get("expected_effect") or "").strip(),
            "main_risk": str(raw.get("main_risk") or "").strip(),
            "preserved_strengths": preserved,
            "inheritance": normalized_inheritance,
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


def _drop_unchanged_patches(
    candidate: dict[str, Any],
    *,
    parent_document: StrategyDocument,
) -> tuple[dict[str, Any], list[str]]:
    """Ignore redundant no-op paragraphs while preserving real candidate edits."""
    current_by_id = {item.id: item.value for item in parent_document.details}
    kept: list[dict[str, str]] = []
    ignored: list[str] = []
    for patch in candidate.get("patches") or []:
        target = str(patch.get("target") or "")
        replacement = str(patch.get("replacement") or "").strip()
        if target in current_by_id and replacement == current_by_id[target]:
            ignored.append(target)
            continue
        kept.append(patch)
    return {**candidate, "patches": kept}, ignored


def _fallback_is_safe(*, failure_stage: str, errors: list[str]) -> bool:
    """Allow only a weak-but-executable semantic candidate to reach matches."""
    if failure_stage != "semantic" or not errors:
        return False
    blocking = (
        "missing_dependency",
        "internal_inconsistency",
        "runtime_boundary",
        "unsupported_capability",
        "unsupported capability",
        "decision_grounding",
        "preserved_strengths",
        "strategy_identity",
        "semantic validator returned no json object",
        "semantic validator rejected the patch",
    )
    normalized = [str(error).strip().casefold() for error in errors if str(error).strip()]
    if not normalized or any(
        marker in error for error in normalized for marker in blocking
    ):
        return False
    return all("underpowered_implementation" in error for error in normalized)


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
    mechanism_prediction = (
        dict(decision.get("mechanism_prediction"))
        if isinstance(decision.get("mechanism_prediction"), dict)
        else {}
    )
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
        "mechanism_family": str(decision.get("mechanism_family") or "").strip(),
        "failure_mode_analysis": (
            dict(decision.get("failure_mode_analysis"))
            if isinstance(decision.get("failure_mode_analysis"), dict)
            else {}
        ),
        "priority_alignment": (
            dict(decision.get("priority_alignment"))
            if isinstance(decision.get("priority_alignment"), dict)
            else {}
        ),
        "mechanism_prediction": mechanism_prediction,
        "retrieval_assessment": (
            dict(decision.get("retrieval_assessment"))
            if isinstance(decision.get("retrieval_assessment"), dict)
            else {}
        ),
        "priority_problem": problem,
        "plan_direction": direction,
        "intervention_package": dict(plan),
        "primary_lever": "other",
        "predictions": [
            str(mechanism_prediction.get("outcome_prediction") or expected_effect).strip()
            or "candidate matches test the supplied hypothesis"
        ],
        "disproof_conditions": [
            str(mechanism_prediction.get("disproof_condition") or "").strip()
            or "the intended mechanism materially changes but the predicted outcome does not"
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
        "inheritance": dict(candidate.get("inheritance") or {}),
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
    retry_feedback: list[str] | None = None,
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
    validation_errors = [str(item).strip() for item in (retry_feedback or []) if str(item).strip()]
    prompt_errors = list(validation_errors)
    events: list[dict[str, Any]] = []
    candidate: dict[str, Any] | None = None
    last_improvement: EvolImprovement | None = None
    latest_applied_improvement: EvolImprovement | None = None
    latest_applied_failure_stage = ""
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

    base_knowledge_runs = _knowledge_runs_for_optimizer(decision, observations)
    knowledge_runs = list(base_knowledge_runs)
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
                validation_errors=prompt_errors,
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
            prompt_errors = [error]
            events.append({"attempt": attempt, "action": "invalid", "error": error, "llm_calls": llm_calls})
            continue

        normalized, error = _normalize_optimizer_candidate(
            raw, parent_document=parent_document
        )
        if normalized is None:
            candidate = raw
            validation_errors.append(error)
            prompt_errors = [error]
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

        normalized, ignored_unchanged = _drop_unchanged_patches(
            normalized,
            parent_document=parent_document,
        )
        if ignored_unchanged:
            events.append(
                {
                    "attempt": attempt,
                    "action": "ignore_unchanged_patches",
                    "valid": True,
                    "ignored_targets": ignored_unchanged,
                    "llm_calls": llm_calls,
                }
            )
            print(
                f"{prefix}OptimizationAgent: ignoring unchanged paragraph "
                f"patches: {', '.join(ignored_unchanged)}",
                flush=True,
            )
        if not normalized["patches"]:
            error = "optimizer candidate contains only unchanged paragraph replacements"
            candidate = normalized
            validation_errors.append(error)
            prompt_errors = [error]
            events.append(
                {
                    "attempt": attempt,
                    "action": "strategy_patch_structure",
                    "valid": False,
                    "error": error,
                    "llm_calls": llm_calls,
                }
            )
            continue

        operations = _patches_to_operations(normalized["patches"])
        rationale = _candidate_rationale(decision=decision, candidate=normalized)
        try:
            patched_text, paragraph_changes = parent_document.apply_patch(operations)
        except ValueError as exc:
            error = str(exc)
            candidate = normalized
            validation_errors.append(error)
            prompt_errors = [error]
            events.append(
                {
                    "attempt": attempt,
                    "action": "apply_strategy_patch",
                    "valid": False,
                    "error": error,
                    "llm_calls": llm_calls,
                }
            )
            if attempt <= MAX_VALIDATION_RETRIES:
                print(
                    f"{prefix}OptimizationAgent: apply patch failed; "
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
        draft_improvement = EvolImprovement(
            analysis=rationale,
            files=payload["files"],
            raw=payload,
        )
        # Keep the newest mechanically applicable generation so a final
        # weak-but-executable semantic result can still be evaluated.
        latest_applied_improvement = draft_improvement

        structure_errors = validate_strategy_patch_structure(
            decision=decision,
            patches=normalized["patches"],
            parent_document=parent_document,
        )
        if structure_errors:
            error = "; ".join(structure_errors)
            latest_applied_failure_stage = "structure"
            validation_errors.append(error)
            prompt_errors = [error]
            events.append(
                {
                    "attempt": attempt,
                    "action": "strategy_patch_structure",
                    "valid": False,
                    "error": error,
                    "llm_calls": llm_calls,
                    "paragraph_changes": paragraph_changes,
                }
            )
            if attempt <= MAX_VALIDATION_RETRIES:
                print(
                    f"{prefix}OptimizationAgent: patch structure failed; "
                    f"retrying ({attempt}/{MAX_VALIDATION_RETRIES}): {error}",
                    flush=True,
                )
            continue

        result = validate_improvement(
            files=draft_improvement.files,
            race=race,
        )
        if not result.ok:
            latest_applied_failure_stage = "basic"
            validation_errors.append(result.error)
            prompt_errors = [result.error]
            events.append(
                {
                    "attempt": attempt,
                    "action": normalized["action"],
                    "valid": False,
                    "error": result.error,
                    "llm_calls": llm_calls,
                    "paragraph_changes": paragraph_changes,
                }
            )
            if attempt <= MAX_VALIDATION_RETRIES:
                print(
                    f"{prefix}OptimizationAgent: basic validation failed; "
                    f"retrying ({attempt}/{MAX_VALIDATION_RETRIES}): {result.error}",
                    flush=True,
                )
            continue

        draft_improvement.files = result.files or draft_improvement.files
        draft_improvement.raw["files"] = dict(draft_improvement.files)
        latest_applied_improvement = draft_improvement
        last_improvement = draft_improvement
        candidate_knowledge = _candidate_knowledge_run(
            candidate_text=patched_text,
            race=race,
            capability_manifest=capability_manifest,
        )
        knowledge_runs = [
            run
            for run in base_knowledge_runs
            if str(run.get("question_id") or "") != "QCANDIDATE"
        ]
        if candidate_knowledge is not None:
            knowledge_runs.append(candidate_knowledge)
            draft_improvement.raw["candidate_knowledge"] = candidate_knowledge
            draft_improvement.analysis["candidate_knowledge"] = candidate_knowledge
            candidate_knowledge_error = find_knowledge_run_error(candidate_knowledge)
            if candidate_knowledge_error:
                error = (
                    "decision_grounding — candidate knowledge — "
                    + candidate_knowledge_error
                )
                latest_applied_failure_stage = "semantic"
                validation_errors.append(error)
                prompt_errors = [error]
                events.append(
                    {
                        "attempt": attempt,
                        "action": "candidate_knowledge",
                        "valid": False,
                        "error": error,
                        "llm_calls": llm_calls,
                        "paragraph_changes": paragraph_changes,
                    }
                )
                if attempt <= MAX_VALIDATION_RETRIES:
                    print(
                        f"{prefix}OptimizationAgent: candidate knowledge failed; "
                        f"retrying ({attempt}/{MAX_VALIDATION_RETRIES}): {error}",
                        flush=True,
                    )
                continue
        semantic_errors = validate_strategy_patch_semantics(
            decision=decision,
            parent_text=parent_text,
            candidate_text=patched_text,
            patches=normalized["patches"],
            inheritance=normalized.get("inheritance"),
            capability_manifest=capability_manifest,
            knowledge_runs=knowledge_runs,
            model=model,
        )
        if semantic_errors:
            error = "; ".join(semantic_errors)
            latest_applied_failure_stage = "semantic"
            validation_errors.append(error)
            prompt_errors = [error]
            events.append(
                {
                    "attempt": attempt,
                    "action": "strategy_patch_semantics",
                    "valid": False,
                    "error": error,
                    "llm_calls": llm_calls,
                    "paragraph_changes": paragraph_changes,
                }
            )
            if attempt <= MAX_VALIDATION_RETRIES:
                print(
                    f"{prefix}OptimizationAgent: patch semantics failed; "
                    f"retrying ({attempt}/{MAX_VALIDATION_RETRIES}): {error}",
                    flush=True,
                )
            continue

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
        print(
            f"{prefix}OptimizationAgent: candidate passed validation",
            flush=True,
        )
        return result, last_improvement, observations, validation_errors, events

    error = validation_errors[-1] if validation_errors else "OptimizationAgent exhausted"
    if latest_applied_improvement is not None and _fallback_is_safe(
        failure_stage=latest_applied_failure_stage,
        errors=prompt_errors,
    ):
        fallback_status = (
            "accepted_after_semantic_retry_exhausted"
            if latest_applied_failure_stage == "semantic"
            else "accepted_after_validation_retry_exhausted"
        )
        warning = {
            "status": fallback_status,
            "failure_stage": latest_applied_failure_stage,
            "errors": list(validation_errors),
        }
        latest_applied_improvement.raw["validation_fallback"] = warning
        latest_applied_improvement.analysis["validation_fallback"] = warning
        if latest_applied_failure_stage == "semantic":
            # Retain the previous field for checkpoint compatibility.
            latest_applied_improvement.raw["semantic_validation"] = warning
            latest_applied_improvement.analysis["semantic_validation"] = warning
        events.append(
            {
                "action": "accept_latest_candidate_after_validation_retry_exhausted",
                "valid": True,
                "warning": error,
                "failure_stage": latest_applied_failure_stage,
                "llm_calls": llm_calls,
            }
        )
        print(
            f"{prefix}OptimizationAgent: validation retries exhausted; "
            "using the latest applicable generated candidate with warnings",
            flush=True,
        )
        return (
            ValidationResult(
                ok=True,
                error=error,
                files=dict(latest_applied_improvement.files),
            ),
            latest_applied_improvement,
            observations,
            validation_errors,
            events,
        )
    return (
        ValidationResult(ok=False, error=error),
        None,
        observations,
        validation_errors,
        events,
    )
