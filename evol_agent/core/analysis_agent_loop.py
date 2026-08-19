from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from functools import partial
from pathlib import Path
import re
from typing import Any

from .checkpoint import EvolCheckpoint, stage_reached
from .config import (
    ANALYSIS_ENABLE_REASONING,
    DEFAULT_ANALYSIS_MODEL,
    MAX_CONCURRENT_MATCH_SUBAGENTS,
    MAX_KNOWLEDGE_QUERIES,
)
from .llm import call_json_llm
from .loop_helpers import (
    abandon_executor,
    analysis_from_json,
    evidence_digest,
    exit_on_keyboard_interrupt,
    fallback_analysis,
    normalize_strategy_contract,
)
from .match_summary import run_fixed_match_summary
from .evidence_retrieval import build_retrieval_evidence_packet
from .prompts import (
    build_cross_match_decision_prompt,
    build_cross_match_discovery_prompt,
)
from .types import AnalysisPipelineResult, BattleAnalysis, GameDigest, ToolObservation
from ..sc2_data_agent import (
    build_knowledge_query,
    find_knowledge_run_error,
    is_knowledge_run_verified,
    run_knowledge_query,
)


_ANALYSIS_ATTEMPTS = 2
_KNOWLEDGE_NEEDS = {"effects", "synergy", "counters", "requirements"}
_FINAL_NEXT_ACTIONS = frozenset(
    {
        "propose_strategy_patch",
        "request_more_matches",
        "inspect_runtime",
        "stop",
    }
)
_CONTROL_CLASSES = frozenset(
    {
        "strategy_fixable",
        "commander_execution",
        "runtime_execution",
        "observation_limited",
    }
)
_ASSESSMENTS = frozenset(
    {
        "plausible_primary",
        "plausible",
        "contributor_not_sufficient",
        "weakly_supported",
        "contradicted",
        "runtime_likely",
    }
)
_CONFIDENCE = frozenset({"low", "medium", "high"})


def _clean_strings(value: Any, *, limit: int | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    result = [text for item in value if (text := str(item).strip())]
    return result[:limit] if limit is not None else result


def _unwrap_analysis(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    raw = result.get("analysis") if isinstance(result.get("analysis"), dict) else result
    return raw if isinstance(raw, dict) else None


def _normalize_pattern_items(raw: Any, *, limit: int = 3) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, str):
            pattern = item.strip()
            evidence: list[str] = []
            extra: dict[str, Any] = {}
        elif isinstance(item, dict):
            pattern = str(item.get("pattern") or item.get("problem") or "").strip()
            evidence = _clean_strings(item.get("evidence"), limit=4)
            extra = dict(item)
        else:
            continue
        if not pattern:
            continue
        row = {"pattern": pattern, "evidence": evidence}
        confidence = str(extra.get("confidence") or "").strip().lower()
        if confidence in _CONFIDENCE:
            row["confidence"] = confidence
        items.append(row)
        if len(items) >= limit:
            break
    return items


def _normalize_unknowns(raw: Any, *, limit: int = 3) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, str):
            unknown = item.strip()
            why = ""
            evidence: list[str] = []
        elif isinstance(item, dict):
            unknown = str(item.get("unknown") or item.get("pattern") or "").strip()
            why = str(item.get("why_it_matters") or "").strip()
            evidence = _clean_strings(item.get("evidence"), limit=4)
        else:
            continue
        if not unknown:
            continue
        items.append({"unknown": unknown, "why_it_matters": why, "evidence": evidence})
        if len(items) >= limit:
            break
    return items


def _normalize_knowledge_questions(raw: Any) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        entities = _clean_strings(item.get("entities"), limit=6)
        needs = [
            need.lower()
            for need in _clean_strings(item.get("needs"), limit=4)
            if need.lower() in _KNOWLEDGE_NEEDS
        ]
        if not question or not entities or not needs:
            continue
        calculations: list[dict[str, Any]] = []
        for calculation in item.get("calculations") or []:
            if not isinstance(calculation, dict):
                continue
            calculation_type = str(calculation.get("type") or "").strip().lower()
            action = str(
                calculation.get("action")
                or calculation.get("unit")
                or calculation.get("entity")
                or ""
            ).strip()
            if calculation_type not in {
                "parallel_production",
                "resource_demand_per_minute",
            } or not action:
                continue
            normalized_calculation: dict[str, Any] = {
                "type": calculation_type,
                "action": action,
                "production_slots": calculation.get("production_slots")
                or calculation.get("producers"),
            }
            if calculation_type == "parallel_production":
                normalized_calculation["quantity"] = calculation.get("quantity")
            calculations.append(normalized_calculation)
            if len(calculations) >= 6:
                break
        questions.append(
            {
                "id": f"Q{len(questions) + 1}",
                "question": question,
                "entities": entities,
                "needs": list(dict.fromkeys(needs)),
                "query_reason": str(
                    item.get("query_reason") or item.get("reason") or question
                ).strip(),
                "evidence_refs": _clean_strings(
                    item.get("evidence_refs") or item.get("evidence"), limit=6
                ),
                "hypothesis_scope": str(
                    item.get("hypothesis_scope") or "static_sc2_fact"
                ).strip(),
                "calculations": calculations,
            }
        )
        if len(questions) >= MAX_KNOWLEDGE_QUERIES:
            break
    return questions


def _normalize_match_evidence_queries(
    raw: Any,
    *,
    fallback_patterns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("query_reason") or item.get("reason") or "").strip()
        refs = _clean_strings(
            item.get("evidence_refs") or item.get("evidence"), limit=8
        )
        if not reason or not refs:
            continue
        queries.append(
            {
                "id": f"M{len(queries) + 1}",
                "query_reason": reason,
                "evidence_refs": refs,
            }
        )
        if len(queries) >= 4:
            break
    if queries:
        return queries
    for pattern in fallback_patterns:
        refs = _clean_strings(pattern.get("evidence"), limit=6)
        if not refs:
            continue
        queries.append(
            {
                "id": f"M{len(queries) + 1}",
                "query_reason": (
                    "Verify the recorded interaction and attribution for: "
                    + str(pattern.get("pattern") or "")
                ),
                "evidence_refs": refs,
            }
        )
        if len(queries) >= 4:
            break
    return queries


def _normalize_experience_query(
    raw: Any,
    *,
    fallback_patterns: list[dict[str, Any]],
) -> dict[str, Any]:
    item = raw if isinstance(raw, dict) else {}
    signature = _clean_strings(item.get("failure_signature"), limit=8)
    if not signature:
        signature = [
            str(pattern.get("pattern") or "").strip()
            for pattern in fallback_patterns
            if str(pattern.get("pattern") or "").strip()
        ][:6]
    reason = str(item.get("query_reason") or item.get("reason") or "").strip()
    if not reason:
        reason = (
            "Retrieve prior successful, rejected, and inconclusive interventions "
            "that address the same observed failure pattern."
        )
    return {"query_reason": reason, "failure_signature": signature}


def _normalize_cross_match_discovery(
    raw: dict[str, Any],
    *,
    knowledge_mode: str,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "discovery returned no JSON object"
    strengths = _normalize_pattern_items(raw.get("strengths"))
    weaknesses = _normalize_pattern_items(raw.get("weaknesses"))
    for item in weaknesses:
        if "confidence" not in item:
            item["confidence"] = "medium"
    unknowns = _normalize_unknowns(raw.get("unknowns"))
    pressure_patterns = _normalize_pattern_items(
        raw.get("opponent_pressure_patterns"), limit=4
    )
    matchup_patterns = _normalize_pattern_items(raw.get("matchup_patterns"), limit=4)
    raw_query_plan = raw.get("query_plan") if isinstance(raw.get("query_plan"), dict) else {}
    questions = (
        _normalize_knowledge_questions(
            raw_query_plan.get("game_knowledge_queries")
            or raw.get("knowledge_questions")
        )
        if knowledge_mode == "enabled"
        else []
    )
    fallback_patterns = [*weaknesses, *pressure_patterns, *matchup_patterns]
    query_plan = {
        "match_evidence_queries": _normalize_match_evidence_queries(
            raw_query_plan.get("match_evidence_queries"),
            fallback_patterns=fallback_patterns,
        ),
        "experience_query": _normalize_experience_query(
            raw_query_plan.get("experience_query"),
            fallback_patterns=fallback_patterns,
        ),
        "game_knowledge_queries": questions,
    }
    return (
        {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "unknowns": unknowns,
            "opponent_pressure_patterns": pressure_patterns,
            "matchup_patterns": matchup_patterns,
            "knowledge_questions": questions,
            "query_plan": query_plan,
        },
        "",
    )


def _normalize_priority_problem(raw: Any) -> tuple[dict[str, Any] | None, str]:
    if isinstance(raw, list):
        return None, "priority_problem must be one object, not a list"
    if isinstance(raw, str):
        problem = raw.strip()
        raw = {"problem": problem} if problem else {}
    if not raw:
        return {}, ""
    if not isinstance(raw, dict):
        return None, "priority_problem must be an object"
    problem = str(raw.get("problem") or raw.get("pattern") or "").strip()
    if not problem:
        return {}, ""
    control_class = str(raw.get("control_class") or "strategy_fixable").strip().lower()
    if control_class not in _CONTROL_CLASSES:
        control_class = "strategy_fixable"
    confidence = str(raw.get("confidence") or "medium").strip().lower()
    if confidence not in _CONFIDENCE:
        confidence = "medium"
    return (
        {
            "problem_id": "P1",
            "problem": problem,
            "evidence": _clean_strings(raw.get("evidence"), limit=4),
            "control_class": control_class,
            "strategy_fixable": control_class == "strategy_fixable",
            "confidence": confidence,
            "consequence": str(raw.get("consequence") or "").strip(),
        },
        "",
    )


def _normalize_mechanism_family(raw: Any, plan: dict[str, Any] | None) -> str:
    value = str(raw or "").strip().lower()
    if not value and plan:
        value = str(plan.get("direction") or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value[:80] or "unclassified_strategy_mechanism"


def _normalize_plan(raw: Any) -> tuple[dict[str, Any] | None, str]:
    if raw in (None, "", []):
        return None, ""
    if isinstance(raw, str):
        direction = raw.strip()
        material_behavior_change = ""
        coordinated_changes: list[dict[str, str]] = []
        preserve: list[str] = []
    elif isinstance(raw, dict):
        direction = str(
            raw.get("direction") or raw.get("name") or raw.get("plan") or ""
        ).strip()
        material_behavior_change = str(
            raw.get("material_behavior_change") or ""
        ).strip()
        coordinated_changes = []
        for item in raw.get("coordinated_changes") or []:
            if not isinstance(item, dict):
                continue
            change = str(item.get("change") or "").strip()
            why_required = str(item.get("why_required") or "").strip()
            if change and why_required:
                coordinated_changes.append(
                    {"change": change, "why_required": why_required}
                )
            if len(coordinated_changes) >= 10:
                break
        preserve = _clean_strings(raw.get("preserve"), limit=5)
    else:
        return None, "plan must be an object with direction"
    if not direction:
        return None, "plan.direction is required"
    return {
        "direction": direction,
        "material_behavior_change": material_behavior_change,
        "coordinated_changes": coordinated_changes,
        "preserve": preserve,
    }, ""


def _normalize_failure_mode_analysis(
    raw: Any,
) -> tuple[dict[str, str] | None, str]:
    if not isinstance(raw, dict):
        return None, "failure_mode_analysis must be an object"
    fields = (
        "failure_mode",
        "survival_prerequisite",
        "opponent_pressure_pattern",
        "matchup_assessment",
        "counterexample_check",
    )
    normalized = {field: str(raw.get(field) or "").strip() for field in fields}
    missing = [field for field, value in normalized.items() if not value]
    if missing:
        return None, "failure_mode_analysis requires " + ", ".join(missing)
    return normalized, ""


def _normalize_priority_alignment(
    raw: Any,
) -> tuple[dict[str, str] | None, str]:
    if not isinstance(raw, dict):
        return None, "priority_alignment must be an object"
    fields = (
        "selected_priority",
        "higher_priority_assessment",
        "downstream_combat_effect",
    )
    normalized = {field: str(raw.get(field) or "").strip() for field in fields}
    missing = [field for field, value in normalized.items() if not value]
    if missing:
        return None, "priority_alignment requires " + ", ".join(missing)
    return normalized, ""


def _normalize_retrieval_assessment(
    raw: Any,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "retrieval_assessment must be an object"
    query_summary = str(raw.get("query_summary") or "").strip()
    confidence = str(raw.get("confidence") or "").strip().lower()
    if confidence not in _CONFIDENCE:
        return None, "retrieval_assessment.confidence must be low, medium, or high"
    if not query_summary:
        return None, "retrieval_assessment.query_summary is required"
    return (
        {
            "query_summary": query_summary,
            "match_evidence_used": _clean_strings(
                raw.get("match_evidence_used"), limit=8
            ),
            "historical_experience_used": _clean_strings(
                raw.get("historical_experience_used"), limit=6
            ),
            "knowledge_used": _clean_strings(raw.get("knowledge_used"), limit=6),
            "conflicting_evidence": _clean_strings(
                raw.get("conflicting_evidence"), limit=6
            ),
            "confidence": confidence,
        },
        "",
    )


def _validate_retrieval_assessment_links(
    assessment: dict[str, Any],
    *,
    retrieval_evidence: dict[str, Any] | None,
    knowledge_runs: list[dict[str, Any]] | None,
) -> str:
    packet = retrieval_evidence or {}
    match_packet = packet.get("match_record_evidence") or {}
    match_queries = [
        item for item in match_packet.get("queries") or [] if isinstance(item, dict)
    ]
    available_match_ids = {
        str(item.get("query_id") or "").strip()
        for item in match_queries
        if item.get("results")
    }
    used_match = list(assessment.get("match_evidence_used") or [])
    if available_match_ids and not used_match:
        return "retrieval_assessment must use or explicitly conflict with queried match evidence"
    cited_match_ids = {
        match.group(0).upper()
        for text in used_match
        for match in re.finditer(r"\bM\d+\b", str(text), re.IGNORECASE)
    }
    unknown_match_ids = cited_match_ids - available_match_ids
    if unknown_match_ids:
        return "retrieval_assessment cites unknown match query ids: " + ", ".join(
            sorted(unknown_match_ids)
        )

    verified_knowledge_ids = {
        str(run.get("question_id") or "").strip()
        for run in knowledge_runs or []
        if is_knowledge_run_verified(run)
    }
    used_knowledge = list(assessment.get("knowledge_used") or [])
    if verified_knowledge_ids and not used_knowledge:
        return "retrieval_assessment must state which verified knowledge facts were used"
    cited_knowledge_ids = {
        match.group(0).upper()
        for text in used_knowledge
        for match in re.finditer(r"\bQ\d+\b", str(text), re.IGNORECASE)
    }
    unknown_knowledge_ids = cited_knowledge_ids - verified_knowledge_ids
    if unknown_knowledge_ids:
        return "retrieval_assessment cites unknown knowledge query ids: " + ", ".join(
            sorted(unknown_knowledge_ids)
        )
    return ""


def _normalize_mechanism_prediction(
    raw: Any,
) -> tuple[dict[str, str] | None, str]:
    if not isinstance(raw, dict):
        return None, "mechanism_prediction must be an object"
    fields = (
        "expected_change",
        "minimum_material_change",
        "outcome_prediction",
        "combat_success_measure",
        "disproof_condition",
    )
    normalized = {field: str(raw.get(field) or "").strip() for field in fields}
    missing = [field for field, value in normalized.items() if not value]
    if missing:
        return None, (
            "mechanism_prediction requires " + ", ".join(missing)
        )
    return normalized, ""


def _normalize_considered_explanations(raw: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return items
    for item in raw:
        if not isinstance(item, dict):
            continue
        explanation = str(item.get("explanation") or item.get("hypothesis") or "").strip()
        if not explanation:
            continue
        control_class = str(item.get("control_class") or "strategy_fixable").strip().lower()
        if control_class not in _CONTROL_CLASSES:
            control_class = "strategy_fixable"
        assessment = str(item.get("assessment") or "plausible").strip().lower()
        if assessment not in _ASSESSMENTS:
            assessment = "plausible"
        items.append(
            {
                "explanation": explanation,
                "supporting_evidence": _clean_strings(
                    item.get("supporting_evidence") or item.get("evidence"),
                    limit=4,
                ),
                "counterevidence": _clean_strings(item.get("counterevidence"), limit=4),
                "control_class": control_class,
                "assessment": assessment,
            }
        )
        if len(items) >= 4:
            break
    return items


def _adapt_decision_for_optimizer(
    decision: dict[str, Any],
    *,
    strategy_name: str,
) -> dict[str, Any]:
    """Map the simplified Decision schema onto the current Optimizer payload."""
    strengths = decision.get("strengths_to_preserve") or []
    priority = decision.get("priority_problem") or {}
    hypothesis = str(decision.get("hypothesis") or "").strip()
    plan = decision.get("plan") if isinstance(decision.get("plan"), dict) else None
    next_action = str(decision.get("next_action") or "")
    payload = dict(decision)
    payload["wins_to_preserve"] = [
        {"pattern": item["pattern"], "evidence": item["evidence"], "why": ""}
        for item in strengths
        if isinstance(item, dict)
    ]
    payload["problems"] = [priority] if priority.get("problem") else []
    payload["primary_problem"] = priority
    payload["winning_mechanism"] = (
        strengths[0]["pattern"] if strengths and isinstance(strengths[0], dict) else ""
    )
    payload["knowledge_questions"] = []
    plans: list[dict[str, Any]] = []
    if next_action == "propose_strategy_patch" and plan:
        plans.append(
            {
                "id": "D1",
                "name": str(plan.get("direction") or ""),
                "hypothesis": hypothesis,
                "primary_lever": "other",
                "addresses_problem_ids": ["P1"],
                "changes": list(plan.get("coordinated_changes") or []),
                "predictions": [
                    str(
                        (decision.get("mechanism_prediction") or {}).get(
                            "outcome_prediction"
                        )
                        or ""
                    )
                ],
                "disproof_conditions": [
                    str(
                        (decision.get("mechanism_prediction") or {}).get(
                            "disproof_condition"
                        )
                        or ""
                    )
                ],
                "capability_mapping": {},
                "expected_benefit": "",
                "risk_to_winning_mechanism": "",
                "preserve": list(plan.get("preserve") or []),
            }
        )
    payload["candidate_plans"] = plans
    payload["optimization_targets"] = [
        {
            "plan_id": item["id"],
            "addresses_problem_ids": item["addresses_problem_ids"],
            "strategy_change": item["name"],
            "changes": item["changes"],
        }
        for item in plans
    ]
    payload["repeated_failures"] = [
        {
            "problem_id": priority.get("problem_id") or "P1",
            "cause": priority.get("problem") or "",
            "consequence": priority.get("consequence") or "",
            "seen_in": priority.get("evidence") or [],
            "strategy_fixable": bool(priority.get("strategy_fixable")),
            "control_class": priority.get("control_class") or "",
            "confidence": priority.get("confidence") or "medium",
        }
    ] if priority.get("problem") else []
    payload["strategy_name"] = strategy_name
    return payload


def _normalize_cross_match_decision(
    raw: dict[str, Any],
    *,
    strategy_name: str,
    require_retrieval_assessment: bool = False,
    retrieval_evidence: dict[str, Any] | None = None,
    knowledge_runs: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "decision returned no JSON object"
    next_action = str(raw.get("next_action") or "").strip()
    if next_action not in _FINAL_NEXT_ACTIONS:
        return None, (
            "next_action must be propose_strategy_patch, request_more_matches, "
            "inspect_runtime, or stop"
        )
    action_reason = str(raw.get("action_reason") or "").strip()
    strengths = _normalize_pattern_items(
        raw.get("strengths_to_preserve") or raw.get("strengths")
    )
    priority, priority_error = _normalize_priority_problem(raw.get("priority_problem"))
    if priority_error:
        return None, priority_error
    hypothesis = str(raw.get("hypothesis") or "").strip()
    plan, plan_error = _normalize_plan(raw.get("plan"))
    mechanism_family = _normalize_mechanism_family(
        raw.get("mechanism_family"), plan
    )
    mechanism_prediction, mechanism_error = _normalize_mechanism_prediction(
        raw.get("mechanism_prediction")
    )
    failure_mode_analysis, failure_mode_error = _normalize_failure_mode_analysis(
        raw.get("failure_mode_analysis")
    )
    priority_alignment, priority_alignment_error = _normalize_priority_alignment(
        raw.get("priority_alignment")
    )
    retrieval_assessment, retrieval_error = _normalize_retrieval_assessment(
        raw.get("retrieval_assessment")
    )
    if plan_error and next_action == "propose_strategy_patch":
        return None, plan_error

    if next_action == "propose_strategy_patch":
        control = str((priority or {}).get("control_class") or "")
        if control in {"runtime_execution", "commander_execution"}:
            next_action = "inspect_runtime"
            action_reason = action_reason or (
                "Priority problem is an execution defect, not a strategy.md change."
            )
            plan = None
            hypothesis = ""
        elif not priority or not priority.get("problem") or not priority.get("evidence"):
            return None, "propose_strategy_patch requires priority_problem with evidence"
        elif not priority.get("strategy_fixable"):
            return None, "propose_strategy_patch requires control_class=strategy_fixable"
        elif not hypothesis:
            return None, "propose_strategy_patch requires hypothesis"
        elif not plan:
            return None, "propose_strategy_patch requires plan.direction"
        elif mechanism_error or not mechanism_prediction:
            return None, mechanism_error or (
                "propose_strategy_patch requires mechanism_prediction"
            )
        elif not plan.get("material_behavior_change"):
            return None, "propose_strategy_patch requires plan.material_behavior_change"
        elif not plan.get("coordinated_changes"):
            return None, "propose_strategy_patch requires plan.coordinated_changes"
        elif failure_mode_error or not failure_mode_analysis:
            return None, failure_mode_error or (
                "propose_strategy_patch requires failure_mode_analysis"
            )
        elif priority_alignment_error or not priority_alignment:
            return None, priority_alignment_error or (
                "propose_strategy_patch requires priority_alignment"
            )
        elif require_retrieval_assessment and (
            retrieval_error or not retrieval_assessment
        ):
            return None, retrieval_error or (
                "propose_strategy_patch requires retrieval_assessment"
            )
        elif require_retrieval_assessment and retrieval_assessment:
            retrieval_link_error = _validate_retrieval_assessment_links(
                retrieval_assessment,
                retrieval_evidence=retrieval_evidence,
                knowledge_runs=knowledge_runs,
            )
            if retrieval_link_error:
                return None, retrieval_link_error
    elif next_action in {"request_more_matches", "stop"}:
        if not action_reason:
            return None, f"{next_action} requires action_reason"
        plan = None
    else:
        plan = None

    decision = {
        "strengths_to_preserve": strengths,
        "priority_problem": priority or {},
        "hypothesis": hypothesis,
        "mechanism_family": mechanism_family,
        "failure_mode_analysis": failure_mode_analysis or {},
        "priority_alignment": priority_alignment or {},
        "mechanism_prediction": mechanism_prediction or {},
        "retrieval_assessment": retrieval_assessment or {},
        "next_action": next_action,
        "action_reason": action_reason,
        "considered_explanations": _normalize_considered_explanations(
            raw.get("considered_explanations")
        ),
        "plan": plan,
        "evidence_limits": _clean_strings(raw.get("evidence_limits")),
        "strategy_contract": normalize_strategy_contract(
            raw.get("strategy_contract"), strategy_name=strategy_name
        ),
    }
    return _adapt_decision_for_optimizer(decision, strategy_name=strategy_name), ""


def _normalize_batch_analysis(
    raw: dict[str, Any],
    *,
    strategy_name: str,
    knowledge_mode: str = "enabled",
    allow_query_knowledge: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    """Compatibility wrapper: final schema is the Decision payload."""
    del knowledge_mode, allow_query_knowledge
    return _normalize_cross_match_decision(raw, strategy_name=strategy_name)


def _run_analysis_json(
    *,
    build_prompt,
    model: str,
    normalizer,
    events: list[dict[str, Any]],
    label: str,
) -> tuple[dict[str, Any] | None, list[str], int]:
    schema_errors: list[str] = []
    payload: dict[str, Any] | None = None
    calls = 0
    for attempt in range(1, _ANALYSIS_ATTEMPTS + 1):
        calls += 1
        result = call_json_llm(
            build_prompt(schema_errors),
            model=model,
            is_reasoning=ANALYSIS_ENABLE_REASONING,
        )
        raw = _unwrap_analysis(result)
        if raw is None:
            error = f"{label} returned no analysis object"
            normalized = None
        else:
            normalized, error = normalizer(raw)
        events.append({"attempt": attempt, "action": label, "error": error})
        if normalized is not None:
            payload = normalized
            break
        schema_errors.append(error)
    return payload, schema_errors, calls


def _run_cross_match_discovery(
    *,
    strategy_name: str,
    race: str,
    summaries: list[BattleAnalysis],
    skill_texts: dict[str, str],
    knowledge_mode: str,
    model: str,
    prior_experiences: list[Any] | None,
    capability_manifest: dict[str, Any] | None,
    events: list[dict[str, Any]],
    prefix: str,
) -> tuple[dict[str, Any] | None, list[str], int]:
    print(
        f"{prefix}AnalysisAgent: cross-match discovery over {len(summaries)} match summaries",
        flush=True,
    )
    payload, errors, calls = _run_analysis_json(
        build_prompt=lambda schema_errors: build_cross_match_discovery_prompt(
            strategy_name=strategy_name,
            race=race,
            single_game_analyses=summaries,
            skill_texts=skill_texts,
            validation_errors=schema_errors,
            knowledge_mode=knowledge_mode,
            prior_experiences=prior_experiences or [],
            capability_manifest=capability_manifest or {},
        ),
        model=model,
        normalizer=lambda raw: _normalize_cross_match_discovery(
            raw, knowledge_mode=knowledge_mode
        ),
        events=events,
        label="cross_match_discovery",
    )
    if payload is not None:
        print(
            f"{prefix}AnalysisAgent: discovery strengths={len(payload['strengths'])} "
            f"weaknesses={len(payload['weaknesses'])} unknowns={len(payload['unknowns'])} "
            f"knowledge_questions={len(payload['knowledge_questions'])}",
            flush=True,
        )
        events.append(
            {
                "action": "cross_match_discovery",
                "strength_count": len(payload["strengths"]),
                "weakness_count": len(payload["weaknesses"]),
                "unknown_count": len(payload["unknowns"]),
                "knowledge_question_count": len(payload["knowledge_questions"]),
            }
        )
    return payload, errors, calls


def _run_cross_match_decision(
    *,
    strategy_name: str,
    race: str,
    summaries: list[BattleAnalysis],
    skill_texts: dict[str, str],
    knowledge_mode: str,
    model: str,
    prior_experiences: list[Any] | None,
    capability_manifest: dict[str, Any] | None,
    discovery: dict[str, Any],
    knowledge_runs: list[dict[str, Any]],
    retrieval_evidence: dict[str, Any],
    events: list[dict[str, Any]],
    prefix: str,
) -> tuple[dict[str, Any] | None, list[str], int]:
    print(f"{prefix}AnalysisAgent: cross-match decision", flush=True)
    payload, errors, calls = _run_analysis_json(
        build_prompt=lambda schema_errors: build_cross_match_decision_prompt(
            strategy_name=strategy_name,
            race=race,
            single_game_analyses=summaries,
            skill_texts=skill_texts,
            validation_errors=schema_errors,
            knowledge_mode=knowledge_mode,
            prior_experiences=prior_experiences or [],
            discovery=discovery,
            knowledge_runs=knowledge_runs,
            retrieval_evidence=retrieval_evidence,
            capability_manifest=capability_manifest or {},
        ),
        model=model,
        normalizer=lambda raw: _normalize_cross_match_decision(
            raw,
            strategy_name=strategy_name,
            require_retrieval_assessment=True,
            retrieval_evidence=retrieval_evidence,
            knowledge_runs=knowledge_runs,
        ),
        events=events,
        label="cross_match_decision",
    )
    if payload is not None:
        print(
            f"{prefix}AnalysisAgent: next_action={payload.get('next_action')}",
            flush=True,
        )
        events.append(
            {
                "action": "cross_match_decision",
                "next_action": payload.get("next_action"),
                "priority_problem": (payload.get("priority_problem") or {}).get("problem"),
            }
        )
    return payload, errors, calls


def _observations_from_runs(
    runs: list[dict[str, Any]], *, prefix: str
) -> list[ToolObservation]:
    observations: list[ToolObservation] = []
    for index, run in enumerate(runs, 1):
        verified = is_knowledge_run_verified(run)
        error = "" if verified else (
            find_knowledge_run_error(run)
            or str(run.get("error") or "knowledge query failed")
        )
        run["ok"] = verified
        run["error"] = error
        observations.append(
            ToolObservation(
                tool="sc2_knowledge",
                args={
                    "question_id": run.get("question_id"),
                    "query": run.get("query"),
                    "query_reason": run.get("query_reason"),
                    "evidence_refs": list(run.get("evidence_refs") or []),
                    "hypothesis_scope": run.get("hypothesis_scope"),
                },
                result={"answer": run.get("answer"), "error": error},
                ok=verified,
                summary=str(run.get("answer") if verified else error),
                status="complete" if verified else "failed",
            )
        )
        print(
            f"{prefix}AnalysisAgent: knowledge {index}/{len(runs)} "
            f"question={run.get('question_id')} status={'ok' if verified else 'failed'}",
            flush=True,
        )
    return observations


def _run_knowledge_queries(
    questions: list[dict[str, Any]],
    *,
    race: str,
    checkpoint: EvolCheckpoint | None,
    prefix: str,
) -> list[dict[str, Any]]:
    if not questions:
        return []
    cached: dict[str, dict[str, Any]] = {}
    if checkpoint is not None:
        for run in checkpoint.load_knowledge_results():
            question_id = str(run.get("question_id") or "").strip()
            if question_id:
                cached[question_id] = run

    runs: list[dict[str, Any]] = []
    retrieval_evidence: dict[str, Any] = {}
    print(
        f"{prefix}AnalysisAgent: resolving {len(questions)} deterministic knowledge question(s)",
        flush=True,
    )
    for question in questions:
        question_id = str(question.get("id") or "").strip()
        expected_query = build_knowledge_query(question, race=race)
        previous = cached.get(question_id)
        if (
            previous
            and is_knowledge_run_verified(previous)
            and str(previous.get("query") or "").strip() == expected_query.strip()
        ):
            run = previous
        else:
            run = run_knowledge_query(question, race=race)
            if checkpoint is not None:
                checkpoint.save_knowledge_result(run)
        run["question"] = str(question.get("question") or run.get("question") or "")
        run["ok"] = is_knowledge_run_verified(run)
        if run["ok"]:
            run["error"] = ""
        runs.append(run)
    return runs


def _summarize_matches(
    *,
    strategy_name: str,
    race: str,
    records: list[Any],
    skill_texts: dict[str, str],
    model: str,
    prefix: str,
    checkpoint: EvolCheckpoint | None,
    summary_seed_checkpoint: EvolCheckpoint | None = None,
) -> tuple[
    list[GameDigest],
    list[BattleAnalysis],
    int,
    list[dict[str, Any]],
    list[str],
]:
    if checkpoint is not None and stage_reached(checkpoint.stage, "match_summaries"):
        values = checkpoint.load_match_summaries()
        print(
            f"{prefix}AnalysisAgent: resume loaded {len(values[1])} match summaries",
            flush=True,
        )
        return values

    results: dict[
        int,
        tuple[GameDigest, BattleAnalysis, bool, list[str], list[dict[str, Any]]],
    ] = {}
    reused_paths: set[str] = set()
    if summary_seed_checkpoint is not None:
        try:
            seed_digests, seed_analyses, seed_completed, seed_events, _seed_errors = (
                summary_seed_checkpoint.load_match_summaries()
            )
            completed_paths = {
                str(Path(str(event.get("record_path") or "")).resolve())
                for event in seed_events
                if isinstance(event, dict) and bool(event.get("completed"))
            }
            assume_all_completed = not seed_events and seed_completed >= len(seed_digests)
            seeded = {}
            for digest, analysis in zip(seed_digests, seed_analyses):
                path = str(Path(digest.record_path).resolve())
                if assume_all_completed or path in completed_paths:
                    seeded[path] = (digest, analysis)
            for game_index, record in enumerate(records, 1):
                path = str(Path(record.file).resolve())
                cached = seeded.get(path)
                if cached is None:
                    continue
                digest, analysis = cached
                results[game_index] = (
                    digest,
                    analysis,
                    True,
                    [],
                    [
                        {
                            "action": "reuse_match_summary",
                            "source_checkpoint": str(summary_seed_checkpoint.run_dir),
                        }
                    ],
                )
                reused_paths.add(path)
        except (OSError, ValueError) as exc:
            print(
                f"{prefix}AnalysisAgent: summary seed ignored: {exc}",
                flush=True,
            )

    pending_records = [
        (game_index, record)
        for game_index, record in enumerate(records, 1)
        if game_index not in results
    ]
    worker_count = min(MAX_CONCURRENT_MATCH_SUBAGENTS, len(pending_records))
    if reused_paths:
        print(
            f"{prefix}AnalysisAgent: reused {len(reused_paths)} match summaries; "
            f"summarizing {len(pending_records)} new matches"
            + (f" (max_concurrency={worker_count})" if pending_records else ""),
            flush=True,
        )
    else:
        print(
            f"{prefix}AnalysisAgent: summarizing {len(records)} matches "
            f"(max_concurrency={worker_count})",
            flush=True,
        )

    if pending_records:
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="evol-match",
        )
        futures = {}
        try:
            for game_index, record in pending_records:
                task = partial(
                    run_fixed_match_summary,
                    strategy_name=strategy_name,
                    race=race,
                    record=record,
                    game_index=game_index,
                    model=model,
                    prefix=prefix,
                )
                futures[executor.submit(copy_context().run, task)] = (
                    game_index,
                    record,
                )
            for future in as_completed(futures):
                game_index, record = futures[future]
                try:
                    results[game_index] = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve the remaining batch
                    error = f"Match summary crashed: {type(exc).__name__}: {exc}"
                    digest = evidence_digest(record, game_index)
                    digest.summary = error
                    results[game_index] = (
                        digest,
                        fallback_analysis(
                            strategy_name=strategy_name,
                            race=race,
                            records=[record],
                            reason=error,
                        ),
                        False,
                        [error],
                        [{"action": "crashed", "error": error}],
                    )
        except KeyboardInterrupt:
            abandon_executor(executor, futures)
            exit_on_keyboard_interrupt("stopped during match summaries")
        else:
            executor.shutdown(wait=True)

    digests: list[GameDigest] = []
    analyses: list[BattleAnalysis] = []
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    completed = 0
    for game_index, record in enumerate(records, 1):
        digest, analysis, ok, item_errors, item_events = results[game_index]
        digests.append(digest)
        analyses.append(analysis)
        completed += int(ok)
        errors.extend(f"match_{game_index:03d}: {error}" for error in item_errors)
        events.append(
            {
                "stage": "match_summary",
                "game_index": game_index,
                "record_path": record.file,
                "completed": ok,
                "reused": str(Path(record.file).resolve()) in reused_paths,
                "events": item_events,
                "errors": item_errors,
            }
        )

    if checkpoint is not None and completed:
        checkpoint.save_match_summaries(
            game_digests=digests,
            single_game_analyses=analyses,
            completed_matches=completed,
            events=events,
            errors=errors,
        )
        print(
            f"{prefix}AnalysisAgent: checkpoint saved match_summaries -> {checkpoint.run_dir}",
            flush=True,
        )
    return digests, analyses, completed, events, errors


def run_analysis_agent_loop(
    *,
    strategy_name: str,
    race: str,
    records: list[Any],
    skill_texts: dict[str, str],
    model: str = "",
    knowledge_mode: str = "enabled",
    prefix: str = "  ",
    checkpoint: EvolCheckpoint | None = None,
    summary_seed_checkpoint: EvolCheckpoint | None = None,
    prior_experiences: list[Any] | None = None,
    capability_manifest: dict[str, Any] | None = None,
) -> AnalysisPipelineResult:
    capability_manifest = capability_manifest or {}
    model = str(model or "").strip() or DEFAULT_ANALYSIS_MODEL
    if not records:
        analysis = fallback_analysis(
            strategy_name=strategy_name,
            race=race,
            records=[],
            reason="No match records were supplied.",
        )
        return AnalysisPipelineResult(completed=False, battle_analysis=analysis)

    if checkpoint is not None and stage_reached(checkpoint.stage, "analysis_complete"):
        analysis, observations, trace, events, errors = (
            checkpoint.load_analysis_complete()
        )
        digests, summaries, _completed, match_events, match_errors = (
            checkpoint.load_match_summaries()
            if (checkpoint.run_dir / "match_summaries.json").is_file()
            else ([], [], 0, [], [])
        )
        print(
            f"{prefix}AnalysisAgent: resume loaded completed batch analysis",
            flush=True,
        )
        return AnalysisPipelineResult(
            completed=True,
            game_digests=digests,
            single_game_analyses=summaries,
            battle_analysis=analysis,
            tool_observations=observations,
            knowledge_trace=trace,
            errors=[*match_errors, *errors],
            events=[*match_events, *events],
        )

    digests, summaries, completed, match_events, errors = _summarize_matches(
        strategy_name=strategy_name,
        race=race,
        records=records,
        skill_texts=skill_texts,
        model=model,
        prefix=prefix,
        checkpoint=checkpoint,
        summary_seed_checkpoint=summary_seed_checkpoint,
    )
    if completed == 0:
        analysis = fallback_analysis(
            strategy_name=strategy_name,
            race=race,
            records=records,
            reason="No match summary produced usable evidence.",
        )
        return AnalysisPipelineResult(
            completed=False,
            game_digests=digests,
            single_game_analyses=summaries,
            battle_analysis=analysis,
            errors=errors,
            events=match_events,
        )

    degraded = len(records) - completed
    if degraded:
        errors.append(
            f"{degraded} of {len(records)} summaries are degraded; trajectory details are uncertain."
        )

    payload: dict[str, Any] | None = None
    discovery: dict[str, Any] | None = None
    analysis_events: list[dict[str, Any]] = []
    observations: list[ToolObservation] = []
    questions: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    llm_cross_match_calls = 0

    if checkpoint is not None and stage_reached(checkpoint.stage, "batch_analysis"):
        loaded = checkpoint.load_batch_analysis()
        payload, resume_error = _normalize_cross_match_decision(
            loaded, strategy_name=strategy_name
        )
        if resume_error:
            errors.append(f"old checkpoint analysis was not reusable: {resume_error}")
            payload = None
        elif payload is not None:
            retrieval_evidence = (
                dict(loaded.get("retrieval_evidence") or {})
                if isinstance(loaded.get("retrieval_evidence"), dict)
                else {}
            )
            runs = checkpoint.load_knowledge_results()
            observations = _observations_from_runs(runs, prefix=prefix) if runs else []
            questions = [
                {
                    "id": run.get("question_id"),
                    "question": run.get("question"),
                    "query_reason": run.get("query_reason"),
                    "evidence_refs": list(run.get("evidence_refs") or []),
                    "hypothesis_scope": run.get("hypothesis_scope"),
                }
                for run in runs
            ]

    if payload is None:
        if checkpoint is not None and checkpoint.has_cross_match_discovery():
            try:
                loaded_discovery, discovery_error = _normalize_cross_match_discovery(
                    checkpoint.load_cross_match_discovery(),
                    knowledge_mode=knowledge_mode,
                )
            except (OSError, ValueError) as exc:
                loaded_discovery, discovery_error = None, str(exc)
            if loaded_discovery is None:
                errors.append(f"old discovery was not reusable: {discovery_error}")
            else:
                discovery = loaded_discovery
                print(
                    f"{prefix}AnalysisAgent: resume loaded cross-match discovery",
                    flush=True,
                )

        if discovery is None:
            discovery, discovery_errors, discovery_calls = _run_cross_match_discovery(
                strategy_name=strategy_name,
                race=race,
                summaries=summaries,
                skill_texts=skill_texts,
                knowledge_mode=knowledge_mode,
                model=model,
                prior_experiences=prior_experiences,
                capability_manifest=capability_manifest,
                events=analysis_events,
                prefix=prefix,
            )
            llm_cross_match_calls += discovery_calls
            errors.extend(f"discovery: {item}" for item in discovery_errors)
            if discovery is not None and checkpoint is not None:
                checkpoint.save_cross_match_discovery(discovery)
                print(
                    f"{prefix}AnalysisAgent: checkpoint saved discovery -> {checkpoint.run_dir}",
                    flush=True,
                )

        if discovery is None:
            analysis = fallback_analysis(
                strategy_name=strategy_name,
                race=race,
                records=records,
                reason="Cross-match Discovery failed to produce a usable diagnosis.",
            )
            return AnalysisPipelineResult(
                completed=False,
                game_digests=digests,
                single_game_analyses=summaries,
                battle_analysis=analysis,
                errors=errors,
                events=[*match_events, *analysis_events],
            )

        questions = list(discovery.get("knowledge_questions") or [])
        retrieval_evidence = build_retrieval_evidence_packet(
            records=records,
            discovery=discovery,
            prior_experiences=prior_experiences,
        )
        match_query_summary = retrieval_evidence.get("match_record_evidence") or {}
        history_query_summary = (
            retrieval_evidence.get("historical_experience_evidence") or {}
        )
        print(
            f"{prefix}AnalysisAgent: retrieval record_refs="
            f"{match_query_summary.get('reference_count', 0)} "
            f"history_results={len(history_query_summary.get('results') or [])}",
            flush=True,
        )
        analysis_events.append(
            {
                "action": "retrieve_evidence",
                "record_reference_count": match_query_summary.get(
                    "reference_count", 0
                ),
                "history_result_count": len(
                    history_query_summary.get("results") or []
                ),
                "errors": list(match_query_summary.get("errors") or []),
            }
        )
        if knowledge_mode == "enabled" and questions:
            runs = _run_knowledge_queries(
                questions,
                race=race,
                checkpoint=checkpoint,
                prefix=prefix,
            )
            observations = _observations_from_runs(runs, prefix=prefix)

        payload, decision_errors, decision_calls = _run_cross_match_decision(
            strategy_name=strategy_name,
            race=race,
            summaries=summaries,
            skill_texts=skill_texts,
            knowledge_mode=knowledge_mode,
            model=model,
            prior_experiences=prior_experiences,
            capability_manifest=capability_manifest,
            discovery=discovery,
            knowledge_runs=runs,
            retrieval_evidence=retrieval_evidence,
            events=analysis_events,
            prefix=prefix,
        )
        llm_cross_match_calls += decision_calls
        errors.extend(f"decision: {item}" for item in decision_errors)

    if payload is None:
        analysis = fallback_analysis(
            strategy_name=strategy_name,
            race=race,
            records=records,
            reason="Cross-match Decision failed to produce one usable next action.",
        )
        return AnalysisPipelineResult(
            completed=False,
            game_digests=digests,
            single_game_analyses=summaries,
            battle_analysis=analysis,
            errors=errors,
            events=[*match_events, *analysis_events],
        )

    failed_questions = [
        str(run.get("question_id") or "")
        for run in runs
        if not is_knowledge_run_verified(run)
    ]
    if failed_questions:
        payload["evidence_limits"] = list(
            dict.fromkeys(
                [
                    *payload.get("evidence_limits", []),
                    f"Knowledge unavailable for: {', '.join(failed_questions)}",
                ]
            )
        )
    payload["knowledge_used"] = [
        {
            "question_id": run.get("question_id"),
            "query_reason": run.get("query_reason"),
            "evidence_refs": list(run.get("evidence_refs") or []),
            "hypothesis_scope": run.get("hypothesis_scope"),
            "finding": run.get("answer"),
        }
        for run in runs
        if is_knowledge_run_verified(run)
    ]
    payload["knowledge_queries"] = [
        {
            "question_id": run.get("question_id"),
            "query_reason": run.get("query_reason"),
            "evidence_refs": list(run.get("evidence_refs") or []),
            "hypothesis_scope": run.get("hypothesis_scope"),
            "ok": is_knowledge_run_verified(run),
        }
        for run in runs
    ]
    if retrieval_evidence:
        payload["retrieval_evidence"] = retrieval_evidence
    if checkpoint is not None:
        checkpoint.save_batch_analysis(payload)
        print(
            f"{prefix}AnalysisAgent: checkpoint saved final batch analysis -> {checkpoint.run_dir}",
            flush=True,
        )
    battle_analysis = analysis_from_json(
        strategy_name=strategy_name,
        race=race,
        records=records,
        data=payload,
    )
    knowledge_ok = sum(1 for run in runs if is_knowledge_run_verified(run))
    knowledge_trace = {
        "knowledge_mode": knowledge_mode,
        "discovery": discovery,
        "questions": questions,
        "runs": runs,
        "retrieval_evidence": retrieval_evidence,
        "failed_questions": failed_questions,
    }
    analysis_events.append(
        {
            "action": "analysis_complete",
            "llm_cross_match_calls": llm_cross_match_calls,
            "knowledge_questions": len(questions),
            "knowledge_answers_ok": knowledge_ok,
            "analysis": battle_analysis.raw,
        }
    )
    if checkpoint is not None:
        checkpoint.save_analysis_complete(
            battle_analysis=battle_analysis,
            tool_observations=observations,
            knowledge_trace=knowledge_trace,
            events=analysis_events,
            errors=errors,
        )
        print(
            f"{prefix}AnalysisAgent: checkpoint saved analysis_complete -> {checkpoint.run_dir}",
            flush=True,
        )

    return AnalysisPipelineResult(
        completed=True,
        game_digests=digests,
        single_game_analyses=summaries,
        battle_analysis=battle_analysis,
        tool_observations=observations,
        knowledge_trace=knowledge_trace,
        errors=errors,
        events=[*match_events, *analysis_events],
    )
