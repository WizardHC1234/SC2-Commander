from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from functools import partial
from pathlib import Path
import re
from typing import Any

from .checkpoint import (
    EvolCheckpoint,
    battle_analysis_from_dict,
    stage_reached,
)
from .config import (
    CROSS_MATCH_DECISION_ENABLE_REASONING,
    CROSS_MATCH_DISCOVERY_ENABLE_REASONING,
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
from .match_summary_cache import MatchSummaryCache
from .evidence_retrieval import build_retrieval_evidence_packet
from .prompts import (
    build_cross_match_decision_prompt,
    build_cross_match_discovery_prompt,
    build_optimization_package_prompt,
)
from .terran_build_order_simulator import simulate_terran_first_commitment
from .types import AnalysisPipelineResult, BattleAnalysis, GameDigest, ToolObservation
from ..sc2_data_agent import (
    build_knowledge_query,
    find_knowledge_run_error,
    is_knowledge_run_verified,
    resolve_knowledge_entities,
    run_knowledge_query,
)


# Each rejection is returned verbatim in the next prompt, so allow enough
# attempts for the model to repair a malformed or weakly grounded decision.
_ANALYSIS_ATTEMPTS = 4
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
_STRATEGY_AREAS = (
    "goal_identity",
    "economy_expansion",
    "production_order_capacity",
    "technology_composition",
    "attack_timing_objective",
    "reinforcement_retreat_cleanup",
)
_STRATEGY_AREA_DECISIONS = frozenset({"preserve", "revise"})
_OUTCOME_RELATIONSHIPS = frozenset(
    {
        "winning_mechanism_not_reproduced",
        "winning_mechanism_reproduced_but_failed",
        "mixed",
        "uncertain",
    }
)
_WINDOW_EFFECTS = frozenset({"earlier", "similar", "later", "unknown"})
_PACKAGE_NEXT_ACTIONS = frozenset({"evaluate_candidate_packages", "inspect_runtime"})
_FAILURE_STAGES = frozenset(
    {
        "before_core_mechanism",
        "during_commitment_or_engagement",
        "after_successful_engagement",
        "mixed",
    }
)
_PRESERVATION_EFFECTS = frozenset(
    {"preserve", "improve", "evidence_supported_tradeoff"}
)
_REQUIREMENT_FACT_TERMS = (
    "cost",
    "resource",
    "mineral",
    "gas",
    "supply",
    "build",
    "production",
    "producer",
    "prerequisite",
    "timing",
    "delay",
    "throughput",
    "queue",
    "upgrade",
    "reinforcement",
)
_COUNTER_FACT_TERMS = (
    "counter",
    "matchup",
    "composition",
    "versus",
    "against",
    "克制",
    "兵种搭配",
)
_EFFECT_FACT_TERMS = (
    "effect",
    "ability",
    "upgrade",
    "synergy",
    "support",
    "效果",
    "升级",
    "协同",
)


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


def _discovery_fact_text(discovery: dict[str, Any]) -> str:
    """Collect only diagnosis text used to ground a fallback data query."""
    contract = discovery.get("strategy_contract") or {}
    contrast = discovery.get("outcome_contrast") or {}
    values: list[str] = [
        str(contract.get("core_win_mechanism") or ""),
        str(contract.get("critical_power_window") or ""),
        *[str(item) for item in contract.get("core_commitments") or []],
        str(contrast.get("loss_shortfall") or ""),
        str(contrast.get("causal_difference") or ""),
    ]
    for field in (
        "weaknesses",
        "unknowns",
        "opponent_pressure_patterns",
        "matchup_patterns",
    ):
        for item in discovery.get(field) or []:
            if not isinstance(item, dict):
                continue
            values.extend(
                str(item.get(key) or "")
                for key in ("pattern", "unknown", "why_it_matters")
            )
    return " ".join(value.strip() for value in values if value.strip())


def _discovery_evidence_refs(discovery: dict[str, Any]) -> list[str]:
    contrast = discovery.get("outcome_contrast") or {}
    refs = [
        *list(contrast.get("winning_evidence") or []),
        *list(contrast.get("loss_evidence") or []),
    ]
    for field in (
        "strengths",
        "weaknesses",
        "opponent_pressure_patterns",
        "matchup_patterns",
    ):
        for item in discovery.get(field) or []:
            if isinstance(item, dict):
                refs.extend(item.get("evidence") or [])
    return list(dict.fromkeys(str(item).strip() for item in refs if str(item).strip()))[:6]


def _ensure_strategy_fact_query(
    discovery: dict[str, Any],
    *,
    knowledge_mode: str,
) -> dict[str, Any]:
    """Ensure strategy-factual diagnoses reach the deterministic Data Agent.

    Discovery remains the query planner. This fallback only covers the common
    failure mode where it makes production, timing, upgrade, or matchup claims
    but accidentally returns no structured query.
    """
    if knowledge_mode != "enabled" or discovery.get("knowledge_questions"):
        return discovery
    fact_text = _discovery_fact_text(discovery)
    if not fact_text:
        return discovery
    folded = fact_text.casefold()
    needs: list[str] = []
    if any(term in folded for term in _REQUIREMENT_FACT_TERMS):
        needs.append("requirements")
    elif any(term in folded for term in _COUNTER_FACT_TERMS):
        needs.append("counters")
    elif any(term in folded for term in _EFFECT_FACT_TERMS):
        needs.append("effects")
    if not needs:
        return discovery
    try:
        resolved = resolve_knowledge_entities(fact_text)
    except (OSError, ValueError):
        resolved = []
    entities = list(
        dict.fromkeys(
            str(item.get("name") or "").strip()
            for item in resolved
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        )
    )[:6]
    if not entities:
        return discovery
    if needs == ["requirements"]:
        requested_facts = "costs, production or build times, producers, and prerequisites"
    elif needs == ["counters"]:
        requested_facts = "deterministic counter relationships"
    else:
        requested_facts = "deterministic effects and capabilities"
    question = {
        "id": "Q1",
        "question": (
            "Using the bundled SC2 dataset tools as the source of truth, verify "
            f"{requested_facts} for {', '.join(entities)} before strategy editing."
        ),
        "entities": entities,
        "needs": needs,
        "query_reason": (
            "The discovery diagnosis relies on these static SC2 facts, but did not "
            "return a structured Data Agent query."
        ),
        "evidence_refs": _discovery_evidence_refs(discovery),
        "hypothesis_scope": "verify_the_primary_strategy_fact_dependency",
        "calculations": [],
        "source": "deterministic_fallback_from_discovery",
    }
    output = dict(discovery)
    output["knowledge_questions"] = [question]
    query_plan = dict(output.get("query_plan") or {})
    query_plan["game_knowledge_queries"] = [question]
    output["query_plan"] = query_plan
    return output


def _retrievable_evidence_ref(value: str) -> str:
    """Normalize common `Game N: ... at Ts` model output for record lookup."""
    text = str(value or "").strip()
    if re.search(r"Game\s+\d+\s*@\s*\d", text, re.IGNORECASE):
        return text
    game = re.search(r"Game\s+(\d+)", text, re.IGNORECASE)
    time = re.search(r"(?:@|\bat)\s*(\d+(?:\.\d+)?)\s*s\b", text, re.IGNORECASE)
    if not game or not time:
        return text
    detail = text.split(":", 1)[1].strip() if ":" in text else text
    return f"Game {game.group(1)} @ {time.group(1)}s: {detail}"


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
        refs = [_retrievable_evidence_ref(ref) for ref in refs]
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
        refs = [
            _retrievable_evidence_ref(ref)
            for ref in _clean_strings(pattern.get("evidence"), limit=6)
        ]
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
    strategy_name: str = "strategy",
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
    outcome_contrast = _normalize_outcome_contrast(raw.get("outcome_contrast"))
    raw_query_plan = raw.get("query_plan") if isinstance(raw.get("query_plan"), dict) else {}
    questions = (
        _normalize_knowledge_questions(
            raw_query_plan.get("game_knowledge_queries")
            or raw.get("knowledge_questions")
        )
        if knowledge_mode == "enabled"
        else []
    )
    fallback_patterns = [
        *weaknesses,
        {
            "pattern": outcome_contrast.get("loss_shortfall") or "loss outcome contrast",
            "evidence": outcome_contrast.get("loss_evidence") or [],
        },
        {
            "pattern": outcome_contrast.get("winning_pattern") or "winning outcome contrast",
            "evidence": outcome_contrast.get("winning_evidence") or [],
        },
        *pressure_patterns,
        *matchup_patterns,
    ]
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
    discovery = {
        "strategy_contract": normalize_strategy_contract(
            raw.get("strategy_contract"), strategy_name=strategy_name
        ),
        "outcome_contrast": outcome_contrast,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "unknowns": unknowns,
        "opponent_pressure_patterns": pressure_patterns,
        "matchup_patterns": matchup_patterns,
        "knowledge_questions": questions,
        "query_plan": query_plan,
    }
    return (
        _ensure_strategy_fact_query(discovery, knowledge_mode=knowledge_mode),
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
        contact_window_effect = "unknown"
        new_hard_prerequisites: list[str] = []
        production_tradeoffs: list[str] = []
        window_tradeoff_evidence: list[str] = []
        why_window_remains_favorable = ""
        preservation_checks: list[dict[str, Any]] = []
        composition_change_allowed = False
        retreat_change_allowed = False
        stage_scope_evidence: list[str] = []
        stage_scope_reason = ""
        strategy_area_audit: list[dict[str, Any]] = []
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
            if change:
                coordinated_changes.append(
                    {
                        "change": change,
                        "why_required": why_required
                        or "Required by the selected coherent intervention.",
                    }
                )
            if len(coordinated_changes) >= 10:
                break
        preserve = _clean_strings(raw.get("preserve"), limit=5)
        contact_window_effect = str(
            raw.get("contact_window_effect") or "unknown"
        ).strip().lower()
        if contact_window_effect not in _WINDOW_EFFECTS:
            contact_window_effect = "unknown"
        new_hard_prerequisites = _clean_strings(
            raw.get("new_hard_prerequisites"), limit=8
        )
        production_tradeoffs = _clean_strings(
            raw.get("production_tradeoffs"), limit=8
        )
        window_tradeoff_evidence = _clean_strings(
            raw.get("window_tradeoff_evidence"), limit=6
        )
        why_window_remains_favorable = str(
            raw.get("why_window_remains_favorable") or ""
        ).strip()
        composition_change_allowed = bool(raw.get("composition_change_allowed"))
        retreat_change_allowed = bool(raw.get("retreat_change_allowed"))
        stage_scope_evidence = _clean_strings(
            raw.get("stage_scope_evidence"), limit=6
        )
        stage_scope_reason = str(raw.get("stage_scope_reason") or "").strip()
        strategy_area_audit = []
        seen_areas: set[str] = set()
        for item in raw.get("strategy_area_audit") or []:
            if not isinstance(item, dict):
                continue
            area = str(item.get("area") or "").strip().lower()
            decision = str(item.get("decision") or "").strip().lower()
            if (
                area not in _STRATEGY_AREAS
                or area in seen_areas
                or decision not in _STRATEGY_AREA_DECISIONS
            ):
                continue
            required_change = str(item.get("required_change") or "").strip()
            if decision == "revise" and not required_change:
                continue
            strategy_area_audit.append(
                {
                    "area": area,
                    "decision": decision,
                    "finding": str(item.get("finding") or "").strip(),
                    "required_change": required_change,
                    "evidence": _clean_strings(item.get("evidence"), limit=4),
                }
            )
            seen_areas.add(area)
        preservation_checks = []
        for item in raw.get("preservation_checks") or []:
            if not isinstance(item, dict):
                continue
            invariant = str(item.get("invariant") or "").strip()
            effect = str(item.get("effect") or "").strip().lower()
            if not invariant or effect not in _PRESERVATION_EFFECTS:
                continue
            preservation_checks.append(
                {
                    "invariant": invariant,
                    "effect": effect,
                    "reason": str(item.get("reason") or "").strip(),
                    "evidence": _clean_strings(item.get("evidence"), limit=4),
                }
            )
            if len(preservation_checks) >= 8:
                break
    else:
        return None, "plan must be an object with direction"
    if not direction:
        return None, "plan.direction is required"
    if not material_behavior_change:
        material_behavior_change = direction
    if not coordinated_changes:
        coordinated_changes = [
            {
                "change": direction,
                "why_required": "Implements the evidence-supported optimization direction.",
            }
        ]
    return {
        "direction": direction,
        "material_behavior_change": material_behavior_change,
        "coordinated_changes": coordinated_changes,
        "preserve": preserve,
        "contact_window_effect": contact_window_effect,
        "new_hard_prerequisites": new_hard_prerequisites,
        "production_tradeoffs": production_tradeoffs,
        "window_tradeoff_evidence": window_tradeoff_evidence,
        "why_window_remains_favorable": why_window_remains_favorable,
        "preservation_checks": preservation_checks,
        "composition_change_allowed": composition_change_allowed,
        "retreat_change_allowed": retreat_change_allowed,
        "stage_scope_evidence": stage_scope_evidence,
        "stage_scope_reason": stage_scope_reason,
        "strategy_area_audit": strategy_area_audit,
    }, ""


def _optional_nonnegative_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _normalize_timing_components(
    raw: Any,
    *,
    slot_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return rows
    for item in raw:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip()
        try:
            quantity = int(item.get("quantity") or 0)
            slots = int(item.get(slot_key) or 0)
        except (TypeError, ValueError):
            continue
        if action and quantity > 0 and slots > 0:
            rows.append({"action": action, "quantity": quantity, slot_key: slots})
    return rows[:16]


def _normalize_timing_package(raw: Any) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "timing package must be an object"
    economy_raw = raw.get("economy") if isinstance(raw.get("economy"), dict) else {}
    economy: dict[str, int | None] = {}
    for field in (
        "worker_target_before_commitment",
        "base_target_before_commitment",
        "gas_workers_before_commitment",
    ):
        value = economy_raw.get(field)
        if value in (None, ""):
            economy[field] = None
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = -1
        economy[field] = parsed if parsed >= 0 else None
    gate_components = _normalize_timing_components(
        raw.get("gate_components"), slot_key="production_slots"
    )
    setup_actions = _normalize_timing_components(
        raw.get("setup_actions"), slot_key="parallel_slots"
    )
    if not gate_components:
        return None, "timing package requires at least one gate component"
    return {
        "economy": economy,
        "gate_components": gate_components,
        "setup_actions": setup_actions,
    }, ""


def _normalize_candidate_package_proposal(
    raw: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "optimization package proposal returned no JSON object"
    next_action = str(raw.get("next_action") or "").strip()
    if next_action not in _PACKAGE_NEXT_ACTIONS:
        return None, "next_action must be evaluate_candidate_packages or inspect_runtime"
    priority, priority_error = _normalize_priority_problem(raw.get("priority_problem"))
    if priority_error:
        return None, priority_error
    failure_mode, _failure_error = _normalize_failure_mode_analysis(
        raw.get("failure_mode_analysis")
    )
    parent_timing_package, parent_error = _normalize_timing_package(
        raw.get("parent_timing_package")
    )
    common = {
        "strengths_to_preserve": _normalize_pattern_items(
            raw.get("strengths_to_preserve") or raw.get("strengths")
        ),
        "priority_problem": priority or {},
        "failure_mode_analysis": failure_mode,
        "next_action": next_action,
        "action_reason": str(raw.get("action_reason") or "").strip(),
        "evidence_limits": _clean_strings(raw.get("evidence_limits")),
    }
    if next_action == "inspect_runtime":
        return {
            **common,
            "candidate_packages": [],
            "parent_timing_package": parent_timing_package or {},
        }, ""
    if parent_error or parent_timing_package is None:
        return None, (
            "evaluate_candidate_packages requires a usable parent_timing_package: "
            + parent_error
        )

    packages: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw.get("candidate_packages") or [], 1):
        if not isinstance(item, dict):
            continue
        package_id = str(item.get("id") or f"P{index}").strip().upper()
        hypothesis = str(item.get("hypothesis") or "").strip()
        plan, plan_error = _normalize_plan(item.get("plan"))
        timing_raw = item.get("timing_budget")
        timing_raw = timing_raw if isinstance(timing_raw, dict) else {}
        timing_package, timing_error = _normalize_timing_package(
            timing_raw.get("package")
        )
        target_latest = _optional_nonnegative_number(
            timing_raw.get("target_latest_first_commitment_seconds")
        )
        maximum_added = _optional_nonnegative_number(
            timing_raw.get("maximum_added_feasibility_seconds")
        )
        budget_basis = _clean_strings(timing_raw.get("budget_basis"), limit=5)
        if (
            not package_id
            or package_id in seen_ids
            or not hypothesis
            or plan_error
            or plan is None
            or timing_error
            or timing_package is None
            or (target_latest is None and maximum_added is None)
            or not budget_basis
        ):
            continue
        seen_ids.add(package_id)
        packages.append(
            {
                "id": package_id,
                "hypothesis": hypothesis,
                "plan": plan,
                "timing_budget": {
                    "target_latest_first_commitment_seconds": target_latest,
                    "maximum_added_feasibility_seconds": maximum_added,
                    "budget_basis": budget_basis,
                    "package": timing_package,
                },
                "engagement_assessment": (
                    {
                        key: str((item.get("engagement_assessment") or {}).get(key) or "").strip()
                        for key in (
                            "intended_contact_window",
                            "own_package_role",
                            "observed_opponent_package",
                            "counter_and_upgrade_relationship",
                            "reinforcement_and_continuity",
                        )
                    }
                    if isinstance(item.get("engagement_assessment"), dict)
                    else {}
                ),
                "expected_effect": str(item.get("expected_effect") or "").strip(),
                "main_risk": str(item.get("main_risk") or "").strip(),
            }
        )
        if len(packages) >= 3:
            break
    if len(packages) < 2:
        return None, (
            "evaluate_candidate_packages requires two or three usable hypothesis packages"
        )
    return {
        **common,
        "parent_timing_package": parent_timing_package,
        "candidate_packages": packages,
    }, ""


def _compact_window_state(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        key: value.get(key)
        for key in ("army", "technology", "buildings", "intel")
        if value.get(key) not in (None, "", [], {})
    }


def _nearest_window_event(
    summary: BattleAnalysis,
    target_seconds: float | None,
) -> dict[str, Any] | None:
    if not isinstance(target_seconds, (int, float)):
        return None
    events = [
        item
        for item in (summary.raw.get("events") or [])
        if isinstance(item, dict) and isinstance(item.get("time_s"), (int, float))
    ]
    if not events:
        return None
    event = min(events, key=lambda item: abs(float(item["time_s"]) - target_seconds))
    event_time = float(event["time_s"])
    return {
        "time_s": round(event_time, 1),
        "offset_from_target_s": round(event_time - float(target_seconds), 1),
        "trigger": str(event.get("trigger") or ""),
        "own_state": _compact_window_state(event.get("own_state")),
        "enemy_observed": _compact_window_state(event.get("enemy_observed")),
        "enemy_truth": _compact_window_state(event.get("enemy_truth")),
    }


def _empirical_opponent_windows(
    summaries: list[BattleAnalysis] | None,
    *,
    parent_seconds: float | None,
    candidate_seconds: float | None,
) -> list[dict[str, Any]]:
    """Join simulated package times to the nearest recorded opponent snapshots."""
    rows: list[dict[str, Any]] = []
    for game_index, summary in enumerate(summaries or [], 1):
        parent_snapshot = _nearest_window_event(summary, parent_seconds)
        candidate_snapshot = _nearest_window_event(summary, candidate_seconds)
        engagements = [
            item
            for item in (summary.raw.get("major_engagements") or [])
            if isinstance(item, dict)
            and isinstance(item.get("time_s"), (int, float))
        ]
        first_relevant = next(
            (
                item
                for item in sorted(engagements, key=lambda row: float(row["time_s"]))
                if not isinstance(candidate_seconds, (int, float))
                or float(item["time_s"]) >= float(candidate_seconds)
            ),
            engagements[0] if engagements else None,
        )
        engagement = None
        if isinstance(first_relevant, dict):
            engagement = {
                key: first_relevant.get(key)
                for key in (
                    "time_s",
                    "own_force_before",
                    "enemy_observed",
                    "enemy_truth",
                    "own_force_after",
                    "own_reinforcement_after",
                    "retreat_policy",
                    "loss_timing",
                    "outcome",
                )
                if first_relevant.get(key) not in (None, "", [], {})
            }
        if parent_snapshot or candidate_snapshot or engagement:
            rows.append(
                {
                    "game": game_index,
                    "result": str(summary.raw.get("result") or ""),
                    "parent_window": parent_snapshot,
                    "candidate_window": candidate_snapshot,
                    "first_engagement_at_or_after_candidate_window": engagement,
                }
            )
        if len(rows) >= 10:
            break
    return rows


def _evaluate_candidate_package_budgets(
    proposal: dict[str, Any],
    *,
    race: str,
    summaries: list[BattleAnalysis] | None = None,
) -> list[dict[str, Any]]:
    packages = list(proposal.get("candidate_packages") or [])
    if str(race or "").strip().casefold() != "terran":
        return [
            {
                "id": item.get("id"),
                "status": "unavailable_for_race",
                "complete": False,
            }
            for item in packages
            if isinstance(item, dict)
        ]
    parent = simulate_terran_first_commitment(
        proposal.get("parent_timing_package") or {}
    )
    parent_time = parent.get("earliest_feasible_time_seconds")
    parent_cost = (
        parent.get("total_cost")
        if isinstance(parent.get("total_cost"), dict)
        else {}
    )
    reports: list[dict[str, Any]] = []
    for item in packages:
        if not isinstance(item, dict):
            continue
        budget = (
            item.get("timing_budget")
            if isinstance(item.get("timing_budget"), dict)
            else {}
        )
        candidate = simulate_terran_first_commitment(budget.get("package") or {})
        candidate_time = candidate.get("earliest_feasible_time_seconds")
        target_latest = budget.get("target_latest_first_commitment_seconds")
        maximum_added = budget.get("maximum_added_feasibility_seconds")
        delta = (
            float(candidate_time) - float(parent_time)
            if isinstance(candidate_time, (int, float))
            and isinstance(parent_time, (int, float))
            else None
        )
        latest_ok = (
            None
            if target_latest is None or not isinstance(candidate_time, (int, float))
            else float(candidate_time) <= float(target_latest)
        )
        delay_ok = (
            None
            if maximum_added is None or delta is None
            else delta <= float(maximum_added)
        )
        candidate_cost = (
            candidate.get("total_cost")
            if isinstance(candidate.get("total_cost"), dict)
            else {}
        )
        complete = bool(parent.get("complete") and candidate.get("complete"))
        within_budget = complete and latest_ok is not False and delay_ok is not False
        reports.append(
            {
                "id": item.get("id"),
                "status": (
                    "within_budget" if within_budget else "over_or_unresolved_budget"
                ),
                "complete": complete,
                "parent_earliest_feasible_time_seconds": parent_time,
                "candidate_earliest_feasible_time_seconds": candidate_time,
                "earliest_feasible_timing_delta_seconds": (
                    round(delta, 3) if delta is not None else None
                ),
                "target_latest_first_commitment_seconds": target_latest,
                "maximum_added_feasibility_seconds": maximum_added,
                "target_latest_satisfied": latest_ok,
                "maximum_added_delay_satisfied": delay_ok,
                "gate_cost_delta": {
                    key: round(
                        float(candidate_cost.get(key) or 0.0)
                        - float(parent_cost.get(key) or 0.0),
                        3,
                    )
                    for key in ("minerals", "gas", "supply")
                },
                "declared_candidate_package": dict(budget.get("package") or {}),
                "empirical_opponent_windows": _empirical_opponent_windows(
                    summaries,
                    parent_seconds=(
                        float(parent_time)
                        if isinstance(parent_time, (int, float))
                        else None
                    ),
                    candidate_seconds=(
                        float(candidate_time)
                        if isinstance(candidate_time, (int, float))
                        else None
                    ),
                ),
                "bottlenecks": list(candidate.get("bottlenecks") or []),
                "warnings": list(candidate.get("warnings") or []),
                "errors": list(candidate.get("errors") or []),
            }
        )
    return reports


def _normalize_failure_mode_analysis(
    raw: Any,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return {}, ""
    fields = (
        "failure_mode",
        "survival_prerequisite",
        "opponent_pressure_pattern",
        "matchup_assessment",
        "counterexample_check",
    )
    normalized = {field: str(raw.get(field) or "").strip() for field in fields}
    covered_failures = _clean_strings(raw.get("covered_failures"), limit=10)
    unexplained_failures = _clean_strings(raw.get("unexplained_failures"), limit=10)
    counterexamples = _clean_strings(raw.get("counterexamples"), limit=10)
    normalized.update(
        {
            "covered_failures": covered_failures,
            "unexplained_failures": unexplained_failures,
            "counterexamples": counterexamples,
        }
    )
    # These fields preserve the decision agent's compact combat-causal chain for
    # the optimizer and experiment history. They are optional for compatibility
    # with older checkpoints, but new prompts request all of them.
    for field in (
        "failure_stage",
        "gate_attainment_and_launch",
        "earliest_strategy_fixable_link",
        "why_later_levers_do_not_outrank_it",
        "commitment_and_contact_timing",
        "own_package_at_contact",
        "opponent_package_and_growth",
        "post_contact_continuity",
        "production_feasibility",
        "optimization_implication",
    ):
        value = str(raw.get(field) or "").strip()
        if value:
            normalized[field] = value
    stage = str(raw.get("failure_stage") or "").strip().lower()
    if stage and stage not in _FAILURE_STAGES:
        stage = ""
    if stage:
        normalized["failure_stage"] = stage
    return normalized, ""


def _normalize_outcome_contrast(raw: Any) -> dict[str, Any]:
    """Keep the win/loss comparison explicit across analysis and optimization."""
    value = raw if isinstance(raw, dict) else {}
    relationship = str(
        value.get("loss_relationship_to_wins")
        or value.get("relationship")
        or "uncertain"
    ).strip().lower()
    if relationship not in _OUTCOME_RELATIONSHIPS:
        relationship = "uncertain"
    return {
        "winning_pattern": str(value.get("winning_pattern") or "").strip(),
        "winning_evidence": _clean_strings(value.get("winning_evidence"), limit=6),
        "loss_shortfall": str(value.get("loss_shortfall") or "").strip(),
        "loss_evidence": _clean_strings(value.get("loss_evidence"), limit=6),
        "loss_relationship_to_wins": relationship,
        "causal_difference": str(value.get("causal_difference") or "").strip(),
        "preservation_rule": str(value.get("preservation_rule") or "").strip(),
    }


def _normalize_priority_alignment(
    raw: Any,
) -> tuple[dict[str, str] | None, str]:
    if not isinstance(raw, dict):
        return {}, ""
    fields = (
        "selected_priority",
        "higher_priority_assessment",
        "downstream_combat_effect",
    )
    normalized = {field: str(raw.get(field) or "").strip() for field in fields}
    return normalized, ""


def _normalize_retrieval_assessment(
    raw: Any,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return {}, ""
    query_summary = str(raw.get("query_summary") or "").strip()
    confidence = str(raw.get("confidence") or "").strip().lower()
    if confidence not in _CONFIDENCE:
        confidence = "medium"
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


_STATIC_DEFENSE_DIRECTION_TERMS = (
    "missile turret",
    "photon cannon",
    "spore crawler",
    "spine crawler",
    "bunker",
    "static defense",
    "static detection",
    "防空塔",
    "光子炮",
    "地堡",
    "静态防御",
)


def _static_defense_direction_error(plan: dict[str, Any] | None) -> str:
    """Keep static base structures out of the selected evolution mechanism."""
    if not isinstance(plan, dict):
        return ""
    selected_text = " ".join(
        [
            str(plan.get("direction") or ""),
            str(plan.get("material_behavior_change") or ""),
            *[
                " ".join(
                    [
                        str(item.get("change") or ""),
                        str(item.get("why_required") or ""),
                    ]
                )
                for item in (plan.get("coordinated_changes") or [])
                if isinstance(item, dict)
            ],
        ]
    ).casefold()
    if any(term in selected_text for term in _STATIC_DEFENSE_DIRECTION_TERMS):
        return (
            "the selected optimization direction cannot use static defensive "
            "structures as its primary mechanism; choose a mobile army, executable "
            "combat control, readiness, production, timing, or recovery change"
        )
    return ""


def _runtime_attribution_error(
    *,
    next_action: str,
    action_reason: str,
    priority: dict[str, Any] | None,
    hypothesis: str,
    failure_mode_analysis: dict[str, Any] | None,
) -> str:
    """Require group-level causal evidence before escalating auto-retreat."""
    if next_action != "inspect_runtime":
        return ""
    evidence = _clean_strings((priority or {}).get("evidence"), limit=20)
    covered = _clean_strings(
        (failure_mode_analysis or {}).get("covered_failures"), limit=20
    )
    combined = " ".join([action_reason, hypothesis, *evidence, *covered]).casefold()
    if "auto-retreat" not in combined and "auto retreat" not in combined:
        return ""

    causal_items: list[str] = []
    for item in [*evidence, *covered]:
        folded = item.casefold()
        identifies_main_force = (
            "main_force" in folded
            or "main force" in folded
            or "group_0" in folded
        )
        establishes_order = (
            "loss_timing=override_before_losses" in folded
            or "override_before_losses" in folded
            or "retreat before losses" in folded
            or "retreat preceded losses" in folded
        )
        if identifies_main_force and establishes_order:
            causal_items.append(item)
    if len({item.casefold() for item in causal_items}) < 2:
        return (
            "inspect_runtime based on auto-retreat requires at least two distinct "
            "match evidence items that identify the affected main force/group_0 "
            "and explicitly establish loss_timing=override_before_losses; a "
            "reinforcement-group retreat or a global army inventory at the same "
            "timestamp does not establish a runtime-caused main-force collapse"
        )
    return ""


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
    missing = [
        field
        for field in ("expected_change", "minimum_material_change", "disproof_condition")
        if not normalized[field]
    ]
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
    """Keep the live Decision payload small; readers handle old checkpoints."""
    return {**decision, "strategy_name": strategy_name}


def _normalize_cross_match_decision(
    raw: dict[str, Any],
    *,
    strategy_name: str,
    require_retrieval_assessment: bool = False,
    retrieval_evidence: dict[str, Any] | None = None,
    knowledge_runs: list[dict[str, Any]] | None = None,
    fallback_strategy_contract: dict[str, Any] | None = None,
    fallback_outcome_contrast: dict[str, Any] | None = None,
    require_outcome_contract: bool = False,
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
    outcome_contrast = _normalize_outcome_contrast(
        raw.get("outcome_contrast") or fallback_outcome_contrast
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
    if next_action == "propose_strategy_patch" and plan and not mechanism_prediction:
        material_change = str(
            plan.get("material_behavior_change") or plan.get("direction") or hypothesis
        ).strip()
        mechanism_prediction = {
            "expected_change": material_change,
            "minimum_material_change": material_change,
            "outcome_prediction": hypothesis or material_change,
            "combat_success_measure": (
                "Improve the decisive engagement, continued pressure, or match outcome."
            ),
            "disproof_condition": (
                "The intended behavior is observed in candidate matches but combat and "
                "match performance do not improve."
            ),
        }
        mechanism_error = ""
    runtime_attribution_error = _runtime_attribution_error(
        next_action=next_action,
        action_reason=action_reason,
        priority=priority,
        hypothesis=hypothesis,
        failure_mode_analysis=failure_mode_analysis,
    )
    if runtime_attribution_error:
        return None, runtime_attribution_error

    if next_action == "propose_strategy_patch":
        control = str((priority or {}).get("control_class") or "")
        if control in {"runtime_execution", "commander_execution"}:
            next_action = "inspect_runtime"
            action_reason = action_reason or (
                "Priority problem is an execution defect, not a strategy.md change."
            )
            runtime_attribution_error = _runtime_attribution_error(
                next_action=next_action,
                action_reason=action_reason,
                priority=priority,
                hypothesis=hypothesis,
                failure_mode_analysis=failure_mode_analysis,
            )
            if runtime_attribution_error:
                return None, runtime_attribution_error
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
            return None, mechanism_error or "propose_strategy_patch requires a usable change"
        elif require_outcome_contract and not failure_mode_analysis.get("failure_stage"):
            return None, (
                "propose_strategy_patch requires failure_mode_analysis.failure_stage"
            )
        elif require_outcome_contract and any(
            not failure_mode_analysis.get(field)
            for field in (
                "gate_attainment_and_launch",
                "earliest_strategy_fixable_link",
                "why_later_levers_do_not_outrank_it",
                "commitment_and_contact_timing",
                "own_package_at_contact",
                "opponent_package_and_growth",
                "post_contact_continuity",
                "production_feasibility",
                "optimization_implication",
            )
        ):
            return None, (
                "propose_strategy_patch requires a complete causal-order analysis "
                "from gate attainment through contact and post-contact continuity"
            )
        elif require_outcome_contract and not plan.get("stage_scope_reason"):
            return None, (
                "propose_strategy_patch requires plan.stage_scope_reason for "
                "composition and retreat permissions"
            )
        elif (
            require_outcome_contract
            and (
                plan.get("composition_change_allowed")
                or plan.get("retreat_change_allowed")
            )
        ) and len(
            {
                str(item).strip().casefold()
                for item in (plan.get("stage_scope_evidence") or [])
                if str(item).strip()
            }
        ) < 2:
            return None, (
                "composition or retreat changes require at least two distinct "
                "plan.stage_scope_evidence references"
            )
        elif require_outcome_contract and not outcome_contrast.get("preservation_rule"):
            return None, (
                "propose_strategy_patch requires outcome_contrast.preservation_rule "
                "derived from wins and losses"
            )
        elif require_outcome_contract and not plan.get("preservation_checks"):
            return None, (
                "propose_strategy_patch requires plan.preservation_checks for the "
                "Champion's winning mechanisms"
            )
        elif (
            require_outcome_contract
            and plan.get("contact_window_effect") == "later"
            and not plan.get("window_tradeoff_evidence")
        ):
            return None, (
                "a later contact window requires plan.window_tradeoff_evidence"
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
        "outcome_contrast": outcome_contrast,
        "priority_problem": priority or {},
        "hypothesis": hypothesis,
        "mechanism_family": mechanism_family,
        "failure_mode_analysis": failure_mode_analysis if not failure_mode_error else {},
        "priority_alignment": priority_alignment if not priority_alignment_error else {},
        "mechanism_prediction": mechanism_prediction or {},
        "retrieval_assessment": retrieval_assessment if not retrieval_error else {},
        "next_action": next_action,
        "action_reason": action_reason,
        "considered_explanations": _normalize_considered_explanations(
            raw.get("considered_explanations")
        ),
        "plan": plan,
        "selected_package_id": str(raw.get("selected_package_id") or "").strip(),
        "selected_timing_budget": (
            dict(raw.get("selected_timing_budget"))
            if isinstance(raw.get("selected_timing_budget"), dict)
            else {}
        ),
        "selected_package_budget": (
            dict(raw.get("selected_package_budget"))
            if isinstance(raw.get("selected_package_budget"), dict)
            else {}
        ),
        "candidate_packages": [
            dict(item)
            for item in (raw.get("candidate_packages") or [])
            if isinstance(item, dict)
        ],
        "package_budget_reports": [
            dict(item)
            for item in (raw.get("package_budget_reports") or [])
            if isinstance(item, dict)
        ],
        "evidence_limits": _clean_strings(raw.get("evidence_limits")),
        "strategy_contract": normalize_strategy_contract(
            raw.get("strategy_contract") or fallback_strategy_contract,
            strategy_name=strategy_name,
        ),
    }
    return _adapt_decision_for_optimizer(decision, strategy_name=strategy_name), ""


def _normalize_package_selection(
    raw: dict[str, Any],
    *,
    proposal: dict[str, Any],
    package_budget_reports: list[dict[str, Any]],
    strategy_name: str,
    fallback_strategy_contract: dict[str, Any] | None,
    fallback_outcome_contrast: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "package selection returned no JSON object"
    selected_id = str(raw.get("selected_package_id") or "").strip().upper()
    selected = next(
        (
            item
            for item in (proposal.get("candidate_packages") or [])
            if isinstance(item, dict)
            and str(item.get("id") or "").strip().upper() == selected_id
        ),
        None,
    )
    if selected is None:
        return None, "selected_package_id must name one proposed optimization package"
    selected_report = next(
        (
            item
            for item in package_budget_reports
            if isinstance(item, dict)
            and str(item.get("id") or "").strip().upper() == selected_id
        ),
        {},
    )
    merged = {
        **raw,
        "strengths_to_preserve": proposal.get("strengths_to_preserve") or [],
        "priority_problem": proposal.get("priority_problem") or {},
        "failure_mode_analysis": proposal.get("failure_mode_analysis") or {},
        "hypothesis": selected.get("hypothesis"),
        "plan": selected.get("plan"),
    }
    payload, error = _normalize_cross_match_decision(
        merged,
        strategy_name=strategy_name,
        fallback_strategy_contract=fallback_strategy_contract,
        fallback_outcome_contrast=fallback_outcome_contrast,
    )
    if payload is None:
        return None, error
    payload["selected_package_id"] = selected_id
    payload["candidate_packages"] = list(proposal.get("candidate_packages") or [])
    payload["package_budget_reports"] = list(package_budget_reports)
    payload["selected_package_budget"] = dict(selected_report)
    payload["selected_timing_budget"] = dict(selected.get("timing_budget") or {})
    payload["selected_engagement_assessment"] = dict(
        selected.get("engagement_assessment") or {}
    )
    payload["expected_effect"] = str(selected.get("expected_effect") or "").strip()
    payload["main_risk"] = str(selected.get("main_risk") or "").strip()
    return payload, ""


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
    is_reasoning: bool,
) -> tuple[dict[str, Any] | None, list[str], int]:
    schema_errors: list[str] = []
    payload: dict[str, Any] | None = None
    calls = 0
    for attempt in range(1, _ANALYSIS_ATTEMPTS + 1):
        calls += 1
        result = call_json_llm(
            build_prompt(schema_errors),
            model=model,
            is_reasoning=is_reasoning,
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
            raw,
            knowledge_mode=knowledge_mode,
            strategy_name=strategy_name,
        ),
        events=events,
        label="cross_match_discovery",
        is_reasoning=CROSS_MATCH_DISCOVERY_ENABLE_REASONING,
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
    print(f"{prefix}AnalysisAgent: generating optimization packages", flush=True)
    proposal, proposal_errors, proposal_calls = _run_analysis_json(
        build_prompt=lambda schema_errors: build_optimization_package_prompt(
            strategy_name=strategy_name,
            race=race,
            single_game_analyses=summaries,
            skill_texts=skill_texts,
            validation_errors=schema_errors,
            prior_experiences=prior_experiences or [],
            discovery=discovery,
            knowledge_runs=knowledge_runs,
            retrieval_evidence=retrieval_evidence,
            capability_manifest=capability_manifest or {},
        ),
        model=model,
        normalizer=_normalize_candidate_package_proposal,
        events=events,
        label="optimization_package_generation",
        is_reasoning=CROSS_MATCH_DECISION_ENABLE_REASONING,
    )
    if proposal is None:
        return None, proposal_errors, proposal_calls
    if proposal.get("next_action") == "inspect_runtime":
        payload = {
            "strategy_name": strategy_name,
            "strategy_contract": normalize_strategy_contract(
                discovery.get("strategy_contract"), strategy_name=strategy_name
            ),
            "outcome_contrast": _normalize_outcome_contrast(
                discovery.get("outcome_contrast")
            ),
            "strengths_to_preserve": proposal.get("strengths_to_preserve") or [],
            "priority_problem": proposal.get("priority_problem") or {},
            "hypothesis": "",
            "failure_mode_analysis": proposal.get("failure_mode_analysis") or {},
            "mechanism_prediction": {},
            "plan": None,
            "candidate_packages": [],
            "package_budget_reports": [],
            "next_action": "inspect_runtime",
            "action_reason": proposal.get("action_reason") or "Runtime inspection required.",
            "evidence_limits": proposal.get("evidence_limits") or [],
        }
        return payload, proposal_errors, proposal_calls

    print(
        f"{prefix}DataAgent: preflighting "
        f"{len(proposal.get('candidate_packages') or [])} optimization package(s)",
        flush=True,
    )
    budget_reports = _evaluate_candidate_package_budgets(
        proposal,
        race=race,
        summaries=summaries,
    )
    for report in budget_reports:
        print(
            f"{prefix}DataAgent: package {report.get('id')} "
            f"earliest={report.get('candidate_earliest_feasible_time_seconds')}s "
            f"delta={report.get('earliest_feasible_timing_delta_seconds')}s "
            f"opponent_windows={len(report.get('empirical_opponent_windows') or [])} "
            f"status={report.get('status')}",
            flush=True,
        )
    events.append(
        {
            "action": "optimization_package_preflight",
            "package_count": len(budget_reports),
            "reports": budget_reports,
        }
    )

    print(f"{prefix}AnalysisAgent: selecting optimization package", flush=True)
    payload, selection_errors, selection_calls = _run_analysis_json(
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
            candidate_package_payload=proposal,
            package_budget_reports=budget_reports,
        ),
        model=model,
        normalizer=lambda raw: _normalize_package_selection(
            raw,
            proposal=proposal,
            package_budget_reports=budget_reports,
            strategy_name=strategy_name,
            fallback_strategy_contract=discovery.get("strategy_contract"),
            fallback_outcome_contrast=discovery.get("outcome_contrast"),
        ),
        events=events,
        label="optimization_package_selection",
        is_reasoning=CROSS_MATCH_DECISION_ENABLE_REASONING,
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
                "selected_package_id": payload.get("selected_package_id"),
                "priority_problem": (payload.get("priority_problem") or {}).get("problem"),
            }
        )
    return (
        payload,
        [*proposal_errors, *selection_errors],
        proposal_calls + selection_calls,
    )


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
                result={
                    "answer": run.get("answer"),
                    "error": error,
                    "knowledge_run": dict(run),
                },
                ok=verified,
                summary=str(run.get("answer") if verified else error),
                status="complete" if verified else "failed",
            )
        )
        print(
            f"{prefix}DataAgent: knowledge {index}/{len(runs)} "
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
        f"{prefix}DataAgent: resolving {len(questions)} deterministic knowledge question(s)",
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
    match_summary_cache_path: Path | None = None,
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
    summary_cache = MatchSummaryCache(match_summary_cache_path)
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

    for game_index, record in enumerate(records, 1):
        if game_index in results:
            continue
        cached = summary_cache.get(
            record,
            strategy_name=strategy_name,
            race=race,
            model=model,
        )
        if cached is None:
            continue
        summary = cached["summary"]
        digest = evidence_digest(record, game_index)
        raw_digest = cached.get("digest")
        cached_digest_summary = (
            str(raw_digest.get("summary") or "").strip()
            if isinstance(raw_digest, dict)
            else ""
        )
        digest.summary = cached_digest_summary or (
            f"{summary.get('result') or record.result} "
            f"duration_s={summary.get('duration_s')} "
            f"events={len(summary.get('events') or [])}"
        )
        digest.raw["summary"] = digest.summary
        digest.raw["analysis"] = summary
        digest.raw["summary_input"] = {
            "format": "fixed_match_timeline_v2",
            "source": "persistent_cache",
        }
        analysis = battle_analysis_from_dict(cached["summary"])
        results[game_index] = (
            digest,
            analysis,
            True,
            [],
            [
                {
                    "action": "reuse_match_summary",
                    "source_cache": str(match_summary_cache_path or ""),
                    "cached_by": cached.get("source") or "persistent_cache",
                }
            ],
        )
        reused_paths.add(str(Path(record.file).resolve()))

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
                    result = future.result()
                    results[game_index] = result
                    digest, analysis, ok, item_errors, _item_events = result
                    if ok:
                        summary_cache.put(
                            record,
                            strategy_name=strategy_name,
                            race=race,
                            model=model,
                            summary=analysis.raw,
                            errors=item_errors,
                            source="analysis_agent",
                            digest=digest.raw,
                        )
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
    match_summary_cache_path: Path | None = None,
    prior_experiences: list[Any] | None = None,
    capability_manifest: dict[str, Any] | None = None,
    retry_feedback: list[str] | None = None,
) -> AnalysisPipelineResult:
    capability_manifest = capability_manifest or {}
    prior_experiences = list(prior_experiences or [])
    # A generation retry must carry the preceding failure into both the
    # cross-match analysis and the later optimizer.  Keep this argument
    # optional so existing callers that only supply prior_experiences remain
    # source-compatible, and avoid adding the same context twice when the
    # caller already recorded it there.
    feedback = [str(item).strip() for item in (retry_feedback or []) if str(item).strip()]
    if feedback and not any(
        isinstance(item, dict) and item.get("kind") == "generation_retry_feedback"
        for item in prior_experiences
    ):
        prior_experiences.append(
            {"kind": "generation_retry_feedback", "errors": feedback}
        )
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
        match_summary_cache_path=match_summary_cache_path,
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
