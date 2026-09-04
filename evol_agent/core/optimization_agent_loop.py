from __future__ import annotations

import importlib
import re
from typing import Any

from .config import (
    CANDIDATE_GENERATION_ENABLE_REASONING,
    DEFAULT_OPTIMIZATION_MODEL,
    MAX_VALIDATION_RETRIES,
)
from .llm import call_json_llm
from .prompts import build_candidate_prompt
from .strategy_patch_validator import (
    validate_strategy_patch_semantics,
    validate_strategy_patch_structure,
)
from .types import BattleAnalysis, EvolImprovement, ToolObservation, ValidationResult
from ..optimization.strategy_document import StrategyDocument, paragraph_hash
from ..sc2_data_agent.bridge import find_knowledge_run_error, run_knowledge_query
from ..validation import validate_improvement


def _unwrap_candidate(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    raw = result.get("candidate") if isinstance(result.get("candidate"), dict) else result
    return raw if isinstance(raw, dict) else None


def _compact_contact_evidence(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep only trajectory rows needed to compare alternative contact windows."""
    packet = raw.get("retrieval_evidence")
    match_packet = packet.get("match_record_evidence") if isinstance(packet, dict) else {}
    rows: list[dict[str, Any]] = []
    for query in (match_packet or {}).get("queries") or []:
        if not isinstance(query, dict):
            continue
        for result in query.get("results") or []:
            if not isinstance(result, dict):
                continue
            game_index = result.get("game_index")
            interaction = result.get("interaction_check") or {}
            for timeline_row in result.get("timeline_rows") or []:
                if not isinstance(timeline_row, dict):
                    continue
                army = timeline_row.get("army") or []
                enemy = timeline_row.get("enemy") or []
                truth = timeline_row.get("opponent_truth_after_match") or []
                groups = []
                for group in timeline_row.get("groups") or []:
                    if not isinstance(group, list):
                        continue
                    groups.append(
                        {
                            "role": group[1] if len(group) > 1 else None,
                            "count": group[2] if len(group) > 2 else None,
                            "power": group[3] if len(group) > 3 else None,
                            "zone": group[4] if len(group) > 4 else None,
                            "composition": group[5] if len(group) > 5 else None,
                            "near_enemy_count": group[6] if len(group) > 6 else None,
                            "near_enemy_power": group[7] if len(group) > 7 else None,
                            "near_enemy_composition": group[8] if len(group) > 8 else None,
                            "mode": group[11] if len(group) > 11 else None,
                            "retreat_ratio": group[12] if len(group) > 12 else None,
                            "command_age": group[13] if len(group) > 13 else None,
                            "command_source": group[14] if len(group) > 14 else None,
                            "objective_status": group[16] if len(group) > 16 else None,
                        }
                    )
                rows.append(
                    {
                        "game_index": game_index,
                        "time_s": timeline_row.get("time_s"),
                        "trigger": timeline_row.get("trigger"),
                        "own": {
                            "army_supply": army[0] if len(army) > 0 else None,
                            "army_power": army[1] if len(army) > 1 else None,
                            "composition": army[2] if len(army) > 2 else None,
                            "training": army[3] if len(army) > 3 else None,
                        },
                        "known_enemy": {
                            "combat_composition": enemy[3] if len(enemy) > 3 else None,
                            "known_types": enemy[8] if len(enemy) > 8 else None,
                        },
                        "enemy_truth": {
                            "supply": truth[2] if len(truth) > 2 else None,
                            "army_units": truth[4] if len(truth) > 4 else None,
                            "structures": truth[5] if len(truth) > 5 else None,
                            "upgrades": truth[7] if len(truth) > 7 else None,
                        },
                        "combat": timeline_row.get("combat"),
                        "groups": groups,
                        "interaction": interaction.get("classification"),
                    }
                )
                if len(rows) >= 16:
                    return rows
    return rows


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
        "strategy_contract": (
            dict(raw.get("strategy_contract"))
            if isinstance(raw.get("strategy_contract"), dict)
            else {}
        ),
        "strengths_to_preserve": strengths,
        "wins_to_preserve": list(raw.get("wins_to_preserve") or strengths),
        "winning_mechanism": str(raw.get("winning_mechanism") or "").strip(),
        "cross_outcome_comparison": list(
            raw.get("cross_outcome_comparison") or []
        ),
        "outcome_contrast": (
            dict(raw.get("outcome_contrast"))
            if isinstance(raw.get("outcome_contrast"), dict)
            else {}
        ),
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
        "selected_package_id": str(raw.get("selected_package_id") or ""),
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
        "selected_engagement_assessment": (
            dict(raw.get("selected_engagement_assessment"))
            if isinstance(raw.get("selected_engagement_assessment"), dict)
            else {}
        ),
        "data_agent_assessment": (
            dict(raw.get("data_agent_assessment"))
            if isinstance(raw.get("data_agent_assessment"), dict)
            else {}
        ),
        "selected_history_assessment": (
            dict(raw.get("selected_history_assessment"))
            if isinstance(raw.get("selected_history_assessment"), dict)
            else {}
        ),
        "direction_audit": (
            dict(raw.get("direction_audit"))
            if isinstance(raw.get("direction_audit"), dict)
            else {}
        ),
        "package_budget_reports": [
            dict(item)
            for item in (raw.get("package_budget_reports") or [])
            if isinstance(item, dict)
        ],
        "retrieval_assessment": (
            dict(raw.get("retrieval_assessment"))
            if isinstance(raw.get("retrieval_assessment"), dict)
            else {}
        ),
        "plan": plan,
        "next_action": str(raw.get("next_action") or ""),
        "knowledge_used": knowledge_used,
        "contact_evidence": _compact_contact_evidence(raw),
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
    parent_text: str = "",
    race: str,
    capability_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    """Build deterministic facts for every executable action named by a candidate."""
    comparison_text = f"{parent_text}\n{candidate_text}"
    macro_contract = capability_manifest.get("macro_contract")
    available = (
        macro_contract.get("available_actions")
        if isinstance(macro_contract, dict)
        else []
    )
    try:
        action_module = importlib.import_module(f"commander.races.{race.casefold()}.actions")
        action_specs = dict(getattr(action_module, "ACTION_SPECS", {}) or {})
    except (ImportError, AttributeError, TypeError, ValueError):
        action_specs = {}

    def normalized_words(value: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", str(value or "").casefold())
        normalized: list[str] = []
        for word in words:
            if len(word) > 4 and word.endswith("ies"):
                word = word[:-3] + "y"
            elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
                word = word[:-1]
            normalized.append(word)
        return normalized

    normalized_text = " ".join(normalized_words(comparison_text))

    def phrase_is_named(value: str) -> bool:
        phrase = " ".join(normalized_words(value))
        return bool(phrase and re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", normalized_text))

    def description_aliases(action: str) -> list[str]:
        spec = action_specs.get(action)
        description = str(getattr(spec, "description", "") or "")
        if not description:
            return []
        first_sentence = description.split(".", 1)[0]
        aliases = re.findall(r"\(([^)]+)\)", first_sentence)
        head = re.sub(r"\([^)]*\)", " ", first_sentence)
        head = re.sub(r"^(?:absolute|train|build|research|morph)\s+", "", head, flags=re.I)
        head = re.split(
            r"\b(?:count|from|for|to|at|with|requires|using|into)\b",
            head,
            maxsplit=1,
            flags=re.I,
        )[0].strip(" :-")
        if head:
            aliases.append(head)
        return aliases

    def action_is_named(action: str) -> bool:
        raw = str(action or "").strip()
        if not raw:
            return False
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(raw)}(?![A-Za-z0-9_])",
            comparison_text,
            re.IGNORECASE,
        ):
            return True
        target = re.sub(r"^(train|build|research|morph)_?", "", raw).replace(
            "techlab", "tech_lab"
        )
        # Strategy files use natural names and irregular plurals, while some
        # executor ids use internal SC2 names (Combat Shield -> shieldwall).
        if phrase_is_named(target.replace("_", " ")):
            return True
        if any(phrase_is_named(alias) for alias in description_aliases(raw)):
            return True
        if raw == "build_gas" and re.search(r"\b(?:gas|refiner(?:y|ies))\b", comparison_text, re.I):
            return True
        return False

    selected = {str(action) for action in (available or []) if action_is_named(str(action))}
    # Timing a named unit also needs verified facts for its complete structural
    # dependency chain. Do not require every dependency to be repeated in prose.
    pending = list(selected)
    while pending:
        action = pending.pop()
        spec = action_specs.get(action)
        for dependency in getattr(spec, "dependencies", ()) or ():
            dependency = str(dependency)
            if dependency in available and dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    actions = [str(action) for action in (available or []) if str(action) in selected]
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


def _document_changes(
    parent: StrategyDocument,
    candidate: StrategyDocument,
) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    if parent.summary != candidate.summary:
        changes.append(
            {
                "op": "replace_summary",
                "target": "summary",
                "old": parent.summary,
                "new": candidate.summary,
            }
        )
    parent_by_id = {item.id: item for item in parent.details}
    candidate_by_id = {item.id: item for item in candidate.details}
    for item in parent.details:
        revised = candidate_by_id.get(item.id)
        if revised is None:
            changes.append(
                {
                    "op": "remove_detail",
                    "target": item.id,
                    "old": item.value,
                    "new": "",
                }
            )
        elif revised.title != item.title or revised.value != item.value:
            changes.append(
                {
                    "op": "replace_detail",
                    "target": item.id,
                    "old": item.value,
                    "new": revised.value,
                }
            )
    for item in candidate.details:
        if item.id not in parent_by_id:
            changes.append(
                {
                    "op": "add_detail",
                    "target": item.id,
                    "old": "",
                    "new": item.value,
                }
            )
    if [item.id for item in parent.details] != [item.id for item in candidate.details]:
        changes.append(
            {
                "op": "reorder_details",
                "target": "details",
                "old": ",".join(item.id for item in parent.details),
                "new": ",".join(item.id for item in candidate.details),
            }
        )
    return changes


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

    normalized_patches: list[dict[str, str]] = []
    strategy_md = str(raw.get("strategy_md") or "").strip()
    document_changes: list[dict[str, str]] = []
    if strategy_md:
        try:
            candidate_document = StrategyDocument.parse(strategy_md)
        except ValueError as exc:
            return None, f"generated strategy.md is invalid: {exc}"
        strategy_md = candidate_document.render()
        document_changes = _document_changes(parent_document, candidate_document)
        if not document_changes:
            return None, "generated strategy.md is unchanged from the Champion"
        parent_by_id = {item.id: item for item in parent_document.details}
        for item in candidate_document.details:
            current = parent_by_id.get(item.id)
            if current is None or current.value == item.value:
                continue
            normalized_patches.append(
                {
                    "target": item.id,
                    "expected_old_hash": paragraph_hash(current.value),
                    "replacement": item.value,
                    "why_required": "changed by full-document strategy generation",
                }
            )
    else:
        # Legacy checkpoint compatibility. Live prompts request strategy_md, but
        # previously saved optimizer responses may still contain paragraph patches.
        patches = raw.get("patches")
        if not isinstance(patches, list) or not patches:
            return None, "optimizer must return a complete strategy_md document"
        detail_ids = {item.id for item in parent_document.details}
        seen: set[str] = set()
        for item in patches:
            if not isinstance(item, dict):
                return None, "each patch must be an object"
            target = str(item.get("target") or "").strip()
            replacement = str(item.get("replacement") or item.get("value") or "").strip()
            why_required = str(item.get("why_required") or "").strip()
            expected_old_hash = str(item.get("expected_old_hash") or "").strip()
            if target in {"", "summary"} or str(item.get("op") or "") == "replace_summary":
                return None, "legacy patch may not modify # Summary"
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
            "strategy_md": strategy_md,
            "document_changes": document_changes,
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
    """Evaluate a retry-exhausted candidate only for non-blocking review notes.

    Structural, executable, knowledge, and blocking semantic errors mean that the
    generated document is not a valid test of the selected hypothesis.  The outer
    evolution loop can retain the Champion and continue the configured generation
    budget instead of spending matches on that invalid candidate.
    """
    if failure_stage != "semantic" or not errors:
        return False
    normalized = [str(error).strip().lower() for error in errors if str(error).strip()]
    if not normalized:
        return False
    if all(error.startswith("non-blocking") for error in normalized):
        return True
    # The exact-document duplicate guard in the outer evolution runner still
    # prevents replaying an identical strategy.  After every semantic retry has
    # been used, a basic-valid, non-identical candidate may be evaluated even if
    # the model-only history judge still considers its causal family similar.
    # Match outcomes are the final evidence and this avoids an endless analysis
    # loop caused solely by an over-broad semantic equivalence verdict.
    return all("mechanism history" in error for error in normalized)


def _verified_inheritance(
    prior_experiences: list[Any] | None,
    *,
    current_strategy: str = "",
) -> dict[str, Any]:
    """Collect score-improving changes on the current Champion ancestry only."""
    experiments = [
        item
        for item in (prior_experiences or [])
        if isinstance(item, dict)
        and str(item.get("experiment_id") or "").strip()
    ]
    by_candidate = {
        str(item.get("candidate") or "").strip(): item
        for item in experiments
        if str(item.get("candidate") or "").strip()
        and str(item.get("decision") or "").strip().lower() == "accepted"
    }
    lineage_ids: set[str] = set()
    cursor = str(current_strategy or "").strip()
    restrict_to_lineage = bool(cursor)
    visited: set[str] = set()
    while cursor and cursor not in visited:
        visited.add(cursor)
        experiment = by_candidate.get(cursor)
        if experiment is None:
            break
        lineage_ids.add(str(experiment.get("experiment_id") or "").strip())
        cursor = str(
            experiment.get("mutation_parent") or experiment.get("parent") or ""
        ).strip()

    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in experiments:
        experiment_id = str(item.get("experiment_id") or "").strip()
        if restrict_to_lineage and experiment_id not in lineage_ids:
            continue
        ledger = item.get("inheritance")
        if isinstance(ledger, dict):
            for change in ledger.get("verified_changes") or []:
                if not isinstance(change, dict):
                    continue
                key = str(change.get("experiment_id") or change.get("change") or "").strip()
                if key and key not in seen:
                    verified.append(dict(change))
                    seen.add(key)
        try:
            score_delta = float(item.get("score_delta"))
        except (TypeError, ValueError):
            score_delta = 0.0
        implementation = str(
            item.get("implementation_verdict") or "unknown"
        ).strip().lower()
        if (
            str(item.get("decision") or "").strip().lower() == "accepted"
            and score_delta > 0.0
            and implementation == "implemented"
        ):
            key = experiment_id or str(item.get("primary_change") or "").strip()
            if key and key not in seen:
                verified.append(
                    {
                        "experiment_id": experiment_id,
                        "difficulty": str(item.get("difficulty") or ""),
                        "mechanism_family": str(item.get("mechanism_family") or ""),
                        "change": str(
                            item.get("primary_change")
                            or item.get("plan_direction")
                            or item.get("hypothesis")
                            or ""
                        ),
                        "evidence": str(item.get("lesson") or ""),
                        "score_delta": score_delta,
                    }
                )
                seen.add(key)
    if not verified:
        return {}
    return {
        "verified_changes": verified[-12:],
        "preservation_rule": (
            "Preserve each trajectory-realized Champion improvement unless current "
            "cross-match evidence directly supports revising it."
        ),
    }


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
    changes = list(candidate.get("document_changes") or candidate.get("patches") or [])
    return {
        "strategy_contract": dict(decision.get("strategy_contract") or {}),
        "outcome_contrast": dict(decision.get("outcome_contrast") or {}),
        "strengths_to_preserve": list(decision.get("strengths_to_preserve") or []),
        "inheritance": dict(decision.get("inheritance") or {}),
        "priority_problem": (
            dict(priority) if isinstance(priority, dict) else {"problem": problem}
        ),
        "hypothesis": hypothesis,
        "mechanism_family": str(decision.get("mechanism_family") or direction).strip(),
        "failure_mode_analysis": (
            dict(decision.get("failure_mode_analysis"))
            if isinstance(decision.get("failure_mode_analysis"), dict)
            else {}
        ),
        "mechanism_prediction": mechanism_prediction,
        "selected_package_id": str(decision.get("selected_package_id") or ""),
        "selected_timing_budget": dict(decision.get("selected_timing_budget") or {}),
        "selected_package_budget": dict(decision.get("selected_package_budget") or {}),
        "selected_engagement_assessment": dict(
            decision.get("selected_engagement_assessment") or {}
        ),
        "data_agent_assessment": dict(
            decision.get("data_agent_assessment") or {}
        ),
        "selected_history_assessment": dict(
            decision.get("selected_history_assessment") or {}
        ),
        "direction_audit": dict(decision.get("direction_audit") or {}),
        "candidate_package_evaluations": [
            dict(item)
            for item in (decision.get("package_budget_reports") or [])
            if isinstance(item, dict)
        ],
        "plan_direction": direction,
        "intervention_package": dict(plan),
        "preserved_strengths": list(candidate.get("preserved_strengths") or strengths),
        "document_changes": [item for item in changes if isinstance(item, dict)],
        "primary_change": direction,
        "expected_effect": expected_effect,
        "main_risk": main_risk,
        "patches": [
            {
                "target": item.get("target"),
                "why_required": item.get("why_required")
                or "changed in the generated complete strategy",
            }
            for item in changes
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
    prior_experiences: list[Any] | None = None,
) -> tuple[
    ValidationResult,
    EvolImprovement | None,
    list[ToolObservation],
    list[str],
    list[dict[str, Any]],
]:
    """Implement one Cross-match hypothesis as a complete strategy.md document."""
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
    timing_feedback_sent = False
    parent_text = str(skill_texts.get("strategy.md") or "")
    try:
        parent_document = StrategyDocument.parse(parent_text)
    except ValueError as exc:
        error = f"parent strategy.md cannot be patched: {exc}"
        return ValidationResult(ok=False, error=error), None, observations, [error], events

    decision = extract_final_cross_match_decision(battle_analysis)
    inheritance = _verified_inheritance(
        prior_experiences,
        current_strategy=strategy_name,
    )
    if inheritance:
        decision = {**decision, "inheritance": inheritance}
    if not decision["hypothesis"] or not str((decision.get("plan") or {}).get("direction") or "").strip():
        error = "Optimizer requires a Cross-match hypothesis and plan.direction"
        return ValidationResult(ok=False, error=error), None, observations, [error], events

    base_knowledge_runs = (
        _knowledge_runs_for_optimizer(decision, observations)
        if knowledge_mode == "enabled"
        else []
    )
    knowledge_runs = list(base_knowledge_runs)
    print(
        f"{prefix}OptimizationAgent: generating complete strategy.md for "
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
            is_reasoning=CANDIDATE_GENERATION_ENABLE_REASONING,
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

        ignored_unchanged: list[str] = []
        if not normalized.get("strategy_md"):
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
        if not normalized.get("strategy_md") and not normalized["patches"]:
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

        rationale = _candidate_rationale(decision=decision, candidate=normalized)
        operations: list[dict[str, str]] = []
        if normalized.get("strategy_md"):
            patched_text = str(normalized["strategy_md"])
            paragraph_changes = list(normalized.get("document_changes") or [])
        else:
            operations = _patches_to_operations(normalized["patches"])
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

        structure_errors = []
        if not normalized.get("strategy_md"):
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
        knowledge_runs = [
            run
            for run in base_knowledge_runs
            if str(run.get("question_id") or "") != "QCANDIDATE"
        ]
        candidate_knowledge = None
        if knowledge_mode == "enabled":
            print(
                f"{prefix}DataAgent: validating candidate action dependencies",
                flush=True,
            )
            candidate_knowledge = _candidate_knowledge_run(
                candidate_text=patched_text,
                parent_text=parent_text,
                race=race,
                capability_manifest=capability_manifest,
            )
        elif attempt == 1:
            print(
                f"{prefix}Model-only ablation: skipped DataAgent candidate validation",
                flush=True,
            )
            events.append(
                {
                    "attempt": attempt,
                    "action": "skip_external_candidate_validation",
                    "valid": True,
                    "llm_calls": llm_calls,
                }
            )
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
        semantic_audit: dict[str, Any] = {}
        semantic_errors = (
            validate_strategy_patch_semantics(
                decision=decision,
                parent_text=parent_text,
                candidate_text=patched_text,
                patches=normalized["patches"],
                capability_manifest=capability_manifest,
                knowledge_runs=knowledge_runs,
                inheritance=inheritance,
                prior_experiences=prior_experiences,
                audit_output=semantic_audit,
                race=race,
                model=model,
            )
            if knowledge_mode == "enabled"
            else []
        )
        if semantic_audit:
            draft_improvement.raw["deterministic_feasibility_audit"] = semantic_audit
            draft_improvement.analysis["deterministic_feasibility_audit"] = semantic_audit
        timing_report = (
            semantic_audit.get("contact_timing_report")
            if isinstance(semantic_audit, dict)
            and isinstance(semantic_audit.get("contact_timing_report"), dict)
            else {}
        )
        timing_delta = timing_report.get("earliest_feasible_timing_delta_seconds")
        plan = decision.get("plan") if isinstance(decision.get("plan"), dict) else {}
        contact_effect = str(plan.get("contact_window_effect") or "unknown").strip().lower()
        timing_justification = str(plan.get("why_window_remains_favorable") or "").strip()
        selected_budget = (
            decision.get("selected_timing_budget")
            if isinstance(decision.get("selected_timing_budget"), dict)
            else {}
        )
        target_latest = selected_budget.get("target_latest_first_commitment_seconds")
        maximum_added = selected_budget.get("maximum_added_feasibility_seconds")
        actual_candidate_time = timing_report.get(
            "candidate_earliest_feasible_time_seconds"
        )
        budget_violations: list[str] = []
        if timing_report.get("complete") is True:
            if (
                isinstance(target_latest, (int, float))
                and isinstance(actual_candidate_time, (int, float))
                and float(actual_candidate_time) > float(target_latest)
            ):
                budget_violations.append(
                    f"earliest feasible commitment {float(actual_candidate_time):.1f}s "
                    f"exceeds the selected package limit {float(target_latest):.1f}s"
                )
            if (
                isinstance(maximum_added, (int, float))
                and isinstance(timing_delta, (int, float))
                and float(timing_delta) > float(maximum_added)
            ):
                budget_violations.append(
                    f"added feasibility delay {float(timing_delta):.1f}s exceeds the "
                    f"selected package allowance {float(maximum_added):.1f}s"
                )
        if not timing_feedback_sent and budget_violations:
            timing_feedback_sent = True
            feedback = (
                "Program-calculated strategy timing violates the selected optimization "
                "package budget: "
                + "; ".join(budget_violations)
                + ". Regenerate the complete strategy without adding first-commitment "
                "requirements outside the selected package."
            )
            candidate = payload
            validation_errors.append(feedback)
            prompt_errors = [feedback]
            events.append(
                {
                    "attempt": attempt,
                    "action": "selected_package_budget_feedback",
                    "valid": False,
                    "error": feedback,
                    "llm_calls": llm_calls,
                }
            )
            print(
                f"{prefix}OptimizationAgent: package-budget feedback; revising once: "
                f"{feedback}",
                flush=True,
            )
            continue
        if (
            not timing_feedback_sent
            and timing_report.get("complete") is True
            and isinstance(timing_delta, (int, float))
            and float(timing_delta) > 30.0
            and contact_effect != "later"
            and not timing_justification
        ):
            timing_feedback_sent = True
            feedback = (
                "Program-calculated earliest feasible first commitment is "
                f"{float(timing_delta):.1f}s later than the Champion. Revise the complete "
                "strategy once to recover the intended contact window, or explicitly make "
                "the later package's combat advantage and survival tradeoff clear."
            )
            candidate = payload
            validation_errors.append(feedback)
            prompt_errors = [feedback]
            events.append(
                {
                    "attempt": attempt,
                    "action": "contact_timing_feedback",
                    "valid": False,
                    "error": feedback,
                    "llm_calls": llm_calls,
                }
            )
            print(
                f"{prefix}OptimizationAgent: timing feedback; revising once: {feedback}",
                flush=True,
            )
            continue
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
                    f"{prefix}OptimizationAgent: candidate semantics failed; "
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
            "using the latest generated candidate and continuing to match evaluation",
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
