from __future__ import annotations

from typing import Any

from .loop_helpers import action_summary, improvement_from_action, sync_improvement_raw
from .config import (
    DEFAULT_OPTIMIZATION_MODEL,
    MAX_EVOL_AGENT_STEPS,
    MAX_VALIDATION_RETRIES,
    OPTIMIZATION_ENABLE_REASONING,
)
from .llm import call_json_llm
from .prompts import build_optimization_agent_prompt
from .types import BattleAnalysis, EvolImprovement, ToolObservation, ValidationResult
from ..validation import (
    normalize_improvement_knowledge_citations,
    normalize_improvement_match_references,
    validate_improvement,
    validate_improvement_metadata,
)


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
) -> tuple[ValidationResult, EvolImprovement | None, list[ToolObservation], list[str], list[dict[str, Any]]]:
    model = str(model or "").strip() or DEFAULT_OPTIMIZATION_MODEL
    observations = list(initial_tool_observations)
    validation_errors: list[str] = []
    loop_events: list[dict[str, Any]] = []
    validation_retries = 0
    candidate: dict[str, Any] | None = None
    analysis_observation_count = len(initial_tool_observations)

    print(
        f"{prefix}OptimizationAgent: optimizing {race}/{strategy_name} "
        f"(knowledge={knowledge_mode})",
        flush=True,
    )
    for step in range(1, MAX_EVOL_AGENT_STEPS + 1):
        action = call_json_llm(
            build_optimization_agent_prompt(
                strategy_name=strategy_name,
                race=race,
                battle_analysis=battle_analysis,
                skill_texts=skill_texts,
                tool_observations=observations,
                validation_errors=validation_errors,
                candidate=candidate,
                knowledge_mode=knowledge_mode,
            ),
            model=model,
            is_reasoning=OPTIMIZATION_ENABLE_REASONING,
        )
        if not action:
            validation_errors.append("Optimization Agent returned no JSON action")
            loop_events.append({"step": step, "action": "invalid", "error": validation_errors[-1]})
            continue

        name = str(action.get("action") or "")
        print(f"{prefix}OptimizationAgent: {action_summary(action)}", flush=True)
        loop_events.append({"step": step, "action": name, "raw": action})

        if name == "verify_candidate":
            validation_errors.append(
                "verify_candidate is disabled; use knowledge already provided by Analysis sub-agents"
            )
            continue

        if name in ("draft_improvement", "revise_candidate", "final_improvement"):
            improvement = improvement_from_action(action)
            changes_made = improvement.analysis.get("changes_made")
            if not isinstance(changes_made, list) or not changes_made:
                validation_errors.append("optimization analysis requires non-empty changes_made")
                continue
            influenced = improvement.analysis.get("knowledge_influenced_changes")
            change_knowledge = [
                item
                for change in changes_made
                if isinstance(change, dict)
                for item in (change.get("knowledge_used") or [])
                if str(item).strip()
            ]
            has_verified_knowledge = any(observation.ok for observation in observations)
            has_failed_knowledge = any(not observation.ok for observation in observations)
            if knowledge_mode == "enabled" and has_verified_knowledge:
                if not isinstance(influenced, list) or not any(
                    str(item).strip() for item in influenced
                ):
                    # Soft: do not block drafts missing this bookkeeping field.
                    print(
                        f"{prefix}OptimizationAgent: warning: enabled mode usually "
                        "lists knowledge_influenced_changes; continuing anyway",
                        flush=True,
                    )
            if knowledge_mode == "enabled" and not has_verified_knowledge:
                if (
                    isinstance(influenced, list)
                    and any(str(item).strip() for item in influenced)
                ) or change_knowledge:
                    print(
                        f"{prefix}OptimizationAgent: warning: knowledge citations "
                        "present without verified knowledge; continuing anyway",
                        flush=True,
                    )
            if knowledge_mode == "enabled" and has_failed_knowledge:
                unverified = improvement.analysis.get("unverified_changes")
                if not isinstance(unverified, list) or not any(
                    str(item).strip() for item in unverified
                ):
                    print(
                        f"{prefix}OptimizationAgent: warning: failed knowledge "
                        "queries usually appear in unverified_changes; continuing",
                        flush=True,
                    )
            if knowledge_mode == "disabled":
                if (
                    isinstance(influenced, list)
                    and any(str(item).strip() for item in influenced)
                ) or change_knowledge:
                    print(
                        f"{prefix}OptimizationAgent: warning: disabled mode usually "
                        "keeps knowledge fields empty; continuing anyway",
                        flush=True,
                    )
            sync_improvement_raw(improvement, [])
            candidate = improvement.raw
            finalized_targets = (
                battle_analysis.raw.get("optimization_targets")
                if isinstance(battle_analysis.raw, dict)
                else None
            )
            if not isinstance(finalized_targets, list):
                finalized_targets = battle_analysis.optimization_targets
            allowed_problem_ids = {
                str(target.get("problem_id"))
                for target in finalized_targets
                if isinstance(target, dict) and str(target.get("problem_id") or "").strip()
            }
            if not allowed_problem_ids:
                allowed_problem_ids = {
                    str(change.get("problem_id"))
                    for change in changes_made
                    if isinstance(change, dict)
                    and str(change.get("problem_id") or "").strip()
                }
            allowed_match_references = {
                str(reference).strip()
                for target in finalized_targets
                if isinstance(target, dict)
                for reference in (target.get("match_evidence") or [])
                if str(reference).strip()
            }
            fix_notes = normalize_improvement_match_references(
                improvement.analysis,
                allowed_match_references,
            )
            if fix_notes:
                print(
                    f"{prefix}OptimizationAgent: auto-normalized "
                    f"{len(fix_notes)} approximate match citation(s)",
                    flush=True,
                )
            verified_knowledge_indices = {
                index
                for index, observation in enumerate(observations, 1)
                if (
                    observation.status == "complete"
                    or (not observation.status and observation.ok)
                )
            }
            knowledge_references = {
                index: (
                    str(observation.summary or "").strip()
                    or str((observation.result or {}).get("answer") or "").strip()
                    or f"Q{index}"
                )
                for index, observation in enumerate(observations, 1)
                if index in verified_knowledge_indices
            }
            knowledge_fix_notes = normalize_improvement_knowledge_citations(
                improvement.analysis,
                verified_knowledge_indices,
                knowledge_references=knowledge_references,
            )
            if knowledge_fix_notes:
                print(
                    f"{prefix}OptimizationAgent: auto-normalized "
                    f"{len(knowledge_fix_notes)} knowledge citation(s)",
                    flush=True,
                )
            metadata_error = validate_improvement_metadata(
                analysis=improvement.analysis,
                files=improvement.files,
                current_strategy=str(skill_texts.get("strategy.md") or ""),
                allowed_problem_ids=allowed_problem_ids,
                verified_knowledge_available=has_verified_knowledge,
                allowed_match_references=allowed_match_references,
                verified_knowledge_indices=verified_knowledge_indices,
            )
            if metadata_error:
                validation_errors.append(metadata_error)
                validation_retries += 1
                if validation_retries > MAX_VALIDATION_RETRIES:
                    return (
                        ValidationResult(ok=False, error=metadata_error),
                        improvement,
                        observations,
                        validation_errors,
                        loop_events,
                    )
                continue
            result = validate_improvement(
                files=improvement.files,
                race=battle_analysis.race,
            )
            if result.ok:
                improvement.files = result.files or improvement.files
                sync_improvement_raw(improvement, [])
                loop_events.append(
                    {
                        "step": step,
                        "action": "completed",
                        "analysis_observation_count": analysis_observation_count,
                    }
                )
                return result, improvement, observations, validation_errors, loop_events
            validation_errors.append(result.error)
            validation_retries += 1
            if validation_retries > MAX_VALIDATION_RETRIES:
                print(
                    f"{prefix}OptimizationAgent: validation failed after "
                    f"{MAX_VALIDATION_RETRIES} revise retries: {result.error}",
                    flush=True,
                )
                return result, improvement, observations, validation_errors, loop_events
            print(
                f"{prefix}OptimizationAgent: validation failed "
                f"(revise retry {validation_retries}/{MAX_VALIDATION_RETRIES}): "
                f"{result.error}",
                flush=True,
            )
            continue

        validation_errors.append(
            "Optimization Agent action must be draft_improvement or revise_candidate"
        )

    return ValidationResult(ok=False, error="Optimization Agent exhausted"), None, observations, validation_errors, loop_events
