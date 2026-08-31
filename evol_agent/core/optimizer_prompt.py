from __future__ import annotations

import json
from typing import Any

from .context import render_knowledge_results
from .types import BattleAnalysis, ToolObservation


def _compact_program_preflight(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value.get(key)
        for key in (
            "id",
            "status",
            "parent_earliest_feasible_time_seconds",
            "candidate_earliest_feasible_time_seconds",
            "earliest_feasible_timing_delta_seconds",
            "target_latest_first_commitment_seconds",
            "maximum_added_feasibility_seconds",
            "target_latest_satisfied",
            "maximum_added_delay_satisfied",
            "gate_cost_delta",
            "parent_support_aware_combat_estimate",
            "candidate_support_aware_combat_estimate",
            "support_aware_combat_delta",
            "bottlenecks",
            "warnings",
            "errors",
            "knowledge_status",
        )
        if value.get(key) not in (None, "", [], {})
    }


def _compact_optimization_brief(decision: dict[str, Any]) -> dict[str, Any]:
    plan = decision.get("plan") if isinstance(decision.get("plan"), dict) else {}
    failure = (
        decision.get("failure_mode_analysis")
        if isinstance(decision.get("failure_mode_analysis"), dict)
        else {}
    )
    return {
        "strategy_core": dict(decision.get("strategy_contract") or {}),
        "champion_lineage": dict(decision.get("inheritance") or {}),
        "strengths_to_preserve": list(decision.get("strengths_to_preserve") or []),
        "win_loss_contrast": dict(decision.get("outcome_contrast") or {}),
        "priority_problem": dict(decision.get("priority_problem") or {}),
        "hypothesis": str(decision.get("hypothesis") or ""),
        "failure_stage": str(failure.get("failure_stage") or ""),
        "contact_and_combat": {
            key: failure.get(key)
            for key in (
                "gate_attainment_and_launch",
                "commitment_and_contact_timing",
                "own_package_at_contact",
                "opponent_package_and_growth",
                "post_contact_continuity",
                "production_feasibility",
                "optimization_implication",
            )
            if failure.get(key)
        },
        "intervention": {
            "selected_package_id": str(decision.get("selected_package_id") or ""),
            "direction": str(plan.get("direction") or ""),
            "material_behavior_change": str(
                plan.get("material_behavior_change") or plan.get("direction") or ""
            ),
            "coordinated_changes": list(plan.get("coordinated_changes") or []),
            "preserve": list(plan.get("preserve") or []),
            "contact_window_effect": str(plan.get("contact_window_effect") or "unknown"),
            "new_hard_prerequisites": list(plan.get("new_hard_prerequisites") or []),
            "production_tradeoffs": list(plan.get("production_tradeoffs") or []),
            "declared_time_budget": dict(
                decision.get("selected_timing_budget") or {}
            ),
            "program_preflight": _compact_program_preflight(
                decision.get("selected_package_budget")
            ),
            "engagement_assessment": dict(
                decision.get("selected_engagement_assessment") or {}
            ),
            "data_agent_assessment": dict(
                decision.get("data_agent_assessment") or {}
            ),
            "history_assessment": dict(
                decision.get("selected_history_assessment") or {}
            ),
        },
        "evidence_limits": list(decision.get("evidence_limits") or []),
    }


def build_candidate_prompt(
    *,
    strategy_name: str,
    race: str,
    battle_analysis: BattleAnalysis,
    skill_texts: dict[str, str],
    tool_observations: list[ToolObservation],
    validation_errors: list[str],
    candidate: dict | None,
    knowledge_mode: str,
    capability_manifest: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    knowledge_runs: list[dict[str, Any]] | None = None,
) -> str:
    """Generate one coherent complete strategy from a compact evidence brief."""
    del tool_observations
    from .prompts import RUNTIME_CONTRACT

    parent_text = str(skill_texts.get("strategy.md") or "")
    decision_payload = decision or dict(battle_analysis.raw or {})
    brief = _compact_optimization_brief(decision_payload)
    previous_candidate = ""
    if isinstance(candidate, dict):
        previous_candidate = str(candidate.get("strategy_md") or "").strip()
        if not previous_candidate and isinstance(candidate.get("files"), dict):
            previous_candidate = str(candidate["files"].get("strategy.md") or "").strip()

    revision_context = ""
    if validation_errors:
        revision_context = f"""
The previous candidate needs a correction. Regenerate the whole strategy.md, preserve the selected intervention and every already-valid paragraph, and fix only the concrete issues below. Do not reopen hypothesis selection or add a compensating second mechanism.

Issues or timing feedback:
{chr(10).join(f'- {item}' for item in validation_errors)}

Previous candidate:
{previous_candidate or 'Unavailable.'}
"""

    action = "revise_candidate" if validation_errors else "draft_candidate"
    knowledge_text = (
        render_knowledge_results(knowledge_runs or [])
        if knowledge_mode == "enabled"
        else (
            "Disabled for the model-only ablation. Use only the supplied strategy, "
            "trajectory analysis, experiment history, and your own internal knowledge."
        )
    )
    return f"""You are EvolAgent's Strategy Optimizer for StarCraft II. The analysis and package-selection stages are complete. Generate the entire replacement strategy.md as one coherent strategy by implementing only the selected optimization brief. Do not select another hypothesis, repeat the match analysis, combine rejected packages, or fill audit forms.

Rewrite procedure:
1. Start from the Current Champion. Treat champion_lineage and strengths_to_preserve as winning evidence, and copy unrelated sections without redesigning them.
2. Implement only the selected material behavior change and its necessary dependencies. Preserve cited gains, make a reversal genuinely reverse the failed behavior, and make a material repair address only its named missing dependency.
3. Review the complete strategy chronologically across economy, production, technology, composition, attack timing, reinforcement, and recovery. Treat program_preflight earliest_feasible_time_seconds as a lower bound, keep the selected useful opponent window, and ensure post-commitment production continues without displacing a preserved priority.
4. If the intervention changes the first attack, state one observable gate and attack when it is met. Do not copy its numerical threshold into recovery, and do not interrupt a favorable ongoing attack merely to rebuild the opening gate.
5. Give every newly introduced or newly resumed unit or production facility a practical stage target and a final cap when production continues. Do not add targets for unaffected units. Keep the explicit final composition at no more than 200 supply including workers and support units.
6. Use only supported high-level macro and army objectives. Do not require hidden state, coordinates, unit tags, fixed-composition detachments, or scripted group splitting and merging.
7. Keep the strategy concise and internally consistent. Do not add a second optimization mechanism, duplicate warnings, or prose that behaves like a detailed state machine.

{revision_context}

Runtime contract:
{RUNTIME_CONTRACT}

Strategy: {strategy_name}
Race: {race}

Compact optimization brief:
{json.dumps(brief, ensure_ascii=False, indent=2)}

Verified deterministic SC2 knowledge:
{knowledge_text}

Executor capability manifest:
{json.dumps(capability_manifest or {}, ensure_ascii=False, separators=(',', ':'))}

Current Champion strategy.md:
{parent_text}

Return one JSON object only:
{{
  "action":"{action}",
  "strategy_md":"complete Markdown document containing # Summary and # Details in that order",
  "changes_made":[{{"problem":"evidence-supported problem","change":"coherent strategy correction","evidence":"brief match evidence"}}],
  "preserved_strengths":["successful behavior retained from the Champion"],
  "expected_effect":"expected change in contact, combat, continuation, or match outcome",
  "main_risk":"most important regression risk to evaluate"
}}
"""


__all__ = ["build_candidate_prompt"]
