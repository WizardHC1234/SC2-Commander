from __future__ import annotations

import json
from typing import Any

from .context import render_knowledge_results
from .optimization_policy import OPTIMIZATION_POLICY
from .types import BattleAnalysis, ToolObservation


def _compact_optimization_brief(decision: dict[str, Any]) -> dict[str, Any]:
    plan = decision.get("plan") if isinstance(decision.get("plan"), dict) else {}
    failure = (
        decision.get("failure_mode_analysis")
        if isinstance(decision.get("failure_mode_analysis"), dict)
        else {}
    )
    return {
        "strategy_core": dict(decision.get("strategy_contract") or {}),
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
            "program_preflight": dict(
                decision.get("selected_package_budget") or {}
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
    del knowledge_mode, tool_observations
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
The previous candidate needs one correction pass. Regenerate the whole strategy.md, keep its evidence-supported idea, fix the concrete issues below, and return the complete strategy again.

Issues or timing feedback:
{chr(10).join(f'- {item}' for item in validation_errors)}

Previous candidate:
{previous_candidate or 'Unavailable.'}
"""

    action = "revise_candidate" if validation_errors else "draft_candidate"
    return f"""You are EvolAgent's Strategy Optimizer for StarCraft II. Analyze the supplied compact evidence and Generate the entire replacement strategy.md. You may coordinate changes across economy, production, technology, army composition, attack timing, reinforcement, retreat, and cleanup when they support the same causal improvement. Inspect every part of the strategy, but preserve unrelated behavior and do not fill audit forms. Do not select among candidate plans; implement the evidence-supported optimization brief as one coherent strategy.

{OPTIMIZATION_POLICY}

Important implementation guidance:
- Preserve the Champion's successful behavior unless the evidence brief explicitly supports revising it.
- Implement the selected optimization package rather than combining it with rejected packages. Correct an internally inconsistent wording detail when necessary, but preserve the selected hypothesis, coordinated changes, and program-calculated time budget.
- Units and production facilities should have practical staged targets when the candidate materially changes their production. Do not invent targets for unaffected units or phases.
- Keep every explicit final army feasible at no more than 200 supply, counting workers plus every final combat and support unit.
- A support unit or upgrade is part of the first-attack gate only when the candidate intentionally makes it a mandatory prerequisite. Otherwise it may join as available reinforcement.
- Keep attack timing, own and enemy composition at contact, production opportunity cost, post-contact reinforcement, and the 1800-second match limit connected to the final win plan.
- The selected package's earliest-feasible preflight and latest useful commitment bound are a budget for the generated strategy. Do not silently add first-attack prerequisites that push the actual generated strategy beyond that budget.
- If the evidence identifies a runtime execution failure rather than a strategy defect, do not disguise it as another strategy threshold change.
- Use only supported runtime controls. The strategy may specify high-level macro targets and army objectives, but not coordinates, unit tags, manual abilities, formation, transport loading, or unit-level micro.
- Return concise strategy text. Prefer one observable instruction over repeated warnings and narrow exceptions.

{revision_context}

Runtime contract:
{RUNTIME_CONTRACT}

Strategy: {strategy_name}
Race: {race}

Compact optimization brief:
{json.dumps(brief, ensure_ascii=False, indent=2)}

Verified deterministic SC2 knowledge:
{render_knowledge_results(knowledge_runs or [])}

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
