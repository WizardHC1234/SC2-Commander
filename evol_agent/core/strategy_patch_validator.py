from __future__ import annotations

import json
import re
from typing import Any

from .config import (
    CONTACT_TIMING_EXTRACTION_ENABLE_REASONING,
    DEFAULT_OPTIMIZATION_MODEL,
    MECHANISM_HISTORY_ENABLE_REASONING,
    STRATEGY_SEMANTIC_VALIDATION_ENABLE_REASONING,
)
from .context import json_compact_block, render_knowledge_results, render_optimizer_decision
from .llm import call_json_llm
from .optimization_policy import HARD_VALIDATION_POLICY
from .terran_build_order_simulator import simulate_terran_first_commitment
from ..optimization.strategy_document import StrategyDocument


def _timing_audit_required(decision: dict[str, Any]) -> bool:
    plan = decision.get("plan") if isinstance(decision.get("plan"), dict) else {}
    effect = str(plan.get("contact_window_effect") or "").strip().lower()
    return bool(
        effect in {"earlier", "later", "similar", "same", "unchanged", "unknown"}
        or plan.get("new_hard_prerequisites")
        or plan.get("production_tradeoffs")
    )


def build_contact_timing_extraction_prompt(
    *,
    decision: dict[str, Any],
    parent_text: str,
    candidate_text: str,
    knowledge_runs: list[dict[str, Any]] | None = None,
) -> str:
    """Ask the model only for structured gate/setup extraction, not arithmetic."""
    return f"""Extract the production package that determines first meaningful contact.

Do not judge whether the candidate is better. Do not calculate durations or costs.
Map every unit, structure, add-on, and upgrade to the exact executor action id shown
in verified knowledge. The code will perform the arithmetic after this extraction.

For both parent and candidate:
- gate_components are units or upgrades that must be complete before the first planned commitment. Use the explicit quantity and the number of production slots allocated to that action before commitment.
- setup_actions are structures, add-ons, gas buildings, or expansions explicitly required before that gate package is complete. Include absolute quantities and available parallel construction slots; do not include optional later-game production.
- economy records only the economy explicitly required before first commitment: living SCV target, completed base target, and workers assigned to gas. Use null when the strategy does not specify a value; do not estimate income or time.
- first commitment is the first strategically meaningful offensive commitment, not merely the earliest possible unit completion.
- identify whether a newly added gate component can prevent the parent commitment from launching, and whether the candidate contains an explicit fallback that preserves the parent contact window.
- a component required by plan.material_behavior_change or plan.coordinated_changes before first commitment remains a gate component even if candidate prose later calls it support or optional. Record that contradiction in notes rather than silently omitting the component.

Cross-match Decision and contact evidence:
{render_optimizer_decision(decision)}

Verified action metadata:
{render_knowledge_results(knowledge_runs or [])}

Parent strategy.md:
{parent_text}

Candidate strategy.md:
{candidate_text}

Return JSON only:
{{
  "timing_model": {{
    "parent": {{
      "economy": {{
        "worker_target_before_commitment": null,
        "base_target_before_commitment": null,
        "gas_workers_before_commitment": null
      }},
      "gate_components": [
        {{"action":"train_unit","quantity":1,"production_slots":1}}
      ],
      "setup_actions": [
        {{"action":"build_structure","quantity":1,"parallel_slots":1}}
      ]
    }},
    "candidate": {{
      "economy": {{
        "worker_target_before_commitment": null,
        "base_target_before_commitment": null,
        "gas_workers_before_commitment": null
      }},
      "gate_components": [
        {{"action":"train_unit","quantity":1,"production_slots":1}}
      ],
      "setup_actions": [
        {{"action":"build_structure","quantity":1,"parallel_slots":1}}
      ]
    }},
    "new_hard_gate_components": ["new component, or empty"],
    "fallback_preserves_parent_window": false,
    "notes": []
  }}
}}
"""


def _normalize_action_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def _knowledge_action_facts(
    knowledge_runs: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    for run in knowledge_runs or []:
        for evidence in run.get("dataset_evidence") or []:
            if not isinstance(evidence, dict):
                continue
            result = evidence.get("result")
            if not isinstance(result, dict):
                continue
            for row in result.get("action_facts") or []:
                if not isinstance(row, dict):
                    continue
                action = str(row.get("action") or "").strip()
                if action:
                    facts[_normalize_action_name(action)] = dict(row)
    return facts


def _calculate_timing_package(
    package: Any,
    facts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(package, dict):
        return {"complete": False, "errors": ["package is missing"]}
    result = simulate_terran_first_commitment(
        package,
        knowledge_facts=facts,
    )
    trace = list(result.pop("trace", []) or [])
    milestones: dict[str, float] = {}
    for item in trace:
        if item.get("event") == "complete" and str(item.get("action") or ""):
            milestones[str(item["action"])] = float(item.get("time") or 0.0)
    result["completion_milestones_seconds"] = {
        action: round(seconds, 3) for action, seconds in sorted(milestones.items())
    }
    return result


def _drop_implicit_supply_rows(package: Any) -> Any:
    """Drop malformed Depot rows when supply is intentionally simulator-derived.

    Strategies commonly require continuous Supply Depot construction without an
    absolute pre-commitment count. The simulator already derives the minimum Depot
    count from requested workers and units. A model-extracted row with neither a
    positive quantity nor parallel capacity must therefore be omitted instead of
    making the entire timing report unusable.
    """
    if not isinstance(package, dict):
        return package
    normalized = dict(package)
    setup_actions: list[Any] = []
    for item in package.get("setup_actions") or []:
        if not isinstance(item, dict):
            setup_actions.append(item)
            continue
        action = str(item.get("action") or "").strip().casefold()
        if action == "build_supply_depot":
            quantity = item.get("quantity")
            slots = item.get("parallel_slots")
            quantity_missing = (
                not isinstance(quantity, (int, float)) or quantity <= 0
            )
            slots_missing = not isinstance(slots, (int, float)) or slots <= 0
            if quantity_missing and slots_missing:
                continue
        setup_actions.append(dict(item))
    normalized["setup_actions"] = setup_actions
    return normalized


def _build_contact_timing_report(
    timing_model: Any,
    knowledge_runs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if not isinstance(timing_model, dict):
        return {"complete": False, "errors": ["timing_model is missing"]}
    facts = _knowledge_action_facts(knowledge_runs)
    parent_package = _drop_implicit_supply_rows(timing_model.get("parent"))
    candidate_package = _drop_implicit_supply_rows(timing_model.get("candidate"))
    parent = _calculate_timing_package(parent_package, facts)
    candidate = _calculate_timing_package(candidate_package, facts)
    errors = list(parent.get("errors") or []) + list(candidate.get("errors") or [])
    parent_time = float(parent.get("earliest_feasible_time_seconds") or 0.0)
    candidate_time = float(candidate.get("earliest_feasible_time_seconds") or 0.0)
    delay = candidate_time - parent_time if parent_time and candidate_time else None
    parent_cost = parent.get("total_cost") or {}
    candidate_cost = candidate.get("total_cost") or {}
    return {
        "complete": bool(parent.get("complete") and candidate.get("complete") and not errors),
        "parent": parent,
        "candidate": candidate,
        "parent_earliest_feasible_time_seconds": round(parent_time, 3) if parent_time else None,
        "candidate_earliest_feasible_time_seconds": round(candidate_time, 3) if candidate_time else None,
        "earliest_feasible_timing_delta_seconds": round(delay, 3)
        if delay is not None
        else None,
        "gate_cost_delta": {
            key: round(float(candidate_cost.get(key) or 0.0) - float(parent_cost.get(key) or 0.0), 3)
            for key in ("minerals", "gas", "supply")
        },
        "new_hard_gate_components": list(
            timing_model.get("new_hard_gate_components") or []
        ),
        "fallback_preserves_parent_window": bool(
            timing_model.get("fallback_preserves_parent_window")
        ),
        "declared_packages": {
            "parent": dict(parent_package or {}),
            "candidate": dict(candidate_package or {}),
        },
        "errors": errors,
        "evidence_warnings": list(
            dict.fromkeys(
                list(parent.get("warnings") or []) + list(candidate.get("warnings") or [])
            )
        ),
        "interpretation": (
            "These are resource-feasible package completion estimates from the project runtime start. "
            "They exclude decision latency, assembly, travel, and combat. Compare any added "
            "minimum delay with empirical opponent growth before accepting the candidate."
        ),
    }


_GENERIC_WHY = {
    "improves the strategy",
    "better scouting",
    "more consistent",
    "improves scouting",
    "better scouting is useful",
}

_COMMANDER_OWNED_DETAIL_IDS = frozenset({"scouting", "scans"})

def build_strategy_patch_validation_prompt(
    *,
    decision: dict[str, Any],
    parent_text: str,
    candidate_text: str,
    patches: list[dict[str, Any]],
    capability_manifest: dict[str, Any] | None = None,
    knowledge_runs: list[dict[str, Any]] | None = None,
    inheritance: dict[str, Any] | None = None,
    preservation_audit: list[dict[str, Any]] | None = None,
    prior_experiences: list[Any] | None = None,
    contact_timing_report: dict[str, Any] | None = None,
) -> str:
    return f"""You are validating a strategy patch represented as a complete SC2 natural-language strategy revision. Act as a narrow semantic and executability reviewer. Do not re-rank hypotheses, propose a better strategy, require extra detail, or reject a candidate merely because it changes goal wording, unit composition, or retreat behavior.

{HARD_VALIDATION_POLICY}

The selected evidence-supported intervention is authoritative:
{render_optimizer_decision(decision)}

Apply three additional consistency checks only:
- Every material change must implement the selected hypothesis or be a necessary dependency of it; an unrelated second objective is blocking.
- The overall timing and manner of gaining advantage must remain the selected combat style unless the intervention explicitly and evidentially revises it. Exact goal wording, unit roster, quantities, production, and retreat rules are not protected fields.
- Main Attack Gate is only for the first planned commitment. Recovery and Cleanup may change when post-contact evidence or the selected hypothesis requires it, but copying or strengthening the opening gate in recovery merely for numerical or textual consistency is a blocking internal inconsistency.

Runtime capabilities:
{json_compact_block(capability_manifest or {})}

Verified deterministic knowledge:
{render_knowledge_results(knowledge_runs or [])}

Parent strategy.md:
{parent_text}

Candidate strategy.md:
{candidate_text}

Return JSON only:
{{"valid":true,"errors":[{{"type":"unsupported_action|missing_dependency|internal_inconsistency|supply_cap","location":"strategy paragraph","description":"hard non-executable error","severity":"blocking"}}]}}

Use an empty errors list when the candidate is executable and is a coherent implementation of the selected intervention. Strategic uncertainty, missing audit fields, a debatable unit choice, a supported timing tradeoff, or similarity to history is non-blocking.
"""
    compact_patches = [
        {
            "target": str(item.get("target") or ""),
            "why_required": str(item.get("why_required") or ""),
            "replacement": str(item.get("replacement") or ""),
        }
        for item in patches
        if isinstance(item, dict)
    ]
    from .prompts import RUNTIME_CONTRACT

    compact_experiences = _compact_prior_experiences(prior_experiences or [])

    return f"""You are validating a strategy patch represented as a complete Summary/Details strategy revision.

You are NOT re-ranking alternative hypotheses or choosing a preferred strategy.
You are NOT choosing a better strategy.
You are NOT analyzing the matches.
You are NOT judging whether another causal hypothesis would have been better.

The Cross-Match Decision has already selected one primary failure mode and one
coherent intervention package in plan.

Check only whether the candidate is a clean implementation of that hypothesis.

Summary defines strategy identity and the overall win mechanism; the titled Details
paragraphs define development and army use. Main Attack Gate applies only to the
first planned attack. Recovery and Cleanup is a separate post-engagement rule and
must not copy, synchronize with, or silently strengthen the opening gate.

Return a blocking error only when at least one of these conditions is present:

1. The candidate omits a coordinated change, cannot produce the declared material
   behavior change, or adds an unrelated second objective.
2. The complete candidate contains a direct contradiction, misses a prerequisite
   required by its own rules, or contradicts supplied deterministic SC2 knowledge.
3. The candidate requires a control or observation that the runtime contract does
   not provide.
4. It removes or reverses a defining Champion mechanism without declaring and
   justifying that change in the inheritance ledger.
5. A production or supply bound required below is missing or impossible.
6. The candidate claims to preserve the Champion's combat style or critical power
   window but introduces a hidden attack prerequisite, production competition, or
   technology delay that materially changes or suppresses that window.
7. The concrete intervention is semantically equivalent to a prior implemented
   and contradicted mechanism, or repeats an exhausted underpowered mechanism
   without directly repairing its recorded failed dependencies.
8. The candidate delays commitment or omits target discovery and cleanup so that it
   has no credible way to locate and destroy all enemy structures within the hard
   1800-second match limit.

Do not re-rank the hypothesis, propose a preferred strategy, or turn uncertainty,
wording style, harmless duplication, and optional refinements into blocking errors.

Strategy style is not a fixed unit roster. The candidate may change units,
upgrades, production, economy, and exact thresholds when those changes preserve
the intended combat style and are supported by the Cross-Match Decision. Judge
style through the timing and manner of gaining advantage: early pressure,
concentrated timing attack, defensive scaling, reinforcement pattern, commitment,
and recovery. Do not reject a candidate merely because it adds or removes a unit.

Use plan.strategy_area_audit as design guidance. Check whether revised areas are implemented and remain compatible with preserved areas, but do not block a candidate merely because an audit entry is missing, concise, or leaves an unaffected paragraph unchanged. Return a blocking error only when the omission creates a direct contradiction, makes the selected package non-executable, or clearly prevents its material behavior change. A coherent intervention may modify many titled paragraphs when they are causal dependencies of the same hypothesis; do not misclassify those dependent changes as unrelated objectives.

Perform a staged production review beyond the first commitment. Check the opening order; relevant pre-commitment and post-commitment producer and unit targets; worker, base, gas, supply, and upgrade support; reinforcement throughput; and late-game completion. Missing detail is non-blocking unless a changed unit or production facility is unbounded in a way that creates a direct contradiction, exceeds supply, or makes the selected package impossible to execute. More production capacity should match the observed resource and queue bottleneck: mineral banking with occupied core queues may justify another relevant mineral-cost producer, while gas banking alone calls for a supported gas use or worker reallocation rather than an unrelated producer.

Check offensive continuity semantically against the inferred combat style and reinforcement_retreat_cleanup audit. A pressure strategy must not silently become a retreat-and-full-rebuild loop that repeatedly suspends a viable offensive and reuses the opening gate. A timing or scaling strategy may preserve an evidence-supported disengage-and-regroup cycle. Treat tactical survival and strategic continuity as evidence-dependent, not as a universal no-retreat or always-retreat rule.

Perform a style-and-window audit:
- infer the Champion's combat style and critical power window from the strategy
  contract and complete parent document;
- compare the parent and candidate launch prerequisites, expected contact window,
  production allocation, and technology depth;
- list every new hard prerequisite for the first planned attack;
- identify competition between support units and core units that share production
  structures or resources;
- reject an undeclared or unjustified delay, a hidden attack gate, or a change that
  turns the strategy into a different combat style;
- allow an evidence-supported timing shift when the Decision explicitly selects
  and justifies that shift.

Also check endgame completion. Winning an army engagement is not sufficient: the
candidate must preserve enough time and non-blocking scouting or scanning instructions
to locate surviving enemy structures and finish the match before 1800 seconds.
Information gathering may support a named attack or cleanup decision but must not
become a hidden prerequisite for an otherwise ready force.

Perform a failure-stage scope audit. Compare failure_mode_analysis.failure_stage,
plan.strategy_area_audit, the selected hypothesis, the complete parent, and the
complete candidate. Composition and retreat/recovery are ordinary evolvable strategy
areas: do not require a separate permission flag. For each material change, decide
whether it implements the selected hypothesis, is a necessary dependency of that
hypothesis, or is an unrelated second objective. Reject only unrelated or
stage-inconsistent changes. In particular, detect when a first-attack gate is copied
into Recovery and Cleanup without post-contact evidence; do not infer that matching
numbers across those stages is inherently consistent.

Perform a contact-window comparison using the deterministic timing report and the
recorded contact evidence. The program reports the earliest resource-feasible time
at which each declared first-commitment package can exist; do not replace or
recalculate those values. Compare the candidate's added minimum delay and own package
against the opponent package observed around the corresponding game period, including
counters, upgrades, combat power, and growth between the two feasible windows. Also consider whether production after
first contact can sustain the intended pressure. Include the configured retreat
ratio, local power at any auto-retreat trigger, force retained after withdrawal,
and time to regroup or re-engage. Runtime auto-retreat fires when the local
own/enemy power ratio falls below retreat_ratio (default 0.6), so a higher value
retreats earlier and a lower value stays committed longer. Do not treat either
direction as inherently better. A later package is valid only when
its matchup-adjusted relative advantage at contact is preserved or improved enough
to offset opponent growth. Do not call a later package favorable merely because it
contains more own units.

A support unit may be optional at first contact or may be a hard gate. It becomes a
blocking inconsistency only when the candidate says the original power window is
preserved while the new support requirement can prevent that attack from launching.

Test strength:
The candidate must be structurally capable of producing the pre-registered
mechanism_prediction.expected_change at or beyond minimum_material_change. Reject
a cosmetic, token, isolated, or clearly underpowered implementation that cannot materially
test the supplied hypothesis. Judge intervention strength from the declared
mechanism and parent-to-candidate strategy difference, never from patch count.
This validates designed test strength only; do not claim that runtime execution or
the realized match mechanism has already been observed.

Perform one candidate-wide production_target_audit. Every unit that the complete
candidate explicitly says to continue, resume, restart, or re-enable must have its
own numerical stage target. If it remains in ongoing reinforcement or late-game
production, it must also have a numerical final count or cap. A temporary unit may
instead have a stage count and explicit stop rule. Do not infer quantities from
remaining supply or prose ratios.

Perform one new_dependency_audit for every unit, upgrade, structure, or production
transition added by the candidate or moved to an earlier stage. Verify its complete
resource and production chain at the stage where it is required. In particular, a
gas-cost target conflicts with an explicit zero-Refinery or mineral-only economy
unless the candidate explicitly changes that economy before the target is produced.
Runtime prerequisite expansion does not repair a direct contradiction in
strategy.md. Also verify producers, add-ons, prerequisite structures, and whether
the new target competes with a protected core unit for the same production slots.

Recompute final_supply from workers and every final combat/support unit. The total
must be complete and no greater than 200.

Judge plan coverage and identity semantically from the complete parent, candidate,
decision and inheritance ledger. Do not use paragraph names, mechanism-family
spelling, or isolated keywords as proof that two rules are equivalent or invalid.

The validator reports errors only; it must not generate replacement patches.

{RUNTIME_CONTRACT}
Capability summary:
{json.dumps(capability_manifest or {}, ensure_ascii=False, indent=2)}

Cross-match Decision:
{render_optimizer_decision(decision)}

Verified knowledge and deterministic feasibility calculations:
{render_knowledge_results(knowledge_runs or [])}

Deterministic resource-aware first-commitment timing report:
{json_compact_block(contact_timing_report or {})}

Prior experiment evidence relevant to mechanism policy:
{json_compact_block(compact_experiences)}

Patches:
{json_compact_block(compact_patches)}

Inheritance ledger:
{json_compact_block(inheritance or {})}

Optimizer preservation audit:
{json_compact_block(preservation_audit or [])}

Parent strategy.md:
{parent_text}

Candidate strategy.md:
{candidate_text}

Return JSON only:
{{
  "valid": true,
  "production_target_audit": [
    {{
      "unit": "unit whose production resumes or continues",
      "instruction": "the production instruction being audited",
      "stage_target": "the unit's explicit numerical production target, or empty",
      "ultimate_goal_target": "the unit's explicit numerical count in Macro.Ultimate Goal, or empty",
      "temporary_stop_rule": "explicit stop condition, or empty when continuously reinforced",
      "verdict": "bounded|missing_stage_target|missing_ultimate_goal_target|invalid_temporary_exception"
    }}
  ],
  "new_dependency_audit": [
    {{
      "target": "new or earlier unit, upgrade, structure, or transition",
      "stage": "when the candidate requires it",
      "gas_required": false,
      "gas_plan": "present|none|not_needed",
      "required_prerequisites": ["required producer, add-on, structure, or prior upgrade"],
      "declared_prerequisites": ["matching candidate instruction"],
      "missing_dependencies": [],
      "shared_production_tradeoff": "competition with protected core production, or empty",
      "verdict": "complete|missing_prerequisite|resource_conflict|stage_conflict"
    }}
  ],
  "final_supply": {{
    "total": 0,
    "calculation": "workers plus every Macro.Ultimate Goal combat/support unit at full supply cost",
    "verdict": "valid|over_200|incomplete"
  }},
  "style_and_window_audit": {{
    "parent_combat_style": "short description",
    "candidate_combat_style": "short description",
    "style_preserved": true,
    "contact_window_effect": "earlier|similar|later|unknown",
    "window_change_justified": true,
    "new_hard_prerequisites": ["new prerequisite, if any"],
    "shared_production_tradeoffs": ["support/core production competition, if any"],
    "hidden_attack_gate": false,
    "verdict": "preserved|evidence_supported_shift|style_drift"
  }},
  "failure_stage_scope_audit": {{
    "failure_stage": "before_core_mechanism|during_commitment_or_engagement|after_successful_engagement|mixed",
    "composition_changed": false,
    "composition_change_relation": "none|implements_selected_hypothesis|necessary_dependency|unrelated",
    "retreat_policy_changed": false,
    "retreat_change_relation": "none|implements_selected_hypothesis|necessary_dependency|unrelated",
    "opening_gate_reused_as_recovery_gate": false,
    "opening_gate_reuse_supported": true,
    "stage_scope_aligned": true,
    "reason": "semantic parent-candidate comparison against the selected failure stage"
  }},
  "contact_window_audit": {{
    "parent_earliest_feasible_time_seconds": 0,
    "candidate_earliest_feasible_time_seconds": 0,
    "own_package_at_candidate_contact": "candidate package expected at contact",
    "opponent_package_at_candidate_contact": "empirical opponent package near that time",
    "opponent_growth_during_wait": "material enemy growth between the two windows",
    "matchup_and_counter_assessment": "how the two packages interact",
    "reinforcement_and_continuity": "retreat threshold and trigger quality, force retained, post-contact production, regroup delay, and ability to re-engage",
    "relative_advantage": "improves|preserves|worsens|unknown",
    "evidence": ["Game N @ Ts: recorded comparison"],
    "verdict": "favorable|preserved|unsupported|unfavorable"
  }},
  "winning_mechanism_audit": {{
    "parent_winning_chain":"causal sequence supported by successful matches",
    "candidate_winning_chain":"how the complete candidate reproduces that sequence",
    "reviewed_invariants":[
      {{"invariant":"protected item from the Decision","candidate_effect":"preserved|improved|evidence_supported_revision|broken","reason":"semantic comparison of parent and candidate"}}
    ],
    "earliest_broken_link":"first lost winning mechanism, or empty",
    "verdict":"preserved|evidence_supported_revision|broken"
  }},
  "errors": [
    {{
      "type": "decision_grounding|unrelated_patch|missing_dependency|underpowered_implementation|internal_inconsistency|preserved_strengths|strategy_identity|runtime_boundary",
      "location": "Summary or Detail paragraph title",
      "description": "what is wrong",
      "severity": "blocking|non-blocking"
    }}
  ]
}}

Set valid=true when there are no blocking issues. Include non-blocking notes
only as non-blocking errors; they must not make valid=false.
"""


def _compact_prior_experiences(items: list[Any]) -> list[dict[str, Any]]:
    experiments = _experiment_history_items(items)
    durable_ids = {
        str(item.get("experiment_id") or "")
        for item in experiments
        if str(item.get("decision") or "").strip().lower() == "accepted"
        or str(item.get("hypothesis_verdict") or "").strip().lower()
        == "contradicted"
        or str(item.get("implementation_verdict") or "").strip().lower()
        == "execution_invalid"
        or item.get("underpowered_retry_exhausted") is True
    }
    selected: list[Any] = []
    for item in items:
        experiment_id = (
            str(item.get("experiment_id") or "") if isinstance(item, dict) else ""
        )
        if experiment_id in durable_ids or item in items[-8:]:
            selected.append(item)
    compact: list[dict[str, Any]] = []
    for item in selected[-24:]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        if kind:
            compact.append(
                {
                    key: item.get(key)
                    for key in (
                        "kind",
                        "difficulty",
                        "mechanism_family",
                        "blocked_mechanism_families",
                        "reason",
                        "rule",
                        "instruction",
                    )
                    if item.get(key) not in (None, "", [], {})
                }
            )
            continue
        intervention = item.get("intervention_package")
        compact.append(
            {
                key: value
                for key, value in {
                    "experiment_id": item.get("experiment_id"),
                    "candidate": item.get("candidate"),
                    "mechanism_family": item.get("mechanism_family"),
                    "primary_change": item.get("primary_change"),
                    "hypothesis": item.get("hypothesis"),
                    "plan_direction": item.get("plan_direction"),
                    "mechanism_prediction": item.get("mechanism_prediction"),
                    "intervention_package": (
                        {
                            name: intervention.get(name)
                            for name in ("direction", "material_behavior_change")
                            if intervention.get(name) not in (None, "", [], {})
                        }
                        if isinstance(intervention, dict)
                        else {}
                    ),
                    "implementation_verdict": item.get("implementation_verdict"),
                    "hypothesis_verdict": item.get("hypothesis_verdict"),
                    "decision": item.get("decision"),
                    "score_delta": item.get("score_delta"),
                    "repairable_underpowered_retry": item.get(
                        "repairable_underpowered_retry"
                    ),
                    "underpowered_retry_exhausted": item.get(
                        "underpowered_retry_exhausted"
                    ),
                    "failed_dependencies": list(item.get("failed_dependencies") or [])[:4],
                    "lesson": item.get("lesson"),
                }.items()
                if value not in (None, "", [], {})
            }
        )
    return compact


def build_mechanism_equivalence_prompt(
    *,
    decision: dict[str, Any],
    parent_text: str,
    candidate_text: str,
    patches: list[dict[str, Any]],
    prior_experiences: list[Any],
) -> str:
    """Build a focused semantic-history check independent of patch validation."""
    history = _compact_prior_experiences(prior_experiences)
    return f"""You are an independent semantic experiment-history judge.

Decide whether the proposed intervention is causally the same as a prior failed intervention. Ignore mechanism-family names, labels, version suffixes, paragraph ids, and wording overlap. Compare the actual strategy behavior being changed, its direction, its required dependencies, the intended contact window, and the causal prediction. For example, raising an attack threshold from 20 to 30 and later from 20 to 50 is normally the same direction even when the labels differ. Conversely, changing attack timing and changing post-contact reinforcement are not equivalent merely because both mention Marines.

Use material_repair only when the candidate keeps the same causal idea but concretely repairs a dependency that prevented the earlier experiment from being implemented. Name the repaired dependency and point to the matching failed dependency in history. A larger numerical change, a renamed mechanism, or another prompt explanation is not a material repair by itself.

Current Cross-match Decision:
{render_optimizer_decision(decision)}

Prior experiment history:
{json_compact_block(history)}

Parent strategy.md:
{parent_text}

Candidate strategy.md:
{candidate_text}

Candidate changes:
{json_compact_block(patches)}

Return JSON only:
{{
  "mechanism_equivalence_audit": {{
    "semantic_relation": "new|material_repair|equivalent_to_prior",
    "related_experiment_ids": ["exact experiment_id from history"],
    "repaired_dependencies": ["concrete dependency repaired, or empty"],
    "reason": "brief comparison of causal behavior, not names",
    "confidence": "high|medium|low"
  }}
}}
"""


def _experiment_history_items(items: list[Any] | None) -> list[dict[str, Any]]:
    return [
        item
        for item in (items or [])
        if isinstance(item, dict)
        and not str(item.get("kind") or "").strip()
        and str(item.get("experiment_id") or "").strip()
    ]


def _latest_gate_execution_issue(items: list[Any] | None) -> dict[str, Any]:
    for item in reversed(_experiment_history_items(items)):
        audit = item.get("gate_execution_audit")
        if not isinstance(audit, dict):
            continue
        status = str(audit.get("status") or "").strip().lower()
        if status in {"execution_issue", "measured"}:
            return dict(audit) if status == "execution_issue" else {}
    return {}


def _normalize_mechanism_equivalence_audit(
    raw: Any,
    *,
    prior_experiences: list[Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    payload = (
        raw.get("mechanism_equivalence_audit")
        if isinstance(raw, dict)
        and isinstance(raw.get("mechanism_equivalence_audit"), dict)
        else raw
    )
    if not isinstance(payload, dict):
        return {}, [
            "decision_grounding — independent mechanism-history audit — "
            "model returned no JSON object"
        ]
    relation = str(payload.get("semantic_relation") or "").strip().lower()
    if relation not in {"new", "material_repair", "equivalent_to_prior"}:
        return {}, [
            "decision_grounding — independent mechanism-history audit — "
            "semantic_relation must be new, material_repair, or equivalent_to_prior"
        ]
    history = _experiment_history_items(prior_experiences)
    by_id = {
        str(item.get("experiment_id") or "").strip(): item for item in history
    }
    related_ids = list(
        dict.fromkeys(
            str(item).strip()
            for item in (payload.get("related_experiment_ids") or [])
            if str(item).strip()
        )
    )
    related = [by_id[item] for item in related_ids if item in by_id]
    repaired = [
        str(item).strip()
        for item in (payload.get("repaired_dependencies") or [])
        if str(item).strip()
    ]
    audit = {
        "semantic_relation": relation,
        "related_experiment_ids": related_ids,
        "repaired_dependencies": repaired,
        "reason": str(payload.get("reason") or "").strip(),
        "confidence": str(payload.get("confidence") or "").strip().lower(),
        "verdict": "allowed",
    }
    errors: list[str] = []
    if relation in {"equivalent_to_prior", "material_repair"} and not related:
        errors.append(
            "decision_grounding — independent mechanism-history audit — semantic "
            "equivalence must reference an exact prior experiment_id"
        )
    if relation == "material_repair" and not repaired:
        errors.append(
            "decision_grounding — material repair — identify the concrete failed "
            "dependency repaired by this candidate"
        )
    if relation == "equivalent_to_prior":
        failed = []
        for item in related:
            implementation = str(
                item.get("implementation_verdict") or "unknown"
            ).strip().lower()
            hypothesis = str(
                item.get("hypothesis_verdict") or "inconclusive"
            ).strip().lower()
            decision = str(item.get("decision") or "").strip().lower()
            if (
                hypothesis == "contradicted"
                or implementation in {"underpowered", "execution_invalid", "unknown"}
                or item.get("underpowered_retry_exhausted") is True
                or decision in {"rejected", "policy_rejected_before_matches"}
            ):
                failed.append(str(item.get("experiment_id") or ""))
        if failed:
            audit["verdict"] = "blocked"
            errors.append(
                "decision_grounding — mechanism history — candidate is semantically "
                "equivalent to failed prior experiment(s): " + ", ".join(failed)
            )
    return audit, list(dict.fromkeys(errors))


def validate_strategy_patch_structure(
    *,
    decision: dict[str, Any],
    patches: list[dict[str, Any]],
    parent_document: StrategyDocument,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(patches, list) or not patches:
        errors.append("strategy patch must contain at least one paragraph change")
        return errors

    hypothesis = str(decision.get("hypothesis") or "").strip()
    if not hypothesis:
        errors.append("cross-match hypothesis is missing")

    detail_ids = {item.id for item in parent_document.details}
    seen: set[str] = set()
    for patch in patches:
        if not isinstance(patch, dict):
            errors.append("each strategy patch must be an object")
            continue
        target = str(patch.get("target") or "").strip()
        replacement = str(patch.get("replacement") or patch.get("value") or "").strip()
        why = str(patch.get("why_required") or "").strip()
        if not target:
            errors.append("patch target is required")
            continue
        if target == "summary":
            errors.append("summary modification is not allowed")
        if target in _COMMANDER_OWNED_DETAIL_IDS:
            errors.append(
                f"{target}: scouting and scanning behavior is Commander-owned "
                "and not evolvable"
            )
        if target not in detail_ids and target != "summary":
            errors.append(f"unknown strategy paragraph: {target}")
        if target in seen:
            errors.append(f"duplicate patch target: {target}")
        seen.add(target)
        if not replacement:
            errors.append(f"{target}: replacement is required")
        else:
            current = next(
                (item for item in parent_document.details if item.id == target),
                None,
            )
            if current is not None and replacement == current.value:
                errors.append(f"{target}: replacement is unchanged")
        if "\n" in str(patch.get("replacement") or patch.get("value") or ""):
            errors.append(
                f"{target}: replacement must be one complete Detail instruction"
            )
        if not why:
            errors.append(f"{target}: why_required is required")
        elif why.lower() in _GENERIC_WHY:
            errors.append(
                f"{target}: why_required must explain a direct hypothesis dependency"
            )
    return errors


def _runtime_boundary_errors(
    patches: list[dict[str, Any]],
    candidate_text: str,
    capability_manifest: dict[str, Any],
) -> list[str]:
    # Runtime feasibility depends on sentence meaning and the supplied capability
    # contract. Keyword/token parsing produced false positives for equivalent names
    # and ordinary strategy prose, so it is intentionally left to the semantic
    # validator. Keep this hook for API compatibility and future structural checks.
    del patches, candidate_text, capability_manifest
    return []


def _semantic_issue(item: Any) -> tuple[str, str]:
    if isinstance(item, dict):
        severity = str(item.get("severity") or "blocking").strip().lower()
        if severity not in {"blocking", "non-blocking"}:
            severity = "blocking"
        parts = [
            str(item.get("type") or "").strip(),
            str(item.get("location") or "").strip(),
            str(item.get("description") or item.get("error") or "").strip(),
        ]
        message = " — ".join(part for part in parts if part)
        if not message:
            message = json.dumps(item, ensure_ascii=False)
        return severity, message
    return "blocking", str(item).strip()


_MISSING_TARGET_MARKERS = (
    "no explicit",
    "not explicit",
    "not specified",
    "not listed",
    "not stated",
    "missing",
    "unspecified",
    "unknown",
    "none",
    "n/a",
)


def _audit_value_is_missing(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return not text or any(marker in text for marker in _MISSING_TARGET_MARKERS)


def _production_contract_errors(payload: dict[str, Any]) -> list[str]:
    """Block explicit impossible production or supply findings, not missing prose."""
    errors: list[str] = []
    audit = payload.get("production_target_audit")
    if isinstance(audit, list):
        for index, row in enumerate(audit, start=1):
            if not isinstance(row, dict):
                continue
            unit = str(row.get("unit") or "").strip()
            label = unit or f"row {index}"
            stage_missing = _audit_value_is_missing(row.get("stage_target"))
            ultimate_missing = _audit_value_is_missing(
                row.get("ultimate_goal_target")
            )
            stop_missing = _audit_value_is_missing(row.get("temporary_stop_rule"))
            verdict = str(row.get("verdict") or "").strip().lower()
            if (
                verdict in {"unbounded", "invalid", "inconsistent"}
                and stage_missing
                and ultimate_missing
                and stop_missing
            ):
                errors.append(
                    f"missing_dependency — production bound for {label} — "
                    "the validator explicitly found an unbounded continuing target"
                )

    final_supply = payload.get("final_supply")
    if isinstance(final_supply, dict):
        total = final_supply.get("total")
        if (
            not isinstance(total, bool)
            and isinstance(total, (int, float))
            and (total < 0 or total > 200)
        ):
            errors.append(
                f"internal_inconsistency — final_supply.total — {total} exceeds "
                "the valid 0-200 supply range"
            )
        verdict = str(final_supply.get("verdict") or "").strip().lower()
        if verdict in {"over_200", "invalid", "impossible"}:
            errors.append(
                "internal_inconsistency — final_supply.verdict — "
                f"validator reported {verdict}"
            )
    return list(dict.fromkeys(errors))


def _style_and_history_contract_errors(
    payload: dict[str, Any],
    *,
    decision: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    style = payload.get("style_and_window_audit")
    if isinstance(style, dict):
        if style.get("style_preserved") is False:
            errors.append(
                "strategy_identity — combat style — candidate changes the "
                "Champion's intended combat style"
            )
        effect = str(style.get("contact_window_effect") or "").strip().lower()
        if effect in {"earlier", "later"} and style.get("window_change_justified") is False:
            errors.append(
                "strategy_identity — critical power window — candidate changes or "
                "suppresses the contact window without support from the selected decision"
            )
        if style.get("hidden_attack_gate") is True:
            errors.append(
                "internal_inconsistency — hidden attack gate — a production, support, "
                "or technology prerequisite can block the declared Main Attack Gate"
            )

    winning = payload.get("winning_mechanism_audit")
    if isinstance(winning, dict):
        reviewed = [
            item
            for item in (winning.get("reviewed_invariants") or [])
            if isinstance(item, dict)
        ]
        if any(
            str(item.get("candidate_effect") or "").strip().lower() == "broken"
            for item in reviewed
        ):
            errors.append(
                "preserved_strengths — broken winning mechanism — candidate removes "
                "a validated Champion advantage"
            )
        winning_verdict = str(winning.get("verdict") or "").strip().lower()
        if winning_verdict == "broken":
            errors.append(
                "preserved_strengths — winning-mechanism verdict — validator reported broken"
            )
        if str(winning.get("earliest_broken_link") or "").strip():
            errors.append(
                "preserved_strengths — earliest broken link — candidate cannot "
                "reproduce the Champion's validated winning chain"
            )

    scope = payload.get("failure_stage_scope_audit")
    failure_mode = (decision or {}).get("failure_mode_analysis") or {}
    expected_stage = str(failure_mode.get("failure_stage") or "").strip().lower()
    if isinstance(scope, dict):
        reported_stage = str(scope.get("failure_stage") or "").strip().lower()
        if expected_stage and reported_stage and reported_stage != expected_stage:
            errors.append(
                "decision_grounding — failure-stage scope — semantic audit used a "
                "different failure stage from the Cross-match Decision"
            )
        composition_relation = str(
            scope.get("composition_change_relation") or "none"
        ).strip().lower()
        if bool(scope.get("composition_changed")) and composition_relation == "unrelated":
            errors.append(
                "unrelated_patch — composition scope — candidate changes the unit "
                "concept without a causal role in the selected hypothesis"
            )
        retreat_relation = str(
            scope.get("retreat_change_relation") or "none"
        ).strip().lower()
        if bool(scope.get("retreat_policy_changed")) and retreat_relation == "unrelated":
            errors.append(
                "unrelated_patch — retreat scope — candidate changes retreat policy "
                "without a causal role in the selected hypothesis"
            )
        if (
            scope.get("opening_gate_reused_as_recovery_gate") is True
            and scope.get("opening_gate_reuse_supported") is not True
        ):
            errors.append(
                "internal_inconsistency — recovery scope — candidate reuses the "
                "first-attack gate as a recovery gate without post-contact evidence"
            )
        if scope.get("stage_scope_aligned") is False:
            errors.append(
                "decision_grounding — failure-stage scope — candidate changes do not "
                "match the selected failure stage and scope permissions"
            )
    return list(dict.fromkeys(errors))


def _new_dependency_contract_errors(payload: dict[str, Any]) -> list[str]:
    """Enforce resource and prerequisite closure for candidate-added targets."""

    audit = payload.get("new_dependency_audit")
    if not isinstance(audit, list):
        return []
    errors: list[str] = []
    for index, row in enumerate(audit, start=1):
        if not isinstance(row, dict):
            continue
        target = str(row.get("target") or "").strip()
        label = target or f"row {index}"
        gas_required = row.get("gas_required") is True
        gas_plan = str(row.get("gas_plan") or "").strip().lower()
        if gas_required and gas_plan != "present":
            errors.append(
                f"missing_dependency — gas economy for {label} — a gas-cost target "
                "requires an explicit compatible gas plan before its stage"
            )
        missing = [
            str(item).strip()
            for item in (row.get("missing_dependencies") or [])
            if str(item).strip()
        ]
        if missing:
            errors.append(
                f"missing_dependency — dependency chain for {label} — "
                + "; ".join(missing)
            )
        verdict = str(row.get("verdict") or "").strip().lower()
        if verdict in {"missing_prerequisite", "incomplete", "invalid"}:
            errors.append(
                f"missing_dependency — new_dependency_audit for {label} — "
                f"validator reported {verdict}"
            )
    return list(dict.fromkeys(errors))


def _contact_window_contract_errors(
    payload: dict[str, Any],
    *,
    decision: dict[str, Any] | None = None,
    contact_timing_report: dict[str, Any] | None = None,
) -> list[str]:
    if not _timing_audit_required(decision or {}):
        return []
    report = contact_timing_report or {}
    if report.get("skipped") is True:
        return []
    if report.get("complete") is not True:
        detail = "; ".join(str(item) for item in report.get("errors") or [])
        return [
            "decision_grounding — contact timing calculation — candidate timing "
            "package could not be calculated from verified action metadata"
            + (f": {detail}" if detail else "")
        ]
    audit = payload.get("contact_window_audit")
    if not isinstance(audit, dict):
        return [
            "decision_grounding — contact_window_audit — compare the calculated "
            "candidate contact window with empirical opponent growth"
        ]
    errors: list[str] = []
    timing_summary = (
        f"parent earliest feasible="
        f"{report.get('parent_earliest_feasible_time_seconds')}s, "
        f"candidate earliest feasible="
        f"{report.get('candidate_earliest_feasible_time_seconds')}s, "
        f"minimum timing delta={report.get('earliest_feasible_timing_delta_seconds')}s, "
        f"required-package cost delta={report.get('gate_cost_delta')}"
    )
    relative = str(audit.get("relative_advantage") or "").strip().lower()
    verdict = str(audit.get("verdict") or "").strip().lower()
    evidence = [str(item).strip() for item in audit.get("evidence") or [] if str(item).strip()]
    if relative == "worsens":
        errors.append(
            "preserved_strengths — relative power at contact — candidate does not "
            f"show a preserved or improved matchup-adjusted advantage "
            f"({relative or 'unknown'}); {timing_summary}"
        )
    if verdict == "unfavorable":
        errors.append(
            "preserved_strengths — contact-window verdict — waiting for the candidate "
            f"package is unsupported or unfavorable after opponent growth; {timing_summary}"
        )
    # Missing trajectory evidence is uncertainty to be resolved by candidate
    # matches, not proof that an otherwise executable candidate is invalid.
    # Explicit evidence of a worse contact window remains blocking above.
    return list(dict.fromkeys(errors))


def _blocking_semantic_errors(
    payload: dict[str, Any],
    *,
    decision: dict[str, Any] | None = None,
    contact_timing_report: dict[str, Any] | None = None,
) -> list[str]:
    reported = payload.get("errors") or []
    blocking: list[str] = []
    has_non_blocking = False
    if not isinstance(reported, list):
        reported = [reported]
    for item in reported:
        severity, message = _semantic_issue(item)
        if not message:
            continue
        if severity == "non-blocking":
            has_non_blocking = True
            continue
        blocking.append(message)
    valid = payload.get("valid")
    blocking.extend(_production_contract_errors(payload))
    blocking.extend(_new_dependency_contract_errors(payload))
    blocking.extend(
        _style_and_history_contract_errors(payload, decision=decision)
    )
    blocking.extend(
        _contact_window_contract_errors(
            payload,
            decision=decision,
            contact_timing_report=contact_timing_report,
        )
    )
    if blocking:
        return list(dict.fromkeys(blocking))
    if valid is True or (valid is False and has_non_blocking):
        return []
    if valid is False:
        return ["semantic validator rejected the patch"]
    if valid is not True:
        return ["semantic validator must return valid=true or errors"]
    return []


_HARD_SEMANTIC_TYPES = {
    "unsupported_action",
    "runtime_boundary",
    "missing_dependency",
    "technology_dependency",
    "production_dependency",
    "internal_inconsistency",
    "supply_cap",
    "unexecutable",
}


def _hard_semantic_errors(payload: dict[str, Any]) -> list[str]:
    """Keep model review narrow; strategic quality belongs to match evaluation."""
    reported = payload.get("errors") or []
    if not isinstance(reported, list):
        reported = [reported]
    errors: list[str] = []
    for item in reported:
        if isinstance(item, str):
            message = item.strip()
            if message:
                errors.append(message)
            continue
        if not isinstance(item, dict):
            continue
        issue_type = str(item.get("type") or item.get("category") or "").strip().lower()
        issue_type = re.sub(r"[^a-z0-9]+", "_", issue_type).strip("_")
        severity = str(item.get("severity") or "blocking").strip().lower()
        if issue_type not in _HARD_SEMANTIC_TYPES or severity in {
            "non-blocking",
            "non_blocking",
            "advisory",
            "warning",
        }:
            continue
        _severity, message = _semantic_issue(item)
        if message:
            errors.append(message)
    return list(dict.fromkeys(errors))


def validate_strategy_patch_semantics(
    *,
    decision: dict[str, Any],
    parent_text: str,
    candidate_text: str,
    patches: list[dict[str, Any]],
    capability_manifest: dict[str, Any] | None = None,
    knowledge_runs: list[dict[str, Any]] | None = None,
    inheritance: dict[str, Any] | None = None,
    preservation_audit: list[dict[str, Any]] | None = None,
    prior_experiences: list[Any] | None = None,
    audit_output: dict[str, Any] | None = None,
    race: str = "terran",
    model: str = "",
) -> list[str]:
    capability_manifest = capability_manifest or {}
    errors = _runtime_boundary_errors(patches, candidate_text, capability_manifest)
    gate_execution_issue = _latest_gate_execution_issue(prior_experiences)
    if gate_execution_issue and any(
        str(patch.get("target") or "").strip() == "main_attack_gate"
        for patch in patches
        if isinstance(patch, dict)
    ):
        errors.append(
            "runtime_boundary — main_attack_gate — prior deterministic audit found "
            "that the declared gate was reached but Commander did not launch at "
            "repeated effective decision opportunities; keep the strategy gate "
            "unchanged until runtime execution is repaired"
        )
        return errors

    failed_history = [
        item
        for item in _experiment_history_items(prior_experiences)
        if str(item.get("decision") or "").strip().lower() != "accepted"
        and (
            str(item.get("hypothesis_verdict") or "").strip().lower()
            == "contradicted"
            or str(item.get("implementation_verdict") or "").strip().lower()
            in {"underpowered", "execution_invalid", "unknown"}
            or item.get("underpowered_retry_exhausted") is True
            or str(item.get("decision") or "").strip().lower()
            in {"rejected", "policy_rejected_before_matches"}
        )
    ]
    if failed_history:
        history_result = call_json_llm(
            build_mechanism_equivalence_prompt(
                decision=decision,
                parent_text=parent_text,
                candidate_text=candidate_text,
                patches=patches,
                prior_experiences=prior_experiences or [],
            ),
            model=str(model or "").strip() or DEFAULT_OPTIMIZATION_MODEL,
            is_reasoning=MECHANISM_HISTORY_ENABLE_REASONING,
        )
        history_audit, history_errors = _normalize_mechanism_equivalence_audit(
            history_result,
            prior_experiences=prior_experiences,
        )
        if audit_output is not None:
            audit_output["mechanism_equivalence_audit"] = (
                history_audit
                if history_audit
                else {
                    "status": "unavailable",
                    "warning": "semantic history judge returned no usable audit",
                }
            )
        # An unavailable judge must not stop a run. A positive equivalence verdict
        # is blocking because evaluating it would spend matches on a failed causal
        # direction that has no material repair.
        if history_audit:
            errors.extend(history_errors)
            if history_errors:
                return errors
    contact_timing_report: dict[str, Any] = {}
    if _timing_audit_required(decision) and str(race).strip().casefold() == "terran":
        extraction = call_json_llm(
            build_contact_timing_extraction_prompt(
                decision=decision,
                parent_text=parent_text,
                candidate_text=candidate_text,
                knowledge_runs=knowledge_runs,
            ),
            model=str(model or "").strip() or DEFAULT_OPTIMIZATION_MODEL,
            is_reasoning=CONTACT_TIMING_EXTRACTION_ENABLE_REASONING,
        )
        timing_model = (
            extraction.get("timing_model")
            if isinstance(extraction, dict)
            else None
        )
        canonical_parent = decision.get("parent_timing_package")
        if isinstance(timing_model, dict) and isinstance(canonical_parent, dict):
            timing_model = {
                **timing_model,
                "parent": dict(canonical_parent),
            }
        contact_timing_report = _build_contact_timing_report(
            timing_model,
            knowledge_runs,
        )
    elif _timing_audit_required(decision):
        contact_timing_report = {
            "complete": False,
            "skipped": True,
            "reason": "resource-aware first-commitment simulation currently supports Terran",
        }
    if audit_output is not None and contact_timing_report:
        audit_output["contact_timing_report"] = contact_timing_report
    result = call_json_llm(
        build_strategy_patch_validation_prompt(
            decision=decision,
            parent_text=parent_text,
            candidate_text=candidate_text,
            patches=patches,
            capability_manifest=capability_manifest,
            knowledge_runs=knowledge_runs,
            inheritance=inheritance,
            preservation_audit=preservation_audit,
            prior_experiences=prior_experiences,
            contact_timing_report=contact_timing_report,
        ),
        model=str(model or "").strip() or DEFAULT_OPTIMIZATION_MODEL,
        is_reasoning=STRATEGY_SEMANTIC_VALIDATION_ENABLE_REASONING,
    )
    if not isinstance(result, dict):
        if audit_output is not None:
            audit_output["hard_semantic_review"] = {
                "status": "unavailable",
                "warning": "hard semantic reviewer returned no JSON object",
            }
        return errors
    payload = result.get("validation") if isinstance(result.get("validation"), dict) else result
    if not isinstance(payload, dict):
        if audit_output is not None:
            audit_output["hard_semantic_review"] = {
                "status": "unavailable",
                "warning": "hard semantic reviewer returned no JSON object",
            }
        return errors
    if audit_output is not None:
        audit_output["hard_semantic_review"] = dict(payload)
    errors.extend(_hard_semantic_errors(payload))
    return errors


def validate_strategy_patch(
    *,
    decision: dict[str, Any],
    patches: list[dict[str, Any]],
    parent_document: StrategyDocument,
    capability_manifest: dict[str, Any] | None = None,
    parent_text: str = "",
    candidate_text: str = "",
    model: str = "",
    knowledge_runs: list[dict[str, Any]] | None = None,
    prior_experiences: list[Any] | None = None,
    preservation_audit: list[dict[str, Any]] | None = None,
    race: str = "terran",
) -> list[str]:
    errors = validate_strategy_patch_structure(
        decision=decision,
        patches=patches,
        parent_document=parent_document,
    )
    if errors or not candidate_text:
        return errors
    errors.extend(
        validate_strategy_patch_semantics(
            decision=decision,
            parent_text=parent_text or parent_document.render(),
            candidate_text=candidate_text,
            patches=patches,
            capability_manifest=capability_manifest or {},
            knowledge_runs=knowledge_runs,
            preservation_audit=preservation_audit,
            prior_experiences=prior_experiences,
            race=race,
            model=model,
        )
    )
    return errors


__all__ = [
    "build_mechanism_equivalence_prompt",
    "build_strategy_patch_validation_prompt",
    "validate_strategy_patch",
    "validate_strategy_patch_semantics",
    "validate_strategy_patch_structure",
]
