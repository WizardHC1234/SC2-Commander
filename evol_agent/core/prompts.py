from __future__ import annotations

import json
from typing import Any

from .context import (
    render_battle_analysis,
    render_sc2_knowledge,
    render_batch_match_evidence,
    render_skill_context,
)
from .types import BattleAnalysis, ToolObservation
from ..optimization.strategy_document import StrategyDocument


STRATEGY_MARKDOWN_FORMAT = """strategy.md format:
- Use exactly two level-one headings in this order: # Summary, # Details.
- # Summary is a short prose description of economy shape, core composition, power stage, and win plan.
- # Details contains only `* Title: ...` bullets. Preserve useful topic titles from the current strategy.
- Do not add Resource Costs or Required Tools sections. Runtime action metadata supplies costs, supply, duration, producers, prerequisites, and dependency closure.
- Write ordinary map-agnostic StarCraft II strategy language.
"""


RUNTIME_CONTRACT = """Current runtime contract:
- strategy.md is the only evolvable strategic-intent source. Runtime code under commander/ is fixed.
- One Commander receives the full structured observation and strategy, then emits absolute macro targets, group-level army commands, optional scan/scout requests, and one wake event.
- Macro targets execute concurrently and replace the previous target list. Every still-needed unmet target must be emitted again; ordering expresses resource priority.
- The runtime expands structure prerequisites and automatically adds build_gas when selected actions require gas. Action descriptions are authoritative for costs, supply, duration, producers, research locations, and prerequisites.
- The persistent main group is the operational force. group_1 contains newly produced units far from it and normally reinforces the main force or its current objective.
- Army movement is group-level and uses one semantic destination with an available movement mode. The Commander resolves semantic locations to observed zone IDs.
- At most one SCV scout is active. Scanner Sweep costs Orbital energy. Every decision has one observable wake condition plus a runtime fallback deadline.
- Python/Sharpy handles worker distribution, mining micro, repairs, pathfinding, formations, abilities, targeting, transport handling, transformations, and other unit-level micro.
"""


CONTROLLABLE_OPTIMIZATION_SCOPE = """Controllable strategy scope:
- Objective: improve expected match win rate across repeated games while preserving the strategy's defining style where it remains viable. Completeness, variety, and use of every available action are not objectives.
- Macro: workers, bases, gas, production, technology, upgrades, unit targets, supply, and resource priority.
- Army: gather/readiness conditions, semantic objective, movement intent, reinforcement, retreat, rebuild, and re-engagement.
- Information and redecision: scout/scan purpose, observable request conditions, required information, and meaningful wake checkpoints.
- Strategy identity defines what must be preserved; it is not an extra optimization category.
- Change only the smallest coherent area or combination supported by match evidence. Do not compensate for unavailable micro or runtime behavior in strategy.md.
"""


def build_fixed_match_summary_prompt(
    *,
    strategy_name: str,
    race: str,
    record_manifest: dict,
    match_timeline: str,
) -> str:
    """Build one factual summary prompt with a stable cacheable prefix.

    The factual-summary instructions are shared by every match in a batch.
    Strategy text is deliberately excluded: it belongs to cross-match
    diagnosis, while this stage must describe recorded evidence neutrally.
    Keep fixed instructions before record-specific metadata and timeline so
    providers with prefix caching can reuse the long prefix.
    """
    return f"""You summarize one StarCraft II match for EvolAgent.

The host has placed every Commander snapshot in chronological order in one fixed-schema table. Read every row. Summarize only recorded facts; do not diagnose causes or propose strategy changes.

Keep observed state, requested macro targets, execution progress, and army/recon/wake orders distinct. Keep completed, under-construction, training, and living units distinct. `enemy` is what Commander observed or remembered under fog of war. `opponent_truth_after_match`, when present, is post-match Replay truth that Commander did not know at decision time. For major engagements, compare the nearest pre-fight and post-fight rows. Put unsupported conclusions in evidence_limits.

Before returning, cross-check the summary against the timeline. Use one consistent timestamp for each event across all sections; distinguish the first attack order, first contact, and post-fight observation. For every strategically relevant upgrade, report requested, researching, and completed as different states and include the first observed completion time. Never claim that an upgrade was absent if any timeline row records it completed. If rows are insufficient or conflict, state the uncertainty instead of choosing a convenient interpretation. The metadata result and duration are authoritative for the final outcome.

Strategy: {strategy_name}
Race: {race}

Return one JSON object with exactly these top-level fields:
{{
  "outcome_summary": "concise chronological account",
  "timing_checkpoints": {{
    "first_friendly_attack_order": "time and order, or unknown",
    "first_major_contact": "time and observed forces, or unknown",
    "first_enemy_pressure_on_owned_zone": "time and threatened assets, or unknown",
    "first_retreat_or_major_loss": "time and before/after force, or unknown",
    "relevant_upgrade_completion": ["upgrade: first observed completion time"]
  }},
  "opening_and_economy": ["timed factual evidence"],
  "production_technology_and_composition": ["timed factual evidence"],
  "enemy_intelligence_and_map_state": ["fog-of-war and Replay-truth evidence kept separate"],
  "army_movement_and_engagements": ["group and pre/post-fight evidence"],
  "action_space_selection_summary": {{}},
  "commander_decision_summary": ["emitted targets/orders and issues"],
  "macro_execution_summary": ["requested targets and later progress"],
  "army_execution_summary": ["orders and later observed state"],
  "final_state": "result and final observed state",
  "evidence_limits": ["facts not established by snapshots"]
}}

Keep the lists concise while covering important transitions. Do not include an action wrapper.

Match-specific metadata:
{record_manifest}

Complete fixed match timeline:
{match_timeline}
"""


def build_batch_analysis_prompt(
    *,
    strategy_name: str,
    race: str,
    single_game_analyses: list[BattleAnalysis],
    skill_texts: dict[str, str],
    validation_errors: list[str],
    knowledge_mode: str,
    prior_experiences: list[Any] | None = None,
    capability_manifest: dict[str, Any] | None = None,
) -> str:
    """Build the single cross-match analysis request used by EvolAgent."""
    errors = "\n".join(f"- {error}" for error in validation_errors) or "None"
    knowledge_rule = (
        "Return zero to four focused, non-overlapping knowledge_questions. Ask "
        "only when a missing static SC2 fact can materially change plan selection "
        "or implementation; an empty list is preferred when supplied action "
        "metadata and match evidence are already sufficient."
        if knowledge_mode == "enabled"
        else "Knowledge is disabled; return an empty knowledge_questions list."
    )
    experience_lines: list[str] = []
    for item in prior_experiences or []:
        if isinstance(item, dict):
            keep = (
                "generation",
                "difficulty",
                "primary_change",
                "primary_lever",
                "hypothesis",
                "predictions",
                "disproof_conditions",
                "selected_changes",
                "parent_score",
                "candidate_score",
                "delta",
                "posterior_probability_better",
                "lesson",
            )
            compact_item = {key: item.get(key) for key in keep if key in item}
            evidence = item.get("experiment_evidence")
            if isinstance(evidence, dict):
                compact_evidence = {
                    key: evidence.get(key)
                    for key in (
                        "parent_batch",
                        "candidate_batch",
                        "candidate_minus_parent",
                    )
                    if key in evidence
                }
                for batch_key in ("parent_batch", "candidate_batch"):
                    batch = compact_evidence.get(batch_key)
                    if isinstance(batch, dict):
                        compact_evidence[batch_key] = {
                            key: batch.get(key)
                            for key in ("wins", "draws", "losses", "games", "score")
                            if key in batch
                        }
                compact_item["experiment_evidence"] = compact_evidence
            experience_lines.append(
                json.dumps(compact_item, ensure_ascii=False, separators=(",", ":"))
            )
        else:
            experience_lines.append(str(item))
    experience_text = "\n".join(f"- {item}" for item in experience_lines) or "None"
    strategy_document = StrategyDocument.parse(str(skill_texts.get("strategy.md") or ""))
    strategy_catalog = json.dumps(
        strategy_document.patch_context(), ensure_ascii=False, separators=(",", ":")
    )
    return f"""You are EvolAgent's batch Analysis Agent.

Read the deterministic match evidence, update the causal diagnosis, and choose the next action. This is the only cross-match reasoning call. Do not force a strategy candidate when the evidence instead requires more matches, runtime inspection, or stopping.

{RUNTIME_CONTRACT}

{CONTROLLABLE_OPTIMIZATION_SCOPE}

Strategy: {strategy_name}
Race: {race}

Current strategy paragraph catalog. Candidate changes must use these stable ids;
candidate_rule must be the complete replacement instruction after the bullet title:
{strategy_catalog}

Executor capability manifest:
{json.dumps(capability_manifest or {}, ensure_ascii=False, indent=2)}

Independent factual match summaries:
{render_batch_match_evidence(single_game_analyses)}

Recent rejected-candidate experience:
{experience_text}

Rules:
- Choose next_action before drafting a plan: propose_strategy_patch, request_more_matches, inspect_runtime, or stop. Explain the choice in action_reason.
- Treat matches classified runtime_contaminated as diagnostic evidence about Sharpy/Commander execution, not clean evidence for changing strategy parameters. runtime_suspect is a warning that requires corroboration from repeated behavior or hard errors; it does not by itself invalidate a match. If hard contamination dominates or explains the apparent failure, choose inspect_runtime and return no candidate plans.
- Choose request_more_matches when the current evidence cannot distinguish competing causes. Choose stop when no safe, testable, materially new lever remains.
- Expected match win rate is the sole optimization objective. Intermediate game facts may support diagnosis, but they are not secondary objectives and must not outweigh observed wins, draws, and losses.
- First identify how the current strategy wins and what successful behavior must remain intact.
- Before diagnosing the champion again, explain each recent rejected experiment using experiment_evidence when present: compare outcomes, duration, Commander decision counts, and selected-tool differences against its parent. Distinguish a prediction that was disproved from one that was never observable. Use this experimental result to eliminate or revise causal hypotheses.
- Treat detailed timed evidence as authoritative over a prose outcome summary. Before diagnosing, reconcile attack-order times, contact times, completed upgrades, completed structures, and living-unit counts across each summary; put unresolved contradictions in evidence_limits and do not use them as causal evidence.
- Rank problems and candidate plans by their likely effect on match outcomes. Do not optimize for a more complete build, broader capability coverage, or greater unit variety unless the match evidence connects it to winning.
- Do not infer causation from end-of-game survivor differences alone. For example, winning matches may have more bases or units because they survived longer; require evidence that the proposed difference occurred early enough to plausibly affect the result.
- Identify one to five distinct problems supported by repeated losses, stalls, bad trades, missed timings, resource conflicts, composition failures, scouting failures, or inability to finish. Classify each as strategy_fixable, commander_execution, runtime_execution, or observation_limited. Only strategy_fixable problems may drive candidate plans.
- Trace every problem through current strategy rule -> Commander decision -> observed execution or progress -> later outcome. A requested command is not proof that the action succeeded; confirm execution from later game state.
- Reconstruct the strategy's force-readiness curve against actual enemy pressure. Across matches, identify the first enemy attack that materially threatened workers, production, technology, bases, or the main force; compare its time with the completed and gathered friendly force, completed upgrades, production capacity, unfinished investments, and losses at that moment. Use opponent Replay truth only for post-match diagnosis, never as information the Commander had during play.
- Test timing feasibility from the full dependency chain, not from final costs alone: resource availability -> prerequisite structures -> production structure or add-on -> build, research, and training durations -> completed units and upgrades -> time to gather. Account for sequential use of the same producer or research facility and for work that can proceed in parallel. Prefer observed completion progress from the records; use verified static durations and requirements only to fill missing facts, and do not invent exact completion times.
- A powerful final composition is not viable if its economy, technology, upgrades, production, or unit targets repeatedly leave the strategy unable to survive until that power stage. Diagnose whether the gap comes from excessive early investment, late production, late research, an unrealistic readiness target, or insufficient early fighting units.
- If the strategy already contains a clear executable rule but Commander did not follow it, report an execution limitation instead of proposing the same rule again.
- Only when next_action is propose_strategy_patch, produce exactly one primary coherent candidate plan. It may contain multiple dependent deterministic changes when they jointly solve the same evidenced causal problem. Do not isolate a larger army threshold from the production, resource, supply, and timing changes needed to reach it.
- For every change in the plan, state the current rule, replacement rule, target paragraph id from strategy.md, and why that change is required.
- Unit composition and upgrade changes are first-class plans. Evaluate their expected combat value together with producer capacity, prerequisite timing, gas/mineral demand, supply, assembly time, and the runtime-owned behavior needed to realize that value; never reject a unit merely because some of its behavior is runtime-managed.
- Make every plan reusable across matches. Do not encode one opponent, one recorded match, a map-specific zone ID, group ID, or an exact timestamp copied from the evidence.
- Macro changes must use executable absolute targets and must state priority or capacity changes when units, structures, add-ons, upgrades, or expansions compete for the same resources or producer.
- When combat strength or composition is a problem, compare strengthening the existing core through relevant upgrades, increasing its production or count, adding a support unit, and compatible combinations. Account for upgrade effects, prerequisites, research time, resource cost, research-facility contention, and whether the benefit arrives before the intended power stage; do not assume that adding another unit type is the best correction or that every upgrade should be researched.
- Every candidate plan must remain operational before its intended power stage. When repeated enemy pressure arrives earlier, include the fixed economy, production, technology, upgrade, unit-count, or readiness changes needed to field a survivable force before the evidenced pressure window; do not preserve an attractive late-game composition by assuming it will be allowed to finish.
- Reject a candidate plan whose required structures, add-ons, upgrades, and trained units cannot plausibly complete before its stated readiness or the evidenced enemy-pressure window. When the chain is too slow, change its priorities, capacity, technology depth, upgrade order, unit targets, or power-stage timing rather than merely asserting an earlier attack or defense.
- Static defenses are not a default way to make a strategy complete. Consider them only when repeated evidence shows that direct attacks on workers, production, technology, or a required position materially caused losses, and compare their resource and timing cost against investment in the strategy's force and win condition.
- Army readiness changes must count completed, living units gathered in the persistent main force. Do not treat requested production targets, units still training, or distant reinforcements as an attack-ready army.
- At most one candidate plan may include a bounded information-conditioned branch. It must use information available in the current observation or an explicit scout/scan result, contain a deterministic default path when information is missing, and change only an executable macro priority, semantic objective, composition target, or readiness rule. Never use Replay truth as live information or build an open-ended condition tree.
- Preserve successful strategy content unless a plan explicitly needs a dependent change. Each plan must retain a credible path from economy and production to a survivable core force, attack, reinforcement, recovery, and destruction of remaining enemy bases. State its risk to the winning mechanism.
- Put runtime-only, micro, and unsupported problems in evidence_limits.
- Treat rejected-candidate scores as experimental evidence. A candidate that lost at least 0.10 score is a failed causal test, not merely a weak preference: its primary lever (for example, support-unit addition, upgrade-gated timing, or a recovery-threshold change) is on cooldown for the next candidate.
- A candidate is promoted only by the external win-rate evaluator. Do not claim that an intermediate prediction proves success; use it only to explain why a win-rate change might occur.
- Compare candidate plans with rejected-candidate selected_changes. The next candidate must test a materially different primary lever and root cause after a significant rejection. A smaller count, a modest timing adjustment, or the same unit/upgrade with a different gate is materially equivalent and is not a new test. Retry the same lever only when new current-match evidence directly contradicts the prior result; state that evidence and the substantive reason the regression should not recur.
- Knowledge questions go only to the deterministic SC2 entity dataset. Never ask it for optimal strategy, match timing, Commander behavior, movement modes, or micro.
- Every knowledge question must include the concrete match evidence that motivates it and explain which plan choice or implementation detail its answer can change. Use exact candidate entity names, do not ask facts already established by the strategy, action metadata, or supplied evidence, and preserve source time units.
- Choose needs precisely: requirements returns costs, supply, build/research time, producer, add-ons, and prerequisites; effects returns unit stats, weapons, abilities, energy, cooldown, and upgrade effects; counters returns counter relations; synergy returns synergy relations. Combine needs only when the same question genuinely requires several categories.
- Suitable queries cover missing costs, supply, build/research time, producers, add-on requirements, prerequisite chains, abilities, transformations, and deterministic counter or synergy relations. Do not ask generic encyclopedia attributes that cannot distinguish plans.
- For an evidenced combat-strength problem, use knowledge questions to compare only plausible upgrades and force alternatives that could change a candidate plan, including their effects and requirements. Do not spend questions cataloguing unrelated units, upgrades, or defensive structures.
- When plan feasibility depends on timing, ask for the missing prerequisite, producer, add-on, build, research, and training durations needed to evaluate the whole critical path; do not ask isolated duration facts that cannot change the plan.
- Do not repeat the same entities and fact needs across questions. One broader comparison question should replace overlapping questions such as separate and repeated Marine/Tank build-time queries.
- {knowledge_rule}

Feedback from an earlier malformed response:
{errors}

Return one JSON object only:
{{
  "action": "analyze_batch",
  "analysis": {{
    "next_action":"propose_strategy_patch|request_more_matches|inspect_runtime|stop",
    "action_reason":"why this is the highest-value next action",
    "strategy_contract": {{
      "identity": "economy shape, core force, power stage, posture, and win condition",
      "core_commitments": ["two to five defining commitments"],
      "optimization_boundary": "what may change without replacing the strategy",
      "direction": "preserve|adjust|replace"
    }},
    "winning_mechanism": "how successful matches were won",
    "wins_to_preserve": [
      {{"pattern":"successful behavior","evidence":["Match number and event"],"why":"observed benefit"}}
    ],
    "problems": [
      {{
        "problem_id":"P1",
        "problem":"distinct strategy problem",
        "evidence":["Match number and event"],
        "consequence":"later match effect",
        "control_class":"strategy_fixable|commander_execution|runtime_execution|observation_limited",
        "strategy_fixable":true,
        "confidence":"low|medium|high"
      }}
    ],
    "candidate_plans": [
      {{
        "id":"D1",
        "name":"coherent deterministic plan",
        "hypothesis":"one causal claim tested by this plan",
        "primary_lever":"composition|upgrade|production_capacity|economy|attack_timing|recovery|information_branch|other",
        "addresses_problem_ids":["P1"],
        "changes":[
          {{
            "target_paragraph_id":"production",
            "baseline_rule":"current fixed rule or parameter",
            "candidate_rule":"replacement fixed rule or parameter",
            "why_required":"why this dependent change is needed"
          }}
        ],
        "predictions":["observable result that should change in candidate matches"],
        "disproof_conditions":["observable result that rejects this hypothesis"],
        "capability_mapping":{{
          "macro_actions":["exact action name from capability manifest"],
          "army_controls":["exact army control from capability manifest"],
          "information_controls":[],
          "runtime_dependencies":["runtime-owned behavior used by the plan"],
          "unsupported_dependencies":[]
        }},
        "expected_benefit":"expected match effect",
        "risk_to_winning_mechanism":"possible regression"
      }}
    ],
    "knowledge_questions": [
      {{
        "id":"Q1",
        "plan_ids":["D1"],
        "evidence_motivation":"recorded event or missing fact motivating the query",
        "decision_use":"which plan choice or implementation detail the answer can change",
        "question":"focused static SC2 fact needed for these plans",
        "entities":["exact SC2 Unit or Upgrade name"],
        "needs":["effects|synergy|counters|requirements"]
      }}
    ],
    "evidence_limits":["important uncertainty"]
  }}
}}
"""


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
) -> str:
    """Build a bounded paragraph-patch request for the strategy document."""
    errors = "\n".join(f"- {error}" for error in validation_errors) or "None"
    parent_text = str(skill_texts.get("strategy.md") or "")
    document = StrategyDocument.parse(parent_text)
    patch_context = json.dumps(document.patch_context(), ensure_ascii=False, indent=2)
    candidate_text = json.dumps(candidate or {}, ensure_ascii=False, indent=2)

    if isinstance(candidate, dict) and validation_errors:
        return f"""You are revising an invalid EvolAgent strategy paragraph patch.

Do not redo the battle analysis and do not output a complete strategy.md. Fix only
the validator errors while preserving the same hypothesis, selected plan, and
unrelated parent paragraphs. Each target and expected_old_hash must match the
parent paragraph catalog.

Parent paragraph catalog:
{patch_context}

Invalid candidate:
{candidate_text}

Validator errors:
{errors}

Return one JSON object with action="revise_candidate", the complete rationale
object from the invalid candidate, and a non-empty operations list. Allowed
operations are replace_detail and replace_summary. Modify at most three Detail
paragraphs; Summary should normally remain unchanged.
"""

    return f"""You are EvolAgent's Strategy Optimization Agent.

Select one evidenced candidate plan and return a small paragraph patch for strategy.md. Do not rewrite the complete file and do not modify runtime code. The host applies the patch to the fixed template and preserves every unselected paragraph verbatim.

{RUNTIME_CONTRACT}

{CONTROLLABLE_OPTIMIZATION_SCOPE}

{STRATEGY_MARKDOWN_FORMAT}

Rules:
- The sole optimization objective is higher expected match win rate. Do not add units, upgrades, buildings, scouting rules, or safety clauses merely to make the strategy look comprehensive.
- The external evaluator promotes candidates only from accumulated wins, draws, and losses. Intermediate predictions are diagnostic hypotheses, not alternative acceptance criteria.
- Preserve strategy_contract.identity, core_commitments, and winning_mechanism unless the evidence clearly requires replacing one of them.
- Select exactly one self-contained candidate plan. Its dependent economy, production, and readiness changes must be included in that plan; do not combine plans. Every applied change must be listed in selected_changes with that single source plan and its evidence-supported role.
- Multiple dependent deterministic changes are allowed, but modify at most three Detail paragraphs and only when they implement the same causal plan. Keep unrelated paragraphs unchanged.
- Make the smallest complete text delta required by the selected plan. Preserve unchanged numeric targets, priorities, and Detail bullets verbatim whenever possible so the experiment tests one coherent cause.
- Changes must be reusable across matches. Do not encode one opponent, one recorded match, map-specific zone IDs, group IDs, or exact timestamps copied from the evidence.
- A bounded information-conditioned branch is allowed only when it is part of the selected plan, uses a current observation or explicit scout/scan result, has a deterministic default path, and maps to available controls. Do not use Replay truth or create an open-ended condition tree.
- Preserve pre-existing generic recovery, safety, or scouting conditions unless the selected plan explicitly changes them; do not turn them into new opponent-dependent strategic choices.
- Preserve successful power timing unless the evidence supports changing it. If army size, composition, or technology increases, jointly check the fixed economy, gas, production capacity, supply, and timing needed to reach it; do not raise an attack threshold in isolation.
- Check the candidate's survival path before its main power stage. If the records show earlier enemy pressure, ensure the revised fixed build priorities produce enough completed fighting units and required technology before that pressure window. Delay or reduce competing economy, technology, upgrades, support units, or final targets when necessary; do not rely on the enemy waiting for the final composition.
- Verify the candidate's critical path using recorded resource and completion progress plus verified prerequisites and durations. Include producer and research-facility contention, sequential versus parallel production, and gathering delay. A target being requested or affordable is not evidence that it will be completed and assembled in time.
- Prefer the cheapest coherent correction that strengthens the evidenced win plan. For combat weakness, consider applicable upgrades to the existing core alongside additional production, larger core counts, or support units; select fixed upgrade targets and priorities rather than adding all available upgrades.
- Add static defense only when the selected plan is supported by repeated evidence that losses of workers, production, technology, or a required position prevented the strategy from reaching or converting its winning position. Otherwise keep those resources focused on the economy, production, upgrades, and army used by the win plan.
- Prefer one coherent reusable plan over repeated warnings, match-specific patches, or many narrow exceptions.
- When a new target competes with the core plan for resources or production capacity, state the required priority and capacity change. Do not assume existing production can meet a larger force or faster timing.
- Express required macro outcomes as absolute targets. Attack-readiness counts must refer to completed, living units gathered in the persistent main force, not requested production targets, units still training, or distant reinforcements.
- Use verified knowledge only as factual support. It cannot prove an optimal ratio, timing, or win probability.
- Every instruction must be executable through current Macro, Army, scout, scan, and wake controls. Leave micro to the runtime.
- capability_mapping.macro_actions must list the executable actions needed by the complete candidate strategy, including preserved economy and production foundations. capability_mapping.changed_macro_actions must list only actions introduced or materially changed by the selected candidate plan. Do not compare or confuse these two sets.
- Keep the complete strategy internally consistent and keep explicit end-state supply at or below 200.
- After editing, ensure the complete strategy has a credible path through economy, production capacity, force preparation, engagement, reinforcement, recovery after losses, and eventual destruction of remaining enemy structures.
- Do not copy generic Commander runtime protocol into strategy.md. Write only strategy-specific targets, priorities, timings, readiness, objectives, reinforcement, recovery, and information requirements.
- Write ordinary StarCraft II strategy language and do not mention EvolAgent internals.

Knowledge mode: {knowledge_mode}
Strategy: {strategy_name}
Race: {race}

Executor capability manifest:
{json.dumps(capability_manifest or {}, ensure_ascii=False, indent=2)}

Current strategy paragraph catalog (stable id, parent hash, and value):
{patch_context}

Batch analysis and candidate plans:
{render_battle_analysis(battle_analysis)}

Verified knowledge results:
{render_sc2_knowledge(tool_observations)}

Return one JSON object only:
{{
  "action": "draft_candidate",
  "rationale": {{
    "hypothesis":"one causal claim tested by this candidate",
    "primary_lever":"one primary lever from the selected plan",
    "predictions":["observable candidate-match prediction"],
    "disproof_conditions":["observable condition that rejects the hypothesis"],
    "capability_mapping":{{
      "macro_actions":["exact available action names"],
      "changed_macro_actions":["exact available action names changed by the selected plan"],
      "army_controls":["exact available army controls"],
      "information_controls":[],
      "runtime_dependencies":[],
      "unsupported_dependencies":[]
    }},
    "preserved_strength":"winning mechanism kept intact",
    "selected_plan_ids":["D1"],
    "overall_assessment":"concise evidence-based assessment",
    "selected_changes":[
      {{"source_plan_id":"D1","problem_id":"P1","change":"specific deterministic correction","why":"evidence-supported role in the coherent plan"}}
    ],
    "primary_change":"coherent plan summary",
    "expected_effect":"expected match effect",
    "main_risk":"possible regression to evaluate in games"
  }},
  "operations": [
    {{
      "op":"replace_detail",
      "target":"main_attack_gate",
      "expected_old_hash":"copy the exact hash from the paragraph catalog",
      "value":"replacement instruction only; do not include the bullet title"
    }}
  ]
}}

Use replace_summary only when the selected plan genuinely changes the strategy identity. Do not add, delete, rename, or reorder Detail paragraphs.
"""


__all__ = [
    "CONTROLLABLE_OPTIMIZATION_SCOPE",
    "RUNTIME_CONTRACT",
    "STRATEGY_MARKDOWN_FORMAT",
    "build_batch_analysis_prompt",
    "build_candidate_prompt",
    "build_fixed_match_summary_prompt",
]
