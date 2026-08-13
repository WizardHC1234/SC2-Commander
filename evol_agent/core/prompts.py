from __future__ import annotations

import json
from typing import Any

from .context import (
    render_battle_analysis,
    render_sc2_knowledge,
    render_single_game_analyses,
    render_skill_context,
)
from .types import BattleAnalysis, ToolObservation


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
) -> str:
    """Build the single cross-match analysis request used by EvolAgent."""
    errors = "\n".join(f"- {error}" for error in validation_errors[-3:]) or "None"
    knowledge_rule = (
        "Return five or six focused, non-overlapping knowledge_questions. Each "
        "must be linked to one or more candidate plans and ask static SC2 facts "
        "needed to compare or implement those plans."
        if knowledge_mode == "enabled"
        else "Knowledge is disabled; return an empty knowledge_questions list."
    )
    experience_lines: list[str] = []
    for item in (prior_experiences or [])[-3:]:
        if isinstance(item, dict):
            experience_lines.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
        else:
            experience_lines.append(str(item))
    experience_text = "\n".join(f"- {item}" for item in experience_lines) or "None"
    return f"""You are EvolAgent's batch Analysis Agent.

Read every independent factual match summary and produce several coherent candidate optimization plans. This is the only cross-match analysis call. Diagnose from recorded evidence; do not rewrite strategy.md and do not modify runtime code.

{RUNTIME_CONTRACT}

{CONTROLLABLE_OPTIMIZATION_SCOPE}

Strategy: {strategy_name}
Race: {race}

Current strategy.md:
{render_skill_context(skill_texts)}

Independent factual match summaries:
{render_single_game_analyses(single_game_analyses)}

Recent rejected-candidate experience:
{experience_text}

Rules:
- First identify how the current strategy wins and what successful behavior must remain intact.
- Treat detailed timed evidence as authoritative over a prose outcome summary. Before diagnosing, reconcile attack-order times, contact times, completed upgrades, completed structures, and living-unit counts across each summary; put unresolved contradictions in evidence_limits and do not use them as causal evidence.
- Rank problems and candidate plans by their likely effect on match outcomes. Do not optimize for a more complete build, broader capability coverage, or greater unit variety unless the match evidence connects it to winning.
- Identify one to five distinct strategy-fixable problems supported by repeated losses, stalls, bad trades, missed timings, resource conflicts, composition failures, scouting failures, or inability to finish. Do not collapse different causes into one vague problem.
- Trace every problem through current strategy rule -> Commander decision -> observed execution or progress -> later outcome. A requested command is not proof that the action succeeded; confirm execution from later game state.
- Reconstruct the strategy's force-readiness curve against actual enemy pressure. Across matches, identify the first enemy attack that materially threatened workers, production, technology, bases, or the main force; compare its time with the completed and gathered friendly force, completed upgrades, production capacity, unfinished investments, and losses at that moment. Use opponent Replay truth only for post-match diagnosis, never as information the Commander had during play.
- Test timing feasibility from the full dependency chain, not from final costs alone: resource availability -> prerequisite structures -> production structure or add-on -> build, research, and training durations -> completed units and upgrades -> time to gather. Account for sequential use of the same producer or research facility and for work that can proceed in parallel. Prefer observed completion progress from the records; use verified static durations and requirements only to fill missing facts, and do not invent exact completion times.
- A powerful final composition is not viable if its economy, technology, upgrades, production, or unit targets repeatedly leave the strategy unable to survive until that power stage. Diagnose whether the gap comes from excessive early investment, late production, late research, an unrealistic readiness target, or insufficient early fighting units.
- If the strategy already contains a clear executable rule but Commander did not follow it, report an execution limitation instead of proposing the same rule again.
- Produce two to five alternative, self-contained coherent candidate plans that the Optimization Agent can choose between. Each plan may contain multiple dependent deterministic changes across economy, production capacity, technology, composition, attack timing, reinforcement, or recovery when they jointly solve the same evidenced causal problem. Do not split one required causal chain into separate plans that would later need to be combined, and do not isolate a larger army threshold from the production, resource, supply, and timing changes needed to reach it.
- For every change in a plan, state the current rule, replacement rule, and why that change is required by the plan. Plans must be meaningfully different, not minor numerical variants or overlapping restatements.
- Make every plan reusable across matches. Do not encode one opponent, one recorded match, a map-specific zone ID, group ID, or an exact timestamp copied from the evidence.
- Macro changes must use executable absolute targets and must state priority or capacity changes when units, structures, add-ons, upgrades, or expansions compete for the same resources or producer.
- When combat strength or composition is a problem, compare strengthening the existing core through relevant upgrades, increasing its production or count, adding a support unit, and compatible combinations. Account for upgrade effects, prerequisites, research time, resource cost, research-facility contention, and whether the benefit arrives before the intended power stage; do not assume that adding another unit type is the best correction or that every upgrade should be researched.
- Every candidate plan must remain operational before its intended power stage. When repeated enemy pressure arrives earlier, include the fixed economy, production, technology, upgrade, unit-count, or readiness changes needed to field a survivable force before the evidenced pressure window; do not preserve an attractive late-game composition by assuming it will be allowed to finish.
- Reject a candidate plan whose required structures, add-ons, upgrades, and trained units cannot plausibly complete before its stated readiness or the evidenced enemy-pressure window. When the chain is too slow, change its priorities, capacity, technology depth, upgrade order, unit targets, or power-stage timing rather than merely asserting an earlier attack or defense.
- Static defenses are not a default way to make a strategy complete. Consider them only when repeated evidence shows that direct attacks on workers, production, technology, or a required position materially caused losses, and compare their resource and timing cost against investment in the strategy's force and win condition.
- Army readiness changes must count completed, living units gathered in the persistent main force. Do not treat requested production targets, units still training, or distant reinforcements as an attack-ready army.
- Do not propose a new policy branch whose effect depends on enemy scouting, scan results, detected defenses, or uncertain opponent state. For example, do not propose "if the enemy is weak, attack earlier" or "if the main is fortified, change objectives". A fixed readiness threshold is allowed.
- Preserve successful strategy content unless a plan explicitly needs a dependent change. Each plan must retain a credible path from economy and production to a survivable core force, attack, reinforcement, recovery, and destruction of remaining enemy bases. State its risk to the winning mechanism.
- Put runtime-only, micro, and unsupported problems in evidence_limits.
- Treat rejected-candidate experience as limited supporting evidence, not as a permanent prohibition or a substitute for the current match records.
- Compare candidate plans with rejected-candidate selected_changes. Do not repeat a materially equivalent failed change combination unless current evidence supports retrying it and the new plan states the substantive difference expected to avoid the previous regression.
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
        "strategy_fixable":true,
        "confidence":"low|medium|high"
      }}
    ],
    "candidate_plans": [
      {{
        "id":"D1",
        "name":"coherent deterministic plan",
        "addresses_problem_ids":["P1"],
        "changes":[
          {{
            "baseline_rule":"current fixed rule or parameter",
            "candidate_rule":"replacement fixed rule or parameter",
            "why_required":"why this dependent change is needed"
          }}
        ],
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
) -> str:
    """Build one candidate-generation request with only basic retry feedback."""
    errors = "\n".join(f"- {error}" for error in validation_errors[-3:]) or "None"
    candidate_text = "None"
    if isinstance(candidate, dict):
        files = candidate.get("files")
        if isinstance(files, dict):
            candidate_text = str(files.get("strategy.md") or "None")

    return f"""You are EvolAgent's Strategy Optimization Agent.

Write one complete candidate replacement for strategy.md. The batch Analysis Agent supplied several evidenced problems and coherent candidate plans. Use the analysis and verified knowledge as proposals and factual support, then make the final strategic judgment. Do not modify runtime code.

{RUNTIME_CONTRACT}

{CONTROLLABLE_OPTIMIZATION_SCOPE}

{STRATEGY_MARKDOWN_FORMAT}

Rules:
- The sole optimization objective is higher expected match win rate. Do not add units, upgrades, buildings, scouting rules, or safety clauses merely to make the strategy look comprehensive.
- Preserve strategy_contract.identity, core_commitments, and winning_mechanism unless the evidence clearly requires replacing one of them.
- Select one self-contained candidate plan by default. Combine a dependent change from another plan only when it is indispensable to make the selected plan executable or internally consistent; do not merge independent improvements merely because each is individually plausible. Every applied change must be listed in selected_changes with its source plan and evidence-supported role.
- Multiple dependent deterministic changes are allowed. Keep unrelated Detail bullets unchanged; do not bundle independent speculative improvements merely because they sound useful.
- Changes must be reusable across matches. Do not encode one opponent, one recorded match, map-specific zone IDs, group IDs, or exact timestamps copied from the evidence.
- Do not add an enemy-observation-conditioned branch: no new rule of the form "if/when scouting, scanning, detection, defenses, or enemy state says X, then change attack timing, objective, production, or priority." Use the fixed candidate_rule from the hypothesis instead. Fixed numerical readiness thresholds and fixed times are allowed.
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
- Keep the complete strategy internally consistent and keep explicit end-state supply at or below 200.
- After editing, ensure the complete strategy has a credible path through economy, production capacity, force preparation, engagement, reinforcement, recovery after losses, and eventual destruction of remaining enemy structures.
- Do not copy generic Commander runtime protocol into strategy.md. Write only strategy-specific targets, priorities, timings, readiness, objectives, reinforcement, recovery, and information requirements.
- Write ordinary StarCraft II strategy language and do not mention EvolAgent internals.

Knowledge mode: {knowledge_mode}
Strategy: {strategy_name}
Race: {race}

Current strategy.md:
{render_skill_context(skill_texts)}

Batch analysis and candidate plans:
{render_battle_analysis(battle_analysis)}

Verified knowledge results:
{render_sc2_knowledge(tool_observations)}

Current invalid candidate, if this is a retry:
{candidate_text}

Basic validator feedback:
{errors}

Return one JSON object only:
{{
  "action": "draft_candidate",
  "rationale": {{
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
  "files": {{
    "strategy.md": "# Summary\\n...\\n\\n# Details\\n* Opening: ...\\n* Main Attack Gate: ..."
  }}
}}

After basic validator feedback, use action="revise_candidate" with the same complete schema.
"""


__all__ = [
    "CONTROLLABLE_OPTIMIZATION_SCOPE",
    "RUNTIME_CONTRACT",
    "STRATEGY_MARKDOWN_FORMAT",
    "build_batch_analysis_prompt",
    "build_candidate_prompt",
    "build_fixed_match_summary_prompt",
]
