from __future__ import annotations

import json
from typing import Any

from .context import (
    render_discovery_findings,
    render_knowledge_results,
    render_retrieval_evidence,
    render_single_game_analyses,
    render_skill_context,
)
from .optimizer_prompt import build_candidate_prompt
from .optimization_policy import OPTIMIZATION_POLICY
from .replay_reasoning_examples import REPLAY_GROUNDED_REASONING_EXAMPLES
from .strategy_patch_validator import build_strategy_patch_validation_prompt
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
- Summary describes the strategy identity and win mechanism; Details contains its executable strategic instructions. The first-attack gate is not a reusable recovery threshold.
- Every match has a hard 1800-second (30-minute) game limit. Victory requires destroying all enemy structures before that limit; winning one engagement or clearing known bases is not sufficient.
- One Commander receives the full structured observation and strategy, then emits absolute macro targets, group-level army commands, optional scan/scout requests, and one wake event.
- Macro targets execute concurrently and replace the previous target list. Every still-needed unmet target must be emitted again; ordering expresses resource priority.
- The runtime expands structure prerequisites and automatically adds build_gas when selected actions require gas. Action descriptions are authoritative for costs, supply, duration, producers, research locations, and prerequisites.
- The persistent main group is the operational force. group_1 contains newly produced units far from it and normally reinforces the main force or its current objective.
- Army-group membership is owned by the executor. The language strategy cannot create a custom detachment, reserve an exact number or type of units for a separate mission, keep such a detachment automatically replenished, or explicitly split, merge, and dissolve groups. It may give high-level hold, defend, regroup, or attack intent to groups that actually appear in the current observation; reinforcements join the persistent main group automatically when they reach it.
- Army movement is group-level and uses one semantic destination with an available movement mode. The Commander resolves semantic locations to observed zone IDs.
- At most one SCV scout is active. Scanner Sweep costs Orbital energy. Every decision has one observable wake condition plus a runtime fallback deadline.
- Observable enemy information includes currently visible contents and last_seen_enemy_contents with seconds_since_last_seen. Scan readiness is observable. Each Commander cycle is one decision; later wakes re-evaluate the same strategy after a scan or scout request.
- Do not require judging whether a Scanner Sweep is "safe", hidden opponent truth, or finishing a scan-then-commit sequence inside one decision frame.
- Python/Sharpy handles worker distribution, mining micro, repairs, pathfinding, formations, abilities, targeting, transport handling, transformations, and other unit-level micro.
"""


CONTROLLABLE_OPTIMIZATION_SCOPE = """Controllable strategy scope:
- Objective: improve expected match win rate across repeated games while preserving the strategy's defining style where it remains viable. Completeness, variety, and use of every available action are not objectives.
- Infer the strategy identity and win mechanism from Summary and the complete Details rather than from a hard-coded strategy profile.
- Evolvable strategy content includes economy and expansion, production and unit targets, technology and upgrades, composition, attack readiness and objective, and evidence-supported recovery behavior.
- Scouting and scans may support a named strategic decision or locate remaining enemy structures, but must not become an independent objective or a hidden attack gate. Wake events, route waypoints, unit-level micro, and runtime state machines remain executor concerns.
- Select the highest-priority evidence-supported failure mode using the shared SC2
  Strategic Priority below. Once selected, change every coherent dependency needed
  to produce a material combat-outcome difference; do not shrink the intervention
  merely to minimize edits. Do not compensate for unavailable micro or runtime
  behavior in strategy.md.

Strategic search guidance (not an exhaustive enum; one generation tests one primary causal hypothesis):
- Economy and expansion: worker/base/gas targets and observable expansion conditions, not clock-only expansion.
- Production capacity and resource priority: producer counts, when to scale or stop, and shared mineral/gas/producer/tech-lab ordering.
- Technology and upgrades: timing and whether an upgrade is required before a push; use verified SC2 facts for cost, duration, and prerequisites.
- Unit composition and support-unit balance: ratios and stage targets while keeping the defining army concept.
- Attack readiness, commitment threshold, and timing: observable living-force, upgrade, regroup, and intel gates rather than a clock as the only commit rule.
- Reinforcement and post-engagement behavior: preserve pressure or rebuild coherently according to the strategy style. The Main Attack Gate applies only to the first attack and must not be copied into Recovery and Cleanup.
- Defensive posture may describe where the currently observed operational force should gather or what threat should cause a new decision. It must not require a fixed-composition reserve squad, automatic reserve replenishment, or a later scripted merge, because those group-membership operations are not strategy controls.
"""


SC2_STRATEGIC_PRIORITY = """## SC2 Strategic Priority

The final objective is match victory. Diagnose strategy changes through the combat
chain that produces it, without using a fixed category ranking.

1. Infer the strategy's style, core win mechanism, and intended relative power
   window from strategy.md rather than its filename.
2. For each repeated decisive outcome, compare commitment and first-contact timing,
   the own and opponent packages at contact, force retained and reinforced after
   contact, and any observed production/resource/supply bottleneck.
3. Compare the opponent package at earlier observed states, actual contact, and a
   later state when available. A larger own force is not automatically better if
   waiting lets the opponent reach a stronger counter package.
4. Select the earliest shared strategy-fixable break in this chain. Composition,
   support, upgrades, timing, production, economy, recovery, and information are
   possible causes only when match evidence connects them to a better or more
   survivable engagement.

Use strict causal ordering. When the enemy arrives before the planned gate, first
determine why the gate was not available: production/resource delay, an excessive
gate, or a genuinely unavoidable pressure window. Do not infer "insufficient
defense" from pre-gate defeat alone. Static defense is eligible only when the
current production and commitment window are already viable and repeated evidence
shows that a small defense preserves them. When a gate was satisfied but the force
did not commit, classify the problem as runtime execution rather than raising,
lowering, or rewriting the strategy gate.

When pre-commitment defense is strategy-fixable, express it through executable economy, production, composition, technology, attack-readiness, or high-level posture changes for the observed operational group. Do not propose a fixed-composition reserve squad or any plan that requires the strategy to split, preserve, replenish, transfer, or later merge custom army membership.

Keep the stages distinct during diagnosis. Main Attack Gate controls only the first
planned commitment. Once that commitment occurs, evaluate Engagement and
Reinforcement and Recovery and Cleanup from post-contact evidence; never treat
changing the opening gate as automatic permission to rewrite recovery behavior.

If the planned package is reached but repeatedly collapses, do not merely produce
more of the same package. Check matchup, support, upgrades, engagement timing and
execution. Conversely, losing at the current or a later contact does not rule out
an earlier pressure window; compare the opponent's growth and successful earlier
counterexamples. If the first fight is viable but the army cannot continue, examine
reinforcement, production continuity, economy and recovery.

Support units and upgrades are not automatically first-attack prerequisites. Keep
them parallel with the attack by default. Make one a hard gate only when repeated
contact evidence and deterministic timing show that waiting improves relative
power after accounting for enemy growth and shared production/resource costs.

Scouting or scanning is useful only when the information changes a named strategic
decision or locates surviving enemy structures needed to finish the match. The
30-minute limit makes timely cleanup part of victory: analyze whether the strategy
leaves enough time to find and destroy all enemy bases and structures. Do not use
static defensive structures as the primary evolution mechanism.
"""


def build_fixed_match_summary_prompt(
    *,
    strategy_name: str,
    race: str,
    record_manifest: dict,
    match_timeline: str,
    audit_focus: dict[str, Any] | None = None,
) -> str:
    """Build one factual summary prompt with a stable cacheable prefix.

    The factual-summary instructions are shared by every match in a batch.
    Strategy text is deliberately excluded: it belongs to cross-match
    diagnosis, while this stage must describe recorded evidence neutrally.
    Keep fixed instructions before record-specific metadata and timeline so
    providers with prefix caching can reuse the long prefix.
    """
    focus = audit_focus if isinstance(audit_focus, dict) else {}
    focus_instructions = ""
    probe_schema = ""
    if focus:
        focus_instructions = f"""
This summary is also being used for a post-experiment mechanism audit. Inspect
the complete timeline specifically for the following pre-registered material
change. This does not authorize diagnosis: report only direct observations that
support, fail to show, or leave the material change unknown.

Audit focus:
{json.dumps(focus, ensure_ascii=False, indent=2)}

Populate mechanism_probe with timestamped facts. Use status=observed only when
the timeline directly shows the minimum material change. Use not_observed only
when the complete relevant window is present and the condition is absent. Use
unknown when sampling or missing fields prevent a determination. Do not infer
implementation from strategy text or final match result.
"""
        probe_schema = """,
  "mechanism_probe": {
    "status": "observed|not_observed|unknown",
    "observations": [
      {"time_s": 0, "fact": "directly recorded material-change evidence"}
    ],
    "evidence_limit": "missing field or sampling limitation, or empty"
  }"""

    return f"""You summarize one StarCraft II match for EvolAgent.

Read the complete match timeline from beginning to end. Compress it into a factual interaction timeline.

Select a small number of important recorded interactions, typically 8 to 15 when the match contains that much change. Prefer opening formation, clear economy or technology changes, first important enemy contact, main attack, major fights, retreat or regroup, base losses, and the end state. Do not copy every row.

Record only information explicitly present in the timeline. If a snapshot has army and economy but no buildings or technology, omit those fields. Do not invent, guess, or fill missing state.

Keep two enemy sources strictly separate:
- enemy_observed is what Commander could know at that decision from the live observation.
- enemy_truth is post-match Replay truth from opponent_truth_after_match when that row includes it. Commander did not know this during play.

Do not diagnose. Do not explain why something succeeded or failed. Do not recommend strategy changes. Do not judge a decision as good or bad. Do not write root causes. The metadata result and duration are authoritative for the match outcome.

Strategy: {strategy_name}
Race: {race}

In addition to the compressed timeline, extract two factual views when the
record supports them:
- enemy_pressure_events: enemy pressure on owned bases, production, workers, or
  a defending army. Record the observable cue separately from Replay truth.
- major_engagements: major army contacts, including both forces before/after the
  contact and whether the force held, broke through, withdrew, or was destroyed.
  When a runtime auto-retreat appears, record the force at the last pre-contact row,
  the configured retreat_ratio and local own/enemy power ratio at the trigger, and
  the force after retreat when available. State whether most losses happened before
  or after the trigger and whether the surviving force later regrouped or re-engaged.
  Never describe auto-retreat as the cause of the collapse when the recorded force
  had already collapsed before it fired.

Do not infer an attack merely because the opponent owned an army. If the record
does not establish pressure or a major engagement, return an empty list.
{focus_instructions}

Return one JSON object with exactly these top-level fields:
{{
  "result": "Victory|Defeat|Tie",
  "duration_s": 0,
  "events": [
    {{
      "time_s": 0,
      "trigger": "wake_event",
      "own_state": {{"economy": "...", "buildings": "...", "technology": "...", "army": "..."}},
      "enemy_observed": {{"army": "...", "buildings": "...", "intel": "..."}},
      "enemy_truth": {{"economy": "...", "buildings": "...", "technology": "...", "army": "..."}},
      "commands": ["recorded_action -> target"]
    }}
  ],
  "enemy_pressure_events": [
    {{
      "time_s": 0,
      "observed_cue": "recorded threat, visible enemy, defensive trigger, or loss change",
      "own_defense": "completed living defending force",
      "enemy_observed": "what Commander knew",
      "enemy_truth": "Replay-only composition when available",
      "outcome": "held|continued_pressure|army_broken|base_or_economy_damaged"
    }}
  ],
  "major_engagements": [
    {{
      "time_s": 0,
      "initiator": "own|enemy|unclear",
      "own_force_before": "recorded force",
      "enemy_observed": "what Commander knew",
      "enemy_truth": "Replay-only force when available",
      "own_force_after": "next recorded post-contact force",
      "own_reinforcement_after": "new combat units or regrouping recorded after contact",
      "production_context_before": "recorded producers, resources, supply block, or empty",
      "runtime_override": "auto-retreat trigger and force at trigger, or empty",
      "retreat_policy": "configured retreat_ratio, local power ratio at trigger, and later regroup/re-engagement, or empty",
      "loss_timing": "losses_before_override|override_before_losses|not_observed",
      "outcome": "breakthrough|held|withdrawal|army_broken|unclear"
    }}
  ]{probe_schema}
}}

Omit any event field that the corresponding timeline row does not support. Do not include an action wrapper.

Match-specific metadata:
{record_manifest}

Complete fixed match timeline:
{match_timeline}
"""


def _format_prior_experiences(prior_experiences: list[Any] | None) -> str:
    experience_lines: list[str] = []
    for item in prior_experiences or []:
        if isinstance(item, dict):
            if item.get("kind") == "parent_analysis_seed":
                analysis = (
                    dict(item.get("analysis") or {})
                    if isinstance(item.get("analysis"), dict)
                    else {}
                )
                compact_analysis = {
                    key: analysis.get(key)
                    for key in (
                        "record_mix",
                        "strategy_contract",
                        "repeated_failures",
                        "wins_to_preserve",
                        "cross_outcome_comparison",
                        "optimization_targets",
                        "priority_problem",
                        "hypothesis",
                        "mechanism_family",
                        "failure_mode_analysis",
                        "priority_alignment",
                        "intervention_package",
                        "next_action",
                        "action_reason",
                        "evidence_limits",
                    )
                    if key in analysis
                }
                for key in (
                    "repeated_failures",
                    "wins_to_preserve",
                    "cross_outcome_comparison",
                    "optimization_targets",
                    "evidence_limits",
                ):
                    if isinstance(compact_analysis.get(key), list):
                        compact_analysis[key] = compact_analysis[key][:6]
                experience_lines.append(
                    json.dumps(
                        {
                            "kind": "parent_analysis_seed",
                            "source_record_count": item.get("source_record_count"),
                            "current_record_count": item.get("current_record_count"),
                            "previous_analysis": compact_analysis,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                continue
            if item.get("kind") == "mechanism_search_policy":
                continue
            if item.get("kind") == "generation_retry_feedback":
                errors = [
                    str(error).strip()
                    for error in (item.get("errors") or [])
                    if str(error).strip()
                ]
                experience_lines.append(
                    json.dumps(
                        {"kind": "generation_retry_feedback", "errors": errors[-4:]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                continue
            keep = (
                "experiment_id",
                "generation",
                "difficulty",
                "mutation_parent",
                "comparison_champion",
                "candidate",
                "mechanism_family",
                "mechanism_prediction",
                "hypothesis",
                "implementation_verdict",
                "hypothesis_verdict",
                "runtime_findings",
                "decision",
                "comparison_champion_score",
                "candidate_score",
                "posterior_probability_better",
                "score_delta",
                "primary_change",
                "primary_lever",
                "lesson",
                "reason",
                "inheritance",
                "salvageable_changes",
                "failed_dependencies",
                "combat_evidence",
                "gate_execution_audit",
                "first_commitment_timing",
            )
            compact_item = {key: item.get(key) for key in keep if key in item}
            for key in ("hypothesis", "primary_change", "lesson"):
                if isinstance(compact_item.get(key), str):
                    compact_item[key] = compact_item[key][:1200]
            prediction = compact_item.get("mechanism_prediction")
            if isinstance(prediction, dict):
                compact_item["mechanism_prediction"] = {
                    key: prediction.get(key)
                    for key in (
                        "expected_change",
                        "minimum_material_change",
                        "outcome_prediction",
                        "disproof_condition",
                    )
                    if prediction.get(key) not in (None, "", [], {})
                }
            runtime_findings = compact_item.get("runtime_findings")
            if isinstance(runtime_findings, list):
                compact_item["runtime_findings"] = runtime_findings[:3]
            for list_key in (
                "salvageable_changes",
                "failed_dependencies",
                "combat_evidence",
            ):
                if isinstance(compact_item.get(list_key), list):
                    compact_item[list_key] = compact_item[list_key][:4]
            tested_changes: list[dict[str, str]] = []
            for patch in item.get("patches") or []:
                if not isinstance(patch, dict):
                    continue
                target = str(patch.get("target") or "").strip()
                replacement = str(
                    patch.get("replacement") or patch.get("value") or ""
                ).strip()
                if target or replacement:
                    tested_changes.append(
                        {
                            "target": target,
                            "replacement": replacement[:500],
                        }
                    )
                if len(tested_changes) >= 8:
                    break
            if tested_changes:
                compact_item["tested_changes"] = tested_changes
            evidence = item.get("experiment_evidence")
            if isinstance(evidence, dict):
                compact_evidence = {
                    key: evidence.get(key)
                    for key in (
                        "parent_batch",
                        "comparison_champion_batch",
                        "candidate_batch",
                        "candidate_minus_parent",
                        "candidate_minus_comparison_champion",
                    )
                    if key in evidence
                }
                for batch_key in (
                    "parent_batch",
                    "comparison_champion_batch",
                    "candidate_batch",
                ):
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
    return "\n".join(f"- {item}" for item in experience_lines) or "None"


def _cross_match_shared_context(
    *,
    strategy_name: str,
    race: str,
    skill_texts: dict[str, str],
    single_game_analyses: list[BattleAnalysis],
    prior_experiences: list[Any] | None,
) -> str:
    return f"""{RUNTIME_CONTRACT}

{CONTROLLABLE_OPTIMIZATION_SCOPE}

Strategy: {strategy_name}
Race: {race}

Current strategy.md:
{render_skill_context(skill_texts)}

Independent factual match summaries:
{render_single_game_analyses(single_game_analyses)}

Recent experiment history:
{_format_prior_experiences(prior_experiences)}

Experiment-history rules:
- score_delta and match outcomes are direct performance evidence; posterior is not a promotion gate.
- Accepted candidates contribute strengths. Rejected and inconclusive candidates contribute tested changes, execution evidence, and causal lessons, but their strategy text is not inherited.
- Compare the semantic direction, actual realization, contact timing, combat outcome, and score change. A failed direction is evidence, not a permanent ban; it may be repaired when the new version fixes a missing dependency or changes how the idea is realized.
- A parent_analysis_seed summarizes a subset of the current records and must not be counted as another match.
"""


def _compact_discovery_prompt(
    *,
    strategy_name: str,
    race: str,
    single_game_analyses: list[BattleAnalysis],
    skill_texts: dict[str, str],
    validation_errors: list[str],
    knowledge_mode: str,
    prior_experiences: list[Any] | None,
) -> str:
    knowledge_rule = (
        "Return one to three static SC2 knowledge questions whenever the diagnosis or a plausible strategy correction depends on costs, prerequisites, production time, upgrade effects, unit synergy, or counter relationships. When the proposed contact package changes units, upgrades, or timing relative to an observed enemy package, query the relevant counters, synergy, or effects before package selection. Return zero only when no static SC2 fact can affect the decision. Every question requires explicit entities and needs."
        if knowledge_mode == "enabled"
        else "Knowledge lookup is disabled; return no questions."
    )
    errors = "\n".join(f"- {error}" for error in validation_errors) or "None"
    return f"""You are EvolAgent's Cross-Match Discovery Agent and compact evidence analyst. Produce a factual cross-match digest before strategy optimization. Compare wins and losses, identify the strategy's core style and power window, distinguish strategy defects from runtime execution defects, and focus on contact timing, both armies at contact, production feasibility, and post-contact continuation. Do not choose the final optimization direction and do not write replacement strategy text. Use evidence references in the exact form `Game N @ Ts: observation` so deterministic record retrieval can resolve them. Do not return next_action, candidate_plans, candidate_rule, or target_paragraph_id.

{OPTIMIZATION_POLICY}

{knowledge_rule}

Previous schema errors:
{errors}

{_cross_match_shared_context(strategy_name=strategy_name, race=race, skill_texts=skill_texts, single_game_analyses=single_game_analyses, prior_experiences=prior_experiences)}

Return one JSON object only:
{{
  "strategy_contract":{{"style":"combat style","core_win_mechanism":"how the strategy wins","critical_power_window":"intended contact stage","core_commitments":["essential behavior"],"flexible_components":["adjustable behavior"]}},
  "outcome_contrast":{{"winning_pattern":"repeated successful behavior","winning_evidence":["Game evidence"],"loss_shortfall":"repeated shortfall","loss_evidence":["Game evidence"],"loss_relationship_to_wins":"same_mechanism_underperforms|mechanism_not_reached|different_mechanism|uncertain","causal_difference":"most useful contrast","preservation_rule":"what the candidate should retain"}},
  "strengths":[{{"pattern":"behavior supported by wins","evidence":["Game N @ Ts: observation"]}}],
  "weaknesses":[{{"pattern":"repeated strategy-fixable or runtime shortfall","evidence":["Game N @ Ts: observation"],"confidence":"low|medium|high"}}],
  "unknowns":[{{"unknown":"important uncertainty","why_it_matters":"decision impact","evidence":["Game N @ Ts: observation"]}}],
  "opponent_pressure_patterns":[{{"pattern":"pressure timing and package","evidence":["Game N @ Ts: observation"]}}],
  "matchup_patterns":[{{"pattern":"own and enemy packages at decisive contact","evidence":["Game N @ Ts: observation"]}}],
  "query_plan":{{"match_evidence_queries":[{{"query_reason":"recorded interaction to verify","evidence_refs":["Game N @ Ts: observation"]}}],"experience_query":{{"query_reason":"related historical change to retrieve","failure_signature":["observed failure pattern"]}},"game_knowledge_queries":[{{"id":"Q1","question":"Using the bundled SC2 dataset tools as the source of truth, ...","entities":["exact unit, structure, or upgrade name"],"needs":["requirements|effects|synergy|counters"],"query_reason":"why this fact changes the decision","evidence_refs":["Game N @ Ts: observation"],"hypothesis_scope":"bounded factual dependency","calculations":[]}}]}}
}}
"""


def build_cross_match_discovery_prompt(
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
    """Round 1: understand evidence before proposing any strategy change."""
    return _compact_discovery_prompt(
        strategy_name=strategy_name,
        race=race,
        single_game_analyses=single_game_analyses,
        skill_texts=skill_texts,
        validation_errors=validation_errors,
        knowledge_mode=knowledge_mode,
        prior_experiences=prior_experiences,
    )
    capability_text = json.dumps(capability_manifest or {}, ensure_ascii=False, indent=2)
    errors = "\n".join(f"- {error}" for error in validation_errors) or "None"
    knowledge_rule = (
        "When knowledge_mode is enabled, ask a knowledge question only if a missing "
        "static SC2 fact would distinguish competing explanations (for example "
        "production bottleneck vs composition/matchup, composition vs upgrade, or "
        "strategy vs runtime). Allowed needs include effects, counters, synergy, "
        "and requirements (including upgrade/ability effects and prerequisites). "
        "Do not ask which strategy to use, why a named game was lost, or whether "
        "to build a specific unit. An empty list is valid when match evidence "
        "already ranks the explanations."
        if knowledge_mode == "enabled"
        else "Knowledge is disabled; return an empty knowledge_questions list."
    )
    return f"""You are EvolAgent's Cross-Match Discovery Agent.

You are given:
- the current strategy.md
- factual summaries from multiple matches
- recent rejected experiments when available
- the runtime/executor contract needed to understand what strategy.md can control

Your job in this round is to understand the evidence before proposing any strategy change.

Do not propose a patch.
Do not write replacement strategy rules.
Do not choose a target paragraph.
Do not force a causal hypothesis before the evidence supports one.
Do not return next_action, candidate_plans, candidate_rule, or target_paragraph_id.

Analyze across matches, not one match in isolation.

Executor capability manifest:
{capability_text}

{_cross_match_shared_context(
    strategy_name=strategy_name,
    race=race,
    skill_texts=skill_texts,
    single_game_analyses=single_game_analyses,
    prior_experiences=prior_experiences,
)}

Identify:

1. Strategy contract
Before ranking problems, infer the current strategy's gameplay style, core win
mechanism, critical relative-power window, commitments that define its identity,
and components that may be adjusted. This contract must come from the supplied
strategy text, not from a fixed profile or the strategy filename. Then ground that
intent in successful trajectories: identify the repeated sequence that actually
created advantage in wins and protect the parts of that sequence that the losses do
not disprove.

Before listing independent strengths and weaknesses, perform one direct outcome
contrast. Ask whether losses failed because the winning mechanism was not reproduced
or because the same mechanism was reproduced and still proved insufficient. Do not
replace a validated winning mechanism merely to patch the final enemy composition in
a small number of losses. Preserve the earliest causal difference between wins and
losses, not only their final states.

2. Strengths
Repeated behaviors or strategy mechanisms that appear to contribute to successful games
and should be preserved.

3. Weaknesses
Repeated patterns associated with losses, stalls, bad trades, missed timings,
poor information, or failure to finish. Do not yet name a root-cause hypothesis.

4. Unknowns
Questions that cannot be answered reliably from the match evidence alone. These must
be static SC2 mechanism facts, not "which strategy is best" or "why this game was lost".

5. Opponent pressure patterns
Compare when and how opponents pressure owned bases, economy, production, or the
defending army. Record the observable cues, the own defensive package available at
contact, whether the strategy survived until its intended power spike, and concrete
counterexamples. Do not fit a rule to one exact timestamp or one opponent build.

6. Matchup and composition patterns
Compare own and enemy compositions at major engagements across wins and losses.
Separate upgrade differences from unit counters, support balance, defender advantage,
engagement conditions, and execution. If a counter, synergy, or upgrade effect is
needed to distinguish explanations, request a bounded knowledge query.

7. Retrieval plan
Create a bounded, evidence-linked query plan before diagnosis:
- First aggregate opponent pressure across the whole batch: first pressure/contact
  time, observable pressure cues, enemy composition at contact, own completed
  defensive package, whether the intended power spike was reached, and survival
  outcome. Report coverage as match counts and a time range; do not generalize
  from one dramatic loss.
- match_evidence_queries must name why exact recorded rows need to be checked and cite existing `Game N @ Ts` references. Use them especially to verify who initiated an engagement, whether pressure reached an owned zone, what command preceded contact, and whether a strategy gate was already satisfied.
- experience_query must describe the observed failure signature, not a strategy filename or a preferred replacement. It will retrieve both successful and failed prior interventions.
- game_knowledge_queries may request only static SC2 facts needed to distinguish the supported interpretations. Each question must state why it is needed, the match references that motivated it, and the bounded hypothesis scope.
- When production throughput or continuous resource demand matters, add deterministic calculations instead of estimating arithmetic in prose. Use parallel_production with action, quantity, and production_slots, or resource_demand_per_minute with action and production_slots.
- Match every requested fact to the query schema: costs, build or research times, producers, and prerequisites require `requirements`; unit or upgrade behavior requires `effects`; matchup claims require `counters`; production arithmetic requires a deterministic calculation.
- Do not ask the static database how many units will win a dynamic engagement, whether a build is optimal, or what strategy should be chosen. Use recorded same-time army evidence for those judgments.
- Query reasons must describe an uncertainty, not assert an unverified capability. For example, do not claim that a support unit provides anti-air before the database verifies that effect.

Use concrete evidence such as:
"Game 3 @ 420s: ..."

Rules:
- Compare wins and losses whenever possible.
- Every opponent_pressure_pattern and matchup_pattern must state how many matches
  support it and include at least one counterexample when one exists.
- Do not infer causation from final-state differences alone.
- A requested Commander action is not proof that execution succeeded.
- Distinguish Commander-observed enemy information from post-match enemy truth.
- Do not invent missing game facts.
- If a weakness may be caused by Commander execution or runtime execution, say so rather than assuming strategy.md is at fault.
- Trace every important failure through strategy rule -> Commander decision ->
  applied command -> later game state. If the strategy already gave a clear rule
  and the downstream behavior failed, classify it as an execution limit instead
  of recommending a more verbose strategy rule.
- Keep army-fight evidence primary. Earlier attacks, lower resource banks, more
  production, or more scouting are not strengths by themselves unless they create
  a more favorable or more survivable decisive engagement.
- For every repeated decisive failure, compare the evidence available for four
  linked facts: contact timing relative to both sides' power growth, own and enemy
  combat packages at contact, force retained and reinforced after contact, and any
  observed producer/resource/supply bottleneck. These are evidence checks, not a
  fixed optimization-category selector. Do not infer missing values.
- Treat retreat_ratio as part of the engagement policy, not as a standalone score.
  Compare its configured value with the local power ratio at the trigger, losses
  before and after the trigger, surviving main-force power, and time to regroup or
  re-engage. A higher or lower ratio is useful only when that complete sequence
  improves the strategy's intended pressure, survival, or repeated fighting.
- Use rejected-experiment evidence to avoid treating an already-failed lever as a fresh unexplained weakness.
- {knowledge_rule}
- Never ask the knowledge database for optimal strategy, match diagnosis, Commander behavior, movement behavior, or micro.

Feedback from an earlier malformed response:
{errors}

Return one JSON object only:
{{
  "action": "discover_batch",
  "analysis": {{
    "strategy_contract":{{
      "identity":"short description of the current strategy",
      "style":"gameplay style inferred from the strategy text",
      "core_win_mechanism":"how this strategy is intended to create a winning advantage",
      "critical_power_window":"the relative timing or completed package on which that mechanism depends",
      "observed_winning_signature":"repeated causal sequence observed in successful games",
      "winning_evidence":["Game 1 @ 420s: evidence for that sequence"],
      "core_commitments":["defining commitment that should normally be preserved"],
      "protected_invariants":["winning condition that a candidate must retain unless directly disproved"],
      "flexible_components":["component that evidence may justify changing"],
      "optimization_boundary":"what must not be changed into another strategy family",
      "direction":"preserve|adjust|replace"
    }},
    "outcome_contrast":{{
      "winning_pattern":"what repeatedly works in wins",
      "winning_evidence":["Game 1 @ 420s: ..."],
      "loss_shortfall":"earliest important deviation or insufficiency in losses",
      "loss_evidence":["Game 2 @ 390s: ..."],
      "loss_relationship_to_wins":"winning_mechanism_not_reproduced|winning_mechanism_reproduced_but_failed|mixed|uncertain",
      "causal_difference":"why the compared trajectories diverge",
      "preservation_rule":"what later optimization must retain while fixing the loss shortfall"
    }},
    "strengths":[
      {{"pattern":"repeated successful behavior","evidence":["Game 1 @ 420s: ..."]}}
    ],
    "weaknesses":[
      {{"pattern":"repeated problematic behavior","evidence":["Game 2 @ 390s: ..."],"confidence":"medium"}}
    ],
    "unknowns":[
      {{
        "unknown":"what cannot be determined from match evidence",
        "why_it_matters":"why resolving this could change interpretation",
        "evidence":["Game 2 @ 390s: ..."]
      }}
    ],
    "opponent_pressure_patterns":[
      {{"pattern":"repeated pressure pattern and survival outcome","evidence":["Game 2 @ 420s: ..."],"confidence":"medium"}}
    ],
    "matchup_patterns":[
      {{"pattern":"composition or engagement relationship across outcomes","evidence":["Game 3 @ 620s: ...","Game 7 @ 670s: ..."],"confidence":"medium"}}
    ],
    "query_plan":{{
      "match_evidence_queries":[
        {{
          "query_reason":"which attribution or interaction must be verified",
          "evidence_refs":["Game 2 @ 390s: ..."]
        }}
      ],
      "experience_query":{{
        "query_reason":"why analogous prior interventions are relevant",
        "failure_signature":["game phase","pressure pattern","own combat state","matchup context"]
      }},
      "game_knowledge_queries":[
        {{
          "question":"static SC2 factual question",
          "entities":["relevant unit, structure, upgrade, or ability"],
          "needs":["requirements"],
          "query_reason":"which competing interpretation this fact distinguishes",
          "evidence_refs":["Game 2 @ 390s: ..."],
          "hypothesis_scope":"bounded factual relationship, not a strategy recommendation",
          "calculations":[
            {{"type":"parallel_production","action":"train_core_unit","quantity":10,"production_slots":2}},
            {{"type":"resource_demand_per_minute","action":"train_core_unit","production_slots":2}}
          ]
        }}
      ]
    }}
  }}
}}
"""


def build_optimization_package_prompt(
    *,
    strategy_name: str,
    race: str,
    single_game_analyses: list[BattleAnalysis],
    skill_texts: dict[str, str],
    validation_errors: list[str],
    prior_experiences: list[Any] | None,
    discovery: dict[str, Any] | None,
    knowledge_runs: list[dict[str, Any]] | None,
    retrieval_evidence: dict[str, Any] | None,
    parent_timing_package: dict[str, Any] | None = None,
    capability_manifest: dict[str, Any] | None = None,
) -> str:
    errors = "\n".join(f"- {error}" for error in validation_errors) or "None"
    return f"""You are EvolAgent's Optimization-Package Planner. Convert the cross-match evidence into two or three genuinely different, evidence-supported optimization hypotheses. Each hypothesis must be a coherent package rather than one isolated edit. Do not select the winner and do not write strategy.md yet.

{OPTIMIZATION_POLICY}

The deterministic build-order simulator will evaluate every package before selection. The parent package below was extracted separately from the current strategy and is read-only. Do not reproduce, reinterpret, or replace it. Each candidate package must describe the complete first-commitment production package, including unchanged Champion components that remain required. Use exact runtime action identifiers from the supplied strategy and action metadata (`train_*`, `build_*`, and `research_*`). A time budget is not an LLM estimate: provide the evidence-derived latest useful first-commitment bound and allowed delay, while the program calculates earliest_feasible_time_seconds, resources, prerequisites, supply, and production queues. For each package, compare the intended own force with the enemy compositions recorded near the same time, including unit counters, upgrades, defender advantage, reinforcements, and whether continued production can sustain the attack. Do not assume that a larger but later own army is stronger after the opponent's growth.

Every proposed package must be implementable through the current strategy controls. The executor owns membership of the persistent main group and reinforcement group. Reject before returning any hypothesis that requires a fixed-composition reserve or defensive squad, exact unit reservation for a separate mission, automatic subgroup replenishment, or scripted split, transfer, merge, or dissolution. A defensive hypothesis must instead use executable macro targets, composition, technology, attack readiness, or high-level posture for groups that actually appear in observations.

Read-only parent first-commitment package:
{json.dumps(parent_timing_package or {}, ensure_ascii=False, separators=(',', ':'))}

Previous schema errors:
{errors}

Current strategy and match summaries:
{_cross_match_shared_context(strategy_name=strategy_name, race=race, skill_texts=skill_texts, single_game_analyses=single_game_analyses, prior_experiences=prior_experiences)}

Evidence digest:
{render_discovery_findings(discovery or {})}

Verified SC2 knowledge:
{render_knowledge_results(knowledge_runs or [])}

Retrieved match/history evidence:
{render_retrieval_evidence(retrieval_evidence or {})}

Runtime action metadata:
{json.dumps(capability_manifest or {}, ensure_ascii=False, separators=(',', ':'))}

Return JSON only. Use next_action=evaluate_candidate_packages whenever at least two strategy-fixable hypotheses can be tested. Use inspect_runtime only when strategy.md cannot address the supported failure:
{{
  "strengths_to_preserve":[{{"pattern":"successful behavior","evidence":"match reference"}}],
  "priority_problem":{{"problem":"earliest important shortfall","evidence":["at least one match reference"],"control_class":"strategy_fixable|commander_execution|runtime_execution|observation_limited","confidence":"low|medium|high","consequence":"combat or match consequence"}},
  "failure_mode_analysis":{{"failure_stage":"before_core_mechanism|during_commitment_or_engagement|after_successful_engagement|mixed","gate_attainment_and_launch":"whether the intended attack became available and launched","commitment_and_contact_timing":"actual timing contrast","own_package_at_contact":"own army and upgrades","opponent_package_and_growth":"enemy army and growth","post_contact_continuity":"reinforcement and continued attack","production_feasibility":"income, queues, upgrades, and bottlenecks","optimization_implication":"concise implication","covered_failures":["match evidence"],"counterexamples":["conflicting or winning evidence"]}},
  "candidate_packages":[
    {{
      "id":"P1",
      "hypothesis":"causal hypothesis distinguished from the other packages",
      "plan":{{"direction":"coherent optimization direction","material_behavior_change":"observable intended change","coordinated_changes":[{{"change":"related strategy change","why_required":"causal role"}}],"preserve":["Champion behavior to retain"],"contact_window_effect":"earlier|similar|later|unknown"}},
      "timing_budget":{{
        "target_latest_first_commitment_seconds":300,
        "maximum_added_feasibility_seconds":20,
        "budget_basis":["Game N @ Ts: why this window matters"],
        "package":{{"economy":{{"worker_target_before_commitment":null,"base_target_before_commitment":null,"gas_workers_before_commitment":null}},"gate_components":[],"setup_actions":[]}}
      }},
      "engagement_assessment":{{"intended_contact_window":"relative window supported by records","own_package_role":"how the own package fights","observed_opponent_package":"enemy composition and growth near that window","counter_and_upgrade_relationship":"relevant counter, support, and upgrade relationship from evidence and DataAgent knowledge","reinforcement_and_continuity":"whether production, retained force, and reinforcements can sustain pressure after contact"}},
      "expected_effect":"expected contact, combat, continuation, or victory change",
      "main_risk":"most important regression risk"
    }}
  ],
  "next_action":"evaluate_candidate_packages|inspect_runtime",
  "action_reason":"short evidence-based reason",
  "evidence_limits":["uncertainty or runtime issue"]
}}
"""


def build_parent_timing_package_prompt(
    *,
    strategy_name: str,
    race: str,
    strategy_text: str,
    validation_errors: list[str],
    capability_manifest: dict[str, Any] | None = None,
) -> str:
    """Extract the Champion's first-commitment package without optimization context."""
    errors = "\n".join(f"- {error}" for error in validation_errors) or "None"
    return f"""You are EvolAgent's Parent Strategy Package Extractor. Perform only a faithful extraction from the supplied current strategy.md. Do not analyze match results, propose improvements, lower or raise targets, or use the strategy filename as a profile. Do not calculate time, costs, income, or resource feasibility.

Extract the complete package that the current strategy establishes for its first meaningful offensive commitment. Read the strategy chronologically across all bullets instead of looking only at the Main Attack Gate. An economy, expansion, production, or technology requirement belongs to the package when the strategy establishes or maintains it before the main attack, even when that requirement is written in a separate section and does not literally say "before the attack." For example, a base count held until the attack, a worker target pursued during the opening, and explicitly declared producer, add-on, refinery, or required-upgrade counts used to create the gate force are pre-commitment requirements. Do not include a third base explicitly started after the attack, ultimate army goals, reinforcement targets, recovery thresholds, or cleanup behavior.

Use exact runtime action identifiers from the action metadata. Gate components are the units or upgrades whose completion directly opens the first commitment. Setup actions are the strategy's explicit absolute quantities of production structures, add-ons, gas structures, expansions, supply infrastructure, and required upgrades established for that commitment. Include declared producer capacity even though the simulator could infer a minimal prerequisite automatically. Production slots must reflect the strategy's declared producer and add-on capacity. Economy fields contain explicit pre-commitment worker, base, and gas-worker targets; use null only when the strategy truly gives no such target. A positive refinery count belongs in setup_actions as build_gas; do not invent a gas-worker count from it.

Zero means absence. If the strategy says mineral-only, build 0 Refineries, no gas, or otherwise forbids Refineries before the attack, do not emit setup_actions.build_gas and set economy.gas_workers_before_commitment to null. In requirement_coverage, never map such a bullet to setup_actions.build_gas or economy.gas_workers_before_commitment. Map only fields that actually appear in parent_timing_package (for example worker_target_before_commitment), or classify the bullet as behavioral_pre_commitment when it only states that gas is unused.

Every non-null economy field and every gate or setup item must include a verbatim strategy_excerpt supporting the extraction. The excerpt must occur in the supplied strategy.md. Do not copy illustrative values because this prompt contains no example strategy.

production_slots and parallel_slots must be positive integers (>= 1) whenever quantity >= 1. Never emit 0. Never emit a dict of building counts as production_slots; put declared producer or add-on counts in setup_actions instead, and set production_slots to the integer queue capacity used for that gate unit.

Review every Markdown bullet in strategy.md. Return one requirement_coverage entry for every bullet, copying the complete bullet verbatim. Classify it as mapped_pre_commitment when its quantitative timing requirements are represented in the package, behavioral_pre_commitment when it affects behavior but has no quantity represented by this simulator, post_commitment when it applies only after the first commitment, or mixed when it contains both pre- and post-commitment content. Scouting, Scanner Sweep behavior, and pre-attack army posture without numeric simulator fields are behavioral_pre_commitment (or post_commitment for after-attack parts)—never mixed with an empty mapped_to. mapped_to must name only package fields or actions that were actually extracted for that bullet, using economy.<field>, gate_components.<action>, or setup_actions.<action>. Do not invent mapped_to names that are absent from parent_timing_package. A mapped_pre_commitment or mixed entry must have at least one valid mapped_to item. Explicit positive numeric economy, base, refinery, producer, add-on, upgrade, or gate-unit requirements that apply before the attack must not be classified as merely behavioral.

Strategy name: {strategy_name}
Race: {race}

Previous extraction errors:
{errors}

Current strategy.md:
{strategy_text}

Runtime action metadata:
{json.dumps(capability_manifest or {}, ensure_ascii=False, separators=(',', ':'))}

Return JSON only:
{{
  "parent_timing_package":{{
    "economy":{{
      "worker_target_before_commitment":null,
      "base_target_before_commitment":null,
      "gas_workers_before_commitment":null,
      "evidence":{{}}
    }},
    "gate_components":[],
    "setup_actions":[]
  }},
  "requirement_coverage":[{{"strategy_excerpt":"complete Markdown bullet copied verbatim","classification":"mapped_pre_commitment|behavioral_pre_commitment|post_commitment|mixed","mapped_to":["economy.worker_target_before_commitment","setup_actions.build_barracks","gate_components.train_unit"],"reason":"brief chronological classification reason"}}]
}}

Each gate_components item must contain action, quantity, production_slots, and strategy_excerpt. Each setup_actions item must contain action, quantity, parallel_slots, and strategy_excerpt. For every non-null economy field, economy.evidence must contain the same field name mapped to its verbatim strategy excerpt.
"""


def _compact_decision_prompt(
    *,
    strategy_name: str,
    race: str,
    validation_errors: list[str],
    discovery: dict[str, Any] | None,
    knowledge_runs: list[dict[str, Any]] | None,
    retrieval_evidence: dict[str, Any] | None,
    candidate_package_payload: dict[str, Any] | None,
    package_budget_reports: list[dict[str, Any]] | None,
) -> str:
    errors = "\n".join(f"- {error}" for error in validation_errors) or "None"
    return f"""You are EvolAgent's Optimization-Package Selector. Select exactly one proposed package after comparing its causal support, preservation of winning behavior, deterministic production feasibility, resource cost, first-commitment time budget, and matchup-adjusted strength against the empirical opponent snapshots in each budget report. Do not combine packages or invent a new package. A package with status `unresolved` cannot be selected. A package with status `timing_risk` is selectable only when match evidence supports the later contact window and the expected combat advantage outweighs opponent growth. Prefer a `feasible` package that reaches a favorable relative-power window and can sustain the next engagement, not merely one with more own units. Use each report's package-specific DataAgent status and query IDs; do not repeat a capability claim contradicted by verified data. A partial packet may provide valid costs or unit attributes, but its missing relation remains unknown and must not be invented. Do not select a package that requires strategy.md to create, reserve, replenish, split, transfer, or merge a custom fixed-composition army detachment; the executor owns main-force and reinforcement membership. Treat such a package as unsupported even when its production timing is feasible, and select another executable package.

Strategy: {strategy_name}
Race: {race}

Previous schema errors:
{errors}

Evidence digest:
{render_discovery_findings(discovery or {})}

Verified SC2 knowledge:
{render_knowledge_results(knowledge_runs or [])}

Retrieved match/history evidence:
{render_retrieval_evidence(retrieval_evidence or {})}

Proposed optimization packages:
{json.dumps(candidate_package_payload or {}, ensure_ascii=False, indent=2)}

Program-calculated package budgets:
{json.dumps(package_budget_reports or [], ensure_ascii=False, indent=2)}

Return JSON only. Copy the selected package's hypothesis and plan without changing its package contents:
{{
  "selected_package_id":"P1",
  "data_agent_assessment":{{"considered_query_ids":["PKG_P1_REQ","PKG_P1_MATCHUP"],"supporting_findings":["verified fact that supports selection"],"contradicted_claims":["proposal claim contradicted by data"],"rejected_package_ids":["P2"],"limitations":["remaining unavailable fact"]}},
  "mechanism_prediction":{{"expected_change":"observable behavior change","minimum_material_change":"minimum trajectory-level realization","outcome_prediction":"expected match effect","combat_success_measure":"combat or continuation measure","disproof_condition":"what result falsifies this package"}},
  "next_action":"propose_strategy_patch|inspect_runtime",
  "action_reason":"why this package has the best evidence-to-budget tradeoff",
  "evidence_limits":["remaining uncertainty"]
}}
"""


def build_cross_match_decision_prompt(
    *,
    strategy_name: str,
    race: str,
    single_game_analyses: list[BattleAnalysis],
    skill_texts: dict[str, str],
    validation_errors: list[str],
    knowledge_mode: str,
    prior_experiences: list[Any] | None = None,
    capability_manifest: dict[str, Any] | None = None,
    discovery: dict[str, Any] | None = None,
    knowledge_runs: list[dict[str, Any]] | None = None,
    retrieval_evidence: dict[str, Any] | None = None,
    candidate_package_payload: dict[str, Any] | None = None,
    package_budget_reports: list[dict[str, Any]] | None = None,
) -> str:
    """Select one preflighted optimization package."""
    return _compact_decision_prompt(
        strategy_name=strategy_name,
        race=race,
        validation_errors=validation_errors,
        discovery=discovery,
        knowledge_runs=knowledge_runs,
        retrieval_evidence=retrieval_evidence,
        candidate_package_payload=candidate_package_payload,
        package_budget_reports=package_budget_reports,
    )
    del knowledge_mode
    capability_text = json.dumps(capability_manifest or {}, ensure_ascii=False, indent=2)
    errors = "\n".join(f"- {error}" for error in validation_errors) or "None"
    return f"""You are EvolAgent's Cross-Match Decision Agent.

A previous discovery pass analyzed the current batch of matches and identified
strengths, weaknesses, and unresolved questions.

You now have:
- the same factual match summaries
- the current strategy.md
- the discovery findings
- requested SC2 knowledge results, when any
- recent rejected experiments

Your job is to choose ONE most important next step.

Executor capability manifest:
{capability_text}

{_cross_match_shared_context(
    strategy_name=strategy_name,
    race=race,
    skill_texts=skill_texts,
    single_game_analyses=single_game_analyses,
    prior_experiences=prior_experiences,
)}

Discovery findings:
{render_discovery_findings(discovery or {})}

Knowledge results:
{render_knowledge_results(knowledge_runs or [])}

Structured retrieval evidence:
{render_retrieval_evidence(retrieval_evidence or {})}

Retrieval rules:
- Treat recorded timeline rows as the authority for command, observation, zone, and engagement-attribution claims. If a discovery sentence conflicts with the queried rows, revise or reject the discovery sentence.
- Historical experiments provide positive and negative intervention evidence. A similar rejected candidate is evidence against repeating the same concrete package, not automatic proof that every stronger or corrected version of the hypothesis is false.
- Static game knowledge can support costs, prerequisites, effects, counters, and synergies. It cannot diagnose a match by itself and cannot override contrary trajectory evidence.
- Any numerical production-time, throughput, or continuous resource-demand claim must quote a deterministic calculation returned by the knowledge query. If the required calculation is absent, keep the claim qualitative or request more evidence; do not calculate it mentally.
- A production-demand calculation does not establish resource income. Never compare demand against a claimed mineral or gas income unless that income was returned by a deterministic query or measured from match records.
- Every retrieved item used in the final diagnosis must be named in retrieval_assessment. Record conflicting evidence instead of silently dropping it.

First, re-evaluate the discovery findings.

For every important weakness:
- confirm it,
- revise it,
- or reject it.

Knowledge may invalidate an earlier interpretation.
Do not defend a discovery finding merely because it appeared in Round 1.

Preserve strengths that remain supported by evidence.

Confirm or revise discovery.outcome_contrast before selecting an intervention. The
winning pattern and its preservation rule are causal constraints, not decorative
summary text. If losses usually fail to reproduce the winning mechanism, repair the
earliest missing link and do not add a later power spike. A later commitment is
eligible only when repeated losses reproduce the Champion's winning mechanism and
still fail, and the evidence explains why the delayed package should meet a more
favorable opponent despite enemy growth.

{SC2_STRATEGIC_PRIORITY}

{REPLAY_GROUNDED_REASONING_EXAMPLES}

Before selecting the final intervention, identify the 2-4 strongest plausible
explanations for the priority failure pattern that the current evidence actually
supports. Do not invent extra explanations merely to fill the list.

For each explanation record:
1. What evidence supports it?
2. What evidence contradicts it?
3. Is it strategy-fixable?
4. Could Commander/runtime execution explain it better?
5. Does a missing static SC2 fact still need to be resolved? Round 2 cannot query
   Knowledge again; if that fact is still missing, prefer request_more_matches
   only when the remaining unknown is match evidence, not a static fact already
   requested in Round 1.

Do not treat a hypothesis as the primary cause if repeated counterexamples show
the same failure after the hypothesized bottleneck has already been resolved.
Ask of every candidate: if this problem were already resolved in some matches,
did the same failure still occur? If yes, it may be a contributor but cannot
alone explain the main failure pattern.

Before proposing upgrades, technology, expansion, or another delayed power spike,
determine whether the current strategy repeatedly survives the opponent's pressure
until that change can become active. If it does not, the intervention package must
include the necessary survival prerequisites or choose the earlier survival failure
as the priority failure mode. Do not spend more resources on a power spike that the
strategy usually dies before reaching.

Treat unit counters and support relationships as hypotheses to ground, not generic
StarCraft intuition. Exact numerical combat claims must appear in verified knowledge
results or recorded evidence. If the knowledge only establishes a qualitative
effect, keep the hypothesis qualitative; never invent damage multipliers, shots to
kill, timing, or cost values.

Apply SC2 Strategic Priority to both diagnosis and the optimization direction.
Before selecting a lower-priority lever, explicitly state why every relevant
higher-priority explanation is unsupported, already viable, or outside strategy
control. Information, economy, production, timing, or an isolated upgrade cannot
be selected merely because it is easier to patch than the supported combat,
survival, matchup, or power-window problem.

Use discovery.strategy_contract as the interpretation boundary. First classify the
repeated failure as occurring before the core win mechanism becomes available,
during its intended commitment or decisive engagement, or after an initially
successful engagement. Then select the change that repairs that stage without
silently replacing the strategy's gameplay style. The same observed unit counter
can imply an earlier commitment for a pressure strategy, a composition/support
change for a gathered push, or a survival/production change for a scaling strategy;
do not apply one generic response to all three.

Treat failure to destroy all enemy structures by 1800 seconds as an endgame
completion failure even when the strategy won earlier fights. Use trajectory
evidence to distinguish insufficient fighting power from delayed commitment,
stalled reinforcement, stale objectives, and failure to scout or scan for surviving
structures. Any information change must remain non-blocking for an already-ready
attack and must directly support target selection or timely cleanup.

Use that failure stage to declare two narrow permissions in plan. They are false
by default and are not general recommendations:
- composition_change_allowed may be true only when repeated match evidence shows
  that the completed fighting package is itself inadequate at the relevant contact,
  or that a composition change is the necessary survival dependency before the
  core mechanism. A final enemy roster alone is insufficient.
- retreat_change_allowed may be true only when repeated main-force evidence links
  the configured retreat threshold or recovery behavior to the decisive outcome.
  Use loss_timing, retained force, regroup delay, and re-engagement evidence.
  Runtime auto-retreat fires when the local own/enemy power ratio falls below
  retreat_ratio (default 0.6): a higher value retreats earlier, while a lower
  value stays committed longer. Neither direction is inherently an improvement.
- Main Attack Gate and Recovery and Cleanup belong to different stages. A change to
  first-attack timing never requires a numerical or semantic recovery change for
  "consistency." When retreat_change_allowed is false, Recovery and Cleanup must be
  copied unchanged into the candidate.
If failure occurs before the core mechanism and neither condition is supported,
preserve both the unit concept and retreat policy and repair the earlier enabling
link. If the first engagement is viable but continuation fails, composition or
retreat may change only when the post-contact evidence supports that exact lever.
Do not enable both merely to give the Optimizer more freedom.

Strategy identity is primarily the intended combat style and advantage-creation
window, not an immutable unit roster. Units, upgrades, production structures,
economy, and exact thresholds may change when the new package still realizes the
same pressure, gathered-push, defensive-scaling, reinforcement, and recovery style.
Do not reject a useful support unit merely because it was absent from the Champion.
Conversely, do not claim that style is preserved when a new support unit, upgrade,
or production target becomes a hard prerequisite that delays or prevents the
Champion's critical attack window.

For every proposed composition or technology change, explicitly compare:
1. the parent and proposed first-commitment prerequisites;
2. whether expected contact becomes earlier, similar, or later;
3. whether support and core units compete for the same production capacity or
   limiting resource;
4. whether the new unit is optional support, a maintained component, or a hard
   attack gate; and
5. why that timing and package should face a more favorable opponent than the
   Champion does.
An added counter unit is not a complete causal argument. The intervention must
improve the own-versus-enemy package at contact after accounting for the time and
core production sacrificed to obtain it.

## Prior Experiment Interpretation

Candidate selection and causal-hypothesis evaluation are separate. A rejected
candidate proves only that the concrete candidate did not beat its Champion under
the selection rule. It does not by itself prove that the causal direction was
wrong or that a stronger coherent implementation would fail.

An accepted score does not establish the selected causal mechanism. When an
experiment is accepted by score but has implementation_verdict=implemented and
hypothesis_verdict=contradicted, preserve the accepted strategy as Champion but
treat the reason for its gain as unknown. Do not strengthen, increase the dose of,
or relabel that contradicted mechanism in the next experiment. Select a materially
different causal mechanism.

Use prior experiment fields as follows:
- implementation_verdict=underpowered, execution_invalid, or unknown means the
  hypothesis was not adequately tested;
- hypothesis_verdict=inconclusive or not_tested must not be treated as contradiction;
- only hypothesis_verdict=contradicted, supported by evidence that the declared
  minimum material mechanism change occurred while the predicted outcome still
  failed, is evidence against repeating the direction;
- an accepted candidate may support a hypothesis, but mechanism evidence remains
  distinct from the score outcome.

A retry of an underpowered or unaudited hypothesis must describe a materially
stronger intervention, explain how it differs from the previous package, and name
the observable mechanism change it should now produce. Do not repeat a materially
equivalent patch. If repeated attempts remain unauditable, request more evidence,
inspect execution, or choose another evidence-supported hypothesis instead of
indefinitely relabeling the same change.

When history marks underpowered_retry_exhausted, the same causal package is no
longer a repair candidate at that difficulty. Select another mechanism. When an
experiment is implemented and contradicted, changing its family name, adding
"upgraded", "stronger", or "v2", or merely increasing the same support count is
still the same mechanism unless the causal chain and contact-time effect are
materially different.

Assign every proposed experiment a concise mechanism_family identifier describing
the causal mechanism, not a unit name or paragraph name. After two equal-score
tests of the same mechanism, select a different causal direction. After an
adequately implemented experiment is contradicted, do not repeat that mechanism.
Judge semantically equivalent names from the concrete history rather than spelling.

Use structured engagement timing as a hard causal constraint. When an engagement
has loss_timing=losses_before_override, an auto-retreat or other runtime override
may be recorded as a consequence but cannot be selected as the cause of that
collapse. A runtime override may be causal only when evidence shows
override_before_losses.

Attribute every auto-retreat to the exact army group that triggered it. The global
living army inventory is not the affected force. A reinforcement group retreating
with one or a few units is expected survival behavior and is not evidence that the
main force was auto-retreated. Before choosing inspect_runtime for auto-retreat,
cite at least two distinct failed matches whose evidence identifies main_force or
group_0 at the trigger and explicitly states loss_timing=override_before_losses.
Write that timing token verbatim in priority_problem.evidence and
failure_mode_analysis.covered_failures. If group attribution or causal ordering is
missing, request more evidence or choose a supported strategy-fixable contributor;
do not escalate a runtime defect.

Treat retreat_ratio as a tunable engagement threshold, not as a universally good
direction. A higher value preserves units but can abandon a favorable breakthrough;
a lower value accepts more losses but can destroy the force needed for continued
pressure. Select a retreat-policy change only when repeated main-force evidence
links the current threshold to the decisive outcome. Evaluate it through losses
before/after the trigger, force retained, regroup delay, and the next engagement.

Before selecting one primary mechanism, enumerate the failed matches it explains,
the failed matches it does not explain, and concrete counterexamples. A proposed
primary mechanism must cover at least two distinct failed matches. If losses belong
to materially different failure modes and no mechanism covers a repeated subset,
request more matches or select an earlier shared prerequisite instead of forcing
them into one unit-counter explanation.

When repeated decisive fights fail after the current attack gate is met or
exceeded, compare composition/support, upgrades, engagement execution, and the
relative timing window. Explicitly ask whether an earlier smaller force would meet
a weaker opponent, whether the current contact is best, or whether waiting creates
a genuinely stronger relative package. Timing, production or economy is valid only
when this comparison explains the expected combat-outcome change.

If a strategy rule was already satisfied but execution or retreat still looks
abnormal, compare commander_execution and runtime_execution. Do not raise a
strategy threshold only to mask an execution defect; choose inspect_runtime.

Do not call an explanation "the root cause" or "the cause" unless both support
and counterevidence justify it. Prefer "primary tested hypothesis", "most
plausible strategy-fixable contributor", or "best-supported intervention target".

A static counter or synergy fact does not by itself mean "build the counter unit".
Any composition change must still fit strategy identity, existing production,
tech prerequisites, observed enemy information, and resource/timing constraints.

The final hypothesis, when next_action is propose_strategy_patch, must be one of
the considered explanations that was not assessed as contradicted. Do not mark an
explanation contradicted and then select it as the hypothesis.

Then choose exactly one next_action:

- propose_strategy_patch
- request_more_matches
- inspect_runtime
- stop

Choose propose_strategy_patch only when:
- there is a repeated problem supported by match evidence;
- the problem is plausibly controllable through strategy.md;
- there is one concrete, testable hypothesis for why a strategy change may improve match outcomes.

Choose request_more_matches when:
- the current evidence cannot distinguish competing explanations;
- the key behavior is too inconsistent or under-observed.

Choose inspect_runtime when:
- Commander already followed a clear strategy rule but execution failed;
- runtime behavior or tool execution better explains the problem;
- changing strategy.md would only mask an execution defect.

Choose stop when:
- no safe, materially new, testable causal hypothesis remains;
- the defining strategy identity is no longer viable and must not be silently replaced.

Do not suppress a hypothesis merely because a previous candidate was rejected.
Suppress it only when prior mechanism evidence supports hypothesis_verdict=contradicted.
Never repeat a materially equivalent candidate; a same-hypothesis retry must state
the substantive increase or correction in intervention strength.

If next_action is propose_strategy_patch:
- select exactly one priority problem as an object, not a list;
- choose one primary failure mode and one causal account for this generation;
- pre-register one mechanism_prediction with an observable expected change, a
  minimum material change required to count as a real test, an outcome prediction,
  and a disproof condition;
- describe a materially different coherent intervention package in plan;
- preserve relevant strengths;
- audit every strategy area before selecting the package; preserve an area only after confirming that its current rule remains compatible with the optimized whole-game behavior;
- revise every area affected by the selected failure mechanism, including upstream economy and production dependencies and downstream attack, reinforcement, retreat, recovery, and cleanup behavior;
- do not provide final paragraph text, target_paragraph_id, baseline_rule, or candidate_rule.

Optimization-direction rules:
- A causal hypothesis may be chosen only when repeated match evidence supports a plausible connection between the current rule and the observed problem. Do not optimize from StarCraft common sense alone.
- The unit of evolution is one primary failure mode addressed by one coherent intervention package, not one small lever, paragraph, unit, or upgrade. The package must be large enough to produce a clear behavioral difference from the Champion.
- Implement the package with every dependent change required for it to be executable, internally consistent, resource-feasible, prerequisite-complete, survivable until active, and testable.
- The package may coordinate Goal-preserving changes across Macro economy/expansion, production/resource priority, technology/upgrades, unit composition/support balance, and Combat attack readiness, decision-relevant information, reinforcement, recovery, or timely cleanup when they are necessary dependencies of the same hypothesis. Scouting and scans must serve one of those Combat decisions and cannot become a hidden gate; wake events, route waypoints, and runtime state machines are not strategy fields.
- For every proposed change ask: "If this change were removed, would the selected hypothesis become incomplete, internally inconsistent, non-executable, or materially different?" If yes, include it. If no, leave that part of the Champion unchanged.
- Do not combine unrelated improvements merely because all of them appear beneficial. A change that survives the removal test as a separate optimization objective belongs to a later generation.
- A candidate should mutate the Champion, not redesign it from scratch. Preserve successful mechanisms unless current evidence directly contradicts them.
- The optimization may adjust quantities, priorities, timings, prerequisites, support units, and readiness conditions, but must not replace the strategy's defining army concept or win plan unless the current strategy itself explicitly allows that flexibility. If the identity is no longer viable, choose stop rather than swapping strategy family.
- plan.direction states the primary failure mode and causal idea. plan.material_behavior_change states the clearly observable difference from the Champion. plan.coordinated_changes lists the core intervention and every necessary prerequisite or consistency change without writing final paragraph text. Do not limit package size by paragraph count; reject only unrelated changes.
- plan.strategy_area_audit must cover all six named areas. Mark an area revise when the selected mechanism changes its behavior or when leaving it unchanged would create an economic, production, timing, reinforcement, retreat, or endgame inconsistency. A preserve decision requires evidence that the existing rule remains compatible.
- Treat production as a staged whole-game process rather than a final building or unit count. Compare opening order, pre-commitment producer and army targets, post-commitment or midgame producer and army targets, late-game completion targets, producer utilization, mineral and gas banking, worker allocation, upgrades, supply interruptions, and reinforcement demand. Mineral surplus with saturated core queues can justify more of the relevant production capacity; gas surplus alone does not justify a mineral-only producer and instead calls for a supported gas use or worker reallocation.
- Infer offensive continuity from the strategy style and trajectories. For a pressure strategy, do not preserve a retreat-and-full-rebuild loop when it repeatedly interrupts a favorable attack and reinforcement chain. For a scaling or timing strategy, allow regrouping when evidence shows that preserving the army is part of its win mechanism.

The hypothesis must explain why changing strategy.md could plausibly change
external match outcomes.

Match evidence is primary.
Knowledge is only supporting factual context.
Do not query the knowledge database again.

Feedback from an earlier malformed response:
{errors}

Return one JSON object only:
{{
  "action": "decide_batch",
  "analysis": {{
    "strengths_to_preserve":[
      {{"pattern":"...","evidence":["Game 1 @ 420s: ..."]}}
    ],
    "outcome_contrast":{{
      "winning_pattern":"repeated mechanism supported by wins",
      "winning_evidence":["Game 1 @ 420s: ..."],
      "loss_shortfall":"earliest repeated shortfall in losses",
      "loss_evidence":["Game 2 @ 390s: ..."],
      "loss_relationship_to_wins":"winning_mechanism_not_reproduced|winning_mechanism_reproduced_but_failed|mixed|uncertain",
      "causal_difference":"most important causal difference between outcomes",
      "preservation_rule":"winning mechanism that the candidate must retain"
    }},
    "priority_problem":{{
      "problem":"one strategy-fixable problem",
      "evidence":["Game 2 @ 390s: ..."],
      "control_class":"strategy_fixable"
    }},
    "failure_mode_analysis":{{
      "failure_mode":"repeated match-level failure the package will address",
      "failure_stage":"before_core_mechanism|during_commitment_or_engagement|after_successful_engagement|mixed",
      "gate_attainment_and_launch":"whether the intended gate was reached before failure and whether commitment followed at the next effective decision opportunity",
      "earliest_strategy_fixable_link":"earliest causal link that strategy.md can change across repeated failures",
      "why_later_levers_do_not_outrank_it":"why defense, composition, upgrades, or recovery after this link are not a higher-priority explanation",
      "commitment_and_contact_timing":"relative timing of own commitment, actual contact, and opponent power growth",
      "own_package_at_contact":"own composition, support, upgrades, and readiness at decisive contact",
      "opponent_package_and_growth":"opponent composition at contact and material changes caused by waiting",
      "post_contact_continuity":"whether the strategy style requires sustained pressure or deliberate regrouping, plus configured retreat threshold, offensive uptime, force retention, reinforcement flow, regroup delay, and ability to re-engage",
      "production_feasibility":"opening production order, first-commitment throughput, post-commitment producer scaling, queue utilization, mineral and gas banking, worker allocation, supply interruptions, and observed bottlenecks",
      "optimization_implication":"which link in the combat chain should change and why",
      "survival_prerequisite":"whether the strategy survives until its intended power spike and supporting evidence",
      "opponent_pressure_pattern":"cross-match pressure pattern or why it is not decisive",
      "matchup_assessment":"composition, counters, support, upgrades, and engagement conditions across outcomes",
      "counterexample_check":"why wins or resolved cases do not refute the selected failure mode",
      "covered_failures":["Game 2 @ 390s: exact failure explained by this mechanism","Game 5 @ 510s: second independent failure explained"],
      "unexplained_failures":["Game 7 @ 440s: materially different failure not explained, or empty when all are covered"],
      "counterexamples":["Game 1 @ 620s: strongest concrete counterexample and why it does not disprove the selected mechanism"]
    }},
    "priority_alignment":{{
      "selected_priority":"the highest evidence-supported level from SC2 Strategic Priority",
      "higher_priority_assessment":"which higher-priority combat, survival, matchup, or power-window explanations were checked and why none outranks the selection",
      "downstream_combat_effect":"how the selected package materially changes decisive engagement survival or victory"
    }},
    "retrieval_assessment":{{
      "query_summary":"how record, history, and static-knowledge queries changed or confirmed the diagnosis",
      "match_evidence_used":["query id and exact Game N @ Ts finding"],
      "historical_experience_used":["experiment id and lesson, including failed changes when relevant"],
      "knowledge_used":["question id and bounded fact used"],
      "conflicting_evidence":["retrieved evidence that weakens the selected explanation"],
      "confidence":"low|medium|high"
    }},
    "hypothesis":"one causal claim that survived the counterevidence check",
    "mechanism_family":"concise_stable_causal_family_id",
    "mechanism_prediction":{{
      "expected_change":"observable intermediate state the candidate must change",
      "minimum_material_change":"minimum difference required to count as testing the hypothesis",
      "outcome_prediction":"match behavior expected if that mechanism changes",
      "combat_success_measure":"decisive-engagement outcome or force-retention signal expected to improve",
      "disproof_condition":"evidence that would contradict the hypothesis after adequate implementation"
    }},
    "next_action":"propose_strategy_patch",
    "action_reason":"why this is the next step",
    "considered_explanations":[
      {{
        "explanation":"one competing explanation",
        "supporting_evidence":["Game 2 @ 390s: ..."],
        "counterevidence":["Game 1 @ 624s: ..."],
        "control_class":"strategy_fixable",
        "assessment":"plausible_primary|plausible|contributor_not_sufficient|weakly_supported|contradicted|runtime_likely"
      }}
    ],
    "plan":{{
      "direction":"primary failure mode and causal direction of the package",
      "material_behavior_change":"clear match behavior that will differ materially from the Champion",
      "coordinated_changes":[
        {{"change":"core or dependent strategic change","why_required":"why removing it would make the package incomplete or ineffective"}}
      ],
      "preserve":["successful mechanism to keep"],
      "contact_window_effect":"earlier|similar|later|unknown",
      "new_hard_prerequisites":["new condition that can block the first commitment, or empty"],
      "production_tradeoffs":["resource or production capacity taken from the Champion's winning chain, or empty"],
      "window_tradeoff_evidence":["Game N @ Ts evidence justifying a later window, or empty"],
      "why_window_remains_favorable":"why the proposed package preserves or improves relative power at contact",
      "composition_change_allowed":false,
      "retreat_change_allowed":false,
      "stage_scope_evidence":["Game N @ Ts evidence supporting every true permission, or empty"],
      "stage_scope_reason":"why these permissions match failure_stage and the selected hypothesis",
      "strategy_area_audit":[
        {{"area":"goal_identity|economy_expansion|production_order_capacity|technology_composition|attack_timing_objective|reinforcement_retreat_cleanup","decision":"preserve|revise","finding":"what the complete trajectories show in this area","required_change":"coordinated change when decision is revise, otherwise empty","evidence":["Game N @ Ts: supporting observation"]}}
      ],
      "preservation_checks":[
        {{"invariant":"protected winning mechanism","effect":"preserve|improve|evidence_supported_tradeoff","reason":"candidate-level requirement","evidence":["Game 1 @ 420s: ..."]}}
      ]
    }},
    "evidence_limits":[]
  }}
}}
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
    round_index: int = 1,
    round1_analysis: dict[str, Any] | None = None,
    knowledge_observations: list[ToolObservation] | None = None,
    discovery: dict[str, Any] | None = None,
    knowledge_runs: list[dict[str, Any]] | None = None,
) -> str:
    """Compatibility wrapper around the two Cross-match stage prompts."""
    if round_index >= 2:
        return build_cross_match_decision_prompt(
            strategy_name=strategy_name,
            race=race,
            single_game_analyses=single_game_analyses,
            skill_texts=skill_texts,
            validation_errors=validation_errors,
            knowledge_mode=knowledge_mode,
            prior_experiences=prior_experiences,
            capability_manifest=capability_manifest,
            discovery=discovery or round1_analysis,
            knowledge_runs=knowledge_runs,
        )
    return build_cross_match_discovery_prompt(
        strategy_name=strategy_name,
        race=race,
        single_game_analyses=single_game_analyses,
        skill_texts=skill_texts,
        validation_errors=validation_errors,
        knowledge_mode=knowledge_mode,
        prior_experiences=prior_experiences,
        capability_manifest=capability_manifest,
    )


__all__ = [
    "CONTROLLABLE_OPTIMIZATION_SCOPE",
    "RUNTIME_CONTRACT",
    "STRATEGY_MARKDOWN_FORMAT",
    "build_batch_analysis_prompt",
    "build_candidate_prompt",
    "build_cross_match_decision_prompt",
    "build_cross_match_discovery_prompt",
    "build_fixed_match_summary_prompt",
    "build_strategy_patch_validation_prompt",
]
