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
- One Commander receives the full structured observation and strategy, then emits absolute macro targets, group-level army commands, optional scan/scout requests, and one wake event.
- Macro targets execute concurrently and replace the previous target list. Every still-needed unmet target must be emitted again; ordering expresses resource priority.
- The runtime expands structure prerequisites and automatically adds build_gas when selected actions require gas. Action descriptions are authoritative for costs, supply, duration, producers, research locations, and prerequisites.
- The persistent main group is the operational force. group_1 contains newly produced units far from it and normally reinforces the main force or its current objective.
- Army movement is group-level and uses one semantic destination with an available movement mode. The Commander resolves semantic locations to observed zone IDs.
- At most one SCV scout is active. Scanner Sweep costs Orbital energy. Every decision has one observable wake condition plus a runtime fallback deadline.
- Observable enemy information includes currently visible contents and last_seen_enemy_contents with seconds_since_last_seen. Scan readiness is observable. Each Commander cycle is one decision; later wakes re-evaluate the same strategy after a scan or scout request.
- Do not require judging whether a Scanner Sweep is "safe", hidden opponent truth, or finishing a scan-then-commit sequence inside one decision frame.
- A scan request and scan_ready may trigger a later decision, but strategy.md cannot remember that a scan completed, test whether scan_ready previously fired, or use raw observation-field names such as last_seen_enemy_contents as stateful conditions. Do not use runtime transformation identifiers (for example SiegeTankSieged or VikingFighter) as strategy conditions; transformations and their states are runtime-owned evidence only.
- Python/Sharpy handles worker distribution, mining micro, repairs, pathfinding, formations, abilities, targeting, transport handling, transformations, and other unit-level micro.
"""


CONTROLLABLE_OPTIMIZATION_SCOPE = """Controllable strategy scope:
- Objective: improve expected match win rate across repeated games while preserving the strategy's defining style where it remains viable. Completeness, variety, and use of every available action are not objectives.
- The only primary optimization domains are: economy and expansion targets;
  production-building and unit-count targets; technology and upgrade paths; army
  composition; and attack-readiness conditions and strategic objectives.
- Scouting, scanning, wake events, decision-cycle protocols, reinforcement routing,
  retreat, recovery, and cleanup are not optimization domains. Preserve their
  existing behavior. A related paragraph may change only to remove a stale reference
  created by an allowed modification, and that repair must not introduce a new rule.
- Strategy identity defines what must be preserved; it is not an extra optimization category.
- Before ranking failures, infer the strategy's style, core win mechanism, critical
  timing or power spike, core commitments, and flexible components directly from
  the complete strategy.md. Do not infer identity from the strategy filename or a
  hard-coded strategy-family profile. Separate the defining mechanism from its
  current numerical implementation: a worker count, producer count, unit target,
  timing, or readiness threshold is adjustable unless the strategy explicitly
  makes that exact value indispensable to the win mechanism. Merely appearing in
  strategy.md does not make a number a core commitment.
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
- Attack readiness and strategic objective: living-force, upgrade, or other currently
  observable readiness conditions, plus the semantic attack objective. Do not design
  an information-acquisition workflow or a wake-event implementation.
"""


SC2_STRATEGIC_PRIORITY = """## SC2 Strategic Priority

The external objective is match victory through a favorable decisive engagement or
survival with enough combat power to continue the strategy's win plan. Timing,
production, economy, upgrades, scouting, and compliance are mechanisms, not goals.

Use one ordered analysis:
1. Infer the strategy's core win mechanism and critical relative power window from
   the complete strategy.md.
2. Compare wins and losses to locate the first failure: before the mechanism forms,
   while it is executed, or after it has succeeded. Optimize that first failed stage;
   do not replace it with a later symptom.
3. Within that stage, test army-package viability, survival to the intended window,
   commitment timing, production/resource feasibility, technology, and composition,
   using the strongest repeated evidence rather than category labels.
4. Select one strategy-fixable causal hypothesis and include every dependency needed
   to create a material combat-outcome difference.

This is not a fixed ranking enum or an exhaustive enum; evidence determines which
explanation survives within the first failed stage. It is not a deterministic category selector.

For timing strategies, a nominally stronger but later army is not an improvement
unless evidence shows that own gains exceed opponent growth. For scaling strategies,
preserve survival and progression to the intended power spike. If losing games
already reached the planned gate, do not blame gate attainment without explaining
why the reached package or relative window was still inadequate.

Information is useful only when a decision-time observable fact changes an
executable composition, production, timing, commitment, or recovery choice.
Replay-only enemy_truth is diagnosis-only. Scouting or scanning that leaves the
army, target, and commitment unchanged must remain optional and non-blocking.
Scouting, scanning, and wake behavior may explain what the Commander knew or when it
redecided, but EvolAgent must not optimize those mechanisms or add new scan/scout/
wake instructions to strategy.md.

An observed opponent feature is diagnostic evidence by default, not a strategy
condition. Promote it to a live control condition only when cross-match evidence
shows that the condition discriminates between situations requiring different
actions, the alternative action is concrete and executable, and a simpler
unconditional change cannot test the same hypothesis. A counter, technology, or
structure appearing in both successful and failed matches is not discriminative by
itself. Refine the condition or keep it diagnosis-only. Prefer a sufficient plan
with fewer new branches, information requests, and cross-cycle dependencies.

Do not use static defensive construction as the primary mechanism, and do not hide
Commander/runtime execution defects inside a more verbose strategy rule.
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

Select a small number of important recorded interactions, typically 8 to 15 when the match contains that much change. Prefer opening formation, clear economy or technology changes, the last recorded state before the first commitment, the first attack command, first important enemy contact, major fights, the first appearance of a materially stronger enemy package, retreat or regroup, base losses, and the end state. Preserve enough timestamped rows to compare how both armies changed before, at, and after first contact. Do not copy every row.

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
  at the retreat trigger, and after retreat when available. State whether most losses
  happened before or after the trigger. Never describe auto-retreat as the cause of
  the collapse when the recorded force had already collapsed before it fired.

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
      "runtime_override": "auto-retreat trigger and force at trigger, or empty",
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
                        "strategy_mechanism_assessment",
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
                experience_lines.append(
                    json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                )
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
                "strategy_contract",
                "strategy_mechanism_assessment",
                "mechanism_family",
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
                "selected_changes",
                "primary_lever",
                "lesson",
                "reason",
                "inheritance",
            )
            compact_item = {key: item.get(key) for key in keep if key in item}
            for key in ("hypothesis", "primary_change", "lesson"):
                if isinstance(compact_item.get(key), str):
                    compact_item[key] = compact_item[key][:1200]
            runtime_findings = compact_item.get("runtime_findings")
            if isinstance(runtime_findings, list):
                compact_item["runtime_findings"] = runtime_findings[:3]
            selected_changes = compact_item.get("selected_changes")
            if isinstance(selected_changes, list):
                compact_item["selected_changes"] = [
                    {
                        key: change.get(key)
                        for key in ("target", "change", "why")
                        if key in change
                    }
                    for change in selected_changes[:4]
                    if isinstance(change, dict)
                ]
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
- decision is the evolution selection result from fixed 10-game scores.
- score_delta is the direct score change. posterior is statistical evidence
  strength only; it is not a promotion gate.
- Accepted: candidate score was strictly higher in the tested context.
- Rejected: candidate score was strictly lower.
- Inconclusive: the two scores were equal; not proof for or against.
- Same paragraph target is not the same experiment.
- Compare the actual selected_changes and score deltas in chronological order.
  Treat repeated changes to the same target in the same direction as one explored
  trajectory even when mechanism_family is renamed or the numerical dose changes.
  A stronger dose after a contradicted implementation requires new trajectory
  evidence for a threshold effect; "the previous value was not large enough" is
  not new evidence. When successive doses worsen both the mechanism observation
  and match score, change causal direction instead of escalating again.
- A parent_analysis_seed is the previous synthesis of a subset of the current
  records. Preserve findings still supported by the full batch, revise findings
  contradicted by newly added matches, and do not count the seed as another match.
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

1. Strategy identity and core mechanism
Read the complete strategy.md before interpreting any match. Infer its gameplay
style, the mechanism by which it is intended to create a winning engagement, its
critical timing window or power spike, the commitments that define the strategy,
and the components that can change without replacing it. Use strategy content, not
the strategy filename or a hard-coded family profile. This contract must be formed
before weaknesses are ranked. Preserve the causal idea rather than automatically
preserving its current numerical settings. Put current quantities and readiness
thresholds in flexible_components unless the text and match evidence show that the
exact value itself defines the strategy.

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

Treat composition as a time-varying relationship, not a final-state label. For each
repeated decisive pattern, reconstruct when the own intended package became usable,
when the attack was ordered, when first contact occurred, and how the opponent's
army or counter package changed across that interval. Compare at least one earlier
recorded opportunity, actual contact, and a later state when the summaries contain
them. State whether moving contact earlier, keeping it unchanged, or delaying it
would plausibly improve the relative power window. Do not assume that waiting for a
larger own army is beneficial when the opponent is growing or completing counters.

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

Use concrete evidence such as:
"Game 3 @ 420s: ..."

Rules:
- First compare whether the core win mechanism and critical timing were realized
  in wins and losses. Do not let a later unit counter or final-state composition
  overwrite an earlier failure to realize the strategy's intended advantage.
- Separate army readiness time, attack-command time, travel or staging delay, and
  first-contact time. A strategy may have enough units on time yet miss its window
  because commitment or arrival is late.
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
      "style":"gameplay style inferred from strategy.md",
      "core_win_mechanism":"how this strategy is intended to create a winning engagement",
      "critical_timing_or_power_spike":"relative timing window or power state that makes the mechanism work",
      "core_commitments":["behavior whose removal would replace the strategy"],
      "flexible_components":["component that may change without replacing the strategy"],
      "optimization_boundary":"what evolution must preserve unless repeated evidence disproves it",
      "direction":"preserve|adjust|replace"
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
) -> str:
    """Round 2: confirm/revise/reject discovery and choose one next action."""
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
- A database result is a factual premise, not a policy proposal. It must not promote
  a counter, technology, unit, or structure into a scouting condition, attack gate,
  target branch, or delay rule unless match evidence independently passes the
  information-control promotion test below.
- Any numerical production-time, throughput, or continuous resource-demand claim must quote a deterministic calculation returned by the knowledge query. If the required calculation is absent, keep the claim qualitative or request more evidence; do not calculate it mentally.
- A production-demand calculation does not establish resource income. Never compare demand against a claimed mineral or gas income unless that income was returned by a deterministic query or measured from match records.
- Every retrieved item used in the final diagnosis must be named in retrieval_assessment. Record conflicting evidence instead of silently dropping it.

First, re-evaluate the strategy contract inferred in discovery. Confirm or revise
the style, core win mechanism, critical timing or power spike, core commitments,
and flexible components using the complete strategy.md. Do this before ranking
weaknesses. Never substitute a unit name for the strategy's causal mechanism, and
do not treat the current numerical attack gate as immutable merely because it is
written in the parent strategy. Preserve the intended engagement mechanism; allow
the gate value to move when evidence supports an earlier or later realization.

Next, compare wins and losses to determine whether the core mechanism was actually
realized. State whether the repeated failure occurs before the mechanism is formed,
during its intended execution, after it has succeeded, or in mixed stages. In
particular, distinguish "the planned army eventually fought" from "the strategy
reached the relative power window that makes that army effective."

Then re-evaluate the remaining discovery findings.

Before choosing an optimization direction, reconstruct the relative power window
across time. Compare the own package and opponent package at the last usable state
before commitment, at actual first contact, and after the opponent's next material
power increase when those rows exist. Explicitly compare three alternatives:
earlier contact with a smaller own force, current contact, and later contact with a
larger own force but a more developed opponent. The conclusion may favor earlier,
unchanged, or later commitment; never assume one direction from strategy name.
When earlier contact is selected, compare at least a lower readiness threshold and
faster attainment of the current threshold as alternative implementations. Choose
between them from observed force viability, travel time, production timing, and the
opponent package expected at contact; do not preserve the current threshold by
default.

For every important weakness:
- confirm it,
- revise it,
- or reject it.

Knowledge may invalidate an earlier interpretation.
Do not defend a discovery finding merely because it appeared in Round 1.

Preserve strengths that remain supported by evidence.

{SC2_STRATEGIC_PRIORITY}

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

Use next_action=stop only after the evidence rules out every remaining
strategy-fixable direction that preserves the strategy contract. Testing several
numerical doses of one direction does not exhaust the strategy. Before stopping,
explicitly assess whether earlier realization or contact, production acceleration,
composition/support, technology, economy, and strategic objective contain an
untested evidence-supported alternative within the flexible components. In
particular, when later or larger commitment repeatedly worsens the relative power
window, evaluate an earlier-window intervention before concluding that no action is
available. Stop is appropriate when these alternatives are contradicted, infeasible,
outside strategy control, or would replace an identity whose direction is preserve.

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

Interpret successive numerical edits as a directional trajectory, not unrelated
candidates. State whether each edit made actual commitment or first contact earlier,
unchanged, or later, then compare the score and opponent package at contact. When
progressively later commitment produces progressively worse outcomes and earlier
contacts provide positive counterexamples, do not conclude that the original gate
is immutable. Evaluate the opposite timing direction, including a lower readiness
threshold when the smaller force remains viable, alongside faster attainment of the
current threshold. A failed increase disproves further waiting more directly than it
disproves an earlier commitment.

A retry of an underpowered or unaudited hypothesis must describe a materially
stronger intervention, explain how it differs from the previous package, and name
the observable mechanism change it should now produce. Do not repeat a materially
equivalent patch. If repeated attempts remain unauditable, request more evidence,
inspect execution, or choose another evidence-supported hypothesis instead of
indefinitely relabeling the same change.

Assign every proposed experiment a concise mechanism_family identifier describing
the causal mechanism, not a unit name or paragraph name. Use prior experiments as
a hard search policy: after two non-accepted attempts in one mechanism_family,
select a different family; after an implemented experiment is contradicted, block
that family; after execution_invalid, block every retry that depends on the same
unsupported Commander/runtime capability. Do not evade the policy by renaming a
materially equivalent mechanism.

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

Before selecting one primary mechanism, enumerate the failed matches it explains,
the failed matches it does not explain, and concrete counterexamples. A proposed
primary mechanism must cover at least two distinct failed matches. If losses belong
to materially different failure modes and no mechanism covers a repeated subset,
request more matches or select an earlier shared prerequisite instead of forcing
them into one unit-counter explanation.

When repeated decisive fights fail after the current attack gate is met or
exceeded, explicitly consider unit composition / support balance, matchup-dependent
readiness, technology / upgrades, Commander execution, and runtime execution.
Considering them is required; changing composition is not required. Information
quality may explain uncertainty but is not itself an EvolAgent modification domain.

Do not dismiss attack timing merely because the assembled army loses at current or
later contact. If earlier contacts succeed before an opponent power increase while
later contacts fail after it, that is evidence for a timing intervention rather
than proof that the own package is categorically nonviable. Attack timing,
production throughput, resource banking, or gate attainment is valid only when it
reaches such a supported relative power-spike window; the mechanism_prediction must
name the expected change in first-contact time, opponent package at contact, and
combat outcome.

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
- preserve the strategy contract unless repeated evidence explicitly supports an
  identity-level change;
- use strategy_mechanism_assessment to ensure the optimization addresses the stage
  where the core mechanism first fails;
- complete information_grounding whenever an allowed plan branches on currently
  visible or observed enemy state. Replay-only enemy_truth may support diagnosis but
  may never be used as a strategy condition. The plan must use information already
  present in the current observation and may not add scouting, scanning, or wake
  behavior to obtain it;
- for every new information-dependent branch, complete
  information_grounding.control_necessity. Name the exact live condition and action
  difference, compare it with the simplest unconditional alternative, cite at least
  two matches showing discriminative value, and assess successful counterexamples.
  Mere correlation with losses, static counter knowledge, or observability is not
  sufficient. If the cue also occurs in successful matches and no narrower
  decision-relevant condition separates them, keep it diagnosis-only and choose a
  non-information-dependent plan;
- select exactly one priority problem as an object, not a list;
- choose one primary failure mode and one causal account for this generation;
- pre-register one mechanism_prediction with an observable expected change, a
  minimum material change required to count as a real test, an outcome prediction,
  and a disproof condition;
- describe a materially different coherent intervention package in plan;
- preserve relevant strengths;
- do not rewrite unrelated strategy areas;
- choose the primary change only from economy/expansion, production targets,
  technology/upgrades, army composition, or attack readiness/objective. Do not
  propose new scouting, scanning, wake-event, decision-cycle, reinforcement,
  retreat, recovery, or cleanup behavior. Those paragraphs may receive only a
  consistency repair caused by an allowed primary change;
- do not provide final paragraph text, target_paragraph_id, baseline_rule, or candidate_rule.

Optimization-direction rules:
- A causal hypothesis may be chosen only when repeated match evidence supports a plausible connection between the current rule and the observed problem. Do not optimize from StarCraft common sense alone.
- The unit of evolution is one primary failure mode addressed by one coherent intervention package, not one small lever, paragraph, unit, or upgrade. The package must be large enough to produce a clear behavioral difference from the Champion.
- Implement the package with every dependent change required for it to be executable, internally consistent, resource-feasible, prerequisite-complete, survivable until active, and testable.
- The package may coordinate economy/expansion, production/resource priority,
  technology/upgrades, unit composition/support balance, and attack
  readiness/objective when they are necessary parts of the same hypothesis.
- For every proposed change ask: "If this change were removed, would the selected hypothesis become incomplete, internally inconsistent, non-executable, or materially different?" If yes, include it. If no, leave that part of the Champion unchanged.
- Do not combine unrelated improvements merely because all of them appear beneficial. A change that survives the removal test as a separate optimization objective belongs to a later generation.
- A candidate should mutate the Champion, not redesign it from scratch. Preserve successful mechanisms unless current evidence directly contradicts them.
- The optimization may adjust quantities, priorities, timings, prerequisites, support units, and readiness conditions, but must not replace the strategy's defining army concept or win plan unless the current strategy itself explicitly allows that flexibility. If the identity is no longer viable, choose stop rather than swapping strategy family.
- plan.direction states the primary failure mode and causal idea.
  plan.material_behavior_change states the clearly observable difference from the
  Champion. plan.coordinated_changes contains only changes in the five allowed
  domains; the Optimizer handles mechanical stale-reference repairs without turning
  them into additional mechanisms.

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
    "strategy_contract":{{
      "style":"gameplay style inferred from the complete strategy",
      "core_win_mechanism":"how the strategy creates its intended winning engagement",
      "critical_timing_or_power_spike":"relative timing window or power state",
      "core_commitments":["must-preserve behavior"],
      "flexible_components":["adjustable behavior"],
      "optimization_boundary":"identity boundary for this evolution",
      "direction":"preserve|adjust|replace"
    }},
    "strategy_mechanism_assessment":{{
      "core_mechanism_realization":"comparison of realization across wins and losses",
      "critical_timing_comparison":"timestamped comparison of own readiness, attack command, first contact, and opponent growth; contrast earlier/current/later contact",
      "failure_stage":"before_core_mechanism|during_core_mechanism|after_core_mechanism|mixed|unknown",
      "optimization_implication":"what kind of change follows from the first failed stage"
    }},
    "core_mechanism_guard":{{
      "identity_effect":"preserve|adjust|replace",
      "first_commitment_effect":"earlier|same|later|conditional",
      "relative_power_effect":"improve|preserve|weaken|unknown",
      "evidence":["Game 2 @ 390s: timestamped comparison supporting the claimed effect"],
      "delayed_first_commitment_success_evidence":["Game 1 @ 620s: delayed contact succeeded under the same failure context"],
      "justification":"why the selected timing effect preserves or improves the strategy's relative power window"
    }},
    "information_grounding":{{
      "uses_runtime_enemy_information":false,
      "facts":[
        {{
          "claim":"fact used by the proposed strategy branch",
          "source":"enemy_observed|own_observation|static_knowledge|enemy_truth",
          "use":"diagnosis_only|strategy_condition",
          "available_before_decision":true,
          "runtime_supported":true,
          "evidence":["Game 2 @ 390s: ..."]
        }}
      ],
      "execution_rule":"current-observation rule, or no enemy information is used as a strategy condition",
      "control_necessity":{{
        "condition":"exact decision-time observable condition; empty for a non-information plan",
        "action_change":"concrete action when true versus false",
        "simpler_alternative":"simplest unconditional strategy change considered",
        "why_information_is_required":"why that simpler change cannot test the same hypothesis",
        "discriminative_evidence":["Game 2 @ 390s: cue and outcome","Game 5 @ 410s: contrasting cue/action/outcome"],
        "counterexample_assessment":"successful matches containing the cue and whether they invalidate or require refinement of the condition"
      }}
    }},
    "strengths_to_preserve":[
      {{"pattern":"...","evidence":["Game 1 @ 420s: ..."]}}
    ],
    "priority_problem":{{
      "problem":"one strategy-fixable problem",
      "evidence":["Game 2 @ 390s: ..."],
      "control_class":"strategy_fixable"
    }},
    "failure_mode_analysis":{{
      "failure_mode":"repeated match-level failure the package will address",
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
      "preserve":["successful mechanism to keep"]
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
