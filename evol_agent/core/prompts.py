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
- Python/Sharpy handles worker distribution, mining micro, repairs, pathfinding, formations, abilities, targeting, transport handling, transformations, and other unit-level micro.
"""


CONTROLLABLE_OPTIMIZATION_SCOPE = """Controllable strategy scope:
- Objective: improve expected match win rate across repeated games while preserving the strategy's defining style where it remains viable. Completeness, variety, and use of every available action are not objectives.
- Macro: workers, bases, gas, production, technology, upgrades, unit targets, supply, and resource priority.
- Army: gather/readiness conditions, semantic objective, movement intent, reinforcement, retreat, rebuild, and re-engagement.
- Information and redecision: scout/scan purpose, observable request conditions, required information, and meaningful wake checkpoints.
- Strategy identity defines what must be preserved; it is not an extra optimization category.
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
- Reinforcement, retreat, rebuild, re-engagement, and cleanup: keep first-attack and recovery gates consistent.
- Scouting, scanning, information requirements, and redecision triggers: recon must serve a named strategic decision, not "more scouting is always better".
"""


SC2_STRATEGIC_PRIORITY = """## SC2 Strategic Priority

The primary optimization objective is to win the decisive army engagement, or
to survive it with enough retained combat power to continue the win plan. Match
win rate remains the final external objective, but attack time, resource banking,
production synchronization, gate timing, scouting volume, and strategy compliance
are only intermediate mechanisms. Never select one of those surrogate improvements
unless the evidence explains how it should improve decisive combat outcomes.

This is an evidence-driven reasoning preference, not a fixed ranking enum or a
deterministic category selector. Do not choose a weakly supported higher-level
category over a strongly supported lower-level explanation.

Prefer interventions that can directly change the decisive match outcome. When
several strategy-fixable problems are supported, reason backward from the decisive
engagement:

1. First ask whether the assembled army package can take favorable or survivable
   fights against the observed opponent composition. Compare matchup, composition,
   support-unit balance, upgrades, engagement conditions, first-engagement force
   retention, and execution. A repeated catastrophic first engagement outranks
   economy, production, and timing cleanup unless those levers create a materially
   different fighting package or power-spike window.
2. Then ask whether the strategy commits during the correct relative power-spike
   window. Distinguish timing from an absolute unit-count threshold.
3. Then ask whether production capacity and resource allocation create the viable
   army package on time. Additional production must resolve a named timing or
   army-state bottleneck; "more production is stronger" is insufficient.
4. Then ask whether economy and expansion sustain production and enable recovery.
   Their value must be tied to sustained output, a different timing, or rebuilding.
5. Treat technology and upgrades primarily as multipliers of an otherwise viable
   army package unless evidence directly identifies upgrade timing as decisive.
6. Treat scouting and information as valuable only when the observation changes a
   composition, production, timing, readiness, commitment, or recovery decision.

Do not select construction of static defensive structures as the primary evolution
mechanism. Static defenses cannot accompany the mobile army and the executor places
them around owned bases rather than arbitrary forward staging zones. Existing static
defenses may appear in evidence, but improvements must come from the mobile fighting
package, an executable combat control, readiness, production, timing, or recovery.

## Reached-Plan Check

Before selecting a production, economy, or reach-gate timing hypothesis, check
whether repeated losing matches already reached or exceeded the current planned
army and attack gate. If the same decisive failure occurred after the threshold
was reached, failure to reach it is not a sufficient explanation. Explicitly
compare composition, matchup-dependent readiness, engagement conditions, upgrades,
Commander execution, and runtime execution. Do not optimize production or economy
merely to produce more of an army package that still loses decisively.

If the planned package repeatedly succeeds once assembled, treat its viability as
provisionally supported and then compare timing, production synchronization, and
economic efficiency.

## Recovery Priority

Prioritize economy, expansion, production recovery, and rebuild changes when the
first major army package is reasonably viable or competitive but the strategy
repeatedly loses because it cannot restore production or army strength after
attrition or retreat. Do not use more economy as the primary fix for a package
that already loses catastrophically in its first decisive engagement unless the
hypothesis explicitly depends on a different timing created by that economy.

## Information Value

Prefer information-conditioned strategy: observe X, then change composition,
production, timing, readiness, commitment, or recovery. "Scout more" or "scan
more" without a named downstream decision is not a sufficient hypothesis.
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

1. Strengths
Repeated behaviors or strategy mechanisms that appear to contribute to successful games
and should be preserved.

2. Weaknesses
Repeated patterns associated with losses, stalls, bad trades, missed timings,
poor information, or failure to finish. Do not yet name a root-cause hypothesis.

3. Unknowns
Questions that cannot be answered reliably from the match evidence alone. These must
be static SC2 mechanism facts, not "which strategy is best" or "why this game was lost".

4. Opponent pressure patterns
Compare when and how opponents pressure owned bases, economy, production, or the
defending army. Record the observable cues, the own defensive package available at
contact, whether the strategy survived until its intended power spike, and concrete
counterexamples. Do not fit a rule to one exact timestamp or one opponent build.

5. Matchup and composition patterns
Compare own and enemy compositions at major engagements across wins and losses.
Separate upgrade differences from unit counters, support balance, defender advantage,
engagement conditions, and execution. If a counter, synergy, or upgrade effect is
needed to distinguish explanations, request a bounded knowledge query.

6. Retrieval plan
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
readiness, technology / upgrades, information quality before committing, Commander
execution, and runtime execution. Considering them is required; changing
composition is not required.

Do not select attack timing, production throughput, resource banking, or gate
attainment as the priority objective when the assembled army still loses the
decisive engagement. Such a lever is valid only when it creates a materially
different combat package or reaches a supported relative power-spike window, and
the mechanism_prediction must name the expected combat-outcome change.

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
- do not rewrite unrelated strategy areas;
- do not provide final paragraph text, target_paragraph_id, baseline_rule, or candidate_rule.

Optimization-direction rules:
- A causal hypothesis may be chosen only when repeated match evidence supports a plausible connection between the current rule and the observed problem. Do not optimize from StarCraft common sense alone.
- The unit of evolution is one primary failure mode addressed by one coherent intervention package, not one small lever, paragraph, unit, or upgrade. The package must be large enough to produce a clear behavioral difference from the Champion.
- Implement the package with every dependent change required for it to be executable, internally consistent, resource-feasible, prerequisite-complete, survivable until active, and testable.
- The package may coordinate economy/expansion, production/resource priority, technology/upgrades, unit composition/support balance, attack readiness/timing, reinforcement/recovery/cleanup, and scouting/information/redecision when they are necessary dependencies of the same hypothesis. This list is neither an enum nor a requirement to change every area.
- For every proposed change ask: "If this change were removed, would the selected hypothesis become incomplete, internally inconsistent, non-executable, or materially different?" If yes, include it. If no, leave that part of the Champion unchanged.
- Do not combine unrelated improvements merely because all of them appear beneficial. A change that survives the removal test as a separate optimization objective belongs to a later generation.
- A candidate should mutate the Champion, not redesign it from scratch. Preserve successful mechanisms unless current evidence directly contradicts them.
- The optimization may adjust quantities, priorities, timings, prerequisites, support units, and readiness conditions, but must not replace the strategy's defining army concept or win plan unless the current strategy itself explicitly allows that flexibility. If the identity is no longer viable, choose stop rather than swapping strategy family.
- plan.direction states the primary failure mode and causal idea. plan.material_behavior_change states the clearly observable difference from the Champion. plan.coordinated_changes lists the core intervention and every necessary prerequisite or consistency change without writing final paragraph text. Do not limit package size by paragraph count; reject only unrelated changes.

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
