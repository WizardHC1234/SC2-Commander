from __future__ import annotations

import json
from typing import Any

from .context import (
    render_discovery_findings,
    render_knowledge_results,
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
- Change only the smallest coherent area or combination supported by match evidence. Do not compensate for unavailable micro or runtime behavior in strategy.md.

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

This is an evidence-driven reasoning preference, not a fixed ranking enum or a
deterministic category selector. Do not choose a weakly supported higher-level
category over a strongly supported lower-level explanation.

Prefer interventions that can directly change the decisive match outcome. When
several strategy-fixable problems are supported, reason backward from the decisive
engagement:

1. First ask whether the assembled army package can take favorable or survivable
   fights against the observed opponent composition. Compare matchup, composition,
   support-unit balance, engagement conditions, and execution.
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
) -> str:
    """Build one factual summary prompt with a stable cacheable prefix.

    The factual-summary instructions are shared by every match in a batch.
    Strategy text is deliberately excluded: it belongs to cross-match
    diagnosis, while this stage must describe recorded evidence neutrally.
    Keep fixed instructions before record-specific metadata and timeline so
    providers with prefix caching can reuse the long prefix.
    """
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
  ]
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
            keep = (
                "experiment_id",
                "generation",
                "difficulty",
                "parent",
                "candidate",
                "hypothesis",
                "plan_direction",
                "patches",
                "decision",
                "parent_score",
                "candidate_score",
                "delta",
                "posterior_probability_better",
                "score_delta",
                "evaluation",
                "primary_change",
                "primary_lever",
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
- Same paragraph target is not the same experiment."""


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
    del capability_manifest
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

Use concrete evidence such as:
"Game 3 @ 420s: ..."

Rules:
- Compare wins and losses whenever possible.
- Do not infer causation from final-state differences alone.
- A requested Commander action is not proof that execution succeeded.
- Distinguish Commander-observed enemy information from post-match enemy truth.
- Do not invent missing game facts.
- If a weakness may be caused by Commander execution or runtime execution, say so rather than assuming strategy.md is at fault.
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
    "knowledge_questions":[
      {{
        "question":"static SC2 factual question",
        "entities":["relevant unit, structure, upgrade, or ability"],
        "needs":["requirements"]
      }}
    ]
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
) -> str:
    """Round 2: confirm/revise/reject discovery and choose one next action."""
    del capability_manifest, knowledge_mode
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

First, re-evaluate the discovery findings.

For every important weakness:
- confirm it,
- revise it,
- or reject it.

Knowledge may invalidate an earlier interpretation.
Do not defend a discovery finding merely because it appeared in Round 1.

Preserve strengths that remain supported by evidence.

{SC2_STRATEGIC_PRIORITY}

Before selecting the final hypothesis, identify the 2-4 strongest plausible
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

## Prior Experiment Interpretation

Candidate selection and causal-hypothesis evaluation are separate. A rejected
candidate proves only that the concrete candidate did not beat its Champion under
the selection rule. It does not by itself prove that the causal direction was
wrong or that a stronger coherent implementation would fail.

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

When repeated decisive fights fail after the current attack gate is met or
exceeded, explicitly consider unit composition / support balance, matchup-dependent
readiness, technology / upgrades, information quality before committing, Commander
execution, and runtime execution. Considering them is required; changing
composition is not required.

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
- choose exactly one primary causal hypothesis for this generation;
- pre-register one mechanism_prediction with an observable expected change, a
  minimum material change required to count as a real test, an outcome prediction,
  and a disproof condition;
- describe its coherent strategy package direction in plan.direction;
- preserve relevant strengths;
- do not rewrite unrelated strategy areas;
- do not provide final paragraph text, target_paragraph_id, baseline_rule, or candidate_rule.

Optimization-direction rules:
- A causal hypothesis may be chosen only when repeated match evidence supports a plausible connection between the current rule and the observed problem. Do not optimize from StarCraft common sense alone.
- One hypothesis is not one paragraph or one strategy category. Implement it as one coherent strategy package containing every dependent change required for it to be executable, internally consistent, resource-feasible, prerequisite-complete, and testable.
- The package may coordinate economy/expansion, production/resource priority, technology/upgrades, unit composition/support balance, attack readiness/timing, reinforcement/recovery/cleanup, and scouting/information/redecision when they are necessary dependencies of the same hypothesis. This list is neither an enum nor a requirement to change every area.
- For every proposed change ask: "If this change were removed, would the selected hypothesis become incomplete, internally inconsistent, non-executable, or materially different?" If yes, include it. If no, leave that part of the Champion unchanged.
- Do not combine unrelated improvements merely because all of them appear beneficial. A change that survives the removal test as a separate optimization objective belongs to a later generation.
- A candidate should mutate the Champion, not redesign it from scratch. Preserve successful mechanisms unless current evidence directly contradicts them.
- The optimization may adjust quantities, priorities, timings, prerequisites, support units, and readiness conditions, but must not replace the strategy's defining army concept or win plan unless the current strategy itself explicitly allows that flexibility. If the identity is no longer viable, choose stop rather than swapping strategy family.
- plan.direction must answer "What single causal idea is this entire package testing?" and describe the coordinated package without writing final paragraph patches. Good: "Move the selected army package's power spike earlier and align its prerequisites, resource priority, and commitment condition so the intended timing is executable." Bad: "Improve timing." Bad: "Improve timing, expand earlier, change composition, and scout more."

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
    "hypothesis":"one causal claim that survived the counterevidence check",
    "mechanism_prediction":{{
      "expected_change":"observable intermediate state the candidate must change",
      "minimum_material_change":"minimum difference required to count as testing the hypothesis",
      "outcome_prediction":"match behavior expected if that mechanism changes",
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
      "direction":"one coherent strategy package direction testing the selected causal hypothesis",
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

