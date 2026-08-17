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
      "commands": ["build_factory -> 2"]
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

Recent rejected-candidate experience:
{_format_prior_experiences(prior_experiences)}"""


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
        "Knowledge questions are allowed only for missing static SC2 facts that "
        "could materially change interpretation. An empty list is valid."
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
        "entities":["Siege Tank"],
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
- no safe, materially new, testable strategy lever remains.

If a significantly failed rejected experiment already tested the same primary lever,
do not choose an essentially identical change unless this batch has new direct evidence.

If next_action is propose_strategy_patch:
- select exactly one priority problem as an object, not a list;
- state one hypothesis;
- propose one coherent change direction;
- preserve relevant strengths;
- do not rewrite unrelated strategy areas;
- do not provide final paragraph text, target_paragraph_id, baseline_rule, or candidate_rule.

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
    "hypothesis":"one causal claim",
    "next_action":"propose_strategy_patch",
    "action_reason":"why this is the next step",
    "plan":{{
      "direction":"coherent change direction, not paragraph text",
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
]
