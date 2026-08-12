from __future__ import annotations

from .context import render_skill_context


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
    skill_texts: dict[str, str],
    match_timeline: str,
) -> str:
    """Build one factual summary request from one complete compact timeline."""
    return f"""You summarize one StarCraft II match for EvolAgent.

The host has placed every Commander snapshot in chronological order in one fixed-schema table. Read every row. Summarize only recorded facts; do not diagnose causes or propose strategy changes.

Keep observed state, requested macro targets, execution progress, and army/recon/wake orders distinct. Keep completed, under-construction, training, and living units distinct. `enemy` is what Commander observed or remembered under fog of war. `opponent_truth_after_match`, when present, is post-match Replay truth that Commander did not know at decision time. For major engagements, compare the nearest pre-fight and post-fight rows. Put unsupported conclusions in evidence_limits.

Strategy: {strategy_name}
Race: {race}
Match metadata:
{record_manifest}

Current strategy.md:
{render_skill_context(skill_texts)}

Complete fixed match timeline:
{match_timeline}

Return one JSON object with exactly these top-level fields:
{{
  "outcome_summary": "concise chronological account",
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
"""


__all__ = [
    "CONTROLLABLE_OPTIMIZATION_SCOPE",
    "RUNTIME_CONTRACT",
    "STRATEGY_MARKDOWN_FORMAT",
    "build_fixed_match_summary_prompt",
]
