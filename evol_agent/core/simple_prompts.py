from __future__ import annotations

from .context import (
    render_battle_analysis,
    render_sc2_knowledge,
    render_single_game_analyses,
    render_skill_context,
)
from .prompts import (
    CONTROLLABLE_OPTIMIZATION_SCOPE,
    RUNTIME_CONTRACT,
    STRATEGY_MARKDOWN_FORMAT,
)
from .types import BattleAnalysis, ToolObservation


def build_batch_analysis_prompt(
    *,
    strategy_name: str,
    race: str,
    single_game_analyses: list[BattleAnalysis],
    skill_texts: dict[str, str],
    validation_errors: list[str],
    knowledge_mode: str,
    prior_experiences: list[str] | None = None,
) -> str:
    """Build the single cross-match analysis request used by EvolAgent."""
    errors = "\n".join(f"- {error}" for error in validation_errors[-3:]) or "None"
    knowledge_rule = (
        "Return zero to five focused knowledge_questions only when static SC2 "
        "facts are needed to choose or implement the optimization direction."
        if knowledge_mode == "enabled"
        else "Knowledge is disabled; return an empty knowledge_questions list."
    )
    experience_text = "\n".join(f"- {item}" for item in (prior_experiences or [])[-3:]) or "None"
    return f"""You are EvolAgent's batch Analysis Agent.

Read every independent factual match summary and produce one optimization hypothesis. This is the only cross-match analysis call. Diagnose from recorded evidence; do not rewrite strategy.md and do not modify runtime code.

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
- Select one primary strategy-fixable root bottleneck supported by the strongest repeated evidence. Do not turn every symptom into a target.
- Trace the primary problem through current strategy rule -> Commander decision -> observed execution or progress -> later outcome. A requested command is not proof that the action succeeded; confirm execution from later game state.
- If the strategy already contains a clear executable rule but Commander did not follow it, report an execution limitation instead of proposing the same rule again.
- Propose one coherent optimization direction. It may contain dependent edits, but must not independently redesign economy, technology, composition, attack timing, and information policy all at once.
- State how the proposed direction could weaken the current winning mechanism.
- Put runtime-only, micro, and unsupported problems in evidence_limits.
- Treat rejected-candidate experience as limited supporting evidence, not as a permanent prohibition or a substitute for the current match records.
- Knowledge questions go only to the deterministic SC2 entity dataset. Never ask it for optimal strategy, match timing, Commander behavior, movement modes, or micro.
- Every knowledge question must be tied to the selected hypothesis, and its answer must be capable of changing how that hypothesis is implemented. Use exact candidate entity names, do not ask facts already established by the supplied evidence, and preserve source time units.
- Ask the dataset for facts or small evidence-supported comparisons, not for a strategic recommendation. Prefer fewer questions that cover the necessary relationships without overlap.
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
    "primary_problem": {{
      "problem_id":"P1",
      "problem":"single root bottleneck",
      "evidence":["Match number and event"],
      "consequence":"later match effect",
      "strategy_fixable":true,
      "confidence":"low|medium|high"
    }},
    "optimization_hypothesis": {{
      "direction":"one coherent strategy correction",
      "scope":["macro|army|information"],
      "expected_benefit":"expected match effect",
      "risk_to_winning_mechanism":"possible regression"
    }},
    "knowledge_questions": [
      {{
        "id":"Q1",
        "question":"focused static SC2 fact needed for this hypothesis",
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

Write one complete candidate replacement for strategy.md. The batch Analysis Agent has already selected one optimization hypothesis. Do not re-diagnose matches, add unrelated improvements, or modify runtime code.

{RUNTIME_CONTRACT}

{CONTROLLABLE_OPTIMIZATION_SCOPE}

{STRATEGY_MARKDOWN_FORMAT}

Rules:
- Preserve strategy_contract.identity, core_commitments, and winning_mechanism unless direction=replace.
- Implement only optimization_hypothesis. Dependent edits are allowed; unrelated upgrades, units, expansions, and information rules are not.
- Preserve the original power timing. A new support unit, upgrade, building, scout, or scan must not become a mandatory attack prerequisite unless the selected hypothesis specifically requires that delay.
- Prefer one reusable rule with an observable condition over repeated warnings, match-specific patches, or many narrow exceptions.
- When a new target competes with the core plan for resources or production capacity, state the necessary priority and do not delay the core power timing unless the evidence justifies that tradeoff.
- Use verified knowledge only as factual support. It cannot prove an optimal ratio, timing, or win probability.
- Every instruction must be executable through current Macro, Army, scout, scan, and wake controls. Leave micro to the runtime.
- Keep the complete strategy internally consistent and keep explicit end-state supply at or below 200.
- After editing, ensure the complete strategy still has a coherent path through economy and production, force preparation, engagement, reinforcement, recovery, and eventual victory. This is a self-check, not permission to expand unrelated sections.
- Write ordinary StarCraft II strategy language and do not mention EvolAgent internals.

Knowledge mode: {knowledge_mode}
Strategy: {strategy_name}
Race: {race}

Current strategy.md:
{render_skill_context(skill_texts)}

Batch analysis and optimization hypothesis:
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
    "primary_change":"single coherent change",
    "expected_effect":"expected match effect",
    "main_risk":"possible regression to evaluate in games"
  }},
  "files": {{
    "strategy.md": "# Summary\\n...\\n\\n# Details\\n* Opening: ...\\n* Main Attack Gate: ..."
  }}
}}

After basic validator feedback, use action="revise_candidate" with the same complete schema.
"""


__all__ = ["build_batch_analysis_prompt", "build_candidate_prompt"]
