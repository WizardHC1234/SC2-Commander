from __future__ import annotations

import json
from typing import Any

from .context import render_knowledge_results, render_optimizer_decision
from .types import BattleAnalysis, ToolObservation
from ..optimization.strategy_document import StrategyDocument


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
    decision: dict[str, Any] | None = None,
    knowledge_runs: list[dict[str, Any]] | None = None,
) -> str:
    """Ask the Optimizer to implement one Cross-match hypothesis as paragraph patches."""
    del capability_manifest, knowledge_mode, tool_observations
    from .prompts import RUNTIME_CONTRACT
    errors = "\n".join(f"- {error}" for error in validation_errors) or "None"
    parent_text = str(skill_texts.get("strategy.md") or "")
    document = StrategyDocument.parse(parent_text)
    patch_context = json.dumps(document.patch_context(), ensure_ascii=False, indent=2)
    candidate_text = json.dumps(candidate or {}, ensure_ascii=False, indent=2)
    decision_payload = decision or dict(battle_analysis.raw or {})
    knowledge_text = render_knowledge_results(knowledge_runs or [])

    if isinstance(candidate, dict) and validation_errors:
        return f"""You are revising an invalid paragraph patch.

Keep the same Cross-match hypothesis and plan direction.

Fix only the reported validator or critic errors.

You may add, remove, or revise paragraph patches when required to make the
same hypothesis internally consistent.

Do not introduce a new strategic objective.
Do not modify unrelated paragraphs.
Do not modify # Summary.
Do not add, delete, rename, or reorder paragraphs.

Each target and expected_old_hash must match the parent paragraph catalog.

Cross-match Decision:
{render_optimizer_decision(decision_payload)}

Parent paragraph catalog:
{patch_context}

Invalid candidate:
{candidate_text}

Validator errors:
{errors}

Return one JSON object only:
{{
  "action": "revise_candidate",
  "patches":[
    {{
      "target":"existing_detail_id",
      "expected_old_hash":"copy the exact hash from the paragraph catalog",
      "replacement":"complete replacement instruction without the bullet title",
      "why_required":"why this paragraph must change for the same hypothesis"
    }}
  ],
  "expected_effect":"expected match effect",
  "main_risk":"possible regression"
}}
"""

    return f"""You are EvolAgent's Strategy Optimizer.

The Cross-Match Decision Agent has already selected one priority problem,
one hypothesis, and one change direction.

Do not redo the match analysis.
Do not choose a different problem.
Do not replace the hypothesis.
Do not introduce a second optimization objective.
Do not select among candidate plans.

Your job is only to implement the supplied hypothesis as a coherent paragraph
patch to the current strategy.md.

{RUNTIME_CONTRACT}

Read the complete strategy before editing.

Determine every existing Detail paragraph that must change so that:
1. the hypothesis is implemented;
2. dependent rules remain internally consistent;
3. supported strengths remain intact.

There is no fixed maximum number of paragraph patches.

However, every modified paragraph must be necessary for the same hypothesis.
Leave every unrelated paragraph unchanged.

Do not minimize patch count if doing so would leave contradictory thresholds,
priorities, production rules, attack conditions, recovery conditions, or final
targets.

Before returning patches, scan every Detail paragraph for rules that reuse
or depend on the concept being changed.

If the hypothesis changes a threshold, production priority, composition target,
attack condition, recovery condition, expansion timing, scouting requirement,
or technology dependency, update every paragraph that would otherwise contradict
the new rule.

Do not update paragraphs that merely mention related units but remain logically
consistent.

For each patch:
- target an existing paragraph id;
- copy its exact parent hash;
- return the complete replacement instruction;
- explain why this paragraph must change for the hypothesis.

Do not add, delete, rename, or reorder paragraphs.
Do not modify # Summary.
Do not mention match ids, exact recorded timestamps, EvolAgent internals,
or Replay-only enemy truth in strategy.md.
Do not write micro, map-specific zone IDs, or group IDs.
Express attack-readiness counts as completed, living units in the persistent main force.
Keep explicit end-state supply at or below 200.
Write reusable StarCraft II strategy instructions.

Strategy: {strategy_name}
Race: {race}

Cross-match Decision:
{render_optimizer_decision(decision_payload)}

Current strategy.md:
{parent_text}

Paragraph catalog (stable id, parent hash, and value):
{patch_context}

Verified static SC2 facts:
{knowledge_text}

Return one JSON object only:
{{
  "action": "draft_candidate",
  "patches":[
    {{
      "target":"main_attack_gate",
      "expected_old_hash":"copy the exact hash from the paragraph catalog",
      "replacement":"complete replacement instruction without the bullet title",
      "why_required":"why this paragraph must change for the hypothesis"
    }}
  ],
  "expected_effect":"expected match effect",
  "main_risk":"possible regression to evaluate in games"
}}
"""
