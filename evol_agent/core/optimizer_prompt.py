from __future__ import annotations

import json
from typing import Any

from .context import render_knowledge_results, render_optimizer_decision
from .types import BattleAnalysis, ToolObservation
from ..optimization.strategy_document import StrategyDocument


_PATCH_CONSISTENCY_RULES = """
Write exclusive if/else branches when conditions are alternatives. Never require
mutually exclusive states (fresh intel vs stale intel; scan available vs scan
unavailable) to be true at the same time.

Give one paragraph ownership of the attack gate. Dependent paragraphs may follow
that gate; they must not copy a second full copy of the condition.

If recovery rebuilds toward an attack threshold, recover until that same gate is
satisfied. Do not hard-code a lower rebuild count that the gate can immediately
raise.

Use only observable information: living unit counts, last-seen enemy contents
and recency, Orbital energy / scan readiness, and current army progress. Request
a scan or scout when needed, then re-decide on a later wake. Never require
judging whether a Scanner Sweep is "safe".

If Scans and the attack gate both mention vision, they must agree: when energy
is available, request a scan and hold the push; when energy is not available,
apply the same fallback the gate names.

Each replacement must differ from the parent paragraph text. Omit a paragraph
rather than returning an unchanged replacement.
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

Fix only the reported patch validation errors.

You may add, remove, or revise paragraph patches when required to make the same
hypothesis executable and internally consistent. Include missing prerequisite,
resource/production, execution, and stale-target dependencies, but do not introduce
a second independent optimization objective. Preserve the strategy's defining army
concept and win plan.

{_PATCH_CONSISTENCY_RULES}

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

The selected causal hypothesis is the unit of experimentation. Treat plan.direction
as its coherent strategy package direction, not as a single paragraph or strategy
category.

Patch every paragraph that must change for the hypothesis to be executable,
internally consistent, resource-feasible, prerequisite-complete, and semantically
coherent. Do not introduce a second independent optimization objective merely
because it might also improve the strategy.

The supplied mechanism_prediction pre-registers what must materially change for
this candidate to count as a real test. Implement a package strong enough to meet
minimum_material_change in strategy intent. Do not satisfy it with a cosmetic,
token, or merely reworded adjustment. Do not exaggerate strength by adding changes
that are unrelated to the same causal mechanism.

Preserve the strategy's defining army concept and win plan unless the supplied
Cross-match Decision explicitly identifies a supported identity-level problem.

Your job is only to implement the supplied hypothesis as a coherent paragraph
patch to the current strategy.md.

{RUNTIME_CONTRACT}
{_PATCH_CONSISTENCY_RULES}

Read the complete strategy before editing.

Determine every existing Detail paragraph that must change in this order:
1. identify the primary change required by the hypothesis;
2. identify prerequisite dependencies;
3. identify resource and production dependencies;
4. identify execution dependencies;
5. identify stale targets or contradictions in dependent paragraphs;
6. patch every necessary dependency;
7. leave unrelated paragraphs unchanged while preserving supported strengths.

There is no fixed maximum number of paragraph patches.

However, every modified paragraph must be necessary for the same hypothesis.
Leave every unrelated paragraph unchanged.

For every proposed patch ask: "If this patch were removed, would the selected
hypothesis become incomplete, internally inconsistent, non-executable, or
materially different?" Include it only when the answer is yes. A dependency that
the current strategy already satisfies needs no patch.

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

