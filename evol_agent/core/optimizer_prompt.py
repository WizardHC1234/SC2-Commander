from __future__ import annotations

import json
from typing import Any

from .context import render_knowledge_results, render_optimizer_decision
from .types import BattleAnalysis, ToolObservation
from ..optimization.strategy_document import StrategyDocument


_PATCH_CONSISTENCY_RULES = """
Binding strategy-writing contract:

1. strategy_contract is the sole definition of strategy identity and preserved
behavior. priority_problem identifies the one problem, and plan.coordinated_changes
is the complete modification package. Do not infer a second objective from other
diagnostic fields.
2. Changes may implement only economy/expansion targets, production-building and
unit-count targets, technology/upgrades, army composition, or attack readiness and
strategic objective. Do not modify Scouting or Scans. Do not add wake-event,
decision-cycle, reinforcement, retreat, recovery, or cleanup behavior.
3. Main Attack Gate exclusively owns first-attack permission. Pre-Attack Army
Posture only gathers and stages. Existing reinforcement/recovery text may be patched
only to replace a stale target with a reference to Main Attack Gate or the selected
strategic objective; do not add a new rule or copy numerical launch thresholds.
4. Every Commander cycle evaluates the current observation independently. Do not
write Cycle 1/Cycle 2 protocols or require memory that a prior scan, wake, or branch
completed. scan_ready means a scan can be requested; it is not an observation result.
5. Use information_grounding only when plan.coordinated_changes actually depends on
enemy information. enemy_truth is diagnosis-only.
Only decision-time observable and runtime-supported facts may change a strategy
branch. If information does not change the army, target, or commitment, keep it
optional and non-blocking.
6. Keep conditions mutually exclusive and keep every changed threshold consistent.
Every resumed or continuing unit needs a numerical stage target and Ultimate Goal
cap. The deterministic validator checks these targets and the 200-supply limit.
7. Each replacement must differ from the parent paragraph. Omit unchanged text.
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
    del knowledge_mode, tool_observations
    from .prompts import RUNTIME_CONTRACT
    errors = "\n".join(f"- {error}" for error in validation_errors) or "None"
    parent_text = str(skill_texts.get("strategy.md") or "")
    document = StrategyDocument.parse(parent_text)
    patch_context = json.dumps(document.patch_context(), ensure_ascii=False, indent=2)
    candidate_text = json.dumps(candidate or {}, ensure_ascii=False, indent=2)
    decision_payload = decision or dict(battle_analysis.raw or {})
    knowledge_text = render_knowledge_results(knowledge_runs or [])
    capability_text = json.dumps(capability_manifest or {}, ensure_ascii=False, indent=2)
    retry_feedback = (
        """
Prior generation attempts failed. Repair these concrete errors in this candidate;
do not repeat their invalid assumptions:
{errors}
""".format(errors=errors)
        if validation_errors
        else ""
    )

    if isinstance(candidate, dict) and validation_errors:
        return f"""You are revising an invalid paragraph patch.

Keep the same Cross-match hypothesis and plan direction.

Keep strategy_contract binding and keep implementing the same
plan.coordinated_changes. Repairing an error must not introduce a second objective.

Fix only the reported patch validation errors.

You may add, remove, or revise paragraph patches when required to make the same
hypothesis executable and internally consistent. Include missing prerequisite,
resource/production, and stale-target dependencies inside the allowed domains, but
do not introduce a second independent optimization objective. Preserve the
strategy's defining army concept and win plan.

{_PATCH_CONSISTENCY_RULES}

Do not introduce a new strategic objective.
Do not modify unrelated paragraphs.
Do not modify # Summary.
Do not add, delete, rename, or reorder paragraphs.

Each target and expected_old_hash must match the parent paragraph catalog.

Cross-match Decision:
{render_optimizer_decision(decision_payload)}

Executor capability manifest:
{capability_text}

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

    return f"""You are EvolAgent's Strategy Optimizer. Implement the supplied
Cross-match Decision; do not redo analysis, choose another problem, or select among
candidate plans.
Do not select among candidate plans.

Binding priority:
1. Preserve strategy_contract. The strategy_contract was inferred before failure ranking and defines
the style, core win mechanism, critical timing/power spike, and core commitments.
2. Address priority_problem by implementing every item in plan.coordinated_changes.
3. Use information_grounding only when that plan depends on decision-time enemy
information; never turn enemy_truth into a live rule.
4. Add only dependencies required for execution, feasibility, and cross-paragraph
consistency. Preserve unrelated strategy-contract commitments.

Reason backward from decisive army combat. The available levers are economy and
expansion targets, production targets, technology and upgrades, army composition,
and attack readiness/objective. Scouting, scanning, wake behavior, reinforcement,
retreat, recovery, cleanup, static defense, runtime-owned micro, transformations,
transport, targeting, and formation are not primary strategy patches.

{retry_feedback}
{RUNTIME_CONTRACT}
{_PATCH_CONSISTENCY_RULES}

Read the complete strategy. Map each coordinated change to existing paragraphs,
add necessary prerequisites inside the five allowed domains, and leave unrelated
text unchanged. A reinforcement or recovery paragraph may change only when an
allowed edit created a stale reference, and then only to reference Main Attack Gate
or the selected objective. There is no fixed patch-count limit.
There is no fixed maximum number of paragraph patches.

For each patch, target an existing Detail id, copy its exact parent hash, return
one complete replacement instruction, and explain its direct dependency. Do not
modify Summary or paragraph structure. Do not mention match ids, timestamps,
EvolAgent internals, Replay-only truth, map-specific IDs, or group IDs. Express
readiness with completed living units and keep final supply at or below 200.

Strategy: {strategy_name}
Race: {race}

Executor capability manifest:
{capability_text}

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
