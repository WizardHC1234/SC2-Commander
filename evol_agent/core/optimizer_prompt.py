from __future__ import annotations

import json
from typing import Any

from .context import render_knowledge_results, render_optimizer_decision
from .replay_reasoning_examples import REPLAY_GROUNDED_REASONING_EXAMPLES
from .types import BattleAnalysis, ToolObservation


_PATCH_CONSISTENCY_RULES = """
Use the fixed Summary/Details document structure. Summary states the strategy's
identity and win mechanism; Details contains titled executable strategy instructions.
State each strategic fact once rather than duplicating it across paragraphs.

Main Attack Gate is authoritative only for the first planned attack. Recovery and
Cleanup must not copy, reference, synchronize with, or silently strengthen that
opening threshold. Post-engagement behavior must be justified independently by the
selected failure-stage evidence.

Every match ends at 1800 seconds. The complete strategy must leave a credible path
to locate and destroy all enemy structures before that limit. Scouting and
Information may support target selection and cleanup, but it must not delay a ready
attack or create a hidden launch prerequisite.

Every unit that the complete candidate says to continue or resume producing needs
an explicit stage target. Units that remain in ongoing reinforcement also need a
numerical final count or cap in Ultimate Goal. The complete document must make these
bounds unambiguous, and workers plus every final combat/support unit must not exceed
200 supply.

Use only information available in the current observation or supported last-seen
state. Scouting or scans may support a named decision, but they must not create an
unavailable cross-cycle state machine or replace the strategy's core attack logic.

Write mutually exclusive alternatives as clear branches and keep all thresholds
and targets consistent across the complete candidate.
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
    """Ask the Optimizer to implement one hypothesis as a complete strategy.md."""
    del knowledge_mode, tool_observations
    from .prompts import RUNTIME_CONTRACT
    errors = "\n".join(f"- {error}" for error in validation_errors) or "None"
    parent_text = str(skill_texts.get("strategy.md") or "")
    previous_candidate = ""
    if isinstance(candidate, dict):
        previous_candidate = str(candidate.get("strategy_md") or "").strip()
        if not previous_candidate:
            files = candidate.get("files")
            if isinstance(files, dict):
                previous_candidate = str(files.get("strategy.md") or "").strip()
    decision_payload = decision or dict(battle_analysis.raw or {})
    knowledge_text = render_knowledge_results(knowledge_runs or [])
    capability_text = json.dumps(capability_manifest or {}, ensure_ascii=False, indent=2)
    revision_context = ""
    if validation_errors:
        revision_context = f"""
The previous complete candidate was invalid. Regenerate the whole strategy.md while
keeping the same hypothesis and correcting every reported issue.

Previous candidate strategy.md:
{previous_candidate or "Unavailable because the previous response was malformed."}

Validator errors:
{errors}
"""

    action = "revise_candidate" if validation_errors else "draft_candidate"
    return f"""You are EvolAgent's Strategy Optimizer.

The Cross-Match Decision Agent has already selected one primary failure mode,
one hypothesis, and one coherent intervention package.

Do not redo the match analysis.
Do not choose a different problem.
Do not replace the hypothesis.
Do not introduce a second optimization objective.
Do not select among candidate plans.

{REPLAY_GROUNDED_REASONING_EXAMPLES}

Use these demonstrations only to implement the selected Cross-match Decision.
They do not authorize changing its failure stage, hypothesis, permissions, unit
counts, or contact window.

Generate the entire replacement strategy.md as one coherent Summary/Details
document. Do not return paragraph patches. Reconsider the complete relationship
among strategy identity, development, composition, first-attack readiness,
objective, reinforcement, post-engagement behavior, and final supply while
implementing only the selected causal package.

The supplied mechanism_prediction pre-registers what must materially change for
this candidate to count as a real test. Implement a package strong enough to meet
minimum_material_change in strategy intent. Do not satisfy it with a cosmetic,
token, isolated, or merely reworded adjustment. The candidate should produce the
declared plan.material_behavior_change. Do not exaggerate strength by adding
changes that are unrelated to the same causal mechanism.

Preserve strategy_contract.style, core_win_mechanism, critical_power_window, and
core_commitments unless the Cross-match Decision explicitly identifies a supported
identity-level problem. Flexible components may change when required by the tested
hypothesis.

Treat plan.composition_change_allowed and plan.retreat_change_allowed as explicit
scope permissions. When composition permission is false, preserve the Champion's
unit concept: do not introduce or remove a combat/support unit and do not turn an
unselected unit into a new attack requirement. Quantity changes required by the
selected non-composition mechanism may remain possible. When retreat permission is
false, preserve the Champion's explicit retreat rules and the runtime default; do
not add or change retreat_ratio. A true permission allows only the change described
by the selected hypothesis and stage_scope_reason, not an unrelated second lever.

Treat outcome_contrast.preservation_rule, strategy_contract.protected_invariants,
and plan.preservation_checks as required design constraints. Reconstruct the
Champion's successful causal chain before writing the candidate, then verify that
the candidate can still reproduce it. Fix the loss shortfall at its earliest causal
point; do not optimize only the final enemy composition while removing the timing,
production throughput, or reinforcement pattern that produced the wins.

Preserving style does not mean freezing the unit roster. Support units, upgrades,
technology, economy, and production may change when they reinforce the same combat
style and power window. Before making a new unit or upgrade part of Main Attack
Gate, verify that the Decision explicitly intends it to be a hard prerequisite.
Otherwise keep the Champion's launch condition and treat completed support units as
optional participants or later reinforcement. Never claim to preserve an early or
critical timing while adding an AND-condition that can indefinitely prevent it.

Account for production opportunity cost. If a support unit shares a production
structure or limiting resource with the strategy's core unit, the production order
must not silently starve the core or move first contact outside the selected power
window. Stage production targets describe what to produce; they must not create a
second hidden launch gate or contradict Main Attack Gate.

After generation, EvolAgent deterministically simulates the first-commitment package
from the standard Terran start. The simulator accounts for mineral and gas collection,
SCV allocation and saturation, construction workers, prerequisites, supply, add-ons,
and production queues. Make every unit or upgrade required before first commitment
explicit, together with the pre-commitment SCV, base, Refinery, structure, add-on, and
production-slot targets needed to produce it. Do not claim that timing is preserved
by calling a required component optional. If validation reports a later
earliest_feasible_time, revise the material plan or explicitly justify that minimum
delay against the selected opponent-growth evidence; never replace the program's
calculated time with prose arithmetic.

Treat the supplied current strategy as the official Champion and the only
inheritance source. Preserve unrelated useful behavior in the regenerated document.
Earlier non-accepted candidates are evidence, not parent text. Produce an explicit
inheritance ledger and never silently remove a Champion mechanism.

The primary strategic objective is a favorable or survivable decisive army
engagement. Timing, resource use, production synchronization, scouting, and gate
attainment are intermediate mechanisms, not standalone optimization goals. The
candidate must preserve a clear explanation of how its material change improves
the combat_success_measure in the selected mechanism prediction.

Implement the exact causal package selected by analysis. Check that its contact
timing, fighting package, post-contact reinforcement, and production feasibility
support the strategy's core win mechanism. These are consistency checks, not a
fixed category ranking. Do not replace the selected combat-outcome mechanism with
an easier surrogate such as more scouting, a later gate, or extra production.

If the selected package changes retreat behavior, describe it only in
Recovery and Cleanup. A selected evidence-supported rule may specify the
retreat_ratio that Commander should emit with offensive commands. Runtime
auto-retreat fires when the local own/enemy power ratio falls below retreat_ratio
(default 0.6). Do not assume that earlier or later retreat is inherently better:
use recorded loss timing, retained power, regroup delay, and re-engagement. If
retreat policy was not selected or plan.retreat_change_allowed is false, preserve
Recovery and Cleanup exactly.

Do not implement construction of static defensive structures as the primary
evolution mechanism. They cannot accompany the mobile fighting force, and the
executor places them around owned bases rather than arbitrary forward staging
zones. If the supplied plan depends on forward or mobile static defense, it is not
an executable strategy patch.

Keep the candidate concise. Prefer one observable rule over repeated warnings,
copied gates, or many narrow exceptions. Never add strategy text to compensate for
runtime-owned transformations, transport loading, abilities, targeting, formation,
or unit-level micro. If the selected hypothesis depends on one of those behaviors,
the candidate is execution-invalid rather than a strategy patch.

Your job is only to implement the supplied hypothesis as a coherent complete
strategy.md. A globally rewritten document must still make one causal change, not
bundle unrelated improvements.

{revision_context}

{RUNTIME_CONTRACT}

Read the complete Champion strategy before writing the replacement. Preserve the
two required sections: write a short prose # Summary, then a # Details section made
only of `* Title: instruction` bullets. Preserve useful existing titles and keep all
required strategic mechanisms in the regenerated document.

Before returning, perform one document-wide consistency pass:
- exactly one authoritative first-attack gate in Main Attack Gate;
- production stage targets are not hidden attack gates;
- Recovery and Cleanup does not reapply, copy, or strengthen the first-attack gate;
- all technology and unit prerequisites are feasible;
- support-unit production does not silently starve the core power window;
- every continuously produced unit has a stage target and a final count or cap;
- workers plus all final combat/support units use no more than 200 supply;
- the attack and cleanup plan can destroy all enemy structures within 1800 seconds;
- the combat style and evidence-supported contact window remain consistent.

{_PATCH_CONSISTENCY_RULES}

Do not mention match ids, exact recorded timestamps, EvolAgent internals,
or Replay-only enemy truth in strategy.md.
Do not write micro, map-specific zone IDs, or group IDs.
Express attack-readiness counts as completed, living units in the persistent main force.
Keep explicit end-state supply at or below 200.
Write reusable StarCraft II strategy instructions.

Strategy: {strategy_name}
Race: {race}

Executor capability manifest:
{capability_text}

Cross-match Decision:
{render_optimizer_decision(decision_payload)}

Current strategy.md:
{parent_text}

Verified static SC2 facts:
{knowledge_text}

Return one JSON object only:
{{
  "action": "{action}",
  "strategy_md":"complete Markdown document containing # Summary and # Details in that order",
  "inheritance":{{
    "keep":[{{"item":"parent mechanism retained","reason":"evidence or dependency"}}],
    "revise":[{{"item":"parent mechanism changed","reason":"relation to the selected hypothesis"}}],
    "remove":[{{"item":"parent mechanism removed","reason":"evidence-based reason; empty when none"}}]
  }},
  "preservation_audit":[
    {{"invariant":"protected winning mechanism from the Decision","effect":"preserved|improved|evidence_supported_revision|broken","candidate_rule":"where the complete candidate implements it","reason":"why the winning mechanism remains reproducible"}}
  ],
  "expected_effect":"expected match effect",
  "main_risk":"possible regression to evaluate in games"
}}
"""
