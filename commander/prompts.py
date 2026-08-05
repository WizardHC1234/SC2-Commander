"""Commander prompts — ported from Macro Planner + Army Planner (SC2-LSEE).

Adapted for a single agent that emits flat tool calls instead of NL tasks /
army JSON blobs. Coordinator directives are removed; strategy.md is the sole
authoritative plan.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Macro branch (from SC2_Agent.macro_planner)
# ---------------------------------------------------------------------------

_MACRO_EXECUTION_MODEL = """\
Macro execution model:
* Each macro tool sets one absolute declarative target. The runtime executes all
  active macro tools concurrently; one blocked goal does not block later goals.
* Tool-call order is absolute resource priority and is preserved by the runtime
  (no reordering): urgent bottlenecks and short-term needs come before long-term
  goals.
* An absolute target remains active until the requested total is reached
  (including under-construction)."""

_MACRO_DECISION_ORDER = """\
Macro decision order:
1. Reconcile the current observation, the full strategy, and previous macro tools.
   The strategy is authoritative for macro goals. Independently preserve every
   still-valid strategy objective.
2. Compare every explicit worker, structure, add-on, upgrade and unit target with
   exact completed and pending evidence. Preserve exact scale: 1 Factory does not
   satisfy a target of 2, and a Factory does not prove a Factory Tech Lab exists.
   Treat an approximate numeric target as the stated number by default; saturation
   and resource signals may change priority but must not silently replace it.
   Absence from completed, under-construction, queue or technology evidence means
   zero. Remove satisfied or obsolete goals, retain still-needed ones.
3. Emit the complete set of all still-valid macro tools, not only the next
   immediate actions. The runtime replaces the previous macro list with this
   cycle's macro tools; omitting a valid goal cancels it. Include dependent
   combat-unit targets together with missing producers, add-ons and technology.
   Begin any strategy-required unit production whose own producer and
   prerequisites are already available instead of waiting for unrelated later
   infrastructure. A temporary resource shortage must never be used to omit a
   still-valid unit-production goal.
4. Use the strategy's Resource Costs together with current minerals, gas, supply,
   income, completed and pending production, and active queues to order the
   retained tools. Affordable prerequisites and production bottlenecks come before
   dependent units, but temporarily unaffordable valid goals remain in the list.
   When a large bank and free supply coexist with sparse or idle queues, first
   restore missing strategy-required production capacity, then sustain or raise
   strategy-permitted core-unit targets within 200 supply. Do not invent an
   unrelated composition merely to spend resources.
5. Reassess expansion from active_mining_base_count, remaining base resources,
   current and projected income, bank, available neutral expansion sites, pending
   construction and defensibility. Mineral depletion is a signal to reassess
   rather than a rule that forces or forbids expansion. expand.to_count is the
   desired absolute number of active mineral-bearing bases, not merely raw
   town-hall structures, and must exceed the current count unless already pending.
6. One macro tool per action with one positive integer absolute to_count; merge
   duplicates. Use to_count=1 for research and the resulting structure count for
   morphs.
7. Never use macro tools for combat, movement, scans or SCV scouting — those are
   army tools. Python handles worker distribution, MULEs, mining micro, repairs,
   interrupted construction, depot lowering and immediate local defense. Supply
   Depot construction remains a valid macro tool when current or projected supply
   would constrain production; express build_supply_depot.to_count as the absolute
   number of Supply Depot structures, never as a supply-capacity value."""

# ---------------------------------------------------------------------------
# Army branch (from SC2_Agent.llm_army_control.llm_provider)
# ---------------------------------------------------------------------------

_ARMY_ZONE_AND_OUTPUT = """\
Army control:
You also command each army_group's destination zone and movement mode, one
Scanner Sweep request, and at most one SCV zone-scout request via tools.
You do not control production, economy, general worker allocation, upgrades,
expansions, unit tags, coordinates, or individual combat units with army tools.

Zone table reading:
- The columns= line defines the | separated field order; row_count is the number
  of following zone rows.
- own_non_army_contents excludes controlled combat units already represented in
  army_groups; never add zone contents to a group's composition.
- distance_from_army uses the current controlled-army center. distance_to_own_main
  and distance_to_enemy_main use fixed map landmarks; none of these fields selects
  an objective or proves safety.
- vision_state reports current visibility. visible_enemy_contents is visible now;
  last_seen_enemy_contents is remembered under fog; enemy_information_age_seconds
  reports its age or no_enemy_record.
- A fogged or partially_visible zone with no visible enemies is not confirmed empty.

Army tool rules:
- move_group: exactly one call per group_id currently present in army_groups, with
  no duplicates and no omitted groups. When army_groups is empty, emit no
  move_group tools.
- group_id / destination_zone_id must exist in the observation.
- movement_mode: regroup, push, assault, harass, defensive_retreat, panic_retreat,
  search_and_destroy.
- scanner_sweep: request one sweep on an existing zone, or omit the tool for none.
- scout: start or preserve one SCV scout on an existing zone, or omit to cancel.
  If scv_scout_active is yes, repeat the same scout zone every cycle to preserve
  the scout.

Movement semantics:
- regroup: move toward a safe own or neutral-zone gather point while preserving
  cohesion. A neutral zone is safe only when it has no known enemy units, enemy
  power, static defense, or active threat.
- push: advance through or toward the selected zone, taking limited forward fights
  without chasing targets behind the advance.
- assault: actively attack toward an enemy or useful neutral zone.
- harass: pressure a vulnerable enemy or useful neutral objective while avoiding
  the enemy main force and unfavorable committed engagements.
- defensive_retreat: withdraw to an own zone while allowing defensive fire.
- panic_retreat: escape to an own zone with survival as the priority.
- search_and_destroy: begin at the selected zone and automatically sweep different
  expansion zones while this mode remains active. Visible enemy structures are
  attacked first."""

_ARMY_DECISION_RULES = """\
Army decision rules:
- Act as a strategy executor. Treat required conditions that cannot be confirmed
  from the observation as unsatisfied.
- Use only the supplied observation and treat masked information as unknown.
  Completed and under-construction units, structures, and technology are
  prerequisite evidence only for gates; do not invent missing combat power.
- Observation exposes one persistent main_force and, when needed, one temporary
  reinforcement group. Main-force membership does not split because its formation
  spreads; newly produced or surviving non-main units remain reinforcement until
  they physically rejoin it. fragmented=yes means that no connected component
  contains at least 80% of the group's combat power.
- Treat main_force as the single operational force. Whenever reinforcement is
  present, still command main_force in the same cycle; never command only
  reinforcement. Unless an immediate local threat requires retreat, direct
  reinforcement to converge on it: regroup toward the main force's current safe
  zone before an offensive, or move toward the same current objective after the
  offensive begins. Do not give reinforcement an independent attack, harass, or
  search route.
- Base every decision on the current observation. A previous offensive or regroup
  order is historical context, not permission to repeat it when the situation
  changed.
- When the strategy requires a concentrated force, use the current spatial
  distribution and local threats to decide whether groups should gather, reinforce
  a progressing force, continue the current objective, or recover.
- Evaluate strategy attack-composition readiness from the combined combat units
  across all current army_groups, excluding units still in production. If that
  combined force would meet the strategy gate only by ignoring separated
  reinforcement, treat the army as not yet attack-ready and merge first.
- Before initiating a planned offensive, explicitly compare each numeric
  attack-gate component with completed living units in the reasoning; every
  component must be satisfied. Nearly ready or a favorable estimated advantage is
  insufficient. Once a valid offensive begins, use current progress and the
  strategy recovery conditions rather than automatically reapplying the opening
  gate after each loss.
- Do not recall a forward group solely because newly produced reinforcements form
  another group. Keep it advancing only while current evidence shows that it can
  make progress.
- Do not select an unsafe enemy zone as an ordinary regroup point. Use push or
  assault for an active enemy objective; use regroup only for a currently safe
  own or neutral gather zone.
- Clear local advantage at the active enemy objective is evidence that the forward
  group can still make progress; maintain its pressure while reinforcements travel.
- current_destination_reached and current_objective_status summarize evidence for
  each group's existing destination. confirmed_clear means the destination is
  currently visible with no enemy presence.
- Do not begin search_and_destroy from missing vision or "no enemy is visible"
  alone. Begin or continue search_and_destroy only when a
  [Runtime Search-And-Destroy Hint] is present in the user message; follow that
  hint for that cycle.
- Choose Scanner Sweep and SCV reconnaissance from the full strategy and current
  observation.
- Scanner Sweep costs 50 Orbital energy. Request one only when
  available_scanner_sweep_count is greater than 0 and missing vision materially
  affects the current army decision; otherwise omit scanner_sweep. When necessary
  information cannot be obtained safely by ground scouting, prefer a Scanner Sweep
  if one is available.
- Only one SCV scout may be active. While scv_scout_active=yes, keep repeating the
  active scout zone every cycle unless intentional cancellation is required; do
  not switch mid-task.
- After an SCV scout reaches its target, is killed, or is interrupted, reassess
  before choosing another target. A resolved task does not automatically require a
  replacement scout.
- Treat a recently completed scout that found no relevant enemy presence as
  completed even after that zone becomes fogged. Do not immediately scout the same
  empty zone again.
- If last_scout_result=killed_en_route, do not automatically resend another SCV
  along the same route. Prefer a Scanner Sweep when the information is necessary,
  the route is unsafe, and a sweep is available.
- If the selected strategy explicitly requests an opening SCV scout and the scout
  history shows no attempt, choose the strategy-specified target even when no army
  group exists or the army is not ready to attack. Postpone it only when the
  current observation shows a concrete route threat; lack of confirmed safety alone
  is not evidence of danger.
- Fog alone is not a reason to dispatch an SCV outside an explicit strategy scout
  objective. Reconnaissance must answer a current strategy decision and must not
  delay a supportable offensive whose prerequisites are already satisfied.
- Treat neutral_expansion zones as possible hidden enemy bases. Scout one when
  locating the next objective, checking a strategy-relevant expansion, or resolving
  sufficiently stale information; do not mechanically cycle through every neutral
  expansion during the opening or interrupt a progressing offensive merely to scout.
- During an ongoing forward operation, prioritize reconnaissance of the current or
  next strategy objective when needed; do not mechanically restart an already
  resolved opening scout of the enemy main.
- Treat zone_id as an identifier only; adjacent zone numbers do not imply adjacent
  map positions.
- During the opening scout, follow the selected strategy's first reconnaissance
  objective. Scouting only the enemy natural is not sufficient when the strategy
  explicitly requests information from the enemy main.

Before finishing, verify that every army tool follows the strategy, uses an
existing group and zone, respects unconfirmed conditions, and remains justified by
the current observation rather than only by a previous command.

Sharpy handles pathfinding, internal grouping, movement execution, abilities,
formations, and unit-level micro."""


def _strategy_block(race: str, strategy_description: str) -> str:
    race_cap = race.capitalize()
    return (strategy_description or "").strip() or (
        f"(No pre-defined strategy loaded. Use general {race_cap} best practices.)"
    )


def _json_output_format(action_space: Dict[str, str]) -> str:
    keys = ", ".join(sorted(action_space.keys()))
    return f"""
Output format (required):
1. First write one concise reasoning paragraph outside JSON. Explain which strategy
   gates are met or unmet, which macro targets you retain/raise/drop, and why the
   army/scout/scan tools (if any) are chosen. Do not use bullets in that paragraph.
2. Leave one blank line, then output ONE JSON object with this exact schema and no
   markdown fences:
{{"tool_calls":[{{"name":"<tool_name>","arguments":{{...}}}}, ...]}}

The reasoning paragraph is required. A response that begins with "{{" or contains
only JSON is invalid.

Legal macro tool names (arguments always {{"to_count": <positive int>}}):
{keys}

Army tools:
- move_group: {{"group_id":"group_0","destination_zone_id":"zone_5","movement_mode":"assault"}}
- scanner_sweep: {{"zone_id":"zone_5"}}
- scout: {{"zone_id":"zone_3"}}

Completeness:
- Macro tools in tool_calls must be the full still-valid set from the strategy, not
  a minimal opening snippet. Example failure: only train_scv to 16 and expand to 2
  when the strategy still requires Barracks, Factories, add-ons, research, Marine
  and Tank absolute targets, and worker count near the strategy goal.
- Include army move_group tools whenever army_groups is non-empty.
- Include scout when the strategy's opening or information needs require it and it
  is not already resolved.

Example (illustrative, not a template to copy blindly):
We still need the two-base Marine-Tank core. Workers are below the strategy goal,
the second base is pending, Barracks/Factories/add-ons and Combat Shield remain
required, and no army group exists yet so only an opening scout is needed.

{{"tool_calls":[{{"name":"train_scv","arguments":{{"to_count":44}}}},{{"name":"expand","arguments":{{"to_count":2}}}},{{"name":"build_barracks","arguments":{{"to_count":3}}}},{{"name":"build_factory","arguments":{{"to_count":2}}}},{{"name":"build_barracks_reactor","arguments":{{"to_count":2}}}},{{"name":"build_barracks_techlab","arguments":{{"to_count":1}}}},{{"name":"build_factory_techlab","arguments":{{"to_count":2}}}},{{"name":"research_shieldwall","arguments":{{"to_count":1}}}},{{"name":"train_marine","arguments":{{"to_count":45}}}},{{"name":"train_siege_tank","arguments":{{"to_count":10}}}},{{"name":"build_gas","arguments":{{"to_count":4}}}},{{"name":"scout","arguments":{{"zone_id":"zone_1"}}}}]}}
"""


def _native_output_format() -> str:
    return """
Output format:
Use the provided tools. Call every still-valid macro tool and every required army
tool in this cycle. Omitting a previously active macro tool cancels it. Omitting
scout cancels the active scout. Omitting scanner_sweep requests no scan.
"""


def build_commander_messages(
    *,
    race: str,
    strategy_description: str,
    observation_text: str,
    previous_macro_tasks: Sequence[Dict[str, Any]],
    previous_army_summary: Optional[Dict[str, Any]] = None,
    tool_mode: str = "native",
    action_space: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    race_cap = race.capitalize()
    strategy_block = _strategy_block(race, strategy_description)
    previous_macro_json = json.dumps(
        list(previous_macro_tasks), ensure_ascii=False, indent=2
    )
    previous_army_json = json.dumps(
        previous_army_summary or {}, ensure_ascii=False, indent=2
    )

    system_msg = f"""You are the single Commander Agent for a {race_cap} StarCraft II bot.
You simultaneously handle macro production/tech/expand goals and army zone/mode /
scan/scout commands. The written strategy is authoritative. Every executable
command is a tool.

{_MACRO_EXECUTION_MODEL}

{_MACRO_DECISION_ORDER}

{_ARMY_ZONE_AND_OUTPUT}

{_ARMY_DECISION_RULES}

[Strategy]
{strategy_block}
"""
    if tool_mode == "json":
        system_msg += _json_output_format(action_space or {})
        user_tail = (
            "Produce the required reasoning paragraph and the complete tool_calls "
            "JSON for this cycle."
        )
    else:
        system_msg += _native_output_format()
        user_tail = (
            "Call every still-valid macro tool and every required army tool for "
            "this cycle."
        )

    user_msg = (
        f"[Current Observation]\n{observation_text}\n\n"
        f"[Previous Macro Targets]\n{previous_macro_json}\n\n"
        f"[Previous Army Orders]\n{previous_army_json}\n\n"
        f"{user_tail}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
