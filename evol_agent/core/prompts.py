from __future__ import annotations

from .context import (
    render_allowed_match_evidence,
    render_battle_analysis,
    render_sc2_knowledge,
    render_knowledge_runs,
    render_single_game_analyses,
    render_skill_context,
)
from .config import (
    MAX_DIAGNOSED_PROBLEMS,
)
from .types import BattleAnalysis, ToolObservation


STRATEGY_MARKDOWN_FORMAT = """House strategy.md format (must match the structure of the supplied current strategy.md):
- Output exactly two level-one headings in this order, each followed by a blank line: # Summary, # Details.
- Do not add #/# ## / ### headings, tables, code fences, numbered lists, or checkbox lists.
- # Summary: one short prose paragraph (or two). No bullets in Summary.
- # Details: only asterisk bullets of the form "* Title: sentence...". Title is a short topic label ending with a colon (for example "* Main Attack Gate: ..."). Prefer preserving the current strategy's topic titles and edit inside those bullets; add a new titled bullet only when a new topic is required.
- Typical Detail topics to keep when present: Opening and Economy; Expansion; Production; Technology; Scouting; Scans; Pre-Attack Army Posture; Main Attack Gate; Attack Objective; Engagement and Reinforcement; Recovery and Cleanup; Ultimate Goal.
- Do not add Resource Costs or Required Tools sections. Costs, supply, durations, producers, prerequisites, and dependency closure belong to the runtime action catalog rather than strategy.md.
- Write ordinary StarCraft II English. Do not invent a different outline style.
"""

RUNTIME_CONTRACT = """Current runtime contract:
- strategy.md is the sole strategic-intent source. EvolAgent may optimize only strategy.md; executable mechanics come from the race action catalog.
- One Commander receives the full structured observation and the full strategy. In one decision it emits macro targets, army commands, optional scan/scout requests, and exactly one wake event. Current decision records use trigger_reason values such as commander_bootstrap, wake_event, wake_fallback_timeout, and auto_retreat_triggered.
- Macro tools use positive integer absolute to_count targets. All emitted targets execute concurrently and the previous macro list is replaced each decision, so every still-valid unmet target must be emitted again. Tool order is resource priority, not a sequential barrier.
- The runtime expands structural prerequisites and automatically includes build_gas when any selected action has positive vespene cost. Tool descriptions—not strategy prose—are authoritative for costs, supply, duration, producers, research locations, and prerequisites.
- Army commands use observed group_id and destination zone_id values. The persistent main group remains the operational force; group_1 is a temporary reinforcement group for newly produced units far from the main force and should normally merge into or reinforce the same objective, not launch an independent campaign.
- Army movement modes are regroup, push, assault, hold, contain, harass, defensive_retreat, panic_retreat, and search_and_destroy. The strategy uses semantic locations; the Commander resolves them to current zone IDs.
- Scanner Sweep costs 50 Orbital energy. At most one SCV zone scout is active. Every decision arms a model-authored wake condition with a runtime deadline fallback.
- Supply structures are legal macro tools and supply must stay ahead of demand. Python/Sharpy still handle MULE usage, worker distribution, mining micro, repairs, interrupted construction, depot lowering, local defense, pathfinding, formations, abilities, and unit-level micro.
- Sharpy handles pathfinding, formations, abilities, and unit-level micro. The strategy cannot directly order coordinates, unit tags, individual-unit actions, transport or garrison loading, transformation or mode toggles, targeted abilities, or other unexposed micro.
- Runtime code lives under commander/ and is never copied into or modified with an optimized strategy.
"""

EVIDENCE_RULES = """Evidence interpretation:
- Current Commander records contain one full structured observation per decision. Missing fields remain unknown, not evidence that the game state was empty or zero.
- `enemy` is what Commander knew under fog of war. `opponent_truth_after_match`, when present, is post-match Replay evidence from the opponent's own player view; use it to measure information gaps and actual enemy economy/composition, but never claim Commander knew it at decision time.
- Absence from the chunks you inspected does not prove that an action never occurred elsewhere in the trajectory.
- Macro Targets and Army Commands are requested outputs; compare them with later Production, Own Forces, Army Group, Combat, and Execution evidence before calling them successful.
- Rejected decisions, reflection issues, command issues, automatic retreat overrides, and LLM errors are runtime evidence rather than strategy behavior.
- Trace a problem through strategy -> Commander decision -> runtime execution -> later observation before assigning a cause.
- If strategy.md already gave a clear, executable rule and a downstream agent ignored it, record that as an unresolved runtime limit instead of repeating the rule in the strategy.
"""

AGGREGATE_ANALYSIS_RULES = """Cross-match analysis rules:
- First identify the current strategy's identity from strategy.md: its economy shape, core fighting force, intended power stage, pressure/defense posture, and route to victory. Express this as one compact strategy_contract rather than a second strategy document.
- Classify the optimization direction as preserve, adjust, or replace. Use preserve when the identity is supported and only execution details need repair; use adjust when support, upgrades, economy amounts, scouting, or gates should change while the core identity remains; use replace only when cross-match evidence shows the core win plan itself is failing and successful evidence does not support preserving it.
- The optimization_boundary must state which changes remain inside the current style and which would change its identity. Do not use fixed loss-count thresholds to choose a direction; justify it from wins, losses, and the causal evidence.
- Prefer patterns repeated across matches. A single high-impact failure may be retained only when its causal chain is clear; label its evidence scope.
- Compare wins, losses, and successful periods before changing strategy identity. Preserve only rules supported by successful evidence unless they caused a demonstrated tradeoff.
- Merge targets that share one root cause. Return at most five high-priority optimization targets; do not turn every observation into an optimization target.
- Before assigning problem_ids, merge descriptions that are different points in the same causal chain. A cause, its immediate symptom, and its downstream consequence should not become separate optimization problems unless they require genuinely different strategy corrections.
- Keep strategy defects separate from unresolved runtime or downstream execution limits. Do not ask the strategy file to compensate for behavior it cannot control.
- An optimization target is valid only if the single Commander, the legal tool catalog, Python automation, or Sharpy can actually execute the proposed correction within the runtime contract. Otherwise mark it strategy_fixable=false / unresolved runtime limit.
- Diagnose SC2 causes through the actual progression of a match: economy and expansion; production capacity and utilization; supply, technology and upgrades; composition and support; gathered main-force readiness and reinforcement flow; scouting, detection and information freshness; engagement selection and trades; defense and map control; recovery and destruction of remaining enemy bases.
- When match summaries show repeated mineral starvation—especially a near-empty mineral bank with a large unused gas surplus and free supply, while production sits idle or underused—treat expansion timing, worker saturation, and macro resource priority as a candidate optimization target. Prefer earlier or additional mining bases / workers when income cannot fund the mineral-heavy core build; do not treat stacked gas alone as a reason to force more gas units. Distinguish delayed expand under saturation or depleting patches, expand that started too early and starved the core build, and concurrent expand-plus-heavy-spend without an explicit priority. Cite bank, income, saturation, depletion, pending versus completed expand, and production-idle evidence. Supply structures are both automated by the runtime and available as legal macro tools; chronic supply blocks may therefore be strategy-fixable through an earlier supply target when the strategy's worker and army growth predictably outruns automation.
- Treat prolonged information gaps as strategy-fixable when they precede surprise contact or wrong engagement selection for that plan. Scout/scan corrections must fit the strategy style (timing attack, two-base all-in, macro/third, defensive gather, etc.): choose priority, frequency, and scan-versus-SCV mix from the plan's information needs. Style sets priority order, not a hard destination whitelist—empty/forward mineral expansions remain valid scout targets when the plan needs to know whether they are taken, safe to expand, or hiding army. Do not collapse scouting to only enemy_natural and the planned third.
- Prefer combat-winning corrections over static or base-defense stacking. Engagements, army composition, trades, reinforcement, and initiative matter more than adding walls, bunkers, missile turrets, photon cannons, spine/spore crawlers, or similar static defense as the main fix. Use static defense only as a small, bounded exception when detection or a brief hold is otherwise unavailable—and never as a substitute for a fight plan.
- When losses involve composition, poor trades, air or ground control, harassment, or failed attacks, make the optimization target a combat-winning plan: adjust or extend the fighting composition (one or more unit types), improve gather/reinforce/retreat, research fight-relevant upgrades on the core army, or change engagement selection—not a defensive delay or isolated reaction.
- Prefer mobile composition answers that keep the army effective in the field: healers and other escort units that move with the force, hard counters, and researches that unlock or strengthen unit abilities. Do not treat scans, static defense, or base buildings as substitutes for those options.
- When poor trades show a missing fight tool on units that are already being produced, consider Macro research that unlocks abilities or key combat upgrades before opening an unrelated tech tree solely to add a new unit type.
- Macro production must be a fixed plan: absolute train/build/research targets gated only by own-side state (tech complete, building counts, bank, gather phase). Do not write enemy-detection forks that switch between named production targets.
- Do not rely only on reacting after an enemy unit is observed when the response would arrive too late. For repeated or high-risk enemy tech paths, prefer a bounded preemptive preparation such as scouting timing, prerequisite tech, flexible production path, upgrade priority, or a small multi-unit buffer that joins the army.
- Production lead time matters: if a response needs a new building line or multiple units, do not make first sighting of the threat the start trigger. Lock the fixed production targets from own-side gates before the first planned fight.
- Every proposed correction must be executable by the Commander through absolute macro targets, group destination/movement commands, scan/scout requests, and wake conditions, or by Python automation / Sharpy. If none can execute it, mark strategy_fixable=false. Do not send Commander-runtime questions to the SC2 Data Agent.
- For every claimed cause, connect an observed state or decision to a later SC2 consequence. Do not confuse correlation, a losing final state, or a downstream symptom with the cause.
- Check the command chain separately: strategy text -> Commander reasoning -> emitted macro/army/wake tools -> validation or command issue -> later execution state. Mark whether the problem is strategy-fixable.
- Each optimization target must name the observed problem, one reusable strategy correction, and the match-summary evidence that supports it.
"""

SC2_STRATEGY_DESIGN_RULES = """SC2 strategy design:
- Preserve a coherent race-appropriate win condition and phase progression. Use observable game-state triggers; use exact timings only when evidence supports a timing dependency.
- Preserve only the parts of the current strategy that are supported by successful evidence. Change composition, tech, economy, or timing when the records show the current plan cannot reliably win.
- The revised strategy must be able to win the game by fighting: take favorable engagements, reinforce through trades, regain initiative, and destroy enemy bases. Do not steer the plan toward a defense-first identity. Static defense, detection buildings, scans, bunkers/cannons/spines, and attack delays are bounded secondary measures, not the primary win condition.
- When losses involve army composition, poor trades, air or ground control, harassment, or failed attacks, write a combat-winning plan. Prefer extending the mobile fighting composition—possibly with several unit types—and/or researching upgrades that unlock or strengthen abilities on the core army. Treat healers and other army escorts as ordinary composition choices alongside hard counters. Do not default to turtling or static base defense. Encode research as Macro absolute targets with prerequisites and resource priority; leave ability casting and micro to Sharpy.
- Macro build and train rules must list fixed absolute targets. Gate them with own-side conditions only (structures complete, add-ons, upgrades done, bank thresholds, attack-gate gather phase). Never choose what to produce from enemy observations, remembered enemy composition, or "otherwise train X" branches.
- Classify every durable rule by the control available to the single Commander before writing it:
  - Macro tools: fixed absolute worker, supply, production, technology, expansion, add-on, unit, and research targets, with explicit resource priority when paths conflict.
  - Army tools: semantic destination intent, allowed movement modes, gathered-main-force attack gates, reinforcement toward the same objective, and need-based scan/scout requests—not formations, abilities, or unit tags.
  - Wake tool: observable conditions for the next useful decision; the strategy may describe meaningful checkpoints but must not encode runtime predicate names.
  - Python automation / Sharpy: do not restate MULE, worker distribution, repair, local-defense, or unexposed micro/abilities as strategy objectives.
- Do not rely only on reacting after an enemy unit is observed when the response would arrive too late. For repeated or high-risk enemy tech paths, include a bounded preemptive preparation such as scouting timing, prerequisite tech, fixed production path, upgrade priority, or a small multi-unit buffer that joins the army.
- Production lead time matters: if a response needs a new building line or multiple units, do not make first contact with the threat the start trigger. Lock fixed absolute targets from own-side gates before the first planned fight.
- Any added escort unit, research path, or counter unit must use fixed absolute counts and own-side start/stop gates (prerequisites, bank, gather phase). Do not start or switch Macro production when enemy X is seen.
- Keep attack readiness consistent with the required core army, escorts, and researched fight upgrades. Do not use a clock-only hard push deadline that forces an attack before the stated core composition can realistically exist.
- Resolve shared production, add-on, tech-path, upgrade, and gas-budget conflicts with explicit ordering so the core win condition is funded before optional extras.
- Cover the complete combat cycle: gather, select a favorable engagement, reinforce, retreat or rebuild after losses, regain initiative, obtain urgent vision or detection when justified, and destroy remaining enemy bases.
- Keep map information matched to the strategy style. Style changes which checks come first (e.g. timing plans often prioritize enemy_natural/tech and pre-push scans; macro plans often check thirds earlier), but it must not forbid scouting other empty or forward mineral expansions when those answers matter. Opening scout alone is not a default answer—derive scout/scan rules from the intended win condition and phase. Runtime still allows at most one SCV zone scout at a time.
- Keep the strategy concise and internally consistent. Prefer one clear rule with observable conditions over repeated warnings or many narrow exceptions.
- Produce reusable strategy rules for any race/build under this runtime—not instructions tied to one named opening, opponent, match, map, exact timestamp, group ID, or zone ID. Semantic zone roles such as own_main, own_natural, enemy_natural, and enemy_main may be used.
"""

EXECUTABLE_STRATEGY_RULES = """Executable strategy boundaries for SC2-Commander:
Every rule written into strategy.md must be executable by the single Commander, its legal tool catalog, Python automation, or Sharpy. A good-sounding plan that these controls cannot carry out is a failed optimization.

Commander macro control:
- Use absolute numeric targets whenever a specific unit, structure, supply structure, add-on, upgrade, morph, worker count, or expansion is required.
- Production targets must be fixed. Do not write detection-gated Macro forks that select different production targets from enemy observations. Commit to one absolute composition and own-side start gates.
- State prerequisite and priority ordering when shared production, add-ons, tech labs, reactors, or gas budgets create a conflict.
- The Commander emits the complete set of still-valid macro targets each decision; omission cancels a previous target. Targets execute concurrently and tool order provides resource priority, but the runtime has no general cross-task barrier, pause-until, or mutual-exclusion primitive. Do not claim that one target is guaranteed to finish before another starts; express the supported prerequisite, absolute targets, and resource priority instead.
- Add-on actions target aggregate add-on counts. They cannot select a particular Barracks, Factory, Starport, or swap add-ons between specific building instances.
- Supply Depot is a valid macro target even though AutoDepot also supplies a safety net. MULE usage, worker distribution, mining micro, repairs, interrupted construction, depot lowering, and immediate local-defense micro remain automatic.
- Each macro tool is one legal action plus an absolute to_count. The Commander cannot place buildings at coordinates, choose a particular producer instance, or invent actions outside the selected race action catalog. Costs, supply, base time, producer, prerequisites, and dependency closure come from that catalog and must not be copied into strategy.md.

Commander army control:
- Strategy text should express semantic destinations or objective types, never literal zone_id values such as zone_3. At runtime the Commander maps that intent to observed group_id and zone_id values.
- Base readiness and attack gates on the gathered persistent main force, its composition and support, upgrades, nearby threats, and visible or remembered enemy information with age. Do not require exact unit formation, type-specific positions inside a group, adjacency percentages, or a zone-adjacency graph: those are not exposed.
- group_1 is a temporary reinforcement group for newly produced or surviving units that are physically far from the persistent main force. It should converge on the main force or its current objective, not conduct an independent attack, harassment, or search campaign; it merges automatically after rejoining.
- Do not use a clock-only or partial-count attack gate that would commit an incomplete gathered force.
- Allowed movement intent maps only to: regroup, push, assault, harass, hold, contain, defensive_retreat, panic_retreat, search_and_destroy. Do not invent modes such as kite, surround, or siege-up-here.
- Each army-group command has one destination zone. Do not prescribe multi-hop routes, waypoints, or different destinations for different unit types inside the same group. Separate combat groups may be ordered to the same safe regroup zone.
- Use Scanner Sweep only as a strategy-level vision or detection requirement; each request costs 50 Orbital energy and is need-based. Use SCV scouting for non-urgent information when the route and replacement cost are acceptable. Do not add artificial scan cooldowns or claim that a request was executed merely because it was output.
- Scouting must fit the strategy style and win condition. Style decides priority and cadence (what to check first, how often, SCV versus scan)—not a whitelist limited to enemy_natural and the next Command Center. Empty/forward mineral expansions, enemy_main, and army-location checks remain allowed when they answer a real decision for that plan. Runtime allows at most one SCV zone scout at a time, so sequence destinations when multiple checks are needed; do not require simultaneous multi-SCV scouts.
- Keep information age bounded relative to the plan: if the strategy is about to push, expand, or commit composition, stale intel that would change that decision should trigger the next scout or scan rather than blind execution. Avoid blanket bans on replacement scouts when the style still needs follow-up vision.
- A scan request and army commands can be emitted in one decision. If movement must wait for scan evidence, describe two observable decisions: hold or regroup at a safe zone while requesting the scan, then advance only after later evidence confirms sufficient vision.
- Do not duplicate generic runtime protocol in the strategy unless this strategy needs a specific exception or threshold. The Commander system prompt already handles group identity, command persistence, group_1 reinforcement, retreat, and search-and-destroy persistence.
- Army tools cannot control production, economy, upgrades, expansions, unit tags, coordinates, transport or garrison loading, transformation or mode toggles, targeted abilities, or other unexposed micro. Leave production to macro tools and micro to Python automation / Sharpy.

Commander wake control:
- Every accepted Commander decision includes one wake condition based on observable state. Strategy text may define strategically meaningful checkpoints, but must not mention internal predicate names or depend on unobservable events.

Economy and supply:
- Use actual base-resource observations for depletion and expansion decisions; do not assume a fixed mineral or gas total for every map. Expansion guidance must account for remaining resources, saturation, production demand, safety, and whether an expansion is already pending.
- When optimizing expansion timing from records of mineral shortage (including near-zero minerals with a large unused gas bank), write observable Macro-facing gates rather than clock-only expand deadlines: active mining base count, worker saturation, remaining minerals/gas on owned bases, mineral/gas bank, whether an expansion is already pending, and competing production demand. Prefer rules such as expand or add workers when current bases are saturated or patches are depleting and mineral income cannot fund the core build—while avoiding a pure gas-unit diversion just because gas is stacked. Delay expand only while the mineral bank is funding a required short production spike that would otherwise be starved by a 400-mineral Command Center.
- If mineral starvation coexists with an expand objective, state an explicit macro resource priority among expand, workers, and the core combat build so the Commander does not keep a 400-mineral Command Center competing blindly with Barracks/Factory/Starport production (or the reverse when income is the bottleneck).
- Supply construction is legal macro control and may be made explicit when the strategy's growth requires lead time. Before returning an exact end-state composition, calculate worker supply plus every combat unit's count times catalog supply; the total must not exceed 200. Never claim that listed targets fill 200 supply unless the arithmetic is correct.

Knowledge and micro:
- Do not guess costs, prerequisites, production chains, abilities, counters, or timings. Use verified deterministic knowledge when available. Put unsupported facts in evidence_limits / unverified_changes rather than inventing them.
- Preserve time units from the dataset. Do not treat raw game-loop values as seconds unless a documented conversion is supplied.
- Keep strategic intent in Markdown and leave coordinates, formations, unit abilities, individual targeting, pathfinding, and other unexposed micro to the runtime.

Mandatory sentence-level executability check:
- Internally classify every imperative clause in Details as Macro, Army, or automatic runtime behavior. Keep this classification internal; write the strategy itself in ordinary StarCraft II language.
- A Macro clause is allowed only when it can be expressed as one legal race action with an absolute target, or as prerequisite/resource priority among such actions. Do not name internal action keys in strategy.md.
- An Army clause is allowed only when it can be decided from the Army View and expressed through group-level destination, allowed movement intent, visible target priority, one scan request, or one scout request.
- Automatic behavior may be stated briefly as an assumption, but never as an objective that the strategy asks an agent to perform.
- If a clause requires an unavailable observation or control primitive, Analysis must mark it strategy_fixable=false; Optimization must omit it and record the limitation in unresolved_limits. Do not disguise an unsupported outcome with more natural-sounding prose.
"""


def build_fixed_match_summary_prompt(
    *,
    strategy_name: str,
    race: str,
    record_manifest: dict,
    skill_texts: dict[str, str],
    match_timeline: str,
) -> str:
    """Build one direct summary request from the complete fixed match table."""
    return f"""You summarize one StarCraft II match for EvolAgent.

The host has already placed every Commander snapshot in chronological order in one fixed-schema table. Read every row. Do not request tools and do not select a subset of the trajectory.

Summarize only recorded facts. Keep observed state, requested macro targets, execution progress, and army/recon/wake orders distinct. Keep completed, under-construction, training, and living units distinct. The `enemy` column is what Commander observed or remembered under fog of war. `opponent_truth_after_match`, when non-empty, is separate post-match Replay truth that Commander did not know at decision time; use it to report actual enemy development and information gaps without treating it as decision-time knowledge. For a major engagement, compare the nearest pre-fight and post-fight rows, including commands, forces, losses, and territorial result. Do not diagnose root causes, compare matches, or propose strategy changes; the cross-match Analysis Agent handles that. Put anything the table cannot establish in evidence_limits.

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
  "outcome_summary": "concise chronological account of how the game developed and ended",
  "opening_and_economy": ["timed factual economy, expansion, supply, income, bank, saturation or depletion evidence"],
  "production_technology_and_composition": ["timed structures, queues, upgrades, living and training composition evidence"],
  "enemy_intelligence_and_map_state": ["scout/scan, visible or remembered enemy, post-match truth when available, information gaps, bases and zone control evidence"],
  "army_movement_and_engagements": ["group movement and major pre-fight/post-fight evidence"],
  "action_space_selection_summary": {{}},
  "commander_decision_summary": ["emitted targets/orders and validation issues"],
  "macro_execution_summary": ["absolute targets and later observed progress"],
  "army_execution_summary": ["group, scan and scout orders plus later observed state"],
  "final_state": "result and final observed economy, production, forces, bases and enemy presence",
  "evidence_limits": ["facts not established by the fixed snapshots"]
}}

Keep the lists concise while covering all important transitions. Do not repeat the table or include an action wrapper.
"""




def build_analysis_agent_prompt(
    *,
    strategy_name: str,
    race: str,
    single_game_analyses: list[BattleAnalysis],
    skill_texts: dict[str, str],
    tool_observations: list[ToolObservation],
    validation_errors: list[str],
    phase: str,
    diagnosis: dict | None,
    knowledge_mode: str,
    knowledge_runs: list[dict] | None = None,
) -> str:
    errors = "\n".join(f"- {error}" for error in validation_errors[-5:]) or "None"
    common = f"""You are EvolAgent's cross-match StarCraft II Analysis Agent.

Compare factual match summaries and identify the smallest evidence-backed strategy changes that improve winning. Separate match evidence, deterministic SC2 knowledge, and runtime limitations.

{RUNTIME_CONTRACT}

{EVIDENCE_RULES}

{AGGREGATE_ANALYSIS_RULES}

{SC2_STRATEGY_DESIGN_RULES}

{EXECUTABLE_STRATEGY_RULES}

Strategy: {strategy_name}
Race: {race}

Current strategy.md:
{render_skill_context(skill_texts)}
"""
    if phase == "diagnose":
        match_summaries = render_single_game_analyses(single_game_analyses)
        if knowledge_mode == "enabled":
            knowledge_gate = f"""
## Knowledge Questions (optional in enabled mode)

Knowledge scope: the deterministic SC2 knowledge reader answers only from the bundled entity dataset (`data_sc2_260701`) and the Commander's action catalog. It can retrieve unit/upgrade effects, executable costs and times, producers and prerequisites, and structured counter/synergy relations. It does not know this bot's agents, movement modes, attack gates, group merge behavior, fight micro, match-record events, map tactics, optimal ratios, or optimal timing. Decide those from match evidence and RUNTIME_CONTRACT; never ask them as knowledge_questions.

How to ask:
- Return one knowledge question for each independent strategy-fixable decision whose answer depends on static SC2 knowledge. Multiple questions are expected when different diagnosed problems require different entities or relations; return none only when no such decision exists.
- Merge questions only when they concern the same strategy decision and substantially the same entities. Do not collapse unrelated diagnosed problems into one broad question, and do not create questions merely to reach a count.
- One question covers one closely related strategy choice; do not bundle unrelated composition, economy, scouting, and timing questions.
- Ask only for verifiable facts that distinguish named candidates. Never ask the dataset which option is best, optimal, recommended, or most cost-effective; AnalysisAgent must make that choice from returned facts, match evidence, and strategy_contract.
- Do not ask a question whose only purpose is to retrieve costs, supply, times, producers, or prerequisites. The host attaches those action-catalog facts automatically when a decision-relevant entity question is asked.
- Provide `entities` with 1-6 exact SC2 Unit or Upgrade names. Provide `needs` with only the relevant values from `effects`, `synergy`, `counters`, and `requirements`.
- Ask for the decision-relevant relationship, not a shopping list of database fields. The host automatically adds available costs, times, producers, action dependencies, entity cards, and requested relations.
- Link the question to one or more diagnosed problem_ids. Do not answer it and do not name database tools.
- Link a problem_id only when the returned facts could directly change that problem's proposed correction. Do not attach a question to adjacent problems merely to broaden apparent coverage; scouting and economy problems normally come from match evidence unless a named entity choice genuinely depends on static facts.
- Do not re-ask facts already established by the current strategy or match evidence unless the records expose a contradiction.
- Ask which verified effects or relations distinguish the exact candidate entities named by the diagnosis.
- Do not ask for optimal timing or ratios, fight micro, or Commander/runtime behavior.
"""
        else:
            knowledge_gate = """
## Knowledge Questions (ablation)

- Knowledge mode is disabled. Still diagnose problems, but return an empty knowledge_questions list. The host will withhold Data Agent runs.
"""
        return common + f"""
Independent factual match summaries (available in full only during diagnosis):
{match_summaries}

Compare the match summaries, diagnose concrete problems, and then emit Data Agent questions when knowledge is enabled. The finish phase will rely on this diagnosis instead of receiving all match summaries again. Do not answer static SC2 encyclopedia facts from memory. Do not plan low-level query schemas.

Rules:
- Return one to {MAX_DIAGNOSED_PROBLEMS} problems supported by named match evidence.
- Mark runtime-only problems strategy_fixable=false.
- Do not invent costs, counters, production arithmetic, or capability details here.
- Preserve successful behavior as well as failures when the summaries support it.
- Record the current strategy contract, cross-outcome differences, uncertainty, and evidence limits now so Finish does not have to reconstruct them from omitted summaries.
- Keep the JSON bounded: at most three concise evidence strings per problem or win pattern, five wins_to_preserve patterns, and five cross_outcome_comparison strings.
{knowledge_gate}
Host feedback:
{errors}

Return one JSON action only:
{{
  "action": "diagnose",
  "diagnosis": {{
    "strategy_contract": {{
      "identity": "one concise description of economy shape, core force, power stage, posture and win condition",
      "core_commitments": ["two to five rules that define this strategy rather than generic runtime behavior"],
      "optimization_boundary": "what may change without becoming a different strategy, and what would cross that boundary",
      "direction": "preserve|adjust|replace"
    }},
    "problems": [
      {{
        "problem_id": "P1",
        "problem": "observed SC2 problem",
        "evidence": ["match number and event"],
        "consequence": "later match effect",
        "strategy_fixable": true
      }}
    ],
    "wins_to_preserve": [
      {{
        "win_id": "W1",
        "pattern": "successful evidence-backed behavior",
        "evidence": ["Match number and event"],
        "why": "observed benefit"
      }}
    ],
    "cross_outcome_comparison": ["evidence-backed difference between wins and losses"],
    "knowledge_questions": [
      {{
        "id": "Q1",
        "problem_ids": ["P1"],
        "question": "decision-focused static-knowledge question about the proposed strategy choice",
        "entities": ["exact SC2 entity name from the proposed choice"],
        "needs": ["effects"]
      }}
    ],
    "evidence_limits": ["uncertainty that must survive into final analysis"]
  }}
}}
"""

    return common + f"""
The structured cross-match diagnosis below was produced from all supplied match summaries:
{diagnosis}

Knowledge mode: {knowledge_mode}

Compact verified knowledge results:
{render_knowledge_runs(knowledge_runs or [])}

Interpretation:
- In enabled mode, a knowledge result is verified only when it carries successful deterministic dataset evidence. Treat the compact answer as authoritative for the diagnosed problems linked by problem_ids / question ids.
- In disabled mode, knowledge was withheld for ablation. Keep knowledge_used empty and do not present model memory as verified data.
- A failed or withheld sub-agent run is an evidence limit, not verified knowledge.
- Use only facts stated in the compact verified answer. Do not turn static facts into asserted fight outcomes, optimal ratios, or win probabilities.
- A verified result proves only the facts it actually states. If the requested comparison or quantitative effect is absent, keep it unresolved; `ok` does not authorize filling the gap from model memory.
- Costs, times, and prerequisites may compare feasibility, but they do not by themselves prove that a candidate is best or justify new army counts, worker targets, attack gates, or expansion timing.
- Carry strategy_contract, wins_to_preserve, cross_outcome_comparison, problem evidence, and uncertainty forward from the diagnosis. Never discard a preserved win pattern merely because the corresponding raw match summary is not repeated here.
- Choose finish_analysis now. There is no query_more step; the host already resolved the diagnosed knowledge_questions.

Host feedback:
{errors}

Return one JSON action only:
{{
  "action": "finish_analysis",
  "analysis": {{
    "strategy_contract": {{
      "identity": "carry the diagnosed strategy identity forward unchanged",
      "core_commitments": ["carry diagnosed commitments forward unchanged"],
      "optimization_boundary": "carry the diagnosed boundary forward unchanged",
      "direction": "preserve|adjust|replace"
    }},
    "repeated_failures": [
      {{"problem_id":"P1","cause":"observed cause","consequence":"later effect","seen_in":["match and event"],"strategy_fixable":true,"confidence":"low|medium|high"}}
    ],
    "wins_to_preserve": [
      {{"pattern":"successful behavior","seen_in":["match and event"],"why":"observed benefit"}}
    ],
    "cross_outcome_comparison": ["evidence-backed difference"],
    "optimization_targets": [
      {{
        "problem_id":"P1",
        "problem":"strategy-fixable problem",
        "match_evidence":["match and event"],
        "strategy_change":"reusable combat-winning correction with Macro absolute targets and/or Army gates; use an own-side timing gate when production lead time matters; may add units and/or research; avoid static base defense and sighting-only tech starts",
        "knowledge_used":["fact from knowledge result Qid, or empty in disabled mode"],
        "decision_effect":"how knowledge changed or confirmed the choice, or that it was withheld",
        "confidence":"low|medium|high"
      }}
    ],
    "combat_win_requirements": ["win favorable field fights, reinforce through trades, regain initiative, destroy bases"],
    "knowledge_used": [
      {{"problem_ids":["P1"],"question_ids":["Q1"],"finding":"verified sub-agent finding","decision_effect":"effect on strategy selection"}}
    ],
    "evidence_limits": ["unresolved evidence or failed knowledge query"]
  }}
}}

Return at most five optimization targets, all linked to diagnosed problem_id values.
"""


def build_optimization_agent_prompt(
    *,
    strategy_name: str,
    race: str,
    battle_analysis: BattleAnalysis,
    skill_texts: dict[str, str],
    tool_observations: list[ToolObservation],
    validation_errors: list[str],
    candidate: dict | None,
    knowledge_mode: str,
    remaining_verify_calls: int | None = None,
) -> str:
    errors = "\n".join(f"- {error}" for error in validation_errors[-5:]) or "None"
    candidate_text = str(candidate) if candidate else "None"
    return f"""You are EvolAgent's Strategy Optimization Agent.

Write a complete replacement for strategy.md from the finalized cross-match analysis. Do not re-diagnose matches or modify runtime code.

{RUNTIME_CONTRACT}

{SC2_STRATEGY_DESIGN_RULES}

{EXECUTABLE_STRATEGY_RULES}

{STRATEGY_MARKDOWN_FORMAT}

Required strategy.md format:
- Match the house format above exactly. Mirror the structure and bullet style of the Current strategy.md provided below; change content where evidence requires it, but do not change the document genre.
- Output exactly # Summary and # Details, in that order and non-empty. Do not add # Resource Costs, # Required Tools, or any other heading.
- Summary briefly defines the economy shape, core composition, and win plan as a short paragraph (no bullets).
- Details contains the complete executable strategy as "* Title: ..." bullets for macro progression and army decisions. Use compact titled bullets with observable triggers, absolute targets where needed, attack readiness based on the gathered persistent main force, group_1 reinforcement toward that force or its objective, retreat and recovery, need-based scouting or scanning, and cleanup intent. Do not turn Details into a copy of the Commander system prompt. Do not switch to hyphen bullets, numbered lists, or untitled freeform paragraphs.
- Do not repeat costs, supply, base times, producers, research locations, prerequisites, or dependency closure in strategy.md. The selected runtime action catalog supplies those mechanics directly to the Commander.

Rules:
- Treat the finalized strategy_contract as the optimization boundary. Do not silently reinterpret its identity, commitments, or direction.
- For direction=preserve, retain the economy shape, core force, power stage, posture, and win condition; change only evidence-backed execution details. For direction=adjust, retain the identity and core commitments while allowing evidence-backed support units, upgrades, economy amounts, scouting, priorities, and readiness gates inside optimization_boundary. For direction=replace, change the identity only to address the analysis evidence that justified replacement.
- If a proposed optimization target crosses optimization_boundary under preserve or adjust, omit that change instead of drifting into another strategy. The revised # Summary must still describe the resulting identity accurately.
- Preserve only strategy behavior supported by successful evidence; change the plan when records show it cannot reliably win.
- Implement the prioritized strategy-fixable targets rather than merely restating their symptoms.
- Optimize for field combat. Prefer composition changes (including escorts/healers and multi-unit mixes), researches that unlock or strengthen core-unit abilities, and clearer gather/reinforce/attack gates. Avoid making static base defense the main change unless evidence is base-only.
- Composition edits may add more than one unit type when evidence supports a mixed package. Research may be chosen instead of or together with new units when it fixes the observed trade. Every train/build/research addition must be a fixed absolute target with own-side gates—never "if enemy X is detected, produce Y."
- Map every new rule to control the single Commander actually has: macro absolute targets and resource priority; army destination, movement mode, gathered-force gates, scan/scout; or an observable wake checkpoint. Drop clauses that need unexposed micro or abilities.
- Before finishing, self-check consistency: every unit, structure, upgrade, or capability named in Summary is produced or enabled by Details; enemy threats are only scouting or engagement evidence, not branches that change the fixed production plan.
- Before finishing, self-check executability against the runtime contract: every macro rule maps to legal absolute-target tools and ordering; every army rule uses semantic destinations, allowed movement modes, and gathered-main-force readiness gates; group_1 only reinforces the persistent main force or its current objective; no rule depends on coordinates, unit tags, transport or garrison loading, transformation or mode toggles, targeted abilities, or other unexposed micro.
- For every Details bullet, perform the mandatory Macro/Army/automatic executability check above. If any imperative clause has no supported mapping, remove that clause instead of approximating it with prose.
- Do not encode unsupported outcomes such as type-specific formations, exact percentages in adjacent zones, multi-zone waypoint routes, selecting a particular production-building instance, or strict "finish target A before target B starts" barriers. Use supported group cohesion/fragmentation, one destination, prerequisites, absolute targets, and resource priority instead.
- Write ordinary StarCraft II strategy language. Do not mention agent internals, observation fields, or runtime APIs.
- Do not rely only on reacting after an enemy unit is observed when the response would arrive too late. For repeated or high-risk enemy tech paths, include a bounded preemptive preparation such as scouting timing, prerequisite tech, fixed production path, upgrade priority, or a small multi-unit buffer that joins the army.
- Production lead time matters: if a response needs a new building line or multiple units, do not make first contact with the threat the start trigger. Prefer own-side timing gates before the first planned fight.
- Any added escort unit, research, or counter unit must state fixed absolute counts and own-side start/stop gates. Do not switch Macro production based on enemy sightings.
- Do not add a clock-only hard attack deadline that forces a push before the stated core army and required escorts or researches can exist.
- In enabled mode, use only the supplied compact verified knowledge answers. Do not call verify_candidate or invent new encyclopedia lookups.
- In disabled mode, do not claim external verification.
- Do not add unsupported thresholds, fight predictions, formations, coordinates, individual micro or unexposed abilities.
- A failed or withheld knowledge result cannot support a strategy change; keep the unresolved claim in unverified_changes.
- Add a new numeric timing, count threshold, or interval only when the current strategy, repeated match evidence, or verified knowledge supports that exact value.
- Keep the strategy map-agnostic: use semantic locations such as own main, natural, safe forward zone, enemy natural, or enemy main; never write literal zone_id values.
- Keep scans and scouts need-based unless repeated match evidence specifically supports a timing requirement.
- Prefer scout/scan rules that match the strategy style for priority and cadence, without locking destinations to only natural/third. Empty or forward mineral expansions may be sequenced when the plan needs to detect a hidden third, choose the next expand, or clear fog before moving. When information age is stale relative to the next planned decision, request the next single worker scout or scan instead of forbidding follow-up scouts. Do not require multiple simultaneous worker scouts.
- Prefer the smallest coherent set of changes that addresses the strongest strategy-fixable evidence.
- Check the combined effect of all changes, not each change in isolation. Added economy, technology, support units, vision requirements, and readiness gates must not collectively move the power stage outside strategy_contract or worsen another diagnosed problem such as an already-late attack.
- When records show unused gas alongside mineral starvation, do not add more gas collection merely because new gas units were added; justify any higher gas target from the combined production demand and preserve enough workers for the mineral-heavy core.
- Keep runtime-only failures in unresolved_limits.
- Reject enemy-detection Macro forks: do not write "if enemy air/cloak is seen, train X; otherwise train Y." Lock one fixed absolute production plan from own-side gates before the first planned fight.
- For every change, provide structured supported_by provenance. Match references must copy an exact match_evidence entry from the finalized optimization_targets only. Do not paraphrase those strings and do not cite repeated_failures.seen_in or other nearby narrative fields. Knowledge references may cite a verified knowledge result by its one-based query_index. An unchanged current-strategy rule may be cited by its exact text.
- A supported_by or new_numeric_claims item with source_type=knowledge must include a separate integer field "query_index": 7 (or the applicable complete result number). Do not put the index only inside "reference".
- Prefer listing newly introduced # Details numbers under new_numeric_claims with a match or complete-knowledge source when available. Static mechanics already supplied by the action catalog do not belong in strategy.md and do not count. Incomplete numeric claims are acceptable; do not invent provenance.
- For every change, classify its required behavior abstractly as macro, army, or automatic in runtime_requirements. Mark supported=true only when the runtime contract supplies the needed observation and control; otherwise omit the behavior from strategy.md and put it in unresolved_limits.
- If validator feedback reports a format error (missing sections, wrong bullet style, literal zone_id values, coordinates, or other unexecutable micro/ability orders), revise the full strategy.md until the house format is valid and executable.

Knowledge mode: {knowledge_mode}
Strategy: {strategy_name}
Race: {race}

Current strategy.md:
{render_skill_context(skill_texts)}

Final cross-match analysis:
{render_battle_analysis(battle_analysis)}

Allowed match_evidence strings (copy these verbatim into supported_by / new_numeric_claims when source_type=match):
{render_allowed_match_evidence(battle_analysis)}

Knowledge observations (compact verified answers):
{render_sc2_knowledge(tool_observations)}

Current candidate:
{candidate_text}

Validator feedback:
{errors}

Return one JSON action only. Prefer drafting when existing knowledge is enough:
{{
  "action": "draft_improvement",
  "analysis": {{
    "main_lesson": "highest-value evidence-backed lesson",
    "combat_win_plan": "how the revised army, production priority, and attack timing defeat the observed threats",
    "changes_made": [
      {{
        "problem_id":"P1",
        "change":"specific strategy change",
        "match_evidence":["analysis evidence"],
        "knowledge_used":["complete knowledge fact or empty"],
        "supported_by":[
          {{"source_type":"match","reference":"specific finalized match evidence"}}
        ],
        "new_numeric_claims":[
          {{"value":"exact new value","source_type":"match","reference":"exact finalized match_evidence entry"}}
        ],
        "runtime_requirements":[
          {{"requirement":"behavior required by this change","support_kind":"macro|army|automatic","supported":true,"reference":"relevant runtime-contract capability"}}
        ],
        "expected_effect":"SC2 effect"
      }}
    ],
    "knowledge_influenced_changes": ["change whose choice was affected by verified knowledge"],
    "unverified_changes": ["exact claim that remains unverified"],
    "unresolved_limits": ["runtime or evidence limit not encoded as strategy"]
  }},
  "files": {{
    "strategy.md": "# Summary\\n...\\n\\n# Details\\n* Opening: ...\\n* Main Attack Gate: ..."
  }}
}}

After validator feedback use action="revise_candidate" with the same complete schema.
"""
