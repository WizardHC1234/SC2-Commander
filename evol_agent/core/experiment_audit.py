from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Any

from .capabilities import build_executor_capability_manifest
from .config import (
    DEFAULT_ANALYSIS_MODEL,
    EXPERIMENT_AUDIT_ENABLE_REASONING,
    MAX_CONCURRENT_MATCH_SUBAGENTS,
)
from .llm import call_json_llm
from .match_summary import run_fixed_match_summary
from .match_summary_cache import MatchSummaryCache
from .types import GameEvidence
from ..analysis.record_reader import find_record_jsons, is_completed_match_record


_IMPLEMENTATION_VERDICTS = {
    "implemented",
    "underpowered",
    "execution_invalid",
    "unknown",
}
_HYPOTHESIS_VERDICTS = {
    "supported",
    "contradicted",
    "inconclusive",
    "not_tested",
}

_OFFENSIVE_MOVEMENT_MODES = frozenset(
    {"assault", "contain", "push", "search_and_destroy"}
)
_ACTION_ENTITY_ALIASES = {
    "buildgas": "refinery",
    "expand": "commandcenter",
    "trainviking": "vikingfighter",
    "trainhellbat": "helliontank",
    "researchyamatocannon": "yamatocannon",
}


_PARENT_ANALYSIS_FIELDS = (
    "strategy_name",
    "race",
    "sample_size",
    "record_mix",
    "strategy_contract",
    "strengths_to_preserve",
    "outcome_contrast",
    "priority_problem",
    "hypothesis",
    "failure_mode_analysis",
    "priority_alignment",
    "mechanism_prediction",
    "retrieval_assessment",
    "winning_mechanism",
    "repeated_failures",
    "evidence_limits",
)


def _compact_parent_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """Keep the prior cross-match conclusion without replaying parent matches."""
    return {
        field: analysis[field]
        for field in _PARENT_ANALYSIS_FIELDS
        if field in analysis and analysis[field] not in (None, "", [], {})
    }


def _evidence_from_batches(
    batch_dirs: list[Path],
    *,
    strategy: str,
) -> list[GameEvidence]:
    records: list[GameEvidence] = []
    for batch_dir in batch_dirs:
        if not batch_dir.is_dir():
            continue
        for path in find_record_jsons(batch_dir):
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            if not is_completed_match_record(data):
                continue
            meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            recorded_strategy = str(meta.get("strategy_id") or "")
            if recorded_strategy and recorded_strategy != strategy:
                continue
            records.append(
                GameEvidence(
                    file=str(path),
                    result=str(meta.get("result") or "?"),
                    duration=str(meta.get("game_duration_formatted") or "?"),
                    timeline="",
                    meta=meta,
                )
            )
    return records


def _normalized_token(value: Any) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _action_entity(action: Any) -> str:
    normalized = _normalized_token(action)
    if normalized in _ACTION_ENTITY_ALIASES:
        return _ACTION_ENTITY_ALIASES[normalized]
    for prefix in ("train", "build", "research", "morph"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def _as_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    return [item for item in (value or []) if isinstance(item, dict)]


def _completed_count(snapshot: dict[str, Any], action: str) -> int:
    observation = (
        snapshot.get("observation_full")
        if isinstance(snapshot.get("observation_full"), dict)
        else snapshot.get("observation")
    )
    if not isinstance(observation, dict):
        return 0
    entity = _action_entity(action)
    if _normalized_token(action) == "expand":
        economy = observation.get("economy") or {}
        return int(economy.get("own_base_count") or 0)
    counts: dict[str, Any] = {}
    own_forces = observation.get("own_forces") or {}
    production = observation.get("production") or {}
    for source in (
        own_forces.get("completed_counts"),
        production.get("completed"),
    ):
        if isinstance(source, dict):
            counts.update(source)
    normalized_counts = {
        _normalized_token(name): int(value or 0) for name, value in counts.items()
    }
    if entity in normalized_counts:
        return normalized_counts[entity]
    if entity == "commandcenter":
        return sum(
            normalized_counts.get(name, 0)
            for name in ("commandcenter", "orbitalcommand", "planetaryfortress")
        )
    return 0


def _upgrade_completed(snapshot: dict[str, Any], action: str) -> bool:
    observation = (
        snapshot.get("observation_full")
        if isinstance(snapshot.get("observation_full"), dict)
        else snapshot.get("observation")
    )
    technology = observation.get("technology") if isinstance(observation, dict) else {}
    upgrades = (
        technology.get("completed_upgrades")
        if isinstance(technology, dict)
        else []
    )
    wanted = _action_entity(action)
    for upgrade in upgrades or []:
        actual = _normalized_token(upgrade)
        if wanted and (wanted == actual or wanted in actual or actual in wanted):
            return True
    return False


def _package_is_reached(snapshot: dict[str, Any], package: dict[str, Any]) -> bool:
    components = [
        item
        for item in (package.get("gate_components") or [])
        if isinstance(item, dict) and str(item.get("action") or "").strip()
    ]
    if not components:
        return False
    for component in components:
        action = str(component.get("action") or "").strip()
        quantity = max(1, int(component.get("quantity") or 1))
        if _normalized_token(action).startswith("research"):
            if not _upgrade_completed(snapshot, action):
                return False
        elif _completed_count(snapshot, action) < quantity:
            return False
    observation = snapshot.get("observation_full") or snapshot.get("observation") or {}
    economy = observation.get("economy") if isinstance(observation, dict) else {}
    economy_contract = package.get("economy") or {}
    if isinstance(economy, dict) and isinstance(economy_contract, dict):
        worker_target = economy_contract.get("worker_target_before_commitment")
        base_target = economy_contract.get("base_target_before_commitment")
        if worker_target not in (None, "") and int(economy.get("workers") or 0) < int(worker_target):
            return False
        if base_target not in (None, "") and int(economy.get("own_base_count") or 0) < int(base_target):
            return False
    return True


def _offensive_command_time(payload: dict[str, Any]) -> tuple[float | None, str]:
    applied_times: list[float] = []
    for snapshot in payload.get("records") or []:
        if not isinstance(snapshot, dict):
            continue
        observation = snapshot.get("observation_full") or snapshot.get("observation") or {}
        army = observation.get("army_control") if isinstance(observation, dict) else {}
        for command in _as_rows(army.get("current_commands") if isinstance(army, dict) else []):
            if str(command.get("movement_mode") or "").strip().lower() not in _OFFENSIVE_MOVEMENT_MODES:
                continue
            try:
                snapshot_time = float(snapshot.get("game_time_seconds") or 0.0)
                command_age = float(command.get("command_age_seconds") or 0.0)
            except (TypeError, ValueError):
                continue
            applied_times.append(max(0.0, snapshot_time - command_age))
    if applied_times:
        return min(applied_times), "applied_runtime_command"
    accepted_times: list[float] = []
    for interaction in payload.get("interactions") or []:
        if not isinstance(interaction, dict) or interaction.get("accepted") is not True:
            continue
        army_policy = interaction.get("army_policy") or {}
        for command in _as_rows(army_policy.get("commands") if isinstance(army_policy, dict) else []):
            if str(command.get("movement_mode") or "").strip().lower() in _OFFENSIVE_MOVEMENT_MODES:
                try:
                    accepted_times.append(float(interaction.get("game_time") or 0.0))
                except (TypeError, ValueError):
                    pass
    return (min(accepted_times), "accepted_commander_command") if accepted_times else (None, "none")


def _accepted_decision_times(payload: dict[str, Any]) -> list[float]:
    result: list[float] = []
    for interaction in payload.get("interactions") or []:
        if not isinstance(interaction, dict) or interaction.get("accepted") is not True:
            continue
        if str(interaction.get("agent") or "commander") != "commander":
            continue
        try:
            result.append(float(interaction.get("game_time") or 0.0))
        except (TypeError, ValueError):
            continue
    return sorted(set(result))


def _audit_gate_execution(
    records: list[GameEvidence],
    *,
    experiment_spec: dict[str, Any],
) -> dict[str, Any]:
    timing = experiment_spec.get("first_commitment_timing")
    declared = timing.get("declared_packages") if isinstance(timing, dict) else {}
    package = declared.get("candidate") if isinstance(declared, dict) else {}
    expected = (
        timing.get("candidate_earliest_feasible_time_seconds")
        if isinstance(timing, dict)
        else None
    )
    if not isinstance(package, dict) or not package.get("gate_components"):
        return {
            "status": "unknown",
            "expected_earliest_feasible_time_seconds": expected,
            "matches": [],
            "execution_issue_matches": 0,
            "evidence_limit": "candidate first-commitment package was not recorded",
        }
    matches: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        try:
            payload = json.loads(Path(record.file).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            payload = {}
        snapshots = sorted(
            [item for item in payload.get("records") or [] if isinstance(item, dict)],
            key=lambda item: float(item.get("game_time_seconds") or 0.0),
        )
        gate_time: float | None = None
        for snapshot in snapshots:
            if _package_is_reached(snapshot, package):
                gate_time = float(snapshot.get("game_time_seconds") or 0.0)
                break
        commitment_time, commitment_source = _offensive_command_time(payload)
        decision_times = _accepted_decision_times(payload)
        missed = 0
        verdict = "gate_never_reached"
        if gate_time is not None:
            if commitment_time is not None and commitment_time < gate_time - 1.0:
                verdict = "committed_before_declared_gate"
            else:
                missed = sum(
                    1
                    for value in decision_times
                    if value >= gate_time - 1.0
                    and (commitment_time is None or value < commitment_time - 1.0)
                )
                if commitment_time is None:
                    verdict = (
                        "gate_met_no_commitment"
                        if missed >= 2
                        else "insufficient_follow_up_after_gate"
                    )
                elif missed >= 2:
                    verdict = "gate_met_delayed_commitment"
                else:
                    verdict = "gate_met_timely_commitment"
        matches.append(
            {
                "game": index,
                "result": record.result,
                "gate_reached_time_seconds": round(gate_time, 3) if gate_time is not None else None,
                "first_commitment_time_seconds": round(commitment_time, 3) if commitment_time is not None else None,
                "commitment_source": commitment_source,
                "missed_effective_decision_opportunities": missed,
                "actual_minus_expected_gate_seconds": (
                    round(gate_time - float(expected), 3)
                    if gate_time is not None and expected not in (None, "")
                    else None
                ),
                "verdict": verdict,
            }
        )
    issue_verdicts = {"gate_met_no_commitment", "gate_met_delayed_commitment"}
    issue_matches = [item for item in matches if item["verdict"] in issue_verdicts]
    gate_times = [
        float(item["gate_reached_time_seconds"])
        for item in matches
        if item["gate_reached_time_seconds"] is not None
    ]
    commitment_times = [
        float(item["first_commitment_time_seconds"])
        for item in matches
        if item["first_commitment_time_seconds"] is not None
    ]
    repeated_issue = len(issue_matches) >= 2
    return {
        "status": "execution_issue" if repeated_issue else "measured",
        "expected_earliest_feasible_time_seconds": expected,
        "median_actual_gate_reached_time_seconds": round(median(gate_times), 3) if gate_times else None,
        "median_actual_first_commitment_time_seconds": round(median(commitment_times), 3) if commitment_times else None,
        "execution_issue_matches": len(issue_matches),
        "gate_reached_matches": len(gate_times),
        "matches": matches,
        "classification_rule": (
            "execution_issue when the declared gate is observed and two or more "
            "effective Commander decisions pass without an applied offensive command; "
            "two repeated matches make the aggregate finding deterministic"
        ),
    }


def _summarize_records(
    records: list[GameEvidence],
    *,
    strategy_name: str,
    race: str,
    model: str,
    label: str,
    cache: MatchSummaryCache,
    audit_focus: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not records:
        return []
    summaries: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, GameEvidence]] = []
    for index, record in enumerate(records, 1):
        # Focused audit summaries are experiment-specific and must never reuse a
        # generic summary that may have omitted the pre-registered mechanism.
        cached = None if audit_focus else cache.get(
            record,
            strategy_name=strategy_name,
            race=race,
            model=model,
        )
        if cached is None:
            pending.append((index, record))
            continue
        summaries[index] = {
            "game": index,
            "result": record.result,
            "summary": cached["summary"],
            "errors": cached["errors"],
        }
    print(
        f"    [{label}: {strategy_name}] reused {len(summaries)} cached summaries; "
        f"summarizing {len(pending)} new matches",
        flush=True,
    )
    if not pending:
        return [summaries[index] for index in sorted(summaries)]
    worker_count = min(MAX_CONCURRENT_MATCH_SUBAGENTS, len(pending))
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix=f"evol-audit-{label}",
    ) as executor:
        futures = {
            executor.submit(
                run_fixed_match_summary,
                strategy_name=strategy_name,
                race=race,
                record=record,
                game_index=index,
                model=model,
                prefix=f"    [{label}: {strategy_name}] ",
                audit_focus=audit_focus,
            ): index
            for index, record in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                digest, analysis, ok, errors, _events = future.result()
                summaries[index] = {
                    "game": index,
                    "result": records[index - 1].result,
                    "summary": analysis.raw,
                    "errors": errors,
                }
                if ok and not audit_focus:
                    cache.put(
                        records[index - 1],
                        strategy_name=strategy_name,
                        race=race,
                        model=model,
                        summary=analysis.raw,
                        errors=errors,
                        source="experiment_audit",
                        digest=(
                            digest.raw
                            if digest is not None and isinstance(digest.raw, dict)
                            else {}
                        ),
                    )
            except Exception as exc:  # preserve the evolution run on audit failure
                summaries[index] = {
                    "game": index,
                    "result": records[index - 1].result,
                    "summary": {},
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
    return [summaries[index] for index in sorted(summaries)]


def build_experiment_audit_prompt(
    *,
    parent_strategy: str,
    candidate_strategy: str,
    experiment_spec: dict[str, Any],
    parent_analysis: dict[str, Any],
    candidate_summaries: list[dict[str, Any]],
    outcome_comparison: dict[str, Any],
    capability_manifest: dict[str, Any],
    gate_execution_audit: dict[str, Any] | None = None,
    parent_new_summaries: list[dict[str, Any]] | None = None,
) -> str:
    return f"""You are EvolAgent's compact post-experiment reviewer. Determine whether the candidate's main intended behavior actually appeared in match trajectories and record what happened at contact and afterward. Do not propose the next strategy and do not require every written strategy detail to appear in every match.

Evaluation rules:
- A requested action is not execution; use later game state or applied commands.
- Mark implemented when the main material change is observably realized in representative candidate matches. Do not require formal audit fields or every coordinated detail.
- Mark execution_invalid when the strategy rule was available but runtime repeatedly failed to apply it, especially when the deterministic gate audit reports an execution issue.
- Compare actual first commitment/contact time, own and enemy packages at contact, decisive engagement result, retained force, reinforcement, continued offense, and final match outcome.
- A score gain supports promotion only when the main change was implemented. A score loss remains useful evidence even when implementation was partial.
- Preserve useful local observations, but do not treat a written-only change as successful.

Experiment intent:
{json.dumps(experiment_spec, ensure_ascii=False, separators=(',', ':'))}

Outcome comparison:
{json.dumps(outcome_comparison, ensure_ascii=False, separators=(',', ':'))}

Deterministic gate and commitment audit:
{json.dumps(gate_execution_audit or {}, ensure_ascii=False, separators=(',', ':'))}

Parent evidence summary:
{json.dumps(parent_analysis, ensure_ascii=False, separators=(',', ':'))}

New parent summaries:
{json.dumps(parent_new_summaries or [], ensure_ascii=False, separators=(',', ':'))}

Candidate match summaries:
{json.dumps(candidate_summaries, ensure_ascii=False, separators=(',', ':'))}

Parent strategy.md:
{parent_strategy}

Candidate strategy.md:
{candidate_strategy}

Return JSON only:
{{
  "implementation_verdict":"implemented|underpowered|execution_invalid|unknown",
  "hypothesis_verdict":"supported|contradicted|inconclusive|not_tested",
  "mechanism_evidence":[{{"side":"candidate","game":"Game 1","evidence":"main change observed in trajectory"}}],
  "combat_evidence":[{{"side":"candidate","game":"Game 1","evidence":"contact timing, packages, engagement, and continuation"}}],
  "runtime_findings":[],
  "salvageable_changes":[],
  "failed_dependencies":[],
  "evidence_limits":[],
  "lesson":"one concise change-outcome lesson"
}}
"""
    return f"""You are EvolAgent's post-experiment mechanism auditor.

The final optimization objective is winning decisive army engagements and then the match. Attack timing, production synchronization, resource banking, scouting, and gate attainment are intermediate mechanisms only.

Audit the pre-registered hypothesis using the parent analysis already produced before candidate generation, any Parent confirmation-match summaries added after that analysis, and the candidate match summaries collected after the change. Do not re-analyze Parent matches already covered by the prior analysis or propose a new strategy in this step.

The experiment is one primary failure mode addressed by a coordinated intervention package. Audit the package as a whole, not merely the easiest individual lever.

Required reasoning discipline:
- First decide whether minimum_material_change was actually realized in candidate matches, not merely written into strategy.md or requested by Commander.
- Check whether the candidate collectively realized the declared intervention_package.coordinated_changes and material_behavior_change. A single upgrade, timing shift, or production change is not implementation when the rest of the required package did not occur.
- Use intervention_package.strategy_area_audit to verify the whole strategy: every area marked revise must produce its intended trajectory-level behavior, including pre-commitment, post-commitment or midgame, and late-game unit and production-facility targets plus post-contact continuity, before the package can be classified as implemented.
- Compare opponent pressure windows and whether the candidate retained enough defensive combat power to reach its intended technology, upgrade, composition, or power spike. Do not evaluate a later power spike while ignoring repeated deaths before it becomes active.
- Compare own/enemy composition and support balance at major engagements. Keep unit-counter or exact numerical claims qualitative unless grounded by the pre-registered verified evidence.
- Trace strategy rule -> Commander decision -> applied command -> later game state.
- A requested action is not proof of execution. Sampled absence is not proof that an event never happened.
- If the hypothesis depends on transformations, abilities, transport loading, targeting, formation, or unit-level micro owned by runtime, mark execution_invalid unless the strategy can affect the mechanism through an exposed high-level control.
- Judge combat success primarily from the decisive engagement result, first-fight survival/force retention, ability to continue the push, and eventual match result.
- A higher score alone does not prove the mechanism. A lower score alone does not contradict it unless the minimum mechanism change occurred.
- Use the deterministic gate-execution audit as authoritative for declared gate attainment and applied offensive-command timing. If it reports execution_issue, classify the failure as runtime execution; do not recommend another change to the strategy's attack gate from those matches.
- Use implemented only when multiple candidate matches show the declared material change. Use underpowered when the strategy changed but the realized mechanism was too weak or inconsistent. Use unknown when the records cannot establish it.
- supported and contradicted are valid only when implementation_verdict=implemented.
- Candidate summaries may contain mechanism_probe. Treat an observed probe as implementation evidence only for that match. not_observed and unknown never prove contradiction. Require multiple observed candidate probes before using implementation_verdict=implemented.
- A rejected or execution-invalid candidate may still contain useful local changes. Separate the failed dependency from any candidate change that was actually realized and repeatedly associated with a better intermediate or combat outcome. Put only evidence-backed local changes in salvageable_changes. Do not salvage a merely written or requested change, and do not recommend inheriting the whole candidate package.

Executor capability manifest:
{json.dumps(capability_manifest, ensure_ascii=False, indent=2)}

Pre-registered experiment:
{json.dumps(experiment_spec, ensure_ascii=False, indent=2)}

Parent strategy.md:
{parent_strategy}

Candidate strategy.md:
{candidate_strategy}

Outcome comparison:
{json.dumps(outcome_comparison, ensure_ascii=False, indent=2)}

Deterministic expected-versus-actual gate and commitment timing:
{json.dumps(gate_execution_audit or {}, ensure_ascii=False, indent=2)}

Parent prior cross-match analysis:
{json.dumps(parent_analysis, ensure_ascii=False, indent=2)}

Parent factual summaries added after the prior analysis:
{json.dumps(parent_new_summaries or [], ensure_ascii=False, indent=2)}

Candidate factual match summaries:
{json.dumps(candidate_summaries, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "implementation_verdict":"implemented|underpowered|execution_invalid|unknown",
  "hypothesis_verdict":"supported|contradicted|inconclusive|not_tested",
  "mechanism_evidence":[
    {{"side":"parent|candidate","game":"Game 1","evidence":"observable event"}}
  ],
  "combat_evidence":[
    {{"side":"parent|candidate","game":"Game 1","evidence":"decisive engagement or force-retention result"}}
  ],
  "runtime_findings":["execution limitation, if any"],
  "salvageable_changes":[
    {{"change":"local candidate change worth reusing","evidence":["Game 2 @ 320s: observed positive effect"],"condition":"when this remains compatible with the Champion contract"}}
  ],
  "failed_dependencies":["specific unavailable or harmful dependency that must not be inherited"],
  "evidence_limits":["important uncertainty"],
  "lesson":"one concise causal lesson for the next generation"
}}
"""


def _observed_mechanism_probe_count(
    candidate_summaries: list[dict[str, Any]] | None,
) -> int:
    count = 0
    for item in candidate_summaries or []:
        summary = item.get("summary") if isinstance(item, dict) else None
        probe = summary.get("mechanism_probe") if isinstance(summary, dict) else None
        if not isinstance(probe, dict) or probe.get("status") != "observed":
            continue
        observations = [
            row
            for row in (probe.get("observations") or [])
            if isinstance(row, dict)
            and str(row.get("fact") or "").strip()
            and row.get("time_s") not in (None, "")
        ]
        if observations:
            count += 1
    return count


def _normalize_audit(
    raw: Any,
    *,
    candidate_summaries: list[dict[str, Any]] | None = None,
    require_observed_probes: bool = False,
    gate_execution_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = raw.get("audit") if isinstance(raw, dict) and isinstance(raw.get("audit"), dict) else raw
    if not isinstance(payload, dict):
        payload = {}
    implementation = str(payload.get("implementation_verdict") or "unknown").strip()
    if implementation not in _IMPLEMENTATION_VERDICTS:
        implementation = "unknown"
    hypothesis = str(payload.get("hypothesis_verdict") or "inconclusive").strip()
    if hypothesis not in _HYPOTHESIS_VERDICTS:
        hypothesis = "inconclusive"

    def rows(name: str) -> list[Any]:
        value = payload.get(name)
        return list(value) if isinstance(value, list) else []

    observed_probe_count = _observed_mechanism_probe_count(candidate_summaries)
    evidence_limits = [str(item) for item in rows("evidence_limits") if str(item)]
    if (
        require_observed_probes
        and implementation == "implemented"
        and observed_probe_count < 2
    ):
        implementation = "underpowered" if observed_probe_count == 1 else "unknown"
        evidence_limits.append(
            "implementation requires observed mechanism probes in at least two "
            f"candidate matches; found {observed_probe_count}"
        )
    if implementation != "implemented" and hypothesis in {"supported", "contradicted"}:
        hypothesis = "not_tested"
    deterministic_gate = dict(gate_execution_audit or {})
    runtime_findings = [str(item) for item in rows("runtime_findings") if str(item)]
    failed_dependencies = [
        str(item) for item in rows("failed_dependencies") if str(item)
    ][:6]
    if deterministic_gate.get("status") == "execution_issue":
        implementation = "execution_invalid"
        hypothesis = "not_tested"
        finding = (
            "declared first-attack gate was reached, but offensive commitment was "
            "missing or delayed across repeated effective Commander decisions"
        )
        if finding not in runtime_findings:
            runtime_findings.append(finding)
        dependency = (
            "runtime launch after gate attainment; do not further modify the strategy "
            "attack gate until Commander execution is repaired"
        )
        if dependency not in failed_dependencies:
            failed_dependencies.append(dependency)

    return {
        "implementation_verdict": implementation,
        "hypothesis_verdict": hypothesis,
        "mechanism_evidence": rows("mechanism_evidence"),
        "combat_evidence": rows("combat_evidence"),
        "runtime_findings": runtime_findings,
        "salvageable_changes": [
            {
                "change": str(item.get("change") or "").strip(),
                "evidence": [
                    str(evidence).strip()
                    for evidence in (item.get("evidence") or [])
                    if str(evidence).strip()
                ][:6],
                "condition": str(item.get("condition") or "").strip(),
            }
            for item in rows("salvageable_changes")
            if isinstance(item, dict)
            and str(item.get("change") or "").strip()
            and item.get("evidence")
        ][:6],
        "failed_dependencies": failed_dependencies[:6],
        "evidence_limits": evidence_limits,
        "lesson": str(payload.get("lesson") or "").strip(),
        "gate_execution_audit": deterministic_gate,
    }


def audit_experiment(
    *,
    race: str,
    parent_strategy_name: str,
    candidate_strategy_name: str,
    parent_strategy: str,
    candidate_strategy: str,
    parent_batch_dirs: list[Path],
    candidate_batch_dirs: list[Path],
    experiment_spec: dict[str, Any],
    outcome_comparison: dict[str, Any],
    model: str = "",
    summary_cache_path: Path | None = None,
    parent_analysis: dict[str, Any] | None = None,
    parent_analysis_record_paths: list[str] | None = None,
) -> dict[str, Any]:
    summary_cache = MatchSummaryCache(summary_cache_path)
    compact_parent_analysis = _compact_parent_analysis(parent_analysis or {})
    parent_records = _evidence_from_batches(
        parent_batch_dirs, strategy=parent_strategy_name
    )
    candidate_records = _evidence_from_batches(
        candidate_batch_dirs, strategy=candidate_strategy_name
    )
    gate_execution_audit = _audit_gate_execution(
        candidate_records,
        experiment_spec=experiment_spec,
    )
    if not compact_parent_analysis or not candidate_records:
        return {
            "implementation_verdict": "unknown",
            "hypothesis_verdict": "inconclusive",
            "mechanism_evidence": [],
            "combat_evidence": [],
            "runtime_findings": [],
            "salvageable_changes": [],
            "failed_dependencies": [],
            "evidence_limits": [
                "parent prior analysis or candidate match records were unavailable"
            ],
            "lesson": "The mechanism could not be audited from the available evidence.",
            "gate_execution_audit": gate_execution_audit,
        }
    selected_model = str(model or "").strip() or DEFAULT_ANALYSIS_MODEL
    analyzed_parent_paths = {
        MatchSummaryCache.key(path)
        for path in (parent_analysis_record_paths or [])
        if str(path).strip()
    }
    new_parent_records = (
        [
            record
            for record in parent_records
            if MatchSummaryCache.key(record.file) not in analyzed_parent_paths
        ]
        if analyzed_parent_paths
        else []
    )
    covered_parent_count = len(parent_records) - len(new_parent_records)
    print(
        f"    [parent: {parent_strategy_name}] reusing prior cross-match analysis "
        f"for {covered_parent_count} matches; {len(new_parent_records)} newly added "
        "parent matches require summaries",
        flush=True,
    )
    parent_new_summaries = _summarize_records(
        new_parent_records,
        strategy_name=parent_strategy_name,
        race=race,
        model=selected_model,
        label="parent-new",
        cache=summary_cache,
    )
    audit_focus = {
        key: value
        for key, value in {
            "minimum_material_change": (
                (experiment_spec.get("mechanism_prediction") or {}).get(
                    "minimum_material_change"
                )
                if isinstance(experiment_spec.get("mechanism_prediction"), dict)
                else ""
            ),
            "expected_change": (
                (experiment_spec.get("mechanism_prediction") or {}).get(
                    "expected_change"
                )
                if isinstance(experiment_spec.get("mechanism_prediction"), dict)
                else ""
            ),
            "material_behavior_change": (
                (experiment_spec.get("intervention_package") or {}).get(
                    "material_behavior_change"
                )
                if isinstance(experiment_spec.get("intervention_package"), dict)
                else ""
            ),
        }.items()
        if str(value or "").strip()
    }
    candidate_summaries = _summarize_records(
        candidate_records,
        strategy_name=candidate_strategy_name,
        race=race,
        model=selected_model,
        label="candidate",
        cache=summary_cache,
        audit_focus=audit_focus,
    )
    raw = call_json_llm(
        build_experiment_audit_prompt(
            parent_strategy=parent_strategy,
            candidate_strategy=candidate_strategy,
            experiment_spec=experiment_spec,
            parent_analysis=compact_parent_analysis,
            candidate_summaries=candidate_summaries,
            outcome_comparison=outcome_comparison,
            capability_manifest=build_executor_capability_manifest(race),
            gate_execution_audit=gate_execution_audit,
            parent_new_summaries=parent_new_summaries,
        ),
        model=selected_model,
        is_reasoning=EXPERIMENT_AUDIT_ENABLE_REASONING,
    )
    return _normalize_audit(
        raw,
        candidate_summaries=candidate_summaries,
        require_observed_probes=False,
        gate_execution_audit=gate_execution_audit,
    )


__all__ = ["audit_experiment", "build_experiment_audit_prompt"]
