from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .capabilities import build_executor_capability_manifest
from .config import DEFAULT_ANALYSIS_MODEL, MAX_CONCURRENT_MATCH_SUBAGENTS
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
    parent_new_summaries: list[dict[str, Any]] | None = None,
) -> str:
    return f"""You are EvolAgent's post-experiment mechanism auditor.

The final optimization objective is winning decisive army engagements and then
the match. Attack timing, production synchronization, resource banking, scouting,
and gate attainment are intermediate mechanisms only.

Audit the pre-registered hypothesis using the parent analysis already produced
before candidate generation, any Parent confirmation-match summaries added after
that analysis, and the candidate match summaries collected after the change. Do
not re-analyze Parent matches already covered by the prior analysis or propose a
new strategy in this step.

The experiment is one primary failure mode addressed by a coordinated intervention
package. Audit the package as a whole, not merely the easiest individual lever.

Required reasoning discipline:
- First decide whether minimum_material_change was actually realized in candidate
  matches, not merely written into strategy.md or requested by Commander.
- Check whether the candidate collectively realized the declared
  intervention_package.coordinated_changes and material_behavior_change. A single
  upgrade, timing shift, or production change is not implementation when the rest
  of the required package did not occur.
- Compare opponent pressure windows and whether the candidate retained enough
  defensive combat power to reach its intended technology, upgrade, composition,
  or power spike. Do not evaluate a later power spike while ignoring repeated deaths
  before it becomes active.
- Compare own/enemy composition and support balance at major engagements. Keep
  unit-counter or exact numerical claims qualitative unless grounded by the
  pre-registered verified evidence.
- Trace strategy rule -> Commander decision -> applied command -> later game state.
- A requested action is not proof of execution. Sampled absence is not proof that
  an event never happened.
- If the hypothesis depends on transformations, abilities, transport loading,
  targeting, formation, or unit-level micro owned by runtime, mark
  execution_invalid unless the strategy can affect the mechanism through an
  exposed high-level control.
- Judge combat success primarily from the decisive engagement result, first-fight
  survival/force retention, ability to continue the push, and eventual match result.
- A higher score alone does not prove the mechanism. A lower score alone does not
  contradict it unless the minimum mechanism change occurred.
- Use implemented only when multiple candidate matches show the declared material
  change. Use underpowered when the strategy changed but the realized mechanism
  was too weak or inconsistent. Use unknown when the records cannot establish it.
- supported and contradicted are valid only when implementation_verdict=implemented.
- Candidate summaries may contain mechanism_probe. Treat an observed probe as
  implementation evidence only for that match. not_observed and unknown never
  prove contradiction. Require multiple observed candidate probes before using
  implementation_verdict=implemented.
- A rejected or execution-invalid candidate may still contain useful local changes.
  Separate the failed dependency from any candidate change that was actually
  realized and repeatedly associated with a better intermediate or combat outcome.
  Put only evidence-backed local changes in salvageable_changes. Do not salvage a
  merely written or requested change, and do not recommend inheriting the whole
  candidate package.

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

    return {
        "implementation_verdict": implementation,
        "hypothesis_verdict": hypothesis,
        "mechanism_evidence": rows("mechanism_evidence"),
        "combat_evidence": rows("combat_evidence"),
        "runtime_findings": [str(item) for item in rows("runtime_findings") if str(item)],
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
        "failed_dependencies": [
            str(item) for item in rows("failed_dependencies") if str(item)
        ][:6],
        "evidence_limits": evidence_limits,
        "lesson": str(payload.get("lesson") or "").strip(),
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
            parent_new_summaries=parent_new_summaries,
        ),
        model=selected_model,
        is_reasoning=True,
    )
    return _normalize_audit(
        raw,
        candidate_summaries=candidate_summaries,
        require_observed_probes=bool(audit_focus),
    )


__all__ = ["audit_experiment", "build_experiment_audit_prompt"]
