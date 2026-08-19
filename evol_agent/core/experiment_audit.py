from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from .capabilities import build_executor_capability_manifest
from .config import DEFAULT_ANALYSIS_MODEL, MAX_CONCURRENT_MATCH_SUBAGENTS
from .llm import call_json_llm
from .match_summary import run_fixed_match_summary
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


class _MatchSummaryCache:
    """Persistent per-record summaries shared by analysis and experiment audit."""

    _SCHEMA = "sc2.experiment_match_summary_cache.v1"

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        if path is None or not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return
        entries = data.get("entries") if isinstance(data, dict) else None
        if isinstance(entries, dict):
            self._entries = {
                str(key): dict(value)
                for key, value in entries.items()
                if isinstance(value, dict)
            }

    @staticmethod
    def _key(path: str | Path) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    @staticmethod
    def _fingerprint(path: str | Path) -> tuple[int, int] | None:
        try:
            stat = Path(path).stat()
        except OSError:
            return None
        return stat.st_size, stat.st_mtime_ns

    def get(
        self,
        record: GameEvidence,
        *,
        strategy_name: str,
        race: str,
    ) -> dict[str, Any] | None:
        fingerprint = self._fingerprint(record.file)
        if fingerprint is None:
            return None
        with self._lock:
            entry = self._entries.get(self._key(record.file))
            if not isinstance(entry, dict):
                return None
            if (
                int(entry.get("size") or -1) != fingerprint[0]
                or int(entry.get("mtime_ns") or -1) != fingerprint[1]
                or str(entry.get("strategy") or "") != strategy_name
                or str(entry.get("race") or "") != race
                or not isinstance(entry.get("summary"), dict)
            ):
                return None
            return {
                "summary": dict(entry["summary"]),
                "errors": [str(item) for item in (entry.get("errors") or [])],
            }

    def put(
        self,
        record: GameEvidence,
        *,
        strategy_name: str,
        race: str,
        summary: dict[str, Any],
        errors: list[str],
        source: str,
    ) -> None:
        fingerprint = self._fingerprint(record.file)
        if fingerprint is None or not summary:
            return
        with self._lock:
            self._entries[self._key(record.file)] = {
                "record_path": str(Path(record.file).resolve()),
                "size": fingerprint[0],
                "mtime_ns": fingerprint[1],
                "strategy": strategy_name,
                "race": race,
                "summary": summary,
                "errors": errors,
                "source": source,
            }
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(
                {"schema": self._SCHEMA, "entries": self._entries},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temp_path.replace(self.path)


_PARENT_ANALYSIS_FIELDS = (
    "strategy_name",
    "race",
    "sample_size",
    "record_mix",
    "strengths_to_preserve",
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
    cache: _MatchSummaryCache,
) -> list[dict[str, Any]]:
    if not records:
        return []
    summaries: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, GameEvidence]] = []
    for index, record in enumerate(records, 1):
        cached = cache.get(
            record,
            strategy_name=strategy_name,
            race=race,
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
            ): index
            for index, record in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                _digest, analysis, ok, errors, _events = future.result()
                summaries[index] = {
                    "game": index,
                    "result": records[index - 1].result,
                    "summary": analysis.raw,
                    "errors": errors,
                }
                if ok:
                    cache.put(
                        records[index - 1],
                        strategy_name=strategy_name,
                        race=race,
                        summary=analysis.raw,
                        errors=errors,
                        source="experiment_audit",
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
) -> str:
    return f"""You are EvolAgent's post-experiment mechanism auditor.

The final optimization objective is winning decisive army engagements and then
the match. Attack timing, production synchronization, resource banking, scouting,
and gate attainment are intermediate mechanisms only.

Audit the pre-registered hypothesis using the parent analysis already produced
before candidate generation and the candidate match summaries collected after the
change. Do not re-analyze parent matches or propose a new strategy in this step.

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
  "evidence_limits":["important uncertainty"],
  "lesson":"one concise causal lesson for the next generation"
}}
"""


def _normalize_audit(raw: Any) -> dict[str, Any]:
    payload = raw.get("audit") if isinstance(raw, dict) and isinstance(raw.get("audit"), dict) else raw
    if not isinstance(payload, dict):
        payload = {}
    implementation = str(payload.get("implementation_verdict") or "unknown").strip()
    if implementation not in _IMPLEMENTATION_VERDICTS:
        implementation = "unknown"
    hypothesis = str(payload.get("hypothesis_verdict") or "inconclusive").strip()
    if hypothesis not in _HYPOTHESIS_VERDICTS:
        hypothesis = "inconclusive"
    if implementation != "implemented" and hypothesis in {"supported", "contradicted"}:
        hypothesis = "not_tested"

    def rows(name: str) -> list[Any]:
        value = payload.get(name)
        return list(value) if isinstance(value, list) else []

    return {
        "implementation_verdict": implementation,
        "hypothesis_verdict": hypothesis,
        "mechanism_evidence": rows("mechanism_evidence"),
        "combat_evidence": rows("combat_evidence"),
        "runtime_findings": [str(item) for item in rows("runtime_findings") if str(item)],
        "evidence_limits": [str(item) for item in rows("evidence_limits") if str(item)],
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
) -> dict[str, Any]:
    summary_cache = _MatchSummaryCache(summary_cache_path)
    compact_parent_analysis = _compact_parent_analysis(parent_analysis or {})
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
            "evidence_limits": [
                "parent prior analysis or candidate match records were unavailable"
            ],
            "lesson": "The mechanism could not be audited from the available evidence.",
        }
    print(
        f"    [parent: {parent_strategy_name}] reusing prior cross-match analysis; "
        "0 parent matches will be summarized",
        flush=True,
    )
    selected_model = str(model or "").strip() or DEFAULT_ANALYSIS_MODEL
    candidate_summaries = _summarize_records(
        candidate_records,
        strategy_name=candidate_strategy_name,
        race=race,
        model=selected_model,
        label="candidate",
        cache=summary_cache,
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
        ),
        model=selected_model,
        is_reasoning=True,
    )
    return _normalize_audit(raw)


__all__ = ["audit_experiment", "build_experiment_audit_prompt"]
