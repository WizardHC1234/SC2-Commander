from __future__ import annotations

from typing import Any

from .config import MATCH_SUBAGENT_ENABLE_REASONING
from .llm import call_json_llm
from .loop_helpers import analysis_from_json, evidence_digest
from .prompts import build_fixed_match_summary_prompt
from .types import BattleAnalysis, GameDigest
from ..analysis.match_record import MatchRecordReader


def _duration_seconds(manifest: dict[str, Any], extracted: dict[str, Any]) -> int | float | None:
    metadata = extracted.get("metadata") if isinstance(extracted.get("metadata"), dict) else {}
    for value in (metadata.get("game_duration_seconds"), manifest.get("duration_s")):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        return int(number) if number.is_integer() else number
    formatted = str(manifest.get("duration") or "").strip()
    if not formatted or formatted == "?":
        return None
    parts = formatted.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 2:
        seconds = numbers[0] * 60 + numbers[1]
    elif len(numbers) == 3:
        seconds = numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    else:
        return None
    return int(seconds) if float(seconds).is_integer() else seconds


def _compact_mapping(value: Any) -> dict[str, Any] | str | None:
    if isinstance(value, dict):
        cleaned = {
            str(key): item
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
        return cleaned or None
    text = str(value or "").strip()
    return text or None


def _compact_event(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    event: dict[str, Any] = {}
    if raw.get("time_s") not in (None, ""):
        event["time_s"] = raw.get("time_s")
    elif raw.get("time") not in (None, ""):
        event["time_s"] = raw.get("time")
    trigger = str(raw.get("trigger") or "").strip()
    if trigger:
        event["trigger"] = trigger
    for key in ("own_state", "enemy_observed", "enemy_truth"):
        cleaned = _compact_mapping(raw.get(key))
        if cleaned:
            event[key] = cleaned
    commands = raw.get("commands")
    if isinstance(commands, list):
        names = [str(item).strip() for item in commands if str(item).strip()]
        if names:
            event["commands"] = names
    return event or None


def _compact_evidence_row(raw: Any, fields: tuple[str, ...]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    row: dict[str, Any] = {}
    if raw.get("time_s") not in (None, ""):
        row["time_s"] = raw.get("time_s")
    elif raw.get("time") not in (None, ""):
        row["time_s"] = raw.get("time")
    for field in fields:
        cleaned = _compact_mapping(raw.get(field))
        if cleaned:
            row[field] = cleaned
    return row or None


def _compact_mechanism_probe(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    status = str(raw.get("status") or "unknown").strip().lower()
    if status not in {"observed", "not_observed", "unknown"}:
        status = "unknown"
    observations: list[dict[str, Any]] = []
    for item in raw.get("observations") or []:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact") or "").strip()
        if not fact:
            continue
        row: dict[str, Any] = {"fact": fact}
        if item.get("time_s") not in (None, ""):
            row["time_s"] = item.get("time_s")
        observations.append(row)
    evidence_limit = str(raw.get("evidence_limit") or "").strip()
    if status == "observed" and not any(
        observation.get("time_s") not in (None, "") for observation in observations
    ):
        status = "unknown"
        evidence_limit = evidence_limit or (
            "observed mechanism evidence had no recorded timestamp"
        )
    return {
        "status": status,
        "observations": observations,
        "evidence_limit": evidence_limit,
    }


def _normalize_summary_payload(
    result: Any,
    *,
    manifest: dict[str, Any],
    duration_s: int | float | None,
) -> dict[str, Any] | None:
    if isinstance(result, dict) and isinstance(result.get("analysis"), dict):
        result = result["analysis"]
    if not isinstance(result, dict) or not result:
        return None
    events_raw = result.get("events")
    if events_raw is None:
        events: list[dict[str, Any]] = []
    elif not isinstance(events_raw, list):
        return None
    else:
        events = [item for raw in events_raw if (item := _compact_event(raw))]
    pressure_raw = result.get("enemy_pressure_events")
    if pressure_raw is None:
        pressure_events: list[dict[str, Any]] = []
    elif not isinstance(pressure_raw, list):
        return None
    else:
        pressure_fields = (
            "observed_cue",
            "own_defense",
            "enemy_observed",
            "enemy_truth",
            "outcome",
        )
        pressure_events = [
            item
            for raw in pressure_raw
            if (item := _compact_evidence_row(raw, pressure_fields))
        ]
    engagements_raw = result.get("major_engagements")
    if engagements_raw is None:
        major_engagements: list[dict[str, Any]] = []
    elif not isinstance(engagements_raw, list):
        return None
    else:
        engagement_fields = (
            "initiator",
            "own_force_before",
            "enemy_observed",
            "enemy_truth",
            "own_force_after",
            "own_reinforcement_after",
            "production_context_before",
            "offensive_command_before",
            "runtime_override",
            "retreat_policy",
            "loss_timing",
            "command_after_override",
            "blocked_offensive_resumed",
            "opening_gate_reapplied_after_commitment",
            "regroup_delay_seconds",
            "reengagement_time_s",
            "outcome",
        )
        major_engagements = [
            item
            for raw in engagements_raw
            if (item := _compact_evidence_row(raw, engagement_fields))
        ]
    payload_duration = result.get("duration_s")
    try:
        normalized_duration = float(payload_duration)
        if normalized_duration.is_integer():
            normalized_duration = int(normalized_duration)
    except (TypeError, ValueError):
        normalized_duration = duration_s
    payload = {
        "result": str(result.get("result") or manifest.get("result") or "").strip(),
        "duration_s": normalized_duration,
        "events": events,
        "enemy_pressure_events": pressure_events,
        "major_engagements": major_engagements,
    }
    mechanism_probe = _compact_mechanism_probe(result.get("mechanism_probe"))
    if mechanism_probe is not None:
        payload["mechanism_probe"] = mechanism_probe
    return payload


def _degraded_payload(
    manifest: dict[str, Any],
    failure_reason: str,
    *,
    duration_s: int | float | None,
) -> dict[str, Any]:
    return {
        "result": str(manifest.get("result") or "unknown"),
        "duration_s": duration_s,
        "events": [],
        "enemy_pressure_events": [],
        "major_engagements": [],
        "evidence_limits": [failure_reason],
        "summary_quality": "degraded",
    }


def run_fixed_match_summary(
    *,
    strategy_name: str,
    race: str,
    record: Any,
    game_index: int,
    model: str,
    prefix: str,
    audit_focus: dict[str, Any] | None = None,
) -> tuple[GameDigest, BattleAnalysis, bool, list[str], list[dict[str, Any]]]:
    """Summarize one complete fixed timeline with one structured LLM call."""
    record_id = f"match_{game_index:03d}"
    record_reader = MatchRecordReader(record.file)
    manifest = record_reader.manifest(record_id)
    timeline = record_reader.fixed_timeline()
    duration_s = _duration_seconds(manifest, record_reader._extract())
    row_count = sum(1 for line in timeline.splitlines() if line.startswith("R "))
    print(
        f"{prefix}MatchSummary {game_index}: {manifest['result']} "
        f"{manifest['duration']} ({row_count} Commander rows, fixed timeline)",
        flush=True,
    )

    result = call_json_llm(
        build_fixed_match_summary_prompt(
            strategy_name=strategy_name,
            race=race,
            record_manifest=manifest,
            match_timeline=timeline,
            audit_focus=audit_focus,
        ),
        model=model,
        is_reasoning=MATCH_SUBAGENT_ENABLE_REASONING,
    )
    payload = _normalize_summary_payload(
        result,
        manifest=manifest,
        duration_s=duration_s,
    )
    errors: list[str] = []
    if payload is not None:
        analysis = analysis_from_json(
            strategy_name=strategy_name,
            race=race,
            records=[record],
            data=payload,
        )
        digest = evidence_digest(record, game_index)
        digest.summary = (
            f"{payload['result']} duration_s={payload['duration_s']} "
            f"events={len(payload['events'])}"
        )
        digest.raw["analysis"] = analysis.raw
        digest.raw["summary_input"] = {
            "format": "fixed_match_timeline_v2",
            "commander_rows": row_count,
            "characters": len(timeline),
        }
        events = [
            {
                "step": 1,
                "action": "finish_match_summary",
                "input_format": "fixed_match_timeline_v2",
                "commander_rows": row_count,
                "input_characters": len(timeline),
                "summary": analysis.raw,
            }
        ]
        return digest, analysis, True, errors, events

    failure_reason = f"{record_id} fixed-timeline summary returned no JSON object"
    errors.append(failure_reason)
    payload = _degraded_payload(manifest, failure_reason, duration_s=duration_s)
    analysis = analysis_from_json(
        strategy_name=strategy_name,
        race=race,
        records=[record],
        data=payload,
    )
    digest = evidence_digest(record, game_index)
    digest.summary = payload["evidence_limits"][0]
    digest.raw["analysis"] = analysis.raw
    digest.raw["summary_quality"] = "degraded"
    events = [
        {
            "step": 1,
            "action": "finish_match_summary",
            "input_format": "fixed_match_timeline_v2",
            "summary_quality": "degraded",
            "error": failure_reason,
            "summary": analysis.raw,
        }
    ]
    return digest, analysis, False, errors, events
