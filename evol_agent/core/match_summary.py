from __future__ import annotations

from typing import Any

from .config import MATCH_SUBAGENT_ENABLE_REASONING
from .llm import call_json_llm
from .loop_helpers import analysis_from_json, evidence_digest
from .prompts import build_fixed_match_summary_prompt
from .types import BattleAnalysis, GameDigest
from ..analysis.match_record import MatchRecordReader


def _degraded_payload(manifest: dict[str, Any], failure_reason: str) -> dict[str, Any]:
    return {
        "outcome_summary": (
            f"Degraded metadata-only summary: result={manifest.get('result', 'unknown')}; "
            f"duration={manifest.get('duration', 'unknown')}; "
            f"chunks={manifest.get('chunk_count', 0)}."
        ),
        "opening_and_economy": [],
        "production_technology_and_composition": [],
        "enemy_intelligence_and_map_state": [],
        "army_movement_and_engagements": [],
        "action_space_selection_summary": dict(
            manifest.get("action_space_selection")
            if isinstance(manifest.get("action_space_selection"), dict)
            else {}
        ),
        "commander_decision_summary": [],
        "macro_execution_summary": [],
        "army_execution_summary": [],
        "final_state": (
            f"Metadata result={manifest.get('result', 'unknown')} at "
            f"duration={manifest.get('duration', 'unknown')}."
        ),
        "evidence_limits": [failure_reason],
        "summary_quality": "degraded",
    }


def run_fixed_match_summary(
    *,
    strategy_name: str,
    race: str,
    record: Any,
    game_index: int,
    skill_texts: dict[str, str],
    model: str,
    prefix: str,
) -> tuple[GameDigest, BattleAnalysis, bool, list[str], list[dict[str, Any]]]:
    """Summarize one complete fixed timeline with one structured LLM call."""
    record_id = f"match_{game_index:03d}"
    record_reader = MatchRecordReader(record.file)
    manifest = record_reader.manifest(record_id)
    timeline = record_reader.fixed_timeline()
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
            skill_texts=skill_texts,
            match_timeline=timeline,
        ),
        model=model,
        is_reasoning=MATCH_SUBAGENT_ENABLE_REASONING,
    )
    if isinstance(result, dict) and isinstance(result.get("analysis"), dict):
        # Accept old action-wrapped output during migration without another LLM call.
        result = result["analysis"]

    errors: list[str] = []
    if isinstance(result, dict) and result:
        payload = dict(result)
        selection = manifest.get("action_space_selection")
        if isinstance(selection, dict) and selection:
            payload["action_space_selection_summary"] = dict(selection)
        analysis = analysis_from_json(
            strategy_name=strategy_name,
            race=race,
            records=[record],
            data=payload,
        )
        digest = evidence_digest(record, game_index)
        digest.summary = str(
            payload.get("outcome_summary") or "Single-match analysis completed."
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
    payload = _degraded_payload(manifest, failure_reason)
    analysis = analysis_from_json(
        strategy_name=strategy_name,
        race=race,
        records=[record],
        data=payload,
    )
    digest = evidence_digest(record, game_index)
    digest.summary = payload["outcome_summary"]
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
