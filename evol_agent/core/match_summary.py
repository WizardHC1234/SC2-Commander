from __future__ import annotations

from typing import Any

from .loop_helpers import analysis_from_json, evidence_digest
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
    model: str,
    prefix: str,
) -> tuple[GameDigest, BattleAnalysis, bool, list[str], list[dict[str, Any]]]:
    """Extract one complete factual match summary without an LLM call."""
    record_id = f"match_{game_index:03d}"
    record_reader = MatchRecordReader(record.file)
    manifest = record_reader.manifest(record_id)
    payload = record_reader.deterministic_features(record_id)
    row_count = int((payload.get("decision_metrics") or {}).get("commander_rows") or 0)
    print(
        f"{prefix}MatchSummary {game_index}: {manifest['result']} "
        f"{manifest['duration']} ({row_count} Commander rows, deterministic features)",
        flush=True,
    )
    analysis = analysis_from_json(
        strategy_name=strategy_name,
        race=race,
        records=[record],
        data=payload,
    )
    digest = evidence_digest(record, game_index)
    digest.summary = payload["outcome_summary"]
    digest.raw["analysis"] = analysis.raw
    digest.raw["summary_quality"] = "deterministic"
    digest.raw["summary_input"] = {
        "format": "deterministic_match_features_v1",
        "commander_rows": row_count,
    }
    events = [
        {
            "step": 1,
            "action": "finish_match_summary",
            "input_format": "deterministic_match_features_v1",
            "summary_quality": "deterministic",
            "summary": analysis.raw,
        }
    ]
    return digest, analysis, True, [], events
