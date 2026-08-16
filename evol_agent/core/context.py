from __future__ import annotations

import json
from typing import Any

from .config import SKILL_FILES
from .types import BattleAnalysis, ToolObservation


def json_block(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def json_compact_block(data: Any) -> str:
    """Lossless JSON serialization with whitespace removed for repeated records."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


def render_skill_context(skill_texts: dict[str, str]) -> str:
    parts: list[str] = []
    for filename in SKILL_FILES:
        content = str(skill_texts.get(filename) or "").strip()
        if content:
            parts.append(f"### {filename}\n{content}")
    return "\n\n".join(parts)


def render_battle_analysis(analysis: BattleAnalysis) -> str:
    return json_block(analysis.raw or analysis.__dict__)


def render_single_game_analyses(analyses: list[BattleAnalysis]) -> str:
    if not analyses:
        return "No completed single-game summaries."
    blocks: list[str] = []
    for index, analysis in enumerate(analyses, 1):
        payload = analysis.raw or analysis.__dict__
        blocks.append(
            f"### Match Summary {index}\n"
            f"record_mix={analysis.record_mix}; sample_size={analysis.sample_size}\n"
            f"{json_compact_block(payload)}"
        )
    return "\n\n".join(blocks)


def render_batch_match_evidence(analyses: list[BattleAnalysis]) -> str:
    """Render aggregate metrics and complete deterministic evidence for every match."""
    if not analyses:
        return "No completed single-game summaries."

    fields = (
        "result",
        "duration",
        "outcome_summary",
        "timing_checkpoints",
        "final_metrics",
        "peak_metrics",
        "completion_milestones_s",
        "upgrade_milestones_s",
        "macro_target_history",
        "milestone_snapshots",
        "decision_metrics",
        "runtime_assessment",
        "action_space_selection_summary",
        "evidence_limits",
    )

    entries: list[tuple[int, BattleAnalysis, dict[str, Any]]] = []
    index_rows: list[dict[str, Any]] = []
    for index, analysis in enumerate(analyses, 1):
        raw = analysis.raw or analysis.__dict__
        raw = raw if isinstance(raw, dict) else {}
        result = str(raw.get("result") or "").strip()
        if not result:
            result = "Victory" if str(analysis.record_mix).startswith("1W/") else "Defeat"
        entries.append((index, analysis, raw))
        index_rows.append(
            {
                "match": index,
                "result": result,
                "duration": raw.get("duration"),
                "timing": raw.get("timing_checkpoints") or {},
                "final": raw.get("final_metrics") or {},
                "peak": raw.get("peak_metrics") or {},
                "runtime": (raw.get("runtime_assessment") or {}).get(
                    "classification", "unknown"
                ),
            }
        )

    def median(values: list[Any]) -> int | float | None:
        numeric = sorted(
            float(value)
            for value in values
            if isinstance(value, (int, float))
        )
        if not numeric:
            return None
        middle = len(numeric) // 2
        value = (
            numeric[middle]
            if len(numeric) % 2
            else (numeric[middle - 1] + numeric[middle]) / 2
        )
        return int(value) if value.is_integer() else round(value, 2)

    outcome_groups: dict[str, list[dict[str, Any]]] = {}
    for row in index_rows:
        outcome_groups.setdefault(str(row["result"]), []).append(row)
    aggregate = {
        "matches": len(index_rows),
        "outcomes": {key: len(value) for key, value in outcome_groups.items()},
        "runtime_classes": {
            value: sum(row["runtime"] == value for row in index_rows)
            for value in sorted({str(row["runtime"]) for row in index_rows})
        },
        "median_by_outcome": {
            outcome: {
                "first_attack_command_s": median(
                    [row["timing"].get("first_attack_command_s") for row in rows]
                ),
                "first_enemy_threat_s": median(
                    [row["timing"].get("first_enemy_threat_s") for row in rows]
                ),
                "peak_army_supply": median(
                    [row["peak"].get("army_supply") for row in rows]
                ),
                "final_own_lost_minerals": median(
                    [row["final"].get("own_lost_minerals") for row in rows]
                ),
            }
            for outcome, rows in outcome_groups.items()
        },
    }

    blocks = [
        "### Batch Aggregate\n" + json_compact_block(aggregate),
        "### All Match Index\n" + json_compact_block(index_rows),
    ]
    for index, analysis, raw in entries:
        evidence = {
            key: raw.get(key)
            for key in fields
            if key in raw
        }
        blocks.append(
            f"### Match Evidence {index}\n"
            f"record_mix={analysis.record_mix}; sample_size={analysis.sample_size}\n"
            f"{json_compact_block(evidence)}"
        )
    return "\n\n".join(blocks)


def render_sc2_knowledge(observations: list[ToolObservation]) -> str:
    if not observations:
        return "No SC2 knowledge results are available."
    blocks: list[str] = []
    for index, observation in enumerate(observations, 1):
        status = observation.status or ("complete" if observation.ok else "failed")
        problem_ids = ""
        plan_ids = ""
        if isinstance(observation.args, dict):
            raw_problem_ids = observation.args.get("problem_ids") or []
            if raw_problem_ids:
                problem_ids = ",".join(str(value) for value in raw_problem_ids)
            else:
                problem_ids = str(observation.args.get("problem_id") or "")
            plan_ids = ",".join(
                str(value) for value in observation.args.get("plan_ids") or []
            )
        header = f"### Knowledge Result {index} ({status})"
        if problem_ids:
            header += f" problems={problem_ids}"
        if plan_ids:
            header += f" plans={plan_ids}"
        # Pass through complete verified answers. Failed runs contribute only
        # their error so fallback dumps cannot masquerade as usable knowledge.
        if observation.ok:
            body = observation.summary or json_block(observation.result)
        else:
            body = str(observation.result.get("error") or observation.summary or "failed")
        blocks.append(f"{header}\nQuery args: {json_block(observation.args)}\nAnswer:\n{body}")
    return "\n\n".join(blocks)


__all__ = [
    "json_block",
    "json_compact_block",
    "render_battle_analysis",
    "render_batch_match_evidence",
    "render_sc2_knowledge",
    "render_single_game_analyses",
    "render_skill_context",
]
