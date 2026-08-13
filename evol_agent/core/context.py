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
    "render_sc2_knowledge",
    "render_single_game_analyses",
    "render_skill_context",
]
