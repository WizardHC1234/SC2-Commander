from __future__ import annotations

import json
from typing import Any

from ..sc2_data_agent.bridge import find_knowledge_run_error, is_knowledge_run_verified
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


def render_allowed_match_evidence(analysis: BattleAnalysis) -> str:
    payload = analysis.raw if isinstance(analysis.raw, dict) else {}
    targets = payload.get("optimization_targets")
    if not isinstance(targets, list):
        targets = analysis.optimization_targets
    lines: list[str] = []
    for target in targets or []:
        if not isinstance(target, dict):
            continue
        problem_id = str(target.get("problem_id") or "").strip() or "?"
        evidence = target.get("match_evidence")
        if not isinstance(evidence, list) or not evidence:
            continue
        lines.append(f"### {problem_id}")
        for item in evidence:
            text = str(item).strip()
            if text:
                lines.append(f"- {text}")
    return "\n".join(lines) if lines else "None"

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
        problem_id = ""
        if isinstance(observation.args, dict):
            problem_id = str(observation.args.get("problem_id") or "")
        header = f"### Knowledge Result {index} ({status})"
        if problem_id:
            header += f" problem_id={problem_id}"
        # Pass through complete verified answers. Failed runs contribute only
        # their error so fallback dumps cannot masquerade as usable knowledge.
        if observation.ok:
            body = observation.summary or json_block(observation.result)
        else:
            body = str(observation.result.get("error") or observation.summary or "failed")
        blocks.append(f"{header}\nQuery args: {json_block(observation.args)}\nAnswer:\n{body}")
    return "\n\n".join(blocks)


def render_knowledge_runs(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "No knowledge query results."
    blocks: list[str] = []
    for index, run in enumerate(runs, 1):
        verified = is_knowledge_run_verified(run)
        status = run.get("status") if verified else "failed"
        status = status or ("complete" if verified else "failed")
        problem_ids = run.get("problem_ids") if isinstance(run.get("problem_ids"), list) else []
        answer = (
            run.get("answer")
            if verified
            else find_knowledge_run_error(run)
            or run.get("error")
            or "knowledge query failed"
        )
        blocks.append(
            f"### Knowledge Query {index} ({status})\n"
            f"question_id: {run.get('question_id') or run.get('problem_id')}\n"
            f"problem_ids: {problem_ids or [run.get('problem_id')]}\n"
            f"query:\n{run.get('query') or '(none)'}\n"
            f"answer:\n{answer or '(empty)'}"
        )
    return "\n\n".join(blocks)
