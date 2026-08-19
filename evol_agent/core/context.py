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
    """Pass complete single-game summaries through without recropping events."""
    return render_single_game_analyses(analyses)


def render_knowledge_results(runs: list[dict[str, Any]]) -> str:
    """Compact knowledge answers for Cross-match Decision, without tool traces."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        row: dict[str, Any] = {
            "question_id": str(run.get("question_id") or run.get("id") or "").strip(),
            "question": str(run.get("question") or "").strip(),
            "query_reason": str(run.get("query_reason") or "").strip(),
            "evidence_refs": list(run.get("evidence_refs") or []),
            "hypothesis_scope": str(run.get("hypothesis_scope") or "").strip(),
            "ok": bool(run.get("ok")),
        }
        if row["ok"]:
            row["answer"] = str(run.get("answer") or "").strip()
        else:
            row["error"] = str(run.get("error") or "knowledge query failed").strip()
        rows.append(row)
    if not rows:
        return "[]"
    return json_compact_block(rows)


def render_retrieval_evidence(packet: dict[str, Any]) -> str:
    """Render deterministic record/history query results as one evidence packet."""
    if not isinstance(packet, dict) or not packet:
        return "{}"
    return json_compact_block(packet)


def render_optimizer_decision(decision: dict[str, Any]) -> str:
    """Compact Cross-match Decision for the Optimizer; no match summaries."""
    priority = decision.get("priority_problem") or {}
    if not isinstance(priority, dict):
        priority = {"problem": str(priority)}
    plan = decision.get("plan") if isinstance(decision.get("plan"), dict) else {}
    payload = {
        "strengths_to_preserve": decision.get("strengths_to_preserve") or [],
        "priority_problem": {
            "problem": str(priority.get("problem") or "").strip(),
            "evidence": list(priority.get("evidence") or [])[:4],
            "control_class": str(priority.get("control_class") or "").strip(),
        },
        "hypothesis": str(decision.get("hypothesis") or "").strip(),
        "failure_mode_analysis": (
            dict(decision.get("failure_mode_analysis"))
            if isinstance(decision.get("failure_mode_analysis"), dict)
            else {}
        ),
        "priority_alignment": (
            dict(decision.get("priority_alignment"))
            if isinstance(decision.get("priority_alignment"), dict)
            else {}
        ),
        "mechanism_prediction": (
            dict(decision.get("mechanism_prediction"))
            if isinstance(decision.get("mechanism_prediction"), dict)
            else {}
        ),
        "retrieval_assessment": (
            dict(decision.get("retrieval_assessment"))
            if isinstance(decision.get("retrieval_assessment"), dict)
            else {}
        ),
        "plan": {
            "direction": str(plan.get("direction") or "").strip(),
            "material_behavior_change": str(
                plan.get("material_behavior_change") or ""
            ).strip(),
            "coordinated_changes": list(plan.get("coordinated_changes") or []),
            "preserve": list(plan.get("preserve") or []),
        },
    }
    return json_compact_block(payload)


def render_discovery_findings(discovery: dict[str, Any]) -> str:
    """Pass Round 1 findings to Round 2 as compact JSON, not scattered prose."""
    payload = {
        "strengths": discovery.get("strengths") or [],
        "weaknesses": discovery.get("weaknesses") or [],
        "unknowns": discovery.get("unknowns") or [],
        "opponent_pressure_patterns": discovery.get("opponent_pressure_patterns") or [],
        "matchup_patterns": discovery.get("matchup_patterns") or [],
        "query_plan": discovery.get("query_plan") or {},
    }
    return json_compact_block(payload)


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
    "render_discovery_findings",
    "render_knowledge_results",
    "render_optimizer_decision",
    "render_retrieval_evidence",
    "render_sc2_knowledge",
    "render_single_game_analyses",
    "render_skill_context",
]
