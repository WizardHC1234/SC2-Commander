"""Bridge from EvolAgent questions to deterministic SC2 knowledge packets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .sc2_data_store import DEFAULT_DATABASE_PATH
from .strategy_knowledge import build_strategy_knowledge, render_strategy_knowledge


KNOWLEDGE_VERIFICATION_SCHEMA = "strategy_knowledge.v3"
DEFAULT_DATA_PATH = DEFAULT_DATABASE_PATH


def find_knowledge_run_error(run: dict[str, Any] | None) -> str:
    """Explain why a persisted deterministic result is not safe to reuse."""
    payload = run if isinstance(run, dict) else {}
    evidence = payload.get("dataset_evidence")
    if not payload.get("ok") and (not isinstance(evidence, list) or not evidence):
        return str(payload.get("error") or "knowledge query failed")
    if payload.get("verification_schema") != KNOWLEDGE_VERIFICATION_SCHEMA:
        return "knowledge result uses an obsolete verification schema"
    if not isinstance(evidence, list) or not evidence:
        return "knowledge result has no deterministic dataset evidence"
    packets = [
        item.get("result")
        for item in evidence
        if isinstance(item, dict) and isinstance(item.get("result"), dict)
    ]
    if not packets:
        return "knowledge result has no deterministic knowledge packet"
    for packet in packets:
        if packet.get("schema") != KNOWLEDGE_VERIFICATION_SCHEMA:
            return "knowledge result contains an obsolete deterministic packet"
        calculation_errors = [
            str(error).strip()
            for error in (packet.get("calculation_errors") or [])
            if str(error).strip()
        ]
        if calculation_errors:
            return "knowledge calculation failed: " + "; ".join(calculation_errors)
        coverage = packet.get("coverage")
        if not isinstance(coverage, dict):
            return "knowledge result has no request coverage audit"
        unresolved_entities = [
            str(value).strip()
            for value in (coverage.get("unresolved_entities") or [])
            if str(value).strip()
        ]
        if unresolved_entities:
            return "knowledge query left requested entities unresolved: " + ", ".join(
                unresolved_entities
            )
        unresolved_actions = [
            str(value).strip()
            for value in (coverage.get("unresolved_actions") or [])
            if str(value).strip()
        ]
        if unresolved_actions:
            return "knowledge query left requested actions unresolved: " + ", ".join(
                unresolved_actions
            )
        unsupported_claims = [
            str(value).strip()
            for value in (coverage.get("unsupported_claims") or [])
            if str(value).strip()
        ]
        if unsupported_claims:
            return "knowledge query requested facts outside the verified dataset: " + "; ".join(
                unsupported_claims
            )
        if coverage.get("complete") is not True:
            limits = [
                str(value).strip()
                for value in (packet.get("missing") or [])
                if str(value).strip()
            ]
            return "knowledge query coverage is incomplete" + (
                ": " + "; ".join(limits) if limits else ""
            )
        requested = int(packet.get("requested_calculation_count") or 0)
        completed = len(
            [
                result
                for result in (packet.get("calculations") or [])
                if isinstance(result, dict)
            ]
        )
        if completed < requested:
            return (
                "knowledge result completed only "
                f"{completed}/{requested} requested calculations"
            )
    if not payload.get("ok"):
        return str(payload.get("error") or "knowledge query failed")
    if not str(payload.get("answer") or "").strip():
        return "knowledge query returned an empty answer"
    return ""


def is_knowledge_run_verified(run: dict[str, Any] | None) -> bool:
    return not find_knowledge_run_error(run)


def build_knowledge_query(item: dict[str, Any], *, race: str = "") -> str:
    """Build a stable checkpoint identity from the question and query fields."""
    parts = [str(item.get("question") or "").strip()]
    entities = [str(value).strip() for value in item.get("entities") or [] if str(value).strip()]
    actions = [str(value).strip() for value in item.get("actions") or [] if str(value).strip()]
    needs = [str(value).strip() for value in item.get("needs") or [] if str(value).strip()]
    plan_ids = [str(value).strip() for value in item.get("plan_ids") or [] if str(value).strip()]
    calculations = [
        value for value in item.get("calculations") or [] if isinstance(value, dict)
    ]
    if entities:
        parts.append(f"entities={','.join(entities)}")
    if actions:
        parts.append(f"actions={','.join(actions)}")
    if needs:
        parts.append(f"needs={','.join(needs)}")
    if plan_ids:
        parts.append(f"plans={','.join(plan_ids)}")
    if calculations:
        import json

        parts.append(
            "calculations="
            + json.dumps(calculations, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    if race:
        parts.append(f"race={race}")
    return " | ".join(part for part in parts if part)


def run_knowledge_query(
    item: dict[str, Any],
    *,
    race: str = "",
    data_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one knowledge question without an LLM planning loop."""
    question_id = str(item.get("id") or item.get("question_id") or "").strip() or "?"
    problem_ids = [
        str(value).strip()
        for value in item.get("problem_ids") or []
        if str(value).strip()
    ]
    plan_ids = [
        str(value).strip()
        for value in item.get("plan_ids") or []
        if str(value).strip()
    ]
    query = build_knowledge_query(item, race=race)
    query_reason = str(item.get("query_reason") or item.get("reason") or "").strip()
    evidence_refs = [
        str(value).strip()
        for value in item.get("evidence_refs") or []
        if str(value).strip()
    ]
    hypothesis_scope = str(item.get("hypothesis_scope") or "").strip()
    try:
        packet = build_strategy_knowledge(
            item,
            race=race,
            data_path=data_path or DEFAULT_DATA_PATH,
        )
        answer = render_strategy_knowledge(packet)
    except Exception as exc:  # noqa: BLE001 - persist the query failure
        return {
            "question_id": question_id,
            "problem_ids": problem_ids,
            "problem_id": problem_ids[0] if problem_ids else question_id,
            "plan_ids": plan_ids,
            "ok": False,
            "query": query,
            "query_reason": query_reason,
            "evidence_refs": evidence_refs,
            "hypothesis_scope": hypothesis_scope,
            "answer": "",
            "error": f"{type(exc).__name__}: {exc}",
        }

    coverage = packet.get("coverage") if isinstance(packet.get("coverage"), dict) else {}
    if not (
        packet.get("entities")
        or packet.get("action_facts")
        or packet.get("calculations")
    ):
        error = "deterministic knowledge query resolved no entities or action facts"
    elif coverage.get("complete") is not True:
        limits = [
            str(value).strip()
            for value in (packet.get("missing") or [])
            if str(value).strip()
        ]
        error = "deterministic knowledge query did not fully cover the request" + (
            ": " + "; ".join(limits) if limits else ""
        )
    else:
        error = ""
    return {
        "question_id": question_id,
        "problem_ids": problem_ids,
        "problem_id": problem_ids[0] if problem_ids else question_id,
        "plan_ids": plan_ids,
        "ok": not error,
        "query": query,
        "query_reason": query_reason,
        "evidence_refs": evidence_refs,
        "hypothesis_scope": hypothesis_scope,
        "answer": answer,
        "error": error,
        "verification_schema": KNOWLEDGE_VERIFICATION_SCHEMA,
        "dataset_evidence": [
            {
                "tool": "get_strategy_knowledge",
                "arguments": {
                    "entities": packet.get("entities") or [],
                    "needs": packet.get("needs") or [],
                    "calculations": item.get("calculations") or [],
                },
                "result": packet,
            }
        ],
    }


__all__ = [
    "KNOWLEDGE_VERIFICATION_SCHEMA",
    "build_knowledge_query",
    "find_knowledge_run_error",
    "is_knowledge_run_verified",
    "run_knowledge_query",
]
