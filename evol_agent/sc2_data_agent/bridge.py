"""Bridge from EvolAgent questions to deterministic SC2 knowledge packets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .sc2_data_store import DEFAULT_DATABASE_PATH
from .strategy_knowledge import build_strategy_knowledge, render_strategy_knowledge


KNOWLEDGE_VERIFICATION_SCHEMA = "strategy_knowledge.v1"
DEFAULT_DATA_PATH = DEFAULT_DATABASE_PATH


def find_knowledge_run_error(run: dict[str, Any] | None) -> str:
    """Explain why a persisted deterministic result is not safe to reuse."""
    payload = run if isinstance(run, dict) else {}
    if not payload.get("ok"):
        return str(payload.get("error") or "knowledge query failed")
    if not str(payload.get("answer") or "").strip():
        return "knowledge query returned an empty answer"
    if payload.get("verification_schema") != KNOWLEDGE_VERIFICATION_SCHEMA:
        return "knowledge result uses an obsolete verification schema"
    evidence = payload.get("dataset_evidence")
    if not isinstance(evidence, list) or not evidence:
        return "knowledge result has no deterministic dataset evidence"
    return ""


def is_knowledge_run_verified(run: dict[str, Any] | None) -> bool:
    return not find_knowledge_run_error(run)


def build_knowledge_query(item: dict[str, Any], *, race: str = "") -> str:
    """Build a stable checkpoint identity from the question and query fields."""
    parts = [str(item.get("question") or "").strip()]
    entities = [str(value).strip() for value in item.get("entities") or [] if str(value).strip()]
    needs = [str(value).strip() for value in item.get("needs") or [] if str(value).strip()]
    if entities:
        parts.append(f"entities={','.join(entities)}")
    if needs:
        parts.append(f"needs={','.join(needs)}")
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
    query = build_knowledge_query(item, race=race)
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
            "ok": False,
            "query": query,
            "answer": "",
            "error": f"{type(exc).__name__}: {exc}",
        }

    error = "" if packet.get("entities") else (
        "deterministic knowledge query resolved no Unit or Upgrade entities"
    )
    return {
        "question_id": question_id,
        "problem_ids": problem_ids,
        "problem_id": problem_ids[0] if problem_ids else question_id,
        "ok": not error,
        "query": query,
        "answer": answer,
        "error": error,
        "verification_schema": KNOWLEDGE_VERIFICATION_SCHEMA,
        "dataset_evidence": [
            {
                "tool": "get_strategy_knowledge",
                "arguments": {
                    "entities": packet.get("entities") or [],
                    "needs": packet.get("needs") or [],
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
