from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from ..analysis.match_record import MatchRecordReader


RETRIEVAL_EVIDENCE_SCHEMA = "evol_retrieval_evidence.v1"
_EVIDENCE_REF_RE = re.compile(
    r"Game\s+(?P<game>\d+)\s*@\s*(?P<start>\d+(?:\.\d+)?)"
    r"(?:\s*[-–—]\s*(?P<end>\d+(?:\.\d+)?))?\s*s?",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")
_STOPWORDS = {
    "about",
    "after",
    "against",
    "before",
    "candidate",
    "change",
    "current",
    "during",
    "evidence",
    "experiment",
    "failure",
    "game",
    "match",
    "strategy",
    "with",
}


def _clean_strings(value: Any, *, limit: int | None = None) -> list[str]:
    if not isinstance(value, list):
        value = [value] if value else []
    rows = list(
        dict.fromkeys(str(item).strip() for item in value if str(item).strip())
    )
    return rows[:limit] if limit is not None else rows


def parse_evidence_reference(value: str) -> dict[str, Any] | None:
    match = _EVIDENCE_REF_RE.search(str(value or ""))
    if match is None:
        return None
    start = float(match.group("start"))
    end = float(match.group("end") or start)
    return {
        "game_index": int(match.group("game")),
        "start_s": min(start, end),
        "end_s": max(start, end),
        "reference": str(value).strip(),
    }


def _timeline_rows(record_path: str) -> list[dict[str, Any]]:
    timeline = MatchRecordReader(record_path).fixed_timeline()
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    for line in timeline.splitlines():
        if line.startswith("SCHEMA "):
            schema = json.loads(line[len("SCHEMA ") :])
            columns = [str(item) for item in schema.get("columns") or []]
        elif line.startswith("R ") and columns:
            values = json.loads(line[2:])
            if not isinstance(values, list):
                continue
            rows.append(
                {
                    column: values[index] if index < len(values) else None
                    for index, column in enumerate(columns)
                }
            )
    return rows


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_timeline_row(row: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "time_s",
        "trigger",
        "phase",
        "production",
        "technology",
        "army",
        "enemy",
        "opponent_truth_after_match",
        "combat",
        "threat",
        "macro_targets",
        "macro_progress_before_decision",
        "groups",
        "zones",
        "orders",
        "accepted_issues",
        "fallback_state",
    )
    return {
        key: row.get(key)
        for key in keep
        if row.get(key) not in (None, "", [], {})
    }


def _rows_for_window(
    rows: list[dict[str, Any]], start_s: float, end_s: float
) -> list[dict[str, Any]]:
    timed = [
        (time_s, row)
        for row in rows
        if (time_s := _number(row.get("time_s"))) is not None
    ]
    if not timed:
        return []
    before = [item for item in timed if item[0] <= start_s]
    after = [item for item in timed if item[0] >= end_s]
    selected: list[tuple[float, dict[str, Any]]] = []
    if before:
        selected.append(max(before, key=lambda item: item[0]))
    else:
        selected.append(min(timed, key=lambda item: abs(item[0] - start_s)))
    if after:
        selected.append(min(after, key=lambda item: item[0]))
    else:
        selected.append(min(timed, key=lambda item: abs(item[0] - end_s)))
    unique: dict[float, dict[str, Any]] = {}
    for time_s, row in selected:
        unique[time_s] = row
    return [_compact_timeline_row(unique[key]) for key in sorted(unique)]


def _interaction_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    own_assault_command = False
    enemy_in_owned_zone = False
    owned_zones_under_attack: list[str] = []
    for row in rows:
        for order in row.get("orders") or []:
            if not isinstance(order, list) or len(order) < 2:
                continue
            name = str(order[0] or "").casefold()
            args = order[1] if isinstance(order[1], dict) else {}
            movement_mode = str(
                args.get("movement_mode") or args.get("mode") or ""
            ).casefold()
            if name == "move_group" and movement_mode in {
                "assault",
                "attack",
                "advance",
                "push",
            }:
                own_assault_command = True
        for zone in row.get("zones") or []:
            if not isinstance(zone, list) or len(zone) < 8:
                continue
            owner = str(zone[1] or "").casefold()
            under_attack = bool(zone[3])
            try:
                visible_enemy = float(zone[7] or 0)
            except (TypeError, ValueError):
                visible_enemy = 0
            if owner == "own" and (under_attack or visible_enemy > 0):
                enemy_in_owned_zone = True
                zone_id = str(zone[0] or "")
                if zone_id:
                    owned_zones_under_attack.append(zone_id)
    if enemy_in_owned_zone and not own_assault_command:
        context = "enemy_pressure"
    elif own_assault_command and not enemy_in_owned_zone:
        context = "own_assault"
    elif own_assault_command and enemy_in_owned_zone:
        context = "contested_or_counterattack"
    else:
        context = "unclear"
    return {
        "classification": context,
        "own_assault_command_observed": own_assault_command,
        "enemy_in_owned_zone_observed": enemy_in_owned_zone,
        "owned_zones_with_pressure": list(dict.fromkeys(owned_zones_under_attack)),
        "limit": "This classifies the queried recorded window; it does not infer hidden movement between snapshots.",
    }


def query_match_records(
    records: list[Any],
    queries: list[dict[str, Any]],
    *,
    max_references: int = 16,
) -> dict[str, Any]:
    """Resolve cited Game N @ Ts references against recorded timeline rows."""
    record_paths = {
        index: str(getattr(record, "file", "") or "")
        for index, record in enumerate(records, 1)
    }
    row_cache: dict[int, list[dict[str, Any]]] = {}
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    used_references = 0
    for query_index, query in enumerate(queries, 1):
        query_id = str(query.get("id") or f"M{query_index}").strip()
        query_reason = str(query.get("query_reason") or query.get("reason") or "").strip()
        query_results: list[dict[str, Any]] = []
        for reference in _clean_strings(query.get("evidence_refs"), limit=8):
            if used_references >= max_references:
                break
            parsed = parse_evidence_reference(reference)
            if parsed is None:
                errors.append(f"{query_id}: could not parse evidence reference {reference!r}")
                continue
            game_index = parsed["game_index"]
            record_path = record_paths.get(game_index)
            if not record_path:
                errors.append(f"{query_id}: Game {game_index} is outside the current batch")
                continue
            try:
                if game_index not in row_cache:
                    row_cache[game_index] = _timeline_rows(record_path)
                rows = row_cache[game_index]
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{query_id}: Game {game_index} timeline query failed: {exc}")
                continue
            used_references += 1
            selected_rows = _rows_for_window(
                rows, parsed["start_s"], parsed["end_s"]
            )
            query_results.append(
                {
                    **parsed,
                    "record_path": record_path,
                    "timeline_rows": selected_rows,
                    "interaction_check": _interaction_check(selected_rows),
                }
            )
        results.append(
            {
                "query_id": query_id,
                "query_reason": query_reason,
                "results": query_results,
            }
        )
        if used_references >= max_references:
            break
    return {
        "source": "recorded_fixed_timeline",
        "query_count": len(results),
        "reference_count": used_references,
        "queries": results,
        "errors": errors,
    }


def _history_text(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item or "")
    fields = (
        "hypothesis",
        "failure_mode_analysis",
        "priority_alignment",
        "mechanism_prediction",
        "intervention_package",
        "plan_direction",
        "primary_change",
        "lesson",
        "decision",
        "implementation_verdict",
        "hypothesis_verdict",
        "gate_execution_audit",
        "first_commitment_timing",
    )
    return json.dumps(
        {key: item.get(key) for key in fields if key in item},
        ensure_ascii=False,
        default=str,
    )


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(value)
        if token.casefold() not in _STOPWORDS
    }


def _compact_history_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"text": str(item)}
    keep = (
        "experiment_id",
        "generation",
        "parent",
        "candidate",
        "hypothesis",
        "failure_mode_analysis",
        "priority_alignment",
        "mechanism_prediction",
        "intervention_package",
        "implementation_verdict",
        "hypothesis_verdict",
        "gate_execution_audit",
        "first_commitment_timing",
        "plan_direction",
        "patches",
        "decision",
        "parent_score",
        "candidate_score",
        "score_delta",
        "primary_change",
        "lesson",
    )
    return {key: item.get(key) for key in keep if key in item}


def query_experiment_history(
    prior_experiences: list[Any] | None,
    query: dict[str, Any],
    *,
    limit: int = 6,
) -> dict[str, Any]:
    """Retrieve relevant successful and failed prior interventions."""
    signature = _clean_strings(query.get("failure_signature"), limit=8)
    query_reason = str(query.get("query_reason") or query.get("reason") or "").strip()
    query_text = " ".join([query_reason, *signature]).strip()
    query_tokens = _tokens(query_text)
    ranked: list[tuple[float, int, Any]] = []
    experiences = list(prior_experiences or [])
    for index, item in enumerate(experiences):
        text = _history_text(item)
        item_tokens = _tokens(text)
        overlap = len(query_tokens & item_tokens)
        coverage = overlap / max(1, len(query_tokens))
        similarity = SequenceMatcher(None, query_text.casefold(), text.casefold()).ratio()
        ranked.append((coverage * 4.0 + similarity, index, item))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    selected = ranked[:limit]
    return {
        "source": "prior_evolution_experiments",
        "query_reason": query_reason,
        "failure_signature": signature,
        "results": [
            {
                "relevance_score": round(score, 4),
                "experience": _compact_history_item(item),
            }
            for score, _index, item in selected
        ],
        "evidence_limit": (
            "No prior experiments were available."
            if not experiences
            else "Similarity retrieves evidence; it does not establish causal equivalence."
        ),
    }


def build_retrieval_evidence_packet(
    *,
    records: list[Any],
    discovery: dict[str, Any],
    prior_experiences: list[Any] | None,
) -> dict[str, Any]:
    query_plan = (
        dict(discovery.get("query_plan"))
        if isinstance(discovery.get("query_plan"), dict)
        else {}
    )
    match_queries = [
        item
        for item in query_plan.get("match_evidence_queries") or []
        if isinstance(item, dict)
    ]
    experience_query = (
        dict(query_plan.get("experience_query"))
        if isinstance(query_plan.get("experience_query"), dict)
        else {}
    )
    return {
        "schema": RETRIEVAL_EVIDENCE_SCHEMA,
        "query_plan": query_plan,
        "match_record_evidence": query_match_records(records, match_queries),
        "historical_experience_evidence": query_experiment_history(
            prior_experiences, experience_query
        ),
    }


__all__ = [
    "RETRIEVAL_EVIDENCE_SCHEMA",
    "build_retrieval_evidence_packet",
    "parse_evidence_reference",
    "query_experiment_history",
    "query_match_records",
]
