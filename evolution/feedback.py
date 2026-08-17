from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from evol_agent.analysis.record_reader import find_record_jsons, is_completed_match_record


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def summarize_batch_evidence(batch_dir: Path) -> dict[str, Any]:
    outcomes = {"victory": 0, "tie": 0, "defeat": 0}
    durations: list[float] = []
    decision_counts: list[int] = []
    selected_tools: set[str] = set()
    selector_fallback_matches = 0
    record_paths: list[str] = []
    for path in find_record_jsons(batch_dir):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not is_completed_match_record(data):
            continue
        meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        result = str(meta.get("result") or "Defeat").strip().lower()
        if result == "draw":
            result = "tie"
        if result not in outcomes:
            result = "defeat"
        outcomes[result] += 1
        duration = _number(meta.get("game_duration_seconds"))
        if duration > 0:
            durations.append(duration)
        interactions = [item for item in data.get("interactions") or [] if isinstance(item, dict)]
        commander = [item for item in interactions if item.get("agent") == "commander"]
        decision_counts.append(len(commander))
        fallback = False
        for item in interactions:
            # Current records store strategy_tool_selection at the interaction
            # top level. Keep the legacy nested read for older batches.
            selected_tools.update(
                str(name) for name in item.get("selected_tools") or [] if str(name)
            )
            fallback = fallback or bool(item.get("fallback_used"))
            decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}
            selection = decision.get("tool_selection") if isinstance(decision.get("tool_selection"), dict) else {}
            selected_tools.update(str(name) for name in selection.get("selected_tools") or [] if str(name))
            fallback = fallback or bool(selection.get("fallback_used"))
        selector_fallback_matches += int(fallback)
        record_paths.append(str(path.resolve()))
    games = sum(outcomes.values())
    return {
        "batch_dir": str(batch_dir.resolve()),
        "games": games,
        "outcomes": outcomes,
        "score": ((outcomes["victory"] + 0.5 * outcomes["tie"]) / games if games else 0.0),
        "duration_seconds": {
            "mean": round(statistics.fmean(durations), 2) if durations else None,
            "median": round(statistics.median(durations), 2) if durations else None,
        },
        "commander_decisions_mean": (
            round(statistics.fmean(decision_counts), 2) if decision_counts else None
        ),
        "selected_tools": sorted(selected_tools),
        "selector_fallback_matches": selector_fallback_matches,
        "record_paths": record_paths,
    }


def compare_batch_evidence(parent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    def delta(path: tuple[str, ...]) -> float | None:
        left: Any = parent
        right: Any = candidate
        for key in path:
            left = left.get(key) if isinstance(left, dict) else None
            right = right.get(key) if isinstance(right, dict) else None
        if left is None or right is None:
            return None
        return round(_number(right) - _number(left), 4)

    return {
        "score_delta": delta(("score",)),
        "mean_duration_seconds_delta": delta(("duration_seconds", "mean")),
        "median_duration_seconds_delta": delta(("duration_seconds", "median")),
        "commander_decisions_mean_delta": delta(("commander_decisions_mean",)),
        "candidate_only_tools": sorted(
            set(candidate.get("selected_tools") or []) - set(parent.get("selected_tools") or [])
        ),
        "parent_only_tools": sorted(
            set(parent.get("selected_tools") or []) - set(candidate.get("selected_tools") or [])
        ),
    }


def combine_batch_evidence(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    games = int(first.get("games") or 0) + int(second.get("games") or 0)
    outcomes = {
        key: int(first.get("outcomes", {}).get(key) or 0)
        + int(second.get("outcomes", {}).get(key) or 0)
        for key in ("victory", "tie", "defeat")
    }

    def weighted(field: str, nested: str | None = None) -> float | None:
        left = first.get(field)
        right = second.get(field)
        if nested:
            left = left.get(nested) if isinstance(left, dict) else None
            right = right.get(nested) if isinstance(right, dict) else None
        left_games = int(first.get("games") or 0)
        right_games = int(second.get("games") or 0)
        present = [(left, left_games), (right, right_games)]
        present = [(value, count) for value, count in present if value is not None and count]
        denominator = sum(count for _value, count in present)
        if not denominator:
            return None
        return round(sum(_number(value) * count for value, count in present) / denominator, 2)

    return {
        "batch_dir": [first.get("batch_dir"), second.get("batch_dir")],
        "games": games,
        "outcomes": outcomes,
        "score": ((outcomes["victory"] + 0.5 * outcomes["tie"]) / games if games else 0.0),
        "duration_seconds": {
            "mean": weighted("duration_seconds", "mean"),
            "median": None,
        },
        "commander_decisions_mean": weighted("commander_decisions_mean"),
        "selected_tools": sorted(
            set(first.get("selected_tools") or []) | set(second.get("selected_tools") or [])
        ),
        "selector_fallback_matches": int(first.get("selector_fallback_matches") or 0)
        + int(second.get("selector_fallback_matches") or 0),
        "record_paths": [
            *list(first.get("record_paths") or []),
            *list(second.get("record_paths") or []),
        ],
    }


__all__ = [
    "combine_batch_evidence",
    "compare_batch_evidence",
    "summarize_batch_evidence",
]

