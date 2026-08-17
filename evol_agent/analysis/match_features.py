from __future__ import annotations

from typing import Any


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_number(value: Any) -> int | float | None:
    number = _number(value)
    if number is None:
        return None
    return int(number) if number.is_integer() else round(number, 2)


def _first_time(rows: list[dict[str, Any]], predicate: Any) -> int | float | None:
    for row in rows:
        if predicate(row):
            return _compact_number(row.get("time"))
    return None


def _peak(rows: list[dict[str, Any]], key: str) -> int | float | None:
    values = [_number(row.get(key)) for row in rows]
    present = [value for value in values if value is not None]
    return _compact_number(max(present)) if present else None


def _counts(value: Any) -> dict[str, int | float]:
    if isinstance(value, dict):
        result: dict[str, int | float] = {}
        for key, raw in value.items():
            number = _compact_number(raw)
            if number is not None and number > 0:
                result[str(key)] = number
        return result
    result = {}
    for part in str(value or "").split("|"):
        name, separator, raw = part.strip().partition("=")
        number = _compact_number(raw) if separator else None
        if name and number is not None and number > 0:
            result[name] = number
    return result


def _milestone_snapshot(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    return {
        key: value
        for key, value in {
            "time_s": _compact_number(row.get("time")),
            "workers": _compact_number(row.get("workers")),
            "army_supply": _compact_number(row.get("army_supply")),
            "minerals": _compact_number(row.get("minerals")),
            "vespene": _compact_number(row.get("vespene")),
            "supply_left": _compact_number(row.get("supply_left")),
            "own_bases": _compact_number(row.get("own_bases")),
            "power_own": _compact_number(row.get("power_own")),
            "power_enemy": _compact_number(row.get("power_enemy")),
            "completed": row.get("completed") or {},
            "under_construction": row.get("under_construction") or {},
            "upgrades": row.get("upgrades") or [],
            "enemy_known": str(row.get("enemy_known") or ""),
            "macro_targets": row.get("macro_targets") or [],
        }.items()
        if value not in (None, "", [], {})
    }


def _execution_issues(observation: dict[str, Any]) -> list[str]:
    execution = _dict(observation.get("execution"))
    found: list[str] = []

    def visit(value: Any, parent: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = str(key).lower()
                if "issue" in normalized or "error" in normalized:
                    for entry in _list(item) if isinstance(item, list) else [item]:
                        text = str(entry or "").strip()
                        if text:
                            found.append(f"{parent + '.' if parent else ''}{key}:{text}")
                elif isinstance(item, (dict, list)):
                    visit(item, f"{parent}.{key}".strip("."))
        elif isinstance(value, list):
            for item in value:
                visit(item, parent)

    visit(execution)
    return list(dict.fromkeys(found))


def extract_match_features(
    *,
    chunks: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build complete factual evidence without asking an LLM to retell a match."""
    rows: list[dict[str, Any]] = []
    decision_issues: list[str] = []
    execution_issues: list[str] = []
    rejected_decisions = 0
    decision_errors = 0
    fragmented_snapshots = 0

    for chunk in chunks:
        observation = _dict(chunk.get("army_observation"))
        if not observation and str(chunk.get("agent_role") or "") != "commander":
            continue
        state = _dict(chunk.get("state"))
        decision = _dict(chunk.get("decision"))
        army_decision = _dict(decision.get("army"))
        control = _dict(observation.get("army_control"))
        groups = [_dict(item) for item in _list(control.get("groups"))]
        fragmented = sum(bool(group.get("is_fragmented")) for group in groups)
        fragmented_snapshots += int(fragmented > 0)

        issues = [
            str(item).strip()
            for item in _list(decision.get("issues")) + _list(army_decision.get("issues"))
            if str(item).strip()
        ]
        decision_issues.extend(issues)
        execution_issues.extend(_execution_issues(observation))
        if decision.get("accepted") is False:
            rejected_decisions += 1
        if str(decision.get("error") or army_decision.get("error") or "").strip():
            decision_errors += 1

        commands = [str(item) for item in _list(army_decision.get("commands"))]
        row = {
            "time": chunk.get("game_time"),
            "trigger": chunk.get("trigger"),
            "workers": state.get("workers"),
            "army_supply": state.get("army_supply"),
            "minerals": state.get("minerals"),
            "vespene": state.get("vespene"),
            "supply_left": state.get("supply_left"),
            "own_bases": state.get("own_bases"),
            "enemy_bases_known": state.get("enemy_bases_known"),
            "power_own": state.get("power_own"),
            "power_enemy": state.get("power_enemy"),
            "own_lost_minerals": state.get("own_lost_minerals"),
            "enemy_lost_minerals": state.get("enemy_lost_minerals"),
            "enemy_rushing": bool(state.get("enemy_rushing")),
            "enemy_proxy": bool(state.get("enemy_proxy")),
            "enemy_cloak": bool(state.get("enemy_cloak")),
            "commands": commands,
            "macro_targets": [
                {
                    "action": str(_dict(item).get("action") or ""),
                    "target": _dict(item).get("to_count"),
                }
                for item in _list(decision.get("macro_tasks"))
                if _dict(item).get("action")
            ],
            "fragmented_groups": fragmented,
            "completed": _counts(state.get("completed_buildings")),
            "under_construction": _counts(state.get("under_construction")),
            "upgrades": [str(item) for item in _list(state.get("upgrades"))],
            "enemy_known": state.get("enemy_known"),
        }
        rows.append(row)

    final = rows[-1] if rows else {}
    first_attack = _first_time(
        rows,
        lambda row: any(
            any(
                word in command.lower()
                for word in ("attack", "advance", "push", "assault")
            )
            for command in row["commands"]
        ),
    )
    first_retreat = _first_time(
        rows,
        lambda row: any("retreat" in command.lower() for command in row["commands"]),
    )
    first_supply_block = _first_time(
        rows,
        lambda row: _number(row.get("supply_left")) is not None
        and float(row["supply_left"]) <= 0,
    )
    first_resource_float = _first_time(
        rows,
        lambda row: (_number(row.get("minerals")) or 0) >= 1000,
    )
    first_second_base = _first_time(
        rows,
        lambda row: (_number(row.get("own_bases")) or 0) >= 2,
    )
    first_threat = _first_time(
        rows,
        lambda row: row["enemy_rushing"] or row["enemy_proxy"] or row["enemy_cloak"],
    )

    def row_at(time_value: int | float | None) -> dict[str, Any] | None:
        if time_value is None:
            return None
        return next(
            (row for row in rows if _compact_number(row.get("time")) == time_value),
            None,
        )

    peak_army_row = max(
        rows,
        key=lambda row: _number(row.get("army_supply")) or -1,
        default=None,
    )
    first_completed_s: dict[str, int | float] = {}
    first_upgrade_s: dict[str, int | float] = {}
    macro_target_stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        time_s = _compact_number(row.get("time"))
        if time_s is None:
            continue
        for entity, count in (row.get("completed") or {}).items():
            if count and entity not in first_completed_s:
                first_completed_s[entity] = time_s
        for upgrade in row.get("upgrades") or []:
            first_upgrade_s.setdefault(str(upgrade), time_s)
        for target in row.get("macro_targets") or []:
            action = str(target.get("action") or "")
            if not action:
                continue
            stats = macro_target_stats.setdefault(
                action,
                {
                    "first_s": time_s,
                    "last_s": time_s,
                    "first_target": target.get("target"),
                    "last_target": target.get("target"),
                },
            )
            stats["last_s"] = time_s
            stats["last_target"] = target.get("target")

    selector = _dict(manifest.get("action_space_selection"))
    selector_failure = bool(
        selector.get("fallback_used")
        or str(selector.get("dependency_error") or "").strip()
    )
    issue_rows = len(set(decision_issues + execution_issues))
    fragmentation_suspect = bool(
        rows and fragmented_snapshots >= max(3, len(rows) // 3)
    )
    runtime_contaminated = bool(
        selector_failure
        or rejected_decisions
        or decision_errors
        or (rows and issue_rows >= max(4, len(rows) // 2))
    )
    classification = (
        "runtime_contaminated"
        if runtime_contaminated
        else "runtime_suspect"
        if fragmentation_suspect
        else "valid_strategy_evidence"
    )
    runtime_signals = list(
        dict.fromkeys(
            [
                *(f"decision:{item}" for item in decision_issues),
                *(f"execution:{item}" for item in execution_issues),
                *(["selector:fallback_or_dependency_error"] if selector_failure else []),
                *(
                    [f"army:fragmented_in_{fragmented_snapshots}_snapshots"]
                    if fragmentation_suspect
                    else []
                ),
            ]
        )
    )

    timing = {
        key: value
        for key, value in {
            "first_second_base_s": first_second_base,
            "first_enemy_threat_s": first_threat,
            "first_attack_command_s": first_attack,
            "first_retreat_command_s": first_retreat,
            "first_supply_block_s": first_supply_block,
            "first_1000_minerals_s": first_resource_float,
        }.items()
        if value is not None
    }
    final_metrics = {
        key: _compact_number(final.get(key))
        for key in (
            "workers",
            "army_supply",
            "minerals",
            "vespene",
            "supply_left",
            "own_bases",
            "enemy_bases_known",
            "power_own",
            "power_enemy",
            "own_lost_minerals",
            "enemy_lost_minerals",
        )
        if _compact_number(final.get(key)) is not None
    }
    peak_metrics = {
        key: value
        for key, value in {
            "workers": _peak(rows, "workers"),
            "army_supply": _peak(rows, "army_supply"),
            "minerals": _peak(rows, "minerals"),
            "vespene": _peak(rows, "vespene"),
            "own_bases": _peak(rows, "own_bases"),
            "power_own": _peak(rows, "power_own"),
        }.items()
        if value is not None
    }
    return {
        "schema": "deterministic_match_features.v1",
        "result": str(manifest.get("result") or "?"),
        "duration": str(manifest.get("duration") or "?"),
        "outcome_summary": (
            f"result={manifest.get('result', '?')}; duration={manifest.get('duration', '?')}; "
            f"commander_rows={len(rows)}; evidence_class="
            f"{classification}"
        ),
        "timing_checkpoints": timing,
        "final_metrics": final_metrics,
        "peak_metrics": peak_metrics,
        "completion_milestones_s": first_completed_s,
        "upgrade_milestones_s": first_upgrade_s,
        "macro_target_history": macro_target_stats,
        "milestone_snapshots": {
            key: value
            for key, value in {
                "first_enemy_threat": _milestone_snapshot(row_at(first_threat)),
                "first_attack_command": _milestone_snapshot(row_at(first_attack)),
                "peak_army": _milestone_snapshot(peak_army_row),
                "final": _milestone_snapshot(final),
            }.items()
            if value
        },
        "decision_metrics": {
            "commander_rows": len(rows),
            "rejected_decisions": rejected_decisions,
            "decision_errors": decision_errors,
            "unique_issue_count": issue_rows,
            "fragmented_snapshots": fragmented_snapshots,
        },
        "runtime_assessment": {
            "classification": classification,
            "contaminated": runtime_contaminated,
            "suspect": fragmentation_suspect,
            "signals": runtime_signals,
        },
        "action_space_selection_summary": selector,
        "evidence_limits": [
            "Facts are deterministic snapshots; causal diagnosis remains batch-level.",
            "Missing timing keys mean the event was not observed, not that it never occurred.",
        ],
        "summary_quality": "deterministic",
    }


__all__ = ["extract_match_features"]

