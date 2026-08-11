from __future__ import annotations

import json
from typing import Any


_LOW_LEVEL_KEYS = {
    "reasoning",
    "previous_decision",
    "signature",
    "target_zones",
    "text",
    "id",
    "unit_tag",
    "unit_tags",
    "position",
    "positions",
    "coordinates",
    "center",
}


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): cleaned
            for key, item in value.items()
            if str(key) not in _LOW_LEVEL_KEYS
            and (cleaned := _clean(item)) not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := _clean(item)) not in (None, "", [], {})
        ]
    if isinstance(value, float):
        return round(value, 1)
    return value


def _values(source: dict[str, Any], keys: list[str]) -> list[Any]:
    return [_clean(source.get(key)) for key in keys]


def _union_keys(chunks: list[dict[str, Any]], section: str) -> list[str]:
    keys: list[str] = []
    for chunk in chunks:
        observation = chunk.get("army_observation")
        if not isinstance(observation, dict):
            continue
        values = observation.get(section)
        if not isinstance(values, dict):
            continue
        for key in values:
            if key not in keys:
                keys.append(str(key))
    return keys


def _base_minerals(economy: dict[str, Any]) -> list[Any]:
    minerals = economy.get("own_base_minerals")
    if not isinstance(minerals, dict):
        return []
    summary = _values(
        minerals,
        ["Full", "Plenty", "Limited", "NearEmpty", "Empty"],
    )
    details = [
        _values(row, ["label", "resources", "minerals_left"])
        for row in minerals.get("details", [])
        if isinstance(row, dict)
    ]
    return [summary, details]


def _base_gas(economy: dict[str, Any]) -> list[list[Any]]:
    gas = economy.get("own_base_gas")
    if not isinstance(gas, dict):
        return []
    return [
        _values(
            row,
            [
                "label",
                "geysers",
                "owned_gas_structure_count",
                "geyser_slots",
                "available_geyser_slots",
                "gas_left",
            ],
        )
        for row in gas.get("details", [])
        if isinstance(row, dict)
    ]


def _groups(control: dict[str, Any]) -> tuple[list[list[Any]], set[str]]:
    rows: list[list[Any]] = []
    relevant_zones: set[str] = set()
    for group in control.get("groups", []):
        if not isinstance(group, dict):
            continue
        command = (
            group.get("current_command")
            if isinstance(group.get("current_command"), dict)
            else {}
        )
        rows.append(
            _values(
                group,
                [
                    "group_id",
                    "role",
                    "unit_count",
                    "power",
                    "nearest_zone_id",
                    "unit_type_counts",
                    "nearby_enemy_count",
                    "nearby_enemy_power",
                    "nearby_enemy_type_counts",
                    "is_fragmented",
                ],
            )
            + _values(
                command,
                ["destination_zone_id", "movement_mode", "retreat_ratio"],
            )
            + _values(
                group,
                [
                    "command_age_seconds",
                    "command_source",
                    "current_destination_reached",
                    "current_objective_status",
                ],
            )
        )
        for zone_id in (
            group.get("nearest_zone_id"),
            command.get("destination_zone_id"),
        ):
            if zone_id:
                relevant_zones.add(str(zone_id))
    return rows, relevant_zones


def _zones(control: dict[str, Any], relevant_zones: set[str]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    keys = [
        "zone_id",
        "owner",
        "zone_role",
        "under_attack",
        "own_units",
        "own_non_army_units",
        "known_enemy_units",
        "visible_enemy_units",
        "remembered_enemy_units",
        "own_contents",
        "visible_enemy_contents",
        "last_seen_enemy_contents",
        "vision_state",
        "enemy_information_age_seconds",
        "own_combat_power",
        "visible_enemy_power",
        "remembered_enemy_power",
        "combat_power_balance",
    ]
    for zone in control.get("zones", []):
        if not isinstance(zone, dict):
            continue
        if not (
            zone.get("owner") in {"own", "enemy"}
            or bool(zone.get("under_attack"))
            or str(zone.get("zone_id") or "") in relevant_zones
        ):
            continue
        rows.append(_values(zone, keys))
    return rows


def _macro_rows(decision: dict[str, Any]) -> tuple[list[list[Any]], set[str]]:
    tasks = decision.get("macro_tasks")
    if not isinstance(tasks, list):
        tasks = decision.get("new_tasks") if isinstance(decision.get("new_tasks"), list) else []
    rows = [
        [str(task.get("action") or task.get("name") or ""), task.get("to_count")]
        for task in tasks
        if isinstance(task, dict)
    ]
    return rows, {str(row[0]) for row in rows if row[0]}


def _macro_progress(decision: dict[str, Any]) -> list[list[Any]]:
    execution = decision.get("execution")
    macro = execution.get("macro") if isinstance(execution, dict) else {}
    if not isinstance(macro, dict):
        return []
    return [
        _values(task, ["action", "current_count", "to_count", "status"])
        for task in macro.get("active_macro_tasks", [])
        if isinstance(task, dict)
    ]


def _orders(decision: dict[str, Any], macro_names: set[str]) -> list[list[Any]]:
    rows = [
        [str(call.get("name") or ""), _clean(call.get("arguments") or {})]
        for call in decision.get("tool_calls", [])
        if isinstance(call, dict) and str(call.get("name") or "") not in macro_names
    ]
    wake_event = decision.get("wake_event")
    if isinstance(wake_event, dict) and not any(row[0] == "set_wake_event" for row in rows):
        rows.append(["set_wake_event", _clean(wake_event)])
    return rows


def _recon(capabilities: dict[str, Any]) -> list[Any]:
    scan = capabilities.get("scan") if isinstance(capabilities.get("scan"), dict) else {}
    scout = (
        capabilities.get("scv_scout")
        if isinstance(capabilities.get("scv_scout"), dict)
        else {}
    )
    return _values(
        scan,
        [
            "orbital_count",
            "orbital_energies",
            "available_scan_count",
            "scan_energy_cost",
            "last_target_zone_id",
            "last_result",
            "last_result_seconds_ago",
        ],
    ) + _values(
        scout,
        [
            "worker_count",
            "active",
            "active_scout_count",
            "active_target_zone_id",
            "last_target_zone_id",
            "last_result",
            "last_result_seconds_ago",
        ],
    )


def build_fixed_match_timeline(
    extracted: dict[str, Any],
    *,
    action_space_selection: dict[str, Any] | None = None,
    file_name: str = "",
    opponent_truth: dict[str, Any] | None = None,
) -> str:
    """Render every Commander snapshot as one fixed-schema compact table row."""
    chunks = extracted.get("chunks") if isinstance(extracted.get("chunks"), list) else []
    combat_keys = _union_keys(chunks, "combat")
    threat_keys = _union_keys(chunks, "threat_flags")
    truth_by_loop = {
        int(snapshot.get("requested_game_loop")): snapshot
        for snapshot in (opponent_truth or {}).get("snapshots", [])
        if isinstance(snapshot, dict)
        and str(snapshot.get("requested_game_loop", "")).isdigit()
    }
    economy_keys = [
        "minerals",
        "vespene",
        "mineral_income",
        "vespene_income",
        "supply_used",
        "supply_cap",
        "supply_free",
        "workers",
        "ideal_workers",
        "own_base_count",
    ]
    schema = {
        "schema": "fixed_match_timeline.v2",
        "format": "fixed arrays; read values using this schema; null/empty means absent at that snapshot",
        "columns": [
            "chunk",
            "time_s",
            "trigger",
            "phase",
            "economy",
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
            "recon",
            "orders",
            "accepted_issues",
            "fallback_state",
        ],
        "economy": economy_keys
        + [
            "active_mining_bases",
            "neutral_expansions",
            "base_minerals[[full,plenty,limited,near_empty,empty],[[label,state,left]]]",
            "base_gas[[label,geysers,owned,slots,available,left]]",
        ],
        "production": [
            "completed",
            "under_construction",
            "workers_en_route",
            "active_queues",
            "producer_addons",
        ],
        "technology": ["completed_upgrades", "upgrades_in_progress"],
        "army": [
            "army_supply",
            "army_power",
            "living_combat_composition",
            "training_combat_composition",
            "completed_counts",
        ],
        "enemy": [
            "race",
            "visible_composition",
            "known_composition",
            "known_combat_composition",
            "known_base_count",
            "last_observation_time",
            "seconds_since_last_seen",
            "macro_build",
            "known_types",
        ],
        "opponent_truth_after_match": [
            "game_loop",
            "resources[mineral,gas]",
            "supply[used,cap,army,workers]",
            "worker_count",
            "army_units",
            "completed_structures",
            "structures_in_progress",
            "completed_upgrades",
            "active_orders",
        ],
        "combat": combat_keys,
        "threat": threat_keys,
        "macro_targets": ["action", "to_count"],
        "macro_progress_before_decision": ["action", "current", "target", "status"],
        "groups": [
            "id",
            "role",
            "count",
            "power",
            "zone",
            "types",
            "near_enemy_count",
            "near_enemy_power",
            "near_enemy_types",
            "fragmented",
            "destination",
            "mode",
            "retreat_ratio",
            "command_age",
            "source",
            "destination_reached",
            "objective_status",
        ],
        "zones": [
            "id",
            "owner",
            "role",
            "under_attack",
            "own_units",
            "own_nonarmy_units",
            "known_enemy",
            "visible_enemy",
            "remembered_enemy",
            "own_contents",
            "visible_enemy_contents",
            "remembered_enemy_contents",
            "vision",
            "intel_age",
            "own_power",
            "visible_enemy_power",
            "remembered_enemy_power",
            "balance",
        ],
        "recon": [
            "orbital_count",
            "energies",
            "scan_available",
            "scan_cost",
            "scan_last_zone",
            "scan_last_result",
            "scan_age",
            "workers",
            "scout_active",
            "scout_count",
            "scout_target",
            "scout_last_zone",
            "scout_last_result",
            "scout_age",
        ],
        "orders": ["non_macro_tool", "arguments"],
        "omitted": [
            "reasoning prose",
            "previous_decision duplicate",
            "unit tags",
            "coordinates",
            "unrelated neutral zones",
            "macro tool calls duplicated by macro_targets",
        ],
    }
    metadata = extracted.get("metadata") if isinstance(extracted.get("metadata"), dict) else {}
    match = _clean(
        {
            "file_name": file_name,
            "map_name": metadata.get("map_name"),
            "matchup": metadata.get("matchup"),
            "my_race": metadata.get("my_race"),
            "enemy_race": metadata.get("enemy_race"),
            "result": metadata.get("result"),
            "game_duration_seconds": metadata.get("game_duration_seconds"),
            "game_duration_formatted": metadata.get("game_duration_formatted"),
            "opponent_id": metadata.get("opponent_id"),
            "strategy_hash": metadata.get("strategy_hash"),
            "commander_model_key": metadata.get("commander_model_key"),
            "save_reason": metadata.get("save_reason"),
            "opponent_truth_source": (
                (opponent_truth or {}).get("source")
                if truth_by_loop
                else "unavailable"
            ),
        }
    )
    rows: list[list[Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        observation = chunk.get("army_observation")
        role = str(chunk.get("agent_role") or "")
        has_observation = isinstance(observation, dict) and bool(observation)
        if not has_observation and role != "commander":
            continue
        observation = observation if has_observation else {}
        decision = chunk.get("decision") if isinstance(chunk.get("decision"), dict) else {}
        economy = observation.get("economy") if isinstance(observation.get("economy"), dict) else {}
        map_control = (
            observation.get("map_control")
            if isinstance(observation.get("map_control"), dict)
            else {}
        )
        production = (
            observation.get("production")
            if isinstance(observation.get("production"), dict)
            else {}
        )
        technology = (
            observation.get("technology")
            if isinstance(observation.get("technology"), dict)
            else {}
        )
        army = (
            observation.get("own_forces")
            if isinstance(observation.get("own_forces"), dict)
            else {}
        )
        enemy = observation.get("enemy") if isinstance(observation.get("enemy"), dict) else {}
        combat = observation.get("combat") if isinstance(observation.get("combat"), dict) else {}
        threat = (
            observation.get("threat_flags")
            if isinstance(observation.get("threat_flags"), dict)
            else {}
        )
        control = (
            observation.get("army_control")
            if isinstance(observation.get("army_control"), dict)
            else {}
        )
        capabilities = (
            observation.get("capabilities")
            if isinstance(observation.get("capabilities"), dict)
            else {}
        )
        time_state = (
            observation.get("time")
            if isinstance(observation.get("time"), dict)
            else {}
        )
        try:
            game_loop = int(time_state.get("game_loop"))
        except (TypeError, ValueError):
            game_loop = -1
        truth = truth_by_loop.get(game_loop, {})
        truth_resources = truth.get("resources") if isinstance(truth.get("resources"), dict) else {}
        truth_supply = truth.get("supply") if isinstance(truth.get("supply"), dict) else {}
        truth_row = (
            [
                truth.get("game_loop"),
                _values(truth_resources, ["minerals", "vespene"]),
                _values(truth_supply, ["used", "cap", "army", "workers"]),
                truth.get("workers"),
                _clean(truth.get("army_units") or {}),
                _clean(truth.get("structures_completed") or {}),
                _clean(truth.get("structures_in_progress") or {}),
                _clean(truth.get("upgrades") or []),
                _clean(truth.get("active_orders") or {}),
            ]
            if truth
            else []
        )
        group_rows, relevant_zones = _groups(control)
        macro_rows, macro_names = _macro_rows(decision)
        rows.append(
            [
                chunk_index,
                _clean(chunk.get("game_time")),
                chunk.get("trigger"),
                chunk.get("phase"),
                _values(economy, economy_keys)
                + [
                    _clean(map_control.get("active_mining_base_count")),
                    _clean(map_control.get("neutral_expansion_count")),
                    _base_minerals(economy),
                    _base_gas(economy),
                ],
                _values(
                    production,
                    [
                        "completed",
                        "under_construction",
                        "workers_en_route",
                        "active_queues",
                        "producer_addons",
                    ],
                ),
                _values(technology, ["completed_upgrades", "upgrades_in_progress"]),
                _values(
                    army,
                    [
                        "army_supply",
                        "army_power",
                        "combat_composition",
                        "training_combat_composition",
                        "completed_counts",
                    ],
                ),
                _values(
                    enemy,
                    [
                        "race",
                        "visible_composition",
                        "known_composition",
                        "known_combat_composition",
                        "known_base_count",
                        "last_observation_time",
                        "seconds_since_last_seen",
                        "macro_build",
                        "known_types",
                    ],
                ),
                truth_row,
                _values(combat, combat_keys),
                _values(threat, threat_keys),
                macro_rows,
                _macro_progress(decision),
                group_rows,
                _zones(control, relevant_zones),
                _recon(capabilities),
                _orders(decision, macro_names),
                [
                    decision.get("accepted"),
                    _clean(decision.get("issues") or []),
                    str(decision.get("error") or ""),
                ],
                _clean(chunk.get("state") or {}) if not observation else {},
            ]
        )
    dumps = lambda value: json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    lines = [
        f"SCHEMA {dumps(schema)}",
        f"MATCH {dumps(match)}",
        f"SELECTOR {dumps(_clean(action_space_selection or {}))}",
    ]
    lines.extend(f"R {dumps(row)}" for row in rows)
    return "\n".join(lines)
