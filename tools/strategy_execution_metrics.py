"""Compute the SC2-LSEE strategy-execution metrics from match records.

The natural-language strategy is manually mapped once to a compact JSON spec.
The same deterministic evaluator then supports different units, quantities,
opponent races, difficulties, and evolved strategy versions.

Primary component metrics:
    economy_completion
    technology_completion
    army_completion
    engagement_trigger_consistency
    engagement_continuation_consistency
    overall_strategy_compliance

``engagement_execution_consistency`` and ``attack_completion`` remain as
backward-compatible aliases for the mean of the available engagement
components.  Matches that never assemble the strategy's attack gate and never
attack have no engagement opportunity, so both engagement components are N/A
rather than artificial perfect scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evol_agent.interaction_schema import (
    STRATEGY_COORDINATOR_INITIAL,
    interaction_get_dict,
)


DEFAULT_ATTACK_MODES = {"push", "assault", "search_and_destroy"}
DEFAULT_ATTACK_ROLES = {"main_force"}
DIFFICULTY_NAMES = {
    "veryeasy",
    "easy",
    "medium",
    "mediumhard",
    "hard",
    "harder",
    "veryhard",
    "cheatvision",
    "cheatmoney",
    "cheatinsane",
}
DIFFICULTY_ORDER = (
    "mediumhard",
    "hard",
    "harder",
    "veryhard",
)
MODEL_ROLE_KEYS = (
    "coordinator_model",
    "macro_model",
    "translator_model",
    "army_model",
)


def difficulty_sort_key(difficulty: str) -> tuple[int, str]:
    normalized = str(difficulty or "unknown").lower()
    try:
        return DIFFICULTY_ORDER.index(normalized), normalized
    except ValueError:
        return len(DIFFICULTY_ORDER), normalized


@dataclass
class Match:
    path: Path
    content_sha256: str
    metadata: dict[str, Any]
    macro: list[dict[str, Any]]
    top: list[dict[str, Any]]
    army: list[dict[str, Any]]
    timeline: list[dict[str, Any]]
    strategy_name: str

    @property
    def match_id(self) -> str:
        return self.path.stem

    @property
    def interval(self) -> float:
        return number(self.metadata.get("interval_seconds")) or 12.0

    @property
    def end_time(self) -> float:
        return number(self.metadata.get("game_duration_seconds"))


@dataclass
class RequirementResult:
    match_id: str
    batch: str
    strategy: str
    category: str
    requirement: str
    score: float
    passed: bool
    evidence: str


def number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def nested(obj: Any, *keys: str, default: Any = None) -> Any:
    current = obj
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def mean_available(*values: Optional[float]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    return statistics.fmean(available) if available else None


def discover_record_files(paths: Iterable[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for input_path in paths:
        if input_path.is_file():
            discovered.add(input_path)
            continue
        batch_dirs = (
            [input_path]
            if input_path.name.startswith("batch_")
            else [
                path
                for path in input_path.iterdir()
                if path.is_dir() and path.name.startswith("batch_")
            ]
        )
        for batch_dir in batch_dirs:
            for path in batch_dir.rglob("*.json"):
                if (
                    path.name == "meta.json"
                    or path.name.startswith("all_compressed_")
                    or path.name.endswith(".enemy_truth.json")
                ):
                    continue
                discovered.add(path)
    return sorted(discovered)


def normalize_current_interaction(item: dict[str, Any]) -> dict[str, Any]:
    """Expose the current Commander observation to the shared metric helpers."""
    normalized = dict(item)
    obs = item.get("observation")
    normalized["observation_full"] = obs

    # Current observations keep route distances in zone_topology.  The old
    # evaluator expects the same value directly on each zone when deciding
    # whether a neutral destination is a forward push.
    if isinstance(obs, dict):
        army_control = nested(obs, "army_control", default={})
        topology_zones = nested(
            army_control,
            "zone_topology",
            "zones",
            default=[],
        )
        distances = {
            str(zone.get("zone_id") or ""): zone.get(
                "path_distance_to_enemy_main"
            )
            for zone in topology_zones
            if isinstance(zone, dict)
        }
        for zone in army_control.get("zones") or []:
            if not isinstance(zone, dict):
                continue
            zone_id = str(zone.get("zone_id") or "")
            if "distance_to_enemy_main" not in zone and zone_id in distances:
                zone["distance_to_enemy_main"] = distances[zone_id]
    return normalized


def load_match(path: Path) -> Match:
    raw_text = path.read_text(encoding="utf-8-sig")
    data = json.loads(raw_text)
    interactions = data.get("interactions") or []
    current = sorted(
        (
            normalize_current_interaction(item)
            for item in interactions
            if isinstance(item, dict)
            and item.get("agent") == "commander"
            and isinstance(item.get("observation"), dict)
        ),
        key=lambda item: number(item.get("game_time")),
    )
    if current:
        metadata = data.get("metadata") or {}
        strategy_name = str(
            metadata.get("strategy_id")
            or metadata.get("strategy")
            or metadata.get("force_strategy")
            or next(
                (
                    item.get("strategy_id")
                    for item in interactions
                    if isinstance(item, dict) and item.get("strategy_id")
                ),
                "",
            )
        )
        return Match(
            path=path,
            content_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            metadata=metadata,
            macro=current,
            top=current,
            army=current,
            timeline=current,
            strategy_name=strategy_name,
        )

    macro = sorted(
        (
            item
            for item in interactions
            if item.get("trigger_reason") == "poll"
            and isinstance(item.get("observation_full"), dict)
        ),
        key=lambda item: number(item.get("game_time")),
    )
    army = sorted(
        (
            item
            for item in interactions
            if item.get("trigger_reason") == "army_control_agent_poll"
            and isinstance(item.get("observation_full"), dict)
        ),
        key=lambda item: number(item.get("game_time")),
    )
    top = sorted(
        (
            item
            for item in interactions
            if item.get("trigger_reason") == "strategy_coordination_poll"
            and isinstance(item.get("observation_full"), dict)
        ),
        key=lambda item: number(item.get("game_time")),
    )
    timeline = sorted(
        (
            item
            for item in interactions
            if isinstance(item.get("observation_full"), dict)
        ),
        key=lambda item: number(item.get("game_time")),
    )
    if not timeline:
        raise ValueError("record contains no structured observations")
    strategy_name = ""
    for item in interactions:
        initial = interaction_get_dict(item, STRATEGY_COORDINATOR_INITIAL)
        if not initial:
            continue
        strategy_name = str(
            initial.get("forced_strategy")
            or initial.get("selected_strategy")
            or ""
        )
        if strategy_name:
            break
    return Match(
        path=path,
        content_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        metadata=data.get("metadata") or {},
        macro=macro,
        top=top,
        army=army,
        timeline=timeline,
        strategy_name=strategy_name,
    )


def batch_name(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("batch_"):
            return parent.name
    return ""


def parse_match_info(path: Path) -> dict[str, str]:
    """Read the simple ``key: value`` fields saved beside each match."""
    if not path.is_file():
        return {}
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip():
            fields[key.strip().lower()] = value.strip()
    return fields


def match_models(
    match_path: Path,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Recover the four gameplay model roles without guessing from filenames."""
    info = parse_match_info(match_path.parent / "match_info.txt")
    metadata = metadata or {}
    commander_model = str(
        info.get("commander_model")
        or metadata.get("commander_model_key")
        or metadata.get("commander_model")
        or ""
    ).strip()
    if commander_model:
        return {"commander_model": commander_model}
    return {
        key: str(info.get(key) or metadata.get(key) or "").strip()
        for key in MODEL_ROLE_KEYS
    }


def model_signature(models: dict[str, str]) -> str:
    """Return one stable model label; mixed-role runs retain their full stack."""
    commander_model = str(models.get("commander_model") or "").strip()
    if commander_model:
        return commander_model
    roles = [str(models.get(key) or "").strip() for key in MODEL_ROLE_KEYS]
    present = [model for model in roles if model]
    if not present:
        return "unknown"
    if len(present) == len(roles) and len(set(present)) == 1:
        return present[0]
    return "/".join(model or "?" for model in roles)


def model_file_slug(model: str) -> str:
    """Create a readable, collision-resistant filename for a model signature."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model or "unknown")).strip("._")
    cleaned = cleaned[:80] or "unknown"
    digest = hashlib.sha256(str(model).encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}_{digest}"


def load_specs(
    spec_paths: Iterable[Path],
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for path in spec_paths:
        spec = json.loads(path.read_text(encoding="utf-8"))
        strategy = str(spec.get("strategy_name") or "")
        if not strategy:
            raise ValueError(f"strategy_name missing in spec: {path}")
        if int(spec.get("schema_version") or 0) != 3:
            raise ValueError(f"unsupported schema_version in spec: {path}")
        if not isinstance(spec.get("attack_gate"), dict) or not spec["attack_gate"]:
            raise ValueError(f"attack_gate missing in spec: {path}")
        unit_supply = spec.get("unit_supply") or {}
        missing_supply = [
            str(unit)
            for unit, target in spec["attack_gate"].items()
            if number(target) > 0 and number(unit_supply.get(unit)) <= 0
        ]
        if missing_supply:
            raise ValueError(
                f"unit_supply missing for attack_gate units in {path}: "
                f"{', '.join(missing_supply)}"
            )
        if strategy in specs:
            raise ValueError(f"duplicate strategy spec: {strategy}")
        spec["_spec_path"] = str(path)
        specs[strategy] = spec
    return specs


def select_spec(
    specs: dict[str, dict[str, Any]],
    match: Match,
) -> Optional[dict[str, Any]]:
    return specs.get(match.strategy_name)


def aliases(spec: dict[str, Any], entity: str) -> list[str]:
    configured = nested(spec, "unit_aliases", entity, default=None)
    if isinstance(configured, list) and configured:
        return [str(value) for value in configured]
    return [entity]


def observation(snapshot: dict[str, Any]) -> dict[str, Any]:
    return snapshot.get("observation_full") or {}


def completed_count(
    snapshot: dict[str, Any],
    spec: dict[str, Any],
    entity: str,
) -> float:
    obs = observation(snapshot)
    counts = nested(obs, "production", "completed", default=None)
    if not isinstance(counts, dict):
        counts = nested(obs, "own_forces", "completed_counts", default={})
    return sum(number(counts.get(alias)) for alias in aliases(spec, entity))


def under_construction_count(
    snapshot: dict[str, Any],
    spec: dict[str, Any],
    entity: str,
) -> float:
    counts = nested(
        observation(snapshot),
        "production",
        "under_construction",
        default={},
    )
    return sum(number(counts.get(alias)) for alias in aliases(spec, entity))


def combat_count(
    snapshot: dict[str, Any],
    spec: dict[str, Any],
    entity: str,
) -> float:
    counts = nested(
        observation(snapshot),
        "own_forces",
        "combat_composition",
        default={},
    )
    return sum(number(counts.get(alias)) for alias in aliases(spec, entity))


def base_count(snapshot: dict[str, Any]) -> float:
    return number(nested(observation(snapshot), "economy", "own_base_count", default=0))


def base_started_count(snapshot: dict[str, Any], spec: dict[str, Any]) -> float:
    # own_base_count already covers completed Command Centers and their morphs.
    return base_count(snapshot) + under_construction_count(snapshot, spec, "COMMANDCENTER")


def first_time(
    snapshots: Iterable[dict[str, Any]],
    predicate: Any,
) -> Optional[float]:
    for snapshot in snapshots:
        if predicate(snapshot):
            return number(snapshot.get("game_time"))
    return None


def snapshot_at_or_before(
    snapshots: Iterable[dict[str, Any]],
    time: float,
) -> Optional[dict[str, Any]]:
    selected = None
    for snapshot in snapshots:
        if number(snapshot.get("game_time")) <= time:
            selected = snapshot
        else:
            break
    return selected


def maximum_completed(
    snapshots: Iterable[dict[str, Any]],
    spec: dict[str, Any],
    entity: str,
) -> float:
    return max(
        (completed_count(snapshot, spec, entity) for snapshot in snapshots),
        default=0.0,
    )


# Observation views rename the SC2 internal Battlecruiser Weapon Refit id to
# the strategy-facing Yamato name. Accept either form in specs and records.
_UPGRADE_NAME_ALIASES = {
    "BATTLECRUISERENABLESPECIALIZATIONS": "YAMATOCANNON",
    "YAMATOCANNON": "YAMATOCANNON",
}


def normalize_upgrade_name(value: Any) -> str:
    name = str(value or "").strip().upper()
    return _UPGRADE_NAME_ALIASES.get(name, name)


def completed_upgrades(snapshot: dict[str, Any]) -> set[str]:
    values = nested(
        observation(snapshot),
        "technology",
        "completed_upgrades",
        default=[],
    )
    return {normalize_upgrade_name(value) for value in values}


def upgrade_satisfied(required: Any, completed: set[str]) -> bool:
    return normalize_upgrade_name(required) in completed


def parse_difficulty(opponent_id: str) -> str:
    for token in reversed(str(opponent_id or "").lower().split(".")):
        if token in DIFFICULTY_NAMES:
            return token
    return "unknown"


def first_main_attack(
    match: Match,
    spec: dict[str, Any],
) -> tuple[Optional[float], Optional[dict[str, Any]]]:
    attack_time, command, _ = first_main_attack_event(match, spec)
    return attack_time, command


def first_main_attack_event(
    match: Match,
    spec: dict[str, Any],
) -> tuple[
    Optional[float],
    Optional[dict[str, Any]],
    Optional[dict[str, Any]],
]:
    for snapshot in logical_army_decision_snapshots(match):
        event = main_attack_event_at(snapshot, spec)
        if event is not None:
            command, group = event
            return number(snapshot.get("game_time")), command, group
    return None, None, None


def group_unit_count(
    group: dict[str, Any],
    spec: dict[str, Any],
    entity: str,
) -> float:
    counts = group.get("unit_type_counts") or {}
    return sum(number(counts.get(alias)) for alias in aliases(spec, entity))


def group_gate_progress(
    group: dict[str, Any],
    spec: dict[str, Any],
) -> float:
    gate = spec.get("attack_gate") or {}
    unit_supply = spec.get("unit_supply") or {}
    if gate and not unit_supply:
        return statistics.fmean(
            min(
                group_unit_count(group, spec, unit) / number(target),
                1.0,
            )
            for unit, target in gate.items()
            if number(target) > 0
        )
    required_supply = sum(
        number(target) * (number(unit_supply.get(unit)) or 1.0)
        for unit, target in gate.items()
        if number(target) > 0
    )
    if required_supply <= 0:
        return 0.0
    completed_supply = sum(
        min(group_unit_count(group, spec, unit), number(target))
        * (number(unit_supply.get(unit)) or 1.0)
        for unit, target in gate.items()
        if number(target) > 0
    )
    return completed_supply / required_supply


def group_gate_complete(
    group: dict[str, Any],
    spec: dict[str, Any],
) -> bool:
    gate = spec.get("attack_gate") or {}
    return bool(gate) and all(
        group_unit_count(group, spec, unit) >= number(target)
        for unit, target in gate.items()
    )


def main_force_groups(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        group
        for group in nested(
            observation(snapshot),
            "army_control",
            "groups",
            default=[],
        )
        if group.get("role") == "main_force"
    ]


def main_group_gate_progress_at(
    snapshot: dict[str, Any],
    spec: dict[str, Any],
) -> float:
    return max(
        (
            group_gate_progress(group, spec)
            for group in main_force_groups(snapshot)
        ),
        default=0.0,
    )


def main_group_gate_complete_at(
    snapshot: dict[str, Any],
    spec: dict[str, Any],
) -> bool:
    return any(
        group_gate_complete(group, spec)
        for group in main_force_groups(snapshot)
    )


def zone_has_enemy_evidence(zone: dict[str, Any]) -> bool:
    return any(
        number(zone.get(field)) > 0
        for field in (
            "known_enemy_units",
            "visible_enemy_units",
            "remembered_enemy_units",
            "known_enemy_power",
            "visible_enemy_power",
            "remembered_enemy_power",
            "enemy_static_power",
        )
    ) or bool(zone.get("visible_enemy_contents")) or bool(
        zone.get("last_seen_enemy_contents")
    )


def is_enemy_side_zone(zone: dict[str, Any]) -> bool:
    return (
        zone.get("owner") == "enemy"
        or str(zone.get("zone_role") or "").startswith("enemy_")
    )


def first_attack_objective_matches(
    command: dict[str, Any],
    snapshot: dict[str, Any],
    spec: dict[str, Any],
) -> bool:
    """Check the strategy-specific first objective against visible state."""
    configured = [
        str(value)
        for value in spec.get("first_attack_zone_roles") or []
        if str(value)
    ]
    if not configured:
        return True
    army_control = nested(observation(snapshot), "army_control", default={})
    zones = {
        str(zone.get("zone_id") or ""): zone
        for zone in army_control.get("zones") or []
    }
    destination = zones.get(str(command.get("destination_zone_id") or "")) or {}
    destination_role = str(destination.get("zone_role") or "")

    # A fallback role is valid only when no preferred objective is currently
    # known.  This represents Battlecruiser's "known expansion, else main"
    # rule without granting all strategies the same broad target set.
    fallback_roles = {
        str(value)
        for value in spec.get("first_attack_fallback_zone_roles") or []
        if str(value)
    }
    known_preferred = any(
        str(zone.get("zone_role") or "") in configured
        and (
            str(zone.get("owner") or "") == "enemy"
            or zone_has_enemy_evidence(zone)
        )
        for zone in zones.values()
    )
    if known_preferred:
        direct_match = destination_role in configured
    else:
        direct_match = destination_role in set(configured) | fallback_roles
    if direct_match:
        return True

    # Ground armies can begin an enemy-main attack with a ``push`` to the next
    # primary-route zone.  That zone is a waypoint in the current action space,
    # not a different strategic objective.  ``assault`` on another base still
    # counts as a different objective.
    if (
        str(command.get("movement_mode") or "") == "push"
        and "enemy_main" in configured
    ):
        topology = {
            str(zone.get("zone_id") or ""): zone
            for zone in nested(
                army_control,
                "zone_topology",
                "zones",
                default=[],
            )
        }
        waypoint = topology.get(
            str(command.get("destination_zone_id") or "")
        ) or {}
        return bool(waypoint.get("on_primary_route"))
    return False


def continuation_objective_matches(
    command: dict[str, Any],
    group: dict[str, Any],
    zones: dict[str, dict[str, Any]],
) -> bool:
    """Allow continued offense, next objectives, and final cleanup."""
    mode = str(command.get("movement_mode") or "")
    if mode == "search_and_destroy":
        return True
    destination = zones.get(str(command.get("destination_zone_id") or "")) or {}
    return is_committed_offensive_destination(command, group, zones) and (
        is_enemy_side_zone(destination)
        or zone_has_enemy_evidence(destination)
    )


def safe_recovery_command(
    command: dict[str, Any],
    zones: dict[str, dict[str, Any]],
) -> bool:
    mode = str(command.get("movement_mode") or "")
    destination = zones.get(str(command.get("destination_zone_id") or "")) or {}
    return mode in {
        "hold",
        "regroup",
        "defensive_retreat",
        "panic_retreat",
    } and str(destination.get("owner") or "") == "own"


def is_committed_offensive_destination(
    command: dict[str, Any],
    group: dict[str, Any],
    zones: dict[str, dict[str, Any]],
) -> bool:
    mode = str(command.get("movement_mode") or "")
    destination = zones.get(command.get("destination_zone_id")) or {}
    if mode == "search_and_destroy":
        return True
    if mode == "assault":
        return (
            destination.get("owner") != "own"
            and (
                is_enemy_side_zone(destination)
                or zone_has_enemy_evidence(destination)
            )
        )
    if mode != "push" or destination.get("owner") == "own":
        return False
    if (
        is_enemy_side_zone(destination)
        or zone_has_enemy_evidence(destination)
    ):
        return True

    current = zones.get(group.get("nearest_zone_id")) or {}
    current_distance = current.get("distance_to_enemy_main")
    destination_distance = destination.get("distance_to_enemy_main")
    return (
        isinstance(current_distance, (int, float))
        and isinstance(destination_distance, (int, float))
        and float(destination_distance) < float(current_distance)
    )


def main_attack_event_at(
    snapshot: dict[str, Any],
    spec: dict[str, Any],
) -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
    """Return the first offensive command and its gathered main force."""
    if policy_parse_failed(snapshot):
        return None

    modes = set(spec.get("attack_modes") or DEFAULT_ATTACK_MODES)
    roles = set(spec.get("attack_group_roles") or DEFAULT_ATTACK_ROLES)
    obs = observation(snapshot)
    groups = {
        group.get("group_id"): group
        for group in nested(obs, "army_control", "groups", default=[])
    }
    zones = {
        zone.get("zone_id"): zone
        for zone in nested(obs, "army_control", "zones", default=[])
    }
    commands = issued_army_commands(snapshot)
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for command in commands or []:
        if command.get("movement_mode") not in modes:
            continue
        group_id = str(command.get("group_id") or "")
        if not group_id:
            continue
        group = groups.get(group_id) or {}
        if number(group.get("unit_count")) <= 0:
            continue
        if not is_committed_offensive_destination(command, group, zones):
            continue
        candidates.append((command, group))

    anchor = next(
        (
            (command, group)
            for command, group in candidates
            if not roles or group.get("role") in roles
        ),
        None,
    )
    if anchor is None:
        return None

    # The current Commander deliberately keeps distant newly produced units in
    # group_1 until they merge into group_0.  They may reinforce the same
    # objective, but they must not make an under-strength gathered main force
    # appear to satisfy the attack gate.
    anchor_command, anchor_group = anchor
    return anchor_command, {
        **anchor_group,
        "group_ids": [anchor_group.get("group_id")],
    }


def main_attack_command_at(
    snapshot: dict[str, Any],
    spec: dict[str, Any],
) -> Optional[dict[str, Any]]:
    event = main_attack_event_at(snapshot, spec)
    return event[0] if event is not None else None


def first_group_gate_time(
    match: Match,
    spec: dict[str, Any],
    group_id: str,
    *,
    no_later_than: Optional[float] = None,
) -> Optional[float]:
    for snapshot in match.army:
        snapshot_time = number(snapshot.get("game_time"))
        if no_later_than is not None and snapshot_time > no_later_than:
            break
        for group in main_force_groups(snapshot):
            if (
                group.get("group_id") == group_id
                and group_gate_complete(group, spec)
            ):
                return snapshot_time
    return None


def first_decision_at_or_after(
    snapshots: list[dict[str, Any]],
    time: float,
) -> Optional[float]:
    return next(
        (
            number(snapshot.get("game_time"))
            for snapshot in snapshots
            if number(snapshot.get("game_time")) >= time
        ),
        None,
    )


def policy_parse_failed(snapshot: dict[str, Any]) -> bool:
    if snapshot.get("agent") == "commander":
        return snapshot.get("accepted") is False or bool(snapshot.get("error"))
    policy = snapshot.get("army_control_agent_policy")
    return isinstance(policy, dict) and bool(policy.get("error"))


def issued_army_commands(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Return commands issued by the decision represented by ``snapshot``."""
    if snapshot.get("agent") == "commander":
        policy = snapshot.get("army_policy")
        commands = policy.get("commands") if isinstance(policy, dict) else []
    else:
        commands = nested(
            snapshot,
            "army_control_agent_policy",
            "parsed",
            "commands",
            default=[],
        )
    return [command for command in commands or [] if isinstance(command, dict)]


def logical_army_decision_snapshots(
    match: Match,
) -> list[dict[str, Any]]:
    """Collapse immediate failed retries into one logical decision.

    Failed outputs are retried much faster than the normal scheduler.
    A retry chain contributes one decision: its first successful result, or
    the final failed attempt when the chain never succeeds.  This prevents
    parser retries from creating artificial 12-second opportunities.
    """
    snapshots = match.army
    if not snapshots:
        return []

    retry_gap = max(match.interval * 0.5, 1e-9)
    decisions: list[dict[str, Any]] = []
    index = 0
    while index < len(snapshots):
        current = snapshots[index]
        if not policy_parse_failed(current):
            decisions.append(current)
            index += 1
            continue

        chosen = current
        cursor = index + 1
        while cursor < len(snapshots):
            previous_time = number(snapshots[cursor - 1].get("game_time"))
            current_time = number(snapshots[cursor].get("game_time"))
            if current_time - previous_time >= retry_gap:
                break
            chosen = snapshots[cursor]
            cursor += 1
            if not policy_parse_failed(chosen):
                break
        decisions.append(chosen)
        index = cursor
    return decisions


def missed_army_opportunities(
    decisions: list[dict[str, Any]],
    first_opportunity_time: float,
    attack_time: float,
) -> int:
    return sum(
        1
        for snapshot in decisions
        if first_opportunity_time
        <= number(snapshot.get("game_time"))
        < attack_time
    )


# After the attack gate is ready, Army is expected to decide on its own.
# Allow one extra normal Army cycle beyond the first ready poll for staging /
# issuing the command; further delay decays with half-life over Army cycles.
TRIGGER_ARMY_GRACE_MISSED_OPPORTUNITIES = 1
TRIGGER_ARMY_HALF_LIFE_MISSED_OPPORTUNITIES = 5


def delayed_attack_trigger_score(missed_opportunities: int) -> float:
    """Score delayed full-strength attacks after the Army grace window."""
    if missed_opportunities <= TRIGGER_ARMY_GRACE_MISSED_OPPORTUNITIES:
        return 1.0
    excess = (
        missed_opportunities - TRIGGER_ARMY_GRACE_MISSED_OPPORTUNITIES
    )
    return float(
        0.5
        ** (
            excess
            / max(TRIGGER_ARMY_HALF_LIFE_MISSED_OPPORTUNITIES, 1)
        )
    )


def evaluate_attack_timing(
    match: Match,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Score readiness and missed executable Army decision opportunities."""
    logical_decisions = logical_army_decision_snapshots(match)
    attack_time, attack_command, attack_group = first_main_attack_event(
        match,
        spec,
    )
    attack_snapshot = (
        snapshot_at_or_before(match.army, attack_time)
        if attack_time is not None
        else None
    )
    objective_correct = (
        first_attack_objective_matches(
            attack_command,
            attack_snapshot,
            spec,
        )
        if attack_command is not None and attack_snapshot is not None
        else None
    )
    attack_group_progress = (
        group_gate_progress(attack_group, spec)
        if attack_group is not None
        else None
    )
    readiness_source = "gathered_main_force"
    common = {
        "commander_interval_seconds": match.interval,
        "trigger_army_grace_missed_opportunities": (
            TRIGGER_ARMY_GRACE_MISSED_OPPORTUNITIES
        ),
        "decision_half_life_opportunities": (
            TRIGGER_ARMY_HALF_LIFE_MISSED_OPPORTUNITIES
        ),
        "attack_time": attack_time,
        "attack_command": attack_command,
        "attack_group_progress": attack_group_progress,
        "attack_readiness_source": readiness_source,
        "objective_correct": objective_correct,
    }

    if attack_time is None:
        gate_time = first_time(
            logical_decisions,
            lambda snapshot: main_group_gate_complete_at(snapshot, spec),
        )
        return {
            **common,
            "gate_time": gate_time,
            "first_opportunity_time": None,
            "missed_army_opportunities": None,
            "attack_offset_seconds": None,
            "score": 0.0 if gate_time is not None else None,
            "evaluable": gate_time is not None,
            "status": (
                "gate_reached_attack_not_issued"
                if gate_time is not None
                else "gate_not_reached_not_evaluable"
            ),
        }

    attack_group_ids = attack_group.get("group_ids") or [
        attack_group.get("group_id")
    ]
    group_id = str(attack_group_ids[0] or "")
    gate_time = first_group_gate_time(
        match,
        spec,
        group_id,
        no_later_than=attack_time,
    )
    offset = attack_time - gate_time if gate_time is not None else None

    if float(attack_group_progress or 0.0) < 1.0:
        readiness_score = float(attack_group_progress or 0.0)
        score = readiness_score if objective_correct is not False else 0.0
        return {
            **common,
            "gate_time": gate_time,
            "first_opportunity_time": None,
            "missed_army_opportunities": None,
            "attack_offset_seconds": offset,
            "score": score,
            "evaluable": True,
            "status": (
                "understrength_wrong_objective"
                if objective_correct is False
                else "understrength_attack"
            ),
        }

    if gate_time is None:
        # The attacking group is complete in the attack snapshot, so that
        # snapshot itself is the first observable gate at the latest.
        gate_time = attack_time
        offset = 0.0

    # Commander decides directly: each accepted decision after gate readiness
    # is one executable opportunity.
    first_opportunity_time = first_decision_at_or_after(
        logical_decisions,
        gate_time,
    )
    if (
        first_opportunity_time is None
        or attack_time < first_opportunity_time
    ):
        first_opportunity_time = attack_time
        missed_opportunities = 0
    else:
        missed_opportunities = missed_army_opportunities(
            logical_decisions,
            first_opportunity_time,
            attack_time,
        )

    if objective_correct is False:
        score = 0.0
        status = "wrong_first_objective"
    elif missed_opportunities <= TRIGGER_ARMY_GRACE_MISSED_OPPORTUNITIES:
        score = 1.0
        status = (
            "on_first_opportunity"
            if missed_opportunities == 0
            else "within_army_grace"
        )
    else:
        score = delayed_attack_trigger_score(missed_opportunities)
        status = "delayed_attack"

    return {
        **common,
        "gate_time": gate_time,
        "first_opportunity_time": first_opportunity_time,
        "missed_army_opportunities": missed_opportunities,
        "attack_offset_seconds": offset,
        "score": score,
        "evaluable": True,
        "status": status,
    }


def applied_group_commands(
    snapshot: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return commands that are already active in the game executor."""
    obs = observation(snapshot)
    army_control = nested(obs, "army_control", default={})
    active = {
        str(command.get("group_id") or ""): command
        for command in army_control.get("current_commands") or []
        if command.get("group_id")
    }
    for group in army_control.get("groups") or []:
        group_id = str(group.get("group_id") or "")
        current = group.get("current_command")
        if group_id and group_id not in active and isinstance(current, dict):
            active[group_id] = current
    return active


def engagement_snapshot_consistency(
    snapshot: dict[str, Any],
    spec: dict[str, Any],
) -> Optional[float]:
    """Return phase-aware unit-weighted consistency for current Commander.

    The operational main force must continue a valid objective, clean up, or
    recover safely after falling below its rebuild gate.  Distant reinforcement
    is consistent only when joining the main force or its current objective.
    Runtime auto-retreat decisions are excluded because they are not authored
    by Commander.
    """
    obs = observation(snapshot)
    army_control = nested(obs, "army_control", default={})
    groups = army_control.get("groups") or []
    if not groups:
        return None
    zones = {
        str(zone.get("zone_id") or ""): zone
        for zone in army_control.get("zones") or []
    }
    commands = applied_group_commands(snapshot)
    main_groups = [group for group in groups if group.get("role") == "main_force"]
    offensive_main_groups = [
        group
        for group in main_groups
        if continuation_objective_matches(
            commands.get(str(group.get("group_id") or "")) or {},
            group,
            zones,
        )
    ]
    offensive_destinations = {
        commands[str(group.get("group_id") or "")].get(
            "destination_zone_id"
        )
        for group in offensive_main_groups
        if str(group.get("group_id") or "") in commands
        and commands[str(group.get("group_id") or "")].get(
            "destination_zone_id"
        )
    }
    main_force_zones = {
        group.get("nearest_zone_id")
        for group in main_groups
        if group.get("nearest_zone_id")
    }
    reinforcement_destinations = offensive_destinations | main_force_zones

    evaluated_units = 0.0
    consistent_units = 0.0
    for group in groups:
        group_id = str(group.get("group_id") or "")
        command = commands.get(group_id) or {}
        units = number(group.get("unit_count"))
        if units <= 0 or str(group.get("command_source") or "") == "auto_retreat":
            continue
        evaluated_units += units
        role = str(group.get("role") or "")
        if role == "main_force":
            offensive = continuation_objective_matches(command, group, zones)
            mode = str(command.get("movement_mode") or "")
            safely_positioned = safe_recovery_command(command, zones)
            recovering = safely_positioned and (
                mode in {"defensive_retreat", "panic_retreat"}
                or not group_gate_complete(group, spec)
            )
            consistent = offensive or recovering
        else:
            mode = str(command.get("movement_mode") or "")
            destination = command.get("destination_zone_id")
            consistent = (
                destination in reinforcement_destinations
                and mode in {"regroup", "push", "assault"}
            )
        if consistent:
            consistent_units += units
    return consistent_units / evaluated_units if evaluated_units > 0 else None


def applied_main_offense_present(
    snapshot: dict[str, Any],
    spec: dict[str, Any],
) -> bool:
    """Whether a main-force offensive command is active in Observation."""
    obs = observation(snapshot)
    army_control = nested(obs, "army_control", default={})
    zones = {
        str(zone.get("zone_id") or ""): zone
        for zone in army_control.get("zones") or []
    }
    commands = applied_group_commands(snapshot)
    modes = set(spec.get("attack_modes") or DEFAULT_ATTACK_MODES)
    roles = set(spec.get("attack_group_roles") or DEFAULT_ATTACK_ROLES)
    return any(
        (not roles or group.get("role") in roles)
        and str(command.get("movement_mode") or "") in modes
        and continuation_objective_matches(command, group, zones)
        for group in army_control.get("groups") or []
        if (
            command := commands.get(str(group.get("group_id") or ""))
        )
    )


def phase_aware_engagement_consistencies(
    snapshots: list[dict[str, Any]],
    spec: dict[str, Any],
) -> list[float]:
    """Return phase-aware continuation values for all later decisions."""
    values: list[float] = []
    for snapshot in snapshots:
        value = engagement_snapshot_consistency(snapshot, spec)
        if value is not None:
            values.append(value)
    return values


def normal_army_opportunity_snapshots(
    match: Match,
    start_time: float,
) -> list[dict[str, Any]]:
    """Return one snapshot per logical Commander decision after ``start_time``."""
    return [
        snapshot
        for snapshot in logical_army_decision_snapshots(match)
        if number(snapshot.get("game_time")) >= start_time
    ]


def evaluate_engagement_continuation(
    match: Match,
    spec: dict[str, Any],
    attack_evaluation: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Score sustained unit commitment after the first valid attack.

    All later logical Commander decisions are evaluated because recovery,
    rebuild, renewed offense, and cleanup are explicit parts of the strategy.
    """
    initiation = attack_evaluation or evaluate_attack_timing(match, spec)
    attack_time = initiation.get("attack_time")
    main_gate_time = first_time(
        logical_army_decision_snapshots(match),
        lambda snapshot: main_group_gate_complete_at(snapshot, spec),
    )
    common = {
        "attack_time": attack_time,
        "global_gate_time": main_gate_time,
        "first_applied_offense_time": None,
        "evaluated_opportunities": 0,
        "continuation_window_opportunities": None,
    }
    if attack_time is None:
        if main_gate_time is None:
            return {
                **common,
                "score": None,
                "status": "gate_not_reached_not_evaluable",
            }
        return {
            **common,
            "score": 0.0,
            "status": "gate_reached_attack_not_issued",
        }

    first_applied_time: Optional[float] = None
    for snapshot in match.army:
        snapshot_time = number(snapshot.get("game_time"))
        if snapshot_time < number(attack_time):
            continue
        if applied_main_offense_present(snapshot, spec):
            first_applied_time = snapshot_time
            break
    if first_applied_time is None:
        return {
            **common,
            "score": 0.0,
            "status": "offense_not_applied",
        }

    window_snapshots = normal_army_opportunity_snapshots(
        match,
        first_applied_time,
    )
    values = phase_aware_engagement_consistencies(
        window_snapshots,
        spec,
    )
    score = statistics.fmean(values) if values else None
    return {
        **common,
        "score": score,
        "status": "evaluated" if values else "not_evaluable_runtime_control",
        "first_applied_offense_time": first_applied_time,
        "evaluated_opportunities": len(values),
    }


def evaluate_engagement_execution(
    match: Match,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Combine condition-aware trigger and continuation consistency."""
    initiation = evaluate_attack_timing(match, spec)
    continuation = evaluate_engagement_continuation(
        match,
        spec,
        initiation,
    )
    trigger_score = initiation.get("score")
    trigger_status = str(initiation.get("status") or "")
    continuation_score = continuation.get("score")
    combined = mean_available(trigger_score, continuation_score)
    return {
        "score": combined,
        "trigger_score": trigger_score,
        "trigger_status": trigger_status,
        # Backward-compatible alias for callers using the previous name.
        "initiation_score": trigger_score,
        "continuation_score": continuation_score,
        "evaluated_opportunities": continuation.get(
            "evaluated_opportunities",
            0,
        ),
        "initiation": initiation,
        "continuation": continuation,
    }


def gate_progress_at(
    snapshot: dict[str, Any],
    spec: dict[str, Any],
) -> float:
    gate = spec.get("attack_gate") or {}
    if not gate:
        return 0.0
    unit_supply = spec.get("unit_supply") or {}
    if not unit_supply:
        return statistics.fmean(
            min(
                completed_count(snapshot, spec, unit) / number(target),
                1.0,
            )
            for unit, target in gate.items()
            if number(target) > 0
        )
    required_supply = sum(
        number(target) * (number(unit_supply.get(unit)) or 1.0)
        for unit, target in gate.items()
        if number(target) > 0
    )
    if required_supply <= 0:
        return 0.0
    completed_supply = sum(
        min(completed_count(snapshot, spec, unit), number(target))
        * (number(unit_supply.get(unit)) or 1.0)
        for unit, target in gate.items()
        if number(target) > 0
    )
    return completed_supply / required_supply


def gate_complete_at(
    snapshot: dict[str, Any],
    spec: dict[str, Any],
) -> bool:
    gate = spec.get("attack_gate") or {}
    return bool(gate) and all(
        completed_count(snapshot, spec, unit) >= number(target)
        for unit, target in gate.items()
    )


def score_base_rule(
    rule: dict[str, Any],
    match: Match,
    spec: dict[str, Any],
    attack_time: Optional[float],
) -> tuple[float, str]:
    rule_type = str(rule.get("type") or "")
    target = int(rule.get("count") or 0)
    tolerance = number(rule.get("tolerance_seconds") or match.interval)

    started_time = first_time(
        match.timeline,
        lambda snapshot: base_started_count(snapshot, spec) >= target,
    )
    completed_time = first_time(
        match.timeline,
        lambda snapshot: base_count(snapshot) >= target,
    )
    maximum = max((base_count(snapshot) for snapshot in match.timeline), default=0.0)

    if rule_type == "at_least_before_attack":
        deadline = attack_time if attack_time is not None else match.end_time
        passed = started_time is not None and started_time <= deadline
    elif rule_type == "start_after_attack":
        passed = (
            attack_time is not None
            and started_time is not None
            and started_time >= attack_time - tolerance
        )
    elif rule_type == "at_least_by_end":
        passed = completed_time is not None
    elif rule_type == "at_most_until_attack":
        deadline = attack_time if attack_time is not None else match.end_time
        passed = all(
            base_started_count(snapshot, spec) <= target
            for snapshot in match.timeline
            if number(snapshot.get("game_time")) <= deadline
        )
    elif rule_type == "at_most_until_end":
        passed = all(
            base_started_count(snapshot, spec) <= target
            for snapshot in match.timeline
        )
    else:
        raise ValueError(f"unsupported base rule: {rule_type}")

    evidence = (
        f"type={rule_type}; target={target}; started_time={started_time}; "
        f"completed_time={completed_time}; max_completed={maximum}"
    )
    return float(passed), evidence


def evaluate_match(
    match: Match,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[RequirementResult]]:
    strategy = str(spec.get("strategy_name") or "unknown")
    engagement_evaluation = evaluate_engagement_execution(match, spec)
    attack_evaluation = engagement_evaluation["initiation"]
    continuation_evaluation = engagement_evaluation["continuation"]
    attack_time = attack_evaluation["attack_time"]
    attack_command = attack_evaluation["attack_command"]
    gate_time = attack_evaluation["gate_time"]
    global_gate_time = first_time(
        match.timeline,
        lambda snapshot: gate_complete_at(snapshot, spec),
    )
    requirements: list[RequirementResult] = []

    def add(
        category: str,
        name: str,
        score: float,
        passed: bool,
        evidence: str,
    ) -> None:
        requirements.append(
            RequirementResult(
                match_id=match.match_id,
                batch=batch_name(match.path),
                strategy=strategy,
                category=category,
                requirement=name,
                score=max(0.0, min(float(score), 1.0)),
                passed=bool(passed),
                evidence=evidence,
            )
        )

    # Economy: average development-target and base-rule groups.
    # Forbidden routes are omitted from scoring: models rarely pursue off-strategy
    # builds, and including near-always-1.0 constraints dilutes real targets.
    worker = spec.get("worker") or {}
    worker_unit = str(worker.get("unit") or "SCV")
    worker_max = maximum_completed(match.timeline, spec, worker_unit)
    economy_target_values: list[float] = []
    worker_target = number(worker.get("target"))
    if worker_target > 0:
        score = min(worker_max / worker_target, 1.0)
        economy_target_values.append(score)
        add(
            "economy",
            f"target_{worker_unit.lower()}",
            score,
            worker_max >= worker_target,
            f"max={worker_max}; target={worker_target}",
        )

    for entity, target_value in (spec.get("economy_targets") or {}).items():
        target = number(target_value)
        observed = maximum_completed(match.timeline, spec, str(entity))
        score = min(observed / target, 1.0) if target > 0 else 1.0
        economy_target_values.append(score)
        add(
            "economy",
            f"target_{str(entity).lower()}",
            score,
            observed >= target,
            f"max={observed}; target={target}",
        )

    base_values: list[float] = []
    for rule in spec.get("base_constraints") or []:
        score, evidence = score_base_rule(rule, match, spec, attack_time)
        base_values.append(score)
        add(
            "economy",
            str(rule.get("name") or rule.get("type") or "base_rule"),
            score,
            math.isclose(score, 1.0),
            evidence,
        )

    economy_completion = mean_available(
        statistics.fmean(economy_target_values) if economy_target_values else None,
        statistics.fmean(base_values) if base_values else None,
    )
    if economy_completion is None:
        raise ValueError("strategy spec has no economy requirements")

    # Technology: only requirements explicitly marked "before_attack" use the
    # first offensive command as their deadline. General strategy requirements
    # use their maximum completed state over the whole match.
    tech_snapshot = (
        snapshot_at_or_before(match.macro, attack_time)
        if attack_time is not None
        else (match.macro[-1] if match.macro else match.timeline[-1])
    )
    if tech_snapshot is None:
        tech_snapshot = match.timeline[-1]

    required_values: list[float] = []
    for entity, target_value in (
        spec.get("required_entities_before_attack") or {}
    ).items():
        target = number(target_value)
        observed = completed_count(tech_snapshot, spec, str(entity))
        score = min(observed / target, 1.0) if target > 0 else 1.0
        required_values.append(score)
        add(
            "technology",
            f"required_{str(entity).lower()}",
            score,
            observed >= target,
            (
                f"observed={observed}; target={target}; "
                f"milestone_time={number(tech_snapshot.get('game_time'))}"
            ),
        )

    for entity, target_value in (
        spec.get("required_entities_by_end") or {}
    ).items():
        target = number(target_value)
        observed = maximum_completed(match.timeline, spec, str(entity))
        score = min(observed / target, 1.0) if target > 0 else 1.0
        required_values.append(score)
        add(
            "technology",
            f"required_{str(entity).lower()}",
            score,
            observed >= target,
            f"max_completed={observed}; target={target}; deadline=match_end",
        )

    upgrade_values: list[float] = []
    milestone_upgrades = completed_upgrades(tech_snapshot)
    for upgrade in spec.get("required_upgrades_before_attack") or []:
        passed = upgrade_satisfied(upgrade, milestone_upgrades)
        upgrade_values.append(float(passed))
        add(
            "technology",
            f"upgrade_{normalize_upgrade_name(upgrade).lower()}",
            float(passed),
            passed,
            f"milestone_time={number(tech_snapshot.get('game_time'))}",
        )

    all_completed_upgrades = set().union(
        *(completed_upgrades(snapshot) for snapshot in match.timeline)
    )
    for upgrade in spec.get("required_upgrades_by_end") or []:
        passed = upgrade_satisfied(upgrade, all_completed_upgrades)
        upgrade_values.append(float(passed))
        add(
            "technology",
            f"upgrade_{normalize_upgrade_name(upgrade).lower()}",
            float(passed),
            passed,
            "deadline=match_end",
        )

    # Forbidden tech routes / upgrades are not scored for the same reason as
    # economy forbids: they are near-always satisfied and dilute required builds.
    technology_completion = mean_available(
        statistics.fmean(required_values) if required_values else None,
        statistics.fmean(upgrade_values) if upgrade_values else None,
    )
    if technology_completion is None:
        raise ValueError("strategy spec has no technology requirements")

    # Army: best simultaneous progress of all completed, living combat units.
    # This deliberately includes units temporarily reserved by PlanZoneDefense.
    army_completion = max(
        (
            gate_progress_at(snapshot, spec)
            for snapshot in match.timeline
        ),
        default=0.0,
    )
    add(
        "army",
        "first_attack_force",
        army_completion,
        global_gate_time is not None,
        (
            "source=all_completed_units; "
            f"strict_global_gate_time={global_gate_time}; "
            f"attack_readiness_gate_time={gate_time}"
        ),
    )

    # Engagement is N/A until the gathered main force either attacks or reaches
    # the strategy gate.  N/A values are omitted from both requirements and the
    # per-match overall rather than being counted as artificial successes.
    engagement_trigger_consistency = engagement_evaluation[
        "trigger_score"
    ]
    engagement_continuation_consistency = engagement_evaluation[
        "continuation_score"
    ]
    engagement_execution_consistency = engagement_evaluation["score"]
    raw_attack_delay = (
        attack_time - gate_time
        if attack_time is not None and gate_time is not None
        else None
    )
    trigger_passed = engagement_evaluation["trigger_status"] in {
        "on_first_opportunity",
        "within_army_grace",
    }
    if engagement_trigger_consistency is not None:
        add(
            "engagement",
            "engagement_trigger",
            engagement_trigger_consistency,
            trigger_passed,
            (
                f"gate_time={gate_time}; attack_time={attack_time}; "
                f"offset={attack_evaluation['attack_offset_seconds']}; "
                "first_opportunity_time="
                f"{attack_evaluation['first_opportunity_time']}; "
                "missed_commander_opportunities="
                f"{attack_evaluation['missed_army_opportunities']}; "
                "grace_opportunities="
                f"{attack_evaluation.get('trigger_army_grace_missed_opportunities', TRIGGER_ARMY_GRACE_MISSED_OPPORTUNITIES)}; "
                "attack_force_progress="
                f"{attack_evaluation['attack_group_progress']}; "
                "attack_readiness_source="
                f"{attack_evaluation['attack_readiness_source']}; "
                "objective_correct="
                f"{attack_evaluation.get('objective_correct')}; "
                f"status={engagement_evaluation['trigger_status']}"
            ),
        )
    if engagement_continuation_consistency is not None:
        add(
            "engagement",
            "engagement_continuation",
            engagement_continuation_consistency,
            math.isclose(engagement_continuation_consistency, 1.0),
            (
                "first_applied_offense_time="
                f"{continuation_evaluation['first_applied_offense_time']}; "
                "evaluated_opportunities="
                f"{continuation_evaluation['evaluated_opportunities']}; "
                "scope=all_post_attack_commander_decisions; "
                f"status={continuation_evaluation['status']}"
            ),
        )

    overall = mean_available(
        economy_completion,
        technology_completion,
        army_completion,
        engagement_trigger_consistency,
        engagement_continuation_consistency,
    )
    if overall is None:
        raise ValueError("strategy spec has no evaluable requirements")
    strict_pass_rate = statistics.fmean(
        float(requirement.passed) for requirement in requirements
    )

    metadata = match.metadata
    models = match_models(match.path, metadata)
    row = {
        "match_id": match.match_id,
        "record_path": str(match.path),
        "record_sha256": match.content_sha256,
        "batch": batch_name(match.path),
        "model": model_signature(models),
        **models,
        "strategy": strategy,
        "enemy_race": str(metadata.get("enemy_race") or ""),
        "matchup": str(metadata.get("matchup") or ""),
        "difficulty": parse_difficulty(str(metadata.get("opponent_id") or "")),
        "map_name": str(metadata.get("map_name") or ""),
        "result": str(metadata.get("result") or ""),
        "duration_s": match.end_time,
        "economy_completion": economy_completion,
        "technology_completion": technology_completion,
        "army_completion": army_completion,
        "engagement_trigger_consistency": engagement_trigger_consistency,
        # Backward-compatible alias for the previous submetric name.
        "engagement_initiation_consistency": (
            engagement_trigger_consistency
        ),
        "engagement_continuation_consistency": (
            engagement_continuation_consistency
        ),
        "engagement_trigger_evaluable": float(
            engagement_trigger_consistency is not None
        ),
        "engagement_continuation_evaluable": float(
            engagement_continuation_consistency is not None
        ),
        # Backward-compatible aliases. New analyses should use the two
        # engagement components above; paper metrics average those five
        # primary components (eco/tech/army/trigger/continuation).
        "engagement_execution_consistency": (
            engagement_execution_consistency
        ),
        "attack_completion": engagement_execution_consistency,
        "overall_strategy_compliance": overall,
        "strict_requirement_pass_rate": strict_pass_rate,
        "gate_reached": float(global_gate_time is not None),
        "gate_time_s": gate_time,
        "global_gate_time_s": global_gate_time,
        "attack_force_progress": attack_evaluation[
            "attack_group_progress"
        ],
        "attack_readiness_source": attack_evaluation[
            "attack_readiness_source"
        ],
        "commander_interval_s": attack_evaluation[
            "commander_interval_seconds"
        ],
        "attack_first_opportunity_time_s": attack_evaluation[
            "first_opportunity_time"
        ],
        "attack_missed_army_opportunities": attack_evaluation[
            "missed_army_opportunities"
        ],
        "attack_decision_half_life_opportunities": attack_evaluation[
            "decision_half_life_opportunities"
        ],
        "attack_evaluable": float(attack_evaluation["evaluable"]),
        "first_attack_objective_correct": attack_evaluation.get(
            "objective_correct"
        ),
        "attack_evaluation_status": attack_evaluation["status"],
        "engagement_trigger_status": engagement_evaluation[
            "trigger_status"
        ],
        "attack_time_s": attack_time,
        "raw_attack_delay_s": raw_attack_delay,
        "attack_mode": attack_command.get("movement_mode") if attack_command else "",
        "engagement_first_applied_offense_time_s": (
            continuation_evaluation["first_applied_offense_time"]
        ),
        "engagement_continuation_opportunities": (
            continuation_evaluation["evaluated_opportunities"]
        ),
        "engagement_continuation_status": continuation_evaluation[
            "status"
        ],
    }
    return row, requirements


PAPER_COMPONENT_METRICS = (
    "economy_completion",
    "technology_completion",
    "army_completion",
    "engagement_trigger_consistency",
    "engagement_continuation_consistency",
)

MAIN_METRICS = (
    *PAPER_COMPONENT_METRICS,
    "engagement_trigger_evaluable",
    "engagement_continuation_evaluable",
    "engagement_execution_consistency",
    "overall_strategy_compliance",
    "strict_requirement_pass_rate",
    "gate_reached",
    "duration_s",
)


def summarize_values(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def numeric_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for metric in MAIN_METRICS:
        values = [
            float(row[metric])
            for row in rows
            if isinstance(row.get(metric), (int, float))
            and math.isfinite(float(row[metric]))
        ]
        if values:
            summary[metric] = summarize_values(values)
    return summary


def grouped_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, str, str, str, str],
        list[dict[str, Any]],
    ] = {}
    for row in rows:
        key = (
            str(row.get("batch") or "(ungrouped)"),
            str(row.get("model") or "unknown"),
            str(row.get("strategy") or ""),
            str(row.get("enemy_race") or ""),
            str(row.get("difficulty") or ""),
            str(row.get("map_name") or ""),
        )
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (
        batch,
        model,
        strategy,
        enemy_race,
        difficulty,
        map_name,
    ), group_rows in sorted(
        groups.items(),
        key=lambda item: (
            item[0][1],
            item[0][2],
            difficulty_sort_key(item[0][4]),
            item[0][0],
            item[0][3],
            item[0][5],
        ),
    ):
        result: dict[str, Any] = {
            "batch": batch,
            "model": model,
            "strategy": strategy,
            "enemy_race": enemy_race,
            "difficulty": difficulty,
            "map_name": map_name,
            "matches": len(group_rows),
        }
        for metric in MAIN_METRICS:
            values = [
                float(row[metric])
                for row in group_rows
                if isinstance(row.get(metric), (int, float))
            ]
            if values:
                result[f"{metric}_n"] = len(values)
                result[f"{metric}_mean"] = statistics.fmean(values)
                result[f"{metric}_sd"] = (
                    statistics.stdev(values) if len(values) >= 2 else 0.0
                )
        result["attack_completion_n"] = result.get(
            "engagement_execution_consistency_n"
        )
        result["attack_completion_mean"] = result.get(
            "engagement_execution_consistency_mean"
        )
        result["attack_completion_sd"] = result.get(
            "engagement_execution_consistency_sd"
        )
        result["engagement_initiation_consistency_n"] = result.get(
            "engagement_trigger_consistency_n"
        )
        result["engagement_initiation_consistency_mean"] = result.get(
            "engagement_trigger_consistency_mean"
        )
        result["engagement_initiation_consistency_sd"] = result.get(
            "engagement_trigger_consistency_sd"
        )
        output.append(result)
    return output


def balanced_paper_metrics(
    cells: list[dict[str, Any]],
    model: str,
    strategy: str,
    difficulty: str,
    batch: Optional[str] = None,
) -> dict[str, Any]:
    """Average paper component means across cells.

    When cells carry ``{metric}_n``, use match-count weights so merging
    unequal batches stays proportional to games rather than batch folders.
    """
    result: dict[str, Any] = {
        "model": model,
        "strategy": strategy,
    }
    if batch is not None:
        result["batch"] = batch
    else:
        batch_names = sorted(
            {
                str(cell.get("batch") or "(ungrouped)")
                for cell in cells
                if cell.get("batch") is not None
            }
        )
        if batch_names:
            result["batch_count"] = len(batch_names)
    result["difficulty"] = difficulty
    for metric in PAPER_COMPONENT_METRICS:
        weighted: list[tuple[float, float]] = []
        for cell in cells:
            if cell.get(f"{metric}_mean") is None:
                continue
            mean = float(number(cell[f"{metric}_mean"]))
            raw_n = cell.get(f"{metric}_n")
            weight = float(number(raw_n)) if raw_n is not None else 1.0
            if weight <= 0:
                continue
            weighted.append((mean, weight))
        if not weighted:
            result[metric] = None
            result[f"{metric}_n"] = 0
            continue
        total_weight = sum(weight for _mean, weight in weighted)
        result[metric] = (
            sum(mean * weight for mean, weight in weighted) / total_weight
            if total_weight > 0
            else None
        )
        result[f"{metric}_n"] = int(total_weight)
    overall_weighted: list[tuple[float, float]] = []
    for cell in cells:
        value = cell.get("overall_strategy_compliance_mean")
        if value is None:
            continue
        weight = float(number(cell.get("overall_strategy_compliance_n") or 1))
        if weight > 0:
            overall_weighted.append((float(value), weight))
    total_overall_weight = sum(weight for _value, weight in overall_weighted)
    result["overall_strategy_compliance"] = (
        sum(value * weight for value, weight in overall_weighted)
        / total_overall_weight
        if total_overall_weight > 0
        else None
    )
    result["overall_strategy_compliance_n"] = int(total_overall_weight)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    extra = sorted({key for row in rows for key in row} - set(fields))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    rows: list[dict[str, Any]],
    summary: dict[str, dict[str, Any]],
) -> None:
    print(f"Matches: {len(rows)}")
    print(f"{'metric':<34} {'mean':>10} {'sd':>10} {'median':>10} {'min':>10} {'max':>10}")
    print("-" * 90)
    for metric in MAIN_METRICS:
        values = summary.get(metric)
        if not values:
            continue
        print(
            f"{metric:<34} "
            f"{values['mean']:>10.3f} "
            f"{values['sd']:>10.3f} "
            f"{values['median']:>10.3f} "
            f"{values['min']:>10.3f} "
            f"{values['max']:>10.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the SC2-LSEE strategy-execution metrics "
            "(five primary components plus overall)."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["game_records"],
        help=(
            "game_records root, batch_* directories, or match JSON files. "
            "When a root directory is given, only its batch_* children are read."
        ),
    )
    parser.add_argument(
        "--spec",
        help="Optional single-strategy specification JSON.",
    )
    parser.add_argument(
        "--spec-dir",
        default="tools/strategy_specs",
        help="Specification directory used for automatic multi-strategy mode.",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["mediumhard", "hard", "harder", "veryhard"],
        help=(
            "Difficulties to include. Defaults to MediumHard through VeryHard. "
            "Pass 'all' to disable difficulty filtering."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="analysis_results/strategy_execution",
        help="Output directory.",
    )
    parser.add_argument(
        "--expected-matches-per-batch",
        type=int,
        default=20,
        help=(
            "Deprecated informational field only. Batch size is no longer used "
            "to include or exclude records (kept for report compatibility)."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-batches",
        action="store_true",
        help=(
            "Deprecated no-op. Incomplete and oversized batches are always "
            "included when records are otherwise valid."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_paths = [Path(value).expanduser().resolve() for value in args.input]
    spec_dir = Path(args.spec_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    allowed_difficulties = {str(value).lower() for value in args.difficulties}
    include_all_difficulties = "all" in allowed_difficulties

    missing = [path for path in input_paths if not path.exists()]
    if missing:
        print(f"[ERROR] input not found: {missing[0]}")
        return 1
    if args.spec:
        spec_path = Path(args.spec).expanduser().resolve()
        if not spec_path.is_file():
            print(f"[ERROR] strategy spec not found: {spec_path}")
            return 1
        spec_paths = [spec_path]
        explicit_spec = True
    else:
        if not spec_dir.is_dir():
            print(f"[ERROR] strategy spec directory not found: {spec_dir}")
            return 1
        spec_paths = sorted(spec_dir.glob("*.json"))
        explicit_spec = False
    try:
        specs = load_specs(spec_paths)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] failed to load strategy specs: {exc}")
        return 1
    if not specs:
        print("[ERROR] no strategy specs found")
        return 1

    files = discover_record_files(input_paths)
    if not files:
        print("[ERROR] no match JSON files found")
        return 1

    game_rows: list[dict[str, Any]] = []
    requirement_rows: list[RequirementResult] = []
    errors: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    raw_batch_counts: dict[str, int] = {}
    for path in files:
        name = batch_name(path)
        raw_batch_counts[name] = raw_batch_counts.get(name, 0) + 1

    loaded_matches: list[Match] = []
    seen_hashes: dict[str, Path] = {}
    seen_ids: dict[str, tuple[str, Path]] = {}
    for path in files:
        try:
            match = load_match(path)
            prior_hash_path = seen_hashes.get(match.content_sha256)
            if prior_hash_path is not None:
                skipped.append(
                    {
                        "record_path": str(path),
                        "strategy": match.strategy_name,
                        "reason": f"duplicate content of {prior_hash_path}",
                    }
                )
                continue
            prior_id = seen_ids.get(match.match_id)
            if prior_id is not None and prior_id[0] != match.content_sha256:
                errors.append(
                    {
                        "record_path": str(path),
                        "error": (
                            "duplicate match_id with different content: "
                            f"{match.match_id}; first={prior_id[1]}"
                        ),
                    }
                )
                continue
            seen_hashes[match.content_sha256] = path
            seen_ids[match.match_id] = (match.content_sha256, path)
            loaded_matches.append(match)
        except Exception as exc:
            errors.append(
                {
                    "record_path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    batch_counts: dict[str, int] = {}
    for match in loaded_matches:
        name = batch_name(match.path)
        batch_counts[name] = batch_counts.get(name, 0) + 1
    all_batch_names = sorted(set(raw_batch_counts) | set(batch_counts))
    batch_rows = [
        {
            "batch": name,
            "record_files": raw_batch_counts.get(name, 0),
            "unique_records": batch_counts.get(name, 0),
            "expected_records": args.expected_matches_per_batch,
            "complete": float(batch_counts.get(name, 0) > 0),
            "included": float(batch_counts.get(name, 0) > 0),
        }
        for name in all_batch_names
    ]

    for match in loaded_matches:
        path = match.path
        try:
            name = batch_name(path)
            difficulty = parse_difficulty(
                str(match.metadata.get("opponent_id") or "")
            )
            if not include_all_difficulties and difficulty not in allowed_difficulties:
                skipped.append(
                    {
                        "record_path": str(path),
                        "strategy": match.strategy_name,
                        "reason": f"difficulty {difficulty} not selected",
                    }
                )
                continue
            if explicit_spec:
                spec = next(iter(specs.values()))
                expected = str(spec.get("strategy_name") or "")
                if match.strategy_name and match.strategy_name != expected:
                    skipped.append(
                        {
                            "record_path": str(path),
                            "strategy": match.strategy_name,
                            "reason": f"does not match explicit spec {expected}",
                        }
                    )
                    continue
            else:
                spec = select_spec(specs, match)
                if spec is None:
                    skipped.append(
                        {
                            "record_path": str(path),
                            "strategy": match.strategy_name,
                            "reason": (
                                "no strategy spec named "
                                f"{match.strategy_name or '(missing)'}"
                            ),
                        }
                    )
                    continue
            row, requirements = evaluate_match(match, spec)
            game_rows.append(row)
            requirement_rows.extend(requirements)
        except Exception as exc:  # Keep other matches usable and record the error.
            errors.append(
                {
                    "record_path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if not game_rows:
        print("[ERROR] all match records failed")
        for error in errors:
            print(f"  {error['record_path']}: {error['error']}")
        return 1

    summary = numeric_summary(game_rows)
    cells = grouped_metrics(game_rows)
    batch_groups = sorted(
        {
            (
                str(cell.get("batch") or "(ungrouped)"),
                str(cell.get("model") or "unknown"),
                str(cell.get("strategy") or ""),
                str(cell.get("difficulty") or "unknown"),
            )
            for cell in cells
        },
        key=lambda item: (
            item[1],
            item[2],
            difficulty_sort_key(item[3]),
            item[0],
        ),
    )
    difficulty_groups = sorted(
        {
            (
                str(cell.get("model") or "unknown"),
                str(cell.get("strategy") or ""),
                str(cell.get("difficulty") or "unknown"),
            )
            for cell in cells
        },
        key=lambda item: (
            item[0],
            item[1],
            difficulty_sort_key(item[2]),
        ),
    )
    model_strategies = sorted(
        {
            (
                str(row.get("model") or "unknown"),
                str(row.get("strategy") or ""),
            )
            for row in game_rows
        }
    )
    paper_batch_rows = []
    paper_rows = []
    paper_overall_rows = []
    for batch, model, strategy, difficulty in batch_groups:
        batch_cells = [
            cell
            for cell in cells
            if str(cell.get("batch") or "(ungrouped)") == batch
            and cell.get("model") == model
            and cell.get("strategy") == strategy
            and str(cell.get("difficulty") or "unknown") == difficulty
        ]
        paper_batch_rows.append(
            balanced_paper_metrics(
                batch_cells,
                model,
                strategy,
                difficulty,
                batch,
            )
        )

    for model, strategy, difficulty in difficulty_groups:
        merged_cells = [
            cell
            for cell in cells
            if cell.get("model") == model
            and cell.get("strategy") == strategy
            and str(cell.get("difficulty") or "unknown") == difficulty
        ]
        paper_rows.append(
            balanced_paper_metrics(
                merged_cells,
                model,
                strategy,
                difficulty,
            )
        )

    for model, strategy in model_strategies:
        strategy_cells = [
            cell
            for cell in cells
            if cell.get("model") == model
            and cell.get("strategy") == strategy
        ]
        paper_overall_rows.append(
            balanced_paper_metrics(
                strategy_cells,
                model,
                strategy,
                "all",
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_game.csv", game_rows)
    write_csv(
        out_dir / "per_requirement.csv",
        [asdict(result) for result in requirement_rows],
    )
    write_csv(out_dir / "grouped_metrics.csv", cells)
    # Default paper table merges batches that share model/strategy/difficulty.
    write_csv(out_dir / "paper_metrics.csv", paper_rows)
    write_csv(out_dir / "paper_metrics_by_batch.csv", paper_batch_rows)
    write_csv(out_dir / "paper_metrics_overall.csv", paper_overall_rows)
    paper_by_model_dir = out_dir / "paper_metrics_by_model"
    paper_batch_by_model_dir = out_dir / "paper_metrics_by_batch_by_model"
    paper_overall_by_model_dir = out_dir / "paper_metrics_overall_by_model"
    for model in sorted({str(row["model"]) for row in paper_rows}):
        model_rows = [row for row in paper_rows if row["model"] == model]
        model_batch_rows = [
            row for row in paper_batch_rows if row["model"] == model
        ]
        model_overall_rows = [
            row for row in paper_overall_rows if row["model"] == model
        ]
        slug = model_file_slug(model)
        paper_by_model_dir.mkdir(parents=True, exist_ok=True)
        write_csv(paper_by_model_dir / f"{slug}.csv", model_rows)
        if model_batch_rows:
            paper_batch_by_model_dir.mkdir(parents=True, exist_ok=True)
            write_csv(
                paper_batch_by_model_dir / f"{slug}.csv",
                model_batch_rows,
            )
        if model_overall_rows:
            paper_overall_by_model_dir.mkdir(parents=True, exist_ok=True)
            write_csv(
                paper_overall_by_model_dir / f"{slug}.csv",
                model_overall_rows,
            )
    write_csv(out_dir / "batch_summary.csv", batch_rows)
    write_csv(
        out_dir / "summary.csv",
        [{"metric": metric, **values} for metric, values in summary.items()],
    )
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "inputs": [str(path) for path in input_paths],
                "spec_paths": [str(path) for path in spec_paths],
                "expected_matches_per_batch": args.expected_matches_per_batch,
                "allow_incomplete_batches": args.allow_incomplete_batches,
                "engagement_execution_metric": {
                    "display_name_zh": "进攻执行一致性",
                    "combination": (
                        "mean of available trigger and continuation scores; "
                        "both are N/A before an engagement opportunity exists"
                    ),
                    "valid_modes": sorted(DEFAULT_ATTACK_MODES),
                    "trigger_roles": sorted(DEFAULT_ATTACK_ROLES),
                    "trigger_force": "gathered main_force only",
                    "continuation_roles": (
                        "main_force plus reinforcement groups joining the "
                        "main force or active objective"
                    ),
                    "requires_commander_decision_accepted": True,
                    "understrength_score": (
                        "required-supply-weighted readiness of the gathered "
                        "main force; distant group_1 reinforcements do not "
                        "complete the trigger gate"
                    ),
                    "army_completion_source": (
                        "maximum required-supply-weighted readiness over all "
                        "completed living units"
                    ),
                    "first_executable_opportunity": (
                        "first logical Commander decision at or after the "
                        "gathered main force reaches gate readiness"
                    ),
                    "late_decay": (
                        "score=1 if missed_army_opportunities <= "
                        f"{TRIGGER_ARMY_GRACE_MISSED_OPPORTUNITIES}; "
                        "else 0.5 ** ((missed - grace) / "
                        f"{TRIGGER_ARMY_HALF_LIFE_MISSED_OPPORTUNITIES})"
                    ),
                    "missed_army_opportunities": (
                        "logical Commander decisions after the first gate-ready "
                        "decision and before the first issued attack"
                    ),
                    "commander_retry_deduplication": (
                        "collapse consecutive fast failed retries into one "
                        "logical decision and retain the first successful "
                        "result or the final failed attempt"
                    ),
                    "no_attack_before_gate": (
                        "trigger=N/A, continuation=N/A"
                    ),
                    "no_attack_after_gate": (
                        "trigger=0, continuation=0"
                    ),
                    "continuation": (
                        "mean over all logical Commander decisions after the "
                        "first applied offense; valid phases are continued or "
                        "renewed offense, cleanup, safe recovery/rebuild, and "
                        "reinforcement joining the main force or objective"
                    ),
                    "continuation_uses_applied_commands": True,
                    "runtime_auto_retreat": (
                        "excluded from the denominator because it is executor "
                        "control rather than a Commander-authored decision"
                    ),
                },
                "batch_summary": batch_rows,
                "specs": list(specs.values()),
                "paper_metrics": paper_rows,
                "paper_metrics_by_batch": paper_batch_rows,
                "paper_metrics_overall": paper_overall_rows,
                "grouped_metrics": cells,
                "matches": game_rows,
                "requirements": [asdict(result) for result in requirement_rows],
                "summary": summary,
                "skipped": skipped,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print_summary(game_rows, summary)
    print()
    print(
        f"Discovered: {len(files)}  Evaluated: {len(game_rows)}  "
        f"Skipped: {len(skipped)}  Errors: {len(errors)}"
    )
    print(f"Wrote: {out_dir}")
    if errors:
        print(f"Warnings: {len(errors)} records failed; see report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
