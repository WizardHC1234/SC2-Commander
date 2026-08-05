from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


ALLOWED_MOVEMENT_MODES = {
    "regroup",
    "push",
    "assault",
    "harass",
    "defensive_retreat",
    "panic_retreat",
    "search_and_destroy",
}
COMMAND_FIELDS = {
    "group_id",
    "destination_zone_id",
    "movement_mode",
}
MOVE_TYPE_BY_MOVEMENT_MODE = {
    "regroup": "ReGroup",
    "push": "Push",
    "assault": "Assault",
    "harass": "Harass",
    "defensive_retreat": "DefensiveRetreat",
    "panic_retreat": "PanicRetreat",
    "search_and_destroy": "SearchAndDestroy",
}


@dataclass(frozen=True)
class ArmyGroupCommand:
    group_id: str
    destination_zone_id: str
    movement_mode: str
    move_type: str


@dataclass(frozen=True)
class ArmyControlPolicy:
    commands: List[ArmyGroupCommand] = field(default_factory=list)
    scan_zone_id: Optional[str] = None
    scout_zone_id: Optional[str] = None


def parse_army_control_policy(data: Dict[str, Any]) -> ArmyControlPolicy:
    if not isinstance(data, dict):
        raise ValueError("army control policy must be a dict")
    allowed_fields = {"commands", "scan_zone_id", "scout_zone_id"}
    if not set(data).issubset(allowed_fields) or "commands" not in data:
        raise ValueError(
            "army control policy must contain commands and optional scan_zone_id/scout_zone_id"
        )

    commands_data = data["commands"]
    if not isinstance(commands_data, list):
        raise ValueError("commands must be a list")
    if len(commands_data) > 3:
        raise ValueError("commands may contain at most three group commands")

    commands = []
    seen_groups = set()
    for command_data in commands_data:
        command = _parse_group_command(command_data)
        if command.group_id in seen_groups:
            raise ValueError(f"duplicate group_id {command.group_id!r}")
        seen_groups.add(command.group_id)
        commands.append(command)
    return ArmyControlPolicy(
        commands,
        _normalize_optional_zone_id(data.get("scan_zone_id"), "scan_zone_id"),
        _normalize_optional_zone_id(data.get("scout_zone_id"), "scout_zone_id"),
    )


def _parse_group_command(data: Any) -> ArmyGroupCommand:
    if not isinstance(data, dict):
        raise ValueError("each group command must be an object")
    missing = COMMAND_FIELDS - set(data)
    extra = set(data) - COMMAND_FIELDS
    if missing:
        raise ValueError(f"missing group command fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"unexpected group command fields: {sorted(extra)}")

    movement_mode = _require_allowed(
        data, "movement_mode", ALLOWED_MOVEMENT_MODES
    )
    return ArmyGroupCommand(
        group_id=_require_group_id(data),
        destination_zone_id=_require_zone_id(data),
        movement_mode=movement_mode,
        move_type=MOVE_TYPE_BY_MOVEMENT_MODE[movement_mode],
    )


def _require_group_id(data: Dict[str, Any]) -> str:
    value = data["group_id"]
    if not isinstance(value, str):
        raise ValueError("group_id must be a string")
    normalized = value.strip()
    if not normalized.startswith("group_") or not normalized[6:].isdigit():
        raise ValueError("group_id must have the form group_<index>")
    return normalized


def _require_zone_id(data: Dict[str, Any]) -> str:
    value = data["destination_zone_id"]
    if not isinstance(value, str):
        raise ValueError("destination_zone_id must be a string")
    normalized = value.strip()
    if not normalized.startswith("zone_") or not normalized[5:].isdigit():
        raise ValueError("destination_zone_id must have the form zone_<index>")
    return normalized


def _normalize_optional_zone_id(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    normalized = value.strip()
    if not normalized.startswith("zone_") or not normalized[5:].isdigit():
        raise ValueError(
            f"{field_name} must have the form zone_<index> or be null"
        )
    return normalized


def _require_allowed(data: Dict[str, Any], key: str, allowed: Set[str]) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    normalized = value.strip()
    if normalized not in allowed:
        raise ValueError(f"{key} must be one of {sorted(allowed)}, got {value!r}")
    return normalized


class InjectedArmyPolicyProvider:
    """Reads the latest Commander-applied policy from ``ai.commander_army_policy``."""

    def get_policy(self, act) -> ArmyControlPolicy:
        policy = getattr(getattr(act, "ai", None), "commander_army_policy", None)
        if isinstance(policy, ArmyControlPolicy):
            return policy
        return ArmyControlPolicy(
            commands=[],
            scan_zone_id=None,
            scout_zone_id=None,
        )
