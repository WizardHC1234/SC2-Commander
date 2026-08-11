"""Typed, single-source action specifications shared by race adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Set, Tuple


ActionFactory = Callable[..., Any]


@dataclass(frozen=True)
class ActionSpec:
    """One model-facing tool and, for macro tools, its runtime implementation."""

    description: str
    action_type: str
    action_func: Optional[ActionFactory] = None
    target_semantics: str = ""
    cost_kind: str = ""
    minerals: int = 0
    vespene: int = 0
    supply: float = 0
    base_time_seconds: float = 0
    production_role: str = ""
    production_location: str = ""
    prerequisites: Tuple[str, ...] = ()
    dependencies: Tuple[str, ...] = ()

    @property
    def is_macro(self) -> bool:
        return self.action_type not in {"army", "meta"}

    @property
    def consumes_gas(self) -> bool:
        return self.is_macro and self.vespene > 0


def macro_action(
    description: str,
    action_type: str,
    action_func: ActionFactory,
    minerals: int,
    vespene: int,
    seconds: float,
    *,
    supply: float = 0,
    target_semantics: str = "absolute_count",
    cost_kind: str = "cost_each",
    production_role: str,
    production_location: str,
    prerequisites: Tuple[str, ...] = (),
    dependencies: Tuple[str, ...] = (),
) -> ActionSpec:
    return ActionSpec(
        description=description,
        action_type=action_type,
        action_func=action_func,
        target_semantics=target_semantics,
        cost_kind=cost_kind,
        minerals=minerals,
        vespene=vespene,
        supply=supply,
        base_time_seconds=seconds,
        production_role=production_role,
        production_location=production_location,
        prerequisites=prerequisites,
        dependencies=dependencies,
    )


def research_action(
    description: str,
    action_func: ActionFactory,
    minerals: int,
    vespene: int,
    seconds: float,
    location: str,
    *,
    prerequisites: Tuple[str, ...] = (),
    dependencies: Tuple[str, ...] = (),
) -> ActionSpec:
    return macro_action(
        description,
        "tech",
        action_func,
        minerals,
        vespene,
        seconds,
        target_semantics="research_once",
        cost_kind="cost",
        production_role="researched_at",
        production_location=location,
        prerequisites=prerequisites,
        dependencies=dependencies,
    )


def control_action(description: str, action_type: str) -> ActionSpec:
    return ActionSpec(description=description, action_type=action_type)


def validate_action_specs(specs: Mapping[str, ActionSpec]) -> None:
    """Fail at import time when a race catalog is internally inconsistent."""
    known = set(specs)
    for action_name, spec in specs.items():
        if spec.action_type not in {"unit", "building", "tech", "army", "meta"}:
            raise ValueError(f"{action_name}: unsupported action_type={spec.action_type!r}")
        if spec.is_macro:
            if spec.action_func is None:
                raise ValueError(f"{action_name}: macro action has no action_func")
            required = {
                "target_semantics": spec.target_semantics,
                "cost_kind": spec.cost_kind,
                "production_role": spec.production_role,
                "production_location": spec.production_location,
            }
            missing = [field for field, value in required.items() if not value]
            if missing:
                raise ValueError(f"{action_name}: missing fields {missing}")
            if spec.base_time_seconds <= 0:
                raise ValueError(f"{action_name}: base_time_seconds must be positive")
        elif spec.action_func is not None:
            raise ValueError(f"{action_name}: control action cannot have action_func")
        unknown_dependencies = set(spec.dependencies) - known
        if unknown_dependencies:
            raise ValueError(
                f"{action_name}: unknown dependencies {sorted(unknown_dependencies)}"
            )


def direct_action_dependencies(
    specs: Mapping[str, ActionSpec],
    action_name: str,
    *,
    gas_action_name: Optional[str] = None,
) -> Set[str]:
    spec = specs.get(action_name)
    if spec is None:
        return set()
    dependencies = set(spec.dependencies)
    if (
        gas_action_name
        and gas_action_name in specs
        and action_name != gas_action_name
        and spec.consumes_gas
    ):
        dependencies.add(gas_action_name)
    return dependencies


def expand_action_dependencies(
    specs: Mapping[str, ActionSpec],
    action_names: Iterable[str],
    *,
    known_action_names: Optional[Iterable[str]] = None,
    gas_action_name: Optional[str] = None,
) -> Set[str]:
    """Compute structural and resource-source dependency closure."""
    known = set(known_action_names) if known_action_names is not None else set(specs)
    selected = {name for name in action_names if name in known}
    pending = list(selected)
    while pending:
        action_name = pending.pop()
        for dependency in direct_action_dependencies(
            specs,
            action_name,
            gas_action_name=gas_action_name,
        ):
            if dependency not in known or dependency in selected:
                continue
            selected.add(dependency)
            pending.append(dependency)
    return selected


def _format_number(value: float) -> str:
    number = float(value or 0)
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}".rstrip("0").rstrip(".")


def render_action_space(specs: Mapping[str, ActionSpec]) -> Dict[str, str]:
    """Render deterministic model-facing descriptions from action specs."""
    rendered: Dict[str, str] = {}
    for action_name, spec in specs.items():
        description = spec.description.strip()
        if not spec.is_macro:
            rendered[action_name] = description
            continue

        cost_parts = [
            f"{_format_number(spec.minerals)}M",
            f"{_format_number(spec.vespene)}G",
        ]
        if spec.supply:
            cost_parts.append(f"{_format_number(spec.supply)}S")
        fields = [
            f"target={spec.target_semantics}",
            f"{spec.cost_kind}={'/'.join(cost_parts)}",
            f"base_time={spec.base_time_seconds:.1f}s",
            f"{spec.production_role}={spec.production_location}",
        ]
        if spec.prerequisites:
            fields.append(f"prerequisites={'+'.join(spec.prerequisites)}")
        rendered[action_name] = f"{description.rstrip('.')}. {'; '.join(fields)}."
    return rendered


__all__ = [
    "ActionSpec",
    "control_action",
    "expand_action_dependencies",
    "macro_action",
    "render_action_space",
    "research_action",
    "validate_action_specs",
]
