"""Tests for automated strategy tool selection."""

from __future__ import annotations

from pathlib import Path

from commander.tool_selection import (
    extract_resource_cost_labels,
    merge_selected_tools,
    required_tools_from_strategy,
    select_tools_for_strategy,
    tools_from_resource_cost_labels,
)
from skills.terran.Action import get_action_space

_ROOT = Path(__file__).resolve().parents[1]
_SKILLS = _ROOT / "skills" / "terran"


def _strategy(name: str) -> str:
    return (_SKILLS / name / "strategy.md").read_text(encoding="utf-8")


def test_extract_battlecruiser_cost_labels():
    labels = extract_resource_cost_labels(_strategy("battlecruiser"))
    assert "Battlecruiser" in labels
    assert "Yamato Cannon" in labels
    assert "Fusion Core" in labels


def test_required_includes_yamato_for_battlecruiser():
    full = get_action_space()
    seed = required_tools_from_strategy(_strategy("battlecruiser"), full_action_space=full)
    required = set(seed["required_tools"])
    assert "research_yamato_cannon" in required
    assert "train_battlecruiser" in required
    assert "build_fusion_core" in required
    assert "build_factory_techlab" in required
    assert "set_wake_event" in required
    assert "move_group" in required
    assert "research_stimpack" not in required
    assert seed["unmatched_labels"] == []


def test_required_tank_has_combat_shield_not_yamato():
    full = get_action_space()
    seed = required_tools_from_strategy(_strategy("tank"), full_action_space=full)
    required = set(seed["required_tools"])
    assert "research_shieldwall" in required
    assert "train_siege_tank" in required
    assert "build_barracks_reactor" in required
    assert "research_yamato_cannon" not in required
    assert "train_battlecruiser" not in required


def test_required_marine_is_minimal():
    full = get_action_space()
    seed = required_tools_from_strategy(_strategy("marine"), full_action_space=full)
    required = set(seed["required_tools"])
    assert "train_marine" in required
    assert "build_barracks" in required
    assert "research_yamato_cannon" not in required
    assert "build_factory" not in required
    # Full catalog is much larger than the marine exposure surface.
    assert len(required) < len(full) // 2


def test_merge_keeps_required_and_army():
    full = get_action_space()
    space = merge_selected_tools(
        full_action_space=full,
        required_tools=["train_marine", "not_a_real_tool"],
        added_tools=["build_barracks", "also_fake"],
    )
    assert "train_marine" in space
    assert "build_barracks" in space
    assert "move_group" in space
    assert "set_wake_event" in space
    assert "not_a_real_tool" not in space


def test_select_without_llm_uses_required_only():
    full = get_action_space()
    out = select_tools_for_strategy(
        strategy_text=_strategy("battlecruiser"),
        full_action_space=full,
        model_key="",
        use_llm=False,
    )
    assert "research_yamato_cannon" in out["action_space"]
    assert out["added_tools"] == []
    assert out["selected_tool_count"] < out["full_tool_count"]
    assert out["selected_tool_count"] == len(out["required_tools"])


def test_unknown_label_reported():
    full = get_action_space()
    mapped, unmatched = tools_from_resource_cost_labels(
        ["Yamato Cannon", "Totally Fake Unit"],
        full_action_space=full,
    )
    assert "research_yamato_cannon" in mapped
    assert "Totally Fake Unit" in unmatched
