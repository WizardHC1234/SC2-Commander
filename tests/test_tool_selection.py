"""Tests for semantic strategy tool selection and dependency expansion."""

from __future__ import annotations

from pathlib import Path

import commander.tool_selection as tool_selection
from commander.tool_selection import (
    build_tool_selection_messages,
    merge_selected_tools,
    select_tools_for_strategy,
)
from commander.races.terran.actions import expand_action_dependencies, get_action_space

_ROOT = Path(__file__).resolve().parents[1]
_SKILLS = _ROOT / "skills" / "terran"


def _strategy(name: str) -> str:
    return (_SKILLS / name / "strategy.md").read_text(encoding="utf-8")


def _selection_result(names):
    import json

    return {
        "content": json.dumps({"select": names}),
        "error": "",
        "latency_seconds": 0.1,
    }


def test_strategy_documents_need_no_resource_or_tool_section():
    for name in ("marine", "tank", "battlecruiser"):
        text = _strategy(name)
        assert "# Resource Costs" not in text
        assert "# Required Tools" not in text


def test_battlecruiser_dependency_closure_is_transitive():
    full = get_action_space()
    expanded = expand_action_dependencies(
        ["train_battlecruiser", "research_yamato_cannon"],
        known_action_names=full,
    )
    assert "build_starport" in expanded
    assert "build_starport_techlab" in expanded
    assert "build_fusion_core" in expanded
    assert "build_factory" in expanded
    assert "build_barracks" in expanded
    assert "build_supply_depot" in expanded
    assert "train_scv" in expanded


def test_level_upgrade_dependency_closure_includes_previous_level():
    full = get_action_space()
    expanded = expand_action_dependencies(
        ["research_infantry_weapons_3"], known_action_names=full
    )
    assert "research_infantry_weapons_2" in expanded
    assert "research_infantry_weapons_1" in expanded
    assert "build_engineering_bay" in expanded
    assert "build_armory" in expanded


def test_semantic_selection_expands_tank_prerequisites(monkeypatch):
    full = get_action_space()
    semantic = [
        "train_marine",
        "train_siege_tank",
        "build_gas",
        "build_barracks_reactor",
        "research_shieldwall",
    ]
    monkeypatch.setattr(
        tool_selection,
        "call_openai_detailed",
        lambda *args, **kwargs: _selection_result(semantic),
    )
    out = select_tools_for_strategy(
        strategy_text=_strategy("tank"),
        full_action_space=full,
        model_key="test-model",
        dependency_resolver=expand_action_dependencies,
    )
    selected = set(out["selected_tools"])
    assert out["fallback_used"] is False
    assert out["semantic_tools"] == semantic
    assert "build_factory" in selected
    assert "build_factory_techlab" in selected
    assert "build_barracks" in selected
    assert "build_barracks_techlab" in selected
    assert "research_yamato_cannon" not in selected
    assert len(selected) < len(full)


def test_marine_selection_stays_compact(monkeypatch):
    full = get_action_space()
    monkeypatch.setattr(
        tool_selection,
        "call_openai_detailed",
        lambda *args, **kwargs: _selection_result(["train_marine"]),
    )
    out = select_tools_for_strategy(
        strategy_text=_strategy("marine"),
        full_action_space=full,
        model_key="test-model",
        dependency_resolver=expand_action_dependencies,
    )
    selected = set(out["selected_tools"])
    assert "train_marine" in selected
    assert "build_barracks" in selected
    assert "build_gas" not in selected
    assert "build_factory" not in selected
    assert "research_yamato_cannon" not in selected


def test_gas_is_available_even_when_semantic_selector_omits_it(monkeypatch):
    full = get_action_space()
    # Dependency expansion must use ActionSpec.vespene, not rendered catalog text.
    full["train_battlecruiser"] = "Deliberately stripped model-facing description."
    monkeypatch.setattr(
        tool_selection,
        "call_openai_detailed",
        lambda *args, **kwargs: _selection_result(["train_battlecruiser"]),
    )
    out = select_tools_for_strategy(
        strategy_text=_strategy("battlecruiser"),
        full_action_space=full,
        model_key="test-model",
        dependency_resolver=expand_action_dependencies,
    )
    assert "build_gas" not in out["semantic_tools"]
    assert "build_gas" in out["dependency_tools"]
    assert "build_gas" in out["action_space"]


def test_missing_dependency_resolver_falls_back_to_full_catalog(monkeypatch):
    full = get_action_space()
    monkeypatch.setattr(
        tool_selection,
        "call_openai_detailed",
        lambda *args, **kwargs: _selection_result(["train_marine"]),
    )
    out = select_tools_for_strategy(
        strategy_text=_strategy("marine"),
        full_action_space=full,
        model_key="test-model",
    )
    assert out["fallback_reason"] == "dependency_resolver_error"
    assert out["dependency_error"] == "missing dependency resolver"
    assert out["action_space"] == full


def test_selection_prompt_uses_whole_strategy_without_required_section():
    messages = build_tool_selection_messages(
        strategy_text=_strategy("tank"), full_action_space=get_action_space()
    )
    combined = "\n".join(message["content"] for message in messages)
    assert "[Strategy]" in combined
    assert 'Return JSON {"select":[...]}' in combined
    assert "catalog descriptions are authoritative" in combined
    assert "REQUIRED tools" not in combined
    assert "strategy Resource Costs" not in combined


def test_no_llm_falls_back_to_full_catalog():
    full = get_action_space()
    out = select_tools_for_strategy(
        strategy_text=_strategy("battlecruiser"),
        full_action_space=full,
        model_key="",
        use_llm=False,
        dependency_resolver=expand_action_dependencies,
    )
    assert out["fallback_used"] is True
    assert out["fallback_reason"] == "llm_disabled"
    assert out["action_space"] == full


def test_invalid_semantic_selection_falls_back_to_full_catalog(monkeypatch):
    full = get_action_space()
    monkeypatch.setattr(
        tool_selection,
        "call_openai_detailed",
        lambda *args, **kwargs: _selection_result(["not_a_real_tool"]),
    )
    out = select_tools_for_strategy(
        strategy_text=_strategy("tank"),
        full_action_space=full,
        model_key="test-model",
        dependency_resolver=expand_action_dependencies,
    )
    assert out["fallback_reason"] == "empty_or_invalid_selection"
    assert out["action_space"] == full


def test_dependency_failure_falls_back_to_full_catalog(monkeypatch):
    full = get_action_space()
    monkeypatch.setattr(
        tool_selection,
        "call_openai_detailed",
        lambda *args, **kwargs: _selection_result(["train_marine"]),
    )

    def broken_resolver(*args, **kwargs):
        raise RuntimeError("broken metadata")

    out = select_tools_for_strategy(
        strategy_text=_strategy("marine"),
        full_action_space=full,
        model_key="test-model",
        dependency_resolver=broken_resolver,
    )
    assert out["fallback_reason"] == "dependency_resolver_error"
    assert "broken metadata" in out["dependency_error"]
    assert out["action_space"] == full


def test_merge_drops_unknown_and_keeps_army_tools():
    full = get_action_space()
    space = merge_selected_tools(
        full_action_space=full,
        selected_tools=["train_marine", "not_a_real_tool"],
    )
    assert "train_marine" in space
    assert "army_intent" in space
    assert "move_group" not in space
    assert "set_wake_event" in space
    assert "not_a_real_tool" not in space
