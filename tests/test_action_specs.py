"""Tests for unified Terran action specs and rendered descriptions."""

from __future__ import annotations

from commander.races.terran.actions import (
    ACTION_SPECS,
    expand_action_dependencies,
    get_action_space,
)


def test_single_action_registry_contains_complete_macro_specs():
    macro_specs = {name: spec for name, spec in ACTION_SPECS.items() if spec.is_macro}
    assert len(ACTION_SPECS) == 74
    assert len(macro_specs) == 70
    for action_name, spec in macro_specs.items():
        assert spec.action_func is not None, action_name
        assert spec.target_semantics, action_name
        assert spec.cost_kind, action_name
        assert spec.base_time_seconds > 0, action_name
        assert spec.production_role, action_name
        assert spec.production_location, action_name


def test_every_macro_description_contains_static_cost_and_time():
    action_space = get_action_space()
    for action_name, spec in ACTION_SPECS.items():
        if not spec.is_macro:
            continue
        description = action_space[action_name]
        assert "target=" in description, action_name
        assert "base_time=" in description, action_name
        assert any(
            marker in description
            for marker in ("cost_each=", "cost=", "incremental_cost=")
        ), action_name


def test_positive_vespene_field_adds_gas_dependency_without_description_parsing():
    for action_name, spec in ACTION_SPECS.items():
        if not spec.consumes_gas:
            continue
        expanded = expand_action_dependencies(
            [action_name], known_action_names=ACTION_SPECS
        )
        assert "build_gas" in expanded, action_name

    marine_dependencies = expand_action_dependencies(
        ["train_marine"], known_action_names=ACTION_SPECS
    )
    assert "build_gas" not in marine_dependencies


def test_siege_tank_description_is_fully_static():
    description = get_action_space()["train_siege_tank"]
    assert "target=absolute_count" in description
    assert "cost_each=150M/125G/3S" in description
    assert "base_time=32.1s" in description
    assert "producer=Factory" in description
    assert "prerequisites=Factory Tech Lab" in description


def test_fusion_core_description_is_fully_static():
    description = get_action_space()["build_fusion_core"]
    assert "cost_each=150M/150G" in description
    assert "base_time=46.4s" in description
    assert "builder=SCV" in description
    assert "prerequisites=Starport" in description


def test_morph_uses_incremental_cost():
    description = get_action_space()["morph_orbital_command"]
    assert "incremental_cost=150M/0G" in description
    assert "producer=Command Center" in description
    assert "prerequisites=Barracks" in description


def test_upgrade_description_contains_research_chain():
    description = get_action_space()["research_infantry_weapons_2"]
    assert "target=research_once" in description
    assert "cost=150M/150G" in description
    assert "base_time=135.7s" in description
    assert "researched_at=Engineering Bay" in description
    assert "prerequisites=Armory+Terran Infantry Weapons Level 1" in description


def test_static_overrides_cover_previously_missing_entries():
    action_space = get_action_space()
    assert "cost=150M/150G; base_time=100.0s" in action_space[
        "research_yamato_cannon"
    ]
    assert "cost=100M/100G; base_time=100.0s" in action_space[
        "research_magfield_accelerator"
    ]
    assert "cost_each=100M/100G; base_time=43.0s" in action_space["build_nuke"]


def test_special_addon_requirement_is_explicit():
    description = get_action_space()["build_factory_techlab"]
    assert "producer=Factory" in description
    assert "prerequisites=completed Factory with free addon slot" in description
