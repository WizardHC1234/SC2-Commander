from __future__ import annotations

from evol_agent.core.terran_build_order_simulator import (
    GAS_RATE_PER_WORKER,
    INITIAL_SUPPLY_CAP,
    INITIAL_SUPPLY_USED,
    INITIAL_WORKERS,
    SimState,
    TerranBuildOrderSimulator,
    simulate_terran_first_commitment,
)


def _marine_package(*, workers: int = 20) -> dict:
    return {
        "economy": {
            "worker_target_before_commitment": workers,
            "base_target_before_commitment": 1,
            "gas_workers_before_commitment": 0,
        },
        "gate_components": [
            {"action": "train_marine", "quantity": 20, "production_slots": 6}
        ],
        "setup_actions": [
            {"action": "build_barracks", "quantity": 6, "parallel_slots": 3}
        ],
    }


def _marine_support_package(*, gas_workers: int) -> dict:
    return {
        "economy": {
            "worker_target_before_commitment": 20,
            "base_target_before_commitment": 1,
            "gas_workers_before_commitment": gas_workers,
        },
        "gate_components": [
            {"action": "train_marine", "quantity": 20, "production_slots": 4},
            {"action": "train_marauder", "quantity": 2, "production_slots": 1},
            {"action": "research_stimpack", "quantity": 1, "production_slots": 1},
        ],
        "setup_actions": [
            {"action": "build_barracks", "quantity": 6, "parallel_slots": 3},
            {
                "action": "build_barracks_techlab",
                "quantity": 1,
                "parallel_slots": 1,
            },
            {"action": "build_gas", "quantity": 2, "parallel_slots": 2},
        ],
    }


def _tank_package() -> dict:
    return {
        "economy": {
            "worker_target_before_commitment": 44,
            "base_target_before_commitment": 2,
            "gas_workers_before_commitment": 12,
        },
        "gate_components": [
            {"action": "train_marine", "quantity": 45, "production_slots": 5},
            {"action": "train_siege_tank", "quantity": 10, "production_slots": 2},
            {"action": "research_shieldwall", "quantity": 1, "production_slots": 1},
        ],
        "setup_actions": [
            {"action": "build_gas", "quantity": 4, "parallel_slots": 2},
            {"action": "build_barracks", "quantity": 3, "parallel_slots": 2},
            {"action": "build_barracks_reactor", "quantity": 2, "parallel_slots": 2},
            {"action": "build_barracks_techlab", "quantity": 1, "parallel_slots": 1},
            {"action": "build_factory", "quantity": 2, "parallel_slots": 2},
            {"action": "build_factory_techlab", "quantity": 2, "parallel_slots": 2},
        ],
    }


def _battlecruiser_package() -> dict:
    return {
        "economy": {
            "worker_target_before_commitment": 50,
            "base_target_before_commitment": 3,
            "gas_workers_before_commitment": 18,
        },
        "gate_components": [
            {"action": "train_battlecruiser", "quantity": 6, "production_slots": 2},
            {"action": "train_thor", "quantity": 4, "production_slots": 2},
            {"action": "train_siege_tank", "quantity": 6, "production_slots": 2},
            {"action": "research_yamato_cannon", "quantity": 1, "production_slots": 1},
        ],
        "setup_actions": [
            {"action": "build_gas", "quantity": 6, "parallel_slots": 3},
            {"action": "build_barracks", "quantity": 1, "parallel_slots": 1},
            {"action": "build_factory", "quantity": 2, "parallel_slots": 2},
            {"action": "build_factory_techlab", "quantity": 2, "parallel_slots": 2},
            {"action": "build_starport", "quantity": 2, "parallel_slots": 2},
            {"action": "build_starport_techlab", "quantity": 2, "parallel_slots": 2},
            {"action": "build_armory", "quantity": 1, "parallel_slots": 1},
            {"action": "build_fusion_core", "quantity": 1, "parallel_slots": 1},
        ],
    }


def test_income_model_matches_runtime_saturation_rates() -> None:
    simulator = TerranBuildOrderSimulator(_marine_support_package(gas_workers=6))
    state = SimState(workers=24, bases=1, refineries=2)

    mineral_rate, gas_rate, mineral_workers, gas_workers = simulator._income_rates(state)

    assert mineral_workers == 18
    assert gas_workers == 6
    assert mineral_rate == 17.0  # 16 ideal + 2 oversaturated at half efficiency
    assert round(gas_rate, 6) == round(6 * GAS_RATE_PER_WORKER, 6)


def test_initial_state_matches_project_runtime_start() -> None:
    state = SimState()

    assert INITIAL_WORKERS == 8
    assert INITIAL_SUPPLY_USED == 8
    assert INITIAL_SUPPLY_CAP == 13
    assert state.workers == 8
    assert state.supply_used == 8
    assert state.supply_cap == 13


def test_simulator_calculates_resource_feasible_marine_timing() -> None:
    result = simulate_terran_first_commitment(_marine_package())

    assert result["complete"] is True
    assert result["earliest_feasible_time_seconds"] == 239.591
    assert result["selected_schedule_policy"] == "economy_first"
    assert result["economy"]["worker_target_before_commitment"] == 20
    assert result["errors"] == []


def test_simulator_calculates_resource_feasible_tank_timing() -> None:
    result = simulate_terran_first_commitment(_tank_package())

    assert result["complete"] is True
    assert result["earliest_feasible_time_seconds"] == 502.255
    assert result["selected_schedule_policy"] == "economy_first"


def test_simulator_calculates_resource_feasible_battlecruiser_timing() -> None:
    result = simulate_terran_first_commitment(_battlecruiser_package())

    assert result["complete"] is True
    assert result["earliest_feasible_time_seconds"] == 641.695
    assert result["selected_schedule_policy"] == "economy_first"
    assert result["total_cost"]["gas"] == 4200.0


def test_gas_support_package_has_material_minimum_delay() -> None:
    parent = simulate_terran_first_commitment(_marine_package())
    candidate = simulate_terran_first_commitment(
        _marine_support_package(gas_workers=6)
    )

    assert parent["complete"] is True
    assert candidate["complete"] is True
    assert candidate["earliest_feasible_time_seconds"] > parent["earliest_feasible_time_seconds"] + 100
    assert candidate["total_cost"]["gas"] >= 175


def test_gas_worker_allocation_changes_feasible_time() -> None:
    three_workers = simulate_terran_first_commitment(
        _marine_support_package(gas_workers=3)
    )
    six_workers = simulate_terran_first_commitment(
        _marine_support_package(gas_workers=6)
    )

    assert three_workers["earliest_feasible_time_seconds"] != six_workers["earliest_feasible_time_seconds"]


def test_gas_worker_target_adds_required_refineries() -> None:
    package = _marine_support_package(gas_workers=6)
    package["setup_actions"] = [
        item for item in package["setup_actions"] if item["action"] != "build_gas"
    ]

    result = simulate_terran_first_commitment(package)

    assert result["complete"] is True
    assert result["targets"]["build_gas"]["count"] == 2


def test_impossible_gas_worker_target_is_rejected() -> None:
    package = _marine_support_package(gas_workers=20)

    result = simulate_terran_first_commitment(package)

    assert result["complete"] is False
    assert any("gas worker target" in error for error in result["errors"])


def test_irregular_upgrade_alias_resolves_to_executor_action() -> None:
    package = _marine_package()
    package["gate_components"].append(
        {
            "action": "research_combat_shield",
            "quantity": 1,
            "production_slots": 1,
        }
    )
    package["setup_actions"].append(
        {
            "action": "build_barracks_techlab",
            "quantity": 1,
            "parallel_slots": 1,
        }
    )
    package["economy"]["gas_workers_before_commitment"] = 3
    package["setup_actions"].append(
        {"action": "build_gas", "quantity": 1, "parallel_slots": 1}
    )

    result = simulate_terran_first_commitment(package)

    assert result["complete"] is True
    assert "research_shieldwall" in result["targets"]


def test_first_commitment_over_200_supply_is_rejected() -> None:
    package = _marine_package()
    package["gate_components"][0]["quantity"] = 190

    result = simulate_terran_first_commitment(package)

    assert result["complete"] is False
    assert any("200-supply cap" in error for error in result["errors"])
