from __future__ import annotations

import pytest

from evol_agent.analysis.support_power import (
    estimate_observation_support_power,
    estimate_support_aware_power,
    estimate_timing_package_support_power,
)


def test_medivac_adds_bounded_sustain_without_changing_direct_power() -> None:
    estimate = estimate_support_aware_power(
        {"MARINE": 12, "SIEGETANK": 3, "MEDIVAC": 1}
    )

    assert estimate["direct_power"] == pytest.approx(21.0)
    assert estimate["support_bonus_power"] == pytest.approx(1.792)
    assert estimate["support_adjusted_power"] == pytest.approx(22.792)
    assert estimate["medivac_count"] == 1
    assert estimate["healable_biological_count"] == 12


def test_medivac_does_not_add_healing_power_to_mechanical_only_force() -> None:
    estimate = estimate_support_aware_power(
        {"SIEGETANK": 6, "VIKINGFIGHTER": 2, "MEDIVAC": 2}
    )

    assert estimate["support_bonus_power"] == 0
    assert estimate["support_adjusted_power"] == estimate["direct_power"]


def test_visible_anti_air_reduces_only_the_support_term() -> None:
    safe = estimate_support_aware_power(
        {"MARINE": 20, "SIEGETANK": 4, "MEDIVAC": 2}
    )
    threatened = estimate_support_aware_power(
        {"MARINE": 20, "SIEGETANK": 4, "MEDIVAC": 2},
        enemy_composition={"MARINE": 20},
    )

    assert threatened["direct_power"] == safe["direct_power"]
    assert threatened["support_bonus_power"] < safe["support_bonus_power"]
    assert threatened["medivac_survival_factor"] < 1


def test_observation_recovers_medivac_from_completed_counts() -> None:
    estimate = estimate_observation_support_power(
        {
            "own_forces": {
                "combat_composition": {"MARINE": 20, "SIEGETANK": 4},
                "completed_counts": {"MARINE": 20, "SIEGETANK": 4, "MEDIVAC": 2},
            },
            "enemy": {"visible_composition": {"MARINE": 10}},
            "technology": {"completed_upgrades": ["SHIELDWALL"]},
            "combat": {
                "controlled_own_army_power": 31.5,
                "visible_enemy_army_power": 10.0,
            },
        }
    )

    assert estimate is not None
    assert estimate["recorded_sharpy_power"] == 31.5
    assert estimate["removed_fixed_support_power"] == 4.0
    assert estimate["direct_power"] == 27.5
    assert estimate["medivac_count"] == 2
    assert estimate["support_adjusted_to_enemy_power_ratio"] > 2.75


def test_timing_package_estimate_reads_gate_units() -> None:
    estimate = estimate_timing_package_support_power(
        {
            "gate_components": [
                {"action": "train_marine", "quantity": 12},
                {"action": "train_siege_tank", "quantity": 3},
                {"action": "train_medivac", "quantity": 1},
            ],
            "setup_actions": [],
        }
    )

    assert estimate["direct_power"] == 21
    assert estimate["support_adjusted_power"] > 21
