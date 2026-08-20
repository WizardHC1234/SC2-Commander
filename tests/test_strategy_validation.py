from __future__ import annotations

from pathlib import Path

import pytest

from evol_agent.validation import validate_strategy_markdown


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("strategy_name", ["marine", "tank", "battlecruiser"])
def test_base_strategies_share_the_same_generic_validation(
    strategy_name: str,
) -> None:
    strategy = (
        ROOT / "skills" / "terran" / strategy_name / "strategy.md"
    ).read_text(encoding="utf-8")

    assert validate_strategy_markdown(strategy, race="terran") is None


def test_strategy_validation_rejects_runtime_owned_transformation_gate() -> None:
    strategy = """# Summary

A reusable Terran attack strategy.

# Details

* Main Attack Gate: Hold until all Siege Tanks are in Siege Mode before attacking.
"""

    error = validate_strategy_markdown(strategy, race="terran")

    assert error is not None
    assert "pre-sieged" in error


def test_strategy_validation_rejects_runtime_transformation_identifier() -> None:
    strategy = """# Summary

A reusable Terran attack strategy.

# Details

* Main Attack Gate: Hold until at least 10 own SiegeTankSieged are observed.
"""

    error = validate_strategy_markdown(strategy, race="terran")

    assert error is not None
    assert "transformation-state" in error


def test_strategy_validation_rejects_impossible_fifth_natural_refinery() -> None:
    strategy = """# Summary

A reusable Terran macro strategy.

# Details

* Economy: Add a fifth Refinery at the natural base before expanding again.
"""

    error = validate_strategy_markdown(strategy, race="terran")

    assert error is not None
    assert "fifth Refinery" in error


def test_supply_budget_does_not_add_attack_gate_to_larger_end_state() -> None:
    strategy = """# Summary

A reusable Terran attack strategy.

# Details

* Ultimate Goal: Continue toward 75 Marines, 14 Siege Tanks, 4 Medivacs, and 44 SCVs; the earlier attack gate remains 45 Marines, 8 Siege Tanks, and 2 Medivacs.
"""

    assert validate_strategy_markdown(strategy, race="terran") is None


def test_supply_budget_still_rejects_a_true_oversized_end_state() -> None:
    strategy = """# Summary

A reusable Terran attack strategy.

# Details

* Ultimate Goal: Continue toward 100 Marines, 20 Siege Tanks, 4 Medivacs, and 44 SCVs.
"""

    error = validate_strategy_markdown(strategy, race="terran")

    assert error is not None
    assert "212 supply" in error
