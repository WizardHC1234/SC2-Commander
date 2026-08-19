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
