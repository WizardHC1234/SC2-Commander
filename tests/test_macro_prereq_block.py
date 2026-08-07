"""Macro Execution blocked= annotations: hard prereq + addon soft-fail."""

from commander.observation import (
    _macro_execution_lines,
    _macro_prereq_block_reason,
)


def test_train_marine_no_barracks_is_blocked():
    production = {"completed": {}, "under_construction": {}}
    assert _macro_prereq_block_reason("train_marine", production) == "blocked=no_barracks"


def test_train_marine_barracks_pending_is_not_blocked():
    production = {
        "completed": {},
        "under_construction": {"BARRACKS": 1},
    }
    assert _macro_prereq_block_reason("train_marine", production) is None


def test_build_barracks_techlab_no_barracks_is_blocked():
    production = {"completed": {}, "under_construction": {}}
    assert (
        _macro_prereq_block_reason("build_barracks_techlab", production)
        == "blocked=no_barracks"
    )


def test_train_siege_tank_no_techlab_is_blocked():
    production = {
        "completed": {"FACTORY": 1},
        "under_construction": {},
        "producer_addons": {"FACTORY": {"with_techlab": 0}},
    }
    assert (
        _macro_prereq_block_reason("train_siege_tank", production)
        == "blocked=no_factory_techlab"
    )


def test_train_siege_tank_techlab_pending_is_not_blocked():
    production = {
        "completed": {"FACTORY": 1},
        "under_construction": {"FACTORYTECHLAB": 1},
        "producer_addons": {"FACTORY": {"with_techlab": 0}},
    }
    assert _macro_prereq_block_reason("train_siege_tank", production) is None


def test_reactor_soft_fail_need_more_barracks():
    """2 Barracks occupied (1 techlab + 1 reactor) cannot reach reactor→2."""
    production = {
        "completed": {"BARRACKS": 2, "BARRACKSTECHLAB": 1, "BARRACKSREACTOR": 1},
        "under_construction": {},
        "producer_addons": {
            "BARRACKS": {
                "ready": 2,
                "with_techlab": 1,
                "with_reactor": 1,
                "no_addon": 0,
            }
        },
    }
    assert (
        _macro_prereq_block_reason(
            "build_barracks_reactor",
            production,
            to_count=2,
            current_count=1,
        )
        == "blocked=need_more_barracks"
    )


def test_reactor_free_slot_is_not_soft_blocked():
    production = {
        "completed": {"BARRACKS": 2, "BARRACKSREACTOR": 1},
        "under_construction": {},
        "producer_addons": {
            "BARRACKS": {
                "ready": 2,
                "with_techlab": 0,
                "with_reactor": 1,
                "no_addon": 1,
            }
        },
    }
    assert (
        _macro_prereq_block_reason(
            "build_barracks_reactor",
            production,
            to_count=2,
            current_count=1,
        )
        is None
    )


def test_reactor_pending_barracks_is_not_soft_blocked():
    production = {
        "completed": {"BARRACKS": 2, "BARRACKSTECHLAB": 1, "BARRACKSREACTOR": 1},
        "under_construction": {"BARRACKS": 1},
        "producer_addons": {
            "BARRACKS": {
                "ready": 2,
                "with_techlab": 1,
                "with_reactor": 1,
                "no_addon": 0,
            }
        },
    }
    assert (
        _macro_prereq_block_reason(
            "build_barracks_reactor",
            production,
            to_count=2,
            current_count=1,
        )
        is None
    )


def test_macro_execution_annotates_soft_fail():
    production = {
        "completed": {"BARRACKS": 2},
        "under_construction": {},
        "producer_addons": {
            "BARRACKS": {
                "ready": 2,
                "with_techlab": 1,
                "with_reactor": 1,
                "no_addon": 0,
            }
        },
    }
    execution = {
        "macro": {
            "status": "active_unsatisfied",
            "active_macro_tasks": [
                {
                    "action": "build_barracks_reactor",
                    "to_count": 2,
                    "current_count": 1,
                    "status": "active_unsatisfied",
                },
            ],
            "last_update_seconds_ago": 1,
            "last_issues": [],
        }
    }
    joined = "\n".join(_macro_execution_lines(execution, production))
    assert "blocked=need_more_barracks" in joined


def test_macro_execution_annotates_only_unsatisfied_failures():
    production = {"completed": {}, "under_construction": {}}
    execution = {
        "macro": {
            "status": "active_unsatisfied",
            "active_macro_tasks": [
                {
                    "action": "train_marine",
                    "to_count": 10,
                    "current_count": 0,
                    "status": "active_unsatisfied",
                },
                {
                    "action": "train_marine",
                    "to_count": 10,
                    "current_count": 10,
                    "status": "target_satisfied",
                },
            ],
            "last_update_seconds_ago": 1,
            "last_issues": [],
        }
    }
    lines = _macro_execution_lines(execution, production)
    joined = "\n".join(lines)
    assert "blocked=no_barracks" in joined
    assert joined.count("blocked=no_barracks") == 1
