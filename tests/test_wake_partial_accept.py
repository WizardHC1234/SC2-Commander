"""Wake-only validation failures must still apply the valid macro/army tools.

Regression test for the macro deadlock: a decision whose wake event is
unreachable (e.g. unit_count on a structure) used to be rejected wholesale
after reflection, so the bot inherited the previous (stale) decision and a
perfectly good macro tool set was discarded cycle after cycle.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

from commander import agent


def _llm_result(content: str) -> Dict[str, Any]:
    return {
        "content": content,
        "raw_message": {},
        "error": "",
        "latency_seconds": 0.1,
        "usage": {},
        "finish_reason": "stop",
    }


def _decision_kwargs() -> Dict[str, Any]:
    return dict(
        race="terran",
        strategy_description="macro then marines",
        observation_text="obs",
        action_space={
            "train_scv": "Set absolute SCV target",
            "build_gas": "Set refinery target",
        },
        model_key="test-model-json",
        full_observation=None,
        ensure_addon_parents=None,
        runtime_hint="",
        map_topology_text="",
    )


def test_wake_only_failure_applies_tools_with_fallback_wake() -> None:
    content = (
        "reasoning paragraph\n\n"
        '{"tool_calls": ['
        '{"name": "train_scv", "arguments": {"to_count": 22}}, '
        '{"name": "set_wake_event", "arguments": {"logic": "any", "conditions": '
        '[{"type": "unit_count_at_least", "unit": "CommandCenter", "count": 2}]}}]}'
    )
    with patch.object(
        agent,
        "call_openai_detailed",
        side_effect=[_llm_result(content), _llm_result(content)],
    ):
        outcome = agent.run_commander_decision(**_decision_kwargs())

    assert outcome["accepted"] is True
    assert outcome["wake_event"] is None
    assert any(
        task.get("action") == "train_scv" and task.get("to_count") == 22
        for task in outcome["tasks"]
    )
    assert any(
        "decision_reflection_exhausted" in str(issue)
        for issue in outcome["issues"]
    )


def test_empty_parse_still_rejects_for_inheritance() -> None:
    with patch.object(
        agent,
        "call_openai_detailed",
        side_effect=[_llm_result("no json here"), _llm_result("still nothing")],
    ):
        outcome = agent.run_commander_decision(**_decision_kwargs())

    assert outcome["accepted"] is False
