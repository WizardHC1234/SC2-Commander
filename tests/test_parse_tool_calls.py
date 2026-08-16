"""Tests for JSON tool_calls parsing / repair."""

from __future__ import annotations

import unittest

from commander.tools import parse_tool_calls_from_content


class ParseToolCallsFromContentTests(unittest.TestCase):
    def test_valid_json_object(self) -> None:
        text = (
            '{"tool_calls":[{"name":"train_scv","arguments":{"to_count":16}},'
            '{"name":"set_wake_event","arguments":{"logic":"any","conditions":'
            '[{"type":"game_time_at_least","seconds":30}}]}}]}'
        )
        calls = parse_tool_calls_from_content(text)
        self.assertEqual(
            [c["name"] for c in calls],
            ["train_scv", "set_wake_event"],
        )

    def test_prose_then_valid_json(self) -> None:
        text = (
            "Workers first, then wake.\n\n"
            '{"tool_calls":[{"name":"train_scv","arguments":{"to_count":16}},'
            '{"name":"set_wake_event","arguments":{"logic":"any","conditions":'
            '[{"type":"game_time_at_least","seconds":30}}]}}]}'
        )
        calls = parse_tool_calls_from_content(text)
        self.assertEqual([c["name"] for c in calls], ["train_scv", "set_wake_event"])

    def test_extra_brace_before_conditions_close(self) -> None:
        # Observed failure mode: ``"seconds":1438}}]`` instead of ``1438}]``.
        text = (
            "Assault the main.\n\n"
            '{"tool_calls":['
            '{"name":"army_intent","arguments":{"mode":"attack",'
            '"zone_id":"zone_15"}},'
            '{"name":"set_wake_event","arguments":{"logic":"any","conditions":'
            '[{"type":"game_time_at_least","seconds":1438}}]}}]}'
        )
        calls = parse_tool_calls_from_content(text)
        self.assertEqual(
            [c["name"] for c in calls],
            ["army_intent", "set_wake_event"],
        )
        wake = calls[-1]["arguments"]
        self.assertEqual(wake["conditions"][0]["seconds"], 1438)

    def test_extra_closing_noise_at_end(self) -> None:
        text = (
            "Continue assault.\n\n"
            '{"tool_calls":['
            '{"name":"army_intent","arguments":{"mode":"attack",'
            '"zone_id":"zone_15"}},'
            '{"name":"set_wake_event","arguments":{"logic":"any","conditions":'
            '[{"type":"game_time_at_least","seconds":1498}}]}]}'
        )
        calls = parse_tool_calls_from_content(text)
        self.assertEqual(
            [c["name"] for c in calls],
            ["army_intent", "set_wake_event"],
        )


if __name__ == "__main__":
    unittest.main()
