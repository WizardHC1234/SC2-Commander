from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..analysis.fixed_timeline import build_fixed_match_timeline
from ..analysis.get_chunk import extract_chunks


class MatchRecordReader:
    """Load one match record and expose its deterministic fixed timeline."""

    def __init__(self, record_path: str | Path) -> None:
        self.record_path = Path(record_path).resolve()
        self._data: dict[str, Any] | None = None
        self._extracted: dict[str, Any] | None = None
        self._enemy_truth: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._data is None:
            data = json.loads(self.record_path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict):
                raise ValueError("match record root must be a JSON object")
            self._data = data
        return self._data

    def _extract(self) -> dict[str, Any]:
        if self._extracted is None:
            self._extracted = extract_chunks(self._load())
        return self._extracted

    def _chunks(self) -> list[dict[str, Any]]:
        chunks = self._extract().get("chunks", [])
        return chunks if isinstance(chunks, list) else []

    def _load_enemy_truth(self) -> dict[str, Any]:
        if self._enemy_truth is not None:
            return self._enemy_truth
        path = self.record_path.with_name(
            f"{self.record_path.stem}.enemy_truth.json"
        )
        if not path.is_file():
            self._enemy_truth = {}
            return self._enemy_truth
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            data = {}
        self._enemy_truth = data if isinstance(data, dict) else {}
        return self._enemy_truth

    def action_space_selection_summary(self) -> dict[str, Any]:
        """Return selector categories as recorded host facts."""
        for chunk_index, chunk in reversed(list(enumerate(self._chunks()))):
            decision = chunk.get("decision") if isinstance(chunk.get("decision"), dict) else {}
            selection = (
                decision.get("tool_selection")
                if isinstance(decision.get("tool_selection"), dict)
                else {}
            )
            if not selection:
                continue

            def names(key: str) -> list[str]:
                raw = selection.get(key)
                if not isinstance(raw, list):
                    return []
                return list(
                    dict.fromkeys(
                        str(item).strip()
                        for item in raw
                        if str(item).strip()
                    )
                )

            selected = names("selected_tools")
            semantic = names("semantic_tools")
            dependencies = names("dependency_tools")
            baseline = names("baseline_tools")
            return {
                "source": "recorded_strategy_tool_selection",
                "chunk_index": chunk_index,
                "selected_tools": selected,
                "semantic_tools": semantic,
                "dependency_tools": dependencies,
                "baseline_tools": baseline,
                "selected_tool_count": (
                    selection.get("selected_tool_count")
                    if selection.get("selected_tool_count") is not None
                    else len(selected)
                ),
                "full_tool_count": selection.get("full_tool_count"),
                "fallback_used": bool(selection.get("fallback_used")),
                "fallback_reason": str(selection.get("fallback_reason") or ""),
                "dependency_error": str(selection.get("dependency_error") or ""),
            }
        return {}

    def manifest(self, record_id: str) -> dict[str, Any]:
        extracted = self._extract()
        metadata = (
            extracted.get("metadata")
            if isinstance(extracted.get("metadata"), dict)
            else {}
        )
        chunks = self._chunks()
        commander_rows = sum(
            1
            for chunk in chunks
            if str(chunk.get("agent_role") or "") == "commander"
            or bool(chunk.get("army_observation"))
        )
        return {
            "record_id": record_id,
            "file_name": self.record_path.name,
            "result": metadata.get("result", "?"),
            "duration": metadata.get("game_duration_formatted", "?"),
            "map": metadata.get("map_name", "?"),
            "matchup": metadata.get("matchup", "?"),
            "opponent": metadata.get("opponent_id", "?"),
            "chunk_count": len(chunks),
            "commander_row_count": commander_rows,
            "opponent_truth": {
                "available": bool(self._load_enemy_truth().get("snapshots")),
                "source": self._load_enemy_truth().get("source"),
                "snapshot_count": self._load_enemy_truth().get("snapshot_count", 0),
            },
            "action_space_selection": self.action_space_selection_summary(),
        }

    def fixed_timeline(self) -> str:
        """Return every Commander snapshot in one compact fixed-schema table."""
        return build_fixed_match_timeline(
            self._extract(),
            action_space_selection=self.action_space_selection_summary(),
            file_name=self.record_path.name,
            opponent_truth=self._load_enemy_truth(),
        )
