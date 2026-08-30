from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from .types import GameEvidence


MATCH_SUMMARY_FORMAT = "fixed_match_timeline_v4_grounded_engagements"


class MatchSummaryCache:
    """Persistent successful per-record summaries shared across EvolAgent stages."""

    _SCHEMA = "sc2.experiment_match_summary_cache.v1"

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        if path is None or not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return
        entries = data.get("entries") if isinstance(data, dict) else None
        if isinstance(entries, dict):
            self._entries = {
                str(key): dict(value)
                for key, value in entries.items()
                if isinstance(value, dict)
            }

    @staticmethod
    def key(path: str | Path) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    @staticmethod
    def _fingerprint(path: str | Path) -> tuple[int, int] | None:
        try:
            stat = Path(path).stat()
        except OSError:
            return None
        return stat.st_size, stat.st_mtime_ns

    def get(
        self,
        record: GameEvidence,
        *,
        strategy_name: str,
        race: str,
        model: str = "",
    ) -> dict[str, Any] | None:
        fingerprint = self._fingerprint(record.file)
        if fingerprint is None:
            return None
        with self._lock:
            entry = self._entries.get(self.key(record.file))
            if not isinstance(entry, dict):
                return None
            cached_model = str(entry.get("model") or "").strip()
            cached_format = str(entry.get("summary_format") or "").strip()
            if (
                int(entry.get("size") or -1) != fingerprint[0]
                or int(entry.get("mtime_ns") or -1) != fingerprint[1]
                or str(entry.get("strategy") or "") != strategy_name
                or str(entry.get("race") or "") != race
                or (cached_model and model and cached_model != model)
                or (cached_format and cached_format != MATCH_SUMMARY_FORMAT)
                or not isinstance(entry.get("summary"), dict)
            ):
                return None
            digest = entry.get("digest")
            return {
                "summary": dict(entry["summary"]),
                "digest": dict(digest) if isinstance(digest, dict) else {},
                "errors": [str(item) for item in (entry.get("errors") or [])],
                "source": str(entry.get("source") or "persistent_cache"),
            }

    def put(
        self,
        record: GameEvidence,
        *,
        strategy_name: str,
        race: str,
        model: str,
        summary: dict[str, Any],
        errors: list[str],
        source: str,
        digest: dict[str, Any] | None = None,
    ) -> None:
        fingerprint = self._fingerprint(record.file)
        if fingerprint is None or not summary:
            return
        with self._lock:
            self._entries[self.key(record.file)] = {
                "record_path": str(Path(record.file).resolve()),
                "size": fingerprint[0],
                "mtime_ns": fingerprint[1],
                "strategy": strategy_name,
                "race": race,
                "model": str(model or "").strip(),
                "summary_format": MATCH_SUMMARY_FORMAT,
                "summary": summary,
                "digest": dict(digest or {}),
                "errors": errors,
                "source": source,
            }
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(
                {"schema": self._SCHEMA, "entries": self._entries},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temp_path.replace(self.path)


__all__ = ["MATCH_SUMMARY_FORMAT", "MatchSummaryCache"]
