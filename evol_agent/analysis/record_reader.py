from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..interaction_schema import (
    STRATEGY_COORDINATOR_INITIAL,
    STRATEGY_COORDINATOR_STRATEGY,
    interaction_get_dict,
    interaction_get_str,
)

from ..core.config import SKILL_FILES, resolve_skill_dir
from ..core.types import GameEvidence


FINAL_SAVE_REASONS = frozenset({"on_end", "match_runner_finally"})


class IncompleteMatchRecordError(ValueError):
    """Raised when a current record is only an autosave/crash snapshot."""


def is_completed_match_record(data: dict[str, Any]) -> bool:
    """Accept final current records and historical records without this field."""
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    save_reason = str(meta.get("save_reason") or "").strip()
    if not save_reason:
        return True
    return save_reason in FINAL_SAVE_REASONS


def extract_strategy_from_record(data: dict[str, Any]) -> Optional[str]:
    meta = data.get("metadata") or {}
    strategy = (
        meta.get("strategy_id")
        or meta.get("strategy")
        or meta.get("force_strategy")
    )
    if strategy:
        return str(strategy)
    for inter in data.get("interactions", []):
        if not isinstance(inter, dict):
            continue
        strategy = inter.get("strategy_id") or inter.get("forced_strategy")
        if strategy:
            return str(strategy)
        strategy = interaction_get_str(inter, STRATEGY_COORDINATOR_STRATEGY)
        if strategy:
            return strategy
        initial = interaction_get_dict(inter, STRATEGY_COORDINATOR_INITIAL)
        strategy = initial.get("forced_strategy") or initial.get("selected_strategy")
        if strategy:
            return str(strategy)
    return None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig") if path.exists() else ""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def skill_dir_for_record(
    record_path: Path,
    strategy_name: str,
    race: str = "terran",
) -> Path:
    """Use the strategy.md saved beside a match/batch record when present."""
    match_dir = record_path.parent
    if (match_dir / "strategy.md").is_file():
        return match_dir
    batch_dir = match_dir.parent
    if (batch_dir / "strategy.md").is_file():
        return batch_dir
    return resolve_skill_dir(strategy_name, race)


def load_skill_texts(skill_dir: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for fname in SKILL_FILES:
        content = read_text(skill_dir / fname)
        if content:
            texts[fname] = content
    return texts


def find_record_jsons(batch_dir: Path) -> list[Path]:
    paths = sorted(batch_dir.rglob("*.json"))
    return [
        path
        for path in paths
        if not path.name.startswith("all_compressed_")
        and not path.name.endswith(".enemy_truth.json")
    ]


def build_record_evidence_baseline(
    record_path: Path,
) -> tuple[str, str, Path, GameEvidence]:
    data = json.loads(record_path.read_text(encoding="utf-8-sig"))
    if not is_completed_match_record(data):
        save_reason = (data.get("metadata") or {}).get("save_reason", "unknown")
        raise IncompleteMatchRecordError(
            f"Match record is not final ({save_reason}): {record_path}"
        )
    meta = data.get("metadata") or {}
    strategy_name = extract_strategy_from_record(data)
    if not strategy_name:
        raise ValueError(f"Could not extract strategy from {record_path}")

    race = str(meta.get("my_race") or "terran").lower()
    skill_dir = skill_dir_for_record(record_path, strategy_name, race)
    if not (skill_dir / "strategy.md").is_file():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

    evidence = GameEvidence(
        file=str(record_path),
        result=str(meta.get("result", "?")),
        duration=str(meta.get("game_duration_formatted", "?")),
        timeline="",
        meta=meta,
    )
    return strategy_name, race, skill_dir, evidence


def group_records_by_strategy(
    record_paths: list[Path],
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for path in record_paths:
        try:
            strategy_name, race, skill_dir, evidence = build_record_evidence_baseline(path)
        except IncompleteMatchRecordError as exc:
            print(f"[EvolAgent] skipped incomplete record: {exc}")
            continue
        key = (race, strategy_name)
        if key not in grouped:
            grouped[key] = {
                "race": race,
                "strategy_name": strategy_name,
                "skill_dir": skill_dir,
                "skill_texts": load_skill_texts(skill_dir),
                "records": [],
            }
        grouped[key]["records"].append(evidence)
    return grouped
