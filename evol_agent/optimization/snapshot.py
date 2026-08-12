from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.config import SKILL_ROOT
from ..validation import validate_strategy_markdown


def output_dir_for_strategy(strategy_name: str, race: str) -> Path:
    base_name = re.sub(r"_opt\d+$", "", strategy_name)
    max_n = 0
    skill_race_dir = SKILL_ROOT / race
    if skill_race_dir.exists():
        pattern = re.compile(rf"^{re.escape(base_name)}_opt(\d+)$")
        for entry in skill_race_dir.iterdir():
            if not entry.is_dir():
                continue
            match = pattern.match(entry.name)
            if match:
                max_n = max(max_n, int(match.group(1)))
    return skill_race_dir / f"{base_name}_opt{max_n + 1}"


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def save_snapshot(
    *,
    source_dir: Path,
    files: dict[str, str],
    output_dir: Path,
    source_info: dict[str, Any],
    race: str = "",
) -> list[dict[str, Any]]:
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("output directory must not overwrite the parent strategy")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"candidate directory already contains files and is immutable: {output_dir}"
        )
    top_content = files.get("strategy.md", "")
    validation_error = validate_strategy_markdown(top_content, race=race)
    if validation_error:
        raise ValueError(validation_error)

    output_dir.mkdir(parents=True, exist_ok=True)
    changes: list[dict[str, Any]] = []
    write_file(output_dir / "strategy.md", top_content)
    changes.append({"file": "strategy.md", "applied": True})
    return changes
