from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.config import SKILL_ROOT, canonical_strategy_folder
from ..validation import validate_strategy_markdown


def output_dir_for_strategy(
    strategy_name: str,
    race: str,
    *,
    overlay_root: Path | None = None,
) -> Path:
    """Allocate the next immutable candidate directory.

    Evolution passes ``overlay_root`` (typically ``evolution_runs/.../strategies``)
    so candidates are not written into ``skills/<race>/``.
    """
    folder = canonical_strategy_folder(strategy_name)
    base_name = re.sub(r"_opt\d+$", "", folder)
    max_n = 0
    search_dir = Path(overlay_root) if overlay_root is not None else (SKILL_ROOT / race)
    if search_dir.exists():
        pattern = re.compile(rf"^{re.escape(base_name)}_opt(\d+)$")
        for entry in search_dir.iterdir():
            if not entry.is_dir():
                continue
            match = pattern.match(entry.name)
            if match:
                max_n = max(max_n, int(match.group(1)))
    search_dir.mkdir(parents=True, exist_ok=True)
    return search_dir / f"{base_name}_opt{max_n + 1}"


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
    allow_validation_warning: bool = False,
) -> list[dict[str, Any]]:
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("output directory must not overwrite the parent strategy")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"candidate directory already contains files and is immutable: {output_dir}"
        )
    top_content = files.get("strategy.md", "")
    validation_error = validate_strategy_markdown(top_content, race=race)
    if validation_error and not allow_validation_warning:
        raise ValueError(validation_error)

    output_dir.mkdir(parents=True, exist_ok=True)
    changes: list[dict[str, Any]] = []
    write_file(output_dir / "strategy.md", top_content)
    change: dict[str, Any] = {"file": "strategy.md", "applied": True}
    if validation_error:
        change["validation_warning"] = validation_error
    changes.append(change)
    return changes
