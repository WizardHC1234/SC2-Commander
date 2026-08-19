"""Summarize SC2 match records by batch folder and report win rates.

Reads ``game_records/<batch>/<match_id>/`` layouts produced by
``scripts/start_batch.ps1`` / ``run_vs_ai.py`` (also compatible with the older
multi-agent ``match_info.txt`` model fields):

* ``{match_id}.json``  — result / duration / map (metadata only)
* ``match_info.txt``   — shared batch config (bot, enemy, strategy, models)

Examples::

    python tools/batch_stats.py
    python tools/batch_stats.py --batch batch_20260728_020940_e1_KairosJunctionL
    python tools/batch_stats.py --group-by difficulty
    python tools/batch_stats.py --per-batch
    python tools/batch_stats.py --list-matches --detail
    python tools/batch_stats.py --json-out batch_summary.json

Default overview merges rows that share strategy + model + opponent race +
difficulty + enemy build across batch folders. Use ``--per-batch`` to keep one
line per folder.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


SKIP_DIR_NAMES = {"_batch_logs", "__pycache__", ".git"}
SKIP_JSON_NAMES = {"meta.json"}
WIN_RESULTS = {"victory", "win"}
# Interrupted / force-quit is treated as a loss for win-rate reporting.
LOSS_RESULTS = {"defeat", "loss", "interrupted"}
TIE_RESULTS = {"tie", "draw"}
GROUP_BY_CHOICES = (
    "batch",
    "difficulty",
    "strategy",
    "enemy",
    "race",
    "map",
    "matchup",
)


_MODEL_SHORT = {
    "kimi-k2.5": "kimi",
    "qwen3-32b-reasoning": "qwen3-32b-r",
    "deepseek-v4-flash": "ds4-flash",
    "deepseek-v4-pro": "ds-pro",
}


@dataclass
class MatchRow:
    match_id: str
    batch: str
    result: str
    duration: str
    map_name: str
    matchup: str
    strategy: str
    json_path: str
    info_path: str = ""
    enemy: str = ""
    difficulty: str = ""
    enemy_build: str = ""
    enemy_race: str = ""
    coordinator_model: str = ""
    macro_model: str = ""
    translator_model: str = ""
    army_model: str = ""
    error: str = ""

    @staticmethod
    def _short_model(name: str) -> str:
        text = (name or "").strip()
        if not text:
            return "-"
        return _MODEL_SHORT.get(text.lower(), text)

    @property
    def model_key(self) -> str:
        """Compact model signature for grouping/display.

        Same model on all roles -> ``kimi`` / ``qwen``.
        Mixed roles -> ``kimi/kimi/kimi/qwen`` (C/M/T/A).
        """
        roles = [
            self.coordinator_model.strip(),
            self.macro_model.strip(),
            self.translator_model.strip(),
            self.army_model.strip(),
        ]
        present = [m for m in roles if m]
        if not present:
            return "-"
        shorts = [self._short_model(m) for m in roles]
        unique = {m for m in present}
        if len(unique) == 1:
            return shorts[0] if shorts[0] != "-" else self._short_model(present[0])
        return "/".join(s if s != "-" else "?" for s in shorts)


@dataclass
class BatchSummary:
    batch: str
    path: str
    total: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    unknown: int = 0
    incomplete: int = 0
    win_rate: Optional[float] = None
    config: dict[str, str] = field(default_factory=dict)
    result_counts: dict[str, int] = field(default_factory=dict)
    matches: list[MatchRow] = field(default_factory=list)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_read_text(path: Path, limit: Optional[int] = None) -> str:
    raw = path.read_bytes()
    if limit is not None and len(raw) > limit:
        raw = raw[:limit]
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_match_info(path: Path) -> dict[str, str]:
    """Parse ``match_info.txt`` key: value lines (plus bot_instruct block)."""
    if not path.is_file():
        return {}
    text = _safe_read_text(path)
    info: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.lower().startswith("bot_instruct:"):
            rest = line.split(":", 1)[1].strip()
            block: list[str] = []
            if rest:
                block.append(rest)
            i += 1
            while i < len(lines):
                nxt = lines[i].rstrip()
                if not nxt:
                    break
                if ":" in nxt and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*:\s", nxt):
                    break
                block.append(nxt)
                i += 1
            info["bot_instruct"] = "\n".join(block).strip()
            continue
        if ":" in line and not line.startswith("="):
            key, value = line.split(":", 1)
            key = key.strip()
            if key and key not in {"SC2 Match Info", "=============="}:
                info[key] = value.strip()
        i += 1
    return info


def load_metadata_only(json_path: Path) -> dict[str, Any]:
    """Parse only the top-level ``metadata`` object (avoids loading huge interactions)."""
    # Metadata sits at the start of our recorder output; 256 KiB is plenty.
    text = _safe_read_text(json_path, limit=256 * 1024).lstrip()
    # Normalize newlines so CRLF records decode cleanly.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text or text[0] not in "{[":
        raise ValueError("empty or non-JSON record")
    key = '"metadata"'
    idx = text.find(key)
    if idx < 0:
        # Fallback: full parse if the file is small enough already buffered
        data = json.loads(_safe_read_text(json_path))
        meta = data.get("metadata") if isinstance(data, dict) else None
        if isinstance(meta, dict):
            return meta
        raise ValueError("metadata field missing")
    colon = text.find(":", idx + len(key))
    if colon < 0:
        raise ValueError("metadata field malformed")
    # json.JSONDecoder.raw_decode does not skip leading whitespace at idx.
    brace = text.find("{", colon)
    if brace < 0:
        raise ValueError("metadata object missing")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text, brace)
    if not isinstance(obj, dict):
        raise ValueError("metadata is not an object")
    return obj


def find_record_json(match_dir: Path) -> Optional[Path]:
    preferred = match_dir / f"{match_dir.name}.json"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        p
        for p in match_dir.glob("*.json")
        if p.name not in SKIP_JSON_NAMES and not p.name.startswith("all_compressed_")
    )
    return candidates[0] if candidates else None


def classify_result(result: str) -> str:
    key = str(result or "").strip().lower()
    if key in WIN_RESULTS:
        return "win"
    if key in LOSS_RESULTS:
        return "loss"
    if key in TIE_RESULTS:
        return "tie"
    if not key or key in {"unknown", "?", "none"}:
        return "unknown"
    return "unknown"


def win_rate(wins: int, losses: int, ties: int = 0) -> Optional[float]:
    decided = wins + losses + ties
    if decided <= 0:
        return None
    return wins / decided


_RACE_ALIASES = {
    "t": "terran",
    "terran": "terran",
    "p": "protoss",
    "protoss": "protoss",
    "z": "zerg",
    "zerg": "zerg",
    "r": "random",
    "random": "random",
}


def parse_enemy_fields(enemy: str) -> tuple[str, str, str, str]:
    """Split ``AI terran / veryeasy / macro`` into label/diff/build/race."""
    text = str(enemy or "").strip()
    if not text:
        return "", "", "", ""
    parts = [p.strip() for p in text.split("/") if p.strip()]
    # parts[0] often like "AI terran"
    difficulty = parts[1] if len(parts) >= 2 else ""
    enemy_build = parts[2] if len(parts) >= 3 else ""
    race = ""
    if parts:
        tokens = re.split(r"[\s_\-]+", parts[0].lower())
        for token in reversed(tokens):
            if token in _RACE_ALIASES:
                race = _RACE_ALIASES[token]
                break
    return text, difficulty, enemy_build, race


def race_from_matchup(matchup: str) -> str:
    """Map ``TvP`` / ``TvZ`` style labels to enemy race."""
    text = str(matchup or "").strip()
    if len(text) >= 3 and text[1].lower() == "v":
        return _RACE_ALIASES.get(text[2].lower(), "")
    return ""


def _majority(values: list[str]) -> str:
    cleaned = [v.strip() for v in values if v and v.strip() and v.strip() != "?"]
    if not cleaned:
        return "-"
    return Counter(cleaned).most_common(1)[0][0]


def _clip(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _short_batch_name(name: str) -> str:
    """Drop common noise so the batch column stays readable."""
    text = (name or "").strip()
    if text.startswith("batch_"):
        text = text[len("batch_") :]
    for suffix in ("_KairosJunctionL", "_KairosJunctionLE"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text or name


def is_match_dir(path: Path) -> bool:
    if not path.is_dir() or path.name in SKIP_DIR_NAMES:
        return False
    if (path / "match_info.txt").is_file():
        return True
    return find_record_json(path) is not None


def discover_batches(records_dir: Path) -> list[Path]:
    """Return batch directories. Loose matches under root become a virtual batch."""
    if not records_dir.is_dir():
        return []

    batches: list[Path] = []
    loose: list[Path] = []
    for child in sorted(records_dir.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or child.name in SKIP_DIR_NAMES:
            continue
        if child.name.startswith("batch_") or any(
            is_match_dir(sub) for sub in child.iterdir() if sub.is_dir()
        ):
            # Prefer dirs that look like batch containers (have match subdirs).
            match_subs = [sub for sub in child.iterdir() if is_match_dir(sub)]
            if match_subs or child.name.startswith("batch_"):
                batches.append(child)
                continue
        if is_match_dir(child):
            loose.append(child)

    if loose:
        # Marker path: parent with special name handled in summarize.
        batches.append(records_dir / "__ungrouped__")
    return batches


def iter_match_dirs(batch_path: Path, records_dir: Path) -> list[Path]:
    if batch_path.name == "__ungrouped__":
        return sorted(
            (
                child
                for child in records_dir.iterdir()
                if is_match_dir(child)
                and not child.name.startswith("batch_")
                and child.name not in SKIP_DIR_NAMES
            ),
            key=lambda p: p.name.lower(),
        )
    if not batch_path.is_dir():
        return []
    return sorted(
        (child for child in batch_path.iterdir() if is_match_dir(child)),
        key=lambda p: p.name.lower(),
    )


def _models_from_info(info: dict[str, str]) -> tuple[str, str, str, str]:
    """Return C/M/T/A model fields; Commander stores a single commander_model."""
    commander = info.get("commander_model", "").strip()
    coordinator = info.get("coordinator_model", "").strip() or commander
    macro = info.get("macro_model", "").strip() or commander
    translator = info.get("translator_model", "").strip() or commander
    army = info.get("army_model", "").strip() or commander
    return coordinator, macro, translator, army


def parse_match(match_dir: Path, batch_name: str) -> MatchRow:
    info_path = match_dir / "match_info.txt"
    info = parse_match_info(info_path)
    json_path = find_record_json(match_dir)
    match_id = info.get("match_id") or match_dir.name
    enemy, difficulty, enemy_build, enemy_race = parse_enemy_fields(
        info.get("enemy", "")
    )
    coordinator_model, macro_model, translator_model, army_model = _models_from_info(
        info
    )

    if json_path is None:
        return MatchRow(
            match_id=match_id,
            batch=batch_name,
            result="Missing",
            duration="?",
            map_name=info.get("map", "?"),
            matchup="?",
            strategy=info.get("force_strategy", "?") or "?",
            json_path="",
            info_path=str(info_path) if info_path.is_file() else "",
            enemy=enemy,
            difficulty=difficulty,
            enemy_build=enemy_build,
            enemy_race=enemy_race or "?",
            coordinator_model=coordinator_model,
            macro_model=macro_model,
            translator_model=translator_model,
            army_model=army_model,
            error="record json not found",
        )

    try:
        meta = load_metadata_only(json_path)
    except Exception as exc:  # noqa: BLE001 - surface per-match errors
        return MatchRow(
            match_id=match_id,
            batch=batch_name,
            result="Error",
            duration="?",
            map_name=info.get("map", "?"),
            matchup="?",
            strategy=info.get("force_strategy", "?") or "?",
            json_path=str(json_path),
            info_path=str(info_path) if info_path.is_file() else "",
            enemy=enemy,
            difficulty=difficulty,
            enemy_build=enemy_build,
            enemy_race=enemy_race or "?",
            coordinator_model=coordinator_model,
            macro_model=macro_model,
            translator_model=translator_model,
            army_model=army_model,
            error=f"{type(exc).__name__}: {exc}",
        )

    strategy = (
        info.get("force_strategy")
        or str(meta.get("strategy") or meta.get("force_strategy") or "")
        or "?"
    )
    matchup = str(meta.get("matchup") or "?")
    if not difficulty or not enemy_build or not enemy_race:
        opponent = str(meta.get("opponent_id") or "")
        # e.g. universal_llm.terran-ai.terran.veryeasy.macro
        parts = [p for p in opponent.split(".") if p]
        if len(parts) >= 2 and not difficulty:
            difficulty = parts[-2]
        if len(parts) >= 1 and not enemy_build:
            enemy_build = parts[-1]
        if not enemy and opponent:
            enemy = opponent
        if not enemy_race:
            for token in reversed(parts):
                mapped = _RACE_ALIASES.get(token.lower())
                if mapped:
                    enemy_race = mapped
                    break
    if not enemy_race:
        enemy_race = race_from_matchup(matchup) or "?"
    return MatchRow(
        match_id=match_id,
        batch=batch_name,
        result=str(meta.get("result") or "Unknown"),
        duration=str(meta.get("game_duration_formatted") or "?"),
        map_name=str(meta.get("map_name") or info.get("map") or "?"),
        matchup=matchup,
        strategy=strategy,
        json_path=str(json_path),
        info_path=str(info_path) if info_path.is_file() else "",
        enemy=enemy,
        difficulty=difficulty,
        enemy_build=enemy_build,
        enemy_race=enemy_race,
        coordinator_model=coordinator_model,
        macro_model=macro_model,
        translator_model=translator_model,
        army_model=army_model,
    )


def merge_batch_config(match_dirs: list[Path]) -> dict[str, str]:
    """Take the first complete match_info; drop per-match-only fields."""
    drop = {"match_id", "timestamp", "run_index", "record_dir"}
    for match_dir in match_dirs:
        info = parse_match_info(match_dir / "match_info.txt")
        if not info:
            continue
        return {k: v for k, v in info.items() if k not in drop}
    return {}


def summarize_batch(batch_path: Path, records_dir: Path) -> BatchSummary:
    batch_name = (
        "(ungrouped)" if batch_path.name == "__ungrouped__" else batch_path.name
    )
    match_dirs = iter_match_dirs(batch_path, records_dir)
    rows = [parse_match(match_dir, batch_name) for match_dir in match_dirs]
    counts: Counter[str] = Counter()
    wins = losses = ties = unknown = incomplete = 0
    for row in rows:
        counts[row.result] += 1
        bucket = classify_result(row.result)
        if row.error or bucket == "unknown" and row.result in {"Missing", "Error"}:
            incomplete += 1
        if bucket == "win":
            wins += 1
        elif bucket == "loss":
            losses += 1
        elif bucket == "tie":
            ties += 1
        else:
            unknown += 1

    display_path = (
        str(records_dir) if batch_path.name == "__ungrouped__" else str(batch_path)
    )
    return BatchSummary(
        batch=batch_name,
        path=display_path,
        total=len(rows),
        wins=wins,
        losses=losses,
        ties=ties,
        unknown=unknown,
        incomplete=incomplete,
        win_rate=win_rate(wins, losses, ties),
        config=merge_batch_config(match_dirs),
        result_counts=dict(sorted(counts.items())),
        matches=rows,
    )


def format_pct(rate: Optional[float]) -> str:
    if rate is None:
        return "-"
    return f"{rate * 100:.1f}%"


def group_key(row: MatchRow, group_by: str) -> str:
    if group_by == "batch":
        return row.batch or "?"
    if group_by == "difficulty":
        return row.difficulty or "?"
    if group_by == "strategy":
        return row.strategy or "?"
    if group_by == "enemy":
        return row.enemy or "?"
    if group_by == "race":
        return row.enemy_race or "?"
    if group_by == "map":
        return row.map_name or "?"
    if group_by == "matchup":
        return row.matchup or "?"
    return row.batch or "?"


def summarize_groups(
    rows: list[MatchRow], group_by: str
) -> list[tuple[str, int, int, int, int, int, Optional[float]]]:
    buckets: dict[str, list[MatchRow]] = defaultdict(list)
    for row in rows:
        buckets[group_key(row, group_by)].append(row)

    out: list[tuple[str, int, int, int, int, int, Optional[float]]] = []
    for key in sorted(buckets.keys(), key=_natural_sort_key):
        group_rows = buckets[key]
        wins = losses = ties = unknown = 0
        for row in group_rows:
            bucket = classify_result(row.result)
            if bucket == "win":
                wins += 1
            elif bucket == "loss":
                losses += 1
            elif bucket == "tie":
                ties += 1
            else:
                unknown += 1
        out.append(
            (
                key,
                len(group_rows),
                wins,
                losses,
                ties,
                unknown,
                win_rate(wins, losses, ties),
            )
        )
    return out


def _count_results(rows: list[MatchRow]) -> tuple[int, int, int, int, int, Optional[float]]:
    wins = losses = ties = unknown = 0
    for row in rows:
        bucket = classify_result(row.result)
        if bucket == "win":
            wins += 1
        elif bucket == "loss":
            losses += 1
        elif bucket == "tie":
            ties += 1
        else:
            unknown += 1
    return len(rows), wins, losses, ties, unknown, win_rate(wins, losses, ties)


_DIFFICULTY_ORDER = {
    "veryeasy": 0,
    "easy": 1,
    "medium": 2,
    "mediumhard": 3,
    "hard": 4,
    "harder": 5,
    "veryhard": 6,
    "cheatvision": 7,
    "cheatmoney": 8,
    "cheatinsane": 9,
}


def _difficulty_sort_key(name: str) -> tuple[int, str]:
    text = (name or "").strip().lower()
    return (_DIFFICULTY_ORDER.get(text, 100), text)


_NATURAL_CHUNK_RE = re.compile(r"(\d+)")


def _natural_sort_key(value: str) -> tuple:
    """Sort ``tank_opt2`` before ``tank_opt10`` (not lexicographic)."""
    text = str(value or "").strip().lower()
    parts: list[tuple[int, object]] = []
    for chunk in _NATURAL_CHUNK_RE.split(text):
        if not chunk:
            continue
        if chunk.isdigit():
            parts.append((1, int(chunk)))
        else:
            parts.append((0, chunk))
    return tuple(parts)


def print_strategy_model_summary(
    rows: list[MatchRow], *, title: str = "Summary by strategy + model"
) -> None:
    """Win rates grouped by strategy + model."""
    if not rows:
        return
    buckets: dict[tuple[str, str], list[MatchRow]] = defaultdict(list)
    for row in rows:
        strategy = (row.strategy or "?").strip() or "?"
        buckets[(strategy, row.model_key)].append(row)

    strat_w, model_w = 24, 22
    header = (
        f"{'strategy':<{strat_w}} {'model':<{model_w}} "
        f"{'n':>3} {'W':>3} {'L':>3} {'T':>3} {'win%':>7}"
    )
    print(title)
    print(header)
    print("-" * len(header))
    for strategy, model in sorted(
        buckets.keys(),
        key=lambda k: (_natural_sort_key(k[0]), k[1].lower()),
    ):
        total, wins, losses, ties, _unknown, rate = _count_results(
            buckets[(strategy, model)]
        )
        print(
            f"{_clip(strategy, strat_w):<{strat_w}} "
            f"{_clip(model, model_w):<{model_w}} "
            f"{total:>3} {wins:>3} {losses:>3} {ties:>3} "
            f"{format_pct(rate):>7}"
        )
    print()


def _batch_column_label(group_rows: list[MatchRow]) -> str:
    """Single batch → short id; multiple → ``N batches``."""
    names = sorted(
        {
            (row.batch or "?").strip() or "?"
            for row in group_rows
            if (row.batch or "").strip()
        }
    )
    if not names:
        return "?"
    if len(names) == 1:
        return _short_batch_name(names[0])
    return f"{len(names)} batches"


def print_batch_overview(
    rows: list[MatchRow], *, per_batch: bool = False
) -> None:
    """Table sorted by strategy / model / race / difficulty.

    By default, rows that share strategy + model + opponent race + difficulty +
    enemy build are merged across batch folders (W/L/T summed). Pass
    ``per_batch=True`` to keep one line per batch folder.

    Loose single-match folders under ``game_records/`` are excluded from the
    main table and printed in a separate section.
    """
    batch_rows = [row for row in rows if row.batch != "(ungrouped)"]
    ungrouped_rows = [row for row in rows if row.batch == "(ungrouped)"]

    buckets: dict[tuple[str, ...], list[MatchRow]] = defaultdict(list)
    for row in batch_rows:
        if per_batch:
            key: tuple[str, ...] = (row.batch or "?",)
        else:
            key = (
                (row.strategy or "?").strip() or "?",
                row.model_key,
                (row.enemy_race or "?").strip() or "?",
                (row.difficulty or "?").strip() or "?",
                (row.enemy_build or "?").strip() or "?",
            )
        buckets[key].append(row)

    # strategy | model | race | diff | build | batch
    strat_w, model_w, race_w, diff_w, build_w, batch_w = 18, 12, 8, 11, 8, 18
    header = (
        f"{'strategy':<{strat_w}} {'model':<{model_w}} {'race':<{race_w}} "
        f"{'diff':<{diff_w}} {'build':<{build_w}} {'batch':<{batch_w}} "
        f"{'n':>3} {'W':>3} {'L':>3} {'T':>3} {'win%':>6}"
    )
    print(header)
    print("-" * len(header))

    if not buckets:
        print("(no batch folders)")
    else:
        records: list[
            tuple[str, str, str, str, str, str, int, int, int, int, Optional[float]]
        ] = []
        for group_rows in buckets.values():
            total, wins, losses, ties, _unknown, rate = _count_results(group_rows)
            records.append(
                (
                    _majority([r.strategy for r in group_rows]),
                    _majority([r.model_key for r in group_rows]),
                    _majority([r.enemy_race for r in group_rows]),
                    _majority([r.difficulty for r in group_rows]),
                    _majority([r.enemy_build for r in group_rows]),
                    _batch_column_label(group_rows),
                    total,
                    wins,
                    losses,
                    ties,
                    rate,
                )
            )
        records.sort(
            key=lambda r: (
                _natural_sort_key(r[0]),
                r[1].lower(),
                r[2].lower(),
                _difficulty_sort_key(r[3]),
                r[4].lower(),
                r[5].lower(),
            )
        )

        prev_strategy = None
        prev_model = None
        prev_race = None
        for (
            strategy,
            model,
            enemy_race,
            difficulty,
            enemy_build,
            batch,
            total,
            wins,
            losses,
            ties,
            rate,
        ) in records:
            if prev_strategy is not None and (
                strategy != prev_strategy
                or model != prev_model
                or enemy_race != prev_race
            ):
                print()
            print(
                f"{_clip(strategy, strat_w):<{strat_w}} "
                f"{_clip(model, model_w):<{model_w}} "
                f"{_clip(enemy_race, race_w):<{race_w}} "
                f"{_clip(difficulty, diff_w):<{diff_w}} "
                f"{_clip(enemy_build, build_w):<{build_w}} "
                f"{_clip(batch, batch_w):<{batch_w}} "
                f"{total:>3} {wins:>3} {losses:>3} {ties:>3} "
                f"{format_pct(rate):>6}"
            )
            prev_strategy = strategy
            prev_model = model
            prev_race = enemy_race

    print("-" * len(header))
    print()
    print_strategy_model_summary(batch_rows)

    if ungrouped_rows:
        print(f"Single matches (ungrouped): {len(ungrouped_rows)}")
        print(
            f"{'match_id':<36} {'race':<8} {'diff':<11} {'build':<8} "
            f"{'strategy':<18} {'result':<8} {'model'}"
        )
        print("-" * 110)
        for row in sorted(
            ungrouped_rows,
            key=lambda r: (
                _natural_sort_key(r.strategy or ""),
                r.model_key.lower(),
                (r.enemy_race or "").lower(),
                _difficulty_sort_key(r.difficulty),
                r.match_id.lower(),
            ),
        ):
            print(
                f"{_clip(row.match_id, 36):<36} "
                f"{_clip(row.enemy_race or '-', 8):<8} "
                f"{_clip(row.difficulty or '-', 11):<11} "
                f"{_clip(row.enemy_build or '-', 8):<8} "
                f"{_clip(row.strategy or '-', 18):<18} "
                f"{_clip(row.result or '-', 8):<8} "
                f"{row.model_key}"
            )
        print()
        print_strategy_model_summary(
            ungrouped_rows, title="Ungrouped summary by strategy + model"
        )


def print_group_table(
    rows: list[MatchRow],
    group_by: str,
    *,
    title: Optional[str] = None,
    per_batch: bool = False,
) -> None:
    if group_by == "batch":
        print_batch_overview(rows, per_batch=per_batch)
        return
    groups = summarize_groups(rows, group_by)
    label = title or f"Win rate by {group_by}"
    print(label)
    print(f"{group_by:<40} {'n':>3} {'W':>3} {'L':>3} {'T':>3} {'?':>3} {'win%':>7}")
    print("-" * 72)
    tw = tl = tt = tu = tn = 0
    for key, total, wins, losses, ties, unknown, rate in groups:
        name = key if len(key) <= 40 else key[:37] + "..."
        print(
            f"{name:<40} {total:>3} {wins:>3} {losses:>3} "
            f"{ties:>3} {unknown:>3} {format_pct(rate):>7}"
        )
        tn += total
        tw += wins
        tl += losses
        tt += ties
        tu += unknown
    print("-" * 72)
    print(
        f"{'TOTAL':<40} {tn:>3} {tw:>3} {tl:>3} {tt:>3} {tu:>3} "
        f"{format_pct(win_rate(tw, tl, tt)):>7}"
    )
    print()


def print_batch_report(summary: BatchSummary, *, list_matches: bool) -> None:
    print("=" * 64)
    print(f"Batch: {summary.batch}")
    print(f"Path : {summary.path}")
    print("-" * 64)
    if summary.config:
        print("Config (shared for this batch):")
        preferred = [
            "bot",
            "enemy",
            "map",
            "force_strategy",
            "coordinator_model",
            "macro_model",
            "translator_model",
            "army_model",
            "batch_name",
            "bot_instruct",
        ]
        shown = set()
        for key in preferred:
            if key in summary.config:
                value = summary.config[key]
                if key == "bot_instruct" and "\n" in value:
                    print(f"  {key}:")
                    for line in value.splitlines():
                        print(f"    {line}")
                else:
                    print(f"  {key}: {value}")
                shown.add(key)
        for key, value in summary.config.items():
            if key not in shown:
                print(f"  {key}: {value}")
    else:
        print("Config: (no match_info.txt found; using JSON metadata only)")

    print("-" * 64)
    print(
        f"Matches: {summary.total}  "
        f"W {summary.wins} / L {summary.losses} / T {summary.ties} / "
        f"? {summary.unknown}  "
        f"win_rate={format_pct(summary.win_rate)}"
    )
    if summary.result_counts:
        detail = ", ".join(f"{k}={v}" for k, v in summary.result_counts.items())
        print(f"Results: {detail}")
    if summary.incomplete:
        print(f"Incomplete/error records: {summary.incomplete}")

    if list_matches and summary.matches:
        print("-" * 64)
        print(f"{'#':<3} {'result':<10} {'dur':<7} {'match_id'}")
        for index, row in enumerate(summary.matches):
            mark = row.result
            extra = f"  ERR={row.error}" if row.error else ""
            print(f"{index:<3} {mark:<10} {row.duration:<7} {row.match_id}{extra}")
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse game_records batches and report win rates."
    )
    parser.add_argument(
        "--records-dir",
        default="",
        help="Root records directory (default: <repo>/game_records)",
    )
    parser.add_argument(
        "--batch",
        action="append",
        default=[],
        help="Only summarize this batch folder name (repeatable)",
    )
    parser.add_argument(
        "--group-by",
        choices=GROUP_BY_CHOICES,
        default="batch",
        help="Primary overview grouping (default: batch)",
    )
    parser.add_argument(
        "--per-batch",
        action="store_true",
        help=(
            "Do not merge batches that share strategy/model/difficulty/build; "
            "print one overview row per batch folder (default merges them)."
        ),
    )
    parser.add_argument(
        "--list-matches",
        action="store_true",
        help="List each match result under every batch",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Also print per-batch config blocks after the overview table",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Write machine-readable summary JSON to this path",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    records_dir = (
        Path(args.records_dir).expanduser().resolve()
        if args.records_dir
        else (_project_root() / "game_records").resolve()
    )
    if not records_dir.is_dir():
        print(f"[ERROR] records dir not found: {records_dir}")
        return 1

    batch_paths = discover_batches(records_dir)
    if args.batch:
        wanted = {name.strip() for name in args.batch if name.strip()}
        batch_paths = [
            path
            for path in batch_paths
            if path.name in wanted
            or (path.name == "__ungrouped__" and "(ungrouped)" in wanted)
        ]
        # Allow direct path / exact folder under records_dir even if empty discovery.
        for name in wanted:
            candidate = Path(name).expanduser()
            if not candidate.is_absolute():
                candidate = records_dir / name
            candidate = candidate.resolve()
            if candidate.is_dir() and candidate not in batch_paths:
                batch_paths.append(candidate)

    if not batch_paths:
        print(f"No batches found under {records_dir}")
        return 0

    summaries = [summarize_batch(path, records_dir) for path in batch_paths]
    all_rows = [row for summary in summaries for row in summary.matches]

    print(f"Records root: {records_dir}")
    print(f"Batches: {len(summaries)}  Matches: {len(all_rows)}")
    print()
    print_group_table(all_rows, args.group_by, per_batch=args.per_batch)

    if args.detail or args.list_matches:
        for summary in summaries:
            print_batch_report(summary, list_matches=args.list_matches)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records_dir": str(records_dir),
            "group_by": args.group_by,
            "groups": [
                {
                    "key": key,
                    "total": total,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "unknown": unknown,
                    "win_rate": rate,
                }
                for key, total, wins, losses, ties, unknown, rate in summarize_groups(
                    all_rows, args.group_by
                )
            ],
            "batches": [asdict(summary) for summary in summaries],
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
