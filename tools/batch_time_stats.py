"""Summarize wall-clock vs in-game duration from SC2 match records.

Reads the same ``game_records/<batch>/<match_id>/`` layout as ``batch_stats.py``:

* ``{match_id}.json``  — metadata has ``game_duration_seconds`` and ``result``
* ``match_info.txt``   — has ``timestamp`` (start real time) and shared config

Computes ``real_seconds`` for each match by looking at the next match's start
within the same batch; the last match uses the file modification time as the
fallback end. This gives a good approximation of how much wall-clock time the
whole match consumed, including LLM latency and game-time waits.

Examples::

    python tools/batch_time_stats.py
    python tools/batch_time_stats.py --batch batch_20260806_135020_e3_KairosJunctionL
    python tools/batch_time_stats.py --group-by strategy
    python tools/batch_time_stats.py --group-by model
    python tools/batch_time_stats.py --group-by difficulty
    python tools/batch_time_stats.py --per-batch
    python tools/batch_time_stats.py --list-matches
    python tools/batch_time_stats.py --json-out time_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tools.batch_stats import (
    SKIP_DIR_NAMES,
    classify_result,
    discover_batches,
    find_record_json,
    is_match_dir,
    iter_match_dirs,
    load_metadata_only,
    parse_enemy_fields,
    parse_match_info,
    win_rate,
)


@dataclass
class TimeRow:
    match_id: str
    batch: str
    result: str
    game_seconds: float
    real_seconds: Optional[float]
    ratio: Optional[float]
    map_name: str
    matchup: str
    strategy: str
    enemy: str
    difficulty: str
    enemy_race: str
    model_key: str
    start_time: Optional[str]
    json_path: str
    info_path: str


@dataclass
class TimeSummary:
    group: str
    total: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    unknown: int = 0
    avg_game_seconds: Optional[float] = None
    avg_real_seconds: Optional[float] = None
    avg_ratio: Optional[float] = None
    min_game_seconds: Optional[float] = None
    max_game_seconds: Optional[float] = None
    min_real_seconds: Optional[float] = None
    max_real_seconds: Optional[float] = None
    matches: list[TimeRow] = field(default_factory=list)


def _parse_timestamp(value: str) -> Optional[datetime]:
    text = (value or "").strip()
    if not text:
        return None
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_match_info_timestamp(info: dict[str, str]) -> Optional[datetime]:
    return _parse_timestamp(info.get("timestamp") or "")


def _fmt_seconds(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _safe_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_min(values: list[float]) -> Optional[float]:
    return min(values) if values else None


def _safe_max(values: list[float]) -> Optional[float]:
    return max(values) if values else None


def _short_model(name: str) -> str:
    short = {
        "kimi-k2.5": "kimi",
        "qwen3-32b": "qwen",
        "deepseek-v4-flash": "ds-flash",
        "deepseek-v4-pro": "ds-pro",
    }
    text = (name or "").strip()
    return short.get(text.lower(), text) if text else "-"


def _extract_config(row: dict[str, str]) -> dict[str, str]:
    drop = {"match_id", "timestamp", "run_index", "record_dir", "bot_instruct"}
    return {k: v for k, v in row.items() if k not in drop}


def _resolve_model_key(info: dict[str, str]) -> str:
    text = info.get("commander_model", "").strip()
    return _short_model(text)


def _resolve_strategy(info: dict[str, str], meta: dict[str, Any]) -> str:
    return (
        info.get("force_strategy")
        or str(meta.get("strategy") or meta.get("force_strategy") or "")
        or "?"
    )


def parse_match_time(match_dir: Path, batch_name: str) -> Optional[TimeRow]:
    info_path = match_dir / "match_info.txt"
    info = parse_match_info(info_path)
    json_path = find_record_json(match_dir)
    if json_path is None:
        return None

    try:
        meta = load_metadata_only(json_path)
    except Exception:
        return None

    enemy, difficulty, _enemy_build, enemy_race = parse_enemy_fields(
        info.get("enemy", "")
    )
    return TimeRow(
        match_id=info.get("match_id") or match_dir.name,
        batch=batch_name,
        result=str(meta.get("result") or "Unknown"),
        game_seconds=float(meta.get("game_duration_seconds") or 0),
        real_seconds=None,
        ratio=None,
        map_name=str(meta.get("map_name") or info.get("map") or "?"),
        matchup=str(meta.get("matchup") or "?"),
        strategy=_resolve_strategy(info, meta),
        enemy=enemy,
        difficulty=difficulty or "?",
        enemy_race=enemy_race or "?",
        model_key=_resolve_model_key(info),
        start_time=info.get("timestamp"),
        json_path=str(json_path),
        info_path=str(info_path) if info_path.is_file() else "",
    )


def compute_real_times(rows: list[TimeRow]) -> list[TimeRow]:
    """Assign real_seconds by looking at next match start in the same batch.

    The last match in a batch falls back to the record JSON modification time.
    """
    by_batch: dict[str, list[TimeRow]] = defaultdict(list)
    for row in rows:
        by_batch[row.batch].append(row)

    for batch, batch_rows in by_batch.items():
        starts: list[tuple[Optional[datetime], TimeRow]] = []
        for row in batch_rows:
            info = parse_match_info(Path(row.info_path)) if row.info_path else {}
            starts.append((_parse_match_info_timestamp(info), row))
        # Sort by start time; if missing, keep directory order.
        sorted_rows = sorted(
            batch_rows,
            key=lambda r: (_parse_match_info_timestamp(
                parse_match_info(Path(r.info_path)) if r.info_path else {}
            ) or datetime.min.replace(tzinfo=timezone.utc), r.match_id),
        )
        for i, row in enumerate(sorted_rows):
            start = _parse_match_info_timestamp(
                parse_match_info(Path(row.info_path)) if row.info_path else {}
            )
            end: Optional[datetime] = None
            if i + 1 < len(sorted_rows):
                end = _parse_match_info_timestamp(
                    parse_match_info(Path(sorted_rows[i + 1].info_path))
                    if sorted_rows[i + 1].info_path
                    else {}
                )
            else:
                try:
                    mtime = os.path.getmtime(Path(row.json_path))
                    end = datetime.fromtimestamp(mtime, tz=timezone.utc)
                except OSError:
                    end = None
            if start and end and end > start:
                row.real_seconds = (end - start).total_seconds()
                row.ratio = (
                    row.real_seconds / row.game_seconds
                    if row.game_seconds > 0
                    else None
                )
    return rows


def group_key(row: TimeRow, group_by: str) -> str:
    if group_by == "batch":
        return row.batch
    if group_by == "strategy":
        return row.strategy
    if group_by == "model":
        return row.model_key
    if group_by == "difficulty":
        return row.difficulty
    if group_by == "enemy":
        return row.enemy
    if group_by == "enemy_race":
        return row.enemy_race
    if group_by == "map":
        return row.map_name
    if group_by == "matchup":
        return row.matchup
    return row.batch


def build_summaries(
    rows: list[TimeRow],
    group_by: str,
    per_batch: bool,
) -> list[TimeSummary]:
    buckets: dict[str, list[TimeRow]] = defaultdict(list)
    for row in rows:
        key = group_key(row, group_by)
        buckets[key].append(row)

    summaries: list[TimeSummary] = []
    for key in sorted(buckets):
        bucket_rows = buckets[key]
        counts = defaultdict(int)
        wins = losses = ties = unknown = 0
        for row in bucket_rows:
            counts[row.result] += 1
            bucket = classify_result(row.result)
            if bucket == "win":
                wins += 1
            elif bucket == "loss":
                losses += 1
            elif bucket == "tie":
                ties += 1
            else:
                unknown += 1

        game_times = [r.game_seconds for r in bucket_rows if r.game_seconds > 0]
        real_times = [r.real_seconds for r in bucket_rows if r.real_seconds is not None]
        ratios = [r.ratio for r in bucket_rows if r.ratio is not None]

        summaries.append(
            TimeSummary(
                group=key,
                total=len(bucket_rows),
                wins=wins,
                losses=losses,
                ties=ties,
                unknown=unknown,
                avg_game_seconds=_safe_mean(game_times),
                avg_real_seconds=_safe_mean(real_times),
                avg_ratio=_safe_mean(ratios),
                min_game_seconds=_safe_min(game_times),
                max_game_seconds=_safe_max(game_times),
                min_real_seconds=_safe_min(real_times),
                max_real_seconds=_safe_max(real_times),
                matches=list(bucket_rows),
            )
        )
    return summaries


def _clip(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _print_table(summaries: list[TimeSummary]) -> None:
    header = (
        f"{'Group':<22} {'N':>3} {'Win':>4} {'Loss':>4} {'Tie':>3} "
        f"{'AvgGame':>8} {'AvgReal':>8} {'Ratio':>6} "
        f"{'GameRange':>18} {'RealRange':>18}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        wr = win_rate(s.wins, s.losses, s.ties)
        wr_text = f"{wr * 100:.0f}%" if wr is not None else "-"
        game_range = (
            f"{_fmt_seconds(s.min_game_seconds)}-{_fmt_seconds(s.max_game_seconds)}"
            if s.min_game_seconds is not None
            else "?"
        )
        real_range = (
            f"{_fmt_seconds(s.min_real_seconds)}-{_fmt_seconds(s.max_real_seconds)}"
            if s.min_real_seconds is not None
            else "?"
        )
        avg_game = _fmt_seconds(s.avg_game_seconds)
        avg_real = _fmt_seconds(s.avg_real_seconds)
        ratio = (
            f"{s.avg_ratio:.2f}"
            if s.avg_ratio is not None
            else "-"
        )
        print(
            f"{_clip(s.group, 22):<22} "
            f"{s.total:>3} {s.wins:>4} {s.losses:>4} {s.ties:>3} "
            f"{avg_game:>8} {avg_real:>8} {ratio:>6} "
            f"{game_range:>18} {real_range:>18}"
        )
    print(f"\nRatio = real wall-clock seconds / game seconds (lower is faster).")
    print(f"Real time estimated from next-match start; last match uses file mtime.")


def _print_matches(summaries: list[TimeSummary]) -> None:
    for s in summaries:
        print(f"\n=== {s.group} ({s.total} matches) ===")
        for row in sorted(s.matches, key=lambda r: r.start_time or ""):
            real = _fmt_seconds(row.real_seconds)
            game = _fmt_seconds(row.game_seconds)
            ratio = f"{row.ratio:.2f}" if row.ratio is not None else "-"
            print(
                f"  {row.match_id:<40} start={row.start_time or '?:?':<16} "
                f"result={_clip(row.result, 8):<8} game={game:>8} "
                f"real={real:>8} ratio={ratio:>6} "
                f"{row.strategy}/{row.model_key}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare wall-clock time and in-game duration from records."
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "game_records",
        help="game_records directory (default: ./game_records)",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default="",
        help="Only one batch folder (name only, no path)",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=(
            "batch",
            "strategy",
            "model",
            "difficulty",
            "enemy",
            "enemy_race",
            "map",
            "matchup",
        ),
        default="batch",
        help="Aggregate key (default: batch)",
    )
    parser.add_argument(
        "--per-batch",
        action="store_true",
        help="Keep one row per batch instead of merging across batches",
    )
    parser.add_argument(
        "--list-matches",
        action="store_true",
        help="Print every match in each group",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write machine-readable summary to this file",
    )
    args = parser.parse_args()

    records_dir = Path(args.records)
    if not records_dir.is_dir():
        print(f"Records directory not found: {records_dir}")
        return

    if args.batch:
        batch_paths = [records_dir / args.batch]
    else:
        batch_paths = discover_batches(records_dir)

    all_rows: list[TimeRow] = []
    for batch_path in batch_paths:
        batch_name = (
            "(ungrouped)" if batch_path.name == "__ungrouped__" else batch_path.name
        )
        for match_dir in iter_match_dirs(batch_path, records_dir):
            row = parse_match_time(match_dir, batch_name)
            if row:
                all_rows.append(row)

    all_rows = compute_real_times(all_rows)
    group_by = "batch" if args.per_batch else args.group_by
    summaries = build_summaries(all_rows, group_by, per_batch=args.per_batch)

    if args.list_matches:
        _print_matches(summaries)
    else:
        _print_table(summaries)

    if args.json_out:
        payload = [
            {
                **asdict(s),
                "matches": [asdict(r) for r in s.matches],
            }
            for s in summaries
        ]
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nWrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
