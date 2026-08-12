"""Summarize wall-clock vs in-game duration from SC2 match records.

Reads the same ``game_records/<batch>/<match_id>/`` layout as ``batch_stats.py``:

* ``{match_id}.json``  — metadata has ``game_duration_seconds`` and ``result``
* ``match_info.txt``   — has ``timestamp`` (start real time) and shared config

Computes ``real_seconds`` (wall-clock match duration) from each match ``.log``:
first→last timestamped line. That includes accelerated in-game simulation time
plus LLM decision pauses — not decision latency alone. Falls back to
``match_info`` timestamp → JSON mtime when the log is missing.

Examples::

    .\\tools\\show_time.ps1
    python tools/batch_time_stats.py
    python tools/batch_time_stats.py --per-batch
    python tools/batch_time_stats.py --group-by strategy
    python tools/batch_time_stats.py --list-matches
    python tools/batch_time_stats.py --json-out time_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)")
_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Support both ``python tools/batch_time_stats.py`` and ``python -m tools.batch_time_stats``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.batch_stats import (
    _MODEL_SHORT,
    _clip,
    _difficulty_sort_key,
    _majority,
    _short_batch_name,
    classify_result,
    discover_batches,
    find_record_json,
    format_pct,
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
    enemy_build: str
    enemy_race: str
    model_key: str
    start_time: Optional[str]
    json_path: str
    info_path: str


@dataclass
class TimeSummary:
    group: str
    strategy: str = "?"
    model_key: str = "?"
    enemy_race: str = "?"
    difficulty: str = "?"
    enemy_build: str = "?"
    batch_label: str = "?"
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
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_log_timestamp(value: str) -> Optional[datetime]:
    text = (value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_match_info_timestamp(info: dict[str, str]) -> Optional[datetime]:
    return _parse_timestamp(info.get("timestamp") or "")


def _find_match_log(match_dir: Path, match_id: str, json_path: Path) -> Optional[Path]:
    candidates = [
        match_dir / f"{match_id}.log",
        json_path.with_suffix(".log"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    logs = sorted(match_dir.glob("*.log"))
    return logs[0] if logs else None


def _wall_seconds_from_log(log_path: Path) -> Optional[float]:
    """Wall-clock span covering accelerated game sim + LLM pauses."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    first: Optional[datetime] = None
    last: Optional[datetime] = None
    for line in text.splitlines():
        if "\x1b" in line:
            line = _ANSI_ESCAPE_RE.sub("", line)
        match = _LOG_TS_RE.match(line)
        if not match:
            continue
        stamp = _parse_log_timestamp(match.group(1))
        if stamp is None:
            continue
        if first is None:
            first = stamp
        last = stamp
    if first is None or last is None or last <= first:
        return None
    return (last - first).total_seconds()


def _wall_seconds_fallback(row: TimeRow) -> Optional[float]:
    """match_info start → JSON mtime when log timestamps are unavailable."""
    start = None
    if row.info_path:
        start = _parse_match_info_timestamp(parse_match_info(Path(row.info_path)))
    if start is None and row.start_time:
        start = _parse_timestamp(row.start_time)
    if start is None:
        return None
    try:
        end = datetime.fromtimestamp(os.path.getmtime(Path(row.json_path)))
    except OSError:
        return None
    if end <= start:
        return None
    return (end - start).total_seconds()


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
    text = (name or "").strip()
    if not text:
        return "-"
    return _MODEL_SHORT.get(text.lower(), text)


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

    enemy, difficulty, enemy_build, enemy_race = parse_enemy_fields(
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
        enemy_build=enemy_build or "?",
        enemy_race=enemy_race or "?",
        model_key=_resolve_model_key(info),
        start_time=info.get("timestamp"),
        json_path=str(json_path),
        info_path=str(info_path) if info_path.is_file() else "",
    )


def compute_real_times(rows: list[TimeRow]) -> list[TimeRow]:
    """Assign per-match wall-clock seconds (accel game + decisions + overhead)."""
    for row in rows:
        match_dir = Path(row.json_path).parent
        log_path = _find_match_log(match_dir, row.match_id, Path(row.json_path))
        wall = _wall_seconds_from_log(log_path) if log_path else None
        if wall is None:
            wall = _wall_seconds_fallback(row)
        row.real_seconds = wall
        if wall is not None and row.game_seconds > 0:
            row.ratio = wall / row.game_seconds
        else:
            row.ratio = None
    return rows


def _result_counts(rows: list[TimeRow]) -> tuple[int, int, int, int, int, Optional[float]]:
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
    total = len(rows)
    return total, wins, losses, ties, unknown, win_rate(wins, losses, ties)


def _time_stats(rows: list[TimeRow]) -> dict[str, Optional[float]]:
    game_times = [r.game_seconds for r in rows if r.game_seconds > 0]
    real_times = [r.real_seconds for r in rows if r.real_seconds is not None]
    ratios = [r.ratio for r in rows if r.ratio is not None]
    return {
        "avg_game_seconds": _safe_mean(game_times),
        "avg_real_seconds": _safe_mean(real_times),
        "avg_ratio": _safe_mean(ratios),
        "min_game_seconds": _safe_min(game_times),
        "max_game_seconds": _safe_max(game_times),
        "min_real_seconds": _safe_min(real_times),
        "max_real_seconds": _safe_max(real_times),
    }


def _batch_column_label(group_rows: list[TimeRow]) -> str:
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


def _make_summary(group: str, rows: list[TimeRow]) -> TimeSummary:
    total, wins, losses, ties, unknown, _rate = _result_counts(rows)
    stats = _time_stats(rows)
    return TimeSummary(
        group=group,
        strategy=_majority([r.strategy for r in rows]),
        model_key=_majority([r.model_key for r in rows]),
        enemy_race=_majority([r.enemy_race for r in rows]),
        difficulty=_majority([r.difficulty for r in rows]),
        enemy_build=_majority([r.enemy_build for r in rows]),
        batch_label=_batch_column_label(rows),
        total=total,
        wins=wins,
        losses=losses,
        ties=ties,
        unknown=unknown,
        matches=list(rows),
        **stats,
    )


def build_overview_summaries(
    rows: list[TimeRow], *, per_batch: bool
) -> list[TimeSummary]:
    """Same merge keys as ``batch_stats.print_batch_overview``."""
    batch_rows = [row for row in rows if row.batch != "(ungrouped)"]
    buckets: dict[tuple[str, ...], list[TimeRow]] = defaultdict(list)
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

    summaries = [_make_summary("|".join(key), group_rows) for key, group_rows in buckets.items()]
    summaries.sort(
        key=lambda s: (
            s.strategy.lower(),
            s.model_key.lower(),
            s.enemy_race.lower(),
            _difficulty_sort_key(s.difficulty),
            s.enemy_build.lower(),
            s.batch_label.lower(),
        )
    )
    return summaries


def build_group_summaries(rows: list[TimeRow], group_by: str) -> list[TimeSummary]:
    buckets: dict[str, list[TimeRow]] = defaultdict(list)
    for row in rows:
        if group_by == "strategy":
            key = row.strategy
        elif group_by == "model":
            key = row.model_key
        elif group_by == "difficulty":
            key = row.difficulty
        elif group_by == "enemy":
            key = row.enemy
        elif group_by == "enemy_race":
            key = row.enemy_race
        elif group_by == "map":
            key = row.map_name
        elif group_by == "matchup":
            key = row.matchup
        else:
            key = row.batch
        buckets[key or "?"].append(row)
    return [_make_summary(key, buckets[key]) for key in sorted(buckets, key=str.lower)]


def _print_overview(rows: list[TimeRow], *, per_batch: bool) -> None:
    batch_rows = [row for row in rows if row.batch != "(ungrouped)"]
    ungrouped_rows = [row for row in rows if row.batch == "(ungrouped)"]
    summaries = build_overview_summaries(rows, per_batch=per_batch)

    strat_w, model_w, race_w, diff_w, build_w, batch_w = 18, 12, 8, 11, 8, 18
    header = (
        f"{'strategy':<{strat_w}} {'model':<{model_w}} {'race':<{race_w}} "
        f"{'diff':<{diff_w}} {'build':<{build_w}} {'batch':<{batch_w}} "
        f"{'n':>3} {'W':>3} {'L':>3} {'T':>3} {'win%':>6} "
        f"{'avgGame':>8} {'avgReal':>8}"
    )
    print(header)
    print("-" * len(header))

    if not summaries:
        print("(no batch folders)")
    else:
        prev_strategy = prev_model = prev_race = None
        for s in summaries:
            if prev_strategy is not None and (
                s.strategy != prev_strategy
                or s.model_key != prev_model
                or s.enemy_race != prev_race
            ):
                print()
            rate = win_rate(s.wins, s.losses, s.ties)
            print(
                f"{_clip(s.strategy, strat_w):<{strat_w}} "
                f"{_clip(s.model_key, model_w):<{model_w}} "
                f"{_clip(s.enemy_race, race_w):<{race_w}} "
                f"{_clip(s.difficulty, diff_w):<{diff_w}} "
                f"{_clip(s.enemy_build, build_w):<{build_w}} "
                f"{_clip(s.batch_label, batch_w):<{batch_w}} "
                f"{s.total:>3} {s.wins:>3} {s.losses:>3} {s.ties:>3} "
                f"{format_pct(rate):>6} "
                f"{_fmt_seconds(s.avg_game_seconds):>8} "
                f"{_fmt_seconds(s.avg_real_seconds):>8}"
            )
            prev_strategy = s.strategy
            prev_model = s.model_key
            prev_race = s.enemy_race

    print("-" * len(header))
    print()
    _print_strategy_model_summary(batch_rows)

    if ungrouped_rows:
        print(f"Single matches (ungrouped): {len(ungrouped_rows)}")
        print(
            f"{'match_id':<36} {'race':<8} {'diff':<11} {'build':<8} "
            f"{'strategy':<18} {'result':<8} {'game':>8} {'model'}"
        )
        print("-" * 120)
        for row in sorted(
            ungrouped_rows,
            key=lambda r: (
                (r.strategy or "").lower(),
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
                f"{_fmt_seconds(row.game_seconds):>8} "
                f"{row.model_key}"
            )
        print()
        _print_strategy_model_summary(
            ungrouped_rows, title="Ungrouped summary by strategy + model"
        )

    print(
        "avgGame = mean in-game duration. "
        "avgReal = mean wall-clock (log first→last; includes accelerated sim + LLM)."
    )


def _print_strategy_model_summary(
    rows: list[TimeRow], *, title: str = "Summary by strategy + model"
) -> None:
    if not rows:
        return
    buckets: dict[tuple[str, str], list[TimeRow]] = defaultdict(list)
    for row in rows:
        strategy = (row.strategy or "?").strip() or "?"
        buckets[(strategy, row.model_key)].append(row)

    strat_w, model_w = 24, 22
    header = (
        f"{'strategy':<{strat_w}} {'model':<{model_w}} "
        f"{'n':>3} {'W':>3} {'L':>3} {'T':>3} {'win%':>7} {'avgGame':>8}"
    )
    print(title)
    print(header)
    print("-" * len(header))
    for strategy, model in sorted(
        buckets.keys(), key=lambda k: (k[0].lower(), k[1].lower())
    ):
        group_rows = buckets[(strategy, model)]
        total, wins, losses, ties, _unknown, rate = _result_counts(group_rows)
        stats = _time_stats(group_rows)
        print(
            f"{_clip(strategy, strat_w):<{strat_w}} "
            f"{_clip(model, model_w):<{model_w}} "
            f"{total:>3} {wins:>3} {losses:>3} {ties:>3} "
            f"{format_pct(rate):>7} "
            f"{_fmt_seconds(stats['avg_game_seconds']):>8}"
        )
    print()


def _print_group_table(summaries: list[TimeSummary], group_by: str) -> None:
    print(f"Game time by {group_by}")
    print(
        f"{group_by:<40} {'n':>3} {'W':>3} {'L':>3} {'T':>3} "
        f"{'win%':>7} {'avgGame':>8} {'avgReal':>8}"
    )
    print("-" * 90)
    tw = tl = tt = tn = 0
    game_all: list[float] = []
    real_all: list[float] = []
    for s in summaries:
        rate = win_rate(s.wins, s.losses, s.ties)
        name = s.group if len(s.group) <= 40 else s.group[:37] + "..."
        print(
            f"{name:<40} {s.total:>3} {s.wins:>3} {s.losses:>3} {s.ties:>3} "
            f"{format_pct(rate):>7} "
            f"{_fmt_seconds(s.avg_game_seconds):>8} "
            f"{_fmt_seconds(s.avg_real_seconds):>8}"
        )
        tn += s.total
        tw += s.wins
        tl += s.losses
        tt += s.ties
        game_all.extend(r.game_seconds for r in s.matches if r.game_seconds > 0)
        real_all.extend(
            r.real_seconds for r in s.matches if r.real_seconds is not None
        )
    print("-" * 90)
    print(
        f"{'TOTAL':<40} {tn:>3} {tw:>3} {tl:>3} {tt:>3} "
        f"{format_pct(win_rate(tw, tl, tt)):>7} "
        f"{_fmt_seconds(_safe_mean(game_all)):>8} "
        f"{_fmt_seconds(_safe_mean(real_all)):>8}"
    )
    print()


def _print_matches(summaries: list[TimeSummary]) -> None:
    for s in summaries:
        label = s.group if s.group else f"{s.strategy}/{s.model_key}"
        print(f"\n=== {label} ({s.total} matches) ===")
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
        help="Primary overview grouping (default: batch = winrate-style table)",
    )
    parser.add_argument(
        "--per-batch",
        action="store_true",
        help="Keep one overview row per batch folder (default merges them)",
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
        print(f"[ERROR] records dir not found: {records_dir}")
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
    print(f"Records root: {records_dir}")
    print(f"Batches: {len(batch_paths)}  Matches: {len(all_rows)}")
    print()

    if args.group_by == "batch":
        summaries = build_overview_summaries(all_rows, per_batch=args.per_batch)
        if args.list_matches:
            _print_matches(summaries)
        else:
            _print_overview(all_rows, per_batch=args.per_batch)
    else:
        summaries = build_group_summaries(all_rows, args.group_by)
        if args.list_matches:
            _print_matches(summaries)
        else:
            _print_group_table(summaries, args.group_by)

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
        print(f"Wrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
