"""Copy match JSON files into a clean hierarchy by experiment axes.

Layout produced under ``--out``::

    <model>/<strategy>/<enemy_race>/<difficulty>/<enemy_build>/<match_id>.json

Only ``*.json`` record files are copied (logs / replays / match_info stay put).
By default only groups with at least ``--min-matches`` (20) JSON records are
exported — matching a completed matrix cell of 20 games. Also writes
``winrate.csv`` and ``winrate.md`` under ``--out``.

Examples::

    python tools/organize_batch_json.py
    python tools/organize_batch_json.py --dry-run
    python tools/organize_batch_json.py --min-matches 1 --out game_records_json
    python tools/organize_batch_json.py --batch batch_20260812_133124_e2_KairosJunctionL
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.batch_stats import (  # noqa: E402
    classify_result,
    discover_batches,
    find_record_json,
    format_pct,
    iter_match_dirs,
    load_metadata_only,
    parse_match,
    parse_match_info,
    win_rate,
)

_SAFE_RE = re.compile(r"[^A-Za-z0-9._+-]+")


@dataclass(frozen=True)
class GroupKey:
    model: str
    strategy: str
    race: str
    difficulty: str
    style: str

    def parts(self) -> tuple[str, str, str, str, str]:
        return (
            _safe_name(self.model),
            _safe_name(self.strategy),
            _safe_name(self.race),
            _safe_name(self.difficulty),
            _safe_name(self.style),
        )


@dataclass
class MatchJson:
    match_id: str
    batch: str
    json_path: Path
    group: GroupKey
    result: str


def _project_root() -> Path:
    return _REPO_ROOT


def _safe_name(value: str, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip() or fallback
    text = text.replace(" ", "_")
    text = _SAFE_RE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or fallback


def _first_nonempty(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text and text not in {"?", "-", "none", "None"}:
            return text
    return ""


def _resolve_model(match_dir: Path, row_model_key: str) -> str:
    info = parse_match_info(match_dir / "match_info.txt")
    json_path = find_record_json(match_dir)
    meta: dict = {}
    if json_path is not None:
        try:
            meta = load_metadata_only(json_path)
        except Exception:  # noqa: BLE001
            meta = {}
    return _first_nonempty(
        info.get("commander_model"),
        meta.get("commander_model_key"),
        meta.get("commander_model"),
        info.get("coordinator_model"),
        row_model_key,
        "unknown",
    )


def _resolve_group(match_dir: Path, batch_name: str) -> Optional[MatchJson]:
    json_path = find_record_json(match_dir)
    if json_path is None:
        return None
    row = parse_match(match_dir, batch_name)
    if not row.json_path:
        return None

    info = parse_match_info(match_dir / "match_info.txt")
    try:
        meta = load_metadata_only(json_path)
    except Exception:  # noqa: BLE001
        meta = {}

    model = _resolve_model(match_dir, row.model_key)
    strategy = _first_nonempty(
        info.get("force_strategy"),
        meta.get("strategy_id"),
        meta.get("strategy"),
        row.strategy,
        "unknown",
    )
    race = _first_nonempty(
        row.enemy_race,
        meta.get("enemy_race"),
        "unknown",
    ).lower()
    difficulty = _first_nonempty(row.difficulty, "unknown").lower()
    style = _first_nonempty(row.enemy_build, "unknown").lower()

    return MatchJson(
        match_id=_safe_name(row.match_id or match_dir.name),
        batch=batch_name,
        json_path=Path(row.json_path),
        group=GroupKey(
            model=model,
            strategy=strategy,
            race=race,
            difficulty=difficulty,
            style=style,
        ),
        result=str(row.result or meta.get("result") or ""),
    )


def collect_matches(
    records_dir: Path,
    *,
    batch_filter: Optional[set[str]] = None,
) -> list[MatchJson]:
    items: list[MatchJson] = []
    for batch_path in discover_batches(records_dir):
        batch_name = (
            "__ungrouped__"
            if batch_path.name == "__ungrouped__"
            else batch_path.name
        )
        if batch_filter and batch_name not in batch_filter:
            continue
        for match_dir in iter_match_dirs(batch_path, records_dir):
            item = _resolve_group(match_dir, batch_name)
            if item is not None:
                items.append(item)
    return items


def _unique_dest(dest_dir: Path, match_id: str) -> Path:
    candidate = dest_dir / f"{match_id}.json"
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = dest_dir / f"{match_id}_{index}.json"
        if not candidate.exists():
            return candidate
        index += 1


def _result_counts(items: list[MatchJson]) -> dict[str, object]:
    wins = losses = ties = unknown = 0
    for item in items:
        kind = classify_result(item.result)
        if kind == "win":
            wins += 1
        elif kind == "loss":
            losses += 1
        elif kind == "tie":
            ties += 1
        else:
            unknown += 1
    rate = win_rate(wins, losses, ties)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "unknown": unknown,
        "decided": wins + losses + ties,
        "win_rate": rate,
        "win_rate_pct": format_pct(rate),
    }


def _csv_escape(value: object) -> str:
    text = str(value)
    if any(ch in text for ch in (",", '"', "\n", "\r")):
        return '"' + text.replace('"', '""') + '"'
    return text


def write_winrate_csv(path: Path, groups: list[dict]) -> None:
    headers = [
        "model",
        "strategy",
        "race",
        "difficulty",
        "style",
        "games",
        "wins",
        "losses",
        "ties",
        "unknown",
        "win_rate",
        "win_rate_pct",
        "path",
        "batches",
    ]
    lines = [",".join(headers)]
    for group in groups:
        rate = group.get("win_rate")
        rate_text = "" if rate is None else f"{float(rate):.6f}"
        lines.append(
            ",".join(
                _csv_escape(v)
                for v in (
                    group.get("model", ""),
                    group.get("strategy", ""),
                    group.get("race", ""),
                    group.get("difficulty", ""),
                    group.get("style", ""),
                    group.get("count", 0),
                    group.get("wins", 0),
                    group.get("losses", 0),
                    group.get("ties", 0),
                    group.get("unknown", 0),
                    rate_text,
                    group.get("win_rate_pct", "-"),
                    group.get("path", ""),
                    ";".join(group.get("batches") or []),
                )
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_winrate_md(path: Path, groups: list[dict], *, min_matches: int) -> None:
    lines = [
        "# Batch win rates",
        "",
        f"Groups with at least **{min_matches}** JSON matches "
        "(model / strategy / race / difficulty / style).",
        "",
        "| Model | Strategy | Race | Difficulty | Style | Games | W | L | T | ? | WinRate |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        lines.append(
            "| {model} | {strategy} | {race} | {difficulty} | {style} | "
            "{count} | {wins} | {losses} | {ties} | {unknown} | {win_rate_pct} |".format(
                model=group.get("model", ""),
                strategy=group.get("strategy", ""),
                race=group.get("race", ""),
                difficulty=group.get("difficulty", ""),
                style=group.get("style", ""),
                count=group.get("count", 0),
                wins=group.get("wins", 0),
                losses=group.get("losses", 0),
                ties=group.get("ties", 0),
                unknown=group.get("unknown", 0),
                win_rate_pct=group.get("win_rate_pct", "-"),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def organize(
    records_dir: Path,
    out_dir: Path,
    *,
    min_matches: int = 20,
    dry_run: bool = False,
    batch_filter: Optional[set[str]] = None,
) -> dict:
    matches = collect_matches(records_dir, batch_filter=batch_filter)
    grouped: dict[GroupKey, list[MatchJson]] = defaultdict(list)
    for item in matches:
        grouped[item.group].append(item)

    selected: dict[GroupKey, list[MatchJson]] = {}
    skipped: list[dict] = []
    for key, items in sorted(
        grouped.items(),
        key=lambda kv: kv[0].parts(),
    ):
        # Prefer unique match_ids (re-runs with same id keep the first path).
        by_id: dict[str, MatchJson] = {}
        for item in items:
            by_id.setdefault(item.match_id, item)
        unique_items = list(by_id.values())
        if len(unique_items) < min_matches:
            skipped.append(
                {
                    "model": key.model,
                    "strategy": key.strategy,
                    "race": key.race,
                    "difficulty": key.difficulty,
                    "style": key.style,
                    "count": len(unique_items),
                    "reason": f"below min_matches={min_matches}",
                }
            )
            continue
        selected[key] = unique_items

    copied = 0
    group_summaries: list[dict] = []
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for key, items in selected.items():
        rel = Path(*key.parts())
        dest_group = out_dir / rel
        if not dry_run:
            dest_group.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        for item in sorted(items, key=lambda m: m.match_id.lower()):
            dest = _unique_dest(dest_group, item.match_id) if not dry_run else (
                dest_group / f"{item.match_id}.json"
            )
            if not dry_run:
                shutil.copy2(item.json_path, dest)
            files.append(dest.name)
            copied += 1
        stats = _result_counts(items)
        group_summaries.append(
            {
                "model": key.model,
                "strategy": key.strategy,
                "race": key.race,
                "difficulty": key.difficulty,
                "style": key.style,
                "path": rel.as_posix(),
                "count": len(items),
                "batches": sorted({m.batch for m in items}),
                "files": files,
                **stats,
            }
        )

    manifest = {
        "source": str(records_dir),
        "out": str(out_dir),
        "min_matches": min_matches,
        "dry_run": dry_run,
        "groups_exported": len(group_summaries),
        "json_copied": copied,
        "groups_skipped": skipped,
        "groups": group_summaries,
    }
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Drop bulky per-file lists from the on-disk manifest; keep them in
        # memory for callers that need them.
        disk_manifest = dict(manifest)
        disk_manifest["groups"] = [
            {k: v for k, v in group.items() if k != "files"}
            for group in group_summaries
        ]
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(disk_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        csv_path = out_dir / "winrate.csv"
        md_path = out_dir / "winrate.md"
        write_winrate_csv(csv_path, group_summaries)
        write_winrate_md(md_path, group_summaries, min_matches=min_matches)
        manifest["manifest"] = str(manifest_path)
        manifest["winrate_csv"] = str(csv_path)
        manifest["winrate_md"] = str(md_path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Organize batch match JSON into "
            "model/strategy/race/difficulty/style folders."
        )
    )
    parser.add_argument(
        "--records-dir",
        default="",
        help="Source game_records root (default: <repo>/game_records)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Destination root (default: <repo>/game_records_json)",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=20,
        help="Only export groups with at least this many JSON matches (default: 20)",
    )
    parser.add_argument(
        "--batch",
        action="append",
        default=[],
        help="Only include this batch folder name (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without copying files",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    records_dir = (
        Path(args.records_dir).expanduser().resolve()
        if args.records_dir
        else (_project_root() / "game_records").resolve()
    )
    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (_project_root() / "game_records_json").resolve()
    )
    if not records_dir.is_dir():
        print(f"[ERROR] records dir not found: {records_dir}")
        return 1

    batch_filter = {name.strip() for name in args.batch if name.strip()} or None
    manifest = organize(
        records_dir,
        out_dir,
        min_matches=max(0, int(args.min_matches)),
        dry_run=bool(args.dry_run),
        batch_filter=batch_filter,
    )

    print(f"source : {manifest['source']}")
    print(f"out    : {manifest['out']}")
    print(f"min    : {manifest['min_matches']}")
    print(f"dry_run: {manifest['dry_run']}")
    print(
        f"exported {manifest['groups_exported']} group(s), "
        f"{manifest['json_copied']} json file(s)"
    )
    for group in manifest["groups"]:
        unknown = int(group.get("unknown") or 0)
        unknown_part = f"/?{unknown}" if unknown else ""
        print(
            f"  + {group['path']}  "
            f"({group['count']} json, "
            f"W{group['wins']}/L{group['losses']}/T{group['ties']}"
            f"{unknown_part} {group['win_rate_pct']}, "
            f"batches={len(group['batches'])})"
        )
    if manifest["groups_skipped"]:
        print(f"skipped {len(manifest['groups_skipped'])} group(s) below min:")
        for group in manifest["groups_skipped"][:30]:
            print(
                f"  - {group['model']}/{group['strategy']}/{group['race']}/"
                f"{group['difficulty']}/{group['style']}  "
                f"({group['count']})"
            )
        remaining = len(manifest["groups_skipped"]) - 30
        if remaining > 0:
            print(f"  ... and {remaining} more")
    if not args.dry_run:
        print(f"manifest : {manifest.get('manifest')}")
        print(f"winrate  : {manifest.get('winrate_csv')}")
        print(f"winrate  : {manifest.get('winrate_md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
