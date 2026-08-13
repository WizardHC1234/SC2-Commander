"""Export Victory decision turns from ``game_records_json`` to SFT jsonl.

Each line::

    {"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}],
     "meta":{...}}

Filter by model / strategy / race / difficulty / style (repeatable flags).
Path layout is expected to be::

    <root>/<model>/<strategy>/<race>/<difficulty>/<style>/<match_id>.json

Examples::

    python tools/export_sft_jsonl.py --model qwen3.5-27b --difficulty hard --strategy tank
    python tools/export_sft_jsonl.py --model deepseek-v4-flash --difficulty harder --difficulty veryhard
    python tools/export_sft_jsonl.py --strategy marine --strategy tank --min-winrate 0.7
    python tools/export_sft_jsonl.py --dry-run --model kimi-k2.5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.batch_stats import classify_result  # noqa: E402

SKIP_JSON_NAMES = {"manifest.json", "winrate.csv", "winrate.md"}


def _project_root() -> Path:
    return _REPO_ROOT


def _norm(value: object) -> str:
    return str(value or "").strip().lower()


def _parse_multi(values: Optional[list[str]]) -> Optional[set[str]]:
    if not values:
        return None
    out: set[str] = set()
    for raw in values:
        for part in str(raw).split(","):
            text = _norm(part)
            if text:
                out.add(text)
    return out or None


def _match_filter(value: str, allowed: Optional[set[str]]) -> bool:
    if not allowed:
        return True
    return _norm(value) in allowed


def _axes_from_path(path: Path, root: Path) -> Optional[dict[str, str]]:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 6:
        return None
    model, strategy, race, difficulty, style = parts[:5]
    return {
        "model": model,
        "strategy": strategy,
        "race": race,
        "difficulty": difficulty,
        "style": style,
    }


def _axes_from_meta(meta: dict[str, Any], fallback: dict[str, str]) -> dict[str, str]:
    opponent = str(meta.get("opponent_id") or "")
    parts = [p for p in opponent.split(".") if p]
    race = str(meta.get("enemy_race") or fallback.get("race") or "unknown")
    difficulty = fallback.get("difficulty") or "unknown"
    style = fallback.get("style") or "unknown"
    if len(parts) >= 2:
        difficulty = parts[-2] or difficulty
        style = parts[-1] or style
    return {
        "model": str(
            meta.get("commander_model_key")
            or meta.get("commander_model")
            or fallback.get("model")
            or "unknown"
        ),
        "strategy": str(
            meta.get("strategy_id")
            or meta.get("strategy")
            or fallback.get("strategy")
            or "unknown"
        ),
        "race": race,
        "difficulty": difficulty,
        "style": style,
    }


def _load_winrate_index(csv_path: Path) -> dict[tuple[str, str, str, str, str], float]:
    if not csv_path.is_file():
        return {}
    index: dict[tuple[str, str, str, str, str], float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (
                _norm(row.get("model")),
                _norm(row.get("strategy")),
                _norm(row.get("race")),
                _norm(row.get("difficulty")),
                _norm(row.get("style")),
            )
            raw = (row.get("win_rate") or "").strip()
            if not raw:
                continue
            try:
                index[key] = float(raw)
            except ValueError:
                continue
    return index


def _is_victory(meta: dict[str, Any]) -> bool:
    return classify_result(str(meta.get("result") or "")) == "win"


def _assistant_text(interaction: dict[str, Any]) -> str:
    text = str(interaction.get("assistant_content") or "").strip()
    if text:
        return text
    # Fallback: rebuild from structured tool_calls when content was not stored.
    tool_calls = interaction.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)
    return ""


def _is_decision_turn(interaction: dict[str, Any]) -> bool:
    if not isinstance(interaction, dict):
        return False
    if interaction.get("accepted") is False:
        return False
    if interaction.get("accepted") is not True and not interaction.get("tool_calls"):
        # Skip tool-selection / thin records unless clearly a decision.
        if "selected_tools" in interaction or "required_tools" in interaction:
            return False
    messages = interaction.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    roles = {str(m.get("role") or "") for m in messages if isinstance(m, dict)}
    if "system" not in roles or "user" not in roles:
        return False
    return bool(_assistant_text(interaction))


def _build_sft_messages(interaction: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for message in interaction.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "")
        if role not in {"system", "user", "assistant"}:
            continue
        if role == "assistant":
            continue  # final assistant comes from accepted content
        out.append({"role": role, "content": content})
    out.append({"role": "assistant", "content": _assistant_text(interaction)})
    return out


def iter_match_json_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        if path.name in SKIP_JSON_NAMES:
            continue
        if path.name.endswith(".csv") or path.name.endswith(".md"):
            continue
        yield path


def export_sft(
    records_root: Path,
    out_path: Path,
    *,
    models: Optional[set[str]] = None,
    strategies: Optional[set[str]] = None,
    races: Optional[set[str]] = None,
    difficulties: Optional[set[str]] = None,
    styles: Optional[set[str]] = None,
    require_victory: bool = True,
    accepted_only: bool = True,
    min_winrate: Optional[float] = None,
    tool_mode: Optional[str] = None,
    limit: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    winrate_index = _load_winrate_index(records_root / "winrate.csv")
    kept = 0
    skipped = {
        "not_victory": 0,
        "filter": 0,
        "winrate": 0,
        "no_decision": 0,
        "accepted": 0,
        "tool_mode": 0,
        "bad_json": 0,
    }
    matches_used = 0
    by_group: dict[str, int] = {}

    out_handle = None
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_handle = out_path.open("w", encoding="utf-8")

    try:
        for path in iter_match_json_files(records_root):
            if limit and kept >= limit:
                break
            fallback = _axes_from_path(path, records_root) or {
                "model": "unknown",
                "strategy": "unknown",
                "race": "unknown",
                "difficulty": "unknown",
                "style": "unknown",
            }
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                skipped["bad_json"] += 1
                continue
            if not isinstance(data, dict):
                skipped["bad_json"] += 1
                continue
            meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            axes = _axes_from_meta(meta, fallback)

            if require_victory and not _is_victory(meta):
                skipped["not_victory"] += 1
                continue
            if not (
                _match_filter(axes["model"], models)
                and _match_filter(axes["strategy"], strategies)
                and _match_filter(axes["race"], races)
                and _match_filter(axes["difficulty"], difficulties)
                and _match_filter(axes["style"], styles)
            ):
                skipped["filter"] += 1
                continue

            group_key = (
                _norm(axes["model"]),
                _norm(axes["strategy"]),
                _norm(axes["race"]),
                _norm(axes["difficulty"]),
                _norm(axes["style"]),
            )
            if min_winrate is not None:
                rate = winrate_index.get(group_key)
                if rate is None or rate < min_winrate:
                    skipped["winrate"] += 1
                    continue

            interactions = data.get("interactions") or []
            if not isinstance(interactions, list):
                continue
            match_kept = 0
            for index, interaction in enumerate(interactions):
                if limit and kept >= limit:
                    break
                if not _is_decision_turn(interaction):
                    skipped["no_decision"] += 1
                    continue
                if accepted_only and interaction.get("accepted") is not True:
                    skipped["accepted"] += 1
                    continue
                mode = str(interaction.get("tool_mode") or "").strip().lower()
                if tool_mode and mode and mode != tool_mode:
                    skipped["tool_mode"] += 1
                    continue
                if tool_mode and not mode and tool_mode != "json":
                    # Older records often omit tool_mode but store JSON content.
                    skipped["tool_mode"] += 1
                    continue

                record = {
                    "messages": _build_sft_messages(interaction),
                    "meta": {
                        "match_id": path.stem,
                        "source": str(path),
                        "interaction_index": index,
                        "game_time": interaction.get("game_time"),
                        "tool_mode": mode or "json",
                        "model": axes["model"],
                        "strategy": axes["strategy"],
                        "race": axes["race"],
                        "difficulty": axes["difficulty"],
                        "style": axes["style"],
                        "result": meta.get("result"),
                        "woken_by": interaction.get("woken_by"),
                        "strategy_hash": interaction.get("strategy_hash")
                        or meta.get("strategy_hash"),
                    },
                }
                if not dry_run and out_handle is not None:
                    out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                match_kept += 1
                g = "/".join(group_key)
                by_group[g] = by_group.get(g, 0) + 1
            if match_kept:
                matches_used += 1
    finally:
        if out_handle is not None:
            out_handle.close()

    return {
        "out": str(out_path),
        "dry_run": dry_run,
        "turns": kept,
        "matches": matches_used,
        "skipped": skipped,
        "by_group": dict(sorted(by_group.items())),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export filtered Victory decision turns to SFT jsonl."
    )
    p.add_argument(
        "--records-dir",
        default="",
        help="Root of organized JSON (default: <repo>/game_records_json)",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output jsonl path (default: <repo>/sft_data/sft.jsonl)",
    )
    p.add_argument("--model", action="append", default=[], help="Model filter (repeatable/csv)")
    p.add_argument("--strategy", action="append", default=[], help="Strategy filter")
    p.add_argument("--race", action="append", default=[], help="Enemy race filter")
    p.add_argument("--difficulty", action="append", default=[], help="Difficulty filter")
    p.add_argument("--style", action="append", default=[], help="Enemy build/style filter")
    p.add_argument(
        "--min-winrate",
        type=float,
        default=None,
        help="Keep only groups whose winrate.csv rate >= this (e.g. 0.7)",
    )
    p.add_argument(
        "--include-losses",
        action="store_true",
        help="Do not require Victory (default: victory only)",
    )
    p.add_argument(
        "--allow-unaccepted",
        action="store_true",
        help="Keep decision turns even when accepted is not True",
    )
    p.add_argument(
        "--tool-mode",
        default="",
        help="Only keep this tool_mode (json/native). Empty = any",
    )
    p.add_argument("--limit", type=int, default=0, help="Max turns to export (0 = all)")
    p.add_argument("--dry-run", action="store_true", help="Count only, do not write")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    records_dir = (
        Path(args.records_dir).expanduser().resolve()
        if args.records_dir
        else (_project_root() / "game_records_json").resolve()
    )
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (_project_root() / "sft_data" / "sft.jsonl").resolve()
    )
    if not records_dir.is_dir():
        print(f"[ERROR] records dir not found: {records_dir}")
        return 1

    summary = export_sft(
        records_dir,
        out_path,
        models=_parse_multi(args.model),
        strategies=_parse_multi(args.strategy),
        races=_parse_multi(args.race),
        difficulties=_parse_multi(args.difficulty),
        styles=_parse_multi(args.style),
        require_victory=not args.include_losses,
        accepted_only=not args.allow_unaccepted,
        min_winrate=args.min_winrate,
        tool_mode=(args.tool_mode or "").strip().lower() or None,
        limit=max(0, int(args.limit)),
        dry_run=bool(args.dry_run),
    )
    print(f"records : {records_dir}")
    print(f"out     : {summary['out']}")
    print(f"dry_run : {summary['dry_run']}")
    print(f"matches : {summary['matches']}")
    print(f"turns   : {summary['turns']}")
    print(f"skipped : {summary['skipped']}")
    if summary["by_group"]:
        print("by_group:")
        for key, count in summary["by_group"].items():
            print(f"  {key}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
