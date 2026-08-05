"""Run the strategy-driven SC2 agent against the built-in game AI.

Edit the ``DEFAULT_*`` values below for convenient local experiments, or pass
explicit CLI arguments. ``--force-strategy <name>`` selects
``SKILL/<race>/<name>/strategy.md``; use ``--force-strategy none`` to let the
Strategy Coordinator select or generate a strategy at game start.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from datetime import datetime
from typing import List, Optional, Sequence

sys.path.insert(1, "python-sc2")

from bot_loader import GameStarter, BotDefinitions
from version import update_version_txt


def _install_interrupt_flush_handlers() -> None:
    """Flush observation JSON on Ctrl+C / terminate even if on_end never runs."""
    try:
        from sharpy.managers.extensions.llm_observation_recorder import (
            _flush_all_active_recorders,
        )
    except Exception:
        return

    def _handler(signum, frame):  # noqa: ANN001
        try:
            _flush_all_active_recorders(reason=f"signal_{signum}")
        except Exception:
            pass
        raise SystemExit(128 + int(signum))

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass

# =============================================================================
# 运行配置 — 改这里即可；``python run_vs_ai.py`` 与 ``play_vs_ai()`` 均以此为准
# CLI 显式传参会覆盖对应项。布尔项支持 ``--flag`` / ``--no-flag`` 覆盖文件默认。
# =============================================================================

OUTPUT_BASE_DIR = "./game_records"

DEFAULT_MY_BOT_NAME = "commander"
DEFAULT_BOT_RACE = "terran"
DEFAULT_BOT_INSTRUCT = ""
DEFAULT_MAP_NAME = "KairosJunctionLE"
DEFAULT_REAL_TIME = False

DEFAULT_ENEMY_RACE = "terran"
DEFAULT_ENEMY_DIFFICULTY = "medium"
DEFAULT_ENEMY_BUILD = "macro"

DEFAULT_COMMANDER_MODEL = "qwen3-32b"
# Deprecated aliases kept for older scripts.
DEFAULT_COORDINATOR_MODEL = DEFAULT_COMMANDER_MODEL
DEFAULT_MACRO_MODEL = DEFAULT_COMMANDER_MODEL
DEFAULT_TRANSLATOR_MODEL = DEFAULT_COMMANDER_MODEL

DEFAULT_FORCE_STRATEGY = "mid_tank"

# --- 其它 ---
DEFAULT_SKIP_VERSION_UPDATE = False  # True：跳过 version.txt 更新（批量并发时防 IO 锁）


class _TeeStream:
    """Mirror direct-run console output into the match log."""

    def __init__(self, console, log_file) -> None:
        self.console = console
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def _resolve_force_strategy(explicit: Optional[str]) -> Optional[str]:
    """解析 force_strategy。

    * ``explicit is None`` — 未在 CLI/调用方指定，使用 ``DEFAULT_FORCE_STRATEGY``
    * ``''`` / ``'none'`` — 显式取消强制
    * 其它非空字符串 — 策略文件夹名
    """
    if explicit is None:
        explicit = DEFAULT_FORCE_STRATEGY
    s = str(explicit or "").strip()
    if not s or s.lower() == "none":
        return None
    return s


def _safe_match_part(value: str, *, max_len: int = 32) -> str:
    text = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value)
    ).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    if max_len > 0 and len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text or "x"


_RACE_CODE = {
    "terran": "T",
    "protoss": "P",
    "zerg": "Z",
    "random": "R",
}

_DIFFICULTY_CODE = {
    "veryeasy": "ve",
    "easy": "e",
    "medium": "m",
    "hard": "h",
    "harder": "hr",
    "veryhard": "vh",
    "cheatvision": "cv",
    "cheatmoney": "cm",
    "cheatinsane": "ci",
}

_BUILD_CODE = {
    "random": "rnd",
    "rush": "rsh",
    "timing": "tim",
    "power": "pwr",
    "macro": "mac",
    "air": "air",
}

# Common ladder / experiment maps → short codes.
_MAP_CODE = {
    "kairosjunction": "KJ",
    "abyssalreef": "AR",
    "acolyte": "AC",
    "automaton": "AU",
    "blueshift": "BS",
    "ceruleanfall": "CF",
    "cyberforest": "CyF",
    "darknesssanctuary": "DS",
    "discobloodbath": "DB",
    "ephemeron": "EP",
    "everdream": "ED",
    "goldenwall": "GW",
    "kingscove": "KC",
    "newrepugnancy": "NR",
    "parasite": "PS",
    "portaleksander": "PA",
    "simulacrum": "SI",
    "submarine": "SU",
    "thunderbird": "TB",
    "triton": "TR",
    "wintersgate": "WG",
    "worldofsleepers": "WoS",
    "zen": "ZEN",
}

_STRATEGY_CODE = {
    "early_marine": "em",
    "mid_tank": "mt",
    "late_battlecruiser": "lbc",
}


def _race_code(race: str) -> str:
    return _RACE_CODE.get(str(race or "").strip().lower(), _safe_match_part(race, max_len=1).upper() or "X")


def _difficulty_code(difficulty: str) -> str:
    key = str(difficulty or "").strip().lower()
    return _DIFFICULTY_CODE.get(key, _safe_match_part(key, max_len=3))


def _build_code(build: str) -> str:
    key = str(build or "").strip().lower()
    return _BUILD_CODE.get(key, _safe_match_part(key, max_len=4))


def _map_code(map_name: str) -> str:
    raw = str(map_name or "")
    cleaned = "".join(ch for ch in raw.lower() if ch.isalnum())
    cleaned = cleaned.replace("le", "") if cleaned.endswith("le") else cleaned
    if cleaned in _MAP_CODE:
        return _MAP_CODE[cleaned]
    # Fallback: first letters of camel/underscore tokens, else first 6 alnum chars.
    tokens = [t for t in _safe_match_part(raw.replace("LE", ""), max_len=40).split("_") if t]
    if len(tokens) >= 2:
        return "".join(tok[:1].upper() for tok in tokens)[:6]
    return _safe_match_part(cleaned or raw, max_len=6)


def _model_code(*models: str) -> str:
    """Compress model keys, e.g. kimi-k2.5 -> k25, DeepSeek-V4-flash -> dsv4f."""
    cleaned = []
    for model in models:
        text = str(model or "").strip()
        if not text:
            continue
        # Keep letters/digits only, drop separators, then truncate.
        compact = "".join(ch for ch in text.lower() if ch.isalnum())
        # Prefer a stable short form for common names.
        if compact.startswith("kimi"):
            digits = "".join(ch for ch in compact if ch.isdigit())
            compact = f"k{digits}" if digits else "kimi"
        elif compact.startswith("deepseek"):
            compact = "ds" + compact.replace("deepseek", "", 1)[:4]
        elif compact.startswith("qwen"):
            compact = "qw" + compact.replace("qwen", "", 1)[:4]
        cleaned.append(_safe_match_part(compact, max_len=8))
    if not cleaned:
        return "nm"
    unique = list(dict.fromkeys(cleaned))
    if len(unique) == 1:
        return unique[0]
    return _safe_match_part("".join(unique), max_len=12)


def _strategy_code(strategy: Optional[str]) -> str:
    key = str(strategy or "").strip()
    if not key:
        return ""
    known = _STRATEGY_CODE.get(key)
    if known:
        return known
    # two_base_tanks_llm_combat_opt3 -> 2bt_o3 style fallback
    compact = _safe_match_part(key, max_len=16)
    parts = [p for p in compact.split("_") if p]
    if not parts:
        return ""
    initials = "".join(p[0] for p in parts if p)
    return _safe_match_part(initials or compact, max_len=8)


def build_match_id(
    *,
    timestamp: str,
    my_bot_name: str,
    enemy_race: str,
    enemy_difficulty: str,
    enemy_build: str,
    map_name: str,
    bot_race: str,
    coordinator_model: str,
    macro_model: str,
    translator_model: str,
    run_index: Optional[int],
    force_strategy: Optional[str] = None,
) -> str:
    """Build a compact, path-safe match folder name.

    Example:
      260723_192457_TvT_hr_mac_KJ_k25_2bt

    Full metadata is also written to ``match_info.txt``
    beside ``{match_id}.json`` and ``{match_id}.SC2Replay``.
    """
    # yyMMdd_HHMMSS keeps uniqueness while saving two characters vs yyyyMMdd.
    ts = str(timestamp or "").strip()
    if len(ts) >= 15 and ts[8] == "_" and ts[:8].isdigit():
        ts = ts[2:]  # 20260723_192457 -> 260723_192457

    matchup = f"{_race_code(bot_race)}v{_race_code(enemy_race)}"
    parts = [
        _safe_match_part(ts, max_len=13),
        matchup,
        _difficulty_code(enemy_difficulty),
        _build_code(enemy_build),
        _map_code(map_name),
        _model_code(coordinator_model, macro_model, translator_model),
    ]
    strategy = _strategy_code(force_strategy)
    if strategy:
        parts.append(strategy)
    if run_index is not None:
        parts.append(f"r{int(run_index)}")

    # Bot name is omitted when it is the default universal bot; rare custom
    # bots keep a tiny tag for disambiguation.
    bot = str(my_bot_name or "").strip().lower()
    if bot and bot not in {"commander", "universal_llm", "universal", "ul"}:
        parts.insert(1, _safe_match_part(bot, max_len=6))

    return _safe_match_part("_".join(parts), max_len=72)

def play_vs_ai(
    *,
    my_bot_name: str = DEFAULT_MY_BOT_NAME,
    map_name: str = DEFAULT_MAP_NAME,
    real_time: bool = DEFAULT_REAL_TIME,
    enemy_race: str = DEFAULT_ENEMY_RACE,
    enemy_difficulty: str = DEFAULT_ENEMY_DIFFICULTY,
    enemy_build: str = DEFAULT_ENEMY_BUILD,
    bot_instruct: str = DEFAULT_BOT_INSTRUCT,
    bot_race: str = DEFAULT_BOT_RACE,
    commander_model: str = DEFAULT_COMMANDER_MODEL,
    coordinator_model: str = "",
    macro_model: str = "",
    translator_model: str = "",
    batch_name: Optional[str] = None,
    run_index: Optional[int] = None,
    output_base_dir: str = OUTPUT_BASE_DIR,
    record_dir_file: Optional[str] = None,
    skip_version_update: bool = DEFAULT_SKIP_VERSION_UPDATE,
    force_strategy: Optional[str] = None,
) -> None:
    force_strategy = _resolve_force_strategy(force_strategy)
    _install_interrupt_flush_handlers()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    if not skip_version_update:
        update_version_txt()

    p2_string = f"ai.{enemy_race}.{enemy_difficulty}.{enemy_build}"
    p1_string = (
        f"{my_bot_name}.{bot_race}"
        if my_bot_name in {"commander", "universal_llm"}
        else my_bot_name
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_for_id = (
        commander_model
        or macro_model
        or coordinator_model
        or DEFAULT_COMMANDER_MODEL
    )
    match_id = build_match_id(
        timestamp=timestamp,
        my_bot_name=my_bot_name,
        enemy_race=enemy_race,
        enemy_difficulty=enemy_difficulty,
        enemy_build=enemy_build,
        map_name=map_name,
        bot_race=bot_race,
        coordinator_model=model_for_id,
        macro_model=model_for_id,
        translator_model=model_for_id,
        run_index=run_index,
        force_strategy=force_strategy,
    )

    base = os.path.abspath(output_base_dir)
    # 如果指定了 batch_name，则归档到单独的批次文件夹下面
    if batch_name:
        batch_slug = _safe_match_part(batch_name, max_len=40)
        record_dir = os.path.join(base, batch_slug, match_id)
    else:
        record_dir = os.path.join(base, match_id)

    os.makedirs(record_dir, exist_ok=True)
    if record_dir_file:
        # The batch launcher cannot know the timestamped match id in advance.
        # Publish the resolved directory so it can archive the console log
        # beside the JSON and replay after this process exits.
        marker_path = os.path.abspath(record_dir_file)
        os.makedirs(os.path.dirname(marker_path), exist_ok=True)
        with open(marker_path, "w", encoding="utf-8") as handle:
            handle.write(record_dir)

    # Human-readable sidecar so the folder id can stay compact.
    army_model = os.environ.get("LLM_ARMY_CONTROL_MODEL", "")
    info_path = os.path.join(record_dir, "match_info.txt")
    with open(info_path, "w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "SC2 Match Info",
                    "==============",
                    f"match_id:          {match_id}",
                    f"timestamp:         {timestamp}",
                    f"bot:               {my_bot_name} ({bot_race})",
                    f"enemy:             AI {enemy_race} / {enemy_difficulty} / {enemy_build}",
                    f"map:               {map_name}",
                    f"force_strategy:    {force_strategy or '(auto)'}",
                    f"coordinator_model: {coordinator_model}",
                    f"macro_model:       {macro_model}",
                    f"translator_model:  {translator_model}",
                    f"army_model:        {army_model or '(default)'}",
                    f"batch_name:        {batch_name or '-'}",
                    f"run_index:         {run_index if run_index is not None else '-'}",
                    f"record_dir:        {record_dir}",
                    "",
                    "bot_instruct:",
                    bot_instruct or "(none)",
                    "",
                ]
            )
        )

    args: List[str] = [
        "run_custom.py",
        "-m", map_name,
        "-p1", p1_string,
        "-p2", p2_string,
        "--record-dir", record_dir,
        "--match-id", match_id,
    ]
    
    model_key = (
        commander_model
        or macro_model
        or coordinator_model
        or DEFAULT_COMMANDER_MODEL
    )

    if real_time:
        args.append("-rt")
    if bot_instruct:
        args.extend(["--instruct", bot_instruct])
    if model_key:
        args.extend(["--commander-model", model_key])
        args.extend(["--macro-model", model_key])
    if force_strategy:
        args.extend(["--force-strategy", force_strategy])

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    # Always mirror console into ``{match_id}.log`` (same as single-match runs).
    # Batch launchers may still redirect stdout to a temp ``fg_run_*.log`` for
    # job monitoring; that file is discarded/renamed after the match.
    direct_log_path = os.path.join(record_dir, f"{match_id}.log")
    direct_log = open(direct_log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(original_stdout, direct_log)
    sys.stderr = _TeeStream(original_stderr, direct_log)

    sys.argv = args

    print("==================================================")
    print(" 正在启动 SC2-Commander 对战...")
    print(f" ▷ 我方阵营 : {my_bot_name} ({bot_race})")
    print(f" ▷ 对手 AI  : {enemy_race.upper()} | 难度: {enemy_difficulty} | 风格: {enemy_build}")
    print(f" ▷ 比赛地图 : {map_name}")
    print(f" ▷ Commander : {model_key}")
    print(f" ▷ Strategy  : {force_strategy or 'none'}")
    print(f" ▷ Runtime   : {sys.executable}")
    if batch_name:
        print(f" ▷ 批次名称 : {batch_name} (任务序号: {run_index})")
    print(f" ▷ 记录目录 : {record_dir}")
    print("==================================================")

    try:
        ladder_bots_path = os.path.join(root_dir, "Bots")
        definitions: BotDefinitions = BotDefinitions(ladder_bots_path)

        starter = GameStarter(definitions)
        starter.play()
    finally:
        if direct_log is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            direct_log.close()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="与 SC2 内置 AI 对战。支持单跑或被批处理脚本调用。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--my-bot-name", default=DEFAULT_MY_BOT_NAME, help="Bot 名称")
    p.add_argument("--map-name", default=DEFAULT_MAP_NAME, help="地图名")
    p.add_argument(
        "--real-time",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REAL_TIME,
        help="实时模式(人类观测)",
    )
    p.add_argument("--enemy-race", default=DEFAULT_ENEMY_RACE, help="对手种族")
    p.add_argument("--enemy-difficulty", default=DEFAULT_ENEMY_DIFFICULTY, help="对手难度")
    p.add_argument("--enemy-build", default=DEFAULT_ENEMY_BUILD, help="对手 AI 风格")
    p.add_argument("--bot-instruct", default=DEFAULT_BOT_INSTRUCT, help="战术指令")
    p.add_argument("--bot-race", default=DEFAULT_BOT_RACE, help="我方种族")
    p.add_argument("--commander-model", default=DEFAULT_COMMANDER_MODEL, help="Commander model key")
    p.add_argument("--coordinator-model", default=DEFAULT_COORDINATOR_MODEL, help="Alias of --commander-model")
    p.add_argument("--macro-model", default=DEFAULT_MACRO_MODEL, help="Alias of --commander-model")
    p.add_argument("--translator-model", default=DEFAULT_TRANSLATOR_MODEL, help="Unused alias")
    p.add_argument("--batch-name", default="", help="记录写入 game_records/<batch-name>/ 归档")
    p.add_argument("--run-index", type=int, default=None, help="批处理序号以防并发冲突")
    p.add_argument("--output-base-dir", default=OUTPUT_BASE_DIR, help="记录根目录")
    p.add_argument(
        "--record-dir-file",
        default="",
        help="Write the resolved match directory here for batch log archival.",
    )
    p.add_argument(
        "--skip-version-update",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SKIP_VERSION_UPDATE,
        help="跳过版本更新防止 IO 锁",
    )
    p.add_argument(
        "--force-strategy",
        default=None,
        metavar="NAME",
        help=(
            f"强制锁定策略（skills/<race>/<name>）；"
            f"未指定时默认 {DEFAULT_FORCE_STRATEGY!r}；"
            f"传 none 取消强制。"
        ),
    )

    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    ns = _parse_args(argv)
    play_vs_ai(
        my_bot_name=ns.my_bot_name,
        map_name=ns.map_name,
        real_time=ns.real_time,
        enemy_race=ns.enemy_race,
        enemy_difficulty=ns.enemy_difficulty,
        enemy_build=ns.enemy_build,
        bot_instruct=ns.bot_instruct,
        bot_race=ns.bot_race,
        commander_model=getattr(ns, "commander_model", None)
        or ns.macro_model
        or ns.coordinator_model,
        batch_name=ns.batch_name or None,
        run_index=ns.run_index,
        output_base_dir=ns.output_base_dir,
        record_dir_file=ns.record_dir_file or None,
        skip_version_update=ns.skip_version_update,
        force_strategy=ns.force_strategy,
    )

if __name__ == "__main__":
    main()
