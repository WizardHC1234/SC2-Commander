from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import mpyq

import sc2_compat

sc2_compat.apply()

from sc2.bot_ai import BotAI
from sc2.main import run_replay


TRUTH_SUFFIX = ".enemy_truth.json"


def enemy_truth_path(record_path: str | Path) -> Path:
    record = Path(record_path).resolve()
    return record.with_name(f"{record.stem}{TRUTH_SUFFIX}")


def replay_path_for_record(record_path: str | Path) -> Path:
    return Path(record_path).resolve().with_suffix(".SC2Replay")


def _patch_linux_start_replay_absolute_path() -> None:
    """burnysc2 rewrites Linux replay paths to bare filenames and requires
    ~/Documents/StarCraft II/Replays. That fails on headless installs where SC2
    accepts absolute paths instead. Patch once to always send an absolute path.
    """
    if platform.system() != "Linux":
        return
    from s2clientprotocol import sc2api_pb2 as sc_pb
    from sc2.controller import Controller

    current = Controller.start_replay
    if getattr(current, "_sc2_commander_abs_path", False):
        return

    async def start_replay(self, replay_path: str, realtime: bool, observed_id: int = 0):
        ifopts = sc_pb.InterfaceOptions(
            raw=True,
            score=True,
            show_cloaked=True,
            raw_affects_selection=True,
            raw_crop_to_playable_area=False,
        )
        path = str(Path(replay_path).expanduser().resolve())
        if not Path(path).is_file():
            raise FileNotFoundError(f"replay not found: {path}")
        req = sc_pb.RequestStartReplay(
            replay_path=path,
            observed_player_id=observed_id,
            realtime=realtime,
            options=ifopts,
        )
        result = await self._execute(start_replay=req)
        assert result.status == 4, (
            f"{result.start_replay.error} - {result.start_replay.error_details}"
        )
        return result

    start_replay._sc2_commander_abs_path = True  # type: ignore[attr-defined]
    Controller.start_replay = start_replay  # type: ignore[method-assign]


def prepare_replay_path_for_sc2(replay_path: Path) -> Path:
    """Return an absolute replay path suitable for run_replay on this OS."""
    replay_path = Path(replay_path).expanduser().resolve()
    if not replay_path.is_file():
        raise FileNotFoundError(f"replay not found: {replay_path}")
    _patch_linux_start_replay_absolute_path()
    return replay_path


def commander_game_loops(record: dict[str, Any]) -> list[int]:
    loops: list[int] = []
    for interaction in record.get("interactions") or []:
        if not isinstance(interaction, dict):
            continue
        if str(interaction.get("agent") or "") != "commander":
            continue
        observation = interaction.get("observation")
        if not isinstance(observation, dict):
            continue
        time_state = observation.get("time")
        raw_loop = time_state.get("game_loop") if isinstance(time_state, dict) else None
        if raw_loop is None:
            snapshot_id = str(observation.get("snapshot_id") or "")
            if snapshot_id.startswith("game_loop:"):
                raw_loop = snapshot_id.partition(":")[2]
        try:
            game_loop = int(raw_loop)
        except (TypeError, ValueError):
            continue
        if game_loop >= 0:
            loops.append(game_loop)
    return sorted(dict.fromkeys(loops))


def _replay_metadata(replay_path: Path) -> dict[str, Any]:
    archive = mpyq.MPQArchive(str(replay_path)).extract()
    raw = archive.get(b"replay.gamemetadata.json")
    if not raw:
        raise ValueError(f"replay metadata missing: {replay_path}")
    metadata = json.loads(raw.decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid replay metadata: {replay_path}")
    return metadata


def _opponent_player_id(metadata: dict[str, Any], commander_player_id: int) -> int:
    player_ids: list[int] = []
    for player in metadata.get("Players") or []:
        if not isinstance(player, dict):
            continue
        try:
            player_ids.append(int(player.get("PlayerID")))
        except (TypeError, ValueError):
            continue
    opponents = [player_id for player_id in player_ids if player_id != commander_player_id]
    if len(opponents) != 1:
        raise ValueError(
            f"expected one replay opponent for player {commander_player_id}, got {player_ids}"
        )
    return opponents[0]


def _counter_names(units: Iterable[Any]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(unit.type_id.name) for unit in units).items(),
            key=lambda item: item[0],
        )
    )


def _unit_type_key(unit: Any) -> int:
    type_id = getattr(unit, "type_id", None)
    if type_id is None:
        return -1
    value = getattr(type_id, "value", type_id)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _ability_name(order: Any) -> str:
    ability = getattr(order, "ability", None)
    ability_id = getattr(ability, "id", None)
    if ability_id is None:
        ability_id = ability
    name = getattr(ability_id, "name", None)
    if name:
        return str(name)
    value = getattr(ability_id, "value", ability_id)
    return str(value or "UNKNOWN")


def _unit_row(unit: Any) -> list[Any]:
    orders = [_ability_name(order) for order in getattr(unit, "orders", [])]
    return [
        int(unit.tag),
        str(unit.type_id.name),
        round(float(unit.position.x), 1),
        round(float(unit.position.y), 1),
        round(float(unit.health), 1),
        round(float(getattr(unit, "shield", 0.0)), 1),
        round(float(getattr(unit, "energy", 0.0)), 1),
        round(float(unit.build_progress), 3),
        orders,
    ]


class ReplayTruthBot(BotAI):
    """Observe one replay player and capture only requested game loops."""

    def __init__(self, *, observed_player_id: int, target_loops: list[int]) -> None:
        super().__init__()
        self.observed_player_id = int(observed_player_id)
        self.target_loops = list(target_loops)
        self.target_index = 0
        self.snapshots: list[dict[str, Any]] = []

    def _prepare_start(self, client, player_id, game_info, game_data, realtime=False):
        # The bundled python-sc2 replay runner passes player_id=0 to BotAI even
        # when RequestStartReplay observes a real player. Repair that old
        # wrapper behavior so own units/resources mean the selected opponent.
        client._player_id = self.observed_player_id
        return super()._prepare_start(
            client,
            self.observed_player_id,
            game_info,
            game_data,
            realtime=realtime,
        )

    def _is_army_unit(self, unit: Any) -> bool:
        entry = self._game_data.units.get(_unit_type_key(unit))
        if entry is None:
            return False
        try:
            return float(entry._proto.food_required) > 0
        except Exception:
            return False

    def _snapshot(self, requested_loop: int) -> dict[str, Any]:
        workers = list(self.workers)
        worker_tags = {unit.tag for unit in workers}
        army_units = [
            unit
            for unit in self.units
            if unit.tag not in worker_tags and self._is_army_unit(unit)
        ]
        completed_structures = [
            unit for unit in self.structures if float(unit.build_progress) >= 1.0
        ]
        incomplete_structures = [
            unit for unit in self.structures if float(unit.build_progress) < 1.0
        ]
        active_orders = Counter(
            _ability_name(order)
            for unit in self.all_own_units
            for order in getattr(unit, "orders", [])
        )
        return {
            "requested_game_loop": requested_loop,
            "game_loop": int(self.state.game_loop),
            "time_seconds": round(float(self.time), 2),
            "resources": {
                "minerals": int(self.minerals),
                "vespene": int(self.vespene),
            },
            "supply": {
                "used": int(self.supply_used),
                "cap": int(self.supply_cap),
                "army": int(self.supply_army),
                "workers": int(self.supply_workers),
            },
            "workers": len(workers),
            "army_units": _counter_names(army_units),
            "structures_completed": _counter_names(completed_structures),
            "structures_in_progress": _counter_names(incomplete_structures),
            "upgrades": sorted(str(upgrade.name) for upgrade in self.state.upgrades),
            "active_orders": dict(sorted(active_orders.items())),
            "unit_schema": [
                "tag",
                "type",
                "x",
                "y",
                "health",
                "shield",
                "energy",
                "build_progress",
                "orders",
            ],
            "units": [_unit_row(unit) for unit in self.all_own_units],
        }

    async def on_step(self, iteration: int) -> None:
        current_loop = int(self.state.game_loop)
        while (
            self.target_index < len(self.target_loops)
            and self.target_loops[self.target_index] <= current_loop
        ):
            requested_loop = self.target_loops[self.target_index]
            self.snapshots.append(self._snapshot(requested_loop))
            self.target_index += 1

        if self.target_index >= len(self.target_loops):
            await self.client.leave()
            return

        next_loop = self.target_loops[self.target_index]
        self.client.game_step = max(1, next_loop - current_loop)


def extract_enemy_truth(
    record_path: str | Path,
    *,
    force: bool = False,
    commander_player_id: int = 1,
) -> Path:
    record_path = Path(record_path).resolve()
    output_path = enemy_truth_path(record_path)
    if output_path.is_file() and not force:
        return output_path

    replay_path = replay_path_for_record(record_path)
    if not record_path.is_file():
        raise FileNotFoundError(f"match record not found: {record_path}")
    if not replay_path.is_file():
        raise FileNotFoundError(f"replay not found: {replay_path}")

    record = json.loads(record_path.read_text(encoding="utf-8-sig"))
    if not isinstance(record, dict):
        raise ValueError(f"match record root must be an object: {record_path}")
    target_loops = commander_game_loops(record)
    if not target_loops:
        raise ValueError(f"no Commander game loops found: {record_path}")

    replay_metadata = _replay_metadata(replay_path)
    opponent_player_id = _opponent_player_id(replay_metadata, commander_player_id)
    bot = ReplayTruthBot(
        observed_player_id=opponent_player_id,
        target_loops=target_loops,
    )
    sc2_replay_path = prepare_replay_path_for_sc2(replay_path)
    run_replay(
        bot,
        str(sc2_replay_path),
        realtime=False,
        observed_id=opponent_player_id,
    )
    captured = {int(row["requested_game_loop"]) for row in bot.snapshots}
    missing = [game_loop for game_loop in target_loops if game_loop not in captured]
    if missing:
        raise RuntimeError(f"replay ended before Commander loops were captured: {missing}")

    payload = {
        "schema": "sc2_opponent_truth.v1",
        "source": "post_match_replay_observed_opponent",
        "record_file": record_path.name,
        "replay_file": replay_path.name,
        "commander_player_id": commander_player_id,
        "opponent_player_id": opponent_player_id,
        "game_version": replay_metadata.get("GameVersion"),
        "data_version": replay_metadata.get("DataVersion"),
        "snapshot_count": len(bot.snapshots),
        "snapshots": bot.snapshots,
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def _record_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value).resolve() for value in args.record]
    if args.batch_dir:
        paths.extend(
            path.resolve()
            for path in Path(args.batch_dir).rglob("*.json")
            if not path.name.endswith(TRUTH_SUFFIX)
        )
    return list(dict.fromkeys(paths))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract post-match opponent truth at Commander decision loops."
    )
    parser.add_argument("--record", action="append", default=[])
    parser.add_argument("--batch-dir", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = _record_paths(args)
    if not paths:
        parser.error("provide --record or --batch-dir")
    failures = 0
    for index, path in enumerate(paths, 1):
        try:
            output = extract_enemy_truth(path, force=args.force)
            print(f"[{index}/{len(paths)}] opponent truth: {output}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"[{index}/{len(paths)}] failed: {path}: {exc}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
