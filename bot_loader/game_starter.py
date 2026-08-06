import glob
import os
import random
import argparse
from configparser import ConfigParser
from datetime import datetime
from typing import List, Optional, Dict

from bot_loader.bot_definitions import BotDefinitions, races, builds, difficulty
from config import get_config
import sc2
from bot_loader.runner import MatchRunner
from sc2 import maps
from sc2.data import Result
from sc2.paths import Paths
from sc2.player import AbstractPlayer, Bot, Human
from sharpy.knowledges import KnowledgeBot
from sharpy.tools import LoggingUtility
from commander.match_defaults import DEFAULT_GAME_TIME_LIMIT_SECONDS

new_line = "\n"

# Used for random map selection
known_melee_maps = (
    "AbyssalReefLE",
    "AcolyteLE",
    "(2)RedshiftLE",
    "(2)DreamcatcherLE",
    "(2)LostandFoundLE",
    "AutomatonLE",
    "BlueshiftLE",
    "CeruleanFallLE",
    "DarknessSanctuaryLE",
    "KairosJunctionLE",
    "ParaSiteLE",
    "PortAleksanderLE",
    # "StasisLE", # Bugged map for bots
    "CyberForestLE",
    "KingsCoveLE",
    "NewRepugnancyLE",
    # Season 3 2019
    "AcropolisLE",
    "DiscoBloodbathLE",
    "EphemeronLE",
    "ThunderbirdLE",
    "TritonLE",
    "WintersGateLE",
    "WorldofSleepersLE",
    # Season 1 2020
    "SImulacrumLE",
    "ZenLE",
    "NightshadeLE",
)


class GameStarter:
    def __init__(self, definitions: BotDefinitions) -> None:
        self.config: ConfigParser = get_config()

        self.definitions = definitions
        self.players = definitions.playable
        self.random_bots = definitions.random_bots

        self.maps = GameStarter.installed_maps()
        self.random_maps = [x for x in known_melee_maps if x in self.maps]

    @staticmethod
    def installed_maps() -> List[str]:
        maps_folder = Paths.MAPS
        map_file_paths = glob.glob(f"{maps_folder}/**/*.SC2Map", recursive=True)

        def get_file_name(path) -> str:
            filename_w_ext = os.path.basename(path)
            filename, file_ext = os.path.splitext(filename_w_ext)
            return filename

        # Use a set to remove duplicate names (same map in multiple folders)
        map_file_names = set(map(get_file_name, map_file_paths))

        map_list = []
        for file_name in sorted(map_file_names):
            map_list.append(file_name)
        return map_list

    def play(self):
        # noinspection PyTypeChecker
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Run a game with custom parameters.",
            epilog=f"""\
Installed maps:
{new_line.join(sorted(self.maps))}

Bots:
{new_line.join(sorted(self.players.keys()))}


For ingame ai, use ai.race.difficulty.build where all arguments are optional
ingame ai defaults to ai.random.veryhard.random

Races:
{new_line.join(races.keys())}

Difficulties:
{new_line.join(difficulty.keys())}

Builds:
{new_line.join(builds.keys())}
                """,
        )

        parser.add_argument(
            "-m",
            "--map",
            help="Name of the map. Defaults to random. The script works with any map present in Starcraft 2 Maps directory.",
            default="random",
        )
        parser.add_argument(
            "-p1", "--player1", help="Name of player 1 bot or human. Defaults to random.", default="random"
        )
        parser.add_argument("-p2", "--player2", help="Name of player 2 bot. Defaults to random.", default="random")
        parser.add_argument("-rt", "--real-time", help="Use real-time mode.", action="store_true")
        parser.add_argument(
            "-r", "--release", help="Use only release config and ignore config local.", action="store_true"
        )
        parser.add_argument("-raw", "--raw_selection", help="Raw affects selection.", action="store_true")
        parser.add_argument(
            "--port", help="starting port to use, i.e. 10 would result in ports 10-17 being used to play."
        )

        parser.add_argument(
            "--requirewin", help="Requires victory for the specified player number (1 or 2) or raise exception."
        )
        parser.add_argument(
            "--record-dir",
            help="Directory where the log, replay, and LLM JSON for this match are written.",
        )
        parser.add_argument(
            "--match-id",
            help="File prefix for this match's log, replay, and LLM JSON.",
        )
        parser.add_argument(
            "--instruct",
            help="Natural-language tactical instruction for the LLM bot.",
            default="",
        )
        parser.add_argument(
            "--commander-model",
            help="Model key from llm/config.json for the Commander agent.",
            default="",
        )
        parser.add_argument(
            "--force-strategy",
            help="Force a strategy folder under skills/<race>/ (e.g. tank).",
            default="",
        )

        args = parser.parse_args()

        player1: str = args.player1

        if player1 == "random":
            player1 = random.choice(list(self.random_bots.keys()))
        elif "human" in player1:
            args.real_time = True
            args.release = True

        map_name = args.map
        if map_name == "random":
            map_name = random.choice(self.random_maps)

        if map_name not in self.maps:
            print(f"map not in Maps:{new_line}{new_line.join(self.maps)}")
            return

        player2: str = args.player2
        if player2 == "random":
            player2 = random.choice(list(self.random_bots.keys()))

        player2_split: List[str] = player2.split(".")
        player2_type: str = player2_split.pop(0)

        player1_split: List[str] = player1.split(".")
        player1_type: str = player1_split.pop(0)

        if player1_type not in self.definitions.player1:
            keys = list(self.definitions.player1.keys())
            print(f"Player1 type {player1} not found in:{new_line} {new_line.join(keys)}")
            return

        player2_bot: Optional[AbstractPlayer]

        if player2_type not in self.definitions.player2:
            keys = list(self.definitions.player2.keys())
            print(f"Enemy type {player2_type} not found in player types:{new_line}{new_line.join(keys)}")
            return
        else:
            player2_bot = self.players[player2_type](player2_split)

        player1_bot: AbstractPlayer = self.players[player1_type](player1_split)

        folder = os.path.abspath(args.record_dir) if args.record_dir else "game"
        os.makedirs(folder, exist_ok=True)

        # Prefer match_id as the file stem so json/replay share the folder name.
        # Compact match_id keeps the full path under Windows MAX_PATH.
        if args.match_id:
            file_stem = args.match_id
        else:
            time = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
            randomizer = random.randint(0, 999999)
            # Randomizer is to make it less likely that games started at the same time have same name
            file_stem = f"{player2}_{map_name}_{time}_{randomizer}"
        path = os.path.join(folder, f"{file_stem}.log")

        if self.config.getboolean("general", "log_file"):
            LoggingUtility.set_logger_file(log_level=self.config["general"]["log_level"], path=path)
        else:
            LoggingUtility.set_logger(log_level=self.config["general"]["log_level"])

        GameStarter.setup_bot(player1_bot, player1, player2, args)
        GameStarter.setup_bot(player2_bot, player2, player1, args)
        GameStarter._set_game_time_limit(
            player1_bot, DEFAULT_GAME_TIME_LIMIT_SECONDS
        )
        GameStarter._set_game_time_limit(
            player2_bot, DEFAULT_GAME_TIME_LIMIT_SECONDS
        )

        print(f"Starting game in {map_name}.")
        print(f"{player1} vs {player2}")

        replay_path = os.path.join(folder, f"{file_stem}.SC2Replay")
        json_path = os.path.join(folder, f"{file_stem}.json")
        # Tell each bot's LLMObservationRecorder where the replay will land so
        # that the JSON observation file is written with the exact same prefix
        # in the exact same folder.
        GameStarter._set_recorder_paths(player1_bot, replay_path, json_path)
        GameStarter._set_recorder_paths(player2_bot, replay_path, json_path)

        runner = MatchRunner()
        result = None
        try:
            result = runner.run_game(
                maps.get(map_name),
                [player1_bot, player2_bot],
                player1_id=player1,
                realtime=args.real_time,
                game_time_limit=DEFAULT_GAME_TIME_LIMIT_SECONDS,
                save_replay_as=replay_path,
                start_port=args.port,
            )
        finally:
            # Force-quit / disconnect may skip BotAI.on_end; still persist JSON.
            # Pass runner result when available. Recorder also remembers on_end
            # Result and skips None-overwrites after a successful on_end save.
            p1_result = GameStarter._player1_result(result)
            GameStarter._flush_recorders(
                player1_bot, reason="match_runner_finally", game_result=p1_result
            )
            GameStarter._flush_recorders(
                player2_bot, reason="match_runner_finally", game_result=None
            )

        if args.requirewin:
            if result is None:
                raise Exception("Game ended before a result was produced!")
            if args.requirewin == "1" and result != Result.Victory:
                raise Exception("Player 1 needed to win the game!")
            if args.requirewin == "2" and result != Result.Defeat:
                raise Exception("Player 2 needed to win the game!")
        # release file handle
        sc2.main.logger.remove()

    @staticmethod
    def setup_bot(player: AbstractPlayer, bot_code, enemy_text: str, args):
        if isinstance(player, Human):
            player.fullscreen = True
        if isinstance(player, Bot) and hasattr(player.ai, "config"):
            my_bot: KnowledgeBot = player.ai
            my_bot.opponent_id = bot_code + "-" + enemy_text
            my_bot.run_custom = True
            my_bot.raw_affects_selection = args.raw_selection
            if getattr(args, "record_dir", None):
                record_dir = os.path.abspath(args.record_dir)
                if hasattr(my_bot, "record_dir"):
                    my_bot.record_dir = record_dir
                recorder = getattr(my_bot, "llm_observation_recorder", None)
                if recorder is not None:
                    recorder.output_folder = record_dir
            if getattr(args, "instruct", None) and hasattr(my_bot, "instruct"):
                my_bot.instruct = args.instruct
            commander_model = getattr(args, "commander_model", None)
            if commander_model and hasattr(my_bot, "commander_model_key"):
                my_bot.commander_model_key = commander_model
            if hasattr(my_bot, "force_strategy"):
                fs = (getattr(args, "force_strategy", "") or "").strip()
                my_bot.force_strategy = fs if fs and fs.lower() != "none" else None
            if args.release:

                my_bot.config = get_config(False)

    @staticmethod
    def _set_game_time_limit(
        player: AbstractPlayer, limit_seconds: float
    ) -> None:
        """Expose the match timeout on the bot so observations can render it."""
        if not isinstance(player, Bot):
            return
        ai = getattr(player, "ai", None)
        if ai is not None:
            setattr(ai, "game_time_limit_seconds", float(limit_seconds))

    @staticmethod
    def _set_recorder_paths(
        player: AbstractPlayer, replay_path: str, json_path: str
    ) -> None:
        """Forward replay/json paths to the bot's LLM observation recorder."""
        if not isinstance(player, Bot):
            return
        recorder = getattr(player.ai, "llm_observation_recorder", None)
        if recorder is not None:
            recorder.replay_save_path = replay_path
            recorder.output_path = json_path

    @staticmethod
    def _player1_result(result) -> Optional[Result]:
        if isinstance(result, Result):
            return result
        if isinstance(result, (list, tuple)) and result:
            first = result[0]
            if isinstance(first, Result):
                return first
        return None

    @staticmethod
    def _flush_recorders(
        player: AbstractPlayer,
        reason: str = "manual",
        game_result: Optional[Result] = None,
    ) -> None:
        if not isinstance(player, Bot):
            return
        recorder = getattr(player.ai, "llm_observation_recorder", None)
        flush = getattr(recorder, "flush_to_disk", None)
        if callable(flush):
            try:
                flush(game_result=game_result, reason=reason)
            except Exception:
                pass

    @staticmethod
    def _set_recorder_replay_path(player: AbstractPlayer, replay_path: str) -> None:
        """Backward-compatible wrapper; prefer ``_set_recorder_paths``. """
        if not isinstance(player, Bot):
            return
        recorder = getattr(player.ai, "llm_observation_recorder", None)
        if recorder is not None:
            recorder.replay_save_path = replay_path
            base, _ext = os.path.splitext(replay_path)
            recorder.output_path = base + ".json"