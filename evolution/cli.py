from __future__ import annotations

import argparse
from pathlib import Path

from .runner import DEFAULT_DIFFICULTIES, EvolutionConfig, EvolutionRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run automatic SC2 strategy evolution")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--commander-model", required=True)
    parser.add_argument("--evolution-model", default="")
    parser.add_argument("--race", default="terran")
    parser.add_argument("--enemy-race", default="terran")
    parser.add_argument("--enemy-build", default="macro")
    parser.add_argument("--map", dest="map_name", default="KairosJunctionLE")
    parser.add_argument("--difficulties", default=",".join(DEFAULT_DIFFICULTIES))
    parser.add_argument("--matches", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--pass-score", type=float, default=0.8)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--knowledge-mode", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--run-dir", default="", help="Existing run directory to resume")
    parser.add_argument("--real-time", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    difficulties = tuple(item.strip().lower() for item in args.difficulties.split(",") if item.strip())
    config = EvolutionConfig(
        strategy=args.strategy,
        commander_model=args.commander_model,
        evolution_model=args.evolution_model,
        race=args.race,
        enemy_race=args.enemy_race,
        enemy_build=args.enemy_build,
        map_name=args.map_name,
        difficulties=difficulties,
        matches_per_batch=args.matches,
        concurrency=args.concurrency,
        pass_score=args.pass_score,
        max_generations=args.max_generations,
        knowledge_mode=args.knowledge_mode,
        real_time=args.real_time,
    )
    runner = EvolutionRunner(config, run_dir=Path(args.run_dir) if args.run_dir else None)
    state = runner.run()
    print(f"Evolution status: {state['status']}")
    print(f"Champion: {state['champion']}")
    print(f"Games used: {state['games_used']}")
    print(f"Run directory: {runner.run_dir}")


if __name__ == "__main__":
    main()
