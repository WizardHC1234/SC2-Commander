from __future__ import annotations

import argparse
from pathlib import Path

from .runner import (
    ANALYSIS_EXPERIENCE_MODES,
    DEFAULT_DIFFICULTIES,
    EvolutionConfig,
    EvolutionRunner,
)


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
    parser.add_argument("--candidate-matches", type=int, default=10)
    parser.add_argument(
        "--candidate-generation-retries",
        type=int,
        default=3,
        help="Retry a failed candidate-generation attempt this many times with its failure feedback",
    )
    parser.add_argument(
        "--confirmation-matches",
        type=int,
        default=4,
        help="Extra games per strategy when initial results differ by at most one outcome point; 0 disables confirmation",
    )
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--mastery-score-threshold", type=float, default=0.90)
    parser.add_argument("--analysis-batch-games", type=int, default=10)
    parser.add_argument("--max-analysis-games-per-generation", type=int, default=20)
    parser.add_argument(
        "--analysis-experience-mode",
        choices=ANALYSIS_EXPERIENCE_MODES,
        default="multi_match",
        help=(
            "multi_match analyzes every supplied trajectory; single_failure "
            "analyzes one deterministically sampled failed trajectory while "
            "retaining all matches for evaluation"
        ),
    )
    parser.add_argument(
        "--analysis-sample-seed",
        type=int,
        default=0,
        help="Fixed seed used by the single_failure ablation",
    )
    parser.add_argument("--max-generations-per-difficulty", type=int, default=10)
    parser.add_argument("--max-total-generations", "--max-generations", type=int, default=50)
    parser.add_argument(
        "--require-full-generation-budget",
        action="store_true",
        help=(
            "Do not stop after curriculum mastery or a per-difficulty budget; "
            "continue until max-total-generations is reached"
        ),
    )
    parser.add_argument("--knowledge-mode", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--run-dir", default="", help="Existing run directory to resume")
    parser.add_argument(
        "--records-dir",
        default="game_records",
        help="Root directory for match records; relative paths are resolved from the project root",
    )
    parser.add_argument(
        "--baseline-batch-dir",
        default="",
        help="Seed a new run from one already-completed champion batch",
    )
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
        candidate_matches=args.candidate_matches,
        candidate_generation_retries=args.candidate_generation_retries,
        confirmation_matches=args.confirmation_matches,
        concurrency=args.concurrency,
        mastery_score_threshold=args.mastery_score_threshold,
        analysis_batch_games=args.analysis_batch_games,
        max_analysis_games_per_generation=args.max_analysis_games_per_generation,
        analysis_experience_mode=args.analysis_experience_mode,
        analysis_sample_seed=args.analysis_sample_seed,
        max_generations_per_difficulty=args.max_generations_per_difficulty,
        max_total_generations=args.max_total_generations,
        require_full_generation_budget=args.require_full_generation_budget,
        knowledge_mode=args.knowledge_mode,
        real_time=args.real_time,
        baseline_batch_dir=args.baseline_batch_dir,
        records_dir=args.records_dir,
    )
    runner = EvolutionRunner(config, run_dir=Path(args.run_dir) if args.run_dir else None)
    state = runner.run()
    print(f"Evolution status: {state['status']}")
    print(f"Champion: {state['champion']}")
    print(f"Games used: {state['games_used']}")
    print(f"Run directory: {runner.run_dir}")


if __name__ == "__main__":
    main()
