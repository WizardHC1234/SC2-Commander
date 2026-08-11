from __future__ import annotations

import argparse
from pathlib import Path

from ..core.agent import EvolAgent
from ..core.config import DEFAULT_KNOWLEDGE_MODE, DEFAULT_MODEL, KNOWLEDGE_MODES
from ..core.types import EvolRunRequest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EvolAgent: per-match summaries, cross-match analysis, then strategy.md optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--record", action="append", default=[], help="Record JSON path; may be repeated")
    parser.add_argument("--batch-dir", default="", help="Directory recursively containing record JSON files")
    parser.add_argument("--strategy", default="", help="Required when selected records contain multiple strategies")
    parser.add_argument("--race", default="terran")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Override analysis and optimization models (llm/config.json pool key)",
    )
    parser.add_argument("--output-dir", default="", help="Optional optimized strategy directory")
    parser.add_argument(
        "--knowledge-mode",
        choices=KNOWLEDGE_MODES,
        default=DEFAULT_KNOWLEDGE_MODE,
        help="Enable or ablate deterministic SC2 knowledge",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run match summaries and cross-match analysis only; do not optimize or write a strategy",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Resume from an EvolAgent run directory containing checkpoint.json",
    )
    args = parser.parse_args()
    request = EvolRunRequest(
        record_paths=[Path(path) for path in args.record],
        batch_dir=Path(args.batch_dir) if args.batch_dir else None,
        strategy_name=args.strategy,
        race=args.race,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        model=args.model,
        knowledge_mode=args.knowledge_mode,
        dry_run=args.dry_run,
        resume_dir=Path(args.resume) if args.resume else None,
    )
    try:
        result = EvolAgent(model=args.model).run(request)
    except KeyboardInterrupt:
        from ..core.loop_helpers import exit_on_keyboard_interrupt

        exit_on_keyboard_interrupt()
    print(f"[{'OK' if result.ok else 'ERROR'}] {result.message}")
    if result.output_dir:
        print(f"Output: {result.output_dir}")
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
