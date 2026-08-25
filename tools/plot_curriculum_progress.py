#!/usr/bin/env python3
"""Plot curriculum-progress curves for strategy-evolution experiments.

The script reads one ``history.csv`` file per strategy, keeps the last record
for each generation, and produces a compact ICASSP-style figure.  Difficulty
levels are encoded as low-saturation horizontal background bands, while line
color and marker shape identify the strategy.

Example:
    python tools/plot_curriculum_progress.py \
        --series Marine=evolution_runs/marine/RUN_ID/history.csv \
        --series Tank=evolution_runs/tank/RUN_ID/history.csv \
        --series Battlecruiser=evolution_runs/battlecruiser/RUN_ID/history.csv \
        --rounds 10 \
        --output analysis_results/strategy_evolution

Preview the single-column layout with synthetic data:
    python tools/plot_curriculum_progress.py --demo \
        --rounds 10 --y-max 3 \
        --output analysis_results/demo_strategy_evolution
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


DIFFICULTIES: tuple[str, ...] = (
    "Harder",
    "VeryHard",
    "CheatVision",
    "CheatMoney",
    "CheatInsane",
)

# Large-area fills are deliberately pale; they should remain secondary to data.
DIFFICULTY_COLORS: tuple[str, ...] = (
    "#4DA3FF",  # bright sky blue
    "#36C7B7",  # bright turquoise
    "#8BCF5B",  # fresh green
    "#FFC857",  # warm gold
    "#E78AAF",  # bright rose
)
DIFFICULTY_ALPHA = 0.09

# Bright, color-blind-friendly line colors paired with distinct marker shapes.
SERIES_STYLES: tuple[tuple[str, str], ...] = (
    ("#2F6BFF", "o"),  # Marine: blue circle
    ("#F28E2B", "s"),  # Tank: orange square
    ("#10A58A", "D"),  # Battlecruiser: teal diamond
)


@dataclass(frozen=True)
class ProgressPoint:
    generation: int
    score: float
    accepted: bool | None
    strategy: str
    difficulty: str


@dataclass(frozen=True)
class ProgressSeries:
    label: str
    source: Path
    points: tuple[ProgressPoint, ...]


def _parse_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _resolve_history_path(source: Path) -> Path:
    if source.is_dir():
        source = source / "history.csv"
    if not source.is_file():
        raise FileNotFoundError(f"history.csv not found: {source}")
    return source


def load_progress_series(label: str, source: Path, rounds: int) -> ProgressSeries:
    """Load at most one final state per generation from an evolution history."""

    history_path = _resolve_history_path(source)
    by_generation: dict[int, tuple[int, ProgressPoint]] = {}
    observed_generations: list[int] = []

    with history_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"generation", "curriculum_progress_score"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{history_path} is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        for row_index, row in enumerate(reader):
            try:
                generation = int(row["generation"])
                score = float(row["curriculum_progress_score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid generation or curriculum score at "
                    f"{history_path}:{row_index + 2}"
                ) from exc

            if generation < 0 or generation > rounds:
                continue
            if not math.isfinite(score):
                raise ValueError(
                    f"Non-finite curriculum score at {history_path}:{row_index + 2}"
                )

            observed_generations.append(generation)
            point = ProgressPoint(
                generation=generation,
                score=score,
                accepted=_parse_bool(row.get("accepted", "")),
                strategy=row.get("strategy", "").strip(),
                difficulty=row.get("difficulty", "").strip(),
            )
            # A difficulty transition can add another row for the same generation.
            # The last row is the state visible at the end of that evolution round.
            by_generation[generation] = (row_index, point)

    if not by_generation:
        raise ValueError(f"No generations in [0, {rounds}] found in {history_path}")

    if any(b < a for a, b in zip(observed_generations, observed_generations[1:])):
        print(
            f"warning: non-monotonic generation order in {history_path}; "
            "plotting the last row for each generation",
            file=sys.stderr,
        )

    points = tuple(by_generation[generation][1] for generation in sorted(by_generation))
    return ProgressSeries(label=label, source=history_path, points=points)


def _parse_series_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "series must use LABEL=PATH, for example Tank=evolution_runs/tank/RUN_ID"
        )
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("both LABEL and PATH must be non-empty")
    return label, Path(raw_path)


def _automatic_y_max(series: Sequence[ProgressSeries]) -> int:
    maximum = max(point.score for item in series for point in item.points)
    # If a point lies exactly on a mastery boundary, reveal the next band too.
    upper = math.ceil(maximum + 1e-6)
    return max(1, min(len(DIFFICULTIES), upper))


def _spread_endpoint_labels(values: Sequence[float], upper: float) -> list[float]:
    """Separate endpoint labels vertically while keeping them inside the axes."""

    if not values:
        return []
    minimum_gap = max(0.10, upper * 0.055)
    lower_bound = upper * 0.04
    upper_bound = upper * 0.96
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    positions = [min(max(value, lower_bound), upper_bound) for _, value in indexed]

    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + minimum_gap)
    overflow = positions[-1] - upper_bound
    if overflow > 0:
        positions = [position - overflow for position in positions]
        for index in range(len(positions) - 2, -1, -1):
            positions[index] = min(positions[index], positions[index + 1] - minimum_gap)

    result = [0.0] * len(values)
    for (original_index, _), position in zip(indexed, positions):
        result[original_index] = position
    return result


def _curriculum_progress(mastered_levels: int, wins: int, games: int) -> float:
    """Compute the paper metric from discrete match outcomes."""

    return mastered_levels + min((wins / games) / 0.9, 1.0)


def build_demo_series(rounds: int) -> list[ProgressSeries]:
    """Return formula-valid synthetic data for visual QA only.

    Each tuple is ``(mastered_levels, wins, games)``.  Scores are deliberately
    computed here instead of being handwritten so every preview point is
    attainable under the paper's evaluation protocol.
    """

    demo_outcomes = {
        "Marine": [
            (0, 17, 20), (1, 17, 20), (2, 5, 10), (2, 3, 10),
            (3, 4, 10), (3, 2, 10), (4, 3, 10), (4, 2, 10),
            (4, 5, 10), (4, 4, 10), (4, 6, 10),
        ],
        "Tank": [
            (0, 6, 10), (0, 8, 14), (0, 7, 10), (1, 2, 10),
            (1, 4, 10), (1, 3, 10), (1, 7, 10), (2, 1, 10),
            (2, 3, 10), (2, 2, 10), (2, 6, 10),
        ],
        "Battlecruiser": [
            (0, 5, 10), (0, 4, 10), (0, 5, 10), (0, 4, 10),
            (0, 8, 14), (0, 6, 10), (0, 8, 14), (0, 7, 10),
            (0, 6, 10), (0, 7, 10), (0, 8, 10),
        ],
    }
    demo_acceptance = {
        "Marine": [True, True, True, True, True, True, True, True, True, True, True],
        "Tank": [True, False, True, True, True, True, False, True, False, True, True],
        "Battlecruiser": [True, True, True, True, False, True, True, True, True, False, True],
    }

    result: list[ProgressSeries] = []
    for label, outcomes in demo_outcomes.items():
        limit = min(rounds + 1, len(outcomes))
        points = tuple(
            ProgressPoint(
                generation=generation,
                score=_curriculum_progress(*outcomes[generation]),
                accepted=demo_acceptance[label][generation],
                strategy=f"demo_{label.lower()}_{generation}",
                difficulty="demo_only",
            )
            for generation in range(limit)
        )
        result.append(
            ProgressSeries(label=label, source=Path("<synthetic-demo>"), points=points)
        )
    return result


def plot_curriculum_progress(
    series: Sequence[ProgressSeries],
    output_stem: Path,
    rounds: int,
    y_max: int | None,
    width: float,
    height: float,
    strategy_legend: bool,
) -> tuple[Path, Path]:
    if not series:
        raise ValueError("at least one series is required")
    if len(series) > len(SERIES_STYLES):
        raise ValueError(f"at most {len(SERIES_STYLES)} strategy series are supported")

    upper = y_max if y_max is not None else _automatic_y_max(series)
    if not 1 <= upper <= len(DIFFICULTIES):
        raise ValueError(f"y-max must be between 1 and {len(DIFFICULTIES)}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # ICASSP single-column width is approximately 3.45 inches.
    fig, ax = plt.subplots(figsize=(width, height))

    for level in range(upper):
        ax.axhspan(
            level,
            level + 1,
            facecolor=DIFFICULTY_COLORS[level],
            edgecolor="none",
            alpha=DIFFICULTY_ALPHA,
            zorder=0,
        )
        ax.text(
            0.015,
            level + 0.06,
            DIFFICULTIES[level],
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            color="#737A81",
            fontsize=5.9,
            fontweight="normal",
            zorder=1,
        )

    line_handles: list[Line2D] = []
    endpoint_values: list[float] = []
    for index, item in enumerate(series):
        color, marker = SERIES_STYLES[index]
        x_values = [point.generation for point in item.points]
        y_values = [point.score for point in item.points]

        (line,) = ax.plot(
            x_values,
            y_values,
            color=color,
            linewidth=1.25,
            marker=marker,
            markersize=4.6,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.45,
            label=item.label,
            zorder=3,
        )
        line_handles.append(line)
        endpoint_values.append(y_values[-1])

    for boundary in range(upper + 1):
        ax.axhline(
            boundary,
            color="#A1A8AE",
            linewidth=0.65 if boundary in {0, upper} else 0.55,
            linestyle="-" if boundary in {0, upper} else (0, (3, 2)),
            alpha=0.75,
            zorder=2,
        )

    right_margin = 0.2 if strategy_legend else 1.45
    ax.set_xlim(-0.2, rounds + right_margin)
    ax.set_ylim(0, upper)
    ax.set_xticks(range(0, rounds + 1, 2))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlabel("Evolution round")
    ax.set_ylabel("Curriculum progress score")

    ax.grid(
        axis="x",
        color="#D7DFE6",
        linewidth=0.45,
        linestyle=(0, (1.5, 2.5)),
        alpha=0.85,
        zorder=1,
    )
    ax.tick_params(axis="both", direction="out", length=2.5, width=0.6)
    for spine in ax.spines.values():
        spine.set_color("#4F5964")
        spine.set_linewidth(0.65)

    if strategy_legend:
        legend_handles: list[Line2D] = list(line_handles)
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=min(2, len(legend_handles)),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.3,
            borderaxespad=0.0,
        )
    else:
        label_positions = _spread_endpoint_labels(endpoint_values, float(upper))
        label_x = rounds + 0.20
        for index, (item, label_y) in enumerate(zip(series, label_positions)):
            color, _ = SERIES_STYLES[index]
            endpoint = item.points[-1]
            ax.annotate(
                item.label,
                xy=(endpoint.generation, endpoint.score),
                xytext=(label_x, label_y),
                textcoords="data",
                ha="left",
                va="center",
                color=color,
                fontsize=7.0,
                fontweight="semibold",
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "linewidth": 0.65,
                    "shrinkA": 1.5,
                    "shrinkB": 2.5,
                },
                zorder=5,
            )

    output_stem = output_stem.with_suffix("")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.tight_layout(pad=0.35)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot curriculum progress for up to three strategy runs."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--series",
        action="append",
        metavar="LABEL=PATH",
        help="strategy label and run directory/history.csv; repeat for each strategy",
    )
    source_group.add_argument(
        "--demo",
        action="store_true",
        help="use synthetic three-strategy data for visual QA only",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="maximum evolution round to include (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_results/strategy_evolution"),
        help="output path stem; both PDF and PNG are written",
    )
    parser.add_argument(
        "--y-max",
        type=int,
        default=None,
        help="number of difficulty bands to show (1-5); default: infer from data",
    )
    parser.add_argument(
        "--strategy-legend",
        action="store_true",
        help="use a conventional strategy legend instead of direct endpoint labels",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=3.45,
        help="figure width in inches (default: 3.45, ICASSP single column)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=2.55,
        help="figure height in inches (default: 2.55)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rounds < 0:
        raise SystemExit("--rounds must be non-negative")

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")

    try:
        if args.demo:
            loaded = build_demo_series(args.rounds)
        else:
            parsed_specs = [_parse_series_spec(spec) for spec in args.series]
            labels = [label for label, _ in parsed_specs]
            if len(set(labels)) != len(labels):
                raise ValueError("strategy labels passed to --series must be unique")
            loaded = [
                load_progress_series(label, source, args.rounds)
                for label, source in parsed_specs
            ]
        pdf_path, png_path = plot_curriculum_progress(
            loaded,
            output_stem=args.output,
            rounds=args.rounds,
            y_max=args.y_max,
            width=args.width,
            height=args.height,
            strategy_legend=args.strategy_legend,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for item in loaded:
        generations = ",".join(str(point.generation) for point in item.points)
        print(f"{item.label}: {len(item.points)} point(s), generations={generations}")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
