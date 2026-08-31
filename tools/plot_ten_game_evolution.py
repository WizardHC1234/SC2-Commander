#!/usr/bin/env python3
"""Plot curriculum progress with one x-step per 10-game evaluation.

Unlike ``plot_curriculum_progress.py`` (x = evolution generation), this script
walks ``history.csv`` in order and treats **every evaluation row** as one step.
A step corresponds to a ~10-game batch (confirm-merged 14-game rows still count
as a single step). Difficulty climbs within the same generation therefore appear
as separate points.

Example:
    python tools/plot_ten_game_evolution.py \
        --series Marine=evolution_runs/marine/20260830_182434 \
        --series Battlecruiser=evolution_runs/battlecruiser/20260830_173416 \
        --max-evals 10 \
        --output analysis_results/ten_game_evolution
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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

DIFFICULTY_COLORS: tuple[str, ...] = (
    "#4DA3FF",
    "#36C7B7",
    "#8BCF5B",
    "#FFC857",
    "#E78AAF",
)
DIFFICULTY_ALPHA = 0.09

SERIES_STYLES: tuple[tuple[str, str], ...] = (
    ("#2F6BFF", "o"),
    ("#F28E2B", "s"),
    ("#10A58A", "D"),
)

GAMES_PER_EVAL = 10


@dataclass(frozen=True)
class EvalPoint:
    eval_index: int
    score: float
    accepted: bool | None
    strategy: str
    difficulty: str
    games: int


@dataclass(frozen=True)
class EvalSeries:
    label: str
    source: Path
    points: tuple[EvalPoint, ...]


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


def _curriculum_progress(mastered_levels: int, wins: int, games: int) -> float:
    if games <= 0:
        raise ValueError("games must be positive")
    return mastered_levels + min((wins / games) / 0.9, 1.0)


def load_ten_game_series(label: str, source: Path, max_evals: int) -> EvalSeries:
    """Load chronological evaluation rows; one x-step per history record."""

    history_path = _resolve_history_path(source)
    points: list[EvalPoint] = []

    with history_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"curriculum_progress_score", "games"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{history_path} is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        for row_index, row in enumerate(reader):
            try:
                score = float(row["curriculum_progress_score"])
                games = int(row["games"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid score/games at {history_path}:{row_index + 2}"
                ) from exc
            if games <= 0 or not math.isfinite(score):
                raise ValueError(
                    f"Non-positive games or non-finite score at "
                    f"{history_path}:{row_index + 2}"
                )

            # Confirm-merged rows (e.g. 14 games) still count as one step; the
            # CSV curriculum_progress_score remains the plotted y-value.
            eval_index = len(points)
            if eval_index > max_evals:
                break

            points.append(
                EvalPoint(
                    eval_index=eval_index,
                    score=score,
                    accepted=_parse_bool(row.get("accepted", "")),
                    strategy=str(row.get("strategy") or "").strip(),
                    difficulty=str(row.get("difficulty") or "").strip(),
                    games=games,
                )
            )

    if not points:
        raise ValueError(f"No evaluation rows found in {history_path}")

    return EvalSeries(label=label, source=history_path, points=tuple(points))


def _parse_series_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "series must use LABEL=PATH, for example Marine=evolution_runs/marine/RUN_ID"
        )
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("both LABEL and PATH must be non-empty")
    return label, Path(raw_path)


def _automatic_y_max(series: Sequence[EvalSeries]) -> int:
    maximum = max(point.score for item in series for point in item.points)
    upper = math.ceil(maximum + 1e-6)
    return max(1, min(len(DIFFICULTIES), upper))


def _spread_endpoint_labels(values: Sequence[float], upper: float) -> list[float]:
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


def build_demo_series(max_evals: int) -> list[EvalSeries]:
    demo_outcomes = {
        "Marine": [
            (0, 8, 10), (0, 9, 10), (1, 10, 10), (2, 9, 10),
            (3, 2, 10), (3, 5, 10), (3, 8, 10), (4, 3, 10),
            (4, 4, 10), (4, 6, 10), (4, 7, 10),
        ],
        "Tank": [
            (0, 6, 10), (0, 7, 10), (0, 8, 10), (1, 4, 10),
            (1, 5, 10), (1, 3, 10), (2, 2, 10), (2, 4, 10),
            (2, 5, 10), (2, 6, 10), (2, 7, 10),
        ],
        "Battlecruiser": [
            (0, 8, 10), (0, 10, 10), (1, 6, 10), (1, 7, 10),
            (1, 8, 10), (2, 3, 10), (2, 5, 10), (2, 6, 10),
            (3, 2, 10), (3, 4, 10), (3, 5, 10),
        ],
    }
    result: list[EvalSeries] = []
    for label, outcomes in demo_outcomes.items():
        limit = min(max_evals + 1, len(outcomes))
        points = tuple(
            EvalPoint(
                eval_index=index,
                score=_curriculum_progress(*outcomes[index]),
                accepted=True,
                strategy=f"demo_{label.lower()}_{index}",
                difficulty="demo",
                games=GAMES_PER_EVAL,
            )
            for index in range(limit)
        )
        result.append(
            EvalSeries(label=label, source=Path("<synthetic-demo>"), points=points)
        )
    return result


def plot_ten_game_evolution(
    series: Sequence[EvalSeries],
    output_stem: Path,
    max_evals: int,
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

    observed_max = max(point.eval_index for item in series for point in item.points)
    x_right = max(max_evals, observed_max)

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
        x_values = [point.eval_index for point in item.points]
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
    ax.set_xlim(-0.2, x_right + right_margin)
    ax.set_ylim(0, upper)
    ax.set_xticks(range(0, x_right + 1, 2 if x_right >= 6 else 1))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlabel("Iteration (evaluation round)")
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
        ax.legend(
            handles=list(line_handles),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=min(2, len(line_handles)),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.3,
            borderaxespad=0.0,
        )
    else:
        label_positions = _spread_endpoint_labels(endpoint_values, float(upper))
        label_x = x_right + 0.20
        for index, (item, label_y) in enumerate(zip(series, label_positions)):
            color, _ = SERIES_STYLES[index]
            endpoint = item.points[-1]
            ax.annotate(
                item.label,
                xy=(endpoint.eval_index, endpoint.score),
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
        description=(
            "Plot curriculum progress with one x-step per 10-game evaluation "
            "(chronological history rows, not generation ids)."
        )
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
        "--max-evals",
        type=int,
        default=10,
        help="maximum 10-game evaluation index to include (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_results/ten_game_evolution"),
        help="output path stem; both PDF and PNG are written",
    )
    parser.add_argument(
        "--y-max",
        type=int,
        default=None,
        help="force curriculum y-axis upper bound (1-5); default is data-driven",
    )
    parser.add_argument("--width", type=float, default=3.45, help="figure width in inches")
    parser.add_argument("--height", type=float, default=2.35, help="figure height in inches")
    parser.add_argument(
        "--strategy-legend",
        action="store_true",
        help="place a top legend instead of endpoint labels",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_evals < 0:
        print("--max-evals must be >= 0", file=sys.stderr)
        return 2

    if args.demo:
        series = build_demo_series(args.max_evals)
    else:
        series = [
            load_ten_game_series(label, path, args.max_evals)
            for label, path in (_parse_series_spec(spec) for spec in args.series)
        ]

    for item in series:
        print(
            f"{item.label}: {len(item.points)} eval(s), "
            f"indices={','.join(str(p.eval_index) for p in item.points)}",
            file=sys.stderr,
        )

    pdf_path, png_path = plot_ten_game_evolution(
        series,
        output_stem=args.output,
        max_evals=args.max_evals,
        y_max=args.y_max,
        width=args.width,
        height=args.height,
        strategy_legend=args.strategy_legend,
    )
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
