"""Build reproducible SC2-LSEE evolution and strategy-consistency reports.

The report deliberately keeps outcome performance and strategy-execution
consistency as separate quantities.  Evolution data comes from ``history.csv``
and ``generation_*/decision.json``.  Consistency data is the ``per_game.csv``
produced by :mod:`tools.strategy_execution_metrics`.

Outputs are tidy CSV files, a machine-readable JSON summary, and paper-ready
PNG/PDF figures.  Plotting is optional so all calculations remain testable on
machines where matplotlib has not yet been installed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


DIFFICULTY_ORDER = (
    "veryeasy",
    "easy",
    "medium",
    "mediumhard",
    "hard",
    "harder",
    "veryhard",
    "cheatvision",
    "cheatmoney",
    "cheatinsane",
)

CONSISTENCY_METRICS = (
    "economy_completion",
    "technology_completion",
    "army_completion",
    "engagement_trigger_consistency",
    "engagement_continuation_consistency",
    "overall_strategy_compliance",
)

METRIC_LABELS = {
    "economy_completion": "Economy",
    "technology_completion": "Technology",
    "army_completion": "Army",
    "engagement_trigger_consistency": "Attack trigger",
    "engagement_continuation_consistency": "Continuation",
    "overall_strategy_compliance": "Overall",
}

PALETTE = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
)


def _float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _int(value: Any, default: int = 0) -> int:
    parsed = _float(value)
    return int(parsed) if parsed is not None else default


def _bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    extras = sorted({key for row in rows for key in row} - set(fields))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extras)
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[Optional[float], Optional[float]]:
    """Return the Wilson 95% interval for a binomial proportion."""
    if trials <= 0:
        return None, None
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def mean_interval(
    values: Iterable[float],
    z: float = 1.959963984540054,
) -> tuple[Optional[float], Optional[float], Optional[float], float]:
    """Return n, mean, and clipped normal-approximation 95% bounds."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None, None, None, 0.0
    mean = statistics.fmean(finite)
    sd = statistics.stdev(finite) if len(finite) >= 2 else 0.0
    margin = z * sd / math.sqrt(len(finite))
    return mean, max(0.0, mean - margin), min(1.0, mean + margin), sd


def difficulty_sort_key(value: str) -> tuple[int, str]:
    normalized = str(value or "unknown").lower()
    try:
        return DIFFICULTY_ORDER.index(normalized), normalized
    except ValueError:
        return len(DIFFICULTY_ORDER), normalized


def load_evolution_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative_games = 0
    for index, raw in enumerate(_read_csv(path)):
        games = _int(raw.get("games"))
        wins = _int(raw.get("wins"))
        draws = _int(raw.get("draws"))
        losses = _int(raw.get("losses"))
        win_rate = _float(raw.get("win_rate"))
        if win_rate is None and games > 0:
            win_rate = wins / games
        score = _float(raw.get("score"))
        if score is None and games > 0:
            score = (wins + 0.5 * draws) / games
        ci_low, ci_high = wilson_interval(wins, games)
        cumulative_games += games
        rows.append(
            {
                "evaluation_index": index,
                "strategy_style": raw.get("strategy_style", ""),
                "generation": _int(raw.get("generation")),
                "strategy": raw.get("strategy", ""),
                "parent": raw.get("parent", ""),
                "difficulty": raw.get("difficulty", "unknown"),
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "games": games,
                "score": score,
                "win_rate": win_rate,
                "win_rate_ci_low": ci_low,
                "win_rate_ci_high": ci_high,
                "mastered_levels": _int(raw.get("mastered_levels")),
                "curriculum_progress_score": _float(
                    raw.get("curriculum_progress_score")
                ),
                "accepted": _bool(raw.get("accepted")),
                "batch": raw.get("batch", ""),
                "cumulative_evaluation_games": cumulative_games,
            }
        )
    return rows


def load_decisions(run_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("generation_*/decision.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        candidate_games = _int(raw.get("candidate_evidence_games"))
        champion_games = _int(raw.get("champion_evidence_games"))
        output.append(
            {
                "generation": _int(raw.get("generation")),
                "difficulty": raw.get("difficulty", "unknown"),
                "parent": raw.get("parent", ""),
                "candidate": raw.get("candidate", ""),
                "parent_score": _float(raw.get("parent_score")),
                "candidate_score": _float(raw.get("candidate_score")),
                "score_delta": _float(
                    raw.get("score_delta"), _float(raw.get("delta"))
                ),
                "decision": raw.get("decision", ""),
                "accepted": _bool(raw.get("accepted")),
                "implementation_verdict": raw.get(
                    "implementation_verdict", ""
                ),
                "hypothesis_verdict": raw.get("hypothesis_verdict", ""),
                "candidate_games": candidate_games,
                "champion_games": champion_games,
                "comparison_games": candidate_games + champion_games,
                "confirmation_used": bool(raw.get("confirmation")),
                "decision_path": str(path),
            }
        )
    return output


def summarize_consistency(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate strategy compliance without mixing it into win rate."""
    metric_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    result_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strategy = str(row.get("strategy") or "unknown")
        difficulty = str(row.get("difficulty") or "unknown")
        result_groups[(strategy, difficulty)].append(row)
        for metric in CONSISTENCY_METRICS:
            value = _float(row.get(metric))
            if value is not None:
                metric_groups[(strategy, difficulty, metric)].append(value)

    metric_summary: list[dict[str, Any]] = []
    for (strategy, difficulty, metric), values in sorted(
        metric_groups.items(),
        key=lambda item: (
            item[0][0],
            difficulty_sort_key(item[0][1]),
            CONSISTENCY_METRICS.index(item[0][2]),
        ),
    ):
        mean, low, high, sd = mean_interval(values)
        metric_summary.append(
            {
                "strategy": strategy,
                "difficulty": difficulty,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "n": len(values),
                "mean": mean,
                "sd": sd,
                "ci_low": low,
                "ci_high": high,
            }
        )

    relation_summary: list[dict[str, Any]] = []
    outcome_summary: list[dict[str, Any]] = []
    for (strategy, difficulty), group in sorted(
        result_groups.items(),
        key=lambda item: (item[0][0], difficulty_sort_key(item[0][1])),
    ):
        results = [str(row.get("result") or "").strip().lower() for row in group]
        wins = sum(result in {"victory", "win"} for result in results)
        draws = sum(result == "draw" for result in results)
        losses = sum(result in {"defeat", "loss"} for result in results)
        games = wins + draws + losses
        overall_values = [
            value
            for row in group
            if (value := _float(row.get("overall_strategy_compliance")))
            is not None
        ]
        mean, low, high, sd = mean_interval(overall_values)
        win_low, win_high = wilson_interval(wins, games)
        relation_summary.append(
            {
                "strategy": strategy,
                "difficulty": difficulty,
                "games": games,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": wins / games if games else None,
                "win_rate_ci_low": win_low,
                "win_rate_ci_high": win_high,
                "overall_strategy_compliance": mean,
                "compliance_sd": sd,
                "compliance_ci_low": low,
                "compliance_ci_high": high,
            }
        )
        by_outcome: dict[str, list[float]] = defaultdict(list)
        for row, result in zip(group, results):
            value = _float(row.get("overall_strategy_compliance"))
            if value is not None and result:
                by_outcome[result].append(value)
        for result, values in sorted(by_outcome.items()):
            outcome_mean, outcome_low, outcome_high, outcome_sd = mean_interval(values)
            outcome_summary.append(
                {
                    "strategy": strategy,
                    "difficulty": difficulty,
                    "result": result,
                    "n": len(values),
                    "mean": outcome_mean,
                    "sd": outcome_sd,
                    "ci_low": outcome_low,
                    "ci_high": outcome_high,
                }
            )
    return metric_summary, relation_summary, outcome_summary


def _matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for figures; install requirements.txt or "
            "rerun with --no-plots"
        ) from exc
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _save_figure(
    figure: Any,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for extension in formats:
        path = output_base.with_suffix(f".{extension}")
        kwargs = {"bbox_inches": "tight"}
        if extension.lower() == "png":
            kwargs["dpi"] = dpi
        figure.savefig(path, **kwargs)
        paths.append(str(path))
    return paths


def plot_evolution_progress(
    rows: list[dict[str, Any]],
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    plt = _matplotlib()
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    xs = [row["evaluation_index"] for row in rows]
    progress = [row["curriculum_progress_score"] for row in rows]
    axes[0].plot(xs, progress, color=PALETTE[0], linewidth=1.7, zorder=1)
    for row in rows:
        accepted = row["accepted"]
        axes[0].scatter(
            row["evaluation_index"],
            row["curriculum_progress_score"],
            s=42,
            marker="o" if accepted else "X",
            facecolor=PALETTE[0] if accepted else "white",
            edgecolor=PALETTE[0] if accepted else PALETTE[3],
            linewidth=1.2,
            zorder=3,
        )
    axes[0].set_ylabel("Curriculum progress score")
    axes[0].set_title("(a) Progress across evaluated strategies", loc="left")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)

    for row in rows:
        x = row["evaluation_index"]
        rate = row["win_rate"]
        low = row["win_rate_ci_low"]
        high = row["win_rate_ci_high"]
        if rate is None:
            continue
        yerr = None
        if low is not None and high is not None:
            yerr = [[rate - low], [high - rate]]
        axes[1].errorbar(
            [x],
            [rate],
            yerr=yerr,
            fmt="o" if row["accepted"] else "X",
            color=PALETTE[0] if row["accepted"] else PALETTE[3],
            markerfacecolor=PALETTE[0] if row["accepted"] else "white",
            capsize=2.5,
            linewidth=1.0,
            markersize=5.5,
        )
    axes[1].axhline(0.9, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].text(
        max(xs) + 0.05 if xs else 0,
        0.9,
        "mastery threshold",
        color="#666666",
        va="bottom",
        fontsize=8,
    )
    axes[1].set_ylim(0, 1.04)
    axes[1].set_ylabel("Win rate (95% Wilson CI)")
    axes[1].set_title("(b) Outcome performance at the active difficulty", loc="left")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    labels = [
        f"{row['strategy']}\n{row['difficulty']}"
        for row in rows
    ]
    axes[1].set_xticks(xs, labels, rotation=30, ha="right")
    axes[1].set_xlabel("Evaluation order")
    figure.tight_layout()
    paths = _save_figure(figure, output_base, formats, dpi)
    plt.close(figure)
    return paths


def plot_sample_budget(
    rows: list[dict[str, Any]],
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(7.2, 3.4))
    xs = [row["evaluation_index"] for row in rows]
    cumulative = [row["cumulative_evaluation_games"] for row in rows]
    axis.step(xs, cumulative, where="mid", color=PALETTE[0], linewidth=1.8)
    axis.scatter(xs, cumulative, color=PALETTE[0], s=28, zorder=3)
    for row in rows:
        axis.annotate(
            f"+{row['games']}",
            (row["evaluation_index"], row["cumulative_evaluation_games"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7.5,
            color="#555555",
        )
    axis.set_ylabel("Cumulative evaluated matches")
    axis.set_xlabel("Evaluation order")
    axis.set_xticks(
        xs,
        [row["strategy"] for row in rows],
        rotation=30,
        ha="right",
    )
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    figure.tight_layout()
    paths = _save_figure(figure, output_base, formats, dpi)
    plt.close(figure)
    return paths


def plot_candidate_deltas(
    rows: list[dict[str, Any]],
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    usable = [row for row in rows if row.get("score_delta") is not None]
    if not usable:
        return []
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(7.2, 3.5))
    xs = list(range(len(usable)))
    deltas = [row["score_delta"] for row in usable]
    colors = [PALETTE[0] if delta > 0 else PALETTE[3] for delta in deltas]
    axis.bar(xs, deltas, color=colors, width=0.62, alpha=0.9)
    axis.axhline(0, color="#555555", linewidth=0.8)
    for x, row in zip(xs, usable):
        axis.text(
            x,
            row["score_delta"] + (0.012 if row["score_delta"] >= 0 else -0.012),
            "accepted" if row["accepted"] else "rejected",
            ha="center",
            va="bottom" if row["score_delta"] >= 0 else "top",
            fontsize=7.5,
        )
    axis.set_ylabel("Candidate score − champion score")
    axis.set_xticks(
        xs,
        [f"{row['candidate']}\n{row['difficulty']}" for row in usable],
        rotation=25,
        ha="right",
    )
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    figure.tight_layout()
    paths = _save_figure(figure, output_base, formats, dpi)
    plt.close(figure)
    return paths


def plot_consistency_components(
    rows: list[dict[str, Any]],
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    if not rows:
        return []
    plt = _matplotlib()
    groups = sorted(
        {(row["strategy"], row["difficulty"]) for row in rows},
        key=lambda item: (item[0], difficulty_sort_key(item[1])),
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    metrics = list(CONSISTENCY_METRICS)
    xs = list(range(len(metrics)))
    lookup = {
        (row["strategy"], row["difficulty"], row["metric"]): row
        for row in rows
    }
    for index, (strategy, difficulty) in enumerate(groups):
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        present_xs: list[int] = []
        for x, metric in zip(xs, metrics):
            item = lookup.get((strategy, difficulty, metric))
            if not item or item["mean"] is None:
                continue
            present_xs.append(x)
            means.append(item["mean"])
            lows.append(item["mean"] - item["ci_low"])
            highs.append(item["ci_high"] - item["mean"])
        axis.errorbar(
            present_xs,
            means,
            yerr=[lows, highs],
            marker="o",
            markersize=4.5,
            linewidth=1.4,
            capsize=2.5,
            color=PALETTE[index % len(PALETTE)],
            label=f"{strategy} · {difficulty}",
        )
    axis.set_xticks(xs, [METRIC_LABELS[metric] for metric in metrics])
    axis.tick_params(axis="x", rotation=20)
    axis.set_ylim(0, 1.04)
    axis.set_ylabel("Strategy consistency (mean and 95% CI)")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis.legend(ncol=2, fontsize=7.5, loc="lower left")
    figure.tight_layout()
    paths = _save_figure(figure, output_base, formats, dpi)
    plt.close(figure)
    return paths


def plot_winrate_consistency(
    rows: list[dict[str, Any]],
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    usable = [
        row
        for row in rows
        if row.get("win_rate") is not None
        and row.get("overall_strategy_compliance") is not None
    ]
    if not usable:
        return []
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    for index, row in enumerate(usable):
        size = 30 + 7 * math.sqrt(max(row["games"], 1))
        axis.scatter(
            row["overall_strategy_compliance"],
            row["win_rate"],
            s=size,
            color=PALETTE[index % len(PALETTE)],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axis.annotate(
            f"{row['strategy']} · {row['difficulty']} (n={row['games']})",
            (row["overall_strategy_compliance"], row["win_rate"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7.2,
        )
    axis.set_xlim(0, 1.04)
    axis.set_ylim(0, 1.04)
    axis.set_xlabel("Overall strategy consistency")
    axis.set_ylabel("Win rate")
    axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.7)
    figure.tight_layout()
    paths = _save_figure(figure, output_base, formats, dpi)
    plt.close(figure)
    return paths


def build_summary(
    history: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    consistency_relation: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted_by_difficulty: dict[str, dict[str, Any]] = {}
    for row in history:
        if row["accepted"]:
            accepted_by_difficulty[row["difficulty"]] = {
                "strategy": row["strategy"],
                "generation": row["generation"],
                "win_rate": row["win_rate"],
                "games": row["games"],
                "curriculum_progress_score": row["curriculum_progress_score"],
            }
    candidate_decisions = [row for row in decisions if row.get("candidate")]
    return {
        "history_rows": len(history),
        "decision_rows": len(decisions),
        "total_evaluation_games": sum(row["games"] for row in history),
        "accepted_candidates": sum(row["accepted"] for row in candidate_decisions),
        "rejected_candidates": sum(not row["accepted"] for row in candidate_decisions),
        "latest_accepted_by_difficulty": accepted_by_difficulty,
        "consistency_groups": consistency_relation,
        "metric_notes": {
            "win_rate_interval": "95% Wilson interval",
            "consistency_interval": (
                "95% normal approximation over per-match scores, clipped to [0, 1]"
            ),
            "outcome_and_consistency": (
                "reported separately; strategy consistency is not included in "
                "the evolution selection score"
            ),
            "total_evaluation_games": (
                "sum of history.csv games; assumes each history row references "
                "a distinct evaluated strategy-difficulty sample"
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SC2-LSEE evolution and strategy-consistency reports."
    )
    parser.add_argument("--run-dir", required=True, help="Evolution run directory.")
    parser.add_argument(
        "--consistency-csv",
        help="Optional per_game.csv from tools/strategy_execution_metrics.py.",
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory (default: <run-dir>/report).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Figure formats.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write calculated CSV/JSON outputs without importing matplotlib.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        print(f"[ERROR] history.csv not found: {history_path}")
        return 1
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else run_dir / "report"
    )
    data_dir = out_dir / "data"
    figure_dir = out_dir / "figures"

    history = load_evolution_history(history_path)
    decisions = load_decisions(run_dir)
    consistency_metrics: list[dict[str, Any]] = []
    consistency_relation: list[dict[str, Any]] = []
    consistency_outcomes: list[dict[str, Any]] = []
    if args.consistency_csv:
        consistency_path = Path(args.consistency_csv).expanduser().resolve()
        if not consistency_path.is_file():
            print(f"[ERROR] consistency CSV not found: {consistency_path}")
            return 1
        consistency_metrics, consistency_relation, consistency_outcomes = (
            summarize_consistency(_read_csv(consistency_path))
        )

    write_csv(data_dir / "evolution_history.csv", history)
    write_csv(data_dir / "candidate_decisions.csv", decisions)
    write_csv(data_dir / "strategy_consistency.csv", consistency_metrics)
    write_csv(data_dir / "winrate_consistency.csv", consistency_relation)
    write_csv(data_dir / "consistency_by_outcome.csv", consistency_outcomes)

    summary = build_summary(history, decisions, consistency_relation)
    figures: list[str] = []
    if not args.no_plots:
        try:
            figures.extend(
                plot_evolution_progress(
                    history,
                    figure_dir / "evolution_progress",
                    args.formats,
                    args.dpi,
                )
            )
            figures.extend(
                plot_sample_budget(
                    history,
                    figure_dir / "sample_budget",
                    args.formats,
                    args.dpi,
                )
            )
            figures.extend(
                plot_candidate_deltas(
                    decisions,
                    figure_dir / "candidate_score_delta",
                    args.formats,
                    args.dpi,
                )
            )
            figures.extend(
                plot_consistency_components(
                    consistency_metrics,
                    figure_dir / "strategy_consistency",
                    args.formats,
                    args.dpi,
                )
            )
            figures.extend(
                plot_winrate_consistency(
                    consistency_relation,
                    figure_dir / "winrate_vs_consistency",
                    args.formats,
                    args.dpi,
                )
            )
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            return 1

    summary["figures"] = figures
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Report written to: {out_dir}")
    print(f"Evolution rows: {len(history)}; decisions: {len(decisions)}")
    if args.consistency_csv:
        print(f"Consistency groups: {len(consistency_relation)}")
    elif not args.no_plots:
        print("Consistency figures skipped: no --consistency-csv was supplied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
