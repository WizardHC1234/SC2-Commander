from __future__ import annotations

import math
from typing import Any, Iterable


def outcome_score(wins: int, draws: int, losses: int) -> float:
    games = int(wins) + int(draws) + int(losses)
    return (int(wins) + 0.5 * int(draws)) / games if games else 0.0


def aggregate_outcomes(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    entries = [item for item in items if isinstance(item, dict)]
    wins = sum(int(item.get("wins") or 0) for item in entries)
    draws = sum(int(item.get("draws") or 0) for item in entries)
    losses = sum(int(item.get("losses") or 0) for item in entries)
    games = wins + draws + losses
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": games,
        "score": outcome_score(wins, draws, losses),
    }


def decide_candidate(candidate_score: float, champion_score: float) -> str:
    if candidate_score > champion_score:
        return "accepted"
    if candidate_score < champion_score:
        return "rejected"
    return "inconclusive"


def posterior_probability_better(
    candidate: dict[str, Any],
    champion: dict[str, Any],
    *,
    bins: int = 4096,
) -> float:
    """Approximate P(candidate win rate > champion win rate).

    A uniform Beta(1, 1) prior is updated from outcomes only. Draws contribute
    half a success and half a failure. Midpoint quadrature keeps this module
    dependency-free and deterministic.
    """

    def parameters(value: dict[str, Any]) -> tuple[float, float]:
        wins = float(value.get("wins") or 0)
        draws = float(value.get("draws") or 0)
        losses = float(value.get("losses") or 0)
        return 1.0 + wins + 0.5 * draws, 1.0 + losses + 0.5 * draws

    def masses(alpha: float, beta: float) -> list[float]:
        log_norm = math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
        values = []
        for index in range(bins):
            x = (index + 0.5) / bins
            values.append(
                math.exp(
                    log_norm
                    + (alpha - 1.0) * math.log(x)
                    + (beta - 1.0) * math.log1p(-x)
                )
            )
        total = sum(values)
        return [value / total for value in values]

    candidate_mass = masses(*parameters(candidate))
    champion_mass = masses(*parameters(champion))
    probability = 0.0
    champion_cdf = 0.0
    for cand, champ in zip(candidate_mass, champion_mass):
        probability += cand * (champion_cdf + 0.5 * champ)
        champion_cdf += champ
    return min(1.0, max(0.0, probability))


__all__ = [
    "aggregate_outcomes",
    "decide_candidate",
    "outcome_score",
    "posterior_probability_better",
]
