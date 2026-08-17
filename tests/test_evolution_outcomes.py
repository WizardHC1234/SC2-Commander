from evolution.outcomes import (
    aggregate_outcomes,
    decide_candidate,
    posterior_probability_better,
)


def test_aggregate_outcomes_uses_only_match_results() -> None:
    aggregate = aggregate_outcomes(
        [
            {"wins": 3, "draws": 1, "losses": 2},
            {"wins": 2, "draws": 0, "losses": 2},
        ]
    )

    assert aggregate == {
        "wins": 5,
        "draws": 1,
        "losses": 4,
        "games": 10,
        "score": 0.55,
    }


def test_posterior_probability_tracks_win_rate_difference() -> None:
    champion = {"wins": 5, "draws": 0, "losses": 5}

    assert posterior_probability_better(
        {"wins": 8, "draws": 0, "losses": 2}, champion
    ) > 0.8
    assert posterior_probability_better(
        {"wins": 2, "draws": 0, "losses": 8}, champion
    ) < 0.2


def test_decide_candidate_uses_score_only() -> None:
    assert decide_candidate(0.8, 0.5) == "accepted"
    assert decide_candidate(0.5, 0.5) == "inconclusive"
    assert decide_candidate(0.2, 0.5) == "rejected"
    assert decide_candidate(1.0, 0.9) == "accepted"
    assert decide_candidate(0.65, 0.70) == "rejected"
    assert decide_candidate(0.70, 0.70) == "inconclusive"
