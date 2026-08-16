from evolution.outcomes import aggregate_outcomes, posterior_probability_better


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
