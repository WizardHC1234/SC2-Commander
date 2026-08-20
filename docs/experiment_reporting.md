# Experiment statistics and plotting

The reporting pipeline is intentionally independent from the online evolution
loop. It reads completed artifacts and writes only to `analysis_results/` by
default. It never changes `state.json`, strategy files, or match records.

## Metrics

Outcome performance and strategy execution are reported separately.

- `win_rate`: victories divided by completed games. Figures use a 95% Wilson
  confidence interval.
- `score`: outcome points used by the run history, where a draw contributes
  half a point.
- `curriculum_progress_score`: the online cross-difficulty progress value
  already stored by the evolution runner.
- `economy_completion`, `technology_completion`, and `army_completion`: the
  three strategy-development components from the deterministic consistency
  evaluator.
- `engagement_trigger_consistency` and
  `engagement_continuation_consistency`: whether the agent starts and continues
  combat according to the strategy when those decisions are evaluable.
- `overall_strategy_compliance`: the mean of the available five consistency
  components. A non-evaluable engagement component is omitted rather than
  counted as a perfect score.

The evolution selection score does not include strategy consistency. The
report adds a separate win-rate-versus-consistency figure so the two properties
can be discussed without conflating them.

## One-command report

From the repository root:

```powershell
.\scripts\generate_evolution_report.ps1 `
  -RunDir "evolution_runs\tank\20260818_134320"
```

This creates the evolution-only CSV files and figures under
`analysis_results/evolution/20260818_134320`.

To include strategy-consistency metrics, pass completed record directories:

```powershell
.\scripts\generate_evolution_report.ps1 `
  -RunDir "evolution_runs\tank\20260818_134320" `
  -Records @(
    "game_records\ev_20260818_134320_g000_champ",
    "game_records\ev_20260818_134320_g001_cand"
  )
```

Each evolved strategy whose composition, build targets, or attack gate changed
should have its own specification JSON in `tools/strategy_specs/`. Reusing
`tank.json` for `tank_opt*` is valid only when the intended measurement is
compliance with the original Tank strategy; it is not an exact measurement of
the evolved strategy. These JSON files are offline metric definitions and are
never read by the evolution agent.

Use `-NoPlots` when only tidy CSV and JSON outputs are needed. Plotting requires
the `matplotlib` dependency listed in `requirements.txt`.

## Outputs

The `data/` directory contains:

- `evolution_history.csv`: one row per history entry, including Wilson bounds
  and cumulative evaluated games.
- `candidate_decisions.csv`: candidate/champion score differences, decisions,
  audit verdicts, and comparison sample counts.
- `strategy_consistency.csv`: mean, standard deviation, and 95% interval for
  each consistency component by strategy and difficulty.
- `winrate_consistency.csv`: paired outcome and overall-consistency summaries.
- `consistency_by_outcome.csv`: consistency split by victory, defeat, or draw.
- `resource_usage.csv`: total and per-game environment time, saved decision
  records, Commander decisions, and game-side LLM interaction counts.

The `figures/` directory contains both raster PNG and vector PDF versions:

- `evolution_progress`: curriculum progress and win rate across evaluations.
- `sample_budget`: cumulative number of evaluated matches.
- `candidate_score_delta`: candidate score minus champion score.
- `strategy_consistency`: the five component metrics plus their overall mean.
- `winrate_vs_consistency`: relationship between successful execution and
  game outcome.
- `resource_usage`: environment hours and game-side LLM interaction counts.

All figure source values remain available in the corresponding CSV files so a
paper result can be reproduced without extracting values from an image.
