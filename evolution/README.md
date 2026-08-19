# Automatic Evolution

This package is the outer experiment controller. It runs matches, calls
`evol_agent` for one candidate, evaluates that candidate, and keeps only a
strictly higher-scoring strategy as the next Champion.

Default curriculum:

```text
Harder -> VeryHard -> CheatVision -> CheatMoney -> CheatInsane
```

Each Champion baseline and each Candidate evaluation starts with 10 matches. When
the observed rates are within one outcome at the smaller sample, the default CLI
runs an additional 4-match confirmation batch for both strategies if their sample
sizes match. If one side already has more evidence, only the smaller side is topped
up to the same game count. Candidate promotion uses the aggregate outcome score:

```text
Candidate score > Champion score -> accepted
Candidate score < Champion score -> rejected
Candidate score = Champion score -> inconclusive
```

After match evaluation, a separate mechanism audit reuses the parent's prior
cross-match analysis and compares it with the candidate trajectories against the
pre-registered mechanism prediction. It records whether the intervention was
implemented, underpowered, execution-invalid, or unknown, and evaluates
decisive-engagement evidence separately from final score. An execution-invalid
candidate is not promoted even if its sampled score is higher. The posterior
probability is retained only as an informational uncertainty statistic; it does
not participate in selection.

If the initial Candidate batch already reaches the difficulty mastery win rate,
the Candidate is accepted immediately, the post-experiment mechanism audit is
skipped, and the curriculum advances to the next difficulty.

When later evidence adds confirmation matches for an unchanged search parent,
the next generation seeds analysis from the latest compatible checkpoint. Match
summaries are reused by resolved record path, so only newly added records call the
summary model. The previous cross-match analysis is supplied as a revisable seed
alongside experiment history; the full current batch remains the evidence authority.

Every proposed experiment records a `mechanism_family`. A family is blocked after
two non-accepted attempts, after an adequately implemented contradiction, or when
it depends on an unsupported execution capability. A blocked candidate is stopped
before match evaluation rather than consuming another batch.

A difficulty is mastered when the current Champion win rate is greater than or
equal to 0.90. Outcome score still assigns 1 to wins, 0.5 to ties, and 0 to
defeats.

Start a new run:

```powershell
.\scripts\start_evolution.ps1 `
  -STRATEGY tank `
  -COMMANDER_MODEL qwen3.5-27b `
  -EVOLUTION_MODEL qwen3.5-27b `
  -MAX_GENERATIONS 10
```

Resume a run with the same configuration:

```powershell
.\scripts\start_evolution.ps1 `
  -STRATEGY tank `
  -COMMANDER_MODEL qwen3.5-27b `
  -EVOLUTION_MODEL qwen3.5-27b `
  -MAX_GENERATIONS 10 `
  -RUN_DIR "D:\path\to\evolution_runs\tank\YYYYMMDD_HHMMSS"
```

Run state and curve-ready data are saved under
`evolution_runs/<strategy>/<timestamp>/state.json` and `history.csv`. Raw match
records remain under `game_records/`, while immutable candidate strategies
remain under `skills/<race>/`.

`history.csv` uses baseline generation 0 and the first candidate as generation 1.
`games` is the number of matches contributing to that strategy row, and `win_rate`
is `wins / games`; the cumulative run budget remains `state.json.games_used`. Confirmation evidence updates the row
of the strategy that played those matches. The online curve metric is
`curriculum_progress_score = mastered_levels + min(win_rate / mastery_threshold, 1)`;
it measures verified curriculum progress and does not impute performance on an
untested difficulty.
