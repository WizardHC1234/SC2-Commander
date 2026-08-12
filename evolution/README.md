# Automatic Evolution

This package is the outer experiment controller. It runs matches, calls
`evol_agent` for one candidate, evaluates that candidate, and keeps only a
strictly better strategy as the next Champion.

Default curriculum:

```text
Harder -> VeryHard -> CheatVision -> CheatMoney -> CheatInsane
```

Each batch uses 10 matches by default. A level is passed at a score of 0.8.
Wins score 1, ties score 0.5, and defeats score 0. Ties between Champion and
candidate retain the Champion.

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
