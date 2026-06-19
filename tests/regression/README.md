# Regression baselines

This directory holds **frozen reference outputs** used to detect regressions in
the solvers, surrogates, and inference pipeline.  Files here are inputs to
tests under `tests/python/` and `tests/cpp/`; they are **not** generated as
part of every test run.

## Layout

```
tests/regression/
  compressible_channel_ma01.json   # baseline: Cf, Ma_max, residuals at Ma=0.1
  compressible_channel_ma03.json   # baseline at Ma=0.3
  bfs_DS1985_default.json          # incompressible BFS reattachment + Cf at Menter defaults
  koh_synthetic_a1_betaStar.json   # KOH posterior summary on synthetic BFS
```

Each baseline file is a small JSON containing scalar quantities of interest
(QoIs) and per-cell statistics needed by a regression test, NOT full field
dumps.  Field-level visualisation belongs in `outputs/` and is not committed.

## Updating baselines

When intentional behaviour changes occur:

1. Run the relevant validation script (`examples/compressible_validation_ladder.py`
   for compressible cases, `koh_example.py --compare` for KOH, etc.).
2. Inspect the diff: regression baselines should change for documented reasons,
   not silently.
3. Replace the baseline file and update the corresponding test's tolerances
   in the same commit.
4. Reference the issue or PR in the commit message so future readers can
   audit which run produced the new baseline.

## Tolerances

Each consuming test owns its own tolerances.  Defaults:

- residuals/Cf:  rel tol 5%
- reattachment x_r/h: abs tol 0.25
- posterior means: abs tol 1.0 * posterior σ
- posterior σ:    rel tol 30% (MCMC noise)

Tests that need tighter or looser tolerances must say so in a comment.
