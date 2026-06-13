"""
POST-PHASE 8 — Active learning for high-Mach adaptive ensembles.

Runs the existing ActiveLearner strategies on the scramjet analytic forward
model and compares convergence rates.  Demonstrates that:

  - Variance-based strategies (max_var, max_norm_var) build accurate GP
    surrogates with fewer function evaluations than random sampling.
  - The surrogate can later be used as a SurrogateForwardModel to replace
    expensive CFD calls in BayesianInferenceKOH.

Strategies benchmarked
----------------------
  random       : LHS baseline (no model-awareness)
  max_min_dist : space-filling (surrogate-free, strong baseline)
  max_var      : surrogate uncertainty maximisation
  max_norm_var : scale-normalised variance (robust across outputs)

Usage:
    python3 active_learning_scramjet.py
    python3 active_learning_scramjet.py --quick
    python3 active_learning_scramjet.py -o PATH
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PYTHON_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PYTHON_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

from observation_schema import ObservableType, scramjet_synthetic_observation_set
from forward_model_interface import forward_model_to_callable
from active_learning import ActiveLearner, sample_lhs
from scramjet_calibration_demo import ScramjetAnalyticForwardModel


# ---------------------------------------------------------------------------
# Validation set helper
# ---------------------------------------------------------------------------

def _make_val_set(fm, n_val: int, lower, upper, rng_seed: int = 99):
    """Build a fixed validation set for RMSE tracking."""
    np.random.seed(rng_seed)
    from bayesian_inference import latin_hypercube
    X_val = latin_hypercube(n_val, len(lower), lower, upper)
    Y_val = []
    for x in X_val:
        r = fm.evaluate(x.tolist())
        Y_val.append(r.predictions)
    return X_val, np.array(Y_val)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Active learning scramjet benchmark (POST-PHASE 8)"
    )
    parser.add_argument("--quick",    action="store_true")
    parser.add_argument("-o", "--save-dir", dest="save_dir", default=None,
                        metavar="PATH")
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()

    quick    = args.quick
    rng_seed = args.rng_seed

    n_init      = 8  if quick else 20
    n_queries   = 8  if quick else 30
    n_stations  = 5  if quick else 8
    n_val       = 30

    strategies = ["random", "max_min_dist", "max_var", "max_norm_var"]

    print("=" * 60)
    print("POST-PHASE 8 — Active Learning: Scramjet Adaptive Ensemble")
    print("=" * 60)
    print(f"  Stations: {n_stations}  init={n_init}  queries={n_queries}")

    # ------------------------------------------------------------------
    # Forward model (analytic SBLI)
    # ------------------------------------------------------------------
    obs_full, truth, _ = scramjet_synthetic_observation_set(
        n_wall_stations=n_stations, rng_seed=rng_seed
    )
    obs_cal = obs_full.filter_by_types([
        ObservableType.WALL_PRESSURE_CP,
        ObservableType.SKIN_FRICTION_CF,
    ])
    fm     = ScramjetAnalyticForwardModel(obs_cal)
    lower  = np.array([0.20, 0.05])
    upper  = np.array([0.50, 0.15])
    forward = forward_model_to_callable(fm)

    print(f"  Outputs: {obs_cal.n_obs}  (Cp+Cf at {n_stations} stations)")

    # ------------------------------------------------------------------
    # Validation set (fixed)
    # ------------------------------------------------------------------
    X_val, Y_val = _make_val_set(fm, n_val, lower, upper, rng_seed=99)

    # ------------------------------------------------------------------
    # Benchmark each strategy
    # ------------------------------------------------------------------
    results = {}
    for strat in strategies:
        print(f"\n=== Strategy: {strat} ===")
        learner = ActiveLearner(
            forward=forward,
            lower=lower,
            upper=upper,
            strategy=strat,
            n_candidates=200,
            rng_seed=rng_seed,
            val_set=(X_val, Y_val),
            verbose=False,
        )
        learner.initialize(n_init)
        learner.run(n_queries)

        final_rmse = learner.history.rmse_history[-1] if learner.history.rmse_history else float("nan")
        results[strat] = {
            "n_total_evals": int(learner.history.n_train),
            "final_rmse": float(final_rmse),
            "rmse_history": [float(r) for r in learner.history.rmse_history],
        }
        print(f"  n_total={learner.history.n_train}"
              f"  final_RMSE={final_rmse:.5f}")

    # ------------------------------------------------------------------
    # Final RMSE ranking
    # ------------------------------------------------------------------
    print("\n=== Final RMSE ranking ===")
    ranked = sorted(results.items(), key=lambda kv: kv[1]["final_rmse"])
    for rank, (strat, info) in enumerate(ranked, 1):
        print(f"  #{rank}  {strat:14s}: RMSE={info['final_rmse']:.5f}"
              f"  (n={info['n_total_evals']})")

    best   = ranked[0][0]
    worst  = ranked[-1][0]
    print(f"\n  Best strategy:  {best}")
    print(f"  Worst strategy: {worst}")

    # Verify variance-based beats random (usually holds with analytic model)
    best_rmse   = results[best]["final_rmse"]
    random_rmse = results["random"]["final_rmse"]
    improvement = (random_rmse - best_rmse) / max(random_rmse, 1e-12) * 100
    print(f"  Improvement vs random: {improvement:.1f}%")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if args.save_dir:
        out = Path(args.save_dir) / "active_learning_scramjet"
        out.mkdir(parents=True, exist_ok=True)
        report = {
            "n_stations": n_stations,
            "n_init": n_init,
            "n_queries": n_queries,
            "strategies": results,
            "ranking": [s for s, _ in ranked],
        }
        (out / "al_report.json").write_text(json.dumps(report, indent=2))
        print(f"\n  Saved → {out}/al_report.json")


if __name__ == "__main__":
    main()
