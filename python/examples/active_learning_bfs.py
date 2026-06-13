"""
PHASE 4 — BFS active-learning vs random sampling demo.

Builds the BFS synthetic case from ``BFSSyntheticCalibration`` and compares
two ensemble-construction strategies for the same number of forward-model
evaluations:

    1. random       (uniform LHS-style sampling — the baseline)
    2. max_min_dist (greedy maximin space-filling)
    3. max_var      (variance-driven; refits the surrogate after each query)

For each strategy we report:

  - cumulative wall time
  - per-output RMSE on a held-out validation set
  - 1σ/2σ/3σ coverage of the validation residuals

Outputs (under ``--save-dir``):
    active_learning_summary.json
    rmse_curves.png
    coverage_curves.png

Usage:
    python3 active_learning_bfs.py
    python3 active_learning_bfs.py --quick                  # tiny budget
    python3 active_learning_bfs.py --strategies random max_min_dist
    python3 active_learning_bfs.py -o results/al_bfs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_BUILD_DIR  = _SCRIPT_DIR.parent.parent / "build"
_PYTHON_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PYTHON_DIR))

try:
    import rans_sst_py as rs
except ImportError:
    print("ERROR: rans_sst_py not built. Run: cmake --build build")
    sys.exit(1)

from bayesian_inference import (
    BFSSyntheticCalibration, _bfs_solver_settings,
)
from surrogate_diagnostics import (
    multi_output_diagnostics, plot_predicted_vs_true,
)
from active_learning import (
    ActiveLearner, cpp_forward_adapter, sample_lhs,
)


def build_calibration():
    cal = BFSSyntheticCalibration(
        nx_up=10, nx_down=20, ny_up=10, ny_down=8,
        param_set=rs.InferenceParameterSet.a1_betaStar(),
        yPlusTarget=5.0,
    )
    return cal


def build_validation_set(forward, lower, upper, n_val: int, rng_seed: int = 999):
    """Generate a validation set on a fixed seed so all strategies share it."""
    X = sample_lhs(n_val, lower, upper, rng_seed=rng_seed)
    valid_X, valid_Y = [], []
    for x in X:
        preds, ok = forward(x)
        if ok:
            valid_X.append(x); valid_Y.append(preds)
    return np.asarray(valid_X), np.asarray(valid_Y)


def run_strategy(strategy: str, forward, lower, upper, n_init: int, n_queries: int,
                 val_set, rng_seed: int, verbose: bool):
    learner = ActiveLearner(forward, lower, upper,
                             strategy=strategy, n_candidates=256,
                             rng_seed=rng_seed, val_set=val_set,
                             verbose=verbose)
    t0 = time.time()
    learner.initialize(n_init=n_init)
    init_t = time.time() - t0
    learner.run(n_queries=n_queries)
    return learner, init_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=None,
                    choices=["random", "max_var", "max_norm_var", "max_min_dist"],
                    help="Subset of strategies to compare (default: all 4)")
    ap.add_argument("--quick", action="store_true",
                    help="Tiny budget (n_init=8, n_queries=10, n_val=15) for fast turn-around")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("-o", "--save-dir", type=Path, default=None,
                    help="Output directory (default: <repo>/results/active_learning_bfs)")
    args = ap.parse_args()

    save_dir = args.save_dir or (_PYTHON_DIR.parent / "results" / "active_learning_bfs")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        n_init, n_queries, n_val = 8, 10, 15
    else:
        n_init, n_queries, n_val = 12, 24, 30

    strategies = args.strategies or ["random", "max_min_dist", "max_var"]

    print("BFS active-learning vs random sampling demo")
    print(f"  strategies:    {strategies}")
    print(f"  n_init:        {n_init}")
    print(f"  n_queries:     {n_queries}")
    print(f"  n_val:         {n_val}")
    print(f"  save_dir:      {save_dir}")

    cal = build_calibration()
    print("\nGenerating synthetic truth observations...")
    np.random.seed(42)
    theta_true, obs_clean = cal.generate_synthetic_truth(verbose=True)

    # Use cal.forward_model directly — it carries a default observation
    # operator that matches the one used in koh_example.
    forward = cpp_forward_adapter(cal.forward_model, koh_n=len(obs_clean))

    lower = np.array(cal.param_set.lower_bounds())
    upper = np.array(cal.param_set.upper_bounds())

    print("\nBuilding validation set (independent LHS)...")
    X_val, Y_val = build_validation_set(forward, lower, upper, n_val=n_val,
                                         rng_seed=999)
    print(f"  Validation: {len(X_val)} valid out of {n_val}")

    summary = {
        "strategies":  strategies,
        "n_init":      n_init,
        "n_queries":   n_queries,
        "n_val":       int(len(X_val)),
        "theta_true":  theta_true.tolist(),
        "obs_clean":   obs_clean.tolist(),
        "results":     {},
    }

    for strat in strategies:
        print(f"\n=== Strategy: {strat} ===")
        learner, init_t = run_strategy(
            strategy=strat, forward=forward, lower=lower, upper=upper,
            n_init=n_init, n_queries=n_queries,
            val_set=(X_val, Y_val), rng_seed=args.rng_seed, verbose=True,
        )
        diag = multi_output_diagnostics(learner.surrog, X_val, Y_val)
        plot_path = save_dir / f"pred_vs_true_{strat}.png"
        plot_predicted_vs_true(learner.surrog, X_val, Y_val, plot_path)
        summary["results"][strat] = {
            "rmse_history":     learner.history.rmse_history,
            "final_mean_rmse":  diag["aggregate"]["mean_rmse"],
            "per_output_rmse":  [p["rmse"] for p in diag["per_output"]],
            "per_output_cov2":  [p["coverage_2sigma"] for p in diag["per_output"]],
            "init_time_s":      float(init_t),
            "query_time_total": float(sum(learner.history.elapsed_s)),
            "n_train":          int(learner.history.n_train),
            "queries":          [list(q) for q in learner.history.queries],
        }

    json_path = save_dir / "active_learning_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # Plot RMSE curves and coverage bars.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for strat in strategies:
            history = summary["results"][strat]["rmse_history"]
            ax.plot(np.arange(1, len(history) + 1), history,
                    "-o", ms=4, label=strat)
        ax.set_xlabel("Acquisition step"); ax.set_ylabel("Validation mean-RMSE")
        ax.legend(); ax.set_title("Surrogate validation error vs strategy")
        fig.tight_layout()
        fig.savefig(save_dir / "rmse_curves.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        cov_keys = ("per_output_cov2",)
        for strat in strategies:
            cov = summary["results"][strat]["per_output_cov2"]
            ax.plot(np.arange(len(cov)), cov, "-o", ms=4, label=strat)
        ax.axhline(0.95, color="k", lw=0.5, ls="--")
        ax.set_ylabel("2σ coverage"); ax.set_xlabel("output channel")
        ax.legend(); ax.set_title("Surrogate predictive-variance calibration (2σ)")
        fig.tight_layout()
        fig.savefig(save_dir / "coverage_curves.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("(matplotlib missing; plots skipped)")

    print(f"\n  Summary -> {json_path}")
    print("\n=== Final RMSE ranking ===")
    ranked = sorted(strategies,
                    key=lambda s: summary["results"][s]["final_mean_rmse"])
    for r, s in enumerate(ranked, 1):
        info = summary["results"][s]
        print(f"  {r}. {s:14s}  mean_rmse={info['final_mean_rmse']:.4g}  "
              f"n_train={info['n_train']}  "
              f"query_time={info['query_time_total']:.1f}s")


if __name__ == "__main__":
    main()
