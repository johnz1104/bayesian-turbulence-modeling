"""
Measure the surrogate acceleration directly: how many GP log-likelihood
evaluations per second the trained surrogate sustains, versus the wall-clock
cost of one full CFD solve.  This is the concrete reason the pipeline is
surrogate-accelerated: MCMC needs ~10^4-10^5 likelihood calls, which is
infeasible against the solver but trivial against the GP.

Real data only: the GP is trained on the actual a1_betaStar ensemble, and the
CFD timing comes from real solves on this machine.  Writes
viz/artifacts/surrogate_benchmark.json.

Usage:
    python3 viz/run_surrogate_benchmark.py
"""

import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "build"))
sys.path.insert(0, str(_REPO_ROOT / "python"))

import numpy as np
import rans_sst_py as rs
from gp_surrogate import GPSurrogate
from run_calibration import build_channel_case


def main():
    ens_path = _SCRIPT_DIR / "artifacts" / "a1_betaStar" / "ensemble.npz"
    if not ens_path.exists():
        print(f"  missing {ens_path}; run: python3 viz/run_calibration.py a1_betaStar")
        return

    d = np.load(ens_path)
    X, y = d["X"], d["y"]
    sur = GPSurrogate()
    sur.train(X, y, optimize_restarts=4, noise_floor=1e-3)

    # Surrogate throughput: time many single-point predictions (the call MCMC makes).
    rng = np.random.default_rng(0)
    n_eval = 20000
    pts = X[rng.integers(0, len(X), size=n_eval)]
    t0 = time.time()
    for p in pts:
        sur.log_likelihood(p)
    t_sur = time.time() - t0
    eval_per_s = n_eval / t_sur

    # CFD cost: time real solves at distinct ensemble points (warm-started in
    # sequence, exactly as run_ensemble drives the solver).  Distinct thetas
    # avoid the warm-start cache returning a previous solve for free.
    param_set = rs.InferenceParameterSet.a1_betaStar()
    _, fm, _ = build_channel_case(param_set)
    n_solve = 4
    thetas = X[rng.integers(0, len(X), size=n_solve)]
    solve_times = []
    for th in thetas:
        t0 = time.time()
        r = fm.evaluate(th.tolist())
        dt = time.time() - t0
        if r.log_lik > -1e5:                 # count only real (converged) solves
            solve_times.append(dt)
    if not solve_times:
        print("  WARNING: no converged CFD solves timed; skipping speedup.")
        s_per_solve = float("nan")
    else:
        s_per_solve = float(np.mean(solve_times))

    speedup = eval_per_s * s_per_solve

    result = {
        "surrogate_eval_per_s": eval_per_s,
        "cfd_s_per_solve": s_per_solve,
        "per_eval_speedup": speedup,
        "n_surrogate_evals_timed": n_eval,
        "n_cfd_solves_timed": n_solve,
        "note": "GP trained on the real a1_betaStar ensemble; CFD timing from real solves on this machine.",
    }
    out = _SCRIPT_DIR / "artifacts" / "surrogate_benchmark.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  surrogate: {eval_per_s:,.0f} eval/s")
    print(f"  CFD:       {s_per_solve:.2f} s/solve  ({1/s_per_solve:.3f} solve/s)")
    print(f"  per-eval speedup: {speedup:,.0f}x")
    print(f"  artifact -> {out}")


if __name__ == "__main__":
    main()
