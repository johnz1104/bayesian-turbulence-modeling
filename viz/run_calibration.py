"""
Run a real surrogate-accelerated Bayesian SST calibration and dump every
artifact the figure scripts need.

This is the single data-producing step for the visualizers.  It runs the actual
pipeline (LHS ensemble of C++ CFD solves -> GP surrogate -> emcee MCMC) on the
turbulent channel case and writes the ensemble, the GP holdout, the full MCMC
chain, the flat posterior samples, and a JSON summary into
``viz/artifacts/<param_set>/``.  Nothing here is synthetic: the likelihood comes
from real SIMPLE solves against the Dean (1978) skin-friction correlation.

Usage:
    python3 viz/run_calibration.py a1_betaStar      # 2-param centrepiece
    python3 viz/run_calibration.py near_wall4        # 4-param sensitivity/identifiability

Seeds are fixed so the artifacts (and therefore the figures) reproduce.
"""

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_BUILD_DIR = _REPO_ROOT / "build"
_PYTHON_DIR = _REPO_ROOT / "python"
sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PYTHON_DIR))

import numpy as np
import rans_sst_py as rs

from bayesian_inference import BayesianInference
from priors import make_prior_from_param_set, make_sampling_prior
from gp_surrogate import GPSurrogate
from param_utils import _get_param_names


# Map CLI name -> (factory, prior builder).  High-D sets use the narrowed
# sampling prior so the LHS ensemble stays in the solver-convergent region.
PARAM_SETS = {
    "a1_betaStar": (rs.InferenceParameterSet.a1_betaStar, "full_box"),
    "near_wall4":  (rs.InferenceParameterSet.near_wall4,  "sampling"),
    "all11":       (rs.InferenceParameterSet.all11,       "sampling"),
}

# The C++ ForwardModel holds references (not copies) to the mesh, observation
# operator, boundary conditions, and settings.  If the Python owners are
# garbage-collected the references dangle and evaluate() returns the failure
# sentinel.  Keep every owner alive for the process lifetime.
_KEEPALIVE = []


def build_channel_case(param_set, nx=40, ny=30):
    """Mesh, BCs, observation, and ForwardModel for the Re_b = 6800 channel.

    Single observation: the Dean (1978) skin-friction correlation at a
    downstream wall station, with 5% measurement uncertainty.
    """
    h = 1.0
    Lx = 10.0 * h
    Ub = 1.0
    Re_b = 6800.0
    nu = Ub * h / Re_b

    # Dean (1978): Cf = 0.073 * Re_b^{-1/4}
    cf_dean = 0.073 * Re_b ** (-0.25)

    mesh = rs.Mesh.make_channel_2d(nx=nx, ny=ny, Lx=Lx, Ly=2.0 * h,
                                   Re=Re_b, yPlusTarget=1.0)
    mesh.compute_wall_distance()

    Tu = 0.05
    k_in = 1.5 * (Ub * Tu) ** 2
    om_in = k_in / (nu * 100.0)
    bcs = rs.FlowBoundaryConditions.channel_defaults(mesh, Ub, k_in, om_in)

    obs = rs.ObservationOperator()
    obs.add_skin_friction(wall_patch="bottom", location=rs.Vec3(7.0, 0, 0),
                          cf_obs=cf_dean, sigma=0.05 * cf_dean, ref_vel=Ub)

    settings = rs.SolverSettings()
    settings.max_iterations = 5000
    settings.convergence_tol = 1e-4
    settings.verbose = False
    settings.report_interval = 500
    settings.alpha_u = 0.7
    settings.alpha_p = 0.5

    fm = rs.ForwardModel(mesh, param_set, obs, bcs, nu, settings,
                         rs.Vec3(Ub, 0, 0), 0.0, k_in, om_in)
    _KEEPALIVE.extend([mesh, obs, bcs, settings, param_set])
    return mesh, fm, cf_dean


def split_rhat(chain):
    """Split-R-hat (Gelman-Rubin) per dimension from an emcee chain.

    chain: (n_steps, n_walkers, ndim).  Each walker is split in half so the
    statistic also catches within-walker non-stationarity.
    """
    n_steps, n_walkers, ndim = chain.shape
    half = n_steps // 2
    a = chain[:half]
    b = chain[half:2 * half]
    # 2*n_walkers segments of length `half`
    segs = np.concatenate([a, b], axis=1)            # (half, 2*n_walkers, ndim)
    m = segs.shape[1]
    n = half
    means = segs.mean(axis=0)                         # (m, ndim)
    grand = means.mean(axis=0)                        # (ndim,)
    B = n * means.var(axis=0, ddof=1)                 # between-segment var
    W = segs.var(axis=0, ddof=1).mean(axis=0)         # within-segment var
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W)


def main():
    parser = argparse.ArgumentParser(description="Real channel-flow SST calibration -> artifacts")
    parser.add_argument("param_set", choices=list(PARAM_SETS.keys()))
    parser.add_argument("--n-ensemble", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--n-walkers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=30)
    parser.add_argument("--reuse-ensemble", action="store_true",
                        help="Skip the CFD ensemble and reload the cached ensemble.npz "
                             "(re-fits the surrogate and re-runs MCMC only).")
    args = parser.parse_args()

    factory, prior_kind = PARAM_SETS[args.param_set]
    param_set = factory()
    ndim = param_set.n_active()
    names = _get_param_names(param_set)

    # Budgets scale with dimensionality; defaults chosen to converge in minutes.
    n_ensemble = args.n_ensemble if args.n_ensemble is not None else (60 if ndim <= 2 else 30 * ndim)
    n_walkers = args.n_walkers if args.n_walkers is not None else max(24, 4 * ndim)
    n_steps = args.n_steps if args.n_steps is not None else (2500 if ndim <= 2 else 3000)
    burn_in = max(500, n_steps // 5)

    print(f"\n=== Channel SST calibration: {args.param_set} ({ndim}-D) ===")
    print(f"  coefficients: {names}")
    print(f"  ensemble={n_ensemble}  walkers={n_walkers}  steps={n_steps}  burn={burn_in}  seed={args.seed}")

    mesh, fm, cf_dean = build_channel_case(param_set, nx=args.nx, ny=args.ny)
    print(f"  mesh: {mesh.n_cells()} cells   Cf_obs (Dean 1978) = {cf_dean:.6f}\n")

    if prior_kind == "sampling":
        prior = make_sampling_prior(param_set, relative_std=0.15, k_sigma=3.0)
    else:
        prior = make_prior_from_param_set(param_set, relative_std=0.15)

    np.random.seed(args.seed)
    bi = BayesianInference(fm, param_set, prior=prior)

    cache = _SCRIPT_DIR / "artifacts" / args.param_set / "ensemble.npz"
    if args.reuse_ensemble and cache.exists():
        print(f"Stage 1 - reusing cached ensemble {cache}")
        d = np.load(cache)
        bi.ensemble_X, bi.ensemble_y = d["X"], d["y"]
        X, y = bi.ensemble_X, bi.ensemble_y
        n_ensemble = len(X)
        n_valid = len(X)
        n_failed = 0
    else:
        print("Stage 1 - LHS ensemble (real CFD solves)")
        bi.run_ensemble(n_samples=n_ensemble, verbose=True)
        X, y = bi.ensemble_X, bi.ensemble_y
        n_valid = len(X)
        n_failed = n_ensemble - n_valid
    print(f"  valid {n_valid}/{n_ensemble}  (failed/diverged {n_failed})")

    # Explicit deterministic holdout for the surrogate diagnostic figure.
    print("\nStage 2 - GP surrogate (ARD-RBF, noise-floored)")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_valid)
    n_test = max(3, n_valid // 5)
    Xtr, ytr = X[idx[n_test:]], y[idx[n_test:]]
    Xte, yte = X[idx[:n_test]], y[idx[:n_test]]
    sur = GPSurrogate()
    sur.train(Xtr, ytr, optimize_restarts=4, noise_floor=1e-3)
    mu_te, var_te = sur.predict_batch(Xte)
    holdout_rmse = float(np.sqrt(np.mean((mu_te - yte) ** 2)))
    ls = sur.lengthscales()
    print(f"  train {len(Xtr)}  holdout {n_test}  holdout RMSE = {holdout_rmse:.4f} (log-lik units)")
    print(f"  ARD lengthscales: " + "  ".join(f"{n}={l:.3f}" for n, l in zip(names, ls)))

    # Install the surrogate trained on the same data the MCMC will use.
    bi.surrogate = sur

    print("\nStage 3 - emcee MCMC (surrogate likelihood)")
    bi.run_mcmc(n_walkers=n_walkers, n_steps=n_steps, burn_in=burn_in,
                thin=1, verbose=True, rng_seed=args.seed)

    chain = bi.sampler.get_chain()                    # (n_steps, n_walkers, ndim)
    flat = bi.samples                                 # (n_eff, ndim)
    rhat = split_rhat(chain)
    af = bi.sampler.acceptance_fraction
    summary = bi.posterior_summary()
    corr = np.corrcoef(flat.T) if ndim > 1 else np.array([[1.0]])

    print("\n=== posterior summary ===")
    bi.print_summary()
    print(f"\n  split-R-hat: " + "  ".join(f"{n}={r:.3f}" for n, r in zip(names, rhat)))
    print(f"  acceptance fraction: mean={np.mean(af):.3f}")

    out_dir = _SCRIPT_DIR / "artifacts" / args.param_set
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "ensemble.npz", X=X, y=y)
    np.savez(out_dir / "chain.npz", chain=chain, flat=flat, acceptance=af, rhat=rhat)
    np.savez(out_dir / "surrogate_holdout.npz",
             X_test=Xte, y_true=yte, y_pred=mu_te, y_var=var_te,
             rmse=holdout_rmse, lengthscales=np.asarray(ls))

    meta = {
        "param_set": args.param_set,
        "ndim": ndim,
        "names": names,
        "case": "turbulent_channel_Reb6800",
        "observation": {"type": "skin_friction_Cf", "ref": "Dean_1978", "cf_obs": float(cf_dean),
                        "rel_sigma": 0.05, "x_station": 7.0},
        "budget": {"n_ensemble": n_ensemble, "n_valid": int(n_valid), "n_failed": int(n_failed),
                   "n_walkers": n_walkers, "n_steps": n_steps, "burn_in": burn_in,
                   "n_posterior_samples": int(len(flat)), "seed": args.seed},
        "surrogate": {"holdout_rmse_loglik": holdout_rmse, "n_train": int(len(Xtr)),
                      "n_holdout": int(n_test), "noise_floor": 1e-3,
                      "ard_lengthscales": {n: float(l) for n, l in zip(names, ls)}},
        "diagnostics": {"split_rhat": {n: float(r) for n, r in zip(names, rhat)},
                        "acceptance_mean": float(np.mean(af)),
                        "acceptance_min": float(np.min(af)),
                        "acceptance_max": float(np.max(af))},
        "posterior": summary,
        "posterior_correlation": {"names": names, "matrix": corr.tolist()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  artifacts -> {out_dir}/")
    print("  files: ensemble.npz  chain.npz  surrogate_holdout.npz  summary.json")


if __name__ == "__main__":
    main()
