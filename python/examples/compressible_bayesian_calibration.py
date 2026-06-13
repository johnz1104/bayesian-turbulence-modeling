"""
PHASE 6 — Compressible Bayesian calibration demo.

Closes the loop on the compressible track:

  1. Build a Ma=0.1 channel with the bound CompressibleForwardModel.
  2. Run a synthetic-truth solve at the SST defaults (a1, β*).
  3. Add 5% Gaussian noise to the resulting Cf observations.
  4. Run a Bayesian calibration with the standard ``BayesianInference``
     pipeline (LHS ensemble -> GP surrogate -> MCMC).
  5. Print posterior summary and verify recovery of truth.

The script prints the same diagnostics as the incompressible BFS example
(``koh_example.py``) so users can see at a glance that the calibration
machinery works for both solvers.

Outputs (under ``--save-dir``):
    posterior_summary.json
    posterior_corner.png      (if matplotlib is available)
    convergence_chain.png     (acceptance / chain trace)

Usage:
    python3 compressible_bayesian_calibration.py
    python3 compressible_bayesian_calibration.py --quick
    python3 compressible_bayesian_calibration.py -o results/comp_calibration
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

from bayesian_inference import BayesianInference


# ---------------- Build case ------------------------------------------

def build_case(nx: int, ny: int, Ma: float = 0.1, Lx: float = 10.0, H: float = 1.0,
               turb_intensity: float = 0.05):
    eos = rs.IdealGasEOS()
    T_in   = 300.0
    p_ref  = 101325.0
    rho_in = eos.density(p_ref, T_in)
    mu_in  = eos.viscosity(T_in)
    a_in   = eos.sound_speed(T_in)
    Uin    = Ma * a_in
    Re     = rho_in * Uin * H / mu_in
    nu_in  = mu_in / rho_in

    mesh = rs.Mesh.make_channel_2d(nx, ny, Lx, H, Re=Re, yPlusTarget=1.0)
    mesh.compute_wall_distance()

    kIn  = 1.5 * (Uin * turb_intensity) ** 2
    omIn = kIn / (nu_in * 100.0)
    bcs  = rs.CompressibleBoundaryConditions.channel_defaults(
        mesh, Uin, T_in, p_ref, kIn, omIn)

    settings = rs.SolverSettings()
    settings.max_iterations      = 3000
    settings.convergence_tol     = 1e-3
    settings.alpha_u             = 0.5
    settings.alpha_p             = 0.2
    settings.alpha_t             = 0.7
    settings.alpha_k             = 0.4
    settings.alpha_omega         = 0.4
    settings.inner_iterations    = 200
    settings.turb_start_iter     = 50
    settings.turb_update_interval = 2
    settings.divergence_limit    = 1e10
    settings.verbose             = False

    return {
        "mesh":     mesh, "eos": eos, "bcs": bcs, "settings": settings,
        "Uin": Uin, "T_in": T_in, "p_ref": p_ref,
        "kIn": kIn, "omIn": omIn,
        "Re": Re, "Ma": Ma,
    }


def build_observations(case: dict, x_stations=(2.5, 5.0, 7.5)):
    """Build a synthetic-truth observation operator on three Cf stations."""
    obs = rs.ObservationOperator()
    for x_loc in x_stations:
        obs.add_skin_friction(
            wall_patch="bottom", location=rs.Vec3(x_loc, 0, 0),
            cf_obs=0.005, sigma=0.001, ref_vel=case["Uin"])
    return obs


def build_forward_model(case: dict, obs: "rs.ObservationOperator",
                         param_set):
    return rs.CompressibleForwardModel(
        mesh=case["mesh"], param_set=param_set, obs_op=obs,
        bcs=case["bcs"], eos=case["eos"], settings=case["settings"],
        u_init=rs.Vec3(case["Uin"], 0, 0),
        p_init=case["p_ref"], T_init=case["T_in"],
        k_init=case["kIn"], omega_init=case["omIn"])


# ---------------- Synthetic truth + noisy observations ----------------

def generate_noisy_truth(fm_truth, param_set, x_stations, Uin,
                          noise_frac: float = 0.05, rng_seed: int = 42):
    """Run truth solve and corrupt with Gaussian noise."""
    np.random.seed(rng_seed)
    coeffs    = rs.SSTCoefficients()
    theta_def = list(param_set.pack(coeffs))
    result    = fm_truth.evaluate(theta_def)
    if not result.predictions:
        raise RuntimeError("truth solve produced no predictions; aborting")
    obs_clean = np.asarray(result.predictions, float)
    sigmas    = np.maximum(noise_frac * np.abs(obs_clean), 1e-6)
    obs_noisy = obs_clean + sigmas * np.random.randn(obs_clean.size)
    return theta_def, obs_clean, obs_noisy, sigmas


def build_observed_forward(case: dict, x_stations,
                            obs_noisy: np.ndarray, sigmas: np.ndarray,
                            param_set):
    """Build a forward model whose observation operator carries the noisy
    Cf targets so the Gaussian likelihood is well-defined."""
    obs_op = rs.ObservationOperator()
    for k, x_loc in enumerate(x_stations):
        obs_op.add_skin_friction(
            wall_patch="bottom", location=rs.Vec3(x_loc, 0, 0),
            cf_obs=float(obs_noisy[k]), sigma=float(sigmas[k]),
            ref_vel=case["Uin"])
    return build_forward_model(case, obs_op, param_set)


# ---------------- Posterior plots --------------------------------------

def save_corner(samples, names, save_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(2.4 * n, 2.4 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i][j] if n > 1 else axes
            if i == j:
                ax.hist(samples[:, i], bins=30, alpha=0.7)
                ax.set_xlabel(names[i])
            elif i > j:
                ax.scatter(samples[:, j], samples[:, i], s=2, alpha=0.4)
                if i == n - 1:  ax.set_xlabel(names[j])
                if j == 0:      ax.set_ylabel(names[i])
            else:
                ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_dir / "posterior_corner.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_chain_plot(sampler, names, save_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    chain = sampler.get_chain()  # shape (n_steps, n_walkers, ndim)
    n     = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.4 * n), sharex=True,
                              squeeze=False)
    for i in range(n):
        axes[i][0].plot(chain[:, :, i], color="k", alpha=0.3, lw=0.5)
        axes[i][0].set_ylabel(names[i])
    axes[-1][0].set_xlabel("MCMC step")
    fig.tight_layout()
    fig.savefig(save_dir / "convergence_chain.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------- Main --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Tiny ensemble (15) and MCMC (200 steps); fast smoke")
    ap.add_argument("--ma", type=float, default=0.1)
    ap.add_argument("--nx", type=int,   default=40)
    ap.add_argument("--ny", type=int,   default=30)
    ap.add_argument("--n-ensemble", type=int, default=None)
    ap.add_argument("--n-steps",    type=int, default=None)
    ap.add_argument("--parallel", action="store_true",
                    help="Use parallel MCMC pool")
    ap.add_argument("-o", "--save-dir", type=Path, default=None)
    args = ap.parse_args()

    save_dir = args.save_dir or (_PYTHON_DIR.parent / "results"
                                  / "compressible_bayesian_calibration")
    save_dir.mkdir(parents=True, exist_ok=True)

    n_ensemble = args.n_ensemble or (15 if args.quick else 30)
    n_steps    = args.n_steps    or (200 if args.quick else 800)

    print("Compressible Bayesian calibration demo (PHASE 6)")
    print(f"  Ma:          {args.ma}")
    print(f"  Mesh:        {args.nx}x{args.ny}")
    print(f"  Ensemble:    {n_ensemble}")
    print(f"  MCMC steps:  {n_steps}")
    print(f"  Parallel:    {args.parallel}")
    print(f"  save_dir:    {save_dir}")

    overall_t0 = time.time()

    # 1) Build truth case + forward model
    case      = build_case(args.nx, args.ny, Ma=args.ma)
    param_set = rs.InferenceParameterSet.a1_betaStar()
    obs_truth = build_observations(case)
    fm_truth  = build_forward_model(case, obs_truth, param_set)

    # 2) Synthetic truth + noisy observations
    print("\n[1/4] Generating synthetic truth observations...")
    x_stations = [2.5, 5.0, 7.5]
    theta_true, obs_clean, obs_noisy, sigmas = generate_noisy_truth(
        fm_truth, param_set, x_stations, case["Uin"], noise_frac=0.05)
    print(f"  truth θ:           {dict(zip(param_set.active_names(), theta_true))}")
    print(f"  Cf truth:          {obs_clean.tolist()}")
    print(f"  Cf noisy:          {obs_noisy.tolist()}")

    # 3) Observed forward model (noisy targets)
    fm_observed = build_observed_forward(case, x_stations, obs_noisy, sigmas,
                                          param_set)

    # 4) Bayesian calibration
    print("\n[2/4] Running ensemble of forward-model evaluations...")
    bi = BayesianInference(fm_observed, param_set)
    bi.run_ensemble(n_samples=n_ensemble, verbose=True)

    print("\n[3/4] Training surrogate...")
    bi.train_surrogate(verbose=True)

    print("\n[4/4] Running MCMC...")
    n_walkers = max(16, 2 * param_set.n_active())
    bi.run_mcmc(n_walkers=n_walkers, n_steps=n_steps,
                burn_in=max(50, n_steps // 4), thin=1,
                parallel=args.parallel, rng_seed=0, verbose=True)

    elapsed = time.time() - overall_t0

    # Posterior summary
    summary = bi.posterior_summary()
    print("\n=== Posterior summary ===")
    print(f"  Total wall time: {elapsed:.1f} s")
    bi.print_summary()

    # Truth comparison
    truth_df = []
    for i, name in enumerate(param_set.active_names()):
        s = summary[name]
        z = abs(s["mean"] - theta_true[i]) / max(s["std"], 1e-9)
        truth_df.append({"name": name, "truth": theta_true[i],
                         "post_mean": s["mean"], "post_std": s["std"],
                         "z_truth": z})
        print(f"  truth {name:>10s} = {theta_true[i]:.4f}  "
              f"posterior μ={s['mean']:.4f} ± {s['std']:.4f}  "
              f"|z|={z:.2f}")

    # Save outputs
    out = {
        "Ma":          args.ma,
        "mesh":        f"{args.nx}x{args.ny}",
        "Re":          case["Re"],
        "n_ensemble":  n_ensemble,
        "n_steps":     n_steps,
        "elapsed_s":   elapsed,
        "x_stations":  x_stations,
        "truth_theta": list(theta_true),
        "obs_clean":   obs_clean.tolist(),
        "obs_noisy":   obs_noisy.tolist(),
        "sigmas":      sigmas.tolist(),
        "posterior":   {k: {kk: vv for kk, vv in summary[k].items()}
                        for k in summary},
        "truth_comparison": truth_df,
    }
    json_path = save_dir / "posterior_summary.json"
    json_path.write_text(json.dumps(out, indent=2))

    # Plots
    save_corner(bi.samples, param_set.active_names(), save_dir)
    save_chain_plot(bi.sampler, param_set.active_names(), save_dir)
    print(f"\n  Summary -> {json_path}")
    print(f"  Plots   -> {save_dir}")


if __name__ == "__main__":
    main()
