"""
PHASE 8 — Multi-case calibration demo.

Joint Bayesian calibration of the two SST coefficients (a1, β*) using two
heterogeneous cases:

    1. ``bfs_synthetic``       (incompressible BFS — DS1985-like obs)
    2. ``channel_compressible`` (Ma=0.1 channel  — Cf at three stations)

The shared θ block is calibrated against both cases simultaneously through
``MultiCaseCalibration``.  This demonstrates the framework that PHASE 8
unlocks; it is also the most expensive script in the repo.

Outputs (under ``--save-dir``):
    multi_case_summary.json
    posterior_corner.png
    convergence_chain.png

Usage:
    python3 multi_case_calibration_demo.py
    python3 multi_case_calibration_demo.py --quick
    python3 multi_case_calibration_demo.py -o results/multi_case_demo
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
    BFSSyntheticCalibration, _bfs_solver_settings, DS1985,
)
from multi_case_calibration import Case, MultiCaseCalibration


# ---------------- Compressible channel case --------------------------

def build_channel_case(quick: bool):
    """Build the Ma=0.1 compressible channel as a multi-case Case."""
    eos    = rs.IdealGasEOS()
    T_in   = 300.0
    p_ref  = 101325.0
    rho_in = eos.density(p_ref, T_in)
    mu_in  = eos.viscosity(T_in)
    Uin    = 0.1 * eos.sound_speed(T_in)
    nu_in  = mu_in / rho_in
    Re     = rho_in * Uin * 1.0 / mu_in

    nx, ny = (16, 12) if quick else (40, 30)
    max_iters = 300 if quick else 3000

    mesh = rs.Mesh.make_channel_2d(nx, ny, 10.0, 1.0, Re=Re, yPlusTarget=1.0)
    mesh.compute_wall_distance()
    kIn  = 1.5 * (Uin * 0.05) ** 2
    omIn = kIn / (nu_in * 100.0)
    bcs  = rs.CompressibleBoundaryConditions.channel_defaults(
        mesh, Uin, T_in, p_ref, kIn, omIn)

    settings = rs.SolverSettings()
    settings.max_iterations = max_iters
    settings.convergence_tol = 1e-2 if quick else 1e-3
    settings.alpha_u = 0.5
    settings.alpha_p = 0.2
    settings.alpha_t = 0.7
    settings.alpha_k = 0.4
    settings.alpha_omega = 0.4
    settings.inner_iterations = 100 if quick else 200
    settings.turb_start_iter = 20 if quick else 50
    settings.turb_update_interval = 2
    settings.divergence_limit = 1e10
    settings.verbose = False

    param_set = rs.InferenceParameterSet.a1_betaStar()

    # Truth Cf at three stations
    obs_truth = rs.ObservationOperator()
    x_stations = [2.5, 5.0, 7.5]
    for x in x_stations:
        obs_truth.add_skin_friction(
            wall_patch="bottom", location=rs.Vec3(x, 0, 0),
            cf_obs=0.005, sigma=0.001, ref_vel=Uin)
    fm_truth = rs.CompressibleForwardModel(
        mesh=mesh, param_set=param_set, obs_op=obs_truth, bcs=bcs, eos=eos,
        settings=settings, u_init=rs.Vec3(Uin, 0, 0),
        p_init=p_ref, T_init=T_in, k_init=kIn, omega_init=omIn)

    np.random.seed(42)
    res = fm_truth.evaluate(list(param_set.pack(rs.SSTCoefficients())))
    obs_clean = np.asarray(res.predictions, float)
    sigmas    = np.maximum(0.05 * np.abs(obs_clean), 1e-6)
    obs_noisy = obs_clean + sigmas * np.random.randn(obs_clean.size)

    obs_used = rs.ObservationOperator()
    for k, x in enumerate(x_stations):
        obs_used.add_skin_friction(
            wall_patch="bottom", location=rs.Vec3(x, 0, 0),
            cf_obs=float(obs_noisy[k]), sigma=float(sigmas[k]),
            ref_vel=Uin)
    fm_used = rs.CompressibleForwardModel(
        mesh=mesh, param_set=param_set, obs_op=obs_used, bcs=bcs, eos=eos,
        settings=settings, u_init=rs.Vec3(Uin, 0, 0),
        p_init=p_ref, T_init=T_in, k_init=kIn, omega_init=omIn)

    case = Case(
        name="channel_Ma01",
        forward_model=fm_used,
        obs_locations=np.array(x_stations, dtype=float),
        obs_values=obs_noisy.astype(float),
        obs_sigmas=sigmas.astype(float),
        weight=1.0,
    )
    return case, param_set


# ---------------- BFS synthetic case --------------------------------

def _quick_bfs_settings(rs):
    """BFS solver settings with reduced iteration cap for quick mode."""
    s = _bfs_solver_settings(rs)
    s.max_iterations   = 4000
    s.convergence_tol  = 1e-3
    s.inner_iterations = 150
    return s


def build_bfs_case(param_set, quick: bool):
    """Build the BFS synthetic case as a multi-case Case."""
    cal = BFSSyntheticCalibration(
        nx_up=10, nx_down=20, ny_up=10, ny_down=8,
        param_set=param_set, yPlusTarget=5.0,
    )

    np.random.seed(42)
    theta_true, obs_clean = cal.generate_synthetic_truth(verbose=False)
    sigmas    = np.maximum(0.05 * np.abs(obs_clean), 1e-6)
    obs_noisy = obs_clean + sigmas * np.random.randn(obs_clean.size)

    obs_used = rs.ObservationOperator()
    obs_used.add_reattachment_length(
        "bottom_wall_down", float(obs_noisy[0]), sigma=float(sigmas[0]))
    cf_stations = DS1985["Cf_stations"][3:6]
    for k, (x_over_h, _, _) in enumerate(cf_stations):
        obs_used.add_skin_friction(
            wall_patch="bottom_wall_down",
            location=rs.Vec3(x_over_h * DS1985["h_s"], 0.0, 0.5),
            cf_obs=float(obs_noisy[1 + k]),
            sigma=float(sigmas[1 + k]),
            ref_vel=cal.Ub)

    solver_settings = _quick_bfs_settings(rs) if quick else _bfs_solver_settings(rs)
    fm_used = rs.ForwardModel(
        cal.mesh, param_set, obs_used, cal.bcs, cal.nu,
        solver_settings,
        rs.Vec3(cal.Ub, 0, 0), 0.0, cal.kIn, cal.omIn,
    )

    locations = np.array([float(obs_noisy[0])]
                          + [x for x, _, _ in cf_stations])
    case = Case(
        name="bfs_DS1985",
        forward_model=fm_used,
        obs_locations=locations,
        obs_values=obs_noisy.astype(float),
        obs_sigmas=sigmas.astype(float),
        weight=1.0,
    )
    # Return cal alongside the case: pybind11 ForwardModel stores references
    # into cal.mesh / cal.bcs, so cal must outlive the ForwardModel.
    return case, cal, theta_true


# ---------------- Plots -------------------------------------------------

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
    fig.savefig(save_dir / "posterior_corner.png", dpi=120)
    plt.close(fig)


# ---------------- Main --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Small budgets: ensemble=10, MCMC=120, coarse meshes")
    ap.add_argument("-o", "--save-dir", type=Path, default=None)
    args = ap.parse_args()

    save_dir = args.save_dir or (_PYTHON_DIR.parent / "results"
                                  / "multi_case_calibration_demo")
    save_dir.mkdir(parents=True, exist_ok=True)

    n_ensemble = 10 if args.quick else 30
    n_steps    = 120 if args.quick else 600

    print("Multi-case calibration demo (PHASE 8)")
    print(f"  quick:        {args.quick}")
    print(f"  n_ensemble:   {n_ensemble}")
    print(f"  n_steps:      {n_steps}")
    print(f"  save_dir:     {save_dir}")

    overall_t0 = time.time()

    print("\n[1/4] Building compressible channel case...")
    channel_case, param_set = build_channel_case(args.quick)

    print("\n[2/4] Building BFS synthetic case (sharing param_set)...")
    bfs_case, _bfs_cal, theta_true = build_bfs_case(param_set, args.quick)

    print("\n[3/4] Running joint multi-case calibration...")
    cal = MultiCaseCalibration(param_set)
    cal.add_case(channel_case)
    cal.add_case(bfs_case)
    cal.run_ensemble(n_samples=n_ensemble, verbose=True)
    cal.train_surrogates(verbose=True)
    cal.run_mcmc(n_walkers=max(8, 4 * param_set.n_active()),
                 n_steps=n_steps, burn_in=max(20, n_steps // 4),
                 thin=1, rng_seed=0, verbose=True)

    print("\n[4/4] Reporting results...")
    cal.print_summary()
    elapsed = time.time() - overall_t0

    # Truth comparison
    summary = cal.posterior_summary()
    truth_df = []
    for i, n in enumerate(param_set.active_names()):
        s = summary[n]
        z = abs(s["mean"] - theta_true[i]) / max(s["std"], 1e-9)
        truth_df.append({"name": n, "truth": float(theta_true[i]),
                         "post_mean": s["mean"], "post_std": s["std"],
                         "z_truth": z})
        print(f"  truth {n:>10s} = {theta_true[i]:.4f}  "
              f"posterior μ={s['mean']:.4f} ± {s['std']:.4f}  "
              f"|z|={z:.2f}")

    out = {
        "elapsed_s": elapsed,
        "n_ensemble": n_ensemble,
        "n_steps": n_steps,
        "cases": cal.case_names(),
        "truth_theta": list(theta_true),
        "posterior": {k: {kk: vv for kk, vv in summary[k].items()}
                       for k in summary},
        "truth_comparison": truth_df,
    }
    json_path = save_dir / "multi_case_summary.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Summary -> {json_path}")

    save_corner(cal.samples, param_set.active_names(), save_dir)
    print(f"  Plots   -> {save_dir}")
    print(f"  Total wall time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
