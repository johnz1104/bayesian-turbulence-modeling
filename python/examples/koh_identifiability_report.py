"""
PHASE 3 — KOH identifiability report.

Runs three calibrations on the BFS synthetic problem (same
``BFSSyntheticCalibration`` used by ``koh_example.py``) and produces a
side-by-side identifiability report:

  1. ``no_discrepancy``    — standard Bayesian inference
  2. ``diagonal``          — KOH with K_δ = I (just σ_δ amplitude)
  3. ``physical_gp``       — KOH with squared-exp kernel on x/h locations

Outputs (under ``--save-dir``):
    koh_identifiability_summary.json
    posterior_widths_shifts.png
    discrepancy_posterior.png

Usage:
    python3 koh_identifiability_report.py
    python3 koh_identifiability_report.py --modes no_discrepancy diagonal
    python3 koh_identifiability_report.py --quick      # tiny ensemble + MCMC
    python3 koh_identifiability_report.py -o results/koh_report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_BUILD_DIR  = _SCRIPT_DIR.parent.parent / "build"
_PYTHON_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PYTHON_DIR))

import numpy as np

try:
    import rans_sst_py as rs
except ImportError:
    print("ERROR: rans_sst_py not built. Run: cmake --build build")
    sys.exit(1)

from bayesian_inference import (
    BFSSyntheticCalibration, DS1985, _bfs_solver_settings,
)
from koh_diagnostics import (
    compare_modes, write_report, make_obs_metadata,
)


def build_calibration(quick: bool):
    """Build the BFS synthetic case + perturbed observation operator.

    Note: the BFS solver is sensitive to mesh size; we keep the proven
    10x20+10x8 mesh in both quick and full modes and let ``--quick`` only
    reduce the MCMC and ensemble counts.
    """
    cal = BFSSyntheticCalibration(
        nx_up=10, nx_down=20, ny_up=10, ny_down=8,
        param_set=rs.InferenceParameterSet.a1_betaStar(),
        yPlusTarget=5.0,
    )
    return cal


def setup_observations(cal):
    """Run the truth solve, build noisy observations, and a fresh obs operator."""
    np.random.seed(42)
    theta_true, obs_clean = cal.generate_synthetic_truth(verbose=True)
    noise = 0.05 * np.maximum(np.abs(obs_clean), 1e-6)
    obs_noisy = obs_clean + noise * np.random.randn(len(obs_clean))

    obs_op = rs.ObservationOperator()
    obs_op.add_reattachment_length(
        "bottom_wall_down", float(obs_noisy[0]), sigma=float(noise[0]))

    cf_stations = DS1985["Cf_stations"][3:6]
    for k, (x_over_h, _, cf_std) in enumerate(cf_stations):
        obs_op.add_skin_friction(
            wall_patch="bottom_wall_down",
            location=rs.Vec3(x_over_h * DS1985["h_s"], 0.0, 0.5),
            cf_obs=float(obs_noisy[1 + k]),
            sigma=float(noise[1 + k]),
            ref_vel=cal.Ub,
        )

    fm_noisy = rs.ForwardModel(
        cal.mesh, cal.param_set, obs_op, cal.bcs, cal.nu,
        _bfs_solver_settings(rs),
        rs.Vec3(cal.Ub, 0, 0), 0.0, cal.kIn, cal.omIn,
    )

    obs_locations = np.array([float(obs_noisy[0])]
                              + [x for x, _, _ in cf_stations])
    metadata = make_obs_metadata([
        {"type": "reattachment", "location": (float(obs_noisy[0]), 0, 0),
         "sigma": float(noise[0]), "group": "reattachment", "units": "h"},
        *[
            {"type": "Cf",
             "location": (x * DS1985["h_s"], 0.0, 0.5),
             "sigma": float(noise[1 + k]), "group": "Cf",
             "units": "dimensionless"}
            for k, (x, _, _) in enumerate(cf_stations)
        ],
    ])
    return fm_noisy, obs_noisy, noise, obs_locations, metadata, theta_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=None,
                    choices=["no_discrepancy", "diagonal", "physical_gp"],
                    help="Subset of modes to run (default: all 3)")
    ap.add_argument("--quick", action="store_true",
                    help="Tiny ensemble (15) and MCMC (200 steps)")
    ap.add_argument("-o", "--save-dir", type=Path, default=None,
                    help="Output directory (default: <repo>/results/koh_report)")
    args = ap.parse_args()

    save_dir = args.save_dir or (_PYTHON_DIR.parent / "results" / "koh_report")

    n_ensemble = 15 if args.quick else 30
    n_steps    = 200 if args.quick else 400

    print("BFS KOH identifiability report")
    print(f"  modes:      {args.modes or ['no_discrepancy', 'diagonal', 'physical_gp']}")
    print(f"  ensemble:   {n_ensemble}")
    print(f"  MCMC steps: {n_steps}")
    print(f"  save_dir:   {save_dir}")

    cal = build_calibration(args.quick)
    fm, obs_values, obs_sigmas, obs_locs, metadata, theta_true = \
        setup_observations(cal)

    comparison = compare_modes(
        forward_model=fm, param_set=cal.param_set,
        obs_locations=obs_locs, obs_values=obs_values, obs_sigmas=obs_sigmas,
        obs_metadata=metadata, modes=args.modes,
        n_ensemble=n_ensemble, n_steps=n_steps,
        rng_seed=0, verbose=True,
    )

    summary_path = write_report(
        comparison, obs_values=obs_values, save_dir=save_dir, truth=theta_true,
    )
    print(f"\n  Summary -> {summary_path}")

    print("\n=== Identifiability flags ===")
    flags = {m: comparison["summaries"][m] for m in comparison["modes"]}
    param_names = cal.param_set.active_names()
    print(f"  truth: {dict(zip(param_names, theta_true.tolist()))}")
    for mode in comparison["modes"]:
        print(f"\n  --- mode={mode} ---")
        s = flags[mode]
        for n in param_names:
            row = s[n]
            print(f"    {n:14s}  μ={row['mean']:+.4f}  σ={row['std']:.4f}  "
                  f"shift={row['shift']:+.2f}σ")


if __name__ == "__main__":
    main()
