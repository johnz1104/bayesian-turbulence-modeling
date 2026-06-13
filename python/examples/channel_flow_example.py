"""
End-to-end example: Bayesian SST calibration on turbulent channel flow.

Runs the full surrogate-accelerated MCMC pipeline against the Dean (1978)
skin-friction correlation for a channel flow at Re_b = 6800.

Usage:
    python3 channel_flow_example.py --demo      # 50 ensemble, 500 MCMC steps (~minutes)
    python3 channel_flow_example.py --quick     # alias for --demo
    python3 channel_flow_example.py             # 200 ensemble, 5000 MCMC steps (~1 hr)
    python3 channel_flow_example.py --report    # save all figures to results/channel/
    python3 channel_flow_example.py -o PATH     # write summary JSON to PATH/

The script resolves the build directory automatically: it looks for
rans_sst_py.*.so in ../../build/ relative to this file's location.
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Locate the build directory relative to this script
_SCRIPT_DIR = Path(__file__).resolve().parent
_BUILD_DIR  = _SCRIPT_DIR.parent.parent / 'build'
_PYTHON_DIR = _SCRIPT_DIR.parent

sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PYTHON_DIR))

try:
    import rans_sst_py as rs
except ImportError:
    print(f"ERROR: Cannot import rans_sst_py.")
    print(f"  Expected .so in: {_BUILD_DIR}")
    print("  Build the project first:")
    print("    cd ../../build && cmake --build .")
    sys.exit(1)

import numpy as np
from bayesian_inference import BayesianInference, make_prior_from_param_set


def build_channel_case(nx=40, ny=30):
    """Build mesh, BCs, observations, and ForwardModel for Re_b = 6800 channel."""
    h    = 1.0
    Lx   = 10.0 * h
    Ub   = 1.0
    Re_b = 6800.0
    nu   = Ub * h / Re_b

    # Dean (1978) skin friction: Cf = 0.073 * Re_b^{-0.25}
    Cf_dean = 0.073 * Re_b ** (-0.25)

    mesh = rs.Mesh.make_channel_2d(nx=nx, ny=ny, Lx=Lx, Ly=2.0 * h,
                                   Re=Re_b, yPlusTarget=1.0)
    mesh.compute_wall_distance()

    Tu   = 0.05
    kIn  = 1.5 * (Ub * Tu) ** 2
    omIn = kIn / (nu * 100.0)

    bcs = rs.FlowBoundaryConditions.channel_defaults(mesh, Ub, kIn, omIn)

    # Observation: Dean correlation Cf at downstream wall (5% measurement uncertainty)
    obs = rs.ObservationOperator()
    obs.add_skin_friction(
        wall_patch='bottom',
        location=rs.Vec3(7.0, 0, 0),
        cf_obs=Cf_dean,
        sigma=0.05 * Cf_dean,
        ref_vel=Ub,
    )

    param_set = rs.InferenceParameterSet.a1_betaStar()

    settings = rs.SolverSettings()
    settings.max_iterations  = 5000
    settings.convergence_tol = 1e-4
    settings.verbose         = False
    settings.report_interval = 500
    settings.alpha_u         = 0.7
    settings.alpha_p         = 0.5

    fm = rs.ForwardModel(
        mesh, param_set, obs, bcs, nu, settings,
        rs.Vec3(Ub, 0, 0), 0.0, kIn, omIn
    )
    return mesh, fm, param_set, nu


def main():
    parser = argparse.ArgumentParser(description='Channel flow Bayesian SST calibration')
    parser.add_argument('--demo',   action='store_true',
                        help='Fast demo: 50 ensemble, 500 MCMC steps')
    parser.add_argument('--quick',  action='store_true',
                        help='Alias for --demo')
    parser.add_argument('--report', action='store_true',
                        help='Save figures to results/channel/')
    parser.add_argument('-o', '--save-dir', default=None, metavar='PATH',
                        help='Write summary JSON to PATH/')
    parser.add_argument('--nx', type=int, default=40, help='Cells in x (default 40)')
    parser.add_argument('--ny', type=int, default=30, help='Cells in y (default 30)')
    args = parser.parse_args()

    fast = args.demo or args.quick
    n_ensemble = 50   if fast else 200
    n_steps    = 500  if fast else 5000
    n_walkers  = 16
    burn_in    = max(50, n_steps // 5)

    print(f"\nChannel flow Bayesian SST calibration")
    print(f"  Mode:     {'demo/quick' if fast else 'production'}")
    print(f"  Ensemble: {n_ensemble}  MCMC steps: {n_steps}  burn-in: {burn_in}")
    print(f"  Mesh:     {args.nx} x {args.ny}\n")

    mesh, fm, param_set, nu = build_channel_case(nx=args.nx, ny=args.ny)
    print(f"  Mesh built: {mesh.n_cells()} cells\n")

    bi = BayesianInference(fm, param_set)

    # Stage 1: ensemble
    print("Stage 1 — LHS ensemble")
    bi.run_ensemble(n_samples=n_ensemble, verbose=True)

    # Stage 2: GP surrogate
    print("\nStage 2 — GP surrogate")
    bi.train_surrogate(verbose=True)

    # Stage 3: MCMC
    print("\nStage 3 — MCMC")
    bi.run_mcmc(n_walkers=n_walkers, n_steps=n_steps,
                burn_in=burn_in, thin=1, verbose=True)

    # Results
    print("\n" + "=" * 60)
    bi.print_summary()
    print("=" * 60)

    if args.report:
        bi.report('results/channel/', mesh=mesh, forward_model=fm)

    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        summary = bi.posterior_summary()
        out = {
            'mode': 'demo' if fast else 'production',
            'mesh': {'nx': args.nx, 'ny': args.ny},
            'n_ensemble': n_ensemble,
            'n_steps': n_steps,
            'posterior': {k: {'mean': v['mean'], 'std': v['std']}
                          for k, v in summary.items()},
        }
        with open(save_path / 'channel_flow_summary.json', 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  Summary -> {save_path / 'channel_flow_summary.json'}")

    print("\nDone.")


if __name__ == '__main__':
    main()
