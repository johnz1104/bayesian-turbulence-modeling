"""
End-to-end example: Bayesian SST calibration on backward-facing step (BFS).

Validates the BFS mesh + forward model at Menter defaults, then runs
a surrogate-accelerated MCMC calibration using synthetic observations
(reattachment length + Cf profile) generated from those same defaults.
Recovering the defaults from the synthetic data validates the full pipeline.

Reference geometry: Driver-Seegmiller (1985), Re_h = 37,600.
  h_s = 1.0 (step height, reference length)
  H_up = 2.0 (upstream channel height), H = 3.0 (total downstream height)
  Lu = 6h (upstream), Ld = 20h (downstream)

Note on timing: The BFS requires conservative SIMPLE relaxation (alpha_u=0.3) to
stabilise the separated flow, making each solve ~30s on the default 1800-cell mesh.
Use a smaller mesh for fast exploratory runs.

Usage:
    python3 bfs_example.py --validate              # mesh + solver only (~30s)
    python3 bfs_example.py --quick                 # 460-cell demo, 30 ensemble (~15 min)
    python3 bfs_example.py --demo --nx_up 10 \\
        --nx_down 20 --ny_up 10 --ny_down 8      # 400-cell demo (~15 min)
    python3 bfs_example.py                         # 1800-cell full run (~2 hr)
    python3 bfs_example.py --report                # also save figures to results/bfs/
    python3 bfs_example.py -o PATH                 # write summary JSON to PATH/
"""

import argparse
import json
import sys
import os
from pathlib import Path

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
from bayesian_inference import (BayesianInference, BFSSyntheticCalibration,
                               DS1985, _bfs_solver_settings)


def validate_bfs_mesh(nx_up=20, nx_down=40, ny_up=20, ny_down=15, yPlusTarget=1.0):
    """Build BFS mesh, solve at Menter defaults, report reattachment length."""
    h_s   = DS1985['h_s']
    H     = DS1985['H']
    Lu    = DS1985['Lu']
    Ld    = DS1985['Ld']
    Re_h  = DS1985['Re_h']
    Ub    = 1.0
    nu    = Ub * h_s / Re_h

    print(f"\n  Building BFS mesh ({nx_up}+{nx_down}) x ({ny_up}+{ny_down}), yPlus={yPlusTarget}")
    mesh = rs.Mesh.make_backward_facing_step_2d(
        nx_up, nx_down, ny_up, ny_down,
        Lu, Ld, h_s, H,
        Re=Re_h, yPlusTarget=yPlusTarget
    )
    mesh.compute_wall_distance()
    print(f"  Mesh: {mesh.n_cells()} cells, {mesh.n_faces()} faces, "
          f"{mesh.n_patches()} patches")

    Tu   = 0.05
    kIn  = 1.5 * (Ub * Tu) ** 2
    omIn = kIn / (nu * 100.0)

    bcs = rs.FlowBoundaryConditions.bfs_defaults(mesh, Ub, kIn, omIn)

    # Observation: reattachment length (used to measure solver quality)
    obs = rs.ObservationOperator()
    obs.add_reattachment_length(
        'bottom_wall_down',
        xr_obs=DS1985['xr_h'],
        sigma=0.5,
    )

    param_set = rs.InferenceParameterSet.a1_betaStar()

    fm = rs.ForwardModel(
        mesh, param_set, obs, bcs, nu, _bfs_solver_settings(rs),
        rs.Vec3(Ub, 0, 0), 0.0, kIn, omIn
    )

    print("\n  Solving at Menter SST defaults...")
    defaults = np.array(param_set.pack(rs.SSTCoefficients()))
    result   = fm.evaluate(defaults.tolist())
    xr       = result.predictions[0] if result.predictions else float('nan')
    print(f"  Status:              {result.status}")
    print(f"  SIMPLE iterations:   {result.simple_iters}")
    print(f"  Reattachment x_r/h:  {xr:.3f}  (DS1985 reference: {DS1985['xr_h']})")
    print(f"  Log-likelihood:      {result.log_lik:.4f}")

    if abs(xr - DS1985['xr_h']) > 3.0:
        print("  WARNING: x_r differs from reference by > 3h "
              "(coarse mesh or unexpected behaviour)")
    else:
        print("  Reattachment within 3h of DS1985 reference — mesh OK")

    return mesh, fm, param_set, nu


def main():
    parser = argparse.ArgumentParser(
        description='Backward-facing step Bayesian SST calibration')
    parser.add_argument('--demo',     action='store_true',
                        help='Fast demo: 30 ensemble, 300 MCMC steps')
    parser.add_argument('--quick',    action='store_true',
                        help='Alias for --demo')
    parser.add_argument('--validate', action='store_true',
                        help='Mesh + solver validation only, skip calibration')
    parser.add_argument('--report',   action='store_true',
                        help='Save figures to results/bfs/')
    parser.add_argument('-o', '--save-dir', default=None, metavar='PATH',
                        help='Write summary JSON to PATH/')
    parser.add_argument('--nx_up',   type=int, default=None)
    parser.add_argument('--nx_down', type=int, default=None)
    parser.add_argument('--ny_up',   type=int, default=None)
    parser.add_argument('--ny_down', type=int, default=None)
    args = parser.parse_args()

    fast = args.demo or args.quick
    # Mesh defaults: demo uses coarser mesh (~460 cells, yPlusTarget=5)
    # Production uses finer mesh (~1800 cells, yPlusTarget=1)
    nx_up   = args.nx_up   if args.nx_up   is not None else (10 if fast else 20)
    nx_down = args.nx_down if args.nx_down is not None else (20 if fast else 40)
    ny_up   = args.ny_up   if args.ny_up   is not None else (10 if fast else 20)
    ny_down = args.ny_down if args.ny_down is not None else ( 8 if fast else 15)
    yPlusTarget = 5.0 if fast else 1.0

    n_ensemble = 30  if fast else 80
    n_steps    = 300 if fast else 1000

    print(f"\nBFS Bayesian SST calibration  (Driver-Seegmiller 1985, Re_h={DS1985['Re_h']:.0f})")
    print(f"  Mode:     {'demo/quick' if fast else 'production'}")
    print(f"  Ensemble: {n_ensemble}  MCMC steps: {n_steps}")
    print(f"  Mesh:     [{nx_up}+{nx_down}] x [{ny_up}+{ny_down}]")

    # Stage 0: validate mesh + solver
    mesh, fm, param_set, nu = validate_bfs_mesh(nx_up, nx_down, ny_up, ny_down,
                                                 yPlusTarget=yPlusTarget)

    if args.validate:
        print("\nValidation complete (--validate flag set, skipping calibration).")
        return

    # Stage 1-3: synthetic calibration
    print("\n" + "=" * 60)
    print("Synthetic calibration (recovering Menter defaults from noisy data)")
    print("=" * 60)

    cal = BFSSyntheticCalibration(
        nx_up=nx_up, nx_down=nx_down,
        ny_up=ny_up, ny_down=ny_down,
        param_set=param_set,
        yPlusTarget=yPlusTarget,
    )

    bi, theta_true = cal.run(
        n_ensemble=n_ensemble,
        n_steps=n_steps,
        noise_frac=0.05,
        verbose=True,
    )

    print("\n" + "=" * 60)
    bi.print_summary()
    print("=" * 60)

    # Validation: posterior mean should be within 2σ of true values
    names = param_set.active_names()
    summary = bi.posterior_summary()
    all_ok = True
    print("\n  Recovery check (posterior vs. truth):")
    for i, name in enumerate(names):
        post_mean = summary[name]['mean']
        post_std  = summary[name]['std']
        truth     = theta_true[i]
        err_sigma = abs(post_mean - truth) / max(post_std, 1e-10)
        status = "OK" if err_sigma < 3.0 else "WARN"
        print(f"    {name:12s}: truth={truth:.4f}  post={post_mean:.4f}±{post_std:.4f}"
              f"  |err|={err_sigma:.1f}σ  [{status}]")
        if err_sigma >= 3.0:
            all_ok = False

    if all_ok:
        print("\n  Synthetic calibration PASSED: all parameters within 3σ of truth.")
    else:
        print("\n  WARNING: some parameters deviate > 3σ from truth "
              "(may need more ensemble points or MCMC steps).")

    if args.report:
        bi.report('results/bfs/', mesh=cal.mesh, forward_model=cal.forward_model)

    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        summary = bi.posterior_summary()
        out = {
            'mode': 'demo' if fast else 'production',
            'mesh': {'nx_up': nx_up, 'nx_down': nx_down,
                     'ny_up': ny_up, 'ny_down': ny_down},
            'n_ensemble': n_ensemble,
            'n_steps': n_steps,
            'posterior': {k: {'mean': v['mean'], 'std': v['std']}
                          for k, v in summary.items()},
            'reference': {'xr_h': DS1985['xr_h']},
        }
        with open(save_path / 'bfs_summary.json', 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  Summary -> {save_path / 'bfs_summary.json'}")

    print("\nDone.")


if __name__ == '__main__':
    main()
