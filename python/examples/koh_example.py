"""
Kennedy-O'Hagan (KOH) model-form uncertainty on the backward-facing step.

Demonstrates how to calibrate SST coefficients while simultaneously inferring
the model-form discrepancy δ(x) ~ GP(0, σ_δ² k(x,x'; l_δ)).  The posterior
over [θ, log σ_δ, log l_δ] separates parameter uncertainty from structural
model error, giving more honest predictive intervals than standard calibration.

Kennedy, M.C. & O'Hagan, A. (2001). Bayesian calibration of computer models.
JRSS-B 63(3), 425-464.

Usage:
    python3 koh_example.py                # 460-cell mesh, demo settings (~10 min)
    python3 koh_example.py --quick        # alias for default demo mode
    python3 koh_example.py --production   # 1800-cell mesh, full settings (~3 hr)
    python3 koh_example.py --compare      # side-by-side: standard vs KOH (~20 min)
    python3 koh_example.py -o PATH        # write summary JSON to PATH/
"""

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_BUILD_DIR  = _SCRIPT_DIR.parent.parent / 'build'
_PYTHON_DIR = _SCRIPT_DIR.parent

sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PYTHON_DIR))

try:
    import rans_sst_py as rs
except ImportError:
    print(f"ERROR: Cannot import rans_sst_py. Build first: cd ../../build && cmake --build .")
    sys.exit(1)

import numpy as np
from bayesian_inference import (
    BayesianInference, BayesianInferenceKOH, BFSSyntheticCalibration,
    KOHLikelihood, DS1985, _bfs_solver_settings,
)


def _recovery_check(bi, theta_true, param_set, label=''):
    """Print per-parameter recovery status vs ground truth."""
    names   = param_set.active_names()
    summary = bi.posterior_summary()
    all_ok  = True
    tag     = f"[{label}] " if label else ""
    print(f"\n  {tag}Recovery check:")
    for i, name in enumerate(names):
        pm   = summary[name]['mean']
        ps   = summary[name]['std']
        truth = theta_true[i]
        err   = abs(pm - truth) / max(ps, 1e-10)
        status = "OK" if err < 3.0 else "WARN"
        print(f"    {name:12s}: truth={truth:.4f}  "
              f"post={pm:.4f}±{ps:.4f}  |err|={err:.1f}σ  [{status}]")
        if err >= 3.0:
            all_ok = False
    return all_ok


def run_standard(cal, n_ensemble, n_steps, verbose):
    """Standard Bayesian calibration (no KOH)."""
    print("\n" + "=" * 60)
    print("Standard calibration (no model-form discrepancy)")
    print("=" * 60)
    bi, theta_true = cal.run(
        n_ensemble=n_ensemble, n_steps=n_steps,
        noise_frac=0.05, verbose=verbose, use_koh=False,
    )
    bi.print_summary()
    ok = _recovery_check(bi, theta_true, cal.param_set, label='standard')
    return bi, theta_true, ok


def run_koh(cal, n_ensemble, n_steps, verbose):
    """KOH calibration with model-form discrepancy."""
    print("\n" + "=" * 60)
    print("KOH calibration (θ + discrepancy hyperparameters)")
    print("=" * 60)
    bi, theta_true = cal.run(
        n_ensemble=n_ensemble, n_steps=n_steps,
        noise_frac=0.05, verbose=verbose, use_koh=True,
    )
    bi.print_summary()
    ok = _recovery_check(bi, theta_true, cal.param_set, label='KOH')

    # Report discrepancy posterior in natural scale
    sd = np.exp(bi.samples[:, cal.param_set.n_active()])
    ld = np.exp(bi.samples[:, cal.param_set.n_active() + 1])
    print(f"\n  Discrepancy amplitude σ_δ: {np.mean(sd):.4f} ± {np.std(sd):.4f}")
    print(f"  Discrepancy lengthscale l_δ: {np.mean(ld):.4f} ± {np.std(ld):.4f} (x/h units)")
    return bi, theta_true, ok


def main():
    parser = argparse.ArgumentParser(
        description='KOH model-form uncertainty on BFS')
    parser.add_argument('--production', action='store_true',
                        help='Full 1800-cell mesh, 80 ensemble, 1000 MCMC steps')
    parser.add_argument('--quick', action='store_true',
                        help='Explicit demo/quick mode (same as default; overrides --production)')
    parser.add_argument('--compare', action='store_true',
                        help='Run both standard and KOH calibrations')
    parser.add_argument('-o', '--save-dir', default=None, metavar='PATH',
                        help='Write summary JSON to PATH/')
    parser.add_argument('--nx_up',   type=int, default=None)
    parser.add_argument('--nx_down', type=int, default=None)
    parser.add_argument('--ny_up',   type=int, default=None)
    parser.add_argument('--ny_down', type=int, default=None)
    args = parser.parse_args()

    production = args.production and not args.quick
    if production:
        nx_up, nx_down = 20, 40
        ny_up, ny_down = 20, 15
        yPlusTarget = 1.0
        n_ensemble, n_steps = 80, 1000
    else:
        nx_up, nx_down = 10, 20
        ny_up, ny_down = 10,  8
        yPlusTarget = 5.0
        n_ensemble, n_steps = 30, 300

    if args.nx_up   is not None: nx_up   = args.nx_up
    if args.nx_down is not None: nx_down = args.nx_down
    if args.ny_up   is not None: ny_up   = args.ny_up
    if args.ny_down is not None: ny_down = args.ny_down

    mode = 'production' if production else 'demo'
    print(f"\nBFS Kennedy-O'Hagan calibration  (DS1985, Re_h={DS1985['Re_h']:.0f})")
    print(f"  Mode:     {mode}")
    print(f"  Mesh:     [{nx_up}+{nx_down}] × [{ny_up}+{ny_down}]  yPlus={yPlusTarget}")
    print(f"  Ensemble: {n_ensemble}  MCMC steps: {n_steps}")

    param_set = rs.InferenceParameterSet.a1_betaStar()
    cal = BFSSyntheticCalibration(
        nx_up=nx_up, nx_down=nx_down,
        ny_up=ny_up, ny_down=ny_down,
        param_set=param_set,
        yPlusTarget=yPlusTarget,
    )

    if args.compare:
        bi_std, theta_true, ok_std = run_standard(cal, n_ensemble, n_steps, True)
        bi_koh, _,          ok_koh = run_koh(cal, n_ensemble, n_steps, True)

        print("\n" + "=" * 60)
        print("Comparison summary:")
        names = param_set.active_names()
        sum_std = bi_std.posterior_summary()
        sum_koh = bi_koh.posterior_summary()
        print(f"\n  {'Parameter':12s}  {'Std σ_post':>10s}  {'KOH σ_post':>10s}  "
              f"{'Inflation':>9s}")
        print("  " + "-" * 48)
        inflation = {}
        for name in names:
            s_std = sum_std[name]['std']
            s_koh = sum_koh[name]['std']
            infl  = s_koh / s_std if s_std > 1e-10 else float('nan')
            inflation[name] = infl
            print(f"  {name:12s}  {s_std:10.4f}  {s_koh:10.4f}  {infl:9.2f}×")
        print("\n  (KOH σ_post > standard σ_post indicates inflated uncertainty")
        print("   from model-form error being modelled explicitly.)")
        print(f"\n  Standard PASSED: {ok_std}")
        print(f"  KOH      PASSED: {ok_koh}")

        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            out = {
                'mode': mode,
                'standard': {k: {'mean': v['mean'], 'std': v['std']}
                              for k, v in sum_std.items()},
                'koh': {k: {'mean': v['mean'], 'std': v['std']}
                        for k, v in sum_koh.items()},
                'inflation': inflation,
                'standard_passed': ok_std,
                'koh_passed': ok_koh,
            }
            with open(save_path / 'koh_summary.json', 'w') as f:
                json.dump(out, f, indent=2)
            print(f"  Summary -> {save_path / 'koh_summary.json'}")
    else:
        bi_koh, theta_true, ok = run_koh(cal, n_ensemble, n_steps, True)
        if ok:
            print("\n  KOH calibration PASSED: θ within 3σ of truth.")
        else:
            print("\n  WARNING: some parameters deviate > 3σ from truth.")

        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            summary = bi_koh.posterior_summary()
            out = {
                'mode': mode,
                'posterior': {k: {'mean': v['mean'], 'std': v['std']}
                              for k, v in summary.items()},
                'koh_passed': ok,
            }
            with open(save_path / 'koh_summary.json', 'w') as f:
                json.dump(out, f, indent=2)
            print(f"  Summary -> {save_path / 'koh_summary.json'}")

    print("\nDone.")


if __name__ == '__main__':
    main()
