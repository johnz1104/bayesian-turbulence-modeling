"""
PHASE 1 / V.1 driver — surrogate breakdown vs. calibration dimensionality.

Sweeps d_θ ∈ {2, 4, 8, 11} on the Re_b=6800 channel, building one checkpointed
LHS ensemble of CFD solves per d_θ and recording GP RMSE / R² / σ-coverage vs.
training-set size on a fixed held-out split.  Emits breakdown curves + a summary
JSON naming the d_θ at which the surrogate degrades materially.

Usage:
    python3 surrogate_scaling_study.py --quick   # d_θ∈{2,4}, n_total=30 (~mins)
    python3 surrogate_scaling_study.py           # full sweep, n_total=120 (~1-2 hr)
    python3 surrogate_scaling_study.py -o PATH    # output directory

Re-runs reuse cached ensembles in the output directory (resumable).
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_BUILD_DIR = _SCRIPT_DIR.parent.parent / "build"
_PYTHON_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PYTHON_DIR))

try:
    import rans_sst_py as rs  # noqa: F401  (import guard / path check)
except ImportError:
    print(f"ERROR: cannot import rans_sst_py from {_BUILD_DIR}; build first.")
    sys.exit(1)

from scaling_study import SurrogateScalingStudy, _param_set_factory


def main():
    ap = argparse.ArgumentParser(description="Surrogate scaling/breakdown study")
    ap.add_argument("--quick", action="store_true",
                    help="Fast subset: d_θ∈{2,4}, n_total=30")
    ap.add_argument("-o", "--out-dir", default=None, help="Output directory")
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--ny", type=int, default=30)
    ap.add_argument("--n-total", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.quick:
        d_thetas = (2, 4)
        n_total = args.n_total or 30
        n_test = 8
        out = args.out_dir or "outputs/scaling_study/channel_quick"
    else:
        d_thetas = (2, 4, 8, 11)
        n_total = args.n_total or 120
        n_test = 30
        out = args.out_dir or "outputs/scaling_study/channel"

    study = SurrogateScalingStudy(
        SurrogateScalingStudy.channel_case_builder(nx=args.nx, ny=args.ny),
        _param_set_factory(),
        d_thetas=d_thetas, n_total=n_total, n_test=n_test,
        rng_seed=args.seed, out_dir=out,
        n_gp_repeats=2 if args.quick else 5,
    )
    verdict = study.run()

    print("\n" + "=" * 70)
    print("SURROGATE BREAKDOWN VERDICT")
    print("=" * 70)
    print(f"  {'d_θ':>4} {'R²@'+str(verdict['reference_budget']):>8} "
          f"{'R²max':>7} {'RMSEmax':>8} {'n→R²≥'+str(verdict['r2_trust']):>9} "
          f"{'cov2σ':>6}")
    for row in verdict["rows"]:
        flag = "  <-- degraded" if row["degraded"] else ""
        de = row["data_to_R2_trust"]
        de_s = str(de) if de is not None else ">budget"
        print(f"  {row['d_theta']:>4} {row['r2_at_ref']:>8.3f} "
              f"{row['r2_max']:>7.3f} {row['rmse_max']:>8.3g} {de_s:>9} "
              f"{row['coverage_2sigma']:>6.2f}{flag}")
    print(f"\n  {verdict['interpretation']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
