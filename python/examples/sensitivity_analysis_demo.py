"""
POST-PHASE 7 — Sensitivity analysis before adjoints.

Ranks SST closure coefficients [a1, β*] by their influence on scramjet
wall observables (Cp, Cf) using two complementary methods:

  - OAT (one-at-a-time): sweeps each parameter while holding others fixed
  - Morris screening:    elementary effects over random trajectories

Results guide the adjoint-implementation decision (Post-Phase 10):
parameters with low normalised sensitivity do not justify adjoint gradients.

Usage:
    python3 sensitivity_analysis_demo.py
    python3 sensitivity_analysis_demo.py --quick
    python3 sensitivity_analysis_demo.py -o PATH
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PYTHON_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PYTHON_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

from observation_schema import ObservableType, scramjet_synthetic_observation_set
from sensitivity_analysis import SensitivityAnalyser
from scramjet_calibration_demo import ScramjetAnalyticForwardModel


def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis demo (POST-PHASE 7)"
    )
    parser.add_argument("--quick",    action="store_true")
    parser.add_argument("-o", "--save-dir", dest="save_dir", default=None,
                        metavar="PATH")
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()

    quick    = args.quick
    rng_seed = args.rng_seed
    n_oat        = 7  if quick else 15
    n_traj       = 8  if quick else 30
    n_stations   = 6  if quick else 10

    print("=" * 60)
    print("POST-PHASE 7 — Sensitivity Analysis before Adjoints")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Build the analytic scramjet forward model (Cp + Cf, n_stations)
    # ------------------------------------------------------------------
    obs_full, truth, _ = scramjet_synthetic_observation_set(
        n_wall_stations=n_stations, rng_seed=rng_seed
    )
    obs_cal = obs_full.filter_by_types([
        ObservableType.WALL_PRESSURE_CP,
        ObservableType.SKIN_FRICTION_CF,
    ])
    fm = ScramjetAnalyticForwardModel(obs_cal)
    print(f"\n  Forward model: {obs_cal.n_obs} outputs  "
          f"(Cp+Cf at {n_stations} stations)")
    print(f"  Parameters:   {fm.parameter_names()}")

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------
    lower   = np.array([0.20, 0.05])
    upper   = np.array([0.50, 0.15])
    nominal = np.array([truth["a1"], truth["betaStar"]])

    sa = SensitivityAnalyser(
        forward_model=fm,
        bounds=(lower, upper),
        param_names=["a1", "betaStar"],
        nominal=nominal,
    )

    print(f"\n[1/2] OAT sweep ({n_oat} points per parameter) ...")
    sa.run_oat(n_points=n_oat, verbose=False)
    oat_imp = sa.oat_importance()
    for name, val in oat_imp.items():
        print(f"  {name:12s}: OAT importance = {val:.4f}")

    print(f"\n[2/2] Morris screening ({n_traj} trajectories) ...")
    sa.run_morris(n_trajectories=n_traj, rng_seed=rng_seed, verbose=False)
    morr_imp = sa.morris_importance()
    for r in sa.morris_results:
        print(f"  {r.param_name:12s}: μ*={r.mu_star:.4e}  σ={r.sigma:.4e}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n=== Sensitivity ranking ===")
    sa.print_report()

    rep = sa.report()
    print(f"\n  Ranking: {' > '.join(rep['ranking'])}")

    # Adjoint decision hint
    min_oat = min(oat_imp.values())
    print(f"\n  Adjoint decision hint:")
    for name in rep["ranking"]:
        score = np.mean([
            rep.get("oat_importance", {}).get(name, 0.0),
            rep.get("morris_importance", {}).get(name, 0.0),
        ])
        advice = "adjoint warranted" if score > 0.2 else "low sensitivity — adjoint low priority"
        print(f"    {name:12s}  score={score:.3f}  → {advice}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if args.save_dir:
        out = Path(args.save_dir) / "sensitivity_analysis"
        out.mkdir(parents=True, exist_ok=True)
        (out / "sensitivity_report.json").write_text(
            json.dumps(rep, indent=2)
        )
        print(f"\n  Saved → {out}/sensitivity_report.json")


if __name__ == "__main__":
    main()
