"""
POST-PHASE 10/11 — Adjoint-readiness diagnostic and final decision.

Evaluates whether the current scramjet calibration problem meets the three
thresholds for adjoint implementation (DECISION_RECORD.md §4):

  1. max normalised sensitivity > 0.5
  2. n_ensemble > 500
  3. d_theta > 5

Reports the diagnostic result and writes a final summary.  This is the
terminal post-phase: if adjoints are not warranted, the project closes with
the GP-surrogate pipeline as the recommended approach.

Usage:
    python3 adjoint_readiness_demo.py
    python3 adjoint_readiness_demo.py --quick
    python3 adjoint_readiness_demo.py -o PATH
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
from adjoint_readiness import run_diagnostic
from scramjet_calibration_demo import ScramjetAnalyticForwardModel


def main():
    parser = argparse.ArgumentParser(
        description="Adjoint-readiness diagnostic (POST-PHASE 10/11)"
    )
    parser.add_argument("--quick",    action="store_true")
    parser.add_argument("-o", "--save-dir", dest="save_dir", default=None,
                        metavar="PATH")
    parser.add_argument("--n-ensemble", type=int, default=None,
                        help="Ensemble size to evaluate (default: 150 full / 40 quick)")
    parser.add_argument("--d-theta", type=int, default=2,
                        help="Number of calibration parameters (default: 2)")
    args = parser.parse_args()

    quick      = args.quick
    n_ensemble = args.n_ensemble or (40 if quick else 150)
    d_theta    = args.d_theta
    n_stations = 6 if quick else 10
    n_oat      = 7 if quick else 15
    n_traj     = 8 if quick else 30

    print("=" * 60)
    print("POST-PHASE 10/11 — Adjoint-Readiness Diagnostic")
    print("=" * 60)
    print(f"  d_theta={d_theta}  n_ensemble={n_ensemble}  stations={n_stations}")

    # ------------------------------------------------------------------
    # 1. Build forward model and run sensitivity analysis
    # ------------------------------------------------------------------
    obs_full, truth, _ = scramjet_synthetic_observation_set(
        n_wall_stations=n_stations, rng_seed=0
    )
    obs_cal = obs_full.filter_by_types([
        ObservableType.WALL_PRESSURE_CP,
        ObservableType.SKIN_FRICTION_CF,
    ])
    fm = ScramjetAnalyticForwardModel(obs_cal)
    lower  = np.array([0.20, 0.05])
    upper  = np.array([0.50, 0.15])

    print("\n[1/2] Running sensitivity analysis ...")
    sa = SensitivityAnalyser(fm, (lower, upper), ["a1", "betaStar"],
                             nominal=np.array([0.31, 0.09]))
    sa.run_oat(n_points=n_oat).run_morris(n_trajectories=n_traj, rng_seed=0)
    sens_report = sa.report()

    print("  Sensitivity ranking:", " > ".join(sens_report["ranking"]))
    oat_imp = sa.oat_importance()
    print("  OAT importance:", {k: f"{v:.4f}" for k, v in oat_imp.items()})

    # ------------------------------------------------------------------
    # 2. Run adjoint-readiness diagnostic
    # ------------------------------------------------------------------
    print("\n[2/2] Adjoint-readiness diagnostic ...")
    result = run_diagnostic(
        sensitivity_report=sens_report,
        n_ensemble=n_ensemble,
        d_theta=d_theta,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # POST-PHASE 11 decision
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("POST-PHASE 11 — Final Adjoint Decision")
    print("=" * 60)

    if result.adjoint_warranted:
        decision = "IMPLEMENT"
        rationale = (
            "All three criteria met. Adjoint gradients should be implemented "
            "to reduce the per-query cost of ensemble generation."
        )
    else:
        decision = "DEFER"
        failing = [k for k, v in result.criteria.items() if not v]
        rationale = (
            f"Criteria not met ({', '.join(failing)}). "
            "The GP-surrogate approach (POST-PHASES 4–8) is sufficient. "
            "See Theory/text/DECISION_RECORD.md for full rationale."
        )

    print(f"\n  Decision:  {decision}")
    print(f"  Rationale: {rationale}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    report = {
        "d_theta":    d_theta,
        "n_ensemble": n_ensemble,
        "sensitivity_report": sens_report,
        "diagnostic": {
            "adjoint_warranted": result.adjoint_warranted,
            "criteria": result.criteria,
            "evidence": result.evidence,
            "thresholds": result.thresholds,
            "ranking": result.ranking,
        },
        "decision":  decision,
        "rationale": rationale,
    }

    if args.save_dir:
        out = Path(args.save_dir) / "adjoint_readiness"
        out.mkdir(parents=True, exist_ok=True)
        (out / "diagnostic_report.json").write_text(
            json.dumps(report, indent=2)
        )
        print(f"\n  Saved → {out}/diagnostic_report.json")

    return 0 if not result.adjoint_warranted else 1


if __name__ == "__main__":
    sys.exit(main())
