"""
POST-PHASE 6 — Multi-fidelity Bayesian calibration study.

Investigates how the posterior over [a1, β*] changes as the observation
fidelity increases.  Three levels are used:

  Level 1 — Lo-fi:    Cf only, 4 wall stations (recovers β* only)
  Level 2 — Med-fi:   Cf + Cp, 6 wall stations (recovers both a1 and β*)
  Level 3 — Hi-fi:    Cf + Cp, 10 wall stations (tighter posteriors)

All three use the same analytic SBLI forward model and synthetic ground-truth
data from ``scramjet_synthetic_observation_set`` (no CFD required).

Usage:
    python3 multi_fidelity_calibration_demo.py
    python3 multi_fidelity_calibration_demo.py --quick
    python3 multi_fidelity_calibration_demo.py -o PATH
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

from observation_schema import (
    ObservableType,
    scramjet_synthetic_observation_set,
)
from bayesian_inference import Prior
from multi_fidelity_calibration import MultiFidelityStudy

# Re-use analytic forward model from POST-PHASE 5
sys.path.insert(0, str(_SCRIPT_DIR))
from scramjet_calibration_demo import ScramjetAnalyticForwardModel


# ---------------------------------------------------------------------------
# Build fidelity levels from a single ground-truth observation set
# ---------------------------------------------------------------------------

def _make_fidelity_levels(truth_obs_full, levels_spec):
    """
    Build (label, obs_set, forward_model) triples from a full observation set.

    Parameters
    ----------
    truth_obs_full : ObservationSet
        Full-fidelity observations with all observable types.
    levels_spec : list of (label, n_stations, obs_types)
        Each tuple defines one level.
    """
    obs_results = []
    for label, n_stations, obs_types in levels_spec:
        # Sub-sample first n_stations x-values in order
        all_xs = sorted(set(o.x for o in truth_obs_full))
        xs_subset = set(all_xs[:n_stations])

        # Build filtered obs_set at subset of stations and selected types
        from observation_schema import ObservationSet
        sub_set = ObservationSet(truth_obs_full.case_id)
        for obs in truth_obs_full:
            if obs.observable_type in obs_types and obs.x in xs_subset:
                sub_set.add(obs)

        fm = ScramjetAnalyticForwardModel(sub_set)
        obs_results.append((label, sub_set, fm))
    return obs_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-fidelity calibration study (POST-PHASE 6)"
    )
    parser.add_argument("--quick",   action="store_true")
    parser.add_argument("-o", "--save-dir", dest="save_dir", default=None,
                        metavar="PATH")
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()

    quick    = args.quick
    rng_seed = args.rng_seed

    n_ensemble = 30 if quick else 120
    n_steps    = 200 if quick else 1500
    n_hi_stations = 10

    print("=" * 60)
    print("POST-PHASE 6 — Multi-fidelity Calibration Study")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Ground-truth: hi-fi synthetic observations (10 stations, all types)
    # ------------------------------------------------------------------
    obs_full, truth, meta = scramjet_synthetic_observation_set(
        n_wall_stations=n_hi_stations, mach=2.0, rng_seed=rng_seed
    )
    print(f"\nGround truth:  a1={truth['a1']:.4f}  betaStar={truth['betaStar']:.4f}")
    print(f"Observations:  {obs_full.n_obs} total"
          f"  ({n_hi_stations} stations × Cp+Cf+qw + 2 scalar locations)")

    # ------------------------------------------------------------------
    # Define fidelity levels
    # ------------------------------------------------------------------
    CF  = ObservableType.SKIN_FRICTION_CF
    CP  = ObservableType.WALL_PRESSURE_CP

    levels_spec = [
        ("lo  (4-sta/Cf)",     4, {CF}),
        ("med (6-sta/Cp+Cf)",  6, {CP, CF}),
        ("hi (10-sta/Cp+Cf)", 10, {CP, CF}),
    ]
    fidelity_levels = _make_fidelity_levels(obs_full, levels_spec)

    for label, obs, fm in fidelity_levels:
        print(f"  {label:26s}: {obs.n_obs:3d} obs")

    # ------------------------------------------------------------------
    # Run study
    # ------------------------------------------------------------------
    prior = Prior(
        means=[0.31, 0.09],
        stds=[0.05, 0.015],
        lower=[0.20, 0.05],
        upper=[0.50, 0.15],
    )

    study = MultiFidelityStudy(
        prior=prior,
        param_names=["a1", "betaStar"],
    )
    for label, obs, fm in fidelity_levels:
        study.add_level(label, obs, fm, koh_mode="diagonal")

    print(f"\nRunning {study.n_levels} levels"
          f"  (ensemble={n_ensemble}, MCMC={n_steps}) ...")
    study.run_all(
        n_ensemble=n_ensemble,
        n_steps=n_steps,
        rng_seed=rng_seed,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n=== Multi-fidelity posterior comparison ===")
    study.print_comparison(truth=truth)

    table = study.comparison_table()

    # Check: hi-fi posterior must be tighter than lo-fi for betaStar
    if len(table["convergence"]) > 0:
        beta_ratios = table["convergence"].get("betaStar", [])
        if len(beta_ratios) >= 2:
            shrinkage = beta_ratios[-1] < 1.0
            print(f"\n  betaStar CI-width shrinkage lo→hi: "
                  f"{beta_ratios[0]:.3f}→{beta_ratios[-1]:.3f}  "
                  f"({'OK' if shrinkage else 'WARN'})")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if args.save_dir:
        out = Path(args.save_dir) / "multi_fidelity_calibration"
        out.mkdir(parents=True, exist_ok=True)
        report = {
            "truth": truth,
            "levels_spec": [
                {"label": label, "n_stations": n, "obs_types": [t.value for t in types]}
                for label, n, types in levels_spec
            ],
            **table,
        }
        (out / "comparison_table.json").write_text(json.dumps(report, indent=2))
        print(f"\n  Saved → {out}/comparison_table.json")


if __name__ == "__main__":
    main()
