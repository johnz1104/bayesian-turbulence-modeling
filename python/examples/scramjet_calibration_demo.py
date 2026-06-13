"""
POST-PHASE 5 — First scramjet-relevant Bayesian calibration experiment.

Demonstrates the full Bayesian/KOH pipeline on synthetic shock-boundary-layer
interaction (SBLI) data without requiring a real hypersonic CFD solver:

  1. Generate synthetic wall observations (Cp, Cf) at Ma=2 using the parametric
     SBLI model from ``observation_schema.scramjet_synthetic_observation_set``.
  2. Build a matching analytic forward model (``ScramjetAnalyticForwardModel``)
     that evaluates the same physics at arbitrary (a1, β*) values.
  3. Run BayesianInferenceKOH with diagonal discrepancy:
       ensemble → GP surrogate → MCMC over [a1, β*, log σ_δ]
  4. Verify posterior recovers the generating parameters within 2σ.

This script does NOT call the C++ RANS solver — it is a pure-Python test bed
for the Bayesian machinery on scramjet-relevant observables.

Usage:
    python3 scramjet_calibration_demo.py            # full run (~2 min)
    python3 scramjet_calibration_demo.py --quick    # fast demo (~15 s)
    python3 scramjet_calibration_demo.py -o PATH    # save results to PATH/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PYTHON_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PYTHON_DIR))

from observation_schema import (
    ObservableType,
    koh_from_observation_set,
    scramjet_synthetic_observation_set,
)
from bayesian_inference import BayesianInferenceKOH, Prior
from forward_model_interface import ForwardModelBase, EvaluationResult


# ---------------------------------------------------------------------------
# Physics helpers (exact match to scramjet_synthetic_observation_set model)
# ---------------------------------------------------------------------------

_X_SHOCK = 0.40
_X_SEP   = 0.35
_X_REAT  = 0.65


def _cp(x, a1, beta_star):
    shock_strength = 0.5 * (1.0 + np.tanh((x - _X_SHOCK) / 0.05))
    sep_dip        = -0.05 * np.exp(-((x - _X_SEP) / 0.05) ** 2)
    return (0.3 + 0.4 * a1 / 0.31) * shock_strength + sep_dip


def _cf(x, a1, beta_star):
    base = 0.002 * beta_star / 0.09
    sep  = -0.0015 * np.exp(-((x - _X_SEP) / 0.06) ** 2)
    reat =  0.0010 * np.exp(-((x - _X_REAT) / 0.08) ** 2)
    return base + sep + reat


# ---------------------------------------------------------------------------
# Analytic forward model
# ---------------------------------------------------------------------------

class ScramjetAnalyticForwardModel(ForwardModelBase):
    """
    Analytic forward model matching the observation_schema SBLI physics.

    Evaluates _cp and _cf (and scalar locations) in the same insertion order
    as the ObservationSet used to build the KOH likelihood.

    Parameters
    ----------
    obs_set : ObservationSet
        The subset of observations used for calibration.  The forward model
        returns predictions in the same order as ``obs_set._observations``.
    """

    _SCALAR_TYPES = {
        ObservableType.SHOCK_LOCATION,
        ObservableType.REATTACHMENT_LOCATION,
        ObservableType.REATTACHMENT_LENGTH,
    }

    def __init__(self, obs_set):
        self._items = [
            (obs.x, obs.observable_type) for obs in obs_set
        ]

    def evaluate(self, theta) -> EvaluationResult:
        a1, beta_star = float(theta[0]), float(theta[1])
        preds = []
        for x, obs_type in self._items:
            if obs_type == ObservableType.WALL_PRESSURE_CP:
                preds.append(float(_cp(x, a1, beta_star)))
            elif obs_type == ObservableType.SKIN_FRICTION_CF:
                preds.append(float(_cf(x, a1, beta_star)))
            elif obs_type in self._SCALAR_TYPES:
                preds.append(float(x))   # location is the observable
            else:
                preds.append(0.0)
        return EvaluationResult(predictions=preds, converged=True,
                                status="analytic")

    def parameter_names(self):
        return ["a1", "betaStar"]


# ---------------------------------------------------------------------------
# Recovery check
# ---------------------------------------------------------------------------

def _recovery_check(samples, truth):
    """Return True if both parameters are recovered within 2σ of truth."""
    param_names = ["a1", "betaStar"]
    truth_vals  = [truth["a1"], truth["betaStar"]]
    all_ok = True

    print("\n=== Posterior summary ===")
    for i, (name, tv) in enumerate(zip(param_names, truth_vals)):
        s     = samples[:, i]
        pm    = float(np.mean(s))
        ps    = float(np.std(s))
        err   = abs(pm - tv) / max(ps, 1e-10)
        tag   = "OK" if err < 2.0 else "WARN"
        if err >= 2.0:
            all_ok = False
        print(f"  {name:12s}: truth={tv:.4f}  post={pm:.4f}±{ps:.4f}"
              f"  |err|={err:.1f}σ  [{tag}]")

    # Discrepancy hyperparameter
    sigma_d = np.exp(samples[:, 2])
    print(f"  {'σ_δ':12s}: {np.mean(sigma_d):.4f} ± {np.std(sigma_d):.4f}"
          f"  (model-form amplitude)")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scramjet KOH calibration demo (POST-PHASE 5)"
    )
    parser.add_argument("--quick",      action="store_true",
                        help="Fast settings: fewer ensemble samples, fewer MCMC steps")
    parser.add_argument("-o", "--save-dir", dest="save_dir", default=None,
                        metavar="PATH",
                        help="Write JSON summary to PATH/scramjet_calibration/")
    parser.add_argument("--n-stations",  type=int, default=None)
    parser.add_argument("--n-ensemble",  type=int, default=None)
    parser.add_argument("--n-steps",     type=int, default=None)
    parser.add_argument("--rng-seed",    type=int, default=0)
    args = parser.parse_args()

    quick = args.quick

    n_stations = args.n_stations or (6  if quick else 10)
    n_ensemble = args.n_ensemble or (40  if quick else 150)
    n_steps    = args.n_steps    or (300 if quick else 2000)
    burn_in    = max(50, n_steps // 5)
    rng_seed   = args.rng_seed

    print("=" * 60)
    print("POST-PHASE 5 — Scramjet KOH Calibration Demo")
    print("=" * 60)
    print(f"  Stations: {n_stations}  Ensemble: {n_ensemble}"
          f"  MCMC steps: {n_steps}  Seed: {rng_seed}")

    # ------------------------------------------------------------------
    # 1. Synthetic observations
    # ------------------------------------------------------------------
    print("\n[1/4] Generating synthetic SBLI observations ...")
    obs_full, truth, metadata = scramjet_synthetic_observation_set(
        n_wall_stations=n_stations, mach=2.0, rng_seed=rng_seed
    )
    print(f"  Truth:   a1={truth['a1']:.4f}  betaStar={truth['betaStar']:.4f}")
    print(f"  Data:    {obs_full.n_obs} obs  "
          f"(Ma={metadata.mach}  Re={metadata.reynolds:.0f})")

    # Calibrate on Cp + Cf (jointly identifies both a1 and betaStar)
    obs_cal = obs_full.filter_by_types([
        ObservableType.WALL_PRESSURE_CP,
        ObservableType.SKIN_FRICTION_CF,
    ])
    print(f"  Calibration set: {obs_cal.n_obs} obs (Cp + Cf)")

    # ------------------------------------------------------------------
    # 2. Forward model + KOH likelihood
    # ------------------------------------------------------------------
    print("\n[2/4] Building forward model and KOH likelihood ...")
    forward_model = ScramjetAnalyticForwardModel(obs_cal)
    koh = koh_from_observation_set(obs_cal, mode="diagonal")
    print(f"  KOH n={koh.n}  mode={koh.mode}")
    assert koh.n == obs_cal.n_obs

    # Prior centred on Menter defaults ±15 %
    prior = Prior(
        means=[0.31, 0.09],
        stds=[0.05, 0.015],
        lower=[0.20, 0.05],
        upper=[0.50, 0.15],
    )
    param_set = {"defaults": [0.31, 0.09], "lower": [0.20, 0.05],
                 "upper": [0.50, 0.15], "names": ["a1", "betaStar"]}

    koh_bi = BayesianInferenceKOH(
        forward_model=forward_model,
        param_set=param_set,
        koh_likelihood=koh,
        theta_prior=prior,
    )

    # ------------------------------------------------------------------
    # 3. Ensemble → surrogate → MCMC
    # ------------------------------------------------------------------
    print(f"\n[3/4] Running ensemble ({n_ensemble} samples) ...")
    t0 = time.time()
    koh_bi.run_ensemble(n_samples=n_ensemble, verbose=False)
    print(f"  Ensemble: {len(koh_bi.ensemble_X)} valid"
          f"  [{time.time()-t0:.1f}s]")

    print("  Training surrogate ...")
    koh_bi.train_surrogate(verbose=False)

    print(f"  Running MCMC ({n_steps} steps) ...")
    t1 = time.time()
    koh_bi.run_mcmc(n_steps=n_steps, burn_in=burn_in, verbose=False,
                    rng_seed=rng_seed)
    print(f"  MCMC complete: {len(koh_bi.samples)} samples"
          f"  [{time.time()-t1:.1f}s]")

    # ------------------------------------------------------------------
    # 4. Recovery check
    # ------------------------------------------------------------------
    print("\n[4/4] Recovery check ...")
    ok = _recovery_check(koh_bi.samples, truth)
    status_str = "PASS" if ok else "WARN"
    print(f"\n  Recovery: {status_str}")

    # ------------------------------------------------------------------
    # Optional: save results
    # ------------------------------------------------------------------
    if args.save_dir:
        out = Path(args.save_dir) / "scramjet_calibration"
        out.mkdir(parents=True, exist_ok=True)

        summary = koh_bi.posterior_summary()
        report = {
            "truth": truth,
            "n_stations": n_stations,
            "n_obs_calibration": obs_cal.n_obs,
            "n_ensemble": int(len(koh_bi.ensemble_X)),
            "n_mcmc_samples": int(len(koh_bi.samples)),
            "koh_mode": koh.mode,
            "posterior": summary,
            "recovery": {
                name: {
                    "truth": float(truth[name]),
                    "post_mean": float(summary[name]["mean"]),
                    "post_std": float(summary[name]["std"]),
                    "err_sigma": abs(summary[name]["mean"] - truth[name])
                                / max(summary[name]["std"], 1e-10),
                }
                for name in ["a1", "betaStar"]
            },
            "status": status_str,
        }
        (out / "posterior_summary.json").write_text(
            json.dumps(report, indent=2)
        )
        print(f"\n  Results saved → {out}/posterior_summary.json")


if __name__ == "__main__":
    main()
