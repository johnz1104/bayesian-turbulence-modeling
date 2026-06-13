"""
KOH Stress Tests — Synthetic High-Mach Scenarios
==================================================
Phase 10 infrastructure hardening.

Verifies that the KOH model behaves sensibly across five synthetic test cases
before applying it to real high-Mach data:

    Case 1: No discrepancy        — KOH should not invent large δ
    Case 2: Smooth spatial δ      — physical GP should outperform diagonal
    Case 3: Shock-localised δ     — localized discrepancy at shock region
    Case 4: Pure measurement noise — δ should not overfit noise as GP signal
    Case 5: High-noise sparse data — should trigger identifiability warning

All data is SYNTHETIC — no real high-Mach measurements are used.

Usage
-----
    python koh_stress_test.py --quick -o /tmp/koh_stress_out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_OBS_X = np.linspace(0.0, 1.0, 8)
X_SHOCK = 0.40
A1_TRUE, BS_TRUE = 0.31, 0.09


def _base_cf(x, a1=A1_TRUE, bs=BS_TRUE):
    base = 0.002 * bs / 0.09
    sep  = -0.0015 * np.exp(-((x - 0.35) / 0.06) ** 2)
    reat =  0.0010 * np.exp(-((x - 0.65) / 0.08) ** 2)
    return base + sep + reat


def _base_cp(x, a1=A1_TRUE, bs=BS_TRUE):
    shock = 0.5 * (1 + np.tanh((x - X_SHOCK) / 0.05))
    return (0.3 + 0.4 * a1 / 0.31) * shock


def _true_predictions(a1, bs):
    return np.concatenate([_base_cp(_OBS_X, a1, bs),
                           _base_cf(_OBS_X, a1, bs)])


def _build_ensemble(n_samples: int, rng_seed: int = 0):
    rng   = np.random.default_rng(rng_seed)
    lower = np.array([0.22, 0.06])
    upper = np.array([0.45, 0.13])
    n, d  = n_samples, 2
    X_lhc = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        X_lhc[:, j] = (lower[j]
                        + (upper[j] - lower[j]) * (perm + rng.random(n)) / n)
    rows_X, rows_Y = [], []
    for theta in X_lhc:
        preds = _true_predictions(*theta)
        if np.all(np.isfinite(preds)):
            rows_X.append(theta)
            rows_Y.append(preds)
    return np.array(rows_X), np.array(rows_Y)


def _run_koh(X_ens, Y_ens, obs_vals, obs_sigs, obs_locs, mode: str,
             n_walkers: int, n_steps: int, burn_in: int,
             optimize_restarts: int = 3) -> Dict:
    from bayesian_inference import BayesianInferenceKOH, Prior, KOHLikelihood

    prior = Prior(
        means=[0.31, 0.09], stds=[0.05, 0.015],
        lower=[0.20, 0.05], upper=[0.50, 0.15],
    )
    param_set = {
        "defaults": [0.31, 0.09],
        "lower":    [0.20, 0.05],
        "upper":    [0.50, 0.15],
        "names":    ["a1", "betaStar"],
    }
    koh = KOHLikelihood(obs_locs, obs_vals, obs_sigs, mode=mode)

    _X, _Y = X_ens, Y_ens

    class _FM:
        def evaluate(self, theta):
            from forward_model_interface import EvaluationResult
            idx = int(np.argmin(np.sum(
                ((_X - np.array(theta)) / np.maximum(_X.std(0), 1e-12)) ** 2,
                axis=1)))
            return EvaluationResult(_Y[idx].tolist(), converged=True,
                                    status="precomputed")
        def parameter_names(self):
            return ["a1", "betaStar"]
        def precomputed_ensemble(self):
            return _X, _Y

    bi = BayesianInferenceKOH(
        forward_model=_FM(),
        param_set=param_set,
        koh_likelihood=koh,
        theta_prior=prior,
    )
    bi.run_ensemble(n_samples=len(X_ens), verbose=False)
    bi.train_surrogate(verbose=False, optimize_restarts=optimize_restarts)
    bi.run_mcmc(n_walkers=n_walkers, n_steps=n_steps,
                burn_in=burn_in, verbose=False)

    summary = bi.posterior_summary()
    a1_std  = summary["a1"]["std"]
    bs_std  = summary["betaStar"]["std"]
    a1_err  = abs(summary["a1"]["mean"] - A1_TRUE)
    bs_err  = abs(summary["betaStar"]["mean"] - BS_TRUE)

    # Discrepancy magnitude (exp of log_sigma_delta if present)
    delta_sigma = None
    if "log_sigma_delta" in summary:
        delta_sigma = float(np.exp(summary["log_sigma_delta"]["mean"]))

    return {
        "mode":        mode,
        "a1_mean":     float(summary["a1"]["mean"]),
        "a1_std":      float(a1_std),
        "a1_error":    float(a1_err),
        "bs_mean":     float(summary["betaStar"]["mean"]),
        "bs_std":      float(bs_std),
        "bs_error":    float(bs_err),
        "delta_sigma": delta_sigma,
        "summary":     summary,
    }


def _stress_case(case_name: str, obs_vals, obs_sigs, obs_locs,
                 X_ens, Y_ens, modes: List[str],
                 n_walkers: int, n_steps: int, burn_in: int,
                 optimize_restarts: int = 3) -> Dict:
    results = {}
    for mode in modes:
        r = _run_koh(X_ens, Y_ens, obs_vals, obs_sigs, obs_locs,
                     mode, n_walkers, n_steps, burn_in,
                     optimize_restarts=optimize_restarts)
        results[mode] = r
    return {"case": case_name, "modes": results}


def run_demo(out_dir: Path, quick: bool = False):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    out_dir.mkdir(parents=True, exist_ok=True)

    n_ensemble        = 15 if quick else 80
    n_walkers         = 4  if quick else 16
    n_steps           = 60  if quick else 500
    burn_in           = 10  if quick else 120
    optimize_restarts = 1  if quick else 3

    rng = np.random.default_rng(99)
    X_ens, Y_ens = _build_ensemble(n_ensemble)

    # Common observation locations
    obs_locs = np.concatenate([_OBS_X, _OBS_X])

    all_cases = []

    # ------------------------------------------------------------------
    # Case 1: No discrepancy — observations = true model at truth
    # ------------------------------------------------------------------
    print("[1/5] Case 1: No discrepancy ...")
    obs_true  = _true_predictions(A1_TRUE, BS_TRUE)
    obs_sigma = np.abs(obs_true) * 0.02 + 1e-8
    obs_noisy = obs_true + rng.standard_normal(len(obs_true)) * obs_sigma

    r1 = _stress_case(
        "no_discrepancy", obs_noisy, obs_sigma, obs_locs,
        X_ens, Y_ens, ["diagonal", "physical_gp"],
        n_walkers, n_steps, burn_in,
        optimize_restarts=optimize_restarts,
    )
    all_cases.append(r1)

    for mode, rd in r1["modes"].items():
        ds = rd.get("delta_sigma")
        ds_str = f"{ds:.4f}" if ds is not None else "N/A"
        print(f"    {mode:12s}: a1_err={rd['a1_error']:.4f}  "
              f"a1_std={rd['a1_std']:.4f}  delta_sigma={ds_str}")

    # ------------------------------------------------------------------
    # Case 2: Smooth spatial discrepancy
    # ------------------------------------------------------------------
    print("[2/5] Case 2: Smooth spatial discrepancy ...")
    smooth_delta = 0.05 * np.sin(np.pi * obs_locs)   # smooth, physically motivated
    obs_sig2     = np.abs(obs_true) * 0.02 + 1e-8
    obs_noisy2   = obs_true + smooth_delta + rng.standard_normal(len(obs_true)) * obs_sig2

    r2 = _stress_case(
        "smooth_discrepancy", obs_noisy2, obs_sig2, obs_locs,
        X_ens, Y_ens, ["diagonal", "physical_gp"],
        n_walkers, n_steps, burn_in,
        optimize_restarts=optimize_restarts,
    )
    all_cases.append(r2)
    for mode, rd in r2["modes"].items():
        print(f"    {mode:12s}: a1_err={rd['a1_error']:.4f}  "
              f"a1_std={rd['a1_std']:.4f}")

    # ------------------------------------------------------------------
    # Case 3: Localised shock-region discrepancy
    # ------------------------------------------------------------------
    print("[3/5] Case 3: Shock-region localised discrepancy ...")
    shock_delta  = np.zeros(len(obs_locs))
    shock_region = np.abs(obs_locs - X_SHOCK) < 0.15
    shock_delta[shock_region] = 0.08   # large discrepancy near shock
    obs_sig3     = np.abs(obs_true) * 0.02 + 1e-8
    obs_noisy3   = obs_true + shock_delta + rng.standard_normal(len(obs_true)) * obs_sig3

    r3 = _stress_case(
        "shock_localised_discrepancy", obs_noisy3, obs_sig3, obs_locs,
        X_ens, Y_ens, ["diagonal", "physical_gp"],
        n_walkers, n_steps, burn_in,
        optimize_restarts=optimize_restarts,
    )
    all_cases.append(r3)
    for mode, rd in r3["modes"].items():
        print(f"    {mode:12s}: a1_err={rd['a1_error']:.4f}  "
              f"a1_std={rd['a1_std']:.4f}")

    # ------------------------------------------------------------------
    # Case 4: Pure measurement noise — no model discrepancy
    # ------------------------------------------------------------------
    print("[4/5] Case 4: Pure measurement noise (high σ_obs) ...")
    obs_sig4   = np.abs(obs_true) * 0.20 + 1e-8   # 20% noise — noisy obs
    obs_noisy4 = obs_true + rng.standard_normal(len(obs_true)) * obs_sig4

    r4 = _stress_case(
        "pure_measurement_noise", obs_noisy4, obs_sig4, obs_locs,
        X_ens, Y_ens, ["diagonal"],
        n_walkers, n_steps, burn_in,
        optimize_restarts=optimize_restarts,
    )
    all_cases.append(r4)
    for mode, rd in r4["modes"].items():
        print(f"    {mode:12s}: a1_err={rd['a1_error']:.4f}  "
              f"a1_std={rd['a1_std']:.4f}")

    # ------------------------------------------------------------------
    # Case 5: High-noise sparse data → identifiability warning
    # ------------------------------------------------------------------
    print("[5/5] Case 5: High-noise sparse data ...")
    sparse_x     = _OBS_X[[0, 3, 7]]   # only 3 Cf stations
    sparse_true  = _base_cf(sparse_x)
    sparse_sigma = np.abs(sparse_true) * 0.50 + 1e-8   # 50% noise
    sparse_obs   = sparse_true + rng.standard_normal(len(sparse_true)) * sparse_sigma

    sparse_X = X_ens.copy()
    sparse_Y = np.array([_base_cf(sparse_x, *th) for th in X_ens])

    r5 = _stress_case(
        "high_noise_sparse", sparse_obs, sparse_sigma, sparse_x,
        sparse_X, sparse_Y, ["diagonal"],
        n_walkers, n_steps, burn_in,
        optimize_restarts=optimize_restarts,
    )
    all_cases.append(r5)
    for mode, rd in r5["modes"].items():
        print(f"    {mode:12s}: a1_err={rd['a1_error']:.4f}  "
              f"a1_std={rd['a1_std']:.4f}  → wide posterior expected")

    # ------------------------------------------------------------------
    # Compile report
    # ------------------------------------------------------------------
    print("\n=== KOH STRESS TEST REPORT ===")
    trustworthy = []
    suspicious  = []

    for case in all_cases:
        for mode, rd in case["modes"].items():
            a1_z   = rd["a1_error"] / max(rd["a1_std"], 1e-6)
            bs_z   = rd["bs_error"] / max(rd["bs_std"], 1e-6)
            overconf = rd["a1_std"] < 0.005 and rd["a1_error"] > 0.05
            if overconf:
                suspicious.append(f"{case['case']} / {mode}")
            elif a1_z < 3 and bs_z < 3:
                trustworthy.append(f"{case['case']} / {mode}")

    print(f"  Trustworthy (|z| < 3): {len(trustworthy)}")
    for t in trustworthy:
        print(f"    ✓ {t}")
    if suspicious:
        print(f"  Suspicious (overconfident): {len(suspicious)}")
        for s in suspicious:
            print(f"    ✗ {s}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    report = {
        "note": "KOH stress test — SYNTHETIC data only",
        "cases": [
            {
                "case":  c["case"],
                "modes": {
                    mode: {
                        k: v for k, v in rd.items() if k != "summary"
                    }
                    for mode, rd in c["modes"].items()
                },
            }
            for c in all_cases
        ],
        "trustworthy": trustworthy,
        "suspicious":  suspicious,
    }
    (out_dir / "koh_stress_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nReport saved to: {out_dir}/koh_stress_report.json")
    print("NOTE: All data is SYNTHETIC — results do not imply real scramjet validation.")


def main():
    parser = argparse.ArgumentParser(
        description="KOH stress tests on synthetic high-Mach scenarios"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Reduced budget for CI")
    parser.add_argument("-o", "--out-dir",
                        default="outputs/koh_stress_tests",
                        help="Output directory")
    args = parser.parse_args()

    run_demo(out_dir=Path(args.out_dir), quick=args.quick)


if __name__ == "__main__":
    main()
