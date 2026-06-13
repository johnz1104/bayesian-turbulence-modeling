"""
Observable Ablation Demo
===========================
Phase 10 infrastructure hardening.

Synthetic planning study: which combination of wall observables (Cp, Cf, qw)
reduces posterior uncertainty the most?  This informs where future
experimental measurements or DNS quantities will have the highest value for
SST calibration.

IMPORTANT: All data is SYNTHETIC — this is a planning study for future
experiments, not a real scramjet calibration result.

Observable combinations tested
--------------------------------
1. Cf only
2. Cp only
3. qw only
4. Cp + Cf
5. Cp + Cf + qw
6. Cp + Cf + qw + shock/separation scalars (if available)

Outputs
-------
    ablation_table.json     — posterior widths and RMSE for each combination
    ablation_summary.png    — bar chart of posterior CI width vs combo
    ablation_detail.json    — full posterior summaries for each combo

Usage
-----
    python observable_ablation_demo.py --quick -o /tmp/ablation_out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic SBLI forward model
# ---------------------------------------------------------------------------

X_SHOCK, X_SEP, X_REAT = 0.40, 0.35, 0.65

A1_TRUE      = 0.31
BETASTAR_TRUE = 0.09
RNG_SEED     = 42

_N_STATIONS = 8
_OBS_X = np.linspace(0.0, 1.0, _N_STATIONS)


def _cp_model(x, a1, beta_star):
    shock = 0.5 * (1 + np.tanh((x - X_SHOCK) / 0.05))
    dip   = -0.05 * np.exp(-((x - X_SEP) / 0.05) ** 2)
    return (0.3 + 0.4 * a1 / 0.31) * shock + dip


def _cf_model(x, a1, beta_star):
    base = 0.002 * beta_star / 0.09
    sep  = -0.0015 * np.exp(-((x - X_SEP)  / 0.06) ** 2)
    reat =  0.0010 * np.exp(-((x - X_REAT) / 0.08) ** 2)
    return base + sep + reat


def _qw_model(x, a1, beta_star):
    base = 50_000.0 * beta_star / 0.09
    peak = 80_000.0 * np.exp(-((x - X_REAT) / 0.07) ** 2)
    return base + peak


def _predict_combo(theta, types: List[str]) -> np.ndarray:
    a1, bs = float(theta[0]), float(theta[1])
    parts  = []
    if "cp"   in types: parts.append(_cp_model(_OBS_X, a1, bs))
    if "cf"   in types: parts.append(_cf_model(_OBS_X, a1, bs))
    if "qw"   in types: parts.append(_qw_model(_OBS_X, a1, bs))
    if "xsep" in types: parts.append(np.array([X_SEP + 0.05 * (a1 / 0.31 - 1.0)]))
    if "xreat" in types: parts.append(np.array([X_REAT + 0.10 * (bs / 0.09 - 1.0)]))
    return np.concatenate(parts)


def _build_obs(types: List[str], noise_frac: float = 0.04, rng_seed: int = 0):
    rng   = np.random.default_rng(rng_seed)
    truth = _predict_combo([A1_TRUE, BETASTAR_TRUE], types)
    sigma = np.abs(truth) * noise_frac + 1e-8
    obs   = truth + rng.standard_normal(len(truth)) * sigma
    locs  = np.concatenate([
        _OBS_X if t in ("cp", "cf", "qw") else np.array([0.5])
        for t in types
    ])
    return obs, sigma, locs, truth


def _build_ensemble(types: List[str], n_samples: int, rng_seed: int = 0):
    rng   = np.random.default_rng(rng_seed + 100)
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
        preds = _predict_combo(theta, types)
        if np.all(np.isfinite(preds)):
            rows_X.append(theta)
            rows_Y.append(preds)
    return np.array(rows_X), np.array(rows_Y)


# ---------------------------------------------------------------------------
# Run one calibration for a given observable combination
# ---------------------------------------------------------------------------

def _run_calibration(label: str, types: List[str],
                     n_ensemble: int, n_walkers: int,
                     n_steps: int, burn_in: int,
                     optimize_restarts: int = 3) -> Dict:
    from bayesian_inference import BayesianInferenceKOH, Prior, KOHLikelihood

    obs_vals, obs_sigs, obs_locs, truth = _build_obs(types)
    X_ens, Y_ens = _build_ensemble(types, n_samples=n_ensemble)

    if len(X_ens) < 6:
        return {"label": label, "error": "insufficient valid ensemble samples"}

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

    koh = KOHLikelihood(obs_locs, obs_vals, obs_sigs, mode="diagonal")

    _Xens, _Yens = X_ens, Y_ens

    class _PreFM:
        def evaluate(self, theta):
            from forward_model_interface import EvaluationResult
            delta = (_Xens - np.array(theta)) / np.maximum(_Xens.std(0), 1e-12)
            idx   = int(np.argmin(np.sum(delta ** 2, axis=1)))
            return EvaluationResult(
                _Yens[idx].tolist(), converged=True, status="precomputed"
            )
        def parameter_names(self):
            return ["a1", "betaStar"]
        def precomputed_ensemble(self):
            return _Xens, _Yens

    bi = BayesianInferenceKOH(
        forward_model=_PreFM(),
        param_set=param_set,
        koh_likelihood=koh,
        theta_prior=prior,
    )
    bi.run_ensemble(n_samples=n_ensemble, verbose=False)
    bi.train_surrogate(verbose=False, optimize_restarts=optimize_restarts)
    bi.run_mcmc(n_walkers=n_walkers, n_steps=n_steps,
                burn_in=burn_in, verbose=False)
    summary = bi.posterior_summary()

    a1_mean   = summary["a1"]["mean"]
    a1_std    = summary["a1"]["std"]
    bs_mean   = summary["betaStar"]["mean"]
    bs_std    = summary["betaStar"]["std"]
    a1_ci95   = summary["a1"]["ci_97.5"] - summary["a1"]["ci_2.5"]
    bs_ci95   = summary["betaStar"]["ci_97.5"] - summary["betaStar"]["ci_2.5"]

    # Posterior predictive RMSE
    if bi.samples is not None and len(bi.samples) > 0:
        theta_mean = bi.samples[:, :2].mean(axis=0)
        pred_mean  = _predict_combo(theta_mean, types)
        pprmse     = float(np.sqrt(np.mean((pred_mean - truth) ** 2)))
    else:
        pprmse = float("nan")

    return {
        "label":        label,
        "types":        types,
        "n_obs":        len(obs_vals),
        "a1_mean":      float(a1_mean),
        "a1_std":       float(a1_std),
        "a1_ci95":       float(a1_ci95),
        "a1_error":      float(abs(a1_mean - A1_TRUE)),
        "betaStar_mean": float(bs_mean),
        "betaStar_std":  float(bs_std),
        "betaStar_ci95": float(bs_ci95),
        "betaStar_error": float(abs(bs_mean - BETASTAR_TRUE)),
        "pp_rmse":      pprmse,
        "summary":      summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo(out_dir: Path, quick: bool = False):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    out_dir.mkdir(parents=True, exist_ok=True)

    n_ensemble       = 15 if quick else 100
    n_walkers        = 4  if quick else 20
    n_steps          = 60  if quick else 600
    burn_in          = 10  if quick else 150
    optimize_restarts = 1  if quick else 3

    all_combos = [
        ("Cf only",             ["cf"]),
        ("Cp only",             ["cp"]),
        ("qw only",             ["qw"]),
        ("Cp + Cf",             ["cp", "cf"]),
        ("Cp + Cf + qw",        ["cp", "cf", "qw"]),
        ("Cp+Cf+qw+locations",  ["cp", "cf", "qw", "xsep", "xreat"]),
    ]
    # Quick mode runs only the 3 most informative combos to keep CI fast.
    combos = all_combos[:3] if quick else all_combos

    print(f"\nObservable ablation study (SYNTHETIC, n_ensemble={n_ensemble})")
    print(f"  True: a1={A1_TRUE}, betaStar={BETASTAR_TRUE}")
    print(f"  Testing {len(combos)} observable combinations\n")

    results = []
    for label, types in combos:
        print(f"  Running: {label} ({len(types)} type(s), "
              f"{_N_STATIONS * len([t for t in types if t in ('cp','cf','qw')]) + sum(1 for t in types if t in ('xsep','xreat'))} obs) ...")
        r = _run_calibration(
            label, types, n_ensemble, n_walkers, n_steps, burn_in,
            optimize_restarts=optimize_restarts,
        )
        results.append(r)
        a1_ci = r.get("a1_ci95", float("nan"))
        bs_ci = r.get("betaStar_ci95", float("nan"))
        print(f"    a1 CI95={a1_ci:.4f}  betaStar CI95={bs_ci:.5f}  "
              f"ppRMSE={r.get('pp_rmse', float('nan')):.4e}")

    # ------------------------------------------------------------------
    # Save ablation table
    # ------------------------------------------------------------------
    table = []
    for r in results:
        table.append({
            "combo":            r["label"],
            "n_obs":            r.get("n_obs", 0),
            "a1_CI95_width":    r.get("a1_ci95", None),
            "betaStar_CI95_width": r.get("betaStar_ci95", None),
            "a1_posterior_error": r.get("a1_error", None),
            "betaStar_posterior_error": r.get("betaStar_error", None),
            "pp_rmse":          r.get("pp_rmse", None),
        })
    (out_dir / "ablation_table.json").write_text(json.dumps(table, indent=2))

    detail = {r["label"]: r.get("summary", {}) for r in results}
    detail["_note"] = (
        "SYNTHETIC PLANNING STUDY — all data is synthetic. "
        "Posterior widths show the relative information content of each "
        "observable combination, not a real scramjet calibration result."
    )
    (out_dir / "ablation_detail.json").write_text(json.dumps(detail, indent=2))

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'Observable combo':<28} {'a1 CI95':>10} {'β* CI95':>12} {'ppRMSE':>12}")
    print("-" * 72)
    for r in results:
        print(f"  {r['label']:<26} "
              f"{r.get('a1_ci95', float('nan')):>10.4f} "
              f"{r.get('betaStar_ci95', float('nan')):>12.5f} "
              f"{r.get('pp_rmse', float('nan')):>12.4e}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r["label"] for r in results]
        a1_widths = [r.get("a1_ci95", 0.0) for r in results]
        bs_widths = [r.get("betaStar_ci95", 0.0) for r in results]

        x = np.arange(len(labels))
        width = 0.35
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(x - width/2, a1_widths, width, label="a1 CI95", color="steelblue")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axes[0].set_ylabel("95% CI width")
        axes[0].set_title("a1 posterior uncertainty")
        axes[0].legend()

        axes[1].bar(x + width/2, bs_widths, width,
                    label="β* CI95", color="darkorange")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axes[1].set_ylabel("95% CI width")
        axes[1].set_title("β* posterior uncertainty")
        axes[1].legend()

        fig.suptitle(
            "Observable Ablation: which measurements reduce uncertainty most?\n"
            "(SYNTHETIC planning study — not real data)",
            fontsize=9, style="italic",
        )
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_summary.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: ablation_summary.png")
    except ImportError:
        print("  (matplotlib not available — skipping plot)")

    print(f"\nOutputs written to: {out_dir}")
    print("NOTE: SYNTHETIC data only — this is a planning study for future")
    print("      experiments/DNS, not a real scramjet calibration result.")


def main():
    parser = argparse.ArgumentParser(
        description="Observable ablation: which measurements reduce posterior uncertainty?"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Reduced budget for CI/testing")
    parser.add_argument("-o", "--out-dir",
                        default="outputs/observable_ablation",
                        help="Output directory")
    args = parser.parse_args()

    run_demo(out_dir=Path(args.out_dir), quick=args.quick)


if __name__ == "__main__":
    main()
