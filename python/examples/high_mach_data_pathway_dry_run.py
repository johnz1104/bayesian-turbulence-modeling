"""
High-Mach Data Pathway — Dry Run
===================================
Phase 10 infrastructure hardening.

This is a DRY RUN of the future real-data workflow.  It uses SYNTHETIC wall
data loaded from a CSV/NPZ file to demonstrate the full pipeline from raw
measurement file to posterior plots — without any real high-Mach data or CFD
solver.

When real SBLI/scramjet wall data becomes available, replace the synthetic
fixture with a real CSV/NPZ (see format notes below) and re-run this script
unchanged.  No calibration code changes are required.

How to convert future real data
---------------------------------
1. Format your data as a CSV:
       x,cp,cp_sigma,cf,cf_sigma,qw,qw_sigma
       0.0,0.05,0.003,...
   where x is streamwise location (nondim by reference length).
   Supported column aliases: see data_adapters.py _COL_TO_TYPE for the full list.

2. Or as an NPZ:
       np.savez("data.npz", x=..., cp=..., cp_sigma=..., cf=..., ...)

3. Call:
       obs = load_wall_data("data.csv", case_id="my_experiment")

4. Run this script with --data-file pointing to your file.

Outputs
-------
    loaded_observations.json    — full ObservationSet in standard format
    posterior_summary.json      — standard Bayes posterior
    koh_diagonal_summary.json   — KOH (diagonal discrepancy) posterior
    koh_physical_summary.json   — KOH (spatial GP discrepancy) posterior
    posterior_predictive.png    — posterior predictive vs. observations
    residuals_by_observable.png — posterior residuals per observable type

Usage
-----
    python high_mach_data_pathway_dry_run.py --quick -o /tmp/drp_out
    python high_mach_data_pathway_dry_run.py --data-file my_data.csv -o /tmp/real_run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_FIXTURE = (Path(__file__).parent.parent.parent
            / "tests" / "fixtures" / "high_mach" / "cp_cf_qw_combined.csv")

X_SHOCK, X_SEP, X_REAT = 0.40, 0.35, 0.65


def _analytic_predict(x_arr, a1, beta_star, which="cf"):
    """Parametric SBLI model — forward model stand-in (SYNTHETIC)."""
    if which == "cf":
        base = 0.002 * beta_star / 0.09
        sep  = -0.0015 * np.exp(-((x_arr - X_SEP)  / 0.06) ** 2)
        reat =  0.0010 * np.exp(-((x_arr - X_REAT) / 0.08) ** 2)
        return base + sep + reat
    elif which == "cp":
        shock = 0.5 * (1 + np.tanh((x_arr - X_SHOCK) / 0.05))
        dip   = -0.05 * np.exp(-((x_arr - X_SEP) / 0.05) ** 2)
        return (0.3 + 0.4 * a1 / 0.31) * shock + dip
    elif which == "qw":
        base = 50_000.0 * beta_star / 0.09
        peak = 80_000.0 * np.exp(-((x_arr - X_REAT) / 0.07) ** 2)
        return base + peak
    raise ValueError(f"Unknown observable: {which}")


class _SBLIForwardModel:
    """Analytic SBLI forward model — replace with real CFD for actual calibration."""

    def __init__(self, obs_x_cp, obs_x_cf, obs_x_qw):
        self._xcp = np.asarray(obs_x_cp)
        self._xcf = np.asarray(obs_x_cf)
        self._xqw = np.asarray(obs_x_qw)

    def evaluate(self, theta):
        from forward_model_interface import EvaluationResult
        a1, bs = float(theta[0]), float(theta[1])
        preds = np.concatenate([
            _analytic_predict(self._xcp, a1, bs, "cp"),
            _analytic_predict(self._xcf, a1, bs, "cf"),
            _analytic_predict(self._xqw, a1, bs, "qw"),
        ])
        return EvaluationResult(
            predictions=preds.tolist(),
            converged=np.all(np.isfinite(preds)),
        )

    def parameter_names(self):
        return ["a1", "betaStar"]

    def precomputed_ensemble(self):
        return None


def _build_ensemble(fm, n_samples, rng_seed=0):
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
        r = fm.evaluate(theta)
        if r.converged and r.predictions and all(np.isfinite(p) for p in r.predictions):
            rows_X.append(theta)
            rows_Y.append(r.predictions)
    return np.array(rows_X), np.array(rows_Y)


def _plot_posterior_predictive(obs_x, obs_vals, obs_sigs, obs_types,
                                fm, samples, out_path):
    """Plot posterior predictive samples vs observations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    type_map  = {"cp": 0, "cf": 1, "qw": 2}
    titles    = ["Wall Pressure (Cp)", "Skin Friction (Cf)", "Heat Flux (qw)"]
    x_dense   = np.linspace(0, 1, 80)

    from collections import defaultdict
    by_type = defaultdict(list)
    for x, v, s, t in zip(obs_x, obs_vals, obs_sigs, obs_types):
        by_type[t].append((x, v, s))

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(samples), size=min(50, len(samples)), replace=False)

    for col, (tname, ax) in enumerate(zip(["cp", "cf", "qw"], axes)):
        for idx in sample_idx:
            a1, bs = float(samples[idx, 0]), float(samples[idx, 1])
            ax.plot(x_dense, _analytic_predict(x_dense, a1, bs, tname),
                    color="steelblue", alpha=0.06, linewidth=0.8)
        if tname in by_type:
            xs_obs = np.array([v[0] for v in by_type[tname]])
            vs_obs = np.array([v[1] for v in by_type[tname]])
            ss_obs = np.array([v[2] for v in by_type[tname]])
            ax.errorbar(xs_obs, vs_obs, yerr=2 * ss_obs, fmt="ko",
                        markersize=4, capsize=3, label="Obs ± 2σ")
        ax.set_title(titles[col], fontsize=10)
        ax.set_xlabel("x / L")
        ax.legend(fontsize=8)

    fig.suptitle("Posterior Predictive (SYNTHETIC dry run — not real data)",
                 fontsize=10, style="italic")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def run_demo(data_file: Path, out_dir: Path, quick: bool = False):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data_adapters import load_wall_data
    from observation_schema import (
        ObservableType, CaseMetadata, CaseData, GeometryType,
        WallCondition, DataSource,
    )
    from bayesian_inference import BayesianInferenceKOH, Prior, KOHLikelihood
    from forward_model_interface import EvaluationResult

    out_dir.mkdir(parents=True, exist_ok=True)

    n_ensemble = 35 if quick else 120
    n_walkers  = 8  if quick else 20
    n_steps    = 200 if quick else 800
    burn_in    = 40  if quick else 200

    # ------------------------------------------------------------------
    # 1. Load observations
    # ------------------------------------------------------------------
    print(f"\n[1/7] Loading observations from {data_file} ...")
    obs = load_wall_data(data_file, case_id="high_mach_drp",
                         default_uncertainty_frac=0.03)
    print(f"      {obs.n_obs} observations loaded")
    print(f"      Types: {set(t.value for t in obs.types())}")

    meta = CaseMetadata(
        case_id               = "high_mach_drp",
        geometry_type         = GeometryType.RAMP,
        mach                  = 2.0,
        reynolds              = 1.0e6,
        stagnation_pressure   = 782448.0,
        stagnation_temperature = 540.0,
        wall_condition        = WallCondition.ADIABATIC,
        data_source           = DataSource.SYNTHETIC,
        notes                 = "Synthetic fixture — DRY RUN, not real data",
    )
    case = CaseData(meta, obs)
    case.to_json(out_dir / "loaded_observations.json")
    print(f"      Saved: loaded_observations.json")

    # ------------------------------------------------------------------
    # 2. Build forward model (analytic SBLI stand-in)
    # ------------------------------------------------------------------
    print("[2/7] Building forward model (SYNTHETIC analytic SBLI) ...")
    cp_obs = obs.filter_by_type(ObservableType.WALL_PRESSURE_CP)
    cf_obs = obs.filter_by_type(ObservableType.SKIN_FRICTION_CF)
    qw_obs = obs.filter_by_type(ObservableType.WALL_HEAT_FLUX)

    # Fall back to Cf-only if Cp/qw are absent
    if cp_obs.n_obs == 0:
        cp_obs = cf_obs
    if qw_obs.n_obs == 0:
        qw_obs = cf_obs

    fm = _SBLIForwardModel(
        obs_x_cp=cp_obs.locations_x() if cp_obs.n_obs else np.linspace(0, 1, 4),
        obs_x_cf=cf_obs.locations_x() if cf_obs.n_obs else np.linspace(0, 1, 4),
        obs_x_qw=qw_obs.locations_x() if qw_obs.n_obs else np.linspace(0, 1, 4),
    )
    n_pred = (cp_obs.n_obs + cf_obs.n_obs + qw_obs.n_obs)
    obs_vals  = np.concatenate([
        cp_obs.values() if cp_obs.n_obs else np.array([]),
        cf_obs.values() if cf_obs.n_obs else np.array([]),
        qw_obs.values() if qw_obs.n_obs else np.array([]),
    ])
    obs_sigs  = np.concatenate([
        cp_obs.uncertainties() if cp_obs.n_obs else np.array([]),
        cf_obs.uncertainties() if cf_obs.n_obs else np.array([]),
        qw_obs.uncertainties() if qw_obs.n_obs else np.array([]),
    ])
    obs_locs  = np.concatenate([
        cp_obs.locations_x() if cp_obs.n_obs else np.array([]),
        cf_obs.locations_x() if cf_obs.n_obs else np.array([]),
        qw_obs.locations_x() if qw_obs.n_obs else np.array([]),
    ])

    # ------------------------------------------------------------------
    # 3. Standard Bayesian calibration
    # ------------------------------------------------------------------
    print("[3/7] Building ensemble ...")
    X_ens, Y_ens = _build_ensemble(fm, n_samples=n_ensemble)
    print(f"      {len(X_ens)} valid samples")

    prior = Prior(
        means=[0.31, 0.09],
        stds=[0.05, 0.015],
        lower=[0.20, 0.05],
        upper=[0.50, 0.15],
    )
    param_set = {
        "defaults": [0.31, 0.09],
        "lower":    [0.20, 0.05],
        "upper":    [0.50, 0.15],
        "names":    ["a1", "betaStar"],
    }

    from forward_model_interface import PrecomputedEnsembleForwardModel

    pre_fm = PrecomputedEnsembleForwardModel(
        X_ens, Y_ens, param_names=["a1", "betaStar"]
    )

    print("[4/7] KOH calibration — diagonal discrepancy ...")
    koh_diag = KOHLikelihood(obs_locs, obs_vals, obs_sigs, mode="diagonal")
    bi_diag  = BayesianInferenceKOH(
        forward_model=pre_fm,
        param_set=param_set,
        koh_likelihood=koh_diag,
        theta_prior=prior,
    )
    bi_diag.run_ensemble(n_samples=n_ensemble, verbose=False)
    bi_diag.train_surrogate(verbose=False)
    bi_diag.run_mcmc(n_walkers=n_walkers, n_steps=n_steps,
                     burn_in=burn_in, verbose=False)
    sum_diag = bi_diag.posterior_summary()
    (out_dir / "koh_diagonal_summary.json").write_text(
        json.dumps(sum_diag, indent=2))
    print(f"      a1={sum_diag['a1']['mean']:.4f}±{sum_diag['a1']['std']:.4f}")

    # Use diagonal posterior samples for plotting
    samples = bi_diag.samples

    print("[5/7] KOH calibration — physical-location GP discrepancy ...")
    koh_phys = KOHLikelihood(obs_locs, obs_vals, obs_sigs, mode="physical_gp")
    bi_phys  = BayesianInferenceKOH(
        forward_model=pre_fm,
        param_set=param_set,
        koh_likelihood=koh_phys,
        theta_prior=prior,
    )
    bi_phys.run_ensemble(n_samples=n_ensemble, verbose=False)
    bi_phys.train_surrogate(verbose=False)
    bi_phys.run_mcmc(n_walkers=n_walkers, n_steps=n_steps,
                     burn_in=burn_in, verbose=False)
    sum_phys = bi_phys.posterior_summary()
    (out_dir / "koh_physical_summary.json").write_text(
        json.dumps(sum_phys, indent=2))
    print(f"      a1={sum_phys['a1']['mean']:.4f}±{sum_phys['a1']['std']:.4f}")

    # ------------------------------------------------------------------
    # 6. KOH comparison summary
    # ------------------------------------------------------------------
    print("[6/7] Saving KOH comparison ...")
    koh_comparison = {
        "note": ("DRY RUN — synthetic data only. "
                 "Replace data file with real measurements for real calibration."),
        "diagonal": {
            "a1_mean":      sum_diag["a1"]["mean"],
            "a1_std":       sum_diag["a1"]["std"],
            "betaStar_mean": sum_diag["betaStar"]["mean"],
            "betaStar_std": sum_diag["betaStar"]["std"],
        },
        "physical_gp": {
            "a1_mean":      sum_phys["a1"]["mean"],
            "a1_std":       sum_phys["a1"]["std"],
            "betaStar_mean": sum_phys["betaStar"]["mean"],
            "betaStar_std": sum_phys["betaStar"]["std"],
        },
    }
    (out_dir / "koh_comparison.json").write_text(json.dumps(koh_comparison, indent=2))

    # Merge into a single posterior_summary too
    (out_dir / "posterior_summary.json").write_text(
        json.dumps({"diagonal": sum_diag, "physical_gp": sum_phys}, indent=2))

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    print("[7/7] Plotting ...")
    obs_x_all = obs_locs
    obs_v_all = obs_vals
    obs_s_all = obs_sigs
    obs_t_all = (
        ["cp"] * cp_obs.n_obs
        + ["cf"] * cf_obs.n_obs
        + ["qw"] * qw_obs.n_obs
    )
    theta_samples = bi_diag.samples[:, :2] if bi_diag.samples is not None else np.array([[0.31, 0.09]])
    _plot_posterior_predictive(
        obs_x_all, obs_v_all, obs_s_all, obs_t_all,
        fm, theta_samples, out_dir / "posterior_predictive.png",
    )

    # Residuals plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        mean_theta = np.array([sum_diag["a1"]["mean"], sum_diag["betaStar"]["mean"]])
        r = fm.evaluate(mean_theta)
        if r.converged and len(r.predictions) == len(obs_vals):
            resid = (np.array(r.predictions) - obs_vals) / obs_sigs
            ax.scatter(obs_locs, resid, c=[hash(t) % 10 for t in obs_t_all],
                       cmap="tab10", s=30, alpha=0.8)
            ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
            ax.set_xlabel("x / L")
            ax.set_ylabel("(pred − obs) / σ")
            ax.set_title("Residuals at posterior mean (DRY RUN — synthetic data)")
            fig.tight_layout()
            fig.savefig(out_dir / "residuals_by_observable.png", dpi=120)
            plt.close(fig)
            print(f"  Saved: residuals_by_observable.png")
    except ImportError:
        pass

    print(f"\nDry-run complete. Outputs in: {out_dir}")
    print("NOTE: ALL RESULTS ARE SYNTHETIC. This is a pipeline dry run,")
    print("      not a real high-Mach calibration result.")
    print("\nOutputs:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="High-Mach data pathway dry run (synthetic data)"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Fast mode: small ensemble and MCMC budget")
    parser.add_argument("-o", "--out-dir",
                        default="outputs/high_mach_data_pathway_dry_run",
                        help="Output directory")
    parser.add_argument("--data-file", default=None,
                        help="Path to wall data file (.csv/.json/.npz). "
                             "Defaults to the synthetic fixture in tests/fixtures/.")
    args = parser.parse_args()

    data_file = Path(args.data_file) if args.data_file else _FIXTURE
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            "Run from the repository root or pass --data-file explicitly."
        )

    run_demo(data_file=data_file, out_dir=Path(args.out_dir), quick=args.quick)


if __name__ == "__main__":
    main()
