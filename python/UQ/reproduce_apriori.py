"""End-to-end reproduce script for the a-priori UQ scaffolding (synthetic only).

Runs the full pipeline on synthetic data with fixed seeds and prints a
reproducible summary. No real DNS is used; this is the demonstration of the
machinery up to the point where DNS plugs in. It exercises, in order:

  1. discrepancy extraction through the DNS-schema interface (recovers a planted
     Reynolds-stress and heat-flux discrepancy);
  2. the generalized-Bayes coverage correction on a synthetic misspecified model
     (standard Bayes is overconfident; the tempered posterior is calibrated);
  3. conformal prediction (nominal coverage, and the honest coverage gap under a
     distribution shift);
  4. the generative conditional-flow model-form model (recovers a bimodal,
     heteroscedastic conditional; every projected sample is realizable), when
     PyTorch is available;
  5. the evaluation harness (CRPS, energy score, reliability, PIT, SBC).

Run:  PYTHONPATH=build:python python3 python/UQ/reproduce_apriori.py
"""
import numpy as np

from . import dns_field as dnsf
from . import generalized_bayes as gb
from . import conformal as cf
from . import evaluation as ev
from . import realizability as rz


def section(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "-" * 70)


def run_discrepancy_extraction():
    section("1. Discrepancy extraction through the DNS-schema interface")
    field, db_true, dq_true = dnsf.fake_dns_field(n=400, seed=0)
    out = field.extract()
    rms_R = np.sqrt(np.mean((out["reynolds_discrepancy"] - db_true) ** 2))
    rms_q = np.sqrt(np.mean((out["heatflux_discrepancy"] - dq_true) ** 2))
    print(f"  features (Pope invariants) shape : {out['features'].shape}")
    print(f"  planted Reynolds discrepancy RMS recovery error : {rms_R:.2e}")
    print(f"  planted heat-flux discrepancy RMS recovery error: {rms_q:.2e}")


def run_generalized_bayes():
    section("2. Generalized-Bayes coverage correction (misspecified model)")
    sigma0, sigma_true, n = 1.0, 2.0, 40
    theta_true = 0.0
    rng = np.random.default_rng(0)
    datasets = [theta_true + sigma_true * rng.normal(size=n) for _ in range(800)]
    cov_std = gb.credible_interval_coverage(datasets, theta_true, sigma0, eta=1.0)
    etas = [gb.calibrate_eta_moment(d, sigma0) for d in datasets]
    cov_temp = np.mean([
        gb.credible_interval(*gb.power_posterior_gaussian(d, sigma0, eta=e))[0]
        <= theta_true <=
        gb.credible_interval(*gb.power_posterior_gaussian(d, sigma0, eta=e))[1]
        for d, e in zip(datasets, etas)])
    print(f"  nominal coverage target           : 0.90")
    print(f"  standard Bayes (eta=1) coverage   : {cov_std:.3f}  (overconfident)")
    print(f"  mean calibrated learning rate eta : {np.mean(etas):.3f} "
          f"(target {(sigma0/sigma_true)**2:.3f})")
    print(f"  tempered-posterior coverage       : {cov_temp:.3f}  (restored)")


def run_conformal():
    section("3. Conformal prediction (nominal coverage and the shift gap)")
    rng = np.random.default_rng(0)
    n = 4000
    xc = rng.uniform(-2, 2, n); yc = np.sin(xc) + 0.3 * rng.normal(size=n)
    xt = rng.uniform(-2, 2, n); yt = np.sin(xt) + 0.3 * rng.normal(size=n)
    lo, hi = cf.split_conformal_intervals(np.sin(xc), yc, np.sin(xt), alpha=0.1)
    cov_in = float(np.mean((yt >= lo) & (yt <= hi)))
    yshift = np.sin(xt) + 1.2 * rng.normal(size=n)       # larger noise, unseen
    cov_sh, gap = cf.conformal_coverage_gap(np.sin(xc), yc, np.sin(xt), yshift, alpha=0.1)
    print(f"  in-distribution coverage (target 0.90) : {cov_in:.3f}")
    print(f"  shifted-distribution coverage          : {cov_sh:.3f}")
    print(f"  honest coverage gap under shift        : {gap:+.3f}")


def run_generative():
    section("4. Generative conditional-flow model-form model")
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        print("  PyTorch not available; skipping the generative demonstration.")
        return
    from . import synthetic as syn
    from .generative import GenerativeDiscrepancyModel
    x, y = syn.conditional_discrepancy_dataset(n=3000, seed=2)
    model = GenerativeDiscrepancyModel(1, 2, n_layers=6, hidden=48, seed=0)
    model.fit(x, y, epochs=180, lr=2e-3, batch=512)
    gen = model.sample(np.full((3000, 1), 1.5), 1)[:, 0, :]
    true = syn.true_conditional_samples(1.5, 3000)
    mid = np.sin(1.5 * 1.5)
    up = np.mean(gen[:, 0] > mid + 0.5)
    print(f"  conditional p(y|x=1.5): gen mean/std {gen[:,0].mean():.2f}/{gen[:,0].std():.2f}"
          f"  true {true[:,0].mean():.2f}/{true[:,0].std():.2f}")
    print(f"  bimodality recovered (upper-mode fraction ~0.5) : {up:.2f}")
    # realizability projection: a separate 5-component anisotropy model whose
    # raw draws can leave the realizable set but whose projected draws cannot.
    rng = np.random.default_rng(3)
    feat5 = rng.uniform(-1, 1, size=(800, 2))
    comp5 = np.concatenate([0.5 * np.tanh(2 * feat5), 0.35 * feat5,
                            0.05 * rng.normal(size=(800, 1))], axis=1)
    amodel = GenerativeDiscrepancyModel(2, 5, n_layers=6, hidden=48, seed=1)
    amodel.fit(feat5, comp5, epochs=80, lr=2e-3, batch=512)
    raw = amodel.components_to_anisotropy(amodel.sample(feat5, 1)[:, 0, :])
    R_raw = 2.0 * (raw + np.eye(3) / 3.0)
    bp = amodel.sample_realizable_anisotropy(feat5, n_per=1)[:, 0, :, :]
    R_proj = 2.0 * (bp + np.eye(3) / 3.0)
    print(f"  fraction realizable  raw / projected            : "
          f"{np.mean(rz.is_realizable(R_raw)):.3f} / "
          f"{np.mean(rz.is_realizable(R_proj, tol=1e-6)):.3f}")


def run_evaluation():
    section("5. Evaluation harness (scoring rules and calibration)")
    rng = np.random.default_rng(0)
    n, m = 3000, 300
    y = rng.normal(size=n)
    cal = rng.normal(size=(n, m))
    mis = 0.3 * rng.normal(size=(n, m))
    print(f"  CRPS (calibrated ensemble)        : {ev.crps_ensemble(y, cal):.3f}")
    print(f"  reliability error  calibrated     : {ev.reliability_error(y, cal):.3f}")
    print(f"  reliability error  overconfident  : {ev.reliability_error(y, mis):.3f}")
    print(f"  PIT uniformity p  calibrated      : {ev.pit_uniformity_pvalue(ev.pit_values(y, cal)):.3f}")
    print(f"  PIT uniformity p  overconfident   : {ev.pit_uniformity_pvalue(ev.pit_values(y, mis)):.1e}")


def main():
    print("A-priori UQ scaffolding: end-to-end reproduce (synthetic data, no DNS)")
    run_discrepancy_extraction()
    run_generalized_bayes()
    run_conformal()
    run_generative()
    run_evaluation()
    print("\nDone. Every number above is from a fixed-seed synthetic run.")


if __name__ == "__main__":
    main()
