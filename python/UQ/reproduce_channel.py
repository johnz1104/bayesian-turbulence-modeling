"""reproduce_channel.py - the plane-channel real-data coverage-correction finding.

End-to-end reproduce script for DNS_plan.md Step 1. It builds the whole real-data
UQ pipeline on the Lee-Moser channel DNS and produces the evidence package for the
finding: does standard Bayesian calibration on real channel DNS produce
overconfident predictions, and do generalized Bayes and conformal restore reliable
coverage, in distribution and across held-out Reynolds numbers.

Stages (each writes to results/channel/, all gitignored and regenerable):
  1. baselines     - matched-Re_tau SST baseline fields (cached; regenerate with --regen-baselines)
  2. discrepancy   - real-data Boussinesq anisotropy discrepancy diagnostics
  3. ensembles     - per-Re calibration ensembles of forward solves (cached; --regen-ensembles)
  4. in-distribution - standard-Bayes overconfidence vs generalized-Bayes / conformal coverage
  5. cross-Re      - leave-one-Reynolds-out and held-out-high-Re coverage
  6. evaluation    - reliability, PIT, CRPS via the UQ evaluation harness
  7. figures+memo  - reliability diagrams, PIT histograms, and the numbers JSON

Reproducibility: fixed seeds, parameters in CONFIG below, ensembles cached. The
expensive step is stage 3 (a few hundred SIMPLE solves); stages 4-7 run from the
cache in seconds. Nothing here tunes the learning rate, the conformal score, or
the model toward the pre-registered criterion.

Usage:
  PYTHONPATH=build:python python3 python/UQ/reproduce_channel.py
  ... --regen-ensembles      # re-run the forward-solve ensembles
  ... --quick                # small ensembles for a smoke run
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "build"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from UQ.datasets import ChannelDNS, CHANNEL_CASES
from UQ.datasets.channel_baseline import ChannelBaselineRANS
from UQ.datasets import channel_discrepancy as cdisc
from UQ.datasets.channel_calibration import ChannelCalibration, CrossReStudy
from UQ import evaluation as ev

CONFIG = {
    "cases": list(CHANNEL_CASES),
    "n_stations": 20,
    "n_ensemble": 48,
    "sigma_floor": 0.005,
    "level": 0.9,
    "seed": 0,
    # forward-model resolution for the calibration ensembles (faster than the
    # baseline config; the developing channel is the same model either way)
    "cfg": {"nx": 40, "ny": 56, "Lx": 18.0,
            "max_iter": 4000, "conv_tol": 1.0e-3, "yplus_target": 0.5},
    # baseline-field resolution (only used when regenerating baselines)
    "baseline_cfg": dict(ChannelBaselineRANS.DEFAULT_CONFIG),
    "param_set": "a1_betaStar",
}
OUT = os.path.join(_HERE, "..", "..", "results", "channel")


# ---- stage 1: baselines ----------------------------------------------------

def stage_baselines(regen):
    path = os.path.join(OUT, "baselines.npz")
    if os.path.exists(path) and not regen:
        return np.load(path)
    print("[baselines] generating matched-Re_tau SST baseline fields ...")
    store = {}
    for n in CONFIG["cases"]:
        dns = ChannelDNS.load(n)
        b = ChannelBaselineRANS.match(dns.re_tau, tol=0.04, max_solves=7)
        store[f"yplus_{n}"] = b.yplus
        store[f"omega_{n}"] = b.omega
        store[f"k_{n}"] = b.k
        store[f"nu_t_{n}"] = b.nu_t
        store[f"U_{n}"] = b.U
        store[f"scalars_{n}"] = np.array([b.nu, b.u_tau, b.re_tau, b.cf, b.cf_dean])
        print(f"  Re_tau {dns.re_tau:.0f} -> {b.re_tau:.0f}  Cf {b.cf:.5f} "
              f"(Dean {b.cf_dean:.5f}, {100*(b.cf-b.cf_dean)/b.cf_dean:+.0f}%)")
    np.savez(path, **store)
    return np.load(path)


# ---- stage 2: discrepancy --------------------------------------------------

def stage_discrepancy(baselines):
    rows = []
    for n in CONFIG["cases"]:
        dns = ChannelDNS.load(n)
        om = np.interp(dns.yplus, baselines[f"yplus_{n}"], baselines[f"omega_{n}"],
                       left=baselines[f"omega_{n}"][0], right=baselines[f"omega_{n}"][-1])
        tau = 1.0 / (0.09 * np.maximum(om, 1e-30))
        d = cdisc.diagnostics(cdisc.channel_discrepancy(dns, tau))
        d["re_tau"] = dns.re_tau
        rows.append(d)
        print(f"  Re_tau {n:>4}: uv_dom={d['uv_dominates_offdiagonal']} "
              f"normal_dom={d['normal_dominates_discrepancy']} "
              f"grows={d['discrepancy_grows_toward_wall']} "
              f"peak|db|={d['db_norm_max']:.3f}@y+={d['yplus_at_db_norm_max']:.1f}")
    return rows


# ---- stage 3: calibration ensembles (expensive, cached) --------------------

def stage_ensembles(regen, quick):
    n_ens = 12 if quick else CONFIG["n_ensemble"]
    cals = {}
    for n in CONFIG["cases"]:
        dns = ChannelDNS.load(n)
        c = ChannelCalibration(dns, param_set=CONFIG["param_set"],
                               n_stations=CONFIG["n_stations"], cfg=CONFIG["cfg"],
                               sigma_floor=CONFIG["sigma_floor"])
        path = os.path.join(OUT, f"ensemble_{n}.npz")
        if os.path.exists(path) and not regen:
            c.load_cache(dict(np.load(path)))
            print(f"  Re_tau {n:>4}: loaded {c.n_valid} ensemble points")
        else:
            print(f"  Re_tau {n:>4}: running {n_ens} forward solves ...", flush=True)
            c.run_ensemble(n=n_ens, seed=CONFIG["seed"])
            np.savez(path, **c.to_cache())
            c.fit_surrogates()
            print(f"           {c.n_valid}/{n_ens} valid")
        cals[n] = c
    return cals


# ---- stage 4: in-distribution overconfidence and correction ----------------

def stage_in_distribution(cals):
    level = CONFIG["level"]
    seed = CONFIG["seed"]
    rng = np.random.default_rng(seed)
    rows = []
    for n in CONFIG["cases"]:
        c = cals[n]
        # standard Bayes (eta = 1): posterior predictive coverage of the DNS QoIs
        post1 = c.sample_posterior(eta=1.0, seed=seed)
        pp1 = c.posterior_predictive(post1, eta=1.0, seed=seed + 1)
        cov1, sharp1 = c.coverage_vs_truth(pp1, level=level)

        # generalized Bayes: learning rate moment-matched on a calibration split
        idx = np.arange(c.n_qoi)
        cal_idx = np.sort(rng.choice(idx, size=c.n_qoi // 2, replace=False))
        test_idx = np.array([i for i in idx if i not in set(cal_idx)])
        eta = c.calibrate_eta(post1, cal_idx)
        post_t = c.sample_posterior(eta=eta, seed=seed)
        pp_t = c.posterior_predictive(post_t, eta=eta, seed=seed + 2)
        cov_t, sharp_t = c.coverage_vs_truth(pp_t, level=level)

        # conformal: width from the calibration-split residuals, coverage on test
        pred_cal = c.point_prediction(post1, cal_idx)
        pred_test = c.point_prediction(post1, test_idx)
        cov_c, half_c, gap_c = c.conformal_coverage(pred_cal, pred_test,
                                                    cal_idx, test_idx,
                                                    alpha=1.0 - level)
        rows.append({
            "re_tau": c.dns.re_tau, "eta": eta,
            "standard_coverage": cov1, "standard_sharpness": sharp1,
            "tempered_coverage": cov_t, "tempered_sharpness": sharp_t,
            "conformal_coverage": cov_c, "conformal_halfwidth": half_c,
        })
        print(f"  Re_tau {n:>4}: standard cov={cov1:.3f}  "
              f"tempered(eta={eta:.3f}) cov={cov_t:.3f}  conformal cov={cov_c:.3f}  "
              f"[nominal {level:.2f}]")
    return rows


# ---- stage 5: cross-Re generalization --------------------------------------

def stage_cross_re(cals):
    study = CrossReStudy(cals)
    cases = CONFIG["cases"]
    level = CONFIG["level"]
    seed = CONFIG["seed"]
    results = {"loro": [], "high_re_holdout": None}

    # leave-one-Reynolds-out
    print("  leave-one-Reynolds-out:")
    for test in cases:
        train = tuple(r for r in cases if r != test)
        eta = _pooled_eta(study, train, test, seed)
        r = study.predict_heldout(train, test, eta=eta, level=level, seed=seed)
        results["loro"].append(r)
        print(f"    held-out Re_tau {test:>4}: standard cov={r['standard_coverage']:.3f} "
              f"tempered cov={r['tempered_coverage']:.3f} "
              f"conformal cov={r['conformal_coverage']:.3f} (gap {r['conformal_gap']:+.3f})")

    # held-out-high-Re split: train low, predict high
    train = (180, 550, 1000)
    print(f"  high-Re holdout (train {train}):")
    hi = []
    for test in (2000, 5200):
        eta = _pooled_eta(study, train, test, seed)
        r = study.predict_heldout(train, test, eta=eta, level=level, seed=seed)
        hi.append(r)
        print(f"    held-out Re_tau {test:>4}: standard cov={r['standard_coverage']:.3f} "
              f"tempered cov={r['tempered_coverage']:.3f} "
              f"conformal cov={r['conformal_coverage']:.3f} (gap {r['conformal_gap']:+.3f})")
    results["high_re_holdout"] = {"train": list(train), "results": hi}
    return results


def _pooled_eta(study, train, test, seed):
    """Moment-match the pooled-posterior learning rate on the training Re only."""
    post = study.pooled_posterior_samples(train, eta=1.0, seed=seed)
    etas = [study.cals[r].calibrate_eta(post, np.arange(study.cals[r].n_qoi))
            for r in train]
    return float(np.mean(etas))


# ---- stage 6: evaluation harness (reliability, PIT, CRPS) -------------------

def stage_evaluation(cals):
    seed = CONFIG["seed"]
    out = {}
    for n in CONFIG["cases"]:
        c = cals[n]
        post1 = c.sample_posterior(eta=1.0, seed=seed)
        eta = c.calibrate_eta(post1, np.arange(c.n_qoi))
        post_t = c.sample_posterior(eta=eta, seed=seed)
        pp1 = c.posterior_predictive(post1, eta=1.0, seed=seed + 1)
        pp_t = c.posterior_predictive(post_t, eta=eta, seed=seed + 2)
        truth = c.qoi_truth
        nominal, emp1 = ev.reliability_curve(truth, pp1.T)
        _, emp_t = ev.reliability_curve(truth, pp_t.T)
        out[n] = {
            "nominal": nominal.tolist(),
            "reliability_standard": emp1.tolist(),
            "reliability_tempered": emp_t.tolist(),
            "reliability_error_standard": ev.reliability_error(truth, pp1.T),
            "reliability_error_tempered": ev.reliability_error(truth, pp_t.T),
            "pit_standard": ev.pit_values(truth, pp1.T).tolist(),
            "pit_tempered": ev.pit_values(truth, pp_t.T).tolist(),
            "crps_standard": ev.crps_ensemble(truth, pp1.T),
            "crps_tempered": ev.crps_ensemble(truth, pp_t.T),
        }
    return out


# ---- figures and memo numbers ----------------------------------------------

def make_figures(indist, evaluation):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [figures] matplotlib unavailable, skipping ({exc})")
        return
    fig_dir = os.path.join(OUT, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # reliability diagram (aggregated over Re, standard vs tempered)
    fig, ax = plt.subplots(figsize=(5, 5))
    nominal = np.array(evaluation[CONFIG["cases"][0]]["nominal"])
    rs = np.mean([evaluation[n]["reliability_standard"] for n in CONFIG["cases"]], axis=0)
    rt = np.mean([evaluation[n]["reliability_tempered"] for n in CONFIG["cases"]], axis=0)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(nominal, rs, "o-", label="standard Bayes")
    ax.plot(nominal, rt, "s-", label="generalized Bayes")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Channel QoI reliability (mean over Re)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "reliability.png"), dpi=140)
    plt.close(fig)

    # PIT histograms (standard vs tempered, pooled)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    pit_s = np.concatenate([evaluation[n]["pit_standard"] for n in CONFIG["cases"]])
    pit_t = np.concatenate([evaluation[n]["pit_tempered"] for n in CONFIG["cases"]])
    for ax, pit, title in ((axes[0], pit_s, "standard Bayes"),
                           (axes[1], pit_t, "generalized Bayes")):
        ax.hist(pit, bins=10, range=(0, 1), color="steelblue", edgecolor="k")
        ax.axhline(len(pit) / 10.0, color="r", ls="--", lw=1)
        ax.set_title(f"PIT: {title}")
        ax.set_xlabel("PIT value")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pit_histograms.png"), dpi=140)
    plt.close(fig)

    # coverage bar chart (in-distribution, per Re)
    fig, ax = plt.subplots(figsize=(7, 4))
    re = [r["re_tau"] for r in indist]
    x = np.arange(len(re))
    ax.bar(x - 0.25, [r["standard_coverage"] for r in indist], 0.25, label="standard")
    ax.bar(x, [r["tempered_coverage"] for r in indist], 0.25, label="generalized Bayes")
    ax.bar(x + 0.25, [r["conformal_coverage"] for r in indist], 0.25, label="conformal")
    ax.axhline(CONFIG["level"], color="k", ls="--", lw=1, label="nominal")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r)}" for r in re])
    ax.set_xlabel("Re_tau")
    ax.set_ylabel("empirical coverage at 90%")
    ax.set_title("In-distribution coverage: overconfidence and correction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "coverage_in_distribution.png"), dpi=140)
    plt.close(fig)
    print(f"  [figures] wrote reliability, PIT, coverage to {fig_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen-baselines", action="store_true")
    ap.add_argument("--regen-ensembles", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("== stage 1: baselines ==")
    baselines = stage_baselines(args.regen_baselines)
    print("== stage 2: discrepancy ==")
    disc = stage_discrepancy(baselines)
    print("== stage 3: calibration ensembles ==")
    cals = stage_ensembles(args.regen_ensembles, args.quick)
    print("== stage 4: in-distribution coverage ==")
    indist = stage_in_distribution(cals)
    print("== stage 5: cross-Re generalization ==")
    cross = stage_cross_re(cals)
    print("== stage 6: evaluation harness ==")
    evaluation = stage_evaluation(cals)
    print("== stage 7: figures and memo numbers ==")
    make_figures(indist, evaluation)

    numbers = {
        "config": CONFIG,
        "discrepancy": disc,
        "in_distribution": indist,
        "cross_re": cross,
        "evaluation": {str(n): {k: v for k, v in evaluation[n].items()
                                if not k.startswith("pit")} for n in CONFIG["cases"]},
    }
    with open(os.path.join(OUT, "finding_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=2, default=float)
    print(f"\nwrote {os.path.join(OUT, 'finding_numbers.json')}")


if __name__ == "__main__":
    main()
