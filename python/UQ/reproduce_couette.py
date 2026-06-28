"""reproduce_couette.py - the Couette cross-flow generalization finding (Step 2).

End-to-end reproduce script for DNS_plan.md Step 2. It calibrates the SST closure
on the real Lee-Moser channel DNS (Step 1, reused), pushes the channel posterior
through the a-posteriori moving-wall Couette forward model, and asks whether the
calibrated UQ covers the held-out Couette DNS at the strict 0.5 percent observation
band (with a 1 percent sensitivity band), in distribution and across the cross-flow
shift.

Stages (each writes to results/couette/, all gitignored and regenerable):
  1. channel ensembles  - per-Re channel calibration ensembles (cached)
  2. couette match+ens  - match Re_tau by nu, then per-Re Couette ensembles (cached)
  3. cross-flow         - channel posterior -> Couette coverage at 0.5% and 1%
  4. within-Couette     - in-distribution and cross-Re coverage on Couette (bonus)

Reproducibility: fixed seeds, parameters in CONFIG below, ensembles cached. The
expensive step is the forward-solve ensembles; the coverage scoring runs from the
cache in seconds. Nothing here tunes the learning rate, the conformal score, or the
model toward the pre-registered criterion (the correction is calibrated on the
channel and applied to Couette).

Usage:
  PYTHONPATH=build:python python3 python/UQ/reproduce_couette.py
  ... --regen-ensembles      # re-run the forward-solve ensembles
  ... --quick                # tiny ensembles / subset for a smoke run
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "build"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from UQ.datasets import (ChannelDNS, CouetteDNS, PipeDNS, RotatingChannelDNS,
                         PIPE_CASES, ROTATING_CASES)
from UQ.datasets.channel_calibration import ChannelCalibration, CrossReStudy
from UQ.datasets.couette_forward import CouetteForwardRANS, CouetteCalibration, COUETTE_CFG
from UQ.datasets.couette_crossflow import CrossFlowStudy
from UQ.datasets import crossflow_companions as companions

CONFIG = {
    "channel_cases": [550, 1000, 2000, 5200],
    "couette_cases": [171, 260, 507, 986],
    "n_ensemble": 40,
    "n_stations": 18,
    "level": 0.9,
    "bands": [0.005, 0.010],
    "seed": 0,
    "param_set": "a1_betaStar",
    # channel forward-model resolution (matches Step 1's calibration config)
    "channel_cfg": {"nx": 40, "ny": 56, "Lx": 18.0,
                    "max_iter": 4000, "conv_tol": 1.0e-3, "yplus_target": 0.5},
    "couette_cfg": dict(COUETTE_CFG),
}
OUT = os.path.join(_HERE, "..", "..", "results", "couette")


def _quick(cfg):
    cfg["channel_cases"] = [1000, 2000]
    cfg["couette_cases"] = [171, 507]
    cfg["n_ensemble"] = 8
    cfg["n_stations"] = 10
    cfg["bands"] = [0.005]
    cfg["couette_cfg"] = dict(cfg["couette_cfg"], ny=64, max_iter=8000)
    return cfg


# ---- stage 1: channel ensembles --------------------------------------------

def stage_channel(regen):
    cals = {}
    for n in CONFIG["channel_cases"]:
        dns = ChannelDNS.load(n)
        c = ChannelCalibration(dns, param_set=CONFIG["param_set"],
                               n_stations=CONFIG["n_stations"],
                               cfg=CONFIG["channel_cfg"], sigma_floor=0.005)
        path = os.path.join(OUT, f"channel_ensemble_{n}.npz")
        if os.path.exists(path) and not regen:
            c.load_cache(dict(np.load(path)))
            print(f"  channel Re_tau {n:>4}: loaded {c.n_valid} ensemble points")
        else:
            print(f"  channel Re_tau {n:>4}: running {CONFIG['n_ensemble']} solves ...",
                  flush=True)
            c.run_ensemble(n=CONFIG["n_ensemble"], seed=CONFIG["seed"])
            np.savez(path, **c.to_cache())
            c.fit_surrogates()
            print(f"           {c.n_valid}/{CONFIG['n_ensemble']} valid")
        cals[n] = c
    return cals


# ---- stage 2: Couette match + ensembles ------------------------------------

def stage_couette(regen):
    cals = {}
    nus = {}
    nu_path = os.path.join(OUT, "couette_matched_nu.json")
    if os.path.exists(nu_path) and not regen:
        nus = {int(k): v for k, v in json.load(open(nu_path)).items()}
    for n in CONFIG["couette_cases"]:
        dns = CouetteDNS.load(n)
        if n not in nus:
            print(f"  couette Re_tau {n:>4}: matching nu ...", flush=True)
            prof = CouetteForwardRANS.match(dns.re_tau, cfg=CONFIG["couette_cfg"])
            nus[n] = prof["matched_nu"]
            print(f"           matched Re_tau {prof['re_tau']:.0f} at nu=1/"
                  f"{1.0/prof['matched_nu']:.0f}")
        c = CouetteCalibration(dns, nu=nus[n], param_set=CONFIG["param_set"],
                               n_stations=CONFIG["n_stations"],
                               cfg=CONFIG["couette_cfg"])
        path = os.path.join(OUT, f"couette_ensemble_{n}.npz")
        if os.path.exists(path) and not regen:
            c.load_cache(dict(np.load(path)))
            print(f"  couette Re_tau {n:>4}: loaded {c.n_valid} ensemble points")
        else:
            print(f"  couette Re_tau {n:>4}: running {CONFIG['n_ensemble']} solves ...",
                  flush=True)
            c.run_ensemble(n=CONFIG["n_ensemble"], seed=CONFIG["seed"])
            np.savez(path, **c.to_cache())
            c.fit_surrogates()
            print(f"           {c.n_valid}/{CONFIG['n_ensemble']} valid")
        cals[n] = c
    json.dump({str(k): v for k, v in nus.items()}, open(nu_path, "w"))
    return cals


# ---- stage 3: cross-flow coverage ------------------------------------------

def stage_crossflow(channel_cals, couette_cals):
    study = CrossFlowStudy(channel_cals, couette_cals)
    res = study.run(rels=tuple(CONFIG["bands"]), level=CONFIG["level"],
                    seed=CONFIG["seed"])
    print(f"  channel-calibrated learning rate eta = {res['eta']:.4f}")
    for rel, rows in res["bands"].items():
        print(f"  band {100*rel:.1f}% observation:")
        for r in rows:
            print(f"    Couette Re_tau {r['re']:>4}: standard={r['standard_coverage']:.3f}"
                  f"  genBayes={r['tempered_coverage']:.3f}"
                  f"  conformal={r['conformal_coverage']:.3f}"
                  f" (gap {r['conformal_gap']:+.3f})  [nominal {CONFIG['level']:.2f}]")
    return res


# ---- stage 4: within-Couette (in-distribution + cross-Re bonus) ------------

def stage_within_couette(couette_cals):
    if len(couette_cals) < 2:
        return {"note": "need >= 2 Couette cases for the cross-Re axis"}
    study = CrossReStudy(couette_cals)
    seed = CONFIG["seed"]
    level = CONFIG["level"]
    # in-distribution: calibrate on Couette, predict the same Couette case
    indist = []
    rng = np.random.default_rng(seed)
    for n, c in couette_cals.items():
        post1 = c.sample_posterior(eta=1.0, seed=seed)
        idx = np.arange(c.n_qoi)
        cal_idx = np.sort(rng.choice(idx, size=c.n_qoi // 2, replace=False))
        eta = c.calibrate_eta(post1, cal_idx)
        post_t = c.sample_posterior(eta=eta, seed=seed)
        cov1, _ = c.coverage_vs_truth(c.posterior_predictive(post1, eta=1.0, seed=seed + 1),
                                      level=level)
        cov_t, _ = c.coverage_vs_truth(c.posterior_predictive(post_t, eta=eta, seed=seed + 2),
                                       level=level)
        indist.append({"re": n, "eta": eta, "standard_coverage": cov1,
                       "tempered_coverage": cov_t})
        print(f"    within-Couette Re_tau {n:>4}: standard={cov1:.3f} genBayes={cov_t:.3f}")
    # cross-Re within Couette: leave-one-Re-out
    loro = []
    cases = list(couette_cals)
    for test in cases:
        train = tuple(r for r in cases if r != test)
        r = study.predict_heldout(train, test, level=level, seed=seed)
        loro.append(r)
        print(f"    held-out Couette Re_tau {test:>4}: standard={r['standard_coverage']:.3f}"
              f" genBayes={r['tempered_coverage']:.3f} conformal={r['conformal_coverage']:.3f}"
              f" (gap {r['conformal_gap']:+.3f})")
    return {"in_distribution": indist, "cross_re_loro": loro}


# ---- stage 5: a-priori companions (pipe, rotating channel) -----------------

def stage_companions():
    pipe = []
    for n in PIPE_CASES:
        if PipeDNS.is_available(n):
            pipe.append(companions.pipe_discrepancy(PipeDNS.load(n)))
            print(f"    pipe Re_tau {pipe[-1]['re_tau']:>5.0f}: peak|db|="
                  f"{pipe[-1]['db_norm_max']:.3f}  normal_dom="
                  f"{pipe[-1]['normal_dominates_discrepancy']}")
    rot = []
    for ro in ROTATING_CASES:
        if RotatingChannelDNS.is_available(ro):
            r = companions.rotating_diagnostic(RotatingChannelDNS.load(ro))
            rot.append(r)
            print(f"    rotating Ro_tau {ro:>5g}: b13_bouss={r['b13_boussinesq_max']:.1e} "
                  f"(=0, structural)  diffuse localisation cost={r['localisation_cost']:.2f}")
    return {"pipe": pipe, "rotating": rot}


# ---- figures ---------------------------------------------------------------

def make_figures(crossflow, within, comps):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [figures] matplotlib unavailable, skipping ({exc})")
        return
    fig_dir = os.path.join(OUT, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # cross-flow coverage: overconfidence and the correction, per band
    bands = list(crossflow["bands"])
    fig, axes = plt.subplots(1, len(bands), figsize=(5.5 * len(bands), 4.2),
                             squeeze=False)
    for ax, rel in zip(axes[0], bands):
        rows = crossflow["bands"][rel]
        re = [r["re"] for r in rows]
        x = np.arange(len(re))
        ax.bar(x - 0.25, [r["standard_coverage"] for r in rows], 0.25, label="standard")
        ax.bar(x, [r["tempered_coverage"] for r in rows], 0.25, label="generalized Bayes")
        ax.bar(x + 0.25, [r["conformal_coverage"] for r in rows], 0.25, label="conformal")
        ax.axhline(CONFIG["level"], color="k", ls="--", lw=1, label="nominal")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in re])
        ax.set_xlabel("Couette Re_tau")
        ax.set_ylabel("empirical coverage at 90%")
        ax.set_title(f"channel -> Couette, {100*rel:.1f}% band")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "crossflow_coverage.png"), dpi=140)
    plt.close(fig)

    # rotating channel: per-component discrepancy band (the diffuse-inflation cost)
    if comps["rotating"]:
        r = comps["rotating"][1 if len(comps["rotating"]) > 1 else 0]
        names = list(r["band_per_component"])
        vals = [r["band_per_component"][n] for n in names]
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["steelblue"] * len(names)
        colors[names.index("uw")] = "crimson"
        ax.bar(names, vals, color=colors)
        ax.axhline(r["global_band"] if "global_band" in r else max(vals),
                   color="k", ls="--", lw=1, label="global band (covers the worst)")
        ax.set_ylabel("per-component band needed (90%)")
        ax.set_title(f"Rotating channel Ro_tau={r['ro_tau']:g}: uneven discrepancy\n"
                     f"(global band over-inflates the median by "
                     f"{r['localisation_cost']:.1f}x; <uw> is structural)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "rotating_diffuse_inflation.png"), dpi=140)
        plt.close(fig)
    print(f"  [figures] wrote crossflow_coverage, rotating_diffuse_inflation to {fig_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen-ensembles", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        _quick(CONFIG)

    os.makedirs(OUT, exist_ok=True)
    print("== stage 1: channel ensembles ==")
    channel_cals = stage_channel(args.regen_ensembles)
    print("== stage 2: Couette match + ensembles ==")
    couette_cals = stage_couette(args.regen_ensembles)
    print("== stage 3: cross-flow coverage (channel -> Couette) ==")
    crossflow = stage_crossflow(channel_cals, couette_cals)
    print("== stage 4: within-Couette coverage (bonus cross-Re) ==")
    within = stage_within_couette(couette_cals)
    print("== stage 5: a-priori companions (pipe, rotating) ==")
    comps = stage_companions()
    print("== stage 6: figures ==")
    make_figures(crossflow, within, comps)

    numbers = {"config": {k: v for k, v in CONFIG.items() if k != "channel_cfg"},
               "cross_flow": crossflow, "within_couette": within,
               "companions": comps}
    out_json = os.path.join(OUT, "finding_numbers.json")
    with open(out_json, "w") as f:
        json.dump(numbers, f, indent=2, default=float)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
