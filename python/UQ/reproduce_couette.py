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
from UQ import cache_fingerprint as cfp
from UQ import conformal as cf

# Physics schema tokens (cache identity; bump exactly when the producing
# model changes, see UQ.cache_fingerprint): v3 marks the final integrated
# post-audit incompressible solver for both flow types. v2 predates integration
# of the wall-molecular diffusion and completed-stress branches.
PHYSICS_CHANNEL = "channel-rans-v3"
PHYSICS_COUETTE = "couette-rans-v3"

CONFIG = {
    "channel_cases": [550, 1000, 2000],
    "couette_cases": [171, 260, 507, 986],
    "n_ensemble": 24,
    "n_stations": 18,
    "level": 0.9,
    "bands": [0.005, 0.010],
    "seed": 0,
    "param_set": "a1_betaStar",
    # channel forward-model resolution (matches Step 1's calibration config)
    # sized for the honest criterion with cold members (see reproduce_channel)
    "channel_cfg": {"nx": 40, "ny": 56, "Lx": 18.0,
                    "max_iter": 20000, "conv_tol": 1.0e-3, "yplus_target": 0.5},
    "couette_cfg": dict(COUETTE_CFG),
}
OUT = os.path.join(_HERE, "..", "..", "results", "couette")


def _conformal_claim_tag(row):
    """Visible guard for diagnostics that cannot make a formal split claim."""
    if row.get("conformal_roles_disjoint", True):
        return ""
    return ("  [single-case fallback: roles NOT disjoint; "
            "excluded from formal conformal claims]")


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
        ident = {"kind": "couette_study_channel_ensemble",
                 "physics": PHYSICS_CHANNEL, "case": n,
                 "n_ensemble": CONFIG["n_ensemble"], "seed": CONFIG["seed"],
                 "param_set": CONFIG["param_set"],
                 "n_stations": CONFIG["n_stations"],
                 "cfg": CONFIG["channel_cfg"], "sigma_floor": 0.005}
        loaded = False
        if os.path.exists(path) and not regen:
            d = dict(np.load(path))
            status, reason = cfp.check(d, ident)
            if status == "mismatch" or (status == "legacy"
                                        and not cfp.legacy_reuse_allowed()):
                print(f"  channel Re_tau {n:>4}: cache REFUSED ({reason}); regenerating")
            else:
                if status == "legacy":
                    print(f"  channel Re_tau {n:>4}: WARNING reusing "
                          f"pre-fingerprint cache (QBTM_ALLOW_LEGACY_CACHE=1); "
                          f"regenerate to stamp it")
                elif reason:
                    print(f"  channel Re_tau {n:>4}: {reason}")
                loaded = c.load_cache(d)
                if loaded:
                    print(f"  channel Re_tau {n:>4}: loaded {c.n_valid} ensemble points")
        if not loaded:
            print(f"  channel Re_tau {n:>4}: running {CONFIG['n_ensemble']} solves ...",
                  flush=True)
            nv = c.run_ensemble(n=CONFIG["n_ensemble"], seed=CONFIG["seed"])
            if nv == 0:
                print(f"  channel Re_tau {n:>4}: EVERY member rejected; aborting")
                sys.exit(1)
            np.savez(path, **cfp.attach(c.to_cache(), ident))
            if not c.fit_surrogates():
                sys.exit(1)
            print(f"           {c.n_valid}/{CONFIG['n_ensemble']} valid")
        cals[n] = c
    return cals


# ---- stage 2: Couette match + ensembles ------------------------------------

def stage_couette(regen):
    cals = {}
    nus = {}
    nu_path = os.path.join(OUT, "couette_matched_nu.json")
    nu_ident = {"kind": "couette_matched_nu", "physics": PHYSICS_COUETTE,
                "cases": CONFIG["couette_cases"], "cfg": CONFIG["couette_cfg"]}
    if os.path.exists(nu_path) and not regen:
        raw = json.load(open(nu_path))
        if "values" in raw and raw.get("fingerprint") == cfp.fingerprint(nu_ident):
            nus = {int(k): v for k, v in raw["values"].items()}
        elif "values" not in raw and cfp.legacy_reuse_allowed():
            print("  couette matched-nu: WARNING reusing pre-fingerprint store "
                  "(QBTM_ALLOW_LEGACY_CACHE=1)")
            nus = {int(k): v for k, v in raw.items()}
        else:
            print("  couette matched-nu: store REFUSED (fingerprint absent or "
                  "stale); re-matching")
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
        ident = {"kind": "couette_ensemble", "physics": PHYSICS_COUETTE,
                 "case": n,
                 "n_ensemble": CONFIG["n_ensemble"], "seed": CONFIG["seed"],
                 "param_set": CONFIG["param_set"],
                 "n_stations": CONFIG["n_stations"],
                 "cfg": CONFIG["couette_cfg"], "matched_nu": nus[n]}
        loaded = False
        if os.path.exists(path) and not regen:
            d = dict(np.load(path))
            status, reason = cfp.check(d, ident)
            if status == "mismatch" or (status == "legacy"
                                        and not cfp.legacy_reuse_allowed()):
                print(f"  couette Re_tau {n:>4}: cache REFUSED ({reason}); regenerating")
            else:
                if status == "legacy":
                    print(f"  couette Re_tau {n:>4}: WARNING reusing "
                          f"pre-fingerprint cache (QBTM_ALLOW_LEGACY_CACHE=1); "
                          f"regenerate to stamp it")
                elif reason:
                    print(f"  couette Re_tau {n:>4}: {reason}")
                loaded = c.load_cache(d)
                if loaded:
                    print(f"  couette Re_tau {n:>4}: loaded {c.n_valid} ensemble points")
        if not loaded:
            print(f"  couette Re_tau {n:>4}: running {CONFIG['n_ensemble']} solves ...",
                  flush=True)
            nv = c.run_ensemble(n=CONFIG["n_ensemble"], seed=CONFIG["seed"])
            if nv == 0:
                print(f"  couette Re_tau {n:>4}: EVERY member rejected; aborting")
                sys.exit(1)
            np.savez(path, **cfp.attach(c.to_cache(), ident))
            if not c.fit_surrogates():
                sys.exit(1)
            print(f"           {c.n_valid}/{CONFIG['n_ensemble']} valid")
        cals[n] = c
    json.dump({"fingerprint": cfp.fingerprint(nu_ident),
               "config": cfp.config_json(nu_ident),
               "code_rev": cfp.code_rev(),
               "values": {str(k): v for k, v in nus.items()}},
              open(nu_path, "w"), indent=1)
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
                  f" (gap {r['conformal_gap']:+.3f})  [nominal {CONFIG['level']:.2f}]"
                  f"{_conformal_claim_tag(r)}")
    return res


# ---- stage 4: within-Couette (in-distribution + cross-Re bonus) ------------

def stage_within_couette(couette_cals):
    if len(couette_cals) < 2:
        return {"note": "need >= 2 Couette cases for the cross-Re axis"}
    study = CrossReStudy(couette_cals)
    seed = CONFIG["seed"]
    level = CONFIG["level"]
    # in-distribution with GENUINELY HELD-OUT stations (post-audit design,
    # mirroring reproduce_channel): the posterior fits Cf plus the odd
    # stations, the even stations split alternately into an eta-calibration
    # set and a test set that never entered any fit; coverage is evaluated on
    # the test stations only, and a pooled sigma-normalized split-conformal
    # line is computed on the never-fitted stations across cases
    indist = []
    pooled_cal, pooled_test = [], []
    for n, c in couette_cals.items():
        idx = np.arange(c.n_qoi)
        fit_idx = np.array([0] + [i for i in idx[1:] if (i - 1) % 2 == 0])
        held = [i for i in idx[1:] if (i - 1) % 2 == 1]
        cal_idx = np.array(held[0::2])
        test_idx = np.array(held[1::2])
        c.refit_likelihood_subset(fit_idx)

        post1 = c.sample_posterior(eta=1.0, seed=seed)
        eta = c.calibrate_eta(post1, cal_idx)
        post_t = c.sample_posterior(eta=eta, seed=seed)
        cov1, _ = c.coverage_vs_truth(
            c.posterior_predictive(post1, eta=1.0, qoi_index=test_idx,
                                   seed=seed + 1),
            qoi_index=test_idx, level=level)
        cov_t, _ = c.coverage_vs_truth(
            c.posterior_predictive(post_t, eta=eta, qoi_index=test_idx,
                                   seed=seed + 2),
            qoi_index=test_idx, level=level)

        pred_cal = c.point_prediction(post1, cal_idx)
        pred_test = c.point_prediction(post1, test_idx)
        pooled_cal.append(np.abs(c.qoi_truth[cal_idx] - pred_cal)
                          / c.qoi_sigma[cal_idx])
        pooled_test.append((n, np.abs(c.qoi_truth[test_idx] - pred_test)
                            / c.qoi_sigma[test_idx]))

        indist.append({"re": n, "eta": eta, "standard_coverage": cov1,
                       "tempered_coverage": cov_t,
                       "n_fit": int(fit_idx.size), "n_cal": int(cal_idx.size),
                       "n_test": int(test_idx.size)})
        c.refit_likelihood_subset(None)   # restore for the cross-Re axis below
        print(f"    within-Couette Re_tau {n:>4}: standard={cov1:.3f} genBayes={cov_t:.3f}")
    q = cf.conformal_quantile(np.concatenate(pooled_cal), alpha=1.0 - level)
    hits_all = []
    for row, (n, r_test) in zip(indist, pooled_test):
        hits = (r_test <= q)
        hits_all.append(hits)
        row["conformal_coverage"] = float(np.mean(hits))
        row["conformal_quantile_sigma_units"] = float(q)
    pooled_cov = float(np.mean(np.concatenate(hits_all)))
    print(f"    pooled held-out-station conformal coverage: {pooled_cov:.3f}")
    # cross-Re within Couette: leave-one-Re-out
    loro = []
    cases = list(couette_cals)
    for test in cases:
        train = tuple(r for r in cases if r != test)
        r = study.predict_heldout(train, test, level=level, seed=seed)
        loro.append(r)
        print(f"    held-out Couette Re_tau {test:>4}: standard={r['standard_coverage']:.3f}"
              f" genBayes={r['tempered_coverage']:.3f} conformal={r['conformal_coverage']:.3f}"
              f" (gap {r['conformal_gap']:+.3f}){_conformal_claim_tag(r)}")
    return {"in_distribution": indist,
            "pooled_conformal_coverage": pooled_cov,
            "cross_re_loro": loro}


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
