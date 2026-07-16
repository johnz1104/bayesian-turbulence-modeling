"""Reproduce the compressible attached-flow heat-flux and Prandtl UQ study.

Fixed-seed production driver for every number the evidence package quotes,
measured against the committed PRE_REGISTRATION.md (splits, sigma levels,
held-out blocks, null shapes fixed there BEFORE these runs). Usage:

    PYTHONPATH=build:python python3 python/UQ/reproduce_compressible.py
        [--quick] [--regen-ensembles] [--results DIR]

Stages (all seeds fixed; ensembles cached under the gitignored results dir):
  1. Ensembles and surrogates for every channel case (24 GV + 2 CKM).
  2. In-distribution coverage per GV case: standard Bayes (eta = 1) versus
     the tempered posterior (learning rate moment-matched on the LIKELIHOOD
     block only), scored separately on the likelihood block and the
     pre-registered HELD-OUT thermal block, at the primary (0.5 percent,
     anchor-floored) and sensitivity (1 percent) sigma levels.
  3. The primary cross-Mach split: calibrate pooled on the M_CLx < 1.0 cases,
     predict every M_CLx >= 1.47 case; per-block coverage, sharpness and the
     conformal gap; the pooled Pr_t marginal recorded.
  4. Leave-one-Re-family-out within the matrix (families at fixed nominal
     Re_tau*), same scoring.
  5. The CKM external check: never calibrated on; predicted from the primary
     pooled posterior.
  6. The flat-plate leg: the pooled Pr_t posterior drives the frozen-mean
     GDH heat-flux profile per plate case against the derived flux at its
     valid stations (the wall-cooling axis). The plate wall-flux clause is
     NOT evaluable through the approved frozen baseline (it predicts no
     B_q); recorded as a stated pre-registration scope note, not silently
     dropped.
  7. The Pr_t posterior against the five MEASURED plate profiles (log-region
     median and quartiles per case).

Every stage writes into one numbers JSON; nothing here is tuned toward any
coverage. A null or negative shape per the pre-registration is reported
exactly as it comes out.
"""
import argparse
import json
import os
import sys

import numpy as np

from UQ.datasets import (GVChannelDNS, GV_CASES, CKMChannelDNS, CKM_CASES,
                         FlatPlateDNS, ZDC_CASES)
from UQ.datasets.compressible_calibration import CompressibleCalibration
from UQ.datasets.crossmach_study import CrossMachStudy
from UQ.datasets.compressible_baseline import FlatPlateFrozenSST
from UQ import cache_fingerprint as cfp

# Physics schema token (cache identity; bump exactly when the producing model
# changes): the producing model here is the python 1-D compressible profile
# baseline, which the 2-D solver audit does not touch, so v1 remains valid
# for pre-audit ensembles once stamped.
PHYSICS = "compressible-profile-v1"


def _conformal_claim_tag(row):
    """Visible guard for diagnostics that cannot make a formal split claim."""
    if row.get("conformal_roles_disjoint", True):
        return ""
    return ("  [single-case fallback: roles NOT disjoint; "
            "excluded from formal conformal claims]")

SEED = 0
LEVEL = 0.9

# pre-registered splits (PRE_REGISTRATION.md): the low-Mach calibration set
# and the high-Mach prediction set, by case tag
LOW_MACH = tuple(c for c in GV_CASES
                 if GVChannelDNS.parse_tag(c)[1] < 1.0)
HIGH_MACH = tuple(c for c in GV_CASES
                  if GVChannelDNS.parse_tag(c)[1] >= 1.47)

# leave-one-Re-family-out: families at fixed nominal Re_tau* level
FAMILY_EDGES = ((0, 130), (130, 200), (200, 300), (300, 500), (500, 1100))


def family_of(case):
    re_nom = GVChannelDNS.parse_tag(case)[0]
    for i, (lo, hi) in enumerate(FAMILY_EDGES):
        if lo <= re_nom < hi:
            return i
    raise ValueError(case)


def build_calibrations(results_dir, n_ensemble, rel, regen, quick):
    """Stage 1: ensembles + surrogates for every channel case, cached."""
    cals = {}
    tags = list(GV_CASES) + list(CKM_CASES)
    if quick:
        tags = [GV_CASES[6], GV_CASES[12], GV_CASES[21], CKM_CASES[0]]
    for j, tag in enumerate(tags):
        dns = GVChannelDNS.load(tag) if tag in GV_CASES \
            else CKMChannelDNS.load(tag)
        cal = CompressibleCalibration(dns, rel_sigma=rel)
        cache = os.path.join(results_dir,
                             f"ensemble_{tag}_rel{rel:g}.npz")
        ident = {"kind": "compressible_ensemble", "physics": PHYSICS,
                 "case": tag,
                 "n_ensemble": n_ensemble, "seed": SEED + j, "rel": rel}
        loaded = False
        if os.path.isfile(cache) and not regen:
            d = {k: v for k, v in np.load(cache).items()}
            status, reason = cfp.check(d, ident)
            # these ensembles are evaluations of the python 1-D profile
            # baseline, independent of the C++ solvers, so a code-revision
            # drift in the solver tree does not invalidate them; the config
            # fingerprint (and the legacy opt-in for unstamped files) is the
            # operative gate
            if status == "mismatch" or (status == "legacy"
                                        and not cfp.legacy_reuse_allowed()):
                print(f"  {tag}: cache REFUSED ({reason}); regenerating")
            else:
                if status == "legacy":
                    print(f"  {tag}: WARNING reusing pre-fingerprint cache "
                          f"(QBTM_ALLOW_LEGACY_CACHE=1); regenerate with "
                          f"--regen-ensembles to stamp it")
                elif reason:
                    print(f"  {tag}: {reason}")
                loaded = cal.load_cache(d)
        if not loaded:
            cal.run_ensemble(n=n_ensemble, seed=SEED + j)
            cal.fit_surrogates()
            np.savez(cache, **cfp.attach(cal.to_cache(), ident))
        cals[tag] = cal
        print(f"  [{j + 1}/{len(tags)}] {tag}: n_valid={cal.n_valid} "
              f"rel_eff={cal.rel_eff:.4f}", flush=True)
    return cals


def in_distribution(cals, quick):
    """Stage 2: per-case standard-versus-tempered coverage, both blocks."""
    out = {}
    tags = [t for t in cals if t in GV_CASES]
    for tag in tags:
        cal = cals[tag]
        post1 = cal.sample_posterior(eta=1.0, seed=SEED,
                                     n_steps=600 if quick else 2000,
                                     burn_in=200 if quick else 500)
        eta = cal.calibrate_eta(post1, cal.lik_index)
        post_t = cal.sample_posterior(eta=eta, seed=SEED,
                                      n_steps=600 if quick else 2000,
                                      burn_in=200 if quick else 500)
        rec = {"eta": eta, "prt": cal.marginal_prt(post_t),
               "edge_fraction_prt": float(np.mean(
                   (post_t[:, 2] < 0.52) | (post_t[:, 2] > 1.48)))}
        for name, e, post in (("standard", 1.0, post1),
                              ("tempered", eta, post_t)):
            for block, idx in (("lik", cal.lik_index),
                               ("thermal", cal.heldout_index)):
                pp = cal.posterior_predictive(post, eta=e, qoi_index=idx,
                                              seed=SEED + 1)
                cov, sharp = cal.coverage_vs_truth(pp, qoi_index=idx,
                                                   level=LEVEL)
                rec[f"{name}_{block}_coverage"] = cov
                rec[f"{name}_{block}_sharpness"] = sharp
        out[tag] = rec
        print(f"  in-dist {tag}: std thermal "
              f"{rec['standard_thermal_coverage']:.2f} -> tempered "
              f"{rec['tempered_thermal_coverage']:.2f} (eta {eta:.2e})",
              flush=True)
    return out


def cross_mach(cals, quick):
    """Stages 3 to 5: the pre-registered splits through CrossMachStudy."""
    study = CrossMachStudy(cals)
    out = {"primary": {}, "lofo": {}, "ckm": {}}
    train = [t for t in LOW_MACH if t in cals]
    tests = [t for t in HIGH_MACH if t in cals]
    for test in tests:
        out["primary"][test] = study.predict_heldout(train, test, seed=SEED)
        r = out["primary"][test]
        print(f"  cross-Mach {test}: std thermal "
              f"{r['standard_thermal_coverage']:.2f} tempered "
              f"{r['tempered_thermal_coverage']:.2f} conformal "
              f"{r['conformal_thermal_coverage']:.2f}"
              f"{_conformal_claim_tag(r)}", flush=True)
    if not quick:
        gv_in = [t for t in cals if t in GV_CASES]
        for fam in sorted(set(family_of(t) for t in gv_in)):
            test_fam = [t for t in gv_in if family_of(t) == fam]
            train_fam = [t for t in gv_in if family_of(t) != fam]
            for test in test_fam:
                out["lofo"][test] = study.predict_heldout(train_fam, test,
                                                          seed=SEED)
    for tag in CKM_CASES:
        if tag in cals:
            out["ckm"][tag] = study.predict_heldout(train, tag, seed=SEED)
    return out


def plate_leg(cals, quick):
    """Stages 6 and 7: the wall-cooling axis and the measured-Pr_t check."""
    study = CrossMachStudy(cals)
    train = [t for t in LOW_MACH if t in cals]
    post1 = study.pooled_posterior_samples(train, 1.0, seed=SEED)
    eta = float(np.mean([cals[t].calibrate_eta(post1, cals[t].lik_index)
                         for t in train]))
    post_t = study.pooled_posterior_samples(train, eta, seed=SEED)
    rng = np.random.default_rng(SEED + 2)
    out = {"eta": eta,
           "prt_posterior": cals[train[0]].marginal_prt(post_t)}
    for tag in ZDC_CASES:
        if not FlatPlateDNS.is_available(tag):
            continue
        d = FlatPlateDNS.load(tag)
        valid = d.q_hat_valid
        truth = d.q_hat[valid, 1]
        scale = np.max(np.abs(d.q_hat[:, 1]))
        sigma = 0.005 * scale
        rec_case = {}
        for name, e, post in (("standard", 1.0, post1),
                              ("tempered", eta, post_t)):
            prt = post[:: max(1, len(post) // 400), 2]
            preds = np.empty((prt.size, truth.size))
            for i, p in enumerate(prt):
                q = FlatPlateFrozenSST(d, coeffs={"Pr_t": float(p)}).closure()
                preds[i] = q["q_gdh_hat"][valid, 1]
            preds = preds + rng.normal(size=preds.shape) \
                * (sigma / np.sqrt(e))
            lo = np.quantile(preds, 0.05, axis=0)
            hi = np.quantile(preds, 0.95, axis=0)
            rec_case[f"{name}_coverage"] = float(np.mean(
                (truth >= lo) & (truth <= hi)))
            rec_case[f"{name}_sharpness"] = float(np.mean(hi - lo))
        log_layer = (d.yplus > 30) & (d.y_outer < 0.5)
        rec_case["measured_prt"] = {
            "median": float(np.median(d.pr_t[log_layer])),
            "q25": float(np.quantile(d.pr_t[log_layer], 0.25)),
            "q75": float(np.quantile(d.pr_t[log_layer], 0.75)),
        }
        rec_case["tw_tr"] = d.wall["tw_tr"]
        rec_case["m_inf"] = d.wall["m_inf"]
        out[tag] = rec_case
        print(f"  plate {tag}: std {rec_case['standard_coverage']:.2f} "
              f"tempered {rec_case['tempered_coverage']:.2f} measured Pr_t "
              f"median {rec_case['measured_prt']['median']:.3f}", flush=True)
    out["scope_note"] = ("the plate wall-flux clause is not evaluable "
                         "through the approved frozen-mean baseline (it "
                         "predicts no B_q); the plate leg evaluates the "
                         "heat-flux profile, stated per the memo")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--regen-ensembles", action="store_true")
    ap.add_argument("--results", default="results/compressible")
    ap.add_argument("--rel", type=float, default=0.005)
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)
    n_ensemble = 24 if args.quick else 64

    print(f"stage 1: ensembles (n={n_ensemble}, rel={args.rel})", flush=True)
    cals = build_calibrations(args.results, n_ensemble, args.rel,
                              args.regen_ensembles, args.quick)
    print("stage 2: in-distribution", flush=True)
    numbers = {"config": {"seed": SEED, "level": LEVEL, "rel": args.rel,
                          "n_ensemble": n_ensemble, "quick": args.quick,
                          "low_mach": list(LOW_MACH),
                          "high_mach": list(HIGH_MACH)}}
    numbers["in_distribution"] = in_distribution(cals, args.quick)
    print("stages 3-5: cross-Mach, LOFO, CKM", flush=True)
    numbers["cross_mach"] = cross_mach(cals, args.quick)
    print("stages 6-7: plate leg", flush=True)
    numbers["plate"] = plate_leg(cals, args.quick)

    path = os.path.join(args.results,
                        "finding_numbers_quick.json" if args.quick
                        else "finding_numbers.json")
    with open(path, "w") as fh:
        json.dump(numbers, fh, indent=1, default=float)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
