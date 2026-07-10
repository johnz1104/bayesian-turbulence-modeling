"""Reproduce the heat-flux model-form study.

Fixed-seed production driver for every number the evidence package quotes,
measured against the committed pre-registration
(UQ-RANS_research/heatflux_modelform/PRE_REGISTRATION.md: splits, targets,
features, scores, bands and seeds fixed there BEFORE these runs). Usage:

    PYTHONPATH=build:python python3 python/UQ/reproduce_heatflux_modelform.py
        [--quick] [--stage all|apriori|conformal] [--results results/heatflux]
        [--cache results/compressible] [--cache-rel01 results/compressible_rel01]

Stages:
  1. Assembly: the (features, dq) pairs of every available case through the
     committed extraction; per-case row counts, M_t spans, the baseline wall
     flux, and the realizability fraction recorded.
  2. In-sample machinery check: train on the full channel matrix, score in
     sample, both targets (wall-normal and the joint family leg).
  3. Leave-one-case-out within the low-Mach training cases (the pre-registered
     in-distribution control, wall-normal target).
  4. The primary cross-Mach split: fit on the low-Mach cases, score every
     held-out high-Mach case (wall-normal for the transfer clause; the joint
     leg for the family clause), plus the never-trained independent-code
     cases and the flat plates (far transfer, wall-normal only).
  5. Leave-one-Mach-family-out over the matrix (the transfer profile along
     the axis, wall-normal target).
  6. The secondary axis: the pinned normalized conformal re-scoring of the
     committed global-correction residuals from the cached calibration
     ensembles, at both committed sigma levels; no recalibration.

Model seeds {0, 1, 2}; every metric is recorded per seed with the seed mean
(the pre-registered criterion reads the seed mean). Every stage writes into
one numbers JSON; the PIT arrays for the figures cache into one npz. Nothing
here is tuned toward any coverage; a null or negative shape per the
pre-registration is reported exactly as it comes out.
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np

from UQ.datasets.heatflux_apriori import (HeatFluxAPriori,
                                          ThermalConformalRescore,
                                          LOW_MACH, HIGH_MACH, MODEL_SEEDS,
                                          EPOCHS, N_SAMPLES, mach_family)
from UQ.datasets.gv_channel import GVChannelDNS, GV_CASES
from UQ.datasets.ckm_channel import CKM_CASES
from UQ.datasets.zdc_flatplate import ZDC_CASES

SEED = 0
CONFORMAL_LEVELS = (0.9, 0.5)


def sample_seed(stage, kind, model_seed, tag):
    """Stable per-evaluation draw seed: depends only on the evaluation
    identity, never on call order, so stage subsets reproduce the full run."""
    key = f"{stage}/{kind}/{model_seed}/{tag}"
    return int(zlib.crc32(key.encode())) & 0x7FFFFFFF


def collapse_seeds(records):
    """Per-metric seed statistics: mean, min, max and the per-seed values.
    Non-float entries (counts, spans) are constant across seeds; keep the
    first."""
    out = {}
    for key in records[0]:
        vals = [r[key] for r in records]
        if isinstance(vals[0], float):
            out[key] = {"mean": float(np.mean(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "per_seed": [float(v) for v in vals]}
        else:
            out[key] = vals[0]
    return out


def fit_and_score(study, train, test, components, kinds, seeds, epochs,
                  n_samples, arrays, stage):
    """Fit each (kind, seed) once on the training cases; score every test
    case; collapse over seeds. Returns {kind: {tag: {metric: stats}}}."""
    out = {}
    for kind in kinds:
        per_tag = {t: [] for t in test}
        for s in seeds:
            model = study.fit(kind, train, components, seed=s, epochs=epochs)
            for tag in test:
                r = study.evaluate(model, tag, components,
                                   n_samples=n_samples,
                                   sample_seed=sample_seed(stage, kind, s,
                                                           tag))
                arr = r.pop("arrays", None)
                if arr is not None:
                    arrays[f"{stage}/{kind}/seed{s}/{tag}/pit_y"] = \
                        arr["pit_y"]
                per_tag[tag].append(r)
        out[kind] = {t: collapse_seeds(rs) for t, rs in per_tag.items()}
        cov = [v.get("coverage_y_0.9", {}).get("mean") for v in
               out[kind].values() if "coverage_y_0.9" in v]
        if cov:
            print(f"  {stage} {kind}: mean coverage_y_0.9 over "
                  f"{len(cov)} cases = {np.mean(cov):.3f}", flush=True)
    return out


def stage_assembly(study):
    out = {"skipped": dict(study.skipped),
           "gv": list(study.gv), "ckm": list(study.ckm),
           "plates": list(study.plates)}
    for tag, rec in study.cases.items():
        out[tag] = {k: rec[k] for k in
                    ("kind", "n", "n_dropped", "m_t_min", "m_t_max",
                     "realizable_fraction", "m_nom", "b_q_base")
                    if k in rec}
        if "re_nom" in rec:
            out[tag]["re_nom"] = rec["re_nom"]
        if "tw_tr" in rec:
            out[tag]["tw_tr"] = rec["tw_tr"]
    n_train_rows = sum(study.cases[t]["n"] for t in study.low_mach())
    out["n_train_rows_low_mach"] = int(n_train_rows)
    print(f"  assembled {len(study.gv)} channel + {len(study.ckm)} "
          f"cross-check + {len(study.plates)} plate cases; "
          f"{n_train_rows} low-Mach training rows", flush=True)
    return out


def stage_apriori(study, numbers, arrays, quick):
    seeds = (0,) if quick else MODEL_SEEDS
    epochs = 40 if quick else EPOCHS
    nsamp = 64 if quick else N_SAMPLES
    low, high = study.low_mach(), study.high_mach()

    if not quick:
        print("stage 2: in-sample machinery check", flush=True)
        numbers["in_sample"] = {}
        for target in ("wall_normal", "joint_xy"):
            numbers["in_sample"][target] = fit_and_score(
                study, study.gv, study.gv, target, ("flow", "gauss"), seeds,
                epochs, nsamp, arrays, f"in_sample_{target}")

    print("stage 3: leave-one-case-out control (low-Mach)", flush=True)
    numbers["loco"] = {k: {} for k in ("flow", "gauss", "pooled")}
    for held, train in study.loco_splits():
        r = fit_and_score(study, train, (held,), "wall_normal",
                          ("flow", "gauss", "pooled"), seeds, epochs, nsamp,
                          arrays, "loco")
        for kind in r:
            numbers["loco"][kind][held] = r[kind][held]

    print("stage 4: cross-Mach primary split", flush=True)
    numbers["cross_mach"] = {}
    numbers["cross_mach"]["wall_normal"] = fit_and_score(
        study, low, high + study.ckm + study.plates, "wall_normal",
        ("flow", "gauss", "pooled"), seeds, epochs, nsamp, arrays,
        "cross_mach_wall_normal")
    numbers["cross_mach"]["joint_xy"] = fit_and_score(
        study, low, high, "joint_xy", ("flow", "gauss"), seeds, epochs,
        nsamp, arrays, "cross_mach_joint_xy")

    if not quick:
        print("stage 5: leave-one-Mach-family-out", flush=True)
        numbers["lomo"] = {k: {} for k in ("flow", "gauss")}
        for fam, test, train in study.lomo_splits():
            r = fit_and_score(study, train, test, "wall_normal",
                              ("flow", "gauss"), seeds, epochs, nsamp,
                              arrays, f"lomo_f{fam}")
            for kind in r:
                for tag in r[kind]:
                    r[kind][tag]["family"] = fam
                    numbers["lomo"][kind][tag] = r[kind][tag]


def stage_conformal(numbers, cache_dirs, quick):
    """The secondary axis, from the cached calibration ensembles."""
    from UQ.datasets.compressible_calibration import CompressibleCalibration

    numbers["conformal"] = {}
    for rel, cache_dir in cache_dirs:
        label = f"rel{rel:g}"
        # the smoke path records the skip without loading and refitting the
        # 24 cached surrogates it would only discard (the load is the
        # expensive part of this stage)
        if quick:
            numbers["conformal"][label] = {"cache_dir": cache_dir,
                                           "status": "quick_skip"}
            continue
        cals, missing = {}, []
        for tag in GV_CASES:
            cache = os.path.join(cache_dir, f"ensemble_{tag}_rel{rel:g}.npz")
            if not os.path.isfile(cache):
                missing.append(tag)
                continue
            cal = CompressibleCalibration(GVChannelDNS.load(tag),
                                          rel_sigma=rel)
            cal.load_cache({k: v for k, v in np.load(cache).items()})
            cals[tag] = cal
        rec = {"cache_dir": cache_dir, "missing": missing}
        # the conformal secondary axis is defined on the committed 8/16 split;
        # run it only when every case of that split is cache-backed, so a
        # partially populated cache records an explicit skip rather than
        # silently computing the pinned mean_coverage/gap keys on a subset
        # (which would also NaN out if a whole side of the split were absent)
        split = list(LOW_MACH) + list(HIGH_MACH)
        have = [t for t in split if t in cals]
        if len(have) < len(split):
            rec["status"] = "incomplete_split"
            rec["split_cases"] = f"{len(have)}/{len(split)}"
            numbers["conformal"][label] = rec
            print(f"  {label}: split {len(have)}/{len(split)} cache-backed;"
                  f" conformal leg skipped (needs the full 8/16 split)",
                  flush=True)
            continue
        rsc = ThermalConformalRescore(cals, LOW_MACH, HIGH_MACH, seed=SEED)
        rec["eta"] = rsc.eta
        for score in ThermalConformalRescore.SCORES:
            rec[score] = {}
            for level in CONFORMAL_LEVELS:
                run = rsc.run(score, level=level)
                rec[score][f"level{level:g}"] = run
                print(f"  {label} {score} at {level:g}: held-out thermal "
                      f"coverage {run['mean_coverage']:.3f} "
                      f"(gap {run['gap']:.3f})", flush=True)
        numbers["conformal"][label] = rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", default="all",
                    choices=("all", "apriori", "conformal"))
    ap.add_argument("--results", default="results/heatflux")
    ap.add_argument("--cache", default="results/compressible")
    ap.add_argument("--cache-rel01", default="results/compressible_rel01")
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)

    gv_cases = GV_CASES
    if args.quick:
        # two low-Mach and two high-Mach cases: the smoke path
        gv_cases = (GV_CASES[6], GV_CASES[12], GV_CASES[9], GV_CASES[21])

    name = "finding_numbers_quick.json" if args.quick \
        else "finding_numbers.json"
    path = os.path.join(args.results, name)
    # stage subsets compose into one numbers file: a prior run's stages are
    # kept and only the stages executed here are (re)written
    numbers = {}
    if os.path.isfile(path):
        with open(path) as fh:
            numbers = json.load(fh)
    numbers["config"] = {
        "seed": SEED, "model_seeds": list(MODEL_SEEDS),
        "epochs": EPOCHS, "n_samples": N_SAMPLES, "quick": args.quick,
        "low_mach": list(LOW_MACH), "high_mach": list(HIGH_MACH),
        "mach_family": {c: mach_family(c) for c in GV_CASES},
    }
    arrays = {}

    print("stage 1: assembly", flush=True)
    study = HeatFluxAPriori.build(
        gv_cases=gv_cases,
        ckm_cases=() if args.quick else CKM_CASES,
        zdc_cases=() if args.quick else ZDC_CASES)
    numbers["assembly"] = stage_assembly(study)

    if args.stage in ("all", "apriori"):
        stage_apriori(study, numbers, arrays, args.quick)
    if args.stage in ("all", "conformal"):
        print("stage 6: normalized conformal re-scoring", flush=True)
        stage_conformal(numbers,
                        ((0.005, args.cache), (0.01, args.cache_rel01)),
                        args.quick)

    with open(path, "w") as fh:
        json.dump(numbers, fh, indent=1, default=float)
    if arrays:
        # quick-suffix the PIT array file exactly as the numbers JSON is
        # suffixed, so a --quick smoke run never overwrites the production
        # arrays the committed figures are built from
        arr_name = "figure_arrays_quick.npz" if args.quick \
            else "figure_arrays.npz"
        np.savez(os.path.join(args.results, arr_name), **arrays)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
