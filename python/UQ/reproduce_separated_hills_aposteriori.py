"""reproduce_separated_hills_aposteriori.py - the coupled model-form ensembles
on the periodic hills, and the cross-geometry propagation in both directions.

Propagates sampled realizable closures from the generative flow and the
Gaussian model-form baseline, and the eigenspace corner family, through the
SAME Reynolds-stress injection on the streamwise-periodic hills solver, and
scores them against the DNS truths (the wall-shear bubble geometry read from
the dense field and the mean velocity at the pinned probes). Then the
cross-geometry clause in its propagated form: the model trained on one
geometry drives the other geometry's coupled ensemble, both directions.
Everything follows UQ-RANS_research/separated_modelform/
METHODS_OPERATIONALIZATION.md sections 1, 3, 6 and 9, pinned before any of
these ensembles ran (coherent shared-latent members, ensemble size 24, seed 0;
non-converged members counted and excluded with the exclusion stated).

Stages (results/separated/ is gitignored and regenerable):
  1. baselines + wrappers      - production-grid hills and BFS + injections
  2. within-hills ensembles    - flow and Gaussian, 24 coupled solves each
  3. eigenspace corners        - Delta_B = 1.0 and 0.5 families on the hills
  4. cross-geometry            - BFS-trained into hills, hills-trained into BFS

Usage:
  PYTHONPATH=build:python python3 python/UQ/reproduce_separated_hills_aposteriori.py
  ... --quick          # coarse grids, short fits, 4-member ensembles
  ... --sensitivity    # add the labeled independent-latent (white) diagnostic
  ... --skip-cross     # within-hills stages only
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "build"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from UQ.datasets.backward_facing_step import BackwardFacingStepDNS
from UQ.datasets.bfs_baseline import BFSBaselineRANS
from UQ.datasets.periodic_hills import PeriodicHillsDNS
from UQ.datasets.hills_baseline import HillsBaselineRANS
from UQ.datasets.separated_aposteriori import BFSAPosteriori, HillsAPosteriori

CONFIG = {
    "bfs_cfg": None,            # None = production defaults of each baseline
    "hills_cfg": None,
    "hills_case": "1p0",
    "kinds": ["flow", "gauss"],
    "epochs": 400,
    "n_members": 24,
    "level": 0.9,
    "seed": 0,
    "eigenspace_deltas": [1.0, 0.5],
}
OUT = os.path.join(_HERE, "..", "..", "results", "separated")


def _quick(cfg):
    cfg["bfs_cfg"] = {"nx_up": 20, "nx_down": 26, "ny_up": 14, "ny_down": 10,
                      "Lu": 10.0, "Ld": 22.0, "max_iter": 8000,
                      "conv_tol": 1.0e-4}
    cfg["hills_cfg"] = {"nx": 48, "ny": 32, "max_iter": 15000,
                        "conv_tol": 1.0e-4, "body_force": 0.0095}
    cfg["epochs"] = 60
    cfg["n_members"] = 4
    return cfg


def _hills_member_summary(members):
    """Per-member scalar record; probe vectors persisted as lists."""
    out = []
    for m in members:
        rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in m.items() if k != "u_probe"}
        if "u_probe" in m:
            rec["u_probe"] = np.asarray(m["u_probe"], float).tolist()
        out.append(rec)
    return out


def _score_hills(study, members, truths, level):
    """The pinned hills quantity set (scalars + probes) for one ensemble."""
    rec = {}
    for key in HillsAPosteriori.SCALAR_KEYS:
        rec[key] = HillsAPosteriori.score_scalar(members, key, truths[key],
                                                 level=level)
    rec["probes"] = study.score_probes(members, level=level)
    rec["members"] = _hills_member_summary(members)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sensitivity", action="store_true",
                    help="add the labeled independent-latent (white) diagnostic")
    ap.add_argument("--skip-cross", action="store_true",
                    help="within-hills stages only")
    args = ap.parse_args()
    if args.quick:
        _quick(CONFIG)

    import torch
    os.makedirs(OUT, exist_ok=True)

    print("== stage 1: baselines + injection wrappers ==", flush=True)
    hills_dns = PeriodicHillsDNS.load(CONFIG["hills_case"])
    hills_base = HillsBaselineRANS.solve(CONFIG["hills_cfg"], dns=hills_dns,
                                         case=CONFIG["hills_case"])
    x_s_dns, x_r_dns, len_dns = hills_dns.bottom_wall_bubble()
    truths = {"separation": x_s_dns, "reattachment": x_r_dns,
              "bubble_length": len_dns}
    print(f"  hills baseline: {hills_base.status}, x_r/h = "
          f"{hills_base.reattachment:.3f} (DNS {x_r_dns:.3f}), U_b = "
          f"{hills_base.bulk_crest:.3f}", flush=True)
    hills = HillsAPosteriori.build(cfg=CONFIG["hills_cfg"], dns=hills_dns,
                                   baseline=hills_base,
                                   case=CONFIG["hills_case"])
    base_ref = hills.inj.run_baseline()
    print(f"  no-injection reference: x_s {base_ref['separation']:.3f} "
          f"x_r {base_ref['reattachment']:.3f} ({base_ref['status']})",
          flush=True)

    numbers = {
        "config": {k: v for k, v in CONFIG.items()},
        "baseline": {
            "status": hills_base.status,
            "separation": base_ref["separation"],
            "reattachment": base_ref["reattachment"],
            "bubble_length": base_ref["bubble_length"],
            "truths_dns": truths,
            "bulk_crest": hills_base.bulk_crest,
            "n_probes": int(hills.inj.probe_x.size),
        },
        "ensembles": {},
        "eigenspace": {},
        "cross_geometry": {},
    }

    print("== stage 2: within-hills probabilistic ensembles ==", flush=True)
    for kind in CONFIG["kinds"]:
        model = hills.train(kind, seed=CONFIG["seed"], epochs=CONFIG["epochs"])
        torch.manual_seed(CONFIG["seed"])
        members = hills.run_ensemble(model, n_members=CONFIG["n_members"],
                                     shared_latent=True)
        rec = _score_hills(hills, members, truths, CONFIG["level"])
        if args.sensitivity:
            torch.manual_seed(CONFIG["seed"])
            white = hills.run_ensemble(model, n_members=CONFIG["n_members"],
                                       shared_latent=False)
            rec["independent_latent_diagnostic"] = {
                "note": "labeled sensitivity: spatially white members, "
                        "never the primary band",
                "reattachment": HillsAPosteriori.score_scalar(
                    white, "reattachment", truths["reattachment"],
                    level=CONFIG["level"]),
            }
        numbers["ensembles"][kind] = rec
        r = rec["reattachment"]
        print(f"  {kind:>5}: {r['n_members'] - r['n_nonconverged']}"
              f"/{r['n_members']} converged, x_r mean "
              f"{r.get('mean', float('nan')):.3f} band {r.get('band')}, "
              f"contains truth: {r.get('contains_truth')}", flush=True)

    print("== stage 3: eigenspace families (three-corner and five-state) ==",
          flush=True)
    numbers["five_state"] = {}
    for delta in CONFIG["eigenspace_deltas"]:
        corners = hills.run_eigenspace(delta_b=delta)
        env = {key: HillsAPosteriori.score_envelope(corners, key, truths[key])
               for key in HillsAPosteriori.SCALAR_KEYS}
        env["probes"] = hills.score_probes(list(corners.values()),
                                           level=CONFIG["level"])
        numbers["eigenspace"][str(delta)] = env
        er = env["reattachment"]
        print(f"  delta_B={delta}: corners {er['corners']} envelope "
              f"{er.get('envelope')} contains truth: "
              f"{er.get('contains_truth')}", flush=True)
        # the documented five-state family (the corrected-solver probe's added
        # reported baseline; deterministic bounding family, so any CRPS read
        # downstream uses the exact discrete-forecast convention)
        five = hills.run_eigenspace(delta_b=delta, five_state=True)
        env5 = {key: HillsAPosteriori.score_envelope(five, key, truths[key])
                for key in HillsAPosteriori.SCALAR_KEYS}
        env5["probes"] = hills.score_probes(list(five.values()),
                                            level=CONFIG["level"])
        numbers["five_state"][str(delta)] = env5
        er5 = env5["reattachment"]
        print(f"  delta_B={delta} five-state: {er5['corners']} envelope "
              f"{er5.get('envelope')} contains truth: "
              f"{er5.get('contains_truth')}", flush=True)

    if not args.skip_cross:
        print("== stage 4: cross-geometry propagation (both directions) ==",
              flush=True)
        bfs_dns = BackwardFacingStepDNS.load()
        bfs_base = BFSBaselineRANS.solve(CONFIG["bfs_cfg"], dns=bfs_dns)
        bfs = BFSAPosteriori.build(cfg=CONFIG["bfs_cfg"], dns=bfs_dns,
                                   baseline=bfs_base)
        truth_bfs = bfs_dns.reattachment_truth()
        print(f"  bfs baseline: {bfs_base.status}, x_r/h = "
              f"{bfs_base.reattachment:.3f} (DNS {truth_bfs})", flush=True)
        for kind in CONFIG["kinds"]:
            m_bfs = bfs.train(kind, seed=CONFIG["seed"],
                              epochs=CONFIG["epochs"])
            torch.manual_seed(CONFIG["seed"])
            b2h = hills.run_ensemble(m_bfs, n_members=CONFIG["n_members"],
                                     shared_latent=True)
            rec_b2h = _score_hills(hills, b2h, truths, CONFIG["level"])

            m_hills = hills.train(kind, seed=CONFIG["seed"],
                                  epochs=CONFIG["epochs"])
            torch.manual_seed(CONFIG["seed"])
            h2b = bfs.run_ensemble(m_hills, n_members=CONFIG["n_members"],
                                   shared_latent=True)
            rec_h2b = {
                "reattachment": BFSAPosteriori.score_reattachment(
                    h2b, truth_bfs, level=CONFIG["level"]),
                "cf": bfs.score_cf(h2b, level=CONFIG["level"]),
            }
            numbers["cross_geometry"][kind] = {"bfs_to_hills": rec_b2h,
                                               "hills_to_bfs": rec_h2b}
            rb = rec_b2h["reattachment"]
            rh = rec_h2b["reattachment"]
            print(f"  {kind:>5}: bfs->hills x_r mean "
                  f"{rb.get('mean', float('nan')):.3f} contains "
                  f"{rb.get('contains_truth')} | hills->bfs x_r mean "
                  f"{rh.get('mean', float('nan')):.3f} contains "
                  f"{rh.get('contains_truth')}", flush=True)

    path = os.path.join(OUT, "hills_aposteriori_numbers.json")
    with open(path, "w") as fh:
        json.dump(numbers, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
