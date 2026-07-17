"""reproduce_separated_aposteriori.py - the coupled model-form ensembles on the
Le-Moin backward-facing step.

Propagates sampled realizable closures from the generative flow and the
Gaussian model-form baseline, and the eigenspace corner family, through the
SAME Reynolds-stress injection to predicted quantities of interest, and scores
them against the DNS truths (reattachment 6.28 and the measured station Cf).
Ensemble construction, exclusion handling, and scoring follow
UQ-RANS_research/separated_modelform/METHODS_OPERATIONALIZATION.md sections 1,
3, 4 and 6 (coherent shared-latent members, ensemble size 24, seed 0;
non-converged members counted and excluded with the exclusion stated).

Stages (results/separated/ is gitignored and regenerable):
  1. baseline + wrappers      - production-grid corrected BFS + injection
  2. training                 - flow and Gaussian on all stations (400 epochs)
  3. ensembles                - 24 coupled solves per probabilistic method
  4. eigenspace corners       - Delta_B = 1.0 and 0.5 families
  5. scoring                  - reattachment and station-Cf records to JSON

Usage:
  PYTHONPATH=build:python python3 python/UQ/reproduce_separated_aposteriori.py
  ... --quick          # coarse grid, short fits, 4-member ensembles
  ... --sensitivity    # add the labeled independent-latent (white) diagnostic
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
from UQ.datasets.separated_aposteriori import BFSAPosteriori

CONFIG = {
    "bfs_cfg": None,            # None = BFSBaselineRANS.DEFAULT_CONFIG (production)
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
    cfg["epochs"] = 60
    cfg["n_members"] = 4
    return cfg


def _member_summary(members, cf_stations=None):
    """Per-member record; wall-Cf curves reduce to the measured stations.

    cf_stations are the downstream station locations; the interpolated
    per-member Cf there is what the pinned conformal overlay calibrates on
    (METHODS_OPERATIONALIZATION.md section 7), so it is persisted.
    """
    out = []
    for m in members:
        rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in m.items() if k not in ("cf_x", "cf")}
        if cf_stations is not None:
            rec["cf_at_stations"] = np.interp(cf_stations, m["cf_x"],
                                              m["cf"]).tolist()
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sensitivity", action="store_true",
                    help="add the labeled independent-latent (white) diagnostic")
    args = ap.parse_args()
    if args.quick:
        _quick(CONFIG)

    import torch
    os.makedirs(OUT, exist_ok=True)

    print("== stage 1: corrected baseline + injection wrapper ==", flush=True)
    dns = BackwardFacingStepDNS.load()
    baseline = BFSBaselineRANS.solve(CONFIG["bfs_cfg"], dns=dns)
    truth = dns.reattachment_truth()
    print(f"  baseline: {baseline.status}, x_r/h = {baseline.reattachment:.3f} "
          f"(DNS {truth})")
    study = BFSAPosteriori.build(cfg=CONFIG["bfs_cfg"], dns=dns,
                                 baseline=baseline)
    xs_all, cf_dns = dns.cf_stations()
    keep = xs_all > 0.0
    cf_stations = xs_all[keep]

    numbers = {
        "config": {k: v for k, v in CONFIG.items()},
        "baseline": {"status": baseline.status,
                     "reattachment": baseline.reattachment,
                     "reattachment_dns": truth,
                     "abs_error": abs(baseline.reattachment - truth)},
        "ensembles": {},
        "eigenspace": {},
    }

    print("== stages 2-3: train and propagate the probabilistic methods ==",
          flush=True)
    for kind in CONFIG["kinds"]:
        model = study.train(kind, seed=CONFIG["seed"], epochs=CONFIG["epochs"])
        torch.manual_seed(CONFIG["seed"])
        members = study.run_ensemble(model, n_members=CONFIG["n_members"],
                                     shared_latent=True)
        rec = {
            "reattachment": BFSAPosteriori.score_reattachment(
                members, truth, level=CONFIG["level"]),
            "cf": study.score_cf(members, level=CONFIG["level"]),
            "members": _member_summary(members, cf_stations=cf_stations),
            "cf_stations_xh": cf_stations.tolist(),
            "cf_measured": cf_dns[keep].tolist(),
        }
        if args.sensitivity:
            torch.manual_seed(CONFIG["seed"])
            white = study.run_ensemble(model, n_members=CONFIG["n_members"],
                                       shared_latent=False)
            rec["independent_latent_diagnostic"] = {
                "note": "labeled sensitivity: spatially white members, "
                        "never the primary band",
                "reattachment": BFSAPosteriori.score_reattachment(
                    white, truth, level=CONFIG["level"]),
            }
        numbers["ensembles"][kind] = rec
        r = rec["reattachment"]
        print(f"  {kind:>5}: {r.get('n_members', 0) - r.get('n_nonconverged', 0)}"
              f"/{r.get('n_members', 0)} converged, "
              f"x_r mean {r.get('mean', float('nan')):.3f} "
              f"band {r.get('band')}, contains truth: "
              f"{r.get('contains_truth')}", flush=True)

    print("== stage 4: eigenspace families (three-corner and five-state) ==",
          flush=True)
    numbers["five_state"] = {}
    for delta in CONFIG["eigenspace_deltas"]:
        corners = study.run_eigenspace(delta_b=delta)
        env = BFSAPosteriori.score_envelope(corners, truth)
        env["cf"] = study.score_cf(list(corners.values()),
                                   level=CONFIG["level"])
        numbers["eigenspace"][str(delta)] = env
        print(f"  delta_B={delta}: corners {env['corners']} envelope "
              f"{env.get('envelope')} contains truth: "
              f"{env.get('contains_truth')}", flush=True)
        # the documented 2017 five-state family (the corrected-solver probe's
        # added reported baseline; scored exactly like the corner family, and
        # as a deterministic bounding family its CRPS convention downstream is
        # the exact M^2 discrete-forecast reading, never the fair estimator)
        five = study.run_eigenspace(delta_b=delta, five_state=True)
        env5 = BFSAPosteriori.score_envelope(five, truth)
        env5["cf"] = study.score_cf(list(five.values()),
                                    level=CONFIG["level"])
        numbers["five_state"][str(delta)] = env5
        print(f"  delta_B={delta} five-state: {env5['corners']} envelope "
              f"{env5.get('envelope')} contains truth: "
              f"{env5.get('contains_truth')}", flush=True)

    path = os.path.join(OUT, "aposteriori_numbers.json")
    with open(path, "w") as fh:
        json.dump(numbers, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
