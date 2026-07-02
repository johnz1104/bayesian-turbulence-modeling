"""reproduce_separated_apriori.py - the a-priori model-form checkpoint on the
real separated discrepancy (the precursor of the separated-flow study).

Fits the generative flow and the Gaussian model-form baseline on the real
(feature, db) pairs of the Le-Moin backward-facing step, against the
limiter-consistent corrected baseline, and scores realizable predictive
anisotropies against the DNS anisotropy per the protocol pinned in
UQ-RANS_research/separated_modelform/METHODS_OPERATIONALIZATION.md section 5:
leave-one-station-out over the six stations plus the train-on-all
in-distribution machinery check, per-component 90 percent coverage/sharpness/
CRPS, the multivariate energy score, and the realizable fraction.

Stages (results/separated/ is gitignored and regenerable):
  1. baseline + discrepancy   - production-grid corrected BFS solve
  2. in-distribution          - train on all stations, score all points
  3. leave-one-station-out    - the pinned cross-station protocol

Reproducibility: fixed seed, parameters in CONFIG below. Nothing here tunes
any model or training choice toward a criterion; the parameters are the pinned
protocol values.

Usage:
  PYTHONPATH=build:python python3 python/UQ/reproduce_separated_apriori.py
  ... --quick     # coarse grid, short fits, fewer samples (smoke run)
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
from UQ.datasets.separated_apriori import SeparatedAPriori, COMPONENT_NAMES

CONFIG = {
    "bfs_cfg": None,            # None = BFSBaselineRANS.DEFAULT_CONFIG (production)
    "kinds": ["flow", "gauss"],
    "epochs": 400,
    "n_samples": 128,
    "level": 0.9,
    "seed": 0,
}
OUT = os.path.join(_HERE, "..", "..", "results", "separated")


def _quick(cfg):
    cfg["bfs_cfg"] = {"nx_up": 20, "nx_down": 26, "ny_up": 14, "ny_down": 10,
                      "Lu": 10.0, "Ld": 22.0, "max_iter": 8000,
                      "conv_tol": 1.0e-4}
    cfg["epochs"] = 60
    cfg["n_samples"] = 48
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        _quick(CONFIG)

    os.makedirs(OUT, exist_ok=True)

    print("== stage 1: corrected baseline + discrepancy (production grid) ==",
          flush=True)
    dns = BackwardFacingStepDNS.load()
    baseline = BFSBaselineRANS.solve(CONFIG["bfs_cfg"], dns=dns)
    print(f"  baseline: {baseline.status}, x_r/h = {baseline.reattachment:.3f} "
          f"(DNS {dns.reattachment_truth()})")
    ap_data = SeparatedAPriori.build(cfg=CONFIG["bfs_cfg"], dns=dns,
                                     baseline=baseline)
    print(f"  training set: {ap_data.n} valid (feature, db) points, "
          f"{len(ap_data.station_xh)} stations")

    print("== stage 2: in-distribution machinery check (train all, score all) ==",
          flush=True)
    all_mask = np.ones(ap_data.n, dtype=bool)
    in_dist = {}
    for kind in CONFIG["kinds"]:
        model = ap_data.fit(kind, all_mask, seed=CONFIG["seed"],
                            epochs=CONFIG["epochs"])
        in_dist[kind] = ap_data.evaluate(model, all_mask,
                                         n_samples=CONFIG["n_samples"],
                                         level=CONFIG["level"])
        r = in_dist[kind]
        print(f"  {kind:>5}: coverage(mean over comps) "
              f"{np.mean(list(r['coverage'].values())):.3f}  "
              f"crps {r['crps_mean']:.4f}  energy {r['energy_score']:.4f}  "
              f"realizable {r['realizable_fraction']:.3f}")

    print("== stage 3: leave-one-station-out (the pinned protocol) ==", flush=True)
    loso = ap_data.leave_one_station_out(kinds=tuple(CONFIG["kinds"]),
                                         seed=CONFIG["seed"],
                                         epochs=CONFIG["epochs"],
                                         n_samples=CONFIG["n_samples"],
                                         level=CONFIG["level"])
    for xh, per in loso.items():
        line = f"  x/h {xh:>5}: "
        for kind in CONFIG["kinds"]:
            r = per[kind]
            line += (f"{kind} cov {np.mean(list(r['coverage'].values())):.3f} "
                     f"crps {r['crps_mean']:.4f} es {r['energy_score']:.4f}   ")
        print(line, flush=True)

    numbers = {
        "config": {k: v for k, v in CONFIG.items()},
        "baseline": {"status": baseline.status,
                     "reattachment": baseline.reattachment,
                     "reattachment_dns": dns.reattachment_truth(),
                     "n_points": ap_data.n},
        "component_names": list(COMPONENT_NAMES),
        "in_distribution": in_dist,
        "leave_one_station_out": {str(k): v for k, v in loso.items()},
    }
    path = os.path.join(OUT, "apriori_numbers.json")
    with open(path, "w") as fh:
        json.dump(numbers, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
