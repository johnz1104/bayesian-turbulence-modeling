"""reproduce_separated_crossgeom.py - hills a-priori coverage and the
cross-geometry transfer of the separated-flow model-form study.

Runs the pinned periodic-hills protocol (METHODS_OPERATIONALIZATION.md section
8) and the pre-registered cross-geometry clause: the dense-field within-hills
coverage (leave-one-band-out plus the in-distribution check), and the transfer
in BOTH directions (train on the backward-facing step, score on the hills, and
the reverse), for the generative flow and the Gaussian baseline through the
identical fit / sample / project / score path.

Stages (results/separated/ is gitignored and regenerable):
  1. baselines + discrepancies  - both geometries on their production grids
  2. within-hills               - in-distribution + leave-one-band-out
  3. cross-geometry             - both directions, both model kinds

Reproducibility: fixed seed, parameters in CONFIG below; nothing is tuned
toward any criterion.

Usage:
  PYTHONPATH=build:python python3 python/UQ/reproduce_separated_crossgeom.py
  ... --quick     # coarse grids, short fits, fewer samples (smoke run)
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
from UQ.datasets.separated_discrepancy import BFSDiscrepancy
from UQ.datasets.periodic_hills import PeriodicHillsDNS
from UQ.datasets.hills_baseline import HillsBaselineRANS
from UQ.datasets.hills_discrepancy import HillsDiscrepancy
from UQ.datasets.separated_apriori import SeparatedAPriori

CONFIG = {
    "bfs_cfg": None,            # None = production defaults of each baseline
    "hills_cfg": None,
    "hills_case": "1p0",
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
    cfg["hills_cfg"] = {"nx": 48, "ny": 32, "max_iter": 15000,
                        "conv_tol": 1.0e-4, "body_force": 0.0095}
    cfg["epochs"] = 60
    cfg["n_samples"] = 48
    return cfg


def _mean_cov(rec):
    return float(np.mean(list(rec["coverage"].values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        _quick(CONFIG)

    os.makedirs(OUT, exist_ok=True)

    print("== stage 1: baselines + discrepancies (both geometries) ==", flush=True)
    bfs_dns = BackwardFacingStepDNS.load()
    bfs_base = BFSBaselineRANS.solve(CONFIG["bfs_cfg"], dns=bfs_dns)
    bfs = SeparatedAPriori(BFSDiscrepancy.build(dns=bfs_dns, baseline=bfs_base))
    print(f"  bfs:   x_r {bfs_base.reattachment:.3f} (DNS 6.28), "
          f"{bfs.n} training points")

    hills_dns = PeriodicHillsDNS.load(CONFIG["hills_case"])
    hills_base = HillsBaselineRANS.solve(CONFIG["hills_cfg"], dns=hills_dns,
                                         case=CONFIG["hills_case"])
    hills_disc = HillsDiscrepancy.build(dns=hills_dns, baseline=hills_base)
    hills = SeparatedAPriori(hills_disc)
    xr_dns_hills = hills_dns.bottom_wall_reattachment()
    print(f"  hills: x_r {hills_base.reattachment:.3f} (DNS field "
          f"{xr_dns_hills:.2f}), U_b {hills_base.bulk_crest:.3f}, "
          f"{hills.n} training points")

    numbers = {
        "config": {k: v for k, v in CONFIG.items()},
        "baselines": {
            "bfs": {"reattachment": bfs_base.reattachment,
                    "reattachment_dns": 6.28, "status": bfs_base.status,
                    "n_points": int(bfs.n)},
            "hills": {"reattachment": hills_base.reattachment,
                      "reattachment_dns": float(xr_dns_hills),
                      "bulk_crest": hills_base.bulk_crest,
                      "status": hills_base.status,
                      "n_points": int(hills.n),
                      "db_by_band": hills_disc.magnitude_by_band()},
        },
        "within_hills": {},
        "cross_geometry": {},
    }

    print("== stage 2: within-hills coverage (pinned band protocol) ==", flush=True)
    all_h = np.ones(hills.n, dtype=bool)
    in_dist = {}
    for kind in CONFIG["kinds"]:
        model = hills.fit(kind, all_h, seed=CONFIG["seed"],
                          epochs=CONFIG["epochs"])
        in_dist[kind] = hills.evaluate(model, all_h,
                                       n_samples=CONFIG["n_samples"],
                                       level=CONFIG["level"])
        r = in_dist[kind]
        print(f"  in-dist {kind:>5}: cov {_mean_cov(r):.3f} "
              f"crps {r['crps_mean']:.4f} es {r['energy_score']:.4f} "
              f"realizable {r['realizable_fraction']:.3f}", flush=True)
    numbers["within_hills"]["in_distribution"] = in_dist
    lobo = hills.leave_one_station_out(kinds=tuple(CONFIG["kinds"]),
                                       seed=CONFIG["seed"],
                                       epochs=CONFIG["epochs"],
                                       n_samples=CONFIG["n_samples"],
                                       level=CONFIG["level"])
    for band, per in lobo.items():
        line = f"  band {band:g}: "
        for kind in CONFIG["kinds"]:
            line += f"{kind} cov {_mean_cov(per[kind]):.3f}  "
        print(line, flush=True)
    numbers["within_hills"]["leave_one_band_out"] = {str(k): v
                                                     for k, v in lobo.items()}

    print("== stage 3: cross-geometry transfer (both directions) ==", flush=True)
    all_b = np.ones(bfs.n, dtype=bool)
    cross = {}
    for kind in CONFIG["kinds"]:
        m_bfs = bfs.fit(kind, all_b, seed=CONFIG["seed"],
                        epochs=CONFIG["epochs"])
        m_hills = hills.fit(kind, all_h, seed=CONFIG["seed"],
                            epochs=CONFIG["epochs"])
        b2h = bfs.evaluate_external(m_bfs, hills,
                                    n_samples=CONFIG["n_samples"],
                                    level=CONFIG["level"])
        h2b = hills.evaluate_external(m_hills, bfs,
                                      n_samples=CONFIG["n_samples"],
                                      level=CONFIG["level"])
        cross[kind] = {"bfs_to_hills": b2h, "hills_to_bfs": h2b}
        print(f"  {kind:>5}: bfs->hills cov {_mean_cov(b2h):.3f} "
              f"crps {b2h['crps_mean']:.4f} es {b2h['energy_score']:.4f} | "
              f"hills->bfs cov {_mean_cov(h2b):.3f} "
              f"crps {h2b['crps_mean']:.4f} es {h2b['energy_score']:.4f}",
              flush=True)
    numbers["cross_geometry"] = cross

    path = os.path.join(OUT, "crossgeom_numbers.json")
    with open(path, "w") as fh:
        json.dump(numbers, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
