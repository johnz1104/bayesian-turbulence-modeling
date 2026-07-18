"""Sensitivity probe for the failed adiabatic gate-B case.

The corrected adiabatic baseline's wall-pressure half-rise sits 2.93 delta0
upstream of the DNS half-rise, outside the pre-registered one-reference-length
band, while the limiter-activation diagnostic shows the corrected production
limiter nearly inactive there (1.7 percent of cells), so the offset is the
underlying SST bubble physics, not the correction. Before deciding whether the
case can rejoin the a-posteriori matrix per-case, this probe measures whether
the study's actual lever, the realizable anisotropy injection, can move the
shock position at all: the moderated eigenspace corner family (delta_B = 0.5)
is injected into the warm-started adiabatic baseline exactly as an
a-posteriori member solve, and the half-rise shift per corner is recorded.

A shift of order one reference length means calibrated corrections could
plausibly close the gate offset (the case is theta-live and can roll back into
the matrix); insensitivity means the position error is structurally outside
the anisotropy-only closure's reach, the pre-registered attribution concern,
and the frozen-mean fallback is the honest path.

Writes results/sbli/adiabatic_probe.json.

Usage: PYTHONPATH=build:python python3 python/UQ/sbli_adiabatic_probe.py
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "build"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from UQ.reproduce_sbli_aposteriori import _load_baseline
from UQ.reproduce_sbli_apriori import _all_records, _impingement_offset
from UQ.datasets.sbli_aposteriori import (cell_conditioning, corner_targets,
                                          landmarks_from_wall)

RESULTS = os.path.join(_HERE, "..", "..", "results", "sbli")


def main():
    records = _all_records()
    rec = records["adiabatic"]
    out = {"baseline_half_rise": None, "dns_half_rise": rec.shock_position(),
           "members": {}}

    # one warm baseline per member solve (independent, order-free)
    cond_base, _ = _load_baseline(records, "adiabatic", RESULTS, quick=False,
                                  member_caps=False, derived_probe=True)
    _features, b_base, mask, _basis_M = cell_conditioning(cond_base, rec)
    targets = corner_targets(b_base, mask, deltas=(0.5,))

    w0 = cond_base.wall()
    off0, x0, dns0 = _impingement_offset(w0, rec)
    out["baseline_half_rise"] = x0
    out["baseline_offset"] = off0
    print(f"baseline half-rise {x0:.3f} vs DNS {dns0:.3f} (offset {off0:.3f})",
          flush=True)

    for label, b_t in targets.items():
        base, _ = _load_baseline(records, "adiabatic", RESULTS, quick=False,
                                 member_caps=True)
        dq0 = np.zeros((b_t.shape[0], 2))
        t0 = time.time()
        base.solver.set_target_correction(np.asarray(b_t, float), dq0, True)
        rep = base.solver.solve()
        w = base.wall()
        off, x_half, _ = _impingement_offset(w, rec)
        lm = landmarks_from_wall(w)
        out["members"][label] = {
            "status": str(rep.status),
            "iterations": int(rep.iterations),
            "half_rise": x_half,
            "offset_vs_dns": off,
            "shift_from_baseline": (None if (x_half is None or x0 is None)
                                    else float(x_half - x0)),
            "landmarks": {k: lm[k] for k in ("x_s", "x_r", "shock")},
            "wall_time_s": round(time.time() - t0, 1),
        }
        print(f"[{label}] {rep.status} iters {rep.iterations} "
              f"half-rise {x_half} shift "
              f"{out['members'][label]['shift_from_baseline']}", flush=True)

    json.dump(out, open(os.path.join(RESULTS, "adiabatic_probe.json"), "w"),
              indent=1)
    print("wrote adiabatic_probe.json", flush=True)


if __name__ == "__main__":
    main()
