"""Rhie-Chow off/on adjudication probe for the separated geometries.

For the backward-facing step and the periodic hills, solve the baseline with
settings.rhie_chow_all_meshes off and on, and record convergence, the
solved-pressure odd-even (checkerboard) energy ratio, and the reattachment
QoI from the same C++ observation each configuration uses. The off/on DELTAS
are the evidence the default-off policy is adjudicated on.

Interpretation aid: the solver's own null-mode gate already activates the
Rhie-Chow face dissipation on OUTLET-FREE meshes (rcActive = !hasOutlet ||
rhie_chow_all_meshes), so on the streamwise-periodic hills the flag is a
no-op by construction and identical off/on rows CONFIRM the gate semantics
rather than a dead flag; only the outlet-bounded BFS discriminates the flag.

Run from the repo root (build/ and python/ resolved relative to this file):
  PYTHONPATH=python python3 python/rc_probe.py
Writes rc_probe_results.json into the working directory.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "build"))
sys.path.insert(0, _HERE)

import numpy as np
import rans_sst_py as rs
from UQ.datasets.bfs_baseline import BFSBaselineRANS
from UQ.datasets.hills_baseline import HillsBaselineRANS


def probe(builder, cfg, **kw):
    fwd = builder(cfg, **kw)
    theta = list(fwd["ps"].pack(fwd["rs"].SSTCoefficients()))
    result = fwd["fm"].evaluate(theta)
    status = str(result.status).split(".")[-1]
    row = {"status": status, "iterations": int(result.simple_iters),
           "reattachment": float(result.predictions[0])}
    if fwd["fm"].has_last_fields():
        p = np.asarray(fwd["fm"].last_fields()["p"], float)
        row["odd_even_energy_ratio_p"] = float(
            rs.odd_even_energy_ratio(fwd["mesh"], list(p)))
    else:
        row["odd_even_energy_ratio_p"] = float("nan")
    return row


def main():
    out = {}
    for name, builder, kw in (("bfs", BFSBaselineRANS.build_forward, {}),
                              ("hills", HillsBaselineRANS.build_forward,
                               {"case": "1p0"})):
        out[name] = {}
        for flag in (False, True):
            tag = "rc_on" if flag else "rc_off"
            print(f"[{name}] {tag} ...", flush=True)
            out[name][tag] = probe(builder, {"rhie_chow_all_meshes": flag}, **kw)
            print(f"[{name}] {tag}: {out[name][tag]}", flush=True)
        off, on = out[name]["rc_off"], out[name]["rc_on"]
        out[name]["delta"] = {
            "reattachment": on["reattachment"] - off["reattachment"],
            "odd_even_energy_ratio_p": (on["odd_even_energy_ratio_p"]
                                        - off["odd_even_energy_ratio_p"]),
            "iterations": on["iterations"] - off["iterations"],
        }
    path = os.path.join(os.getcwd(), "rc_probe_results.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
