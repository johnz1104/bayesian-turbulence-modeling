"""Fair-score recomputation for the separated model-form study (post-audit).

The committed score tables used the biased (M^2, diagonal-included) ensemble
CRPS/energy estimators under a "fair" docstring; the bias penalises SMALL
ensembles, i.e. the 2-3-member eigenspace corner families relative to the
24-member flow/Gaussian ensembles. This script recomputes every
mixed-ensemble-size comparison from the COMMITTED per-member records (no
solving), with both estimators:

  - biased columns must REPRODUCE the committed numbers (validation that the
    records are read exactly as the study scored them);
  - fair columns are the corrected comparison.

Reads:  UQ-RANS_research/separated_modelform/finding_numbers.json (BFS)
        UQ-RANS_research/separated_modelform/hills_numbers.json   (hills)
Writes: UQ-RANS_research/separated_modelform/fair_scores_recompute.json

Run from the repo root:
  PYTHONPATH=python python3 UQ-RANS_research/separated_modelform/recompute_fair_scores.py
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("QBTM_PY", os.path.join(_HERE, "..", "..", "python")))

from UQ import evaluation as ev

PKG = sys.argv[1] if len(sys.argv) > 1 else _HERE
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else PKG


def _crps_pair(truth, values):
    """(fair, biased) scalar-QoI CRPS of a member-value ensemble."""
    y = np.array([truth], dtype=float)
    s = np.asarray(values, dtype=float)[None, :]
    return ev.crps_ensemble(y, s), ev.crps_ensemble_biased(y, s)


def _converged(members):
    return [m for m in members if str(m.get("status", "")) == "Converged"]


def _bfs(out):
    d = json.load(open(os.path.join(PKG, "finding_numbers.json")))
    ap = d["aposteriori"]
    truth = ap["baseline"]["reattachment_dns"]
    rows = {}
    for method in ("flow", "gauss"):
        vals = [m["reattachment"] for m in _converged(ap["ensembles"][method]["members"])]
        fair, biased = _crps_pair(truth, vals)
        rows[method] = {"n_members": len(vals), "crps_fair": fair,
                        "crps_biased": biased,
                        "crps_committed": ap["ensembles"][method]["reattachment"]["crps"]}
    for delta, block in ap["eigenspace"].items():
        vals = list(block["corners"].values())
        if not vals:
            rows[f"eigenspace_{delta}"] = {"n_members": 0, "note": "no converged corners"}
            continue
        fair, biased = _crps_pair(truth, vals)
        rows[f"eigenspace_{delta}"] = {
            "n_members": len(vals), "crps_fair": fair, "crps_biased": biased,
            "crps_committed_uniform_reading": block["crps_uniform_reading"]}
    # cf stations: per-station CRPS averaged over stations (the study's cf crps
    # convention) plus the joint energy score over the station vector
    cf_truth = np.asarray(ap["ensembles"]["flow"]["cf_measured"], dtype=float)
    for method in ("flow", "gauss"):
        mem = _converged(ap["ensembles"][method]["members"])
        cf = np.asarray([m["cf_at_stations"] for m in mem], dtype=float)  # (M, S)
        per_station = np.transpose(cf)                                    # (S, M)
        rows[method]["cf_crps_fair"] = ev.crps_ensemble(cf_truth, per_station)
        rows[method]["cf_crps_biased"] = ev.crps_ensemble_biased(cf_truth, per_station)
        rows[method]["cf_energy_fair"] = ev.energy_score(
            cf_truth[None, :], cf[None, :, :])                            # (1,M,S)
        rows[method]["cf_energy_biased"] = ev.energy_score_biased(
            cf_truth[None, :], cf[None, :, :])
        rows[method]["cf_crps_committed"] = ap["ensembles"][method]["cf"]["crps"]
        rows[method]["cf_energy_committed"] = \
            ap["ensembles"][method]["cf"].get("energy_score")
    out["bfs_reattachment_and_cf"] = {"truth": truth, "rows": rows}


def _hills(out):
    d = json.load(open(os.path.join(PKG, "hills_numbers.json")))
    ap = d["aposteriori"]
    truths = ap["baseline"]["truths_dns"]
    rows = {}
    for method in ("flow", "gauss"):
        mem = _converged(ap["ensembles"][method]["members"])
        rows[method] = {"n_members": len(mem)}
        for qoi in ("separation", "reattachment", "bubble_length"):
            vals = [m[qoi] for m in mem]
            fair, biased = _crps_pair(truths[qoi], vals)
            rows[method][f"{qoi}_crps_fair"] = fair
            rows[method][f"{qoi}_crps_biased"] = biased
            rows[method][f"{qoi}_crps_committed"] = \
                ap["ensembles"][method][qoi].get("crps")
    for delta, block in ap["eigenspace"].items():
        entry = {}
        for qoi in ("separation", "reattachment", "bubble_length"):
            corners = block[qoi]["corners"]
            if not corners:
                entry[qoi] = {"n_members": 0, "note": "all corners excluded"}
                continue
            fair, biased = _crps_pair(truths[qoi], list(corners.values()))
            entry[qoi] = {"n_members": len(corners),
                          "crps_fair": fair, "crps_biased": biased}
        rows[f"eigenspace_{delta}"] = entry
    out["hills_wall_qois"] = {"truths": truths, "rows": rows}


def main():
    out = {"note": ("Post-audit score-convention recomputation from committed "
                    "member records. Biased columns reproduce the committed "
                    "values; for the sampled flow/Gaussian ensembles the fair "
                    "column (Ferro 2014) is the corrected comparison, while "
                    "for the deterministic eigenspace families the biased "
                    "column IS the exact CRPS of the finite discrete forecast "
                    "and the fair column is a sensitivity reading only.")}
    _bfs(out)
    _hills(out)
    path = os.path.join(OUT_DIR, "fair_scores_recompute.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}")
    # loud validation: biased recompute must match the committed numbers
    bfs = out["bfs_reattachment_and_cf"]["rows"]
    for method in ("flow", "gauss"):
        a, b = bfs[method]["crps_biased"], bfs[method]["crps_committed"]
        print(f"  BFS {method}: biased {a:.6f} vs committed {b:.6f} "
              f"({'OK' if abs(a - b) < 1e-9 else 'MISMATCH'}); "
              f"fair {bfs[method]['crps_fair']:.6f}")


if __name__ == "__main__":
    main()
