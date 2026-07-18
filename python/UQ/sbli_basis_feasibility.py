"""Objective-basis feasibility gates for the SBLI db leg (pre-registered).

The amendment declares the db targets as integrity-basis coefficients and
pre-registers feasibility gates checked on the corrected a-priori targets
BEFORE any training: per-fold basis rank and condition over the
interaction-region samples are reported, and the DNS db targets must be
achievable in the basis, gated at median relative reconstruction residual at
most 0.20 with the 90th percentile reported as a diagnostic. On failure the
db leg reverts to the raw-component parameterization for flow and Gaussian
identically, the reversion is stated, and the residuals are published.

This check recomputes the strain and rotation tensors with exactly the
extraction pipeline's helpers at the same strided sample points, forms Pope's
integrity basis, and evaluates UQ.discrepancy.basis_diagnostics against the
extracted db targets, restricted to the interaction-region label the
extraction itself assigns.

Writes results/sbli/basis_feasibility.json.

Usage: PYTHONPATH=build:python python3 python/UQ/sbli_basis_feasibility.py [fold ...]
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "build"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from UQ import discrepancy
from UQ.datasets import sbli_discrepancy
from UQ.reproduce_sbli_aposteriori import _load_baseline
from UQ.reproduce_sbli_apriori import _all_records

RESULTS = os.path.join(_HERE, "..", "..", "results", "sbli")
GATE_MEDIAN = 0.20


def feasibility(records, fold):
    rec = records[fold]
    base, _ = _load_baseline(records, fold, RESULTS, quick=False,
                             derived_probe=True)
    study = sbli_discrepancy.interaction_study(rec, base, stride=(8, 4))
    keep = study["region"] == "interaction"
    db = np.asarray(study["db"], float)[keep]

    # the SAME strided points and gradient/timescale conventions the
    # extraction used, this time keeping the rotation tensor for the basis
    m2 = rec.interior_mask()
    strided = np.zeros_like(m2)
    strided[::8, ::4] = True
    ii, jj = np.nonzero(m2 & strided)
    sample = base.sample_fields(rec.x[ii], rec.y[jj])
    grad_u = sbli_discrepancy._gradients_tensor(sample)
    timescale = 1.0 / np.maximum(0.09 * sample["omega"], 1e-6)
    S, W = discrepancy.strain_rotation(grad_u, timescale)
    T = discrepancy.integrity_basis(S[keep], W[keep])

    # normalize each sample's basis tensors to unit Frobenius norm: the span
    # (hence the reconstruction residual and the rank) is invariant under
    # column scaling, while the raw tensors' scales differ by orders of
    # magnitude across the shock and make the Gram numerically singular; the
    # reported condition number is that of the normalized (angle) structure
    norms = np.linalg.norm(T.reshape(T.shape[0], T.shape[1], 9), axis=2)
    T = T / np.maximum(norms, 1e-300)[:, :, None, None]

    diag = discrepancy.basis_diagnostics(T, db)
    med = float(np.median(diag["rel_residual"]))
    p90 = float(np.percentile(diag["rel_residual"], 90))
    return {
        "n_interaction_samples": int(keep.sum()),
        "rank_median": float(np.median(diag["rank"])),
        "rank_max": int(np.max(diag["rank"])),
        "cond_median": float(np.median(diag["cond"][np.isfinite(diag["cond"])])),
        "rel_residual_median": med,
        "rel_residual_p90": p90,
        "gate_median_max": GATE_MEDIAN,
        "gate_pass": bool(med <= GATE_MEDIAN),
    }


def main():
    records = _all_records()
    folds = sys.argv[1:] or ["s0.5", "s0.75", "s1.0", "s1.4", "s1.9"]
    out = {}
    for fold in folds:
        out[fold] = feasibility(records, fold)
        r = out[fold]
        print(f"[{fold}] n={r['n_interaction_samples']} rank_med "
              f"{r['rank_median']:.0f} cond_med {r['cond_median']:.1e} "
              f"rel_resid med {r['rel_residual_median']:.4f} "
              f"p90 {r['rel_residual_p90']:.4f} "
              f"gate {'PASS' if r['gate_pass'] else 'FAIL'}", flush=True)
    passing = [f for f in out if out[f]["gate_pass"]]
    out["_gate"] = {
        "rule": "median rel reconstruction residual <= 0.20 per fold",
        "all_pass": len(passing) == len(folds),
        "passing": passing,
    }
    json.dump(out, open(os.path.join(RESULTS, "basis_feasibility.json"), "w"),
              indent=1)
    print("wrote basis_feasibility.json", flush=True)


if __name__ == "__main__":
    main()
