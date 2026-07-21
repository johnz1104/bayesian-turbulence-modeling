"""Three-solve pilot for the repaired injection contract (2026-07-20 plan).

Runs, in order, on the s1.0 fold's regenerated targets:

  control   the zero-discrepancy control: db = 0 with the absolute target
            equal to the conditioned baseline, which under the
            stored-discrepancy contract must reproduce the uninjected
            baseline trajectory exactly (the turbulent identity, here at
            the production configuration)
  member    one registered running-k flow member (index 0), full budget,
            early abort off, verbose cell budgets on rejection
  corner    the moderated three-corner member 3C at delta_B = 0.5

The pilot STOPS after reporting; no matrix launches on any outcome (the
reviewer adjudicates). Writes results/sbli/aposteriori/pilot.json.

Usage: PYTHONPATH=build:python python3 python/UQ/sbli_member_pilot.py
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "build"))
sys.path.insert(0, os.path.join(_HERE, ".."))
os.environ.setdefault("QBTM_SBLI_VERBOSE", "1")

from UQ.reproduce_sbli_apriori import _all_records, _configure, _fields_path
from UQ.reproduce_sbli_aposteriori import (_targets_path, dq_to_solver_units,
                                           MEMBER_TOL)

RESULTS = os.path.join(_HERE, "..", "..", "results", "sbli")
FOLD = "s1.0"
BUDGET = 45000


def _solve(records, db, b_abs, dq_dim, mask, label):
    base = _configure(records[FOLD], False, with_shock=True,
                      max_iterations=BUDGET, convergence_tol=MEMBER_TOL)
    prim = np.load(_fields_path(RESULTS, FOLD))["primitive"]
    base.solver.init_field(prim)
    base.solver.set_target_correction(db, b_abs, dq_dim, True, mask=mask)
    t0 = time.time()
    rep = base.solver.solve()
    out = {"status": str(rep.status), "iterations": int(rep.iterations),
           "final_residual": float(rep.final_residual),
           "wall_time_s": round(time.time() - t0, 1)}
    print(f"[pilot {label}] {rep.status} iters {rep.iterations} "
          f"rel {rep.final_residual:.3e} {out['wall_time_s']}s", flush=True)
    return out, base


def main():
    records = _all_records()
    tg = np.load(_targets_path(RESULTS, FOLD, "flow"))
    b_base = np.asarray(tg["b_base"], dtype=float)
    mask = np.asarray(tg["mask"], bool)
    n = b_base.shape[0]
    out = {"fold": FOLD, "budget": BUDGET, "tol": MEMBER_TOL}

    # 1. zero-discrepancy control: must match the uninjected trajectory
    base_ref = _configure(records[FOLD], False, with_shock=True,
                          max_iterations=2000, convergence_tol=1e-30)
    prim = np.load(_fields_path(RESULTS, FOLD))["primitive"]
    base_ref.solver.init_field(prim)
    base_ref.solver.solve()
    f_ref = base_ref.solver.fields()

    ctrl = _configure(records[FOLD], False, with_shock=True,
                      max_iterations=2000, convergence_tol=1e-30)
    ctrl.solver.init_field(prim)
    ctrl.solver.set_target_correction(np.zeros((n, 3, 3)), b_base,
                                      np.zeros((n, 2)), True, mask=mask)
    ctrl.solver.solve()
    f_inj = ctrl.solver.fields()
    diffs = {q: float(np.max(np.abs(np.asarray(f_ref[q])
                                    - np.asarray(f_inj[q]))))
             for q in ("rho", "u", "v", "p", "k", "omega")}
    out["control"] = {"max_abs_diff": diffs,
                      "exact": bool(all(v == 0.0 for v in diffs.values()))}
    print(f"[pilot control] max diffs {diffs} exact={out['control']['exact']}",
          flush=True)

    # 2. one registered running-k flow member
    b_t = np.asarray(tg["b"][0], dtype=float)
    dq = np.asarray(tg["dq"][0], dtype=float)
    # units come from any configured baseline; reuse the control's
    dq_dim = dq_to_solver_units(dq[None], records[FOLD], ctrl.units)[0]
    out["member_flow_0"], _ = _solve(records, b_t - b_base, b_t, dq_dim,
                                     mask, "member flow 0")

    # 3. the moderated corner 3C at delta 0.5
    tgc = np.load(_targets_path(RESULTS, FOLD, "corners"))
    b_c = np.asarray(tgc["3C_d0.5"], dtype=float)
    cb_base = np.asarray(tgc["b_base"], dtype=float)
    cmask = np.asarray(tgc["mask"], bool)
    out["corner_3C_d0.5"], _ = _solve(records, b_c - cb_base, b_c,
                                      np.zeros((n, 2)), cmask,
                                      "corner 3C d0.5")

    path = os.path.join(RESULTS, "aposteriori", "pilot.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
