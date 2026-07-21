"""A/B probe: one s1.0 flow member target, solved with and without the
solver-side injection mask, verbose per-equation residuals, abort off.
Decides whether the phase-3 face-gated mask is what stalls every member."""
import os
import sys
import time

import numpy as np

os.chdir("/Users/johnz11/Downloads/CFD/btm-channel-dns-pipeline")
sys.path.insert(0, "build")
sys.path.insert(0, "python")

from UQ.reproduce_sbli_aposteriori import (_load_baseline, _targets_path,
                                           dq_to_solver_units, MEMBER_TOL)
from UQ.reproduce_sbli_apriori import _all_records

records = _all_records()
RESULTS = "results/sbli"
tg = np.load(_targets_path(RESULTS, "s1.0", "flow"))
b_t = np.asarray(tg["b"][0], dtype=float)
dq = np.asarray(tg["dq"][0], dtype=float)
mask = np.asarray(tg["mask"], bool)

for label, use_mask in (("unmasked", False), ("masked", True)):
    os.environ["QBTM_SBLI_VERBOSE"] = "1"
    base, _ = _load_baseline(records, "s1.0", RESULTS, quick=False,
                             member_caps=True)
    st_dq = dq_to_solver_units(dq[None], records["s1.0"], base.units)[0]
    if use_mask:
        base.solver.set_target_correction(b_t, st_dq, True, mask=mask)
    else:
        base.solver.set_target_correction(b_t, st_dq, True)
    t0 = time.time()
    rep = base.solver.solve()
    print(f"[{label}] {rep.status} iters {rep.iterations} "
          f"final_rel {rep.final_residual:.3e} {time.time()-t0:.0f}s",
          flush=True)
