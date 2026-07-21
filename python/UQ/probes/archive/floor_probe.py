"""Same s1.0 flow member with the injected correction zeroed below
y* = 0.1 (the wedge diagnosis located the vacuum collapse at the
y* = 0.05 edge row, y+ ~ 16, buffer layer). Converge = the mask floor is
the destabilizer; stall = the runaway lives higher up too."""
import os
import sys
import time

import numpy as np

os.chdir("/Users/johnz11/Downloads/CFD/btm-channel-dns-pipeline")
sys.path.insert(0, "build")
sys.path.insert(0, "python")
os.environ["QBTM_SBLI_VERBOSE"] = "1"

from UQ.reproduce_sbli_apriori import _all_records, _configure, _fields_path
from UQ.reproduce_sbli_aposteriori import (_targets_path, dq_to_solver_units,
                                           _load_baseline, cell_conditioning)

records = _all_records()
R = "results/sbli"

cond, _ = _load_baseline(records, "s1.0", R, quick=False, derived_probe=True)
_f, b_base, mask, _M, _S = cell_conditioning(cond, records["s1.0"],
                                             m_t_from_fields=True)
cc = np.asarray(cond.mesh.cell_centers())
y_star = cc[:, 1] / cond.units.delta0

tg = np.load(_targets_path(R, "s1.0", "flow"))
b_t = np.asarray(tg["b"][0], dtype=float)
dq = np.asarray(tg["dq"][0], dtype=float)
low = y_star < 0.1
b_t[low] = b_base[low]
dq[low] = 0.0
print(f"zeroed correction on {int(low.sum())} cells below y*=0.1 "
      f"(mask had {int((mask & low).sum())} of them active)", flush=True)

base = _configure(records["s1.0"], False, with_shock=True,
                  max_iterations=45000, convergence_tol=1e-3)
prim = np.load(_fields_path(R, "s1.0"))["primitive"]
base.solver.init_field(prim)
st_dq = dq_to_solver_units(dq[None], records["s1.0"], base.units)[0]
base.solver.set_target_correction(b_t, st_dq, True)
t0 = time.time()
rep = base.solver.solve()
print(f"RESULT {rep.status} iters {rep.iterations} "
      f"rel {rep.final_residual:.3e} {time.time()-t0:.0f}s", flush=True)
