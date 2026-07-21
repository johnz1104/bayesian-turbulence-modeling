"""Short verbose member solve to locate the persistent admissibility
rejections: one s1.0 flow member, unmasked, 6000-iteration budget."""
import os
import sys
import time

import numpy as np

os.chdir("/Users/johnz11/Downloads/CFD/btm-channel-dns-pipeline")
sys.path.insert(0, "build")
sys.path.insert(0, "python")
os.environ["QBTM_SBLI_VERBOSE"] = "1"

from UQ.reproduce_sbli_apriori import _all_records, _configure, _fields_path
from UQ.reproduce_sbli_aposteriori import _targets_path, dq_to_solver_units

records = _all_records()
R = "results/sbli"
base = _configure(records["s1.0"], False, with_shock=True,
                  max_iterations=6000, convergence_tol=1e-3)
prim = np.load(_fields_path(R, "s1.0"))["primitive"]
base.solver.init_field(prim)
tg = np.load(_targets_path(R, "s1.0", "flow"))
b_t = np.asarray(tg["b"][0], dtype=float)
dq = np.asarray(tg["dq"][0], dtype=float)
st_dq = dq_to_solver_units(dq[None], records["s1.0"], base.units)[0]
base.solver.set_target_correction(b_t, st_dq, True)
t0 = time.time()
rep = base.solver.solve()
print(f"RESULT {rep.status} iters {rep.iterations} "
      f"rel {rep.final_residual:.3e} {time.time()-t0:.0f}s", flush=True)
