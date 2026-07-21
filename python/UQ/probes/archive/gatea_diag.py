import sys, os, time
sys.path.insert(0, "build"); sys.path.insert(0, "python")
import numpy as np
from UQ.reproduce_sbli_apriori import _all_records
from UQ.datasets.sbli_baseline import SBLIBaseline

records = _all_records()
base = SBLIBaseline.configure(records["adiabatic"], with_shock=False,
                              nx=480, ny=224, x_hi=14.0, height=8.0,
                              cfl=300.0, max_iterations=40000,
                              convergence_tol=1e-6, yplus_target=0.05,
                              verbose=True, report_interval=1000)
prim = np.load("results/sbli/fields_gate_a_attached.npz")["primitive"]
base.solver.init_field(prim)
t = time.time()
rep = base.solver.solve()
print("DIAG RESULT:", rep.status, "iters", rep.iterations,
      "final rel_max", rep.final_residual, f"{time.time()-t:.0f}s", flush=True)
