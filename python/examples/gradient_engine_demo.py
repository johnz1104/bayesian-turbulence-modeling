"""
Rung-1 true-model gradient — engine demo / reproducible artifact.

Compares the three gradient engines on the inlet-driven channel and exercises the consumer
wiring, writing a JSON summary.  See DECISION_RECORD.md §4 for the full investigation.

  warm-FD  (eta_jacobian_warm_fd)          — production default; == cold full FD, ~13x cheaper
  frozen-P (eta_jacobian_tangent)          — direction-faithful, ~30% magnitude-biased
  + active-subspace Gram matrix and KOH ∂L/∂θ composition from the warm-FD gradient

  python examples/gradient_engine_demo.py            # full
  python examples/gradient_engine_demo.py --quick    # coarser mesh, fewer points
  python examples/gradient_engine_demo.py -o results/gradient_engine
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "build"), str(_REPO / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rans_sst_py as rs  # noqa: E402
from gradient_inference import (  # noqa: E402
    WarmFDForwardModel, gradient_dataset, koh_loglik_gradient_theta,
)
from identifiability import active_subspace  # noqa: E402
from bayesian_inference import Prior, KOHLikelihood  # noqa: E402

NAMES = ["sigma_k1", "sigma_w1", "beta1", "alpha1", "sigma_k2", "sigma_w2",
         "beta2", "alpha2", "betaStar", "a1", "kappa"]
NONLINEAR = (8, 9)
STRONG = (2, 8, 9)


def channel(quick):
    nx, ny, mi = (20, 14, 1500) if quick else (32, 24, 4000)
    h, Lx, Ub, Re_b = 1.0, 10.0, 1.0, 6800.0
    nu = Ub * h / Re_b
    Cf = 0.073 * Re_b ** -0.25
    mesh = rs.Mesh.make_channel_2d(nx=nx, ny=ny, Lx=Lx, Ly=2.0 * h, Re=Re_b, yPlusTarget=1.0)
    mesh.compute_wall_distance()
    Tu = 0.05; kIn = 1.5 * (Ub * Tu) ** 2; omIn = kIn / (nu * 100.0)
    bcs = rs.FlowBoundaryConditions.channel_defaults(mesh, Ub, kIn, omIn)
    obs = rs.ObservationOperator()
    xs = np.linspace(3.0, 7.0, 4)
    for x in xs:
        obs.add_skin_friction("bottom", rs.Vec3(float(x), 0, 0), Cf, 0.05 * Cf, Ub)
    s = rs.SolverSettings()
    s.max_iterations = mi; s.convergence_tol = 1e-6; s.verbose = False
    s.alpha_u = 0.7; s.alpha_p = 0.5
    return (mesh, obs, bcs, nu, s, rs.Vec3(Ub, 0, 0), 0.0, kIn, omIn), xs, Cf


def main():
    ap = argparse.ArgumentParser(description="Rung-1 gradient engine demo")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("-o", "--out-dir", default="results/gradient_engine")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cargs, xs, Cf = channel(args.quick)
    mesh = cargs[0]
    sens = rs.ParameterSensitivity(*cargs)
    th = list(np.array(rs.InferenceParameterSet.all11().pack(rs.SSTCoefficients())))
    sens.solve_state(th)
    warm_cap = 300 if args.quick else 400

    # --- warm-FD (default) ----------------------------------------------------
    t0 = time.perf_counter()
    rw = sens.eta_jacobian_warm_fd(th, warm_max_iter=warm_cap, warm_tol=1e-5)
    t_warm = time.perf_counter() - t0
    Jw = np.asarray(rw.d_obs_d_theta)

    # --- frozen-pressure tangent ---------------------------------------------
    rt = sens.eta_jacobian_tangent(th, krylov_tol=1e-9, max_iter=5000, fd_step=1e-6)
    Jt = np.asarray(rt.d_obs_d_theta)

    # --- cold full-FD reference on the strong columns (exactness check) -------
    def cold_eta(tv):
        s = rs.ParameterSensitivity(*channel(args.quick)[0]); s.solve_state(list(tv))
        return np.asarray(s.observe(list(tv)), float)
    th0 = np.asarray(th, float)
    t0 = time.perf_counter()
    warm_vs_cold, tangent_ratio = {}, {}
    for j in STRONG:
        hr = 1e-5 if j in NONLINEAR else 5e-4; hj = max(hr * abs(th0[j]), 1e-7)
        e = np.zeros(11); e[j] = hj
        col = (cold_eta(th0 + e) - cold_eta(th0 - e)) / (2 * hj)
        warm_vs_cold[NAMES[j]] = float(np.linalg.norm(Jw[:, j] - col) / max(np.linalg.norm(col), 1e-30))
        tangent_ratio[NAMES[j]] = float(np.linalg.norm(Jt[:, j]) / max(np.linalg.norm(col), 1e-30))
    t_cold6 = time.perf_counter() - t0

    # --- active subspace from a small warm-FD gradient set --------------------
    fm = WarmFDForwardModel(sens, sens.n_obs(), warm_max_iter=warm_cap, warm_tol=1e-5)
    ps = rs.InferenceParameterSet.all11()
    lo = np.array(ps.lower_bounds()); hi = np.array(ps.upper_bounds())
    prior = Prior(means=np.array(th), stds=0.3 * (hi - lo), lower=lo, upper=hi)
    rng = np.random.default_rng(0)
    npts = 3 if args.quick else 5
    thetas = np.array([th] + [np.clip(th0 + rng.uniform(-1, 1, 11) * 0.05 * (hi - lo),
                                      lo + 1e-3, hi - 1e-3) for _ in range(npts - 1)])
    G, valid = gradient_dataset(fm, prior, thetas, out / "grad_cache.npz", verbose=False)
    vals, _vecs, frac = active_subspace(G[valid])

    # --- KOH log-likelihood θ-gradient composition ----------------------------
    koh = KOHLikelihood(xs, np.full(len(xs), Cf), np.full(len(xs), 0.05 * Cf), mode="physical_gp")
    g_theta, g_s, g_l = koh_loglik_gradient_theta(koh, fm.eta(th), Jw, -3.0, 0.2)

    summary = {
        "engine": "warm-FD (default) vs frozen-P tangent vs cold full-FD",
        "warm_fd_time_s": round(t_warm, 2),
        "cold_fd_6solves_time_s": round(t_cold6, 2),
        "warm_fd_vs_cold_relerr_strong": warm_vs_cold,
        "frozen_pressure_tangent_magnitude_ratio_strong": tangent_ratio,
        "frozen_pressure_tangent_converged": bool(all(rt.krylov_converged)),
        "warm_fd_solve_iters": int(rw.n_residual_evals),
        "active_subspace_eigval_fraction": [round(float(x), 4) for x in frac],
        "active_subspace_n_valid_grads": int(valid.sum()),
        "koh_dL_dtheta_strong": {NAMES[j]: float(g_theta[j]) for j in STRONG},
        "koh_dL_dlog_sigma_delta": float(g_s),
        "notes": "warm-FD matches cold full-FD to the FD floor; frozen-P tangent is "
                 "direction-faithful but ~0.77x magnitude (no dp/dθ). See DECISION_RECORD.md.",
    }
    (out / "gradient_engine_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n[gradient_engine_demo] wrote {out/'gradient_engine_summary.json'}")


if __name__ == "__main__":
    main()
