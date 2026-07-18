"""Limiter-activation fractions and spatial maps for the SBLI baselines.

The pre-registration amendment commits every corrected interaction baseline to
reporting where the SST omega-production limiter actually activates, so the
localization to shock feet and separated shear layers is a measured result
rather than a premise. This diagnostic recomputes the activation criterion
from the persisted converged primitive state (results/sbli/fields_<case>.npz,
columns rho, u, v, p, k, omega on the production mesh):

    active  <=>  nu_t S^2 > 10 betaStar k omega

with nu_t = a1 k / max(a1 omega, S F2), S the deviatoric strain magnitude from
structured central differences of the cell-center velocities, F2 the standard
SST blend evaluated with the wall distance (the flat-plate wall at y = 0) and
a constant molecular viscosity from the record's inlet Reynolds number (the
Sutherland variation moves only the sublayer argument of F2 and does not
change the activation locus; stated, not hidden). The recomputation convention
means the maps are diagnostics of the converged state, not solver-internal
flags.

Writes results/sbli/limiter_activation.json (fractions per case, overall and
in the interaction band |x*| <= 5) and one map PNG per case under
results/sbli/figures/.

Usage: PYTHONPATH=build:python python3 python/UQ/sbli_limiter_activation.py [case ...]
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "build"))
sys.path.insert(0, os.path.join(_HERE, ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from UQ.datasets.sbli_interaction import SBLIInteractionDNS
from UQ.datasets.sbli_baseline import SBLIBaseline

RESULTS = os.path.join(_HERE, "..", "..", "results", "sbli")
A1, BETA_STAR = 0.31, 0.09


def _load_record(case, root=None):
    if case == "adiabatic":
        return SBLIInteractionDNS.adiabatic(root)
    return SBLIInteractionDNS.wall_thermal(case.lstrip("s"), root)


def activation(case):
    rec = _load_record(case)
    base = SBLIBaseline.configure(rec, with_shock=True, nx=480, ny=224,
                                  x_hi=14.0, height=8.0, cfl=300.0,
                                  max_iterations=1, convergence_tol=1e-6,
                                  yplus_target=0.05)
    cc = np.asarray(base.mesh.cell_centers())
    prim = np.load(os.path.join(RESULTS, f"fields_{case}.npz"))["primitive"]
    assert prim.shape[0] == cc.shape[0], "cache does not match the mesh"

    # structured (ny, nx) index grid from the cell-center coordinates
    xs = np.unique(np.round(cc[:, 0], 12))
    ys = np.unique(np.round(cc[:, 1], 12))
    nx, ny = xs.size, ys.size
    ix = np.searchsorted(xs, np.round(cc[:, 0], 12))
    iy = np.searchsorted(ys, np.round(cc[:, 1], 12))
    grid = np.full((ny, nx), -1, dtype=int)
    grid[iy, ix] = np.arange(cc.shape[0])
    assert (grid >= 0).all(), "mesh is not a full structured grid"

    def G(col):
        return prim[grid, col]

    u, v = G(1), G(2)
    k = np.maximum(G(4), 0.0)
    om = np.maximum(G(5), 1e-20)

    dudy, dudx = np.gradient(u, ys, xs)
    dvdy, dvdx = np.gradient(v, ys, xs)
    div = dudx + dvdy
    # deviatoric strain magnitude, the solver's production convention
    sxx = dudx - div / 3.0
    syy = dvdy - div / 3.0
    szz = -div / 3.0
    sxy = 0.5 * (dudy + dvdx)
    S = np.sqrt(2.0 * (sxx**2 + syy**2 + szz**2 + 2.0 * sxy**2))

    y = np.maximum(ys[:, None] * np.ones((ny, nx)), 1e-12)
    # solver-units molecular kinematic viscosity at the free stream (the F2
    # sublayer argument only; see the module docstring)
    nu = base.units.mu_inf / base.units.rho_inf
    arg2 = np.maximum(2.0 * np.sqrt(k) / (BETA_STAR * om * y),
                      500.0 * nu / (y**2 * om))
    F2 = np.tanh(arg2**2)
    nut = A1 * k / np.maximum(A1 * om, S * F2)

    active = (nut * S**2) > (10.0 * BETA_STAR * k * om)
    x_star = xs / base.units.delta0
    if hasattr(rec, "x"):
        x_star = x_star + float(rec.x[0])
    elif hasattr(rec, "meta") and "x_lo" in rec.meta:
        x_star = x_star + float(rec.meta["x_lo"])
    band = (np.abs(x_star) <= 5.0)[None, :] * np.ones((ny, nx), bool)

    frac_all = float(np.mean(active))
    frac_band = float(np.mean(active[band])) if band.any() else float("nan")

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.pcolormesh(x_star, ys / base.units.delta0, active.astype(float),
                  cmap="Reds", vmin=0, vmax=1, shading="auto")
    ax.set_xlabel("x*")
    ax.set_ylabel("y / delta0")
    ax.set_ylim(0, 4)
    ax.set_title(f"omega-production limiter activation, {case} "
                 f"(overall {frac_all:.3f}, |x*|<=5 band {frac_band:.3f})")
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures",
                             f"limiter_activation_{case}.png"), dpi=140)
    plt.close(fig)
    return {"fraction_overall": frac_all, "fraction_band_x5": frac_band}


def main():
    cases = sys.argv[1:] or ["adiabatic", "s0.5", "s0.75", "s1.0"]
    out = {}
    for case in cases:
        path = os.path.join(RESULTS, f"fields_{case}.npz")
        if not os.path.isfile(path):
            print(f"[{case}] no fields cache; skipped", flush=True)
            continue
        out[case] = activation(case)
        print(f"[{case}] activation overall {out[case]['fraction_overall']:.4f} "
              f"band {out[case]['fraction_band_x5']:.4f}", flush=True)
    path = os.path.join(RESULTS, "limiter_activation.json")
    existing = json.load(open(path)) if os.path.isfile(path) else {}
    existing.update(out)
    json.dump(existing, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
