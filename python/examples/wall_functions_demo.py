"""
PHASE 7 — Wall-functions / coarse-mesh demo.

Compares three runs side-by-side on the compressible channel:

    1. Fine resolved-LES        (40x30, y+_target=1,  use_wall_functions=False)
    2. Coarse resolved-LES      (24x14, y+_target=30, use_wall_functions=False)
    3. Coarse + wall functions  (24x14, y+_target=30, use_wall_functions=True)

For each run we report convergence, y+ statistics, Cf along the bottom wall,
and (for runs 2 & 3) the Cf relative error against run 1 at x ≈ 5h.

Outputs (under ``--save-dir``):
    wall_functions_summary.json
    cf_overlay.png
    y_plus_summary.png

Usage:
    python3 wall_functions_demo.py
    python3 wall_functions_demo.py --quick
    python3 wall_functions_demo.py -o results/wall_fn
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_BUILD_DIR  = _SCRIPT_DIR.parent.parent / "build"
_PYTHON_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PYTHON_DIR))

try:
    import rans_sst_py as rs
except ImportError:
    print("ERROR: rans_sst_py not built. Run: cmake --build build")
    sys.exit(1)

from wall_diagnostics import (
    y_plus_first_cell, cf_along_wall, coarse_vs_fine_summary,
)


def build_and_run(name: str, *, nx: int, ny: int, yPlusTarget: float,
                   use_wall_functions: bool, max_iters: int,
                   Ma: float = 0.1, Lx: float = 10.0, H: float = 1.0):
    eos    = rs.IdealGasEOS()
    T_in   = 300.0
    p_ref  = 101325.0
    rho_in = eos.density(p_ref, T_in)
    mu_in  = eos.viscosity(T_in)
    Uin    = Ma * eos.sound_speed(T_in)
    Re     = rho_in * Uin * H / mu_in
    nu_in  = mu_in / rho_in

    mesh = rs.Mesh.make_channel_2d(nx, ny, Lx, H, Re=Re,
                                    yPlusTarget=yPlusTarget)
    mesh.compute_wall_distance()
    kIn  = 1.5 * (Uin * 0.05) ** 2
    omIn = kIn / (nu_in * 100.0)
    bcs  = rs.CompressibleBoundaryConditions.channel_defaults(
        mesh, Uin, T_in, p_ref, kIn, omIn)

    obs = rs.ObservationOperator()
    obs.add_skin_friction(
        wall_patch="bottom", location=rs.Vec3(5.0, 0, 0),
        cf_obs=0.005, sigma=0.001, ref_vel=Uin)

    settings = rs.SolverSettings()
    settings.max_iterations      = max_iters
    settings.convergence_tol     = 1e-3
    settings.alpha_u             = 0.5
    settings.alpha_p             = 0.2
    settings.alpha_t             = 0.7
    settings.alpha_k             = 0.4
    settings.alpha_omega         = 0.4
    settings.inner_iterations    = 200
    settings.turb_start_iter     = 50
    settings.turb_update_interval = 2
    settings.divergence_limit    = 1e10
    settings.verbose             = False
    settings.use_wall_functions  = use_wall_functions

    param_set = rs.InferenceParameterSet.a1_betaStar()
    fm = rs.CompressibleForwardModel(
        mesh=mesh, param_set=param_set, obs_op=obs, bcs=bcs, eos=eos,
        settings=settings, u_init=rs.Vec3(Uin, 0, 0),
        p_init=p_ref, T_init=T_in, k_init=kIn, omega_init=omIn)

    t0 = time.time()
    result = fm.evaluate(list(param_set.pack(rs.SSTCoefficients())))
    elapsed = time.time() - t0
    fields = fm.last_fields() if fm.has_last_fields() else None

    out = {
        "name":              name,
        "nx":                nx, "ny": ny, "n_cells": mesh.n_cells(),
        "yPlusTarget":       yPlusTarget,
        "use_wall_functions": use_wall_functions,
        "status":            str(result.status),
        "simple_iters":      int(result.simple_iters),
        "elapsed_s":         elapsed,
        "Cf_obs":            list(result.predictions),
        "fields_available":  fields is not None,
    }
    if fields is not None:
        yplus = y_plus_first_cell(mesh, fields, nu_in, wall_patch="bottom")
        cf    = cf_along_wall(mesh, fields, nu_in, "bottom", ref_vel=Uin)
        out["y_plus"] = {"min":  yplus["min"],  "max":  yplus["max"],
                         "mean": yplus["mean"]}
        out["cf"]     = {"x":  cf["x"].tolist(),
                          "cf": cf["cf"].tolist(),
                          "max": cf["max"], "mean": cf["mean"]}
        out["nu"] = nu_in
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Tiny iteration budgets for fast smoke")
    ap.add_argument("-o", "--save-dir", type=Path, default=None)
    args = ap.parse_args()

    save_dir = args.save_dir or (_PYTHON_DIR.parent / "results"
                                  / "wall_functions_demo")
    save_dir.mkdir(parents=True, exist_ok=True)

    iters_fine   = 1500 if args.quick else 3000
    iters_coarse = 2000 if args.quick else 4000

    print("Wall-functions / coarse-mesh demo (PHASE 7)")
    print(f"  save_dir: {save_dir}")

    print("\n[1/3] Fine resolved (40x30, y+_target=1, wf=OFF)...")
    fine = build_and_run("fine_resolved", nx=40, ny=30,
                          yPlusTarget=1.0, use_wall_functions=False,
                          max_iters=iters_fine)
    print("  status:", fine["status"], " iters:", fine["simple_iters"],
          " y+:", fine.get("y_plus"))

    print("\n[2/3] Coarse resolved (24x14, y+_target=30, wf=OFF)...")
    coarse_res = build_and_run("coarse_resolved", nx=24, ny=14,
                                yPlusTarget=30.0, use_wall_functions=False,
                                max_iters=iters_coarse)
    print("  status:", coarse_res["status"], " iters:", coarse_res["simple_iters"],
          " y+:", coarse_res.get("y_plus"))

    print("\n[3/3] Coarse + wall functions (24x14, y+_target=30, wf=ON)...")
    coarse_wf = build_and_run("coarse_wall_functions", nx=24, ny=14,
                                yPlusTarget=30.0, use_wall_functions=True,
                                max_iters=iters_coarse)
    print("  status:", coarse_wf["status"], " iters:", coarse_wf["simple_iters"],
          " y+:", coarse_wf.get("y_plus"))

    summary = {"runs": [fine, coarse_res, coarse_wf]}
    if all(r["fields_available"] for r in summary["runs"]):
        summary["coarse_resolved_vs_fine"] = coarse_vs_fine_summary(
            coarse_res, fine)
        summary["coarse_wf_vs_fine"]       = coarse_vs_fine_summary(
            coarse_wf, fine)

    json_path = save_dir / "wall_functions_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  Summary -> {json_path}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for r in summary["runs"]:
            if not r["fields_available"]:  continue
            ax.plot(r["cf"]["x"], r["cf"]["cf"], "-o", ms=3, label=r["name"])
        ax.set_xlabel("x"); ax.set_ylabel("Cf"); ax.legend()
        ax.set_title("Skin friction comparison")
        fig.tight_layout()
        fig.savefig(save_dir / "cf_overlay.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        names  = [r["name"] for r in summary["runs"] if r["fields_available"]]
        means  = [r["y_plus"]["mean"] for r in summary["runs"] if r["fields_available"]]
        maxes  = [r["y_plus"]["max"]  for r in summary["runs"] if r["fields_available"]]
        x = np.arange(len(names))
        ax.bar(x - 0.2, means, width=0.4, label="y+_mean")
        ax.bar(x + 0.2, maxes, width=0.4, label="y+_max")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=10)
        ax.set_ylabel("y+"); ax.legend()
        ax.set_title("y+ at first cell — fine vs coarse mesh")
        fig.tight_layout()
        fig.savefig(save_dir / "y_plus_summary.png", dpi=150)
        plt.close(fig)

        print(f"  Plots   -> {save_dir}")
    except ImportError:
        print("(matplotlib missing; plots skipped)")


if __name__ == "__main__":
    main()
