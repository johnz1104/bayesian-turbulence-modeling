"""
Compressible channel-flow example (PHASE 1).

Builds a small compressible channel case from Python and exercises the new
``CompressibleForwardModel`` bindings end to end:

  1. Construct mesh and ideal-gas EOS.
  2. Build a default observation operator (skin friction at three stations).
  3. Build a compressible-channel BC set.
  4. Run ``CompressibleForwardModel.evaluate(theta)`` at the Menter SST defaults.
  5. Print convergence status, Cf, max Mach number, T extrema, and residual
     summary.

Usage:
    python3 compressible_channel.py                  # Ma=0.1, ~16x12 mesh
    python3 compressible_channel.py --ma 0.3 --nx 32 --ny 24
    python3 compressible_channel.py --report results/compressible_channel
"""

from __future__ import annotations

import argparse
import json
import sys
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


def build_case(nx: int, ny: int, Ma: float, Lx: float = 10.0, H: float = 1.0,
               turb_intensity: float = 0.05):
    """Construct the mesh, EOS, BCs, and forward model for a Ma=Ma channel."""
    eos    = rs.IdealGasEOS()
    T_in   = 300.0
    p_ref  = 101325.0
    rho_in = eos.density(p_ref, T_in)
    mu_in  = eos.viscosity(T_in)
    a_in   = eos.sound_speed(T_in)
    Uin    = Ma * a_in
    nu_in  = mu_in / rho_in
    Re     = rho_in * Uin * H / mu_in

    mesh = rs.Mesh.make_channel_2d(nx, ny, Lx, H, Re=Re, yPlusTarget=1.0)
    mesh.compute_wall_distance()

    kIn  = 1.5 * (Uin * turb_intensity) ** 2
    omIn = kIn / (nu_in * 100.0)

    bcs = rs.CompressibleBoundaryConditions.channel_defaults(
        mesh, Uin, T_in, p_ref, kIn, omIn)

    obs = rs.ObservationOperator()
    for x_loc in (0.25 * Lx, 0.5 * Lx, 0.75 * Lx):
        obs.add_skin_friction(
            wall_patch="bottom", location=rs.Vec3(x_loc, 0.0, 0.0),
            cf_obs=0.005, sigma=0.001, ref_vel=Uin)

    settings = rs.SolverSettings()
    settings.max_iterations      = 2000
    settings.convergence_tol     = 1e-3
    settings.divergence_limit    = 1e10
    settings.alpha_u             = 0.5
    settings.alpha_p             = 0.2
    settings.alpha_t             = 0.7
    settings.alpha_k             = 0.4
    settings.alpha_omega         = 0.4
    settings.inner_iterations    = 200
    settings.inner_tolerance     = 1e-4
    settings.turb_start_iter     = 50
    settings.turb_update_interval = 2
    settings.verbose             = False
    settings.report_interval     = 200

    param_set = rs.InferenceParameterSet.a1_betaStar()
    fm = rs.CompressibleForwardModel(
        mesh=mesh, param_set=param_set, obs_op=obs, bcs=bcs, eos=eos,
        settings=settings, u_init=rs.Vec3(Uin, 0, 0),
        p_init=p_ref, T_init=T_in, k_init=kIn, omega_init=omIn)

    return {
        "mesh":      mesh,
        "eos":       eos,
        "bcs":       bcs,
        "obs":       obs,
        "settings":  settings,
        "param_set": param_set,
        "forward":   fm,
        "Uin":       Uin,
        "T_in":      T_in,
        "p_ref":     p_ref,
        "Ma":        Ma,
        "Re":        Re,
        "rho_in":    rho_in,
        "mu_in":     mu_in,
    }


def summarize(case: dict) -> dict:
    """Run the forward model at Menter defaults and gather diagnostics."""
    fm        = case["forward"]
    eos       = case["eos"]
    param_set = case["param_set"]

    theta_def = list(param_set.pack(rs.SSTCoefficients()))
    result    = fm.evaluate(theta_def)
    fields    = fm.last_fields()

    U   = fields["U"]
    p   = fields["p"]
    T   = fields["T"]
    rho = fields["rho"]
    nuT = fields["nuT"]

    Umag    = np.linalg.norm(U, axis=1)
    a_local = np.sqrt(eos.gamma * eos.R * T)
    Ma_local = Umag / a_local

    summary = {
        "status":       str(result.status),
        "log_lik":      float(result.log_lik),
        "simple_iters": int(result.simple_iters),
        "predictions":  list(result.predictions),
        "Ma_inlet":     float(case["Ma"]),
        "Re":           float(case["Re"]),
        "U_max":        float(np.max(Umag)),
        "U_mean":       float(np.mean(Umag)),
        "p_min":        float(np.min(p)),
        "p_max":        float(np.max(p)),
        "T_min":        float(np.min(T)),
        "T_max":        float(np.max(T)),
        "rho_min":      float(np.min(rho)),
        "rho_max":      float(np.max(rho)),
        "Ma_max":       float(np.max(Ma_local)),
        "nuT_max":      float(np.max(nuT)),
    }
    return summary


def print_summary(case: dict, summary: dict) -> None:
    print(f"\nCompressible channel (Ma={case['Ma']:.2f}, Re={case['Re']:.0f})")
    print(f"  Mesh: {case['mesh'].n_cells()} cells")
    print(f"  Status:        {summary['status']}")
    print(f"  SIMPLE iters:  {summary['simple_iters']}")
    print(f"  log L:         {summary['log_lik']:+.3f}")
    print(f"  U_max:         {summary['U_max']:.2f} m/s  "
          f"(U_inlet = {case['Uin']:.2f})")
    print(f"  T range:       [{summary['T_min']:.2f}, {summary['T_max']:.2f}] K")
    print(f"  p range:       [{summary['p_min']:.0f}, {summary['p_max']:.0f}] Pa")
    print(f"  rho range:     [{summary['rho_min']:.4f}, "
          f"{summary['rho_max']:.4f}] kg/m^3")
    print(f"  Ma_max:        {summary['Ma_max']:.4f}")
    print(f"  Cf samples:    {[f'{v:.5f}' for v in summary['predictions']]}")


def main():
    p = argparse.ArgumentParser(description="PHASE 1 compressible channel demo")
    p.add_argument("--ma",    type=float, default=0.1,  help="Inlet Mach number")
    p.add_argument("--nx",    type=int,   default=16,   help="Streamwise cells")
    p.add_argument("--ny",    type=int,   default=12,   help="Wall-normal cells")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: use smaller default mesh (no-op; already fast)")
    p.add_argument("--report", type=str, default=None,
                   help="Directory to write summary.json")
    p.add_argument("-o", "--save-dir", type=str, default=None,
                   help="Alias for --report: directory to write summary.json")
    args = p.parse_args()

    outdir_str = args.save_dir or args.report
    case    = build_case(args.nx, args.ny, args.ma)
    summary = summarize(case)
    print_summary(case, summary)

    if outdir_str:
        outdir = Path(outdir_str)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n  Summary -> {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
