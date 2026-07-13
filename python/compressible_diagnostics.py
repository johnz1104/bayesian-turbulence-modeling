"""
Lightweight diagnostics for the compressible solver (PHASE 2).

Public API:
    run_validation_case(case_spec)  -> dict of QoIs
    column_mass_imbalance(...)       -> float, relative imbalance
    summarise_fields(...)            -> dict of min/max/mean per field

Designed to be imported by both ``examples/compressible_validation_ladder.py``
and ``tests/python/test_compressible_validation.py`` so the regression test and
the user-facing example agree on what is computed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


_REPO_ROOT  = Path(__file__).resolve().parent.parent
_BUILD_DIR  = _REPO_ROOT / "build"
_PYTHON_DIR = _REPO_ROOT / "python"
sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PYTHON_DIR))

import rans_sst_py as rs  # noqa: E402


# ---------- Field-level summaries ----------------------------------------

def summarise_fields(fields: dict, eos) -> dict:
    """Per-field min/max/mean and Mach number summary."""
    U   = fields["U"]
    p   = fields["p"]
    T   = fields["T"]
    rho = fields["rho"]
    nuT = fields["nuT"]
    Umag    = np.linalg.norm(U, axis=1)
    a_local = np.sqrt(eos.gamma * eos.R * T)
    Ma      = Umag / a_local
    return {
        "U_max":     float(np.max(Umag)),
        "U_mean":    float(np.mean(Umag)),
        "p_min":     float(np.min(p)),
        "p_max":     float(np.max(p)),
        "T_min":     float(np.min(T)),
        "T_max":     float(np.max(T)),
        "rho_min":   float(np.min(rho)),
        "rho_max":   float(np.max(rho)),
        "Ma_max":    float(np.max(Ma)),
        "Ma_mean":   float(np.mean(Ma)),
        "nuT_max":   float(np.max(nuT)),
    }


def positivity_ok(fields: dict) -> tuple[bool, dict]:
    """Strict positivity check on rho, p, T."""
    rho = fields["rho"]; p = fields["p"]; T = fields["T"]
    info = {
        "rho_positive": bool(np.all(rho > 0.0)),
        "p_positive":   bool(np.all(p   > 0.0)),
        "T_positive":   bool(np.all(T   > 0.0)),
    }
    return all(info.values()), info


# ---------- Column mass-flux imbalance -----------------------------------

def column_mass_imbalance(mesh, fields, x_tol: float = 1e-9) -> dict:
    """
    Relative mass-flux imbalance for a channel-style mesh (uniform Δx).

    For each unique x-column we evaluate:

        column_flux(x) = Σ_j ρ_ij U_x,ij Δy_j   (Δz absorbed in V)

    where Δy_j is recovered from the cell volume V = Δx · Δy · Δz with the
    column-uniform Δx detected automatically.  This avoids the off-by-one
    binning error that arises when fixed-count bins do not align with the
    actual cell columns.

    Returns ``rel_imbalance = (max-min)/|mean|`` across columns.  Values < 1e-3
    are typical of a well-converged Ma<0.5 channel solution.
    """
    centers = mesh.cell_centers()
    volumes = mesh.cell_volumes()
    rho     = fields["rho"]
    Ux      = fields["U"][:, 0]

    # Detect x-columns by clustering cell x-centers.  Channel meshes use a
    # uniform Δx, so the unique column count equals nx.
    xs_sorted = np.sort(centers[:, 0])
    unique_x  = [xs_sorted[0]]
    for x in xs_sorted[1:]:
        if x - unique_x[-1] > x_tol * max(abs(unique_x[-1]), 1.0):
            unique_x.append(float(x))
    unique_x = np.asarray(unique_x)
    if unique_x.size < 2:
        return {"max_flux": 0.0, "min_flux": 0.0,
                "mean_flux": 0.0, "rel_imbalance": float("nan"),
                "n_columns": int(unique_x.size)}

    dx_col = float(np.median(np.diff(unique_x)))
    fluxes = np.zeros(unique_x.size)
    for k, xk in enumerate(unique_x):
        mask = np.abs(centers[:, 0] - xk) < 0.5 * dx_col
        if not np.any(mask):
            continue
        # column flux = Σ_j ρ U_x V / Δx  (V/Δx = Δy·Δz)
        fluxes[k] = float(np.sum(rho[mask] * Ux[mask] * volumes[mask]) / dx_col)

    nz = fluxes[fluxes != 0.0]
    if nz.size == 0:
        return {"max_flux": 0.0, "min_flux": 0.0,
                "mean_flux": 0.0, "rel_imbalance": float("nan")}
    mean_flux = float(np.mean(nz))
    spread    = float(np.max(nz) - np.min(nz))
    rel = spread / abs(mean_flux) if abs(mean_flux) > 0 else float("nan")
    return {
        "max_flux":      float(np.max(nz)),
        "min_flux":      float(np.min(nz)),
        "mean_flux":     mean_flux,
        "rel_imbalance": rel,
        "n_columns":     int(nz.size),
    }


# ---------- One validation case ------------------------------------------

def make_validation_case(name: str, Ma: float, nx: int = 32, ny: int = 24,
                          Lx: float = 10.0, H: float = 1.0,
                          max_iterations: int = 4000,
                          convergence_tol: float = 1e-3,
                          turb_intensity: float = 0.05,
                          nut_floor_iters: int | None = None) -> dict[str, Any]:
    """Build the mesh + EOS + BCs + solver settings for a Ma=Ma channel."""
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
    bcs  = rs.CompressibleBoundaryConditions.channel_defaults(
        mesh, Uin, T_in, p_ref, kIn, omIn)

    obs = rs.ObservationOperator()
    for x_loc in (0.25 * Lx, 0.5 * Lx, 0.75 * Lx):
        obs.add_skin_friction(
            wall_patch="bottom", location=rs.Vec3(x_loc, 0.0, 0.0),
            cf_obs=0.005, sigma=0.001, ref_vel=Uin)

    settings = rs.SolverSettings()
    settings.max_iterations      = max_iterations
    settings.convergence_tol     = convergence_tol
    settings.divergence_limit    = 1e10
    settings.alpha_u             = 0.5 if Ma <= 0.3 else 0.4
    settings.alpha_p             = 0.2
    settings.alpha_t             = 0.7
    settings.alpha_k             = 0.4
    settings.alpha_omega         = 0.4
    settings.inner_iterations    = 200
    settings.inner_tolerance     = 1e-4
    settings.turb_start_iter     = 50
    settings.turb_update_interval = 2
    settings.verbose             = False
    settings.report_interval     = 500
    # marginal developing-channel cases may need a longer startup floor
    # window than the default (the floor is startup-only; see SolverSettings)
    if nut_floor_iters is not None:
        settings.nut_floor_iters = nut_floor_iters

    param_set = rs.InferenceParameterSet.a1_betaStar()
    fm = rs.CompressibleForwardModel(
        mesh=mesh, param_set=param_set, obs_op=obs, bcs=bcs, eos=eos,
        settings=settings, u_init=rs.Vec3(Uin, 0, 0),
        p_init=p_ref, T_init=T_in, k_init=kIn, omega_init=omIn)

    return {
        "name": name, "Ma": Ma, "Re": Re,
        "Uin": Uin, "T_in": T_in, "p_ref": p_ref,
        "rho_in": rho_in, "mu_in": mu_in,
        "mesh": mesh, "eos": eos, "obs": obs, "bcs": bcs, "settings": settings,
        "param_set": param_set, "forward": fm,
        "nx": nx, "ny": ny, "Lx": Lx, "H": H,
        "max_iterations": max_iterations,
    }


def run_validation_case(case: dict[str, Any]) -> dict[str, Any]:
    """Solve at Menter SST defaults; return a reproducible QoI dict.

    Failures that prevent the solver from leaving any fields behind (typically
    Diverged or InvalidParameters) are recorded in the summary instead of
    crashing the caller, so a multi-case ladder can still report on the
    cases that did run.
    """
    fm        = case["forward"]
    eos       = case["eos"]
    param_set = case["param_set"]
    theta_def = list(param_set.pack(rs.SSTCoefficients()))

    t0 = time.time()
    result = fm.evaluate(theta_def)
    elapsed = time.time() - t0

    status_str = str(result.status)
    summary: dict[str, Any] = {
        "name":                case["name"],
        "Ma":                  case["Ma"],
        "Re":                  case["Re"],
        "mesh_cells":          int(case["mesh"].n_cells()),
        "max_iterations":      case["max_iterations"],
        "simple_iters":        int(result.simple_iters),
        "elapsed_s":           float(elapsed),
        "status":              status_str,
        "converged":           "Converged" in status_str,
        # The C++ enum is `Diverged` but the binding stringifies to
        # "EvaluationStatus.DivergenceDetected"; match either spelling.
        "diverged":            ("Diverged" in status_str
                                or "Divergence" in status_str),
        "log_lik":             float(result.log_lik),
        "Cf_at_stations":      list(result.predictions),
    }

    if not fm.has_last_fields():
        summary.update({
            "fields_available": False,
            "positivity_ok":    False,
            "positivity":       {"rho_positive": False, "p_positive": False,
                                  "T_positive": False},
            "U_max":   float("nan"), "U_mean":  float("nan"),
            "p_min":   float("nan"), "p_max":   float("nan"),
            "T_min":   float("nan"), "T_max":   float("nan"),
            "rho_min": float("nan"), "rho_max": float("nan"),
            "Ma_max":  float("nan"), "Ma_mean": float("nan"),
            "nuT_max": float("nan"),
            "mass_flux": {"max_flux": float("nan"), "min_flux": float("nan"),
                          "mean_flux": float("nan"), "rel_imbalance": float("nan"),
                          "n_columns": 0},
        })
        return summary

    fields = fm.last_fields()
    pos_ok, pos_info = positivity_ok(fields)
    field_summary    = summarise_fields(fields, eos)
    mass_info        = column_mass_imbalance(case["mesh"], fields)

    summary.update({
        "fields_available": True,
        "positivity_ok":    pos_ok,
        "positivity":       pos_info,
        **field_summary,
        "mass_flux":        mass_info,
    })
    return summary


def format_summary_row(s: dict) -> str:
    base = (f"  {s['name']:<14s}  Ma={s['Ma']:<4.2f}  "
            f"status={s['status']:<35s} iters={s['simple_iters']:>5d}")
    if not s.get("fields_available", True):
        return base + "  [no fields available — see status]"
    return (
        base +
        f"  Ma_max={s['Ma_max']:.3f}  "
        f"T=[{s['T_min']:.1f},{s['T_max']:.1f}]K  "
        f"rho_min={s['rho_min']:.4f}  "
        f"mass_imb={s['mass_flux']['rel_imbalance']:.2e}"
    )
