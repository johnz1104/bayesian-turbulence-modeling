"""
PHASE 7 — y+ and wall-function diagnostics.

Public API
----------
y_plus_first_cell(mesh, fields, nu, wall_patch=None, beta_star=0.09)
    For each wall face on the named patch (or all walls), compute y⁺ at the
    first interior cell centre using the SST-consistent estimate
    u_τ = (β*¹ᐟ⁴) √k_p where k_p is the cell-centre TKE.

cf_along_wall(mesh, fields, nu, wall_patch, ref_vel)
    Skin-friction coefficient along a wall patch using the cell-centre
    tangential velocity divided by the wall normal distance — the same
    incompressible Cf approximation used by ObservationOperator.

coarse_vs_fine_summary(coarse, fine)
    JSON-friendly comparison dict including y⁺ and Cf relative error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_PYTHON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PYTHON_DIR))


def _wall_patches(mesh, name: str | None):
    names = mesh.patch_names()
    types = mesh.patch_types()
    out = []
    for n, t in zip(names, types):
        if t != "wall":
            continue
        if name is not None and n != name:
            continue
        out.append(n)
    return out


def y_plus_first_cell(mesh, fields, nu: float, *,
                       wall_patch: str | None = None,
                       beta_star: float = 0.09) -> dict:
    """y⁺ at the first interior cell centre adjacent to each wall face."""
    k = np.asarray(fields["k"], float)
    y_plus_all = []
    patch_breakdown = {}

    for pname in _wall_patches(mesh, wall_patch):
        info  = mesh.wall_patch_data(pname)
        own   = np.asarray(info["owner"], int)
        delta = np.asarray(info["delta"], float)
        delta = np.maximum(delta, 1e-30)
        k_p   = np.maximum(k[own], 1e-30)
        uTau  = np.sqrt(np.sqrt(beta_star) * k_p)
        yp    = uTau * delta / nu
        patch_breakdown[pname] = yp
        y_plus_all.append(yp)

    arr = np.concatenate(y_plus_all) if y_plus_all else np.empty(0)
    return {
        "per_face_y_plus": arr,
        "per_patch":       {n: v.tolist() for n, v in patch_breakdown.items()},
        "min":  float(np.min(arr))    if arr.size else float("nan"),
        "max":  float(np.max(arr))    if arr.size else float("nan"),
        "mean": float(np.mean(arr))   if arr.size else float("nan"),
        "n_below_5":     int(np.sum(arr < 5.0)),
        "n_30_to_300":   int(np.sum((arr >= 30.0) & (arr <= 300.0))),
        "n_total":       int(arr.size),
    }


def cf_along_wall(mesh, fields, nu: float, wall_patch: str,
                   ref_vel: float) -> dict:
    """Skin-friction coefficient along ``wall_patch`` (incompressible form)."""
    info = mesh.wall_patch_data(wall_patch)
    own  = np.asarray(info["owner"], int)
    delta = np.maximum(np.asarray(info["delta"], float), 1e-30)
    normal = np.asarray(info["normal"], float)
    center = np.asarray(info["center"], float)
    U      = np.asarray(fields["U"], float)
    rho_dyn = 0.5 * ref_vel ** 2

    Up    = U[own]                         # (n_faces, 3)
    proj  = np.sum(Up * normal, axis=1)    # n . U
    Up_t  = Up - proj[:, None] * normal     # tangential component
    tau   = nu * np.linalg.norm(Up_t, axis=1) / delta
    cf    = tau / rho_dyn

    return {
        "x":    center[:, 0],
        "y":    center[:, 1],
        "cf":   cf,
        "mean": float(np.mean(cf)) if cf.size else float("nan"),
        "max":  float(np.max(cf))  if cf.size else float("nan"),
    }


def coarse_vs_fine_summary(coarse: dict, fine: dict) -> dict:
    """Compare a coarse wall-function run with a fine resolved-LES run."""
    cf_c = np.asarray(coarse["cf"]["cf"], float)
    cf_f = np.asarray(fine["cf"]["cf"],   float)
    if cf_f.size > cf_c.size:
        x_c = np.asarray(coarse["cf"]["x"], float)
        x_f = np.asarray(fine["cf"]["x"],   float)
        order = np.argsort(x_f)
        cf_f_resampled = np.interp(x_c, x_f[order], cf_f[order])
    else:
        cf_f_resampled = cf_f
    rel = np.abs(cf_c - cf_f_resampled) / np.maximum(np.abs(cf_f_resampled), 1e-12)
    return {
        "coarse_y_plus_mean": coarse["y_plus"]["mean"],
        "fine_y_plus_mean":   fine["y_plus"]["mean"],
        "cf_max_rel_error":   float(np.max(rel)),
        "cf_mean_rel_error":  float(np.mean(rel)),
        "n_compared":         int(cf_c.size),
    }
