"""
Shared validation-case builders for the 1+3+5+7+8 program (Phases 3-6).

Each builder returns a ready ``CaseSpec`` (mesh, forward model, active parameter set,
narrowed sampling prior, observation metadata) for a given closure variant
(0=M_full, 1=M_nolim, 2=M_kw) so the model-evidence, identifiability, and multi-case
studies all consume the *same* canonical cases:

  * channel    — Re_b=6800, attached; Dean (1978) Cf anchor.
  * flat_plate — Re_L=2e6, attached APG-free; Schoenherr Cf anchor (channel mesh +
                 freestream BCs, as in the C++ CLI).
  * bfs        — Re_h=37,600, separated; Driver-Seegmiller (1985) reattachment
                 (the smoothed, differentiable QoI) — the discriminating headline case.

The active subset defaults to ``a1_betaStar`` (d_θ=2): the model-selection question is
*structural* (which closure), so the coefficients are held to the low-D, surrogate-
trustworthy workhorse pair while the closure term-toggle varies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "build"), str(_REPO / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rans_sst_py as rs
from bayesian_inference import make_sampling_prior, DS1985, _bfs_solver_settings


VARIANTS = {0: "M_full", 1: "M_nolim", 2: "M_kw"}


class CaseSpec:
    """Bundle of everything a study needs for one (case, variant)."""

    def __init__(self, name, variant, mesh, forward_model, param_set, prior,
                 nu, obs_locations, obs_values, obs_sigmas, obs_kind):
        self.name = name
        self.variant = variant
        self.variant_name = VARIANTS[variant]
        self.mesh = mesh
        self.fm = forward_model
        self.param_set = param_set
        self.prior = prior
        self.nu = nu
        self.obs_locations = np.asarray(obs_locations, float)
        self.obs_values = np.asarray(obs_values, float)
        self.obs_sigmas = np.asarray(obs_sigmas, float)
        self.obs_kind = obs_kind
        self.n_obs = len(self.obs_values)


def _settings(variant, **over):
    s = rs.SolverSettings()
    s.max_iterations = over.get("max_iterations", 3000)
    s.convergence_tol = 1e-4
    s.verbose = False
    s.alpha_u = over.get("alpha_u", 0.7)
    s.alpha_p = over.get("alpha_p", 0.5)
    s.sst_variant = int(variant)
    return s


def build_channel(variant=0, param_set=None, nx=40, ny=30, n_cf=5):
    h, Lx, Ub, Re_b = 1.0, 10.0, 1.0, 6800.0
    nu = Ub * h / Re_b
    Cf = 0.073 * Re_b ** (-0.25)
    mesh = rs.Mesh.make_channel_2d(nx=nx, ny=ny, Lx=Lx, Ly=2.0 * h, Re=Re_b,
                                   yPlusTarget=1.0)
    mesh.compute_wall_distance()
    Tu = 0.05
    kIn = 1.5 * (Ub * Tu) ** 2
    omIn = kIn / (nu * 100.0)
    bcs = rs.FlowBoundaryConditions.channel_defaults(mesh, Ub, kIn, omIn)
    xs = np.linspace(3.0, 7.0, n_cf)
    obs = rs.ObservationOperator()
    for x in xs:
        obs.add_skin_friction("bottom", rs.Vec3(float(x), 0, 0), Cf, 0.05 * Cf, Ub)
    ps = param_set or rs.InferenceParameterSet.a1_betaStar()
    fm = rs.ForwardModel(mesh, ps, obs, bcs, nu, _settings(variant),
                         rs.Vec3(Ub, 0, 0), 0.0, kIn, omIn)
    prior = make_sampling_prior(ps)
    return CaseSpec("channel", variant, mesh, fm, ps, prior, nu,
                    xs, np.full(n_cf, Cf), np.full(n_cf, 0.05 * Cf), "Cf")


def build_flat_plate(variant=0, param_set=None, nx=60, ny=40, n_cf=5):
    L, Uinf, Re_L = 1.0, 1.0, 2.0e6
    nu = Uinf * L / Re_L
    mesh = rs.Mesh.make_channel_2d(nx=nx, ny=ny, Lx=L, Ly=0.1 * L, Re=Re_L,
                                   yPlusTarget=1.0)
    mesh.compute_wall_distance()
    Tu = 0.01
    kIn = 1.5 * (Uinf * Tu) ** 2
    omIn = kIn / (nu * 100.0)
    bcs = rs.FlowBoundaryConditions.flat_plate_defaults(mesh, Uinf, kIn, omIn)
    xs = np.linspace(0.4, 0.9, n_cf) * L
    obs = rs.ObservationOperator()
    cf_vals = 0.0592 * (Re_L * xs / L) ** (-0.2)        # Schoenherr Cf(x)
    for x, cf in zip(xs, cf_vals):
        obs.add_skin_friction("bottom", rs.Vec3(float(x), 0, 0), float(cf),
                              0.05 * float(cf), Uinf)
    ps = param_set or rs.InferenceParameterSet.a1_betaStar()
    fm = rs.ForwardModel(mesh, ps, obs, bcs, nu,
                         _settings(variant, max_iterations=5000),
                         rs.Vec3(Uinf, 0, 0), 0.0, kIn, omIn)
    prior = make_sampling_prior(ps)
    return CaseSpec("flat_plate", variant, mesh, fm, ps, prior, nu,
                    xs, cf_vals, 0.05 * cf_vals, "Cf")


def build_bfs(variant=0, param_set=None, nx_up=20, nx_down=40, ny_up=20, ny_down=15):
    h_s, H, Lu, Ld, Re_h = (DS1985["h_s"], DS1985["H"], DS1985["Lu"],
                            DS1985["Ld"], DS1985["Re_h"])
    Ub = 1.0
    nu = Ub * h_s / Re_h
    mesh = rs.Mesh.make_backward_facing_step_2d(nx_up, nx_down, ny_up, ny_down,
                                                Lu, Ld, h_s, H, Re=Re_h,
                                                yPlusTarget=1.0)
    mesh.compute_wall_distance()
    Tu = 0.05
    kIn = 1.5 * (Ub * Tu) ** 2
    omIn = kIn / (nu * 100.0)
    bcs = rs.FlowBoundaryConditions.bfs_defaults(mesh, Ub, kIn, omIn)
    xr = DS1985["xr_h"]
    obs = rs.ObservationOperator()
    obs.add_reattachment_length("bottom_wall_down", xr_obs=xr, sigma=0.5)
    ps = param_set or rs.InferenceParameterSet.a1_betaStar()
    s = _bfs_solver_settings(rs)
    s.sst_variant = int(variant)
    fm = rs.ForwardModel(mesh, ps, obs, bcs, nu, s, rs.Vec3(Ub, 0, 0), 0.0, kIn, omIn)
    prior = make_sampling_prior(ps)
    return CaseSpec("bfs", variant, mesh, fm, ps, prior, nu,
                    [xr], [xr], [0.5], "reattachment")


BUILDERS = {"channel": build_channel, "flat_plate": build_flat_plate, "bfs": build_bfs}


def build_case(name, variant=0, param_set=None):
    return BUILDERS[name](variant=variant, param_set=param_set)


if __name__ == "__main__":
    # smoke: each case builds and evaluates finite at Menter defaults (M_full)
    theta = list(rs.InferenceParameterSet.a1_betaStar().pack(rs.SSTCoefficients()))
    for name in ("channel", "flat_plate", "bfs"):
        cs = build_case(name, variant=0)
        r = cs.fm.evaluate(theta)
        print(f"{name:11s} d_obs={cs.n_obs}  status={str(r.status).split('.')[-1]:11s} "
              f"preds[0]={r.predictions[0]:.4f}  loglik={r.log_lik:.3f}")
