"""
ADJOINT GROUNDWORK regression contract — analytic ∂R/∂θ and ∂g/∂θ.

Locks the first (NON-HELD) increment of the discrete adjoint: the EXACT analytic
sensitivity of the discrete residual and of the observation operator to each of the 11
SST coefficients, at a FIXED converged primary state.  Nothing here differentiates the
residual w.r.t. the FIELDS — the held (∂R/∂U)ᵀ core is untouched.

Validation oracle (mirrors python/gradient_inference.py conventions):
  * 4th-order central finite difference of ``ParameterSensitivity.residual(θ)`` /
    ``observe(θ)`` at the FIXED state — i.e. the PARTIAL ∂/∂θ at fixed U, NOT the
    whole-solve total derivative.
  * The stencil is written difference-first  (8·(f₊−f₋) − (f₊₊−f₋₋))/12h  so residual
    entries that are independent of θ_j cancel to EXACT zero.
  * The wall-adjacent ω rows are Dirichlet-PINNED by the solver (ω overridden by the
    wall BC, ω = 60ν/(β1 y²)); their "residual" is a BC-override imbalance of O(1e6–1e7),
    not a PDE residual.  That huge magnitude sets a catastrophic-cancellation floor in any
    FD, so the contract is checked on the genuine PDE rows (these BC rows are excluded —
    the discrete adjoint treats Dirichlet rows specially anyway).  The analytic ∂R/∂θ is
    still computed on them; it simply is not FD-verifiable there.
  * Per-coefficient step: 9 of 11 coefficients enter R LINEARLY (σ_k1/σ_k2 and σ_w1 via
    the diffusion blends, α1/α2 via ω-production, β1/β2 via ω-destruction, σ_w2 via the
    sign-fixed cross-diffusion) — a moderate central step is exact for them; β* and a1
    are NONLINEAR (eddy-viscosity max()/Menter min()/F1 tanh) and use a SMALL step so the
    stencil does not straddle a branch kink (the analytic is one-sided-exact at a kink;
    a straddling FD cannot verify it — confirmed by rel err collapsing as h_nl shrinks).
    h_nl is 1e-6: at the wall-molecular momentum treatment's converged BFS state the
    near-step omega rows sit close enough to a closure kink that a 1e-5 relative step
    straddles it (rel err 1.1e-2 at 1e-5, 1.3e-8 at 1e-6, converged at 1e-7); the gate
    itself is unchanged.

Gate: relative error < 1e-6 for every coefficient whose analytic derivative is
non-trivial; a coefficient with analytic ∂R/∂θ ≡ 0 (κ, unused) must FD to ~0.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

NAMES = ["sigma_k1", "sigma_w1", "beta1", "alpha1", "sigma_k2", "sigma_w2",
         "beta2", "alpha2", "betaStar", "a1", "kappa"]
NONLINEAR = (8, 9)          # betaStar, a1 — the only coefficients nonlinear in R
REL_TOL   = 1e-6            # gate for non-trivial coefficients
TRIVIAL   = 1e-7            # |FD| below this ⇒ coefficient is structurally zero


# --------------------------------------------------------------------------- #
# Fixed-state finite-difference Jacobian of a vector field f(θ)
# --------------------------------------------------------------------------- #
def fd_jacobian(fn, theta0, n_out, *, h_lin=5e-4, h_nl=1e-6, h_floor=1e-7):
    """Difference-first 4th-order central FD of f: R^11 -> R^n_out, at fixed state."""
    theta0 = np.asarray(theta0, float)
    d = len(theta0)
    J = np.zeros((d, n_out))
    for j in range(d):
        hr = h_nl if j in NONLINEAR else h_lin
        h = max(hr * abs(theta0[j]), h_floor)
        e = np.zeros(d); e[j] = h
        fp2 = np.asarray(fn((theta0 + 2 * e).tolist()), float)
        fp1 = np.asarray(fn((theta0 + e).tolist()), float)
        fm1 = np.asarray(fn((theta0 - e).tolist()), float)
        fm2 = np.asarray(fn((theta0 - 2 * e).tolist()), float)
        J[j] = (8.0 * (fp1 - fm1) - (fp2 - fm2)) / (12.0 * h)
    return J


def pde_row_mask(mesh, n_cells, n_state):
    """Boolean mask over the stacked residual that KEEPS genuine PDE rows and drops the
    Dirichlet-pinned ω rows (the wall-adjacent cells whose ω the solver overrides)."""
    keep = np.ones(n_state, bool)
    names, types = mesh.patch_names(), mesh.patch_types()
    for nm, ty in zip(names, types):
        if ty == "wall":
            for ow in np.asarray(mesh.wall_patch_data(nm)["owner"]):
                keep[3 * n_cells + int(ow)] = False     # ω block = [3·nc, 4·nc)
    return keep


# --------------------------------------------------------------------------- #
# Case builders (mirror python/case_library.py, but driving ParameterSensitivity)
# --------------------------------------------------------------------------- #
def _channel(rs):
    h, Lx, Ub, Re_b = 1.0, 10.0, 1.0, 6800.0
    nu = Ub * h / Re_b
    Cf = 0.073 * Re_b ** -0.25
    mesh = rs.Mesh.make_channel_2d(nx=32, ny=24, Lx=Lx, Ly=2.0 * h, Re=Re_b, yPlusTarget=1.0)
    mesh.compute_wall_distance()
    Tu = 0.05; kIn = 1.5 * (Ub * Tu) ** 2; omIn = kIn / (nu * 100.0)
    bcs = rs.FlowBoundaryConditions.channel_defaults(mesh, Ub, kIn, omIn)
    obs = rs.ObservationOperator()
    for x in np.linspace(3.0, 7.0, 4):
        obs.add_skin_friction("bottom", rs.Vec3(float(x), 0, 0), Cf, 0.05 * Cf, Ub)
    s = rs.SolverSettings()
    s.max_iterations = 4000; s.convergence_tol = 1e-5; s.verbose = False
    s.alpha_u = 0.7; s.alpha_p = 0.5
    return mesh, obs, bcs, nu, s, rs.Vec3(Ub, 0, 0), 0.0, kIn, omIn


def _bfs(rs):
    h_s, H, Lu, Ld, Re_h, xr_h = 0.0127, 0.0381, 0.0508, 0.508, 37600.0, 6.26
    Ub = 1.0; nu = Ub * h_s / Re_h
    mesh = rs.Mesh.make_backward_facing_step_2d(16, 32, 16, 12, Lu, Ld, h_s, H,
                                                Re=Re_h, yPlusTarget=1.0)
    mesh.compute_wall_distance()
    Tu = 0.05; kIn = 1.5 * (Ub * Tu) ** 2; omIn = kIn / (nu * 100.0)
    bcs = rs.FlowBoundaryConditions.bfs_defaults(mesh, Ub, kIn, omIn)
    obs = rs.ObservationOperator()
    obs.add_reattachment_length("bottom_wall_down", xr_obs=xr_h * h_s, sigma=0.5)
    obs.add_skin_friction("bottom_wall_down", rs.Vec3(10 * h_s, 0, 0.5), 0.002, 0.001, Ub)
    s = rs.SolverSettings()
    # bounded iterations: the FD-check validates the derivative at the FIXED state and is
    # state-agnostic, so full convergence is not required for the contract.
    s.max_iterations = 1500; s.convergence_tol = 1e-4; s.divergence_limit = 1e8
    s.alpha_u = 0.3; s.alpha_p = 0.2; s.alpha_k = 0.4; s.alpha_omega = 0.4
    s.inner_iterations = 300; s.inner_tolerance = 1e-4
    s.turb_start_iter = 30; s.turb_update_interval = 2; s.verbose = False
    return mesh, obs, bcs, nu, s, rs.Vec3(Ub, 0, 0), 0.0, kIn, omIn


def _theta_points(rs, n_random=3, seed=12345):
    ps = rs.InferenceParameterSet.all11()
    lo = np.array(ps.lower_bounds()); hi = np.array(ps.upper_bounds())
    menter = np.array(ps.pack(rs.SSTCoefficients()))
    rng = np.random.default_rng(seed)
    pts = [menter]
    span = 0.15 * (hi - lo)
    for _ in range(n_random):
        t = menter + rng.uniform(-1, 1, 11) * span
        pts.append(np.clip(t, lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)))
    return [p.tolist() for p in pts]


@pytest.fixture(scope="module")
def channel_sens(rs):
    args = _channel(rs); mesh = args[0]      # keep mesh alive + reuse for the row mask
    sens = rs.ParameterSensitivity(*args)
    status = sens.solve_state(_theta_points(rs)[0])
    return rs, sens, mesh, status


@pytest.fixture(scope="module")
def bfs_sens(rs):
    args = _bfs(rs); mesh = args[0]
    sens = rs.ParameterSensitivity(*args)
    sens.solve_state(_theta_points(rs)[0])
    return rs, sens, mesh


# --------------------------------------------------------------------------- #
# Core contract: analytic ∂R/∂θ vs fixed-state 4th-order central FD
# --------------------------------------------------------------------------- #
def _check_dR(rs, sens, mesh, label):
    n = sens.n_state()
    keep = pde_row_mask(mesh, sens.n_cells(), n)            # drop Dirichlet-pinned ω rows
    worst = 0.0
    for ti, th in enumerate(_theta_points(rs)):
        dR_an = np.asarray(sens.d_residual_d_theta(th))[:, keep]   # (11, nKept)
        dR_fd = fd_jacobian(sens.residual, th, n)[:, keep]
        for j in range(11):
            fnorm = np.linalg.norm(dR_fd[j])
            anorm = np.linalg.norm(dR_an[j])
            if fnorm < TRIVIAL and anorm < TRIVIAL:
                # structurally zero (κ never enters the residual) — must FD to ~0
                assert anorm == 0.0, f"{label} {NAMES[j]}: expected exact 0, got {anorm:.2e}"
                continue
            rel = np.linalg.norm(dR_an[j] - dR_fd[j]) / max(fnorm, 1e-30)
            worst = max(worst, rel)
            assert rel < REL_TOL, (
                f"{label} θ#{ti} {NAMES[j]}: rel err {rel:.2e} "
                f"(|an|={anorm:.3e} |fd|={fnorm:.3e})")
    return worst


def test_residual_sensitivity_channel(channel_sens):
    rs, sens, mesh, status = channel_sens
    assert sens.has_state() and sens.n_obs() == 4
    worst = _check_dR(rs, sens, mesh, "channel")
    assert worst < REL_TOL


def test_residual_sensitivity_bfs(bfs_sens):
    rs, sens, mesh = bfs_sens
    assert sens.has_state()
    worst = _check_dR(rs, sens, mesh, "bfs")
    assert worst < REL_TOL


def test_kappa_residual_is_exactly_zero(channel_sens):
    """κ (index 10) enters no residual term ⇒ ∂R/∂κ ≡ 0 analytically and FD ≈ 0."""
    rs, sens, _mesh, _status = channel_sens
    th = _theta_points(rs)[0]
    dR_an = np.asarray(sens.d_residual_d_theta(th))
    assert np.all(dR_an[10] == 0.0)
    dR_fd = fd_jacobian(sens.residual, th, sens.n_state())
    assert np.linalg.norm(dR_fd[10]) < TRIVIAL


# --------------------------------------------------------------------------- #
# ∂g/∂θ — the (near-)zero observation sensitivity, with the documented reason
# --------------------------------------------------------------------------- #
def _check_dG(rs, sens, label):
    for th in _theta_points(rs):
        dG_an = np.asarray(sens.d_obs_d_theta(th))              # (nObs, 11)
        dG_fd = fd_jacobian(sens.observe, th, sens.n_obs()).T   # (nObs, 11)
        # The QoIs contain no explicit θ; their only closure dependence is nuT, and at
        # the near-wall cells they read, ω is large so the Bradshaw limiter is inactive
        # (nuT = k/ω, θ-independent).  Hence ∂g/∂θ is (near-)zero for every coefficient.
        assert np.max(np.abs(dG_an)) < 1e-9, f"{label}: analytic ∂g/∂θ not ~0"
        assert np.max(np.abs(dG_fd)) < 1e-5, f"{label}: FD ∂g/∂θ not ~0"
        assert np.allclose(dG_an, dG_fd, atol=1e-5)


def test_obs_sensitivity_near_zero_channel(channel_sens):
    rs, sens, _mesh, _status = channel_sens
    _check_dG(rs, sens, "channel")     # SkinFriction QoIs


def test_obs_sensitivity_near_zero_bfs(bfs_sens):
    rs, sens, _mesh = bfs_sens
    _check_dG(rs, sens, "bfs")         # ReattachmentLength + SkinFriction QoIs


# --------------------------------------------------------------------------- #
# Deliverable-1 sanity + cost note
# --------------------------------------------------------------------------- #
def test_residual_small_at_fixed_state(channel_sens):
    """R(U*, θ*) ≈ 0 for the PRIMARY (momentum, k) equations at the fixed state — the
    deliverable-1 sanity check.  The ω block carries the documented O(1e6) wall-pinning
    imbalance (the BC pinning overrides the PDE at wall cells, so the SIMPLE 'converged'
    flag stays unset), so only the momentum/k blocks are asserted small here."""
    rs, sens, _mesh, _status = channel_sens
    nc = sens.n_cells()
    R = np.asarray(sens.residual(_theta_points(rs)[0]))
    assert np.linalg.norm(R[0:nc])      < 1e-4    # momentum-x
    assert np.linalg.norm(R[nc:2 * nc]) < 1e-4    # momentum-y
    assert np.linalg.norm(R[2 * nc:3 * nc]) < 1e-4  # k


def test_sensitivity_cost_is_sub_solve(channel_sens):
    """Assembling ∂R/∂θ for all 11 coefficients is O(1) residual assemblies (no solves);
    it must be far cheaper than one SIMPLE solve."""
    rs, sens, _mesh, _status = channel_sens
    th = _theta_points(rs)[0]
    t0 = time.perf_counter()
    for _ in range(3):
        sens.d_residual_d_theta(th)
    t_sens = (time.perf_counter() - t0) / 3.0

    # a fresh solve from scratch (the cost ∂R/∂θ must beat by a wide margin)
    sens2 = rs.ParameterSensitivity(*_channel(rs))
    t0 = time.perf_counter()
    sens2.solve_state(th)
    t_solve = time.perf_counter() - t0
    assert t_sens < 0.1 * t_solve, f"∂R/∂θ {t_sens:.3f}s vs solve {t_solve:.3f}s"
