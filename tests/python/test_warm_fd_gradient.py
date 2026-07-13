"""
True-model gradient regression contract.

Locks the production warm-started-FD true-model gradient (``ParameterSensitivity.
eta_jacobian_warm_fd`` and the ``gradient_inference.WarmFDForwardModel`` drop-in) and the
frozen-pressure semi-analytic tangent.
Warm-FD is the default over the semi-analytic tangents.

Gates
-----
* warm-FD ∂η/∂θ MATCHES a cold full-FD re-solve (it IS full-FD, warm-started) — same fixed
  point, no frozen-pressure bias.  Checked on the strong coefficients within a defensible
  tolerance (FD floor + bounded-convergence noise on the coarse test mesh).
* warm-FD ∂logL/∂θ matches a finite difference of the scalar log-likelihood (internal
  consistency of the one-pass η + logL warm-FD).
* the frozen-pressure tangent CONVERGES on all 11 coefficients and agrees with warm-FD in
  DIRECTION (sign), with strictly smaller magnitude on the strong coefficients — the
  documented frozen-pressure (~0.77×) bias.
* the WarmFDForwardModel exposes the GradientForwardModel interface and composes with
  make_truemodel_logpost_grad into a finite (log_post, grad).

The held (∂R/∂U)ᵀ adjoint core is untouched — every engine here is matrix-free Jv or a full
warm re-solve; none assembles the analytic ∂R/∂U transpose.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

NAMES = ["sigma_k1", "sigma_w1", "beta1", "alpha1", "sigma_k2", "sigma_w2",
         "beta2", "alpha2", "betaStar", "a1", "kappa"]
NONLINEAR = (8, 9)
STRONG = (2, 8, 9)        # beta1, betaStar, a1 — the dominant skin-friction sensitivities


def _channel(rs, nx=20, ny=14, max_iter=1500):
    """Coarse inlet-velocity/outlet-pressure channel (fast enough for a cold-FD reference)."""
    h, Lx, Ub, Re_b = 1.0, 10.0, 1.0, 6800.0
    nu = Ub * h / Re_b
    Cf = 0.073 * Re_b ** -0.25
    mesh = rs.Mesh.make_channel_2d(nx=nx, ny=ny, Lx=Lx, Ly=2.0 * h, Re=Re_b, yPlusTarget=1.0)
    mesh.compute_wall_distance()
    Tu = 0.05; kIn = 1.5 * (Ub * Tu) ** 2; omIn = kIn / (nu * 100.0)
    bcs = rs.FlowBoundaryConditions.channel_defaults(mesh, Ub, kIn, omIn)
    obs = rs.ObservationOperator()
    for x in np.linspace(3.0, 7.0, 4):
        obs.add_skin_friction("bottom", rs.Vec3(float(x), 0, 0), Cf, 0.05 * Cf, Ub)
    s = rs.SolverSettings()
    s.max_iterations = max_iter; s.convergence_tol = 1e-6; s.verbose = False
    s.alpha_u = 0.7; s.alpha_p = 0.5
    return mesh, obs, bcs, nu, s, rs.Vec3(Ub, 0, 0), 0.0, kIn, omIn


def _menter(rs):
    ps = rs.InferenceParameterSet.all11()
    return list(np.array(ps.pack(rs.SSTCoefficients())))


@pytest.fixture(scope="module")
def channel_case(rs):
    args = _channel(rs)
    mesh = args[0]                       # keep alive
    sens = rs.ParameterSensitivity(*args)
    th = _menter(rs)
    sens.solve_state(th)
    return rs, sens, mesh, th


def _cold_eta(rs, theta):
    """Cold-start full solve at θ, observe — the unbiased FD oracle (fresh case each call)."""
    s = rs.ParameterSensitivity(*_channel(rs))
    s.solve_state(list(theta))
    return np.asarray(s.observe(list(theta)), float)


# --------------------------------------------------------------------------- #
# warm-FD == cold full-FD (it IS full-FD, just warm-started)
# --------------------------------------------------------------------------- #
def test_warm_fd_matches_cold_fd(channel_case):
    rs, sens, _mesh, th = channel_case
    res = sens.eta_jacobian_warm_fd(th, h_rel=5e-4, h_floor=1e-7,
                                    warm_max_iter=400, warm_tol=1e-5)
    J_warm = np.asarray(res.d_obs_d_theta)            # n_obs × 11
    assert all(res.krylov_converged), "warm re-solves diverged"

    th0 = np.asarray(th, float)
    for j in STRONG:                                  # cold-FD reference on the strong columns
        hr = 1e-5 if j in NONLINEAR else 5e-4
        hj = max(hr * abs(th0[j]), 1e-7)
        e = np.zeros(11); e[j] = hj
        col_fd = (_cold_eta(rs, th0 + e) - _cold_eta(rs, th0 - e)) / (2 * hj)
        rel = np.linalg.norm(J_warm[:, j] - col_fd) / max(np.linalg.norm(col_fd), 1e-30)
        assert rel < 2e-2, f"{NAMES[j]}: warm-FD vs cold-FD rel err {rel:.2e}"


def test_coupled_tangent_matches_warm_fd(channel_case):
    """The pressure-coupled semi-analytic tangent (block-preconditioned FGMRES on the 5-block
    saddle, wall-ω Dirichlet rows, BC re-application, wall-ω row scaling) ROBUSTLY converges and
    matches the warm-FD truth on the strong coefficients.  This is the
    contract that distinguishes the working coupled tangent from the earlier non-converging
    attempts (BiCGSTAB diverged, GMRES stalled, segregated tangent-SIMPLE limit-cycled)."""
    rs, sens, _mesh, th = channel_case
    Jw = np.asarray(sens.eta_jacobian_warm_fd(th, warm_max_iter=400, warm_tol=1e-5).d_obs_d_theta)
    rc = sens.eta_jacobian_tangent_coupled(th, krylov_tol=1e-7, max_iter=2000, fd_step=1e-6)
    Jc = np.asarray(rc.d_obs_d_theta)
    for j in STRONG:                                  # betaStar, beta1, a1 (the dominant ones)
        nw = np.linalg.norm(Jw[:, j])
        assert nw > 1e-3, f"{NAMES[j]} unexpectedly weak on this case"
        assert rc.krylov_converged[j], (
            f"{NAMES[j]}: coupled tangent did not converge (relres {rc.krylov_rel_res[j]:.1e})")
        rel = np.linalg.norm(Jc[:, j] - Jw[:, j]) / nw
        assert rel < 6e-2, f"{NAMES[j]}: coupled vs warm-FD rel err {rel:.2e}"


def test_warm_fd_loglik_gradient_consistent(channel_case):
    """∂logL/∂θ from warm-FD matches a finite difference of the scalar log-lik (1 coeff)."""
    rs, sens, _mesh, th = channel_case
    res = sens.eta_jacobian_warm_fd(th, warm_max_iter=400, warm_tol=1e-5)
    g = np.asarray(res.log_lik_gradient)
    assert g.shape == (11,) and np.all(np.isfinite(g))
    assert g[10] == 0.0 or abs(g[10]) < 1e-6          # κ unused ⇒ ~0

    j = 8                                             # betaStar
    th0 = np.asarray(th, float); hj = max(1e-5 * abs(th0[j]), 1e-7)
    e = np.zeros(11); e[j] = hj
    sp = rs.ParameterSensitivity(*_channel(rs)); sp.solve_state(list(th0 + e))
    sm = rs.ParameterSensitivity(*_channel(rs)); sm.solve_state(list(th0 - e))
    g_fd = (sp.log_lik(list(th0 + e)) - sm.log_lik(list(th0 - e))) / (2 * hj)
    assert abs(g[j] - g_fd) <= 0.05 * abs(g_fd) + 1e-6


# --------------------------------------------------------------------------- #
# frozen-pressure tangent — converges, direction-faithful, smaller magnitude
# --------------------------------------------------------------------------- #
def test_frozen_pressure_tangent_direction(channel_case):
    rs, sens, _mesh, th = channel_case
    # max_iter: the corrected SST-2003 omega production adds a betaStar RHS
    # contribution at limiter-active near-wall cells (10 betaStar k omega / nuT),
    # the stiffest part of the preconditioned spectrum; that direction needs
    # roughly five times the pre-correction Krylov budget and then converges
    # cleanly to 1e-9 (verified; no stall).
    rt = sens.eta_jacobian_tangent(th, krylov_tol=1e-9, max_iter=30000, fd_step=1e-6)
    assert all(rt.krylov_converged), f"frozen-P tangent did not converge: {list(rt.krylov_rel_res)}"
    Jt = np.asarray(rt.d_obs_d_theta)

    rw = sens.eta_jacobian_warm_fd(th, warm_max_iter=400, warm_tol=1e-5)
    Jw = np.asarray(rw.d_obs_d_theta)

    for j in STRONG:                                  # same sign, smaller magnitude (no dp/dθ)
        nt = np.linalg.norm(Jt[:, j]); nw = np.linalg.norm(Jw[:, j])
        assert nw > 1e-3
        cos = np.dot(Jt[:, j], Jw[:, j]) / (nt * nw + 1e-30)
        assert cos > 0.97, f"{NAMES[j]}: tangent/warm direction cos {cos:.3f}"
        assert nt < nw, f"{NAMES[j]}: frozen-P magnitude {nt:.3e} not < warm {nw:.3e}"


# --------------------------------------------------------------------------- #
# WarmFDForwardModel drop-in + NUTS log-posterior composition
# --------------------------------------------------------------------------- #
def test_warm_fd_forward_model_dropin(channel_case):
    rs, sens, _mesh, th = channel_case
    from gradient_inference import WarmFDForwardModel, make_truemodel_logpost_grad
    from bayesian_inference import Prior

    fm = WarmFDForwardModel(sens, sens.n_obs(), warm_max_iter=400, warm_tol=1e-5)
    eta = fm.eta(th)
    assert eta.shape == (sens.n_obs(),) and np.all(np.isfinite(eta))
    assert np.isfinite(fm.loglik(th))
    J = fm.eta_jacobian(th)
    assert J.shape == (sens.n_obs(), 11) and np.all(np.isfinite(J))
    g = fm.loglik_gradient(th)
    assert g.shape == (11,) and np.all(np.isfinite(g))
    assert fm.n_solves > 0

    ps = rs.InferenceParameterSet.all11()
    lo = np.array(ps.lower_bounds()); hi = np.array(ps.upper_bounds())
    prior = Prior(means=np.array(th), stds=0.3 * (hi - lo), lower=lo, upper=hi)
    logpg = make_truemodel_logpost_grad(fm, prior)
    lp, grad = logpg(np.asarray(th))
    assert np.isfinite(lp)
    assert grad.shape == (11,) and np.all(np.isfinite(grad))


def test_warm_fd_cheaper_than_cold(channel_case):
    """Warm-FD must beat a cold full-FD Jacobian (the point of warm-starting)."""
    rs, sens, _mesh, th = channel_case
    t0 = time.perf_counter()
    sens.eta_jacobian_warm_fd(th, warm_max_iter=300, warm_tol=1e-5)
    t_warm = time.perf_counter() - t0

    th0 = np.asarray(th, float)
    t0 = time.perf_counter()
    for _ in range(2):                                # 1 cold central-FD column = 2 cold solves
        _cold_eta(rs, th0)
    t_two_cold = time.perf_counter() - t0
    # 11-coeff warm-FD (22 warm solves) must be cheaper than 22 cold solves; compare the
    # per-solve cost: warm ≪ cold ⇒ warm-FD total < cold-FD total by a wide margin.
    assert t_warm < 11 * t_two_cold, f"warm-FD {t_warm:.1f}s not < 11×(2 cold) {11*t_two_cold:.1f}s"
