"""
Integration of the warm-FD true-model gradient into the consumers.

* run_nuts: WarmFDForwardModel composes into the (log_post, grad) callback via
  make_truemodel_logpost_grad (the sampler-recovery gates live in test_nuts.py and are
  independent of the CFD gradient).
* active subspace: the checkpointed per-θ gradient_dataset job array fed by WarmFDForwardModel
  builds the Gram matrix C ≈ (1/N)Σ gᵢgᵢᵀ.
* KOH: ∂L_KOH/∂θ = (∂L_KOH/∂η)·(∂η/∂θ) — the KOH ∂L/∂η composed with the warm-FD observable
  Jacobian, verified end-to-end against a finite difference of the KOH score.

Warm-FD is the engine.  The all-case production runs reuse exactly this
wiring: one WarmFDForwardModel per case, gradient_dataset as the resumable per-case job array.
"""

from __future__ import annotations

import numpy as np
import pytest


def _channel(rs, nx=20, ny=14, max_iter=1500):
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
    s.max_iterations = max_iter; s.convergence_tol = 1e-6; s.verbose = False
    s.alpha_u = 0.7; s.alpha_p = 0.5
    return (mesh, obs, bcs, nu, s, rs.Vec3(Ub, 0, 0), 0.0, kIn, omIn), xs, Cf


def _menter(rs):
    ps = rs.InferenceParameterSet.all11()
    return list(np.array(ps.pack(rs.SSTCoefficients())))


def _prior(rs, th):
    from bayesian_inference import Prior
    ps = rs.InferenceParameterSet.all11()
    lo = np.array(ps.lower_bounds()); hi = np.array(ps.upper_bounds())
    return Prior(means=np.array(th), stds=0.3 * (hi - lo), lower=lo, upper=hi)


@pytest.fixture(scope="module")
def case(rs):
    args, xs, Cf = _channel(rs)
    mesh = args[0]
    sens = rs.ParameterSensitivity(*args)
    th = _menter(rs)
    sens.solve_state(th)
    return rs, sens, mesh, th, xs, Cf


# --------------------------------------------------------------------------- #
# Active-subspace Gram matrix from the warm-FD gradient job array
# --------------------------------------------------------------------------- #
def test_active_subspace_from_warm_fd(case, tmp_path):
    rs, sens, _mesh, th, _xs, _Cf = case
    from gradient_inference import WarmFDForwardModel, gradient_dataset
    from identifiability import active_subspace

    fm = WarmFDForwardModel(sens, sens.n_obs(), warm_max_iter=300, warm_tol=1e-5)
    prior = _prior(rs, th)
    rng = np.random.default_rng(7)
    lo, hi = prior.lower, prior.upper
    thetas = np.array([th] + [np.clip(np.array(th) + rng.uniform(-1, 1, 11) * 0.05 * (hi - lo),
                                      lo + 1e-3, hi - 1e-3) for _ in range(2)])

    G, valid = gradient_dataset(fm, prior, thetas, tmp_path / "grads.npz", verbose=False)
    assert G.shape == (3, 11)
    assert valid.sum() >= 2, "too few valid warm-FD gradients"

    vals, vecs, frac = active_subspace(G[valid])
    assert vals.shape == (11,) and np.all(np.isfinite(vals))
    assert np.all(vals >= -1e-12) and vals[0] > 0
    assert abs(np.sum(frac) - 1.0) < 1e-9
    # resumability: a second call loads the checkpoint, no new solves
    n0 = fm.n_solves
    G2, _ = gradient_dataset(fm, prior, thetas, tmp_path / "grads.npz", verbose=False)
    assert np.allclose(np.nan_to_num(G2), np.nan_to_num(G))
    assert fm.n_solves == n0, "checkpoint not reused (extra solves)"


# --------------------------------------------------------------------------- #
# KOH log-likelihood gradient w.r.t. θ — composition verified end-to-end vs FD
# --------------------------------------------------------------------------- #
def test_koh_loglik_gradient_theta(case):
    rs, sens, _mesh, th, xs, Cf = case
    from gradient_inference import WarmFDForwardModel, koh_loglik_gradient_theta
    from bayesian_inference import KOHLikelihood

    y = np.full(len(xs), Cf)                      # synthetic data near the prediction
    sig = np.full(len(xs), 0.05 * Cf)
    koh = KOHLikelihood(xs, y, sig, mode="physical_gp")
    lsd, lld = -3.0, 0.2

    fm = WarmFDForwardModel(sens, sens.n_obs(), warm_max_iter=400, warm_tol=1e-5)
    eta = fm.eta(th)
    J = fm.eta_jacobian(th)                       # n_obs × 11
    g_theta, g_s, g_l = koh_loglik_gradient_theta(koh, eta, J, lsd, lld)
    assert g_theta.shape == (11,) and np.all(np.isfinite(g_theta))
    assert np.isfinite(g_s) and np.isfinite(g_l)

    # end-to-end FD of the KOH score L(η(θ)) for the strong coefficient betaStar
    j = 8
    th0 = np.asarray(th, float); hj = max(1e-5 * abs(th0[j]), 1e-7)
    def koh_at(tv):
        s = rs.ParameterSensitivity(*_channel(rs)[0]); s.solve_state(list(tv))
        return koh(np.asarray(s.observe(list(tv)), float), lsd, lld)
    e = np.zeros(11); e[j] = hj
    g_fd = (koh_at(th0 + e) - koh_at(th0 - e)) / (2 * hj)
    rel = abs(g_theta[j] - g_fd) / max(abs(g_fd), 1e-9)
    assert rel < 5e-2, f"KOH ∂L/∂betaStar: composed {g_theta[j]:.4e} vs FD {g_fd:.4e} (rel {rel:.2e})"

    # KOH finite/−inf contract preserved on non-finite η
    gz, sz, lz = koh_loglik_gradient_theta(koh, np.full(sens.n_obs(), np.nan), J, lsd, lld)
    assert np.all(gz == 0) and sz == 0.0 and lz == 0.0


# --------------------------------------------------------------------------- #
# NUTS (log_post, grad) callback from warm-FD
# --------------------------------------------------------------------------- #
def test_warm_fd_nuts_callback(case):
    rs, sens, _mesh, th, _xs, _Cf = case
    from gradient_inference import WarmFDForwardModel, make_truemodel_logpost_grad

    fm = WarmFDForwardModel(sens, sens.n_obs(), warm_max_iter=300, warm_tol=1e-5)
    logpg = make_truemodel_logpost_grad(fm, _prior(rs, th))
    lp, grad = logpg(np.asarray(th))
    assert np.isfinite(lp) and grad.shape == (11,) and np.all(np.isfinite(grad))
    # out-of-support θ ⇒ −inf, zero grad (NUTS reject contract)
    bad = np.array(th); bad[0] = 1e9
    lp2, grad2 = logpg(bad)
    assert lp2 == -np.inf and np.all(grad2 == 0)
