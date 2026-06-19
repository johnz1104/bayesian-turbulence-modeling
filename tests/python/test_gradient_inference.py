"""
Regression contracts for the FD-first gradient stack.

Covers the analytic surrogate-gradient gate, the FD primitives, the true-model FD
oracle (against an analytic fake forward model), the truncated-normal prior gradient,
and the checkpointed gradient dataset.  No real CFD is used.
"""

from __future__ import annotations

import numpy as np

from bayesian_inference import GPSurrogate, Prior
from gradient_inference import (central_fd_gradient, surrogate_gradient_check,
                                GradientForwardModel, _prior_logp_and_grad,
                                make_surrogate_logpost_grad, gradient_dataset)


# --------------------------------------------------------------------------- #
# Fake analytic forward model for the FD-oracle tests
# --------------------------------------------------------------------------- #
class _Res:
    def __init__(self, preds, ll):
        self.predictions = list(preds)
        self.log_lik = ll
        self.status = "Converged"


class _AnalyticFM:
    """η(θ) = [θ0², θ0·θ1, sin(θ1)];  log-lik = -0.5 Σ (η-y)²/σ²."""

    def __init__(self, y, sigma):
        self.y = np.asarray(y, float)
        self.sigma = np.asarray(sigma, float)

    def eta(self, t):
        return np.array([t[0] ** 2, t[0] * t[1], np.sin(t[1])])

    def jac(self, t):
        return np.array([[2 * t[0], 0.0],
                         [t[1], t[0]],
                         [0.0, np.cos(t[1])]])

    def evaluate(self, theta_list):
        t = np.asarray(theta_list, float)
        e = self.eta(t)
        ll = -0.5 * np.sum(((e - self.y) / self.sigma) ** 2)
        return _Res(e, float(ll))


# --------------------------------------------------------------------------- #
# FD primitives
# --------------------------------------------------------------------------- #
def test_central_fd_gradient_orders():
    mu = np.array([0.3, -0.2, 0.5])
    f = lambda t: -0.5 * np.sum((t - mu) ** 2)
    t = np.array([0.1, 0.4, -0.1])
    truth = -(t - mu)
    for order in (2, 4):
        g = central_fd_gradient(f, t, order=order)
        assert np.allclose(g, truth, atol=1e-7)


# --------------------------------------------------------------------------- #
# Analytic surrogate gradient gate (the mandatory first check)
# --------------------------------------------------------------------------- #
def test_surrogate_gradient_matches_fd(rs):
    rng = np.random.default_rng(0)
    d, n = 4, 120
    X = rng.uniform(-1, 1, (n, d))
    truth = (np.sin(1.3 * X[:, 0]) + 0.5 * X[:, 1] ** 2
             - 0.7 * X[:, 2] * X[:, 3] + 0.2 * X[:, 0])
    y = truth + 0.02 * rng.standard_normal(n)   # noise -> non-degenerate GP
    gp = GPSurrogate()
    gp.train(X, y, optimize_restarts=4)
    pts = rng.uniform(-0.7, 0.7, (10, d))
    max_rel = surrogate_gradient_check(gp, pts, h=1e-4, order=4)
    assert max_rel < 1e-6, f"analytic surrogate gradient rel err {max_rel:.2e}"


# --------------------------------------------------------------------------- #
# True-model FD oracle vs analytic
# --------------------------------------------------------------------------- #
def test_fd_oracle_loglik_gradient():
    fm = _AnalyticFM(y=[0.2, 0.1, 0.0], sigma=[0.1, 0.1, 0.1])
    gfm = GradientForwardModel(fm, n_obs=3, h=1e-5)
    t = np.array([0.4, 0.3])
    # analytic ∂logL/∂θ = -Σ (η-y)/σ² · J
    e = fm.eta(t); J = fm.jac(t)
    truth = -((e - fm.y) / fm.sigma ** 2) @ J
    g = gfm.loglik_gradient(t)
    assert np.allclose(g, truth, rtol=1e-4, atol=1e-6)


def test_fd_oracle_eta_jacobian():
    fm = _AnalyticFM(y=[0, 0, 0], sigma=[1, 1, 1])
    gfm = GradientForwardModel(fm, n_obs=3, h=1e-5)
    t = np.array([0.4, 0.3])
    Jfd = gfm.eta_jacobian(t)
    assert np.allclose(Jfd, fm.jac(t), rtol=1e-4, atol=1e-6)
    assert Jfd.shape == (3, 2)


# --------------------------------------------------------------------------- #
# Prior gradient
# --------------------------------------------------------------------------- #
def test_prior_logp_and_grad():
    prior = Prior(means=[0.3, 0.5], stds=[0.1, 0.2],
                  lower=[0.0, 0.0], upper=[1.0, 1.0])
    t = np.array([0.35, 0.4])
    lp, g = _prior_logp_and_grad(prior, t)
    z = (t - prior.means) / prior.stds
    assert np.isclose(lp, -0.5 * np.sum(z ** 2))
    assert np.allclose(g, -(t - prior.means) / prior.stds ** 2)
    # out of bounds -> -inf, zero gradient
    lp2, g2 = _prior_logp_and_grad(prior, np.array([1.5, 0.4]))
    assert lp2 == -np.inf and np.all(g2 == 0)


def test_surrogate_logpost_grad_callable(rs):
    rng = np.random.default_rng(1)
    d, n = 3, 90
    X = rng.uniform(0.1, 0.9, (n, d))
    y = -np.sum((X - 0.5) ** 2, axis=1) + 0.01 * rng.standard_normal(n)
    gp = GPSurrogate(); gp.train(X, y, optimize_restarts=3)
    prior = Prior(means=[0.5] * d, stds=[0.15] * d,
                  lower=[0.1] * d, upper=[0.9] * d)
    lpg = make_surrogate_logpost_grad(prior, gp)
    lp, g = lpg(np.array([0.5, 0.5, 0.5]))
    assert np.isfinite(lp) and g.shape == (d,) and np.all(np.isfinite(g))
    lp2, g2 = lpg(np.array([2.0, 0.5, 0.5]))   # out of bounds
    assert lp2 == -np.inf


# --------------------------------------------------------------------------- #
# Checkpointed gradient dataset (active-subspace job array)
# --------------------------------------------------------------------------- #
def test_gradient_dataset_checkpoint(tmp_path):
    fm = _AnalyticFM(y=[0.2, 0.1, 0.0], sigma=[0.2, 0.2, 0.2])
    gfm = GradientForwardModel(fm, n_obs=3, h=1e-5)
    prior = Prior(means=[0.3, 0.3], stds=[0.2, 0.2],
                  lower=[0.0, 0.0], upper=[1.0, 1.0])
    thetas = np.array([[0.3, 0.3], [0.4, 0.2], [0.25, 0.35], [0.5, 0.5]])
    cache = tmp_path / "grads.npz"
    G1, v1 = gradient_dataset(gfm, prior, thetas, cache, verbose=False)
    assert cache.exists() and v1.all() and G1.shape == (4, 2)

    # resumable: sabotage the model; a re-run must load cache, not recompute.
    gfm.fm = None
    G2, v2 = gradient_dataset(gfm, prior, thetas, cache, verbose=False)
    assert np.array_equal(G1, G2) and v2.all()
