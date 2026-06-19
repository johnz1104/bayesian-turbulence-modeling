"""
Regression contract for the analytic gradient of the KOH log-likelihood.

Verifies ``KOHLikelihood.gradient`` (used to make the augmented [θ…, log σ_δ, log l_δ]
posterior differentiable for NUTS) against finite differences of ``KOHLikelihood.__call__``
in both modes.  Acceptance: rel. err < 1e-4 (the §6.2 KOH-gradient gate).  No CFD.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_inference import KOHLikelihood


def _setup(mode):
    rng = np.random.default_rng(0)
    n = 6
    locs = np.linspace(0.0, 5.0, n)
    y = np.array([1.0, 0.9, 0.7, 0.65, 0.6, 0.55])
    sig = np.full(n, 0.05)
    koh = KOHLikelihood(locs, y, sig, mode=mode)
    eta = y + 0.03 * rng.standard_normal(n)      # predictions near the data
    return koh, eta


def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


@pytest.mark.parametrize("mode", ["physical_gp", "diagonal"])
def test_koh_gradient_matches_fd(mode):
    koh, eta = _setup(mode)
    lsd, lld = -1.5, 0.2
    deta, g_s, g_l = koh.gradient(eta, lsd, lld)

    # dL/deta vs FD (component-wise)
    h = 1e-6
    deta_fd = np.zeros(koh.n)
    for j in range(koh.n):
        e = np.zeros(koh.n); e[j] = h
        deta_fd[j] = (koh(eta + e, lsd, lld) - koh(eta - e, lsd, lld)) / (2 * h)
    rel_eta = np.linalg.norm(deta - deta_fd) / max(np.linalg.norm(deta_fd), 1e-12)
    assert rel_eta < 1e-4, f"dL/deta rel err {rel_eta:.2e}"

    # dL/dlog_sigma_delta vs FD
    g_s_fd = _fd(lambda s: koh(eta, s, lld), lsd, 1e-6)
    assert abs(g_s - g_s_fd) / max(abs(g_s_fd), 1e-9) < 1e-4, (g_s, g_s_fd)

    # dL/dlog_l_delta vs FD
    g_l_fd = _fd(lambda L: koh(eta, lsd, L), lld, 1e-6)
    if mode == "physical_gp":
        assert abs(g_l - g_l_fd) / max(abs(g_l_fd), 1e-9) < 1e-4, (g_l, g_l_fd)
    else:
        # diagonal mode: lengthscale has no effect -> both ~0
        assert abs(g_l) < 1e-10 and abs(g_l_fd) < 1e-6


def test_koh_gradient_handles_nonfinite():
    koh, eta = _setup("physical_gp")
    deta, g_s, g_l = koh.gradient(np.full(koh.n, np.nan), -1.0, 0.0)
    assert np.all(deta == 0) and g_s == 0.0 and g_l == 0.0


def test_koh_discrepancy_mean_formula():
    # discrepancy reconstruction primitive: δ̂ = σ_δ² K C⁻¹ (y−η)
    koh, eta = _setup("physical_gp")
    lsd, lld = -1.0, 0.3
    delta = koh.discrepancy_mean(eta, lsd, lld)
    sd = np.exp(lsd)
    K = koh._kernel(np.exp(lld))
    C = sd ** 2 * K + koh._Sigma_eps + 1e-10 * koh._I
    manual = sd ** 2 * (K @ np.linalg.solve(C, koh.y - eta))
    assert np.allclose(delta, manual)
    assert np.all(koh.discrepancy_mean(np.full(koh.n, np.nan), lsd, lld) == 0)
