"""
Regression contracts for hierarchical/transfer + KOH integration.

Covers the posterior-as-prior transfer fit, the KOH-as-evidence-integrand composition,
the augmented KOH gradient log-posterior for NUTS (FD-checked), and the shared-vs-
case-specific discrepancy verdict.  No CFD.
"""

from __future__ import annotations

import numpy as np

from bayesian_inference import Prior, KOHLikelihood
from hierarchical import (fit_transfer_prior, koh_evidence_loglik,
                          koh_logpost_grad, compare_discrepancies)


class _PS:
    def lower_bounds(self): return [0.1, 0.01]
    def upper_bounds(self): return [0.8, 0.2]


def test_fit_transfer_prior():
    rng = np.random.default_rng(0)
    samp = rng.normal([0.31, 0.09], [0.02, 0.005], size=(4000, 2))
    pr = fit_transfer_prior(samp, param_set=_PS())
    assert np.allclose(pr.means, [0.31, 0.09], atol=0.01)
    assert np.allclose(pr.stds, [0.02, 0.005], atol=0.003)
    # support narrowed to ~mean±4σ within the physical box (concentrated for transfer)
    assert np.allclose(pr.lower, np.maximum(pr.means - 4 * pr.stds, [0.1, 0.01]))
    assert np.allclose(pr.upper, np.minimum(pr.means + 4 * pr.stds, [0.8, 0.2]))
    assert np.all(pr.lower > [0.1, 0.01]) and np.all(pr.upper < [0.8, 0.2])


def test_koh_evidence_loglik_composition():
    locs = np.linspace(0, 4, 5)
    y = np.array([1.0, 0.9, 0.8, 0.75, 0.7])
    koh = KOHLikelihood(locs, y, np.full(5, 0.05), mode="physical_gp")
    A = np.random.default_rng(0).standard_normal((5, 2))
    eta_fn = lambda th: A @ th + y          # near data for finite ll
    loglik = koh_evidence_loglik(koh, eta_fn, n_theta=2)
    ext = np.array([0.0, 0.0, -1.5, 0.2])   # [theta(2), log_sigma, log_l]
    assert np.isfinite(loglik(ext))
    bad = np.array([np.inf, 0.0, -1.5, 0.2])
    assert loglik(bad) == -np.inf


def test_koh_logpost_grad_matches_fd():
    locs = np.linspace(0, 4, 5)
    y = np.array([1.0, 0.9, 0.8, 0.75, 0.7])
    koh = KOHLikelihood(locs, y, np.full(5, 0.05), mode="physical_gp")
    rng = np.random.default_rng(1)
    A = 0.1 * rng.standard_normal((5, 2))
    b = y.copy()
    eta_fn = lambda th: A @ th + b
    eta_jac = lambda th: A
    # augmented prior: theta(2) + [log_sigma_delta, log_l_delta]
    prior = Prior(means=[0.3, 0.1, -1.5, 0.0], stds=[0.1, 0.05, 1.0, 1.0],
                  lower=[0.0, 0.0, -6, -4], upper=[1.0, 1.0, 4, 4])
    lpg = koh_logpost_grad(koh, eta_fn, eta_jac, prior, n_theta=2)
    x = np.array([0.32, 0.11, -1.2, 0.3])
    lp, g = lpg(x)
    # 4th-order FD of the log-posterior
    def f(z):
        return lpg(z)[0]
    g_fd = np.zeros(4)
    h = 1e-5
    for j in range(4):
        e = np.zeros(4); e[j] = h
        g_fd[j] = (-f(x + 2 * e) + 8 * f(x + e) - 8 * f(x - e) + f(x - 2 * e)) / (12 * h)
    rel = np.linalg.norm(g - g_fd) / max(np.linalg.norm(g_fd), 1e-12)
    assert rel < 1e-4, f"augmented KOH gradient rel err {rel:.2e}\n{g}\n{g_fd}"


def test_compare_discrepancies_verdict():
    rng = np.random.default_rng(0)
    shared = {"channel": rng.normal(0.05, 0.01, 500),
              "plate": rng.normal(0.052, 0.01, 500),
              "bfs": rng.normal(0.051, 0.01, 500)}
    assert compare_discrepancies(shared)["verdict"] == "shared"
    specific = {"channel": rng.normal(0.02, 0.004, 500),
                "bfs": rng.normal(0.15, 0.01, 500)}
    assert compare_discrepancies(specific)["verdict"] == "case-specific"
