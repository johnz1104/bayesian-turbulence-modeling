"""
Regression contracts for the identifiability suite.

Planted synthetic cases verify each diagnostic and their reconciliation:
a function active in a known low-rank subspace must be recovered (rank + directions),
the posterior-precision spectrum must match a known covariance, and ARD relevance must
rank a known-influential coefficient first.  No CFD.
"""

from __future__ import annotations

import numpy as np

from identifiability import (active_subspace, posterior_covariance_eigenspectrum,
                             ard_relevance, effective_rank, principal_angles,
                             reconcile)


def _planted_gradients(d=6, n=400, seed=0):
    """∇ of f = sin(a1·θ) + 0.5(a2·θ)²  -> gradients lie in span{a1, a2}."""
    rng = np.random.default_rng(seed)
    a1 = np.zeros(d); a1[0] = 1.0
    a2 = np.zeros(d); a2[1] = 1.0; a2[2] = 1.0; a2 /= np.linalg.norm(a2)
    thetas = rng.uniform(-1, 1, (n, d))
    G = (np.cos(thetas @ a1)[:, None] * a1[None, :]
         + (thetas @ a2)[:, None] * a2[None, :])
    return G, np.column_stack([a1, a2])


def test_active_subspace_recovers_planted_rank_and_directions():
    G, planted = _planted_gradients()
    vals, vecs, norm = active_subspace(G)
    # exactly two non-negligible eigenvalues
    assert effective_rank(vals)["rank"] == 2
    assert norm[2] < 1e-6, f"3rd eigenvalue not negligible: {norm[2]}"
    # recovered 2-D subspace aligns with the planted one (small principal angles)
    ang = np.degrees(principal_angles(vecs[:, :2], planted))
    assert np.all(ang < 5.0), f"principal angles too large: {ang}"


def test_inactive_directions_recovered():
    G, _ = _planted_gradients(d=6)
    rep = reconcile([f"c{i}" for i in range(6)], gradients=G)
    # coefficients 3,4,5 are inactive -> tiny loadings in the dominant direction
    dom = rep["dominant_direction"]
    for inactive in ("c3", "c4", "c5"):
        assert dom[inactive] < 0.1, (inactive, dom[inactive])


def test_posterior_precision_matches_known_covariance():
    rng = np.random.default_rng(0)
    true_cov = np.diag([1.0, 0.01, 0.25])     # c1 well-constrained, c0 poorly
    samples = rng.multivariate_normal(np.zeros(3), true_cov, size=20000)
    vals, vecs, _ = posterior_covariance_eigenspectrum(samples, use_precision=True)
    # precision eigenvalues ~ 1/variance: largest ~ 1/0.01 = 100
    assert np.isclose(vals[0], 100.0, rtol=0.15)
    assert np.isclose(vals[-1], 1.0, rtol=0.15)


def test_ard_relevance_ranks_influential_first():
    ls = np.array([0.1, 10.0, 5.0])           # coeff 0 short ℓ -> most relevant
    r = ard_relevance(ls)
    assert np.argmax(r) == 0
    assert np.isclose(np.sum(r), 1.0)


def test_effective_rank_energy_and_gap():
    spec = np.array([10.0, 8.0, 0.01, 0.001, 1e-5])   # clear rank-2 with a big gap
    er = effective_rank(spec, energy=0.95, gap_factor=10.0)
    assert er["rank"] == 2


def test_reconcile_three_way_consistency():
    G, _ = _planted_gradients(d=5, n=400)
    names = [f"c{i}" for i in range(5)]
    # posterior samples constrained in 2 directions (small variance), loose elsewhere
    rng = np.random.default_rng(1)
    cov = np.diag([0.01, 0.01, 1.0, 1.0, 1.0])
    samples = rng.multivariate_normal(np.zeros(5), cov, size=8000)
    ls = np.array([0.1, 0.1, 8.0, 8.0, 8.0])  # ARD: first two relevant
    rep = reconcile(names, samples=samples, gradients=G, lengthscales=ls)
    assert set(rep["methods"]) == {"posterior", "active_subspace", "ard"}
    assert rep["ranks_consistent"], rep["ranks"]
    assert rep["rank_consensus"] == 2
