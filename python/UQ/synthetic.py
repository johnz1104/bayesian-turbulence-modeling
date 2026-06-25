"""Synthetic fake-DNS fields with a known, recoverable model-form discrepancy.

No real DNS is used anywhere in the program; correctness of the UQ machinery is
established on synthetic data whose ground truth is known. This module plants:

  - a deterministic Reynolds-stress discrepancy that ``discrepancy`` must recover
    exactly from the synthetic Reynolds-stress field, and
  - a stochastic, non-Gaussian, heteroscedastic, bimodal conditional discrepancy
    that the generative model must learn and whose every sample must be realizable.
"""
import numpy as np

from . import discrepancy as dq

# A fixed symmetric, traceless "shape" tensor for the planted discrepancy.
_D0 = np.array([[0.20, 0.10, 0.0],
                [0.10, -0.05, 0.0],
                [0.0, 0.0, -0.15]])
_D0 = 0.5 * (_D0 + _D0.T)
_D0 = _D0 - np.trace(_D0) / 3.0 * np.eye(3)


def _random_traceless_gradients(n, rng, scale=1.0):
    """Random divergence-free (traceless) velocity-gradient tensors."""
    g = rng.normal(scale=scale, size=(n, 3, 3))
    tr = np.trace(g, axis1=-2, axis2=-1) / 3.0
    g = g - tr[..., None, None] * np.eye(3)
    return g


def planted_alpha(features):
    """Known scalar amplitude of the planted discrepancy as a function of the
    first two Galilean invariants (smooth, bounded)."""
    l1, l2 = features[..., 0], features[..., 1]
    return 0.3 * np.tanh(0.5 * l1) + 0.15 * np.cos(0.8 * l2)


def make_fake_dns(n=400, seed=0, k=1.0):
    """Construct a synthetic DNS field with a planted Reynolds-stress discrepancy.

    Returns a dict with the velocity gradient, timescale, turbulence energy, the
    full Reynolds-stress tensor, and the GROUND-TRUTH anisotropy discrepancy that
    ``discrepancy.reynolds_discrepancy`` must recover.
    """
    rng = np.random.default_rng(seed)
    grad_u = _random_traceless_gradients(n, rng)
    timescale = np.full(n, 1.0)
    S, W = dq.strain_rotation(grad_u, timescale)
    feats = dq.invariants(S, W)

    b_bouss = dq.boussinesq_anisotropy(S)
    db_true = planted_alpha(feats)[..., None, None] * _D0
    b_dns = b_bouss + db_true
    R_dns = 2.0 * k * (b_dns + np.eye(3) / 3.0)
    return {
        "grad_u": grad_u,
        "timescale": timescale,
        "k": np.full(n, k),
        "S": S,
        "features": feats,
        "R_dns": R_dns,
        "db_true": db_true,
    }


def make_fake_heatflux(n=400, seed=1, Pr_t=0.9):
    """Synthetic heat-flux field with a planted gradient-diffusion discrepancy."""
    rng = np.random.default_rng(seed)
    grad_T = rng.normal(size=(n, 3))
    nu_t = 0.05 + 0.02 * rng.random(n)
    # planted non-gradient-aligned part (Reynolds-analogy breakdown)
    dq_true = np.stack([0.1 * np.sin(grad_T[:, 0]),
                        0.05 * grad_T[:, 1] ** 2 - 0.05,
                        np.zeros(n)], axis=-1)
    q_gdh = dq.gdh_heat_flux(grad_T, nu_t, Pr_t)
    q_dns = q_gdh + dq_true
    return {"grad_T": grad_T, "nu_t": nu_t, "q_dns": q_dns,
            "dq_true": dq_true, "Pr_t": Pr_t}


def conditional_discrepancy_dataset(n=4000, seed=2):
    """A non-Gaussian, heteroscedastic, BIMODAL conditional p(y | x).

    The feature x in [-2, 2] indexes the local flow state; the 2-D target y is a
    stand-in for two anisotropy-discrepancy components. The conditional is an
    equal-weight mixture of two branches whose separation and spread both vary
    with x, so it is multimodal where the branches are far apart and nearly
    unimodal where they merge. The generative model must recover this shape.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n, 1))
    comp = rng.random(n) < 0.5
    sep = 0.8 + 0.6 * np.abs(x[:, 0])            # branch separation grows with |x|
    spread = 0.10 + 0.15 * (1.0 + np.cos(np.pi * x[:, 0]))   # heteroscedastic
    mu1 = np.sin(1.5 * x[:, 0]) + sep
    mu2 = np.sin(1.5 * x[:, 0]) - sep
    y0 = np.where(comp, mu1, mu2) + spread * rng.normal(size=n)
    # second component correlated with the first branch choice
    y1 = 0.5 * x[:, 0] + np.where(comp, 0.4, -0.4) + 0.12 * rng.normal(size=n)
    y = np.stack([y0, y1], axis=-1)
    return x, y


def true_conditional_samples(x_value, m, seed=7):
    """Draw m reference samples from the true p(y | x = x_value) for comparison."""
    rng = np.random.default_rng(seed)
    comp = rng.random(m) < 0.5
    sep = 0.8 + 0.6 * abs(x_value)
    spread = 0.10 + 0.15 * (1.0 + np.cos(np.pi * x_value))
    mu1 = np.sin(1.5 * x_value) + sep
    mu2 = np.sin(1.5 * x_value) - sep
    y0 = np.where(comp, mu1, mu2) + spread * rng.normal(size=m)
    y1 = 0.5 * x_value + np.where(comp, 0.4, -0.4) + 0.12 * rng.normal(size=m)
    return np.stack([y0, y1], axis=-1)
