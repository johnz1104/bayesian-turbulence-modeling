"""
CFD-free demonstration contracts:
  * posterior-as-prior transfer measurably reduces high-fidelity evaluations, and
  * the KOH marginal likelihood composes as the per-β integrand inside evidence.py.
"""

from __future__ import annotations

import numpy as np

from bayesian_inference import Prior, KOHLikelihood, GPSurrogate, latin_hypercube
from hierarchical import fit_transfer_prior, koh_evidence_loglik
from evidence import thermodynamic_integration, stepping_stone


def test_transfer_prior_reduces_evals_for_target_accuracy():
    """
    A transfer prior fit to a low-regime posterior spends the high-fidelity ensemble
    *where the posterior lives* instead of across a broad cold box — so for a target
    posterior accuracy it needs far fewer evaluations.  Measured by a deterministic
    concentration metric (no GP/importance-sampling noise): at a fixed budget the
    transfer-prior LHS design places many more points near θ* and far fewer are wasted.
    """
    rng = np.random.default_rng(0)
    theta_star = np.array([0.31, 0.09])
    lo, hi = np.array([0.1, 0.01]), np.array([0.8, 0.2])
    cold = Prior(means=[0.45, 0.1], stds=[0.2, 0.06], lower=lo, upper=hi)   # broad
    low_post = theta_star + 0.02 * rng.standard_normal((2000, 2))           # low-regime
    warm = fit_transfer_prior(low_post, lower=lo, upper=hi, inflate=1.5)

    n = 24
    np.random.seed(0)
    Xc = latin_hypercube(n, 2, cold.lower, cold.upper)
    np.random.seed(0)
    Xw = latin_hypercube(n, 2, warm.lower, warm.upper)
    dc = np.linalg.norm(Xc - theta_star, axis=1)
    dw = np.linalg.norm(Xw - theta_star, axis=1)

    # transfer concentrates the ensemble near the posterior (median distance halved+)
    assert np.median(dw) < 0.5 * np.median(dc), (np.median(dw), np.median(dc))
    # and yields many more "useful" high-fidelity points (near θ*) at the same budget
    near_w = int(np.sum(dw < 0.08))
    near_c = int(np.sum(dc < 0.08))
    assert near_w > 2 * max(near_c, 1), (near_w, near_c)


def test_koh_composes_as_evidence_integrand():
    """KOH log-marginal-likelihood as the per-β integrand → finite log Z; TI≈SS."""
    rng = np.random.default_rng(0)
    locs = np.linspace(0, 4, 5)
    A = 0.05 * rng.standard_normal((5, 2))
    theta_true = np.array([0.3, 0.1])
    y = A @ theta_true + 0.02 * rng.standard_normal(5)
    koh = KOHLikelihood(locs, y, np.full(5, 0.03), mode="physical_gp")

    def eta_fn(theta):
        return A @ np.asarray(theta)

    n_theta = 2
    loglik = koh_evidence_loglik(koh, eta_fn, n_theta)

    # augmented prior over [θ(2), log σ_δ, log l_δ]
    pm = np.array([0.3, 0.1, -2.0, 0.0])
    psd = np.array([0.1, 0.05, 1.0, 1.0])

    def logprior(ext):
        z = (np.asarray(ext) - pm) / psd
        return -0.5 * np.sum(z * z) - np.sum(np.log(psd * np.sqrt(2 * np.pi)))

    sk = dict(n_steps=700, burn=250, thin=2)
    ti_Z, ti_se, _ = thermodynamic_integration(loglik, logprior, pm,
                                               sampler_kwargs=sk, rng_seed=0)
    ss_Z, ss_se, _ = stepping_stone(loglik, logprior, pm,
                                    sampler_kwargs=sk, rng_seed=0)
    assert np.isfinite(ti_Z) and np.isfinite(ss_Z)
    # the two estimators agree within a few combined SE (KOH composes cleanly)
    assert abs(ti_Z - ss_Z) < max(3.0 * np.hypot(ti_se, ss_se), 1.0), (ti_Z, ss_Z)
