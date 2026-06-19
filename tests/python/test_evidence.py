"""
Regression contracts for model-evidence estimators.

Anchored on the analytic-Gaussian case where Z is closed form: both thermodynamic
integration and stepping-stone must recover it, and must agree with each other.  No
CFD is used (the estimators consume analytic log-lik / log-prior callables).
"""

from __future__ import annotations

import numpy as np
import pytest

from evidence import (power_betas, thermodynamic_integration, stepping_stone,
                      log_evidence, bayes_factor, analytic_gaussian_log_evidence)


def test_power_betas_schedule():
    b = power_betas(20, p=5.0)
    assert b[0] == 0.0 and np.isclose(b[-1], 1.0)
    assert np.all(np.diff(b) > 0)                 # strictly increasing
    # denser near 0: first gap smaller than last gap
    assert (b[1] - b[0]) < (b[-1] - b[-2])


def test_analytic_gaussian_log_evidence_formula():
    y = np.array([0.3, -0.4])
    sigma2, tau2 = 0.5, 1.0
    z = analytic_gaussian_log_evidence(y, sigma2, tau2, 2)
    s = sigma2 + tau2
    manual = -0.5 * (2 * np.log(2 * np.pi * s) + np.sum(y ** 2) / s)
    assert np.isclose(z, manual)


def test_bayes_factor_and_jeffreys():
    # ln B = 2.5 -> log10 ~ 1.086 -> "strong (favors M1)"
    log10B, label = bayes_factor(2.5, 0.0)
    assert np.isclose(log10B, 2.5 / np.log(10))
    assert "strong" in label and "M1" in label
    # reverse favors M2
    _, label2 = bayes_factor(0.0, 5.0)
    assert "M2" in label2 and "decisive" in label2


def test_evidence_selects_generating_model():
    """
    Recovery check: data simulated from closure A → the evidence must prefer A
    over a structurally different closure B (the §6.2 model-selection recovery).
    Analytic linear 'closures' (no CFD) so it is deterministic and unbiased.
    """
    pytest.importorskip("emcee")
    rng = np.random.default_rng(0)
    sigma = 0.05
    A = np.array([[1.0, 0.2], [0.3, 0.9], [0.5, 0.5]])     # generating closure
    B = np.array([[1.0, -0.5], [0.8, 0.2], [0.1, 1.0]])    # different closure
    theta_true = np.array([0.4, 0.3])
    y = A @ theta_true + sigma * rng.standard_normal(3)

    pm, psd = np.array([0.4, 0.3]), np.array([0.3, 0.3])

    def logprior(t):
        z = (t - pm) / psd
        return -0.5 * np.sum(z * z) - np.sum(np.log(psd * np.sqrt(2 * np.pi)))

    def make_ll(M):
        def ll(t):
            d = y - M @ t
            return (-0.5 * np.sum(d * d) / sigma ** 2
                    - 3 * np.log(sigma * np.sqrt(2 * np.pi)))
        return ll

    sk = dict(n_steps=900, burn=300, thin=2)
    ZA, seA, _ = thermodynamic_integration(make_ll(A), logprior, pm,
                                           sampler_kwargs=sk, rng_seed=0)
    ZB, seB, _ = thermodynamic_integration(make_ll(B), logprior, pm,
                                           sampler_kwargs=sk, rng_seed=0)
    # the generating closure A must win, by more than the combined SE
    assert ZA - ZB > 3.0 * np.hypot(seA, seB), (ZA, ZB)


@pytest.mark.parametrize("dim", [1, 2])
def test_ti_ss_recover_analytic_gaussian(dim):
    pytest.importorskip("emcee")
    rng = np.random.default_rng(0)
    sigma2, tau2 = 0.5, 1.0
    y = rng.standard_normal(dim) * np.sqrt(sigma2)

    def loglik(theta):
        d = y - theta
        return -0.5 * (dim * np.log(2 * np.pi * sigma2) + np.sum(d * d) / sigma2)

    def logprior(theta):
        return -0.5 * (dim * np.log(2 * np.pi * tau2) + np.sum(theta * theta) / tau2)

    truth = analytic_gaussian_log_evidence(y, sigma2, tau2, dim)
    res = log_evidence(loglik, logprior, np.zeros(dim), rng_seed=0,
                       sampler_kwargs=dict(n_steps=900, burn=300, thin=2))
    # both estimators within ~0.5 nat of the closed form
    assert abs(res["ti_logZ"] - truth) < 0.5, (res["ti_logZ"], truth)
    assert abs(res["ss_logZ"] - truth) < 0.5, (res["ss_logZ"], truth)
    # and agree with each other
    assert res["agree"], res
