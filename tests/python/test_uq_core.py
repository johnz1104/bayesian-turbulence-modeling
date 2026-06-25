"""Synthetic-data tests for the a-priori UQ scaffolding (numpy/scipy modules).

Each test pins the property the module must have, with a known answer:
realizability projection lands in the barycentric triangle, discrepancy
extraction recovers a planted discrepancy, the scoring rules and calibration
diagnostics separate calibrated from miscalibrated predictors, conformal
prediction hits nominal coverage and exposes the shift gap, and generalized Bayes
restores coverage a misspecified standard-Bayes posterior loses.
"""
import numpy as np
import pytest

from UQ import realizability as rz
from UQ import discrepancy as dq
from UQ import synthetic as syn
from UQ import evaluation as ev
from UQ import conformal as cf
from UQ import generalized_bayes as gb
from UQ import dns_field as dnsf


# ---------------------------------------------------------------- realizability

def test_realizability_isotropic_unchanged():
    R = (2.0 / 3.0) * np.eye(3)[None]          # k = 1, isotropic
    assert bool(rz.is_realizable(R)[0])
    Rp = rz.project_reynolds_stress(R)
    assert np.allclose(Rp, R, atol=1e-10)


def test_realizability_projects_unrealizable_in():
    # a strongly one-component, slightly-too-extreme stress
    R = np.diag([2.9, 0.05, 0.05])[None]
    tr0 = np.trace(R[0])
    Rp = rz.project_reynolds_stress(R)
    assert np.isclose(np.trace(Rp[0]), tr0, atol=1e-7)        # 2k preserved
    assert bool(rz.is_realizable(Rp, tol=1e-7)[0])            # now realizable
    # eigenvalues of the projected anisotropy give non-negative barycentric coords
    c1, c2, c3 = rz.barycentric_coords(Rp)
    assert c1[0] > -1e-7 and c2[0] > -1e-7 and c3[0] > -1e-7


def test_realizability_random_batch_all_in():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(200, 3, 3))
    R = np.einsum("nij,nkj->nik", A, A) + 0.01 * np.eye(3)    # SPD-ish
    Rp = rz.project_reynolds_stress(R)
    assert np.all(rz.is_realizable(Rp, tol=1e-6))


# ------------------------------------------------------------------ discrepancy

def test_recover_planted_reynolds_discrepancy():
    d = syn.make_fake_dns(n=300, seed=3)
    db = dq.reynolds_discrepancy(d["R_dns"], d["S"])
    assert np.allclose(db, d["db_true"], atol=1e-9)


def test_recover_planted_heatflux_discrepancy():
    h = syn.make_fake_heatflux(n=300, seed=4)
    dqv = dq.heatflux_discrepancy(h["q_dns"], h["grad_T"], h["nu_t"], h["Pr_t"])
    assert np.allclose(dqv, h["dq_true"], atol=1e-12)


def test_dns_schema_adapter_recovers_discrepancies():
    """The DNS-schema interface recovers both planted discrepancies end to end."""
    field, db_true, dq_true = dnsf.fake_dns_field(n=300, seed=8)
    assert field.has_heat_flux()
    out = field.extract()
    assert np.allclose(out["reynolds_discrepancy"], db_true, atol=1e-9)
    assert np.allclose(out["heatflux_discrepancy"], dq_true, atol=1e-12)
    # features are the five Pope invariants here (no extras supplied)
    assert out["features"].shape == (300, 5)


def test_invariants_are_rotationally_invariant():
    rng = np.random.default_rng(5)
    g = rng.normal(size=(50, 3, 3))
    S, W = dq.strain_rotation(g, np.ones(50))
    inv0 = dq.invariants(S, W)
    # rotate every tensor by a fixed rotation: invariants must not change
    th = 0.7
    Q = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    Sr = np.einsum("ij,njk,lk->nil", Q, S, Q)
    Wr = np.einsum("ij,njk,lk->nil", Q, W, Q)
    inv1 = dq.invariants(Sr, Wr)
    assert np.allclose(inv0, inv1, atol=1e-10)


def test_integrity_basis_symmetric_traceless():
    rng = np.random.default_rng(6)
    g = rng.normal(size=(20, 3, 3))
    S, W = dq.strain_rotation(g, np.ones(20))
    T = dq.integrity_basis(S, W)                 # (20, 10, 3, 3)
    assert np.allclose(T, np.swapaxes(T, -1, -2), atol=1e-10)     # symmetric
    assert np.allclose(np.trace(T, axis1=-2, axis2=-1), 0.0, atol=1e-10)


# ------------------------------------------------------------------- evaluation

def _crps_gaussian(mu, sigma, y):
    from scipy.stats import norm
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def test_crps_matches_gaussian_closed_form():
    rng = np.random.default_rng(7)
    mu, sigma = 0.3, 1.2
    y = np.array([0.0, 1.0, -2.0, 0.3])
    samples = mu + sigma * rng.normal(size=(4, 60000))
    crps_emp = ev.crps_ensemble(y, samples)
    crps_true = np.mean([_crps_gaussian(mu, sigma, yi) for yi in y])
    assert abs(crps_emp - crps_true) < 0.02


def test_energy_score_reduces_to_crps_in_1d():
    rng = np.random.default_rng(8)
    y = np.array([0.0, 1.0])
    s = rng.normal(size=(2, 4000))
    crps = ev.crps_ensemble(y, s)
    es = ev.energy_score(y[:, None], s[:, :, None])
    assert abs(crps - es) < 0.02


def test_pit_and_reliability_separate_calibrated():
    rng = np.random.default_rng(9)
    n, m = 4000, 400
    y = rng.normal(size=n)
    cal = rng.normal(size=(n, m))                       # calibrated ensemble
    mis = 0.3 * rng.normal(size=(n, m))                 # too narrow -> overconfident
    assert ev.pit_uniformity_pvalue(ev.pit_values(y, cal)) > 0.05
    assert ev.pit_uniformity_pvalue(ev.pit_values(y, mis)) < 1e-3
    assert ev.reliability_error(y, cal) < 0.03
    assert ev.reliability_error(y, mis) > 0.1


def test_sbc_detects_miscalibration():
    rng = np.random.default_rng(10)
    T, L = 2000, 99
    # correct inference: theta_true and the posterior draws are exchangeable
    # (drawn from the same law), so the rank is uniform on {0..L} by construction.
    z = rng.normal(size=(T, L + 1))
    theta_true = z[:, 0]
    good = z[:, 1:]
    # miscalibrated: an overconfident posterior centred at 0 that ignores the
    # truth's spread -> ranks pile at the extremes.
    bad_truth = rng.normal(size=T)
    bad = 0.3 * rng.normal(size=(T, L))
    assert ev.sbc_uniformity_pvalue(ev.sbc_ranks(theta_true, good), L) > 0.05
    assert ev.sbc_uniformity_pvalue(ev.sbc_ranks(bad_truth, bad), L) < 1e-3


# -------------------------------------------------------------------- conformal

def test_split_conformal_nominal_coverage():
    rng = np.random.default_rng(11)
    n = 4000
    x = rng.uniform(-2, 2, n)
    y = np.sin(x) + 0.3 * rng.normal(size=n)
    pred = np.sin(x)                                    # the (good) point predictor
    cal = slice(0, n // 2)
    test = slice(n // 2, n)
    lo, hi = cf.split_conformal_intervals(pred[cal], y[cal], pred[test], alpha=0.1)
    cov = float(np.mean((y[test] >= lo) & (y[test] <= hi)))
    assert 0.86 < cov < 0.94


def test_conformal_coverage_gap_under_shift():
    rng = np.random.default_rng(12)
    n = 3000
    xc = rng.uniform(-2, 2, n); yc = np.sin(xc) + 0.3 * rng.normal(size=n)
    # shifted test set: larger noise the calibration never saw
    xt = rng.uniform(-2, 2, n); yt = np.sin(xt) + 1.2 * rng.normal(size=n)
    cov, gap = cf.conformal_coverage_gap(np.sin(xc), yc, np.sin(xt), yt, alpha=0.1)
    assert gap > 0.05            # honest under-coverage on the shifted data


def test_cqr_nominal_coverage():
    rng = np.random.default_rng(13)
    n = 5000
    x = rng.uniform(-2, 2, n)
    sigma = 0.2 + 0.4 * (x + 2) / 4                     # heteroscedastic
    y = np.sin(x) + sigma * rng.normal(size=n)
    cal = slice(0, n // 2); test = slice(n // 2, n)
    # oracle-ish conditional quantiles (the model the conformaliser corrects)
    from scipy.stats import norm
    qlo = np.sin(x) + norm.ppf(0.05) * sigma
    qhi = np.sin(x) + norm.ppf(0.95) * sigma
    lo, hi = cf.cqr_intervals(qlo[cal], qhi[cal], y[cal], qlo[test], qhi[test], alpha=0.1)
    cov = float(np.mean((y[test] >= lo) & (y[test] <= hi)))
    assert 0.86 < cov < 0.94


# ------------------------------------------------------------ generalized Bayes

def test_generalized_bayes_restores_coverage():
    # misspecified Gaussian: the model assumes sigma0 but the truth is larger.
    sigma0, sigma_true, n = 1.0, 2.0, 40
    theta_true = 0.0
    rng = np.random.default_rng(14)
    datasets = [theta_true + sigma_true * rng.normal(size=n) for _ in range(600)]

    cov_std = gb.credible_interval_coverage(datasets, theta_true, sigma0, eta=1.0)
    assert cov_std < 0.75                # standard Bayes is overconfident

    # moment-matched learning rate per dataset -> recovers eta* ~ (sigma0/sigma_true)^2
    etas = [gb.calibrate_eta_moment(d, sigma0) for d in datasets]
    assert abs(np.mean(etas) - (sigma0 / sigma_true) ** 2) < 0.06

    cov_temp = np.mean([
        (lambda m, v: (m - 1.645 * np.sqrt(v) <= theta_true <= m + 1.645 * np.sqrt(v)))(
            *gb.power_posterior_gaussian(d, sigma0, eta=e))
        for d, e in zip(datasets, etas)])
    assert cov_temp > 0.85               # tempered posterior is calibrated
