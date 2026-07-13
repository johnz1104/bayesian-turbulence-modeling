"""Calibrated-UQ wiring on the channel harness (generalized Bayes + conformal).

These tests inject a synthetic ensemble (no forward solves) into the channel
calibration harness so the UQ wiring is exercised quickly: the surrogates, the
power-posterior tempering, the posterior predictive, conformal, and the coverage
scoring. The synthetic ensemble is deliberately biased (no coefficient can match
the truth), which is the misspecification that makes standard Bayes overconfident
and the correction necessary. The real forward-solve calibration runs in the
reproduce script. Needs the built extension and the DNS data (for the case
geometry and the QoI truth), but no solving.
"""
import numpy as np
import pytest

from UQ.datasets import ChannelDNS

pytestmark = pytest.mark.skipif(
    not ChannelDNS.is_available(180),
    reason="channel DNS_data not present (bulk data is local/gitignored)",
)


def _inject_synthetic_ensemble(c, n=60, bias=0.02, seed=0):
    """A biased, theta-dependent synthetic ensemble fed to the surrogates.

    predictions(theta) = truth + bias + sensitivity . (theta - center): an affine
    response in the coefficients plus an irreducible bias, so the best fit still
    leaves a residual the predictive band must cover.
    """
    rng = np.random.default_rng(seed)
    lo, hi = c.prior.lower, c.prior.upper
    X = rng.uniform(lo, hi, size=(n, c.ndim))
    center = 0.5 * (lo + hi)
    # one random affine sensitivity vector per QoI
    sens = rng.normal(scale=0.3, size=(c.ndim, c.n_qoi))
    preds = (c.qoi_truth[None, :] + bias
             + (X - center) @ sens
             + 0.001 * rng.normal(size=(n, c.n_qoi)))
    resid = (preds - c.qoi_truth[None, :]) / c.qoi_sigma[None, :]
    loglik = -0.5 * np.sum(resid ** 2, axis=1)
    c.X, c.preds, c.loglik = X, preds, loglik
    c.n_valid = n
    c.fit_surrogates(noise_floor=1e-3)


@pytest.fixture(scope="module")
def calib(rs):
    from UQ.datasets.channel_calibration import ChannelCalibration
    dns = ChannelDNS.load(180)
    c = ChannelCalibration(dns, n_stations=12)
    _inject_synthetic_ensemble(c)
    return c


def test_observation_setup(calib):
    """Cf plus one velocity QoI per station, with floored DNS-stdev sigmas."""
    c = calib
    assert c.n_qoi == len(c.y_delta) + 1            # Cf + velocity stations
    assert c.qoi_names[0] == "Cf"
    assert np.all(c.qoi_sigma[1:] >= 0.005 - 1e-12)  # velocity-station sigma floor
    assert c.qoi_sigma[0] == pytest.approx(0.05 * c.qoi_truth[0])  # Cf is 5% relative
    assert 0.0 < c.qoi_truth[0] < 0.02              # Cf magnitude
    assert np.all(np.diff(c.y_delta) > 0)           # stations ordered outward


def test_standard_bayes_is_overconfident(calib):
    """At eta = 1 the predictive band under-covers the biased truth."""
    c = calib
    post = c.sample_posterior(eta=1.0, n_steps=800, burn_in=200)
    pp = c.posterior_predictive(post, eta=1.0)
    cov, _ = c.coverage_vs_truth(pp, level=0.9)
    assert cov < 0.9                                 # overconfident


def test_generalized_bayes_restores_coverage(calib):
    """Tempering (eta < 1) widens the predictive and raises coverage."""
    c = calib
    post1 = c.sample_posterior(eta=1.0, n_steps=800, burn_in=200)
    cov1, sharp1 = c.coverage_vs_truth(c.posterior_predictive(post1, eta=1.0), level=0.9)
    eta = c.calibrate_eta(post1, np.arange(c.n_qoi))
    assert 0.0 < eta <= 1.0
    post_t = c.sample_posterior(eta=eta, n_steps=800, burn_in=200)
    cov_t, sharp_t = c.coverage_vs_truth(c.posterior_predictive(post_t, eta=eta), level=0.9)
    assert cov_t > cov1                              # coverage improves
    assert sharp_t >= sharp1                         # at wider intervals


def test_posterior_predictive_inflates_with_temper(calib):
    """sigma/sqrt(eta) makes the eta<1 predictive strictly wider than eta=1."""
    c = calib
    post = c.sample_posterior(eta=1.0, n_steps=600, burn_in=200)
    w1 = np.std(c.posterior_predictive(post, eta=1.0, seed=3), axis=0).mean()
    w2 = np.std(c.posterior_predictive(post, eta=0.1, seed=3), axis=0).mean()
    assert w2 > w1


def test_conformal_coverage_runs_and_bounds(calib):
    """Split conformal returns a coverage in [0,1] and a finite half-width."""
    c = calib
    post = c.sample_posterior(eta=1.0, n_steps=600, burn_in=200)
    idx = np.arange(c.n_qoi)
    cal = idx[::2]
    test = idx[1::2]
    pc = c.point_prediction(post, cal)
    pt = c.point_prediction(post, test)
    cov, half, gap = c.conformal_coverage(pc, pt, cal, test, alpha=0.1)
    assert 0.0 <= cov <= 1.0
    assert half > 0.0
    assert abs(gap - (0.9 - cov)) < 1e-9


def test_cross_re_conformal_roles_are_disjoint(rs):
    """The conformal leg's case roles never overlap when train allows it.

    Three synthetic calibrations: fit cases, the calibration case, and the
    test case must be pairwise disjoint (the untouched-calibration-set
    requirement at the case level), and the calibration case defaults to the
    training Re nearest the held-out one. A single-case train falls back to
    the shared-case design and says so.
    """
    from UQ.datasets.channel_calibration import ChannelCalibration, CrossReStudy
    dns = ChannelDNS.load(180)
    cals = {}
    for i, re_label in enumerate((180, 550, 1000)):
        c = ChannelCalibration(dns, n_stations=8)
        _inject_synthetic_ensemble(c, n=40, seed=i)
        cals[re_label] = c
    study = CrossReStudy(cals)
    out = study.predict_heldout((180, 550), 1000, level=0.9, seed=0)
    assert out["conformal_roles_disjoint"]
    assert out["cal_case"] == 550, "default calibration case is nearest in Re"
    assert out["cal_case"] not in out["fit_cases"]
    assert 1000 not in out["fit_cases"] and 1000 != out["cal_case"]
    assert 0.0 <= out["conformal_coverage"] <= 1.0
    out1 = study.predict_heldout((550,), 1000, level=0.9, seed=0)
    assert not out1["conformal_roles_disjoint"]
    assert out1["fit_cases"] == [550]
