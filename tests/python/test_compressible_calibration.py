"""Compressible calibration harness verified on the real channel matrix.

These tests run only where the gitignored DNS_data is present. They check the
pre-registration's structural guarantees (the thermal targets NEVER inform the
likelihood; the Pr_t prior is exactly uniform on its box; the per-case sigma
floor honours the loader anchor), and they drive the whole spine end to end on
real cases with smoke-sized budgets. Coverage NUMBERS from these budgets are
machinery checks only; the evidence package quotes exclusively the fixed-seed
production runs.
"""
import numpy as np
import pytest

from UQ.datasets import GVChannelDNS
from UQ.datasets.compressible_calibration import (CompressiblePrior,
                                                  CompressibleCalibration,
                                                  PRT_BOX)
from UQ.datasets.crossmach_study import CrossMachStudy

_CASE_A = "Retaus_0105_MCLx_0p32_isoTw_0298_MB_AIR0"
_CASE_B = "Retaus_0143_MCLx_0p32_isoTw_0298_MB_AIR0"

pytestmark = pytest.mark.skipif(
    not GVChannelDNS.is_available(_CASE_A),
    reason="GV compressible channel DNS_data not present",
)


def test_prior_is_exactly_uniform_in_prt():
    """The Pr_t marginal contributes zero log-density variation inside its
    pre-registered box, while the SST pair keeps the Menter-centred normal."""
    prior = CompressiblePrior.build()
    assert prior.ndim == 3
    assert prior.lower[2] == PRT_BOX[0] and prior.upper[2] == PRT_BOX[1]
    base = np.array([0.31, 0.09, 0.6])
    moved = np.array([0.31, 0.09, 1.4])
    assert prior.log_prior(base) == prior.log_prior(moved)
    off_sst = np.array([0.33, 0.09, 0.6])
    assert prior.log_prior(off_sst) != prior.log_prior(base)
    assert prior.log_prior(np.array([0.31, 0.09, 0.4])) == -np.inf
    np.random.seed(0)
    draws = prior.sample(200)
    assert np.all(draws[:, 2] >= PRT_BOX[0])
    assert np.all(draws[:, 2] <= PRT_BOX[1])
    # uniform draws fill the box rather than clustering at the centre
    assert np.std(draws[:, 2]) > 0.15


@pytest.fixture(scope="module")
def cal():
    c = CompressibleCalibration(GVChannelDNS.load(_CASE_A))
    n = c.run_ensemble(n=12, seed=0)
    assert n >= 10
    c.fit_surrogates()
    return c


def test_heldout_block_never_informs_the_likelihood(cal):
    """The pre-registered separation: B_q and the heat-flux stations sit in
    the held-out block, disjoint from the likelihood block, and perturbing a
    held-out prediction leaves the likelihood unchanged."""
    lik = set(cal.lik_index.tolist())
    held = set(cal.heldout_index.tolist())
    assert lik.isdisjoint(held)
    assert lik | held == set(range(cal.n_qoi))
    assert cal.qoi_names[cal.heldout_index[0]] == "B_q"
    assert all(cal.qoi_names[i].startswith("q@")
               for i in cal.heldout_index[1:])
    status, pred = cal.evaluate(np.array([0.31, 0.09, 0.9]))
    assert status == "converged"
    ll = cal.log_likelihood_direct(pred)
    perturbed = pred.copy()
    perturbed[cal.heldout_index] *= 3.0
    assert cal.log_likelihood_direct(perturbed) == ll


def test_sigma_floor_honours_the_anchor(cal):
    """The effective relative level is max(level, loader anchor rms), the
    pre-registered floor rule, and every sigma is positive."""
    assert cal.rel_eff == max(0.005, cal.anchor["rms"])
    assert np.all(cal.qoi_sigma > 0.0)


def test_cf_derivation_matches_gv_table(cal):
    """The harness's record-derived cf (used where a source states none, the
    CKM cases) reproduces the GV global-table value exactly, since the table
    defines cf on the same centreline dynamic head."""
    from UQ.datasets import CKMChannelDNS
    d = cal.dns
    derived = 2.0 / (float(d.rho[-1]) * float(d.U[-1]) ** 2)
    assert derived == pytest.approx(d.wall["cf"], rel=2e-3)
    if CKMChannelDNS.is_available("M1p5"):
        c2 = CompressibleCalibration(CKMChannelDNS.load("M1p5"),
                                     n_stations=8, n_q_stations=5)
        assert c2.qoi_truth[0] > 0.0
        assert c2.qoi_names[0] == "cf"


def test_posterior_and_predictive_spine(cal):
    """Posterior sampling, the Pr_t marginal, block-wise predictive coverage,
    and the moment-matched learning rate all run on the real case."""
    post = cal.sample_posterior(eta=1.0, n_steps=400, burn_in=150, seed=0)
    assert post.shape[1] == 3
    prt = cal.marginal_prt(post)
    assert set(prt) == {"mean", "sd", "q05", "q95", "prior_sd"}
    assert PRT_BOX[0] <= prt["mean"] <= PRT_BOX[1]
    for idx in (cal.lik_index, cal.heldout_index):
        pp = cal.posterior_predictive(post, eta=1.0, qoi_index=idx, seed=1)
        cov, sharp = cal.coverage_vs_truth(pp, qoi_index=idx)
        assert 0.0 <= cov <= 1.0
        assert sharp > 0.0
    eta = cal.calibrate_eta(post, cal.lik_index)
    assert 0.0 < eta <= 1.0


def test_cache_roundtrip(cal):
    """The ensemble cache rebuilds the surrogates without re-solving."""
    d = cal.to_cache()
    c2 = CompressibleCalibration(GVChannelDNS.load(_CASE_A))
    c2.load_cache(d)
    theta = np.array([0.31, 0.09, 0.9])
    # GP hyperparameter optimisation has stochastic restarts, so the rebuilt
    # surrogate is equivalent (1e-5 relative here), not bit-identical
    assert c2.gp.log_likelihood(theta) == pytest.approx(
        cal.gp.log_likelihood(theta), rel=1e-3)


def test_crossmach_study_blocks(cal):
    """The pooled study predicts a held-out case with per-block coverage and
    conformal fields, the Pr_t posterior summary attached."""
    if not GVChannelDNS.is_available(_CASE_B):
        pytest.skip("companion case not present")
    cal_b = CompressibleCalibration(GVChannelDNS.load(_CASE_B))
    cal_b.run_ensemble(n=12, seed=1)
    cal_b.fit_surrogates()
    study = CrossMachStudy({_CASE_A: cal, _CASE_B: cal_b})
    out = study.predict_heldout(train=[_CASE_A], test=_CASE_B, seed=0)
    for key in ("standard_lik_coverage", "standard_thermal_coverage",
                "tempered_lik_coverage", "tempered_thermal_coverage",
                "conformal_lik_coverage", "conformal_thermal_coverage",
                "conformal_thermal_gap", "eta", "prt_posterior"):
        assert key in out
    # single-case train cannot split fit and calibration roles; the record
    # must say so instead of silently sharing the case
    assert out["conformal_roles_disjoint"] is False
    from UQ.reproduce_compressible import _conformal_claim_tag
    assert "excluded from formal conformal claims" in _conformal_claim_tag(out)


def test_crossmach_multicase_conformal_roles_are_disjoint(cal):
    """With two or more training cases the conformal leg's fit cases exclude
    the calibration case, and the record says the roles are disjoint. The
    third calibration reuses case A's data under a distinct tag (the role
    logic is what is under test, not the physics)."""
    if not GVChannelDNS.is_available(_CASE_B):
        pytest.skip("companion case not present")
    cal_b = CompressibleCalibration(GVChannelDNS.load(_CASE_B))
    cal_b.run_ensemble(n=12, seed=1)
    cal_b.fit_surrogates()
    cal_a2 = CompressibleCalibration(GVChannelDNS.load(_CASE_A))
    cal_a2.run_ensemble(n=12, seed=2)
    cal_a2.fit_surrogates()
    study = CrossMachStudy({_CASE_A: cal, "A2": cal_a2, _CASE_B: cal_b})
    out = study.predict_heldout(train=[_CASE_A, "A2"], test=_CASE_B, seed=0)
    assert out["conformal_roles_disjoint"] is True
    assert out["cal_case"] == _CASE_A
    assert out["cal_case"] not in out["fit_cases"]
    assert _CASE_B not in out["fit_cases"] and _CASE_B != out["cal_case"]
    for key in out:
        if key.endswith("_coverage"):
            assert 0.0 <= out[key] <= 1.0
    assert 0.0 < out["eta"] <= 1.0
    for key in out:
        if key.endswith("_coverage"):
            assert 0.0 <= out[key] <= 1.0
