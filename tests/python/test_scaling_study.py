"""
Regression contracts for the narrowed-prior review + surrogate scaling harness.

These lock the *contracts* of the substrate without running real CFD:
a fake forward model supplies a smooth analytic log-likelihood so the GP path,
the LHS design, checkpointing/resumability, and the breakdown verdict can all be
exercised deterministically and cheaply.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_inference import make_prior_from_param_set, make_sampling_prior
import scaling_study as ss


# --------------------------------------------------------------------------- #
# Fake forward model: smooth concave log-lik so the GP fits it well.
# --------------------------------------------------------------------------- #
class _FakeResult:
    def __init__(self, log_lik):
        self.log_lik = log_lik
        self.predictions = [0.0]
        self.status = "Converged"


class _FakeForwardModel:
    """evaluate(theta) -> result with a smooth quadratic log-lik in theta."""

    def __init__(self, param_set):
        names = param_set.active_names()
        # Menter-ish centres via the prior means; unit scales.
        self.mu = make_prior_from_param_set(param_set).means
        self.s = np.maximum(0.15 * np.abs(self.mu), 1e-3)

    def evaluate(self, theta_list):
        z = (np.asarray(theta_list) - self.mu) / self.s
        return _FakeResult(-0.5 * float(np.sum(z * z)))


def _fake_case_builder(param_set):
    return None, _FakeForwardModel(param_set), 0.0


# --------------------------------------------------------------------------- #
# Prior review (the 9 unexercised coefficients)
# --------------------------------------------------------------------------- #
def test_make_sampling_prior_narrows_within_bounds(rs):
    ps = rs.InferenceParameterSet.all11()
    base = make_prior_from_param_set(ps)
    narrow = make_sampling_prior(ps, relative_std=0.15, k_sigma=3.0)

    # means/stds are unchanged; only the support (LHS box) narrows.
    assert np.allclose(narrow.means, base.means)
    assert np.allclose(narrow.stds, base.stds)

    # narrowed support is a subset of the full physical box.
    assert np.all(narrow.lower >= base.lower - 1e-12)
    assert np.all(narrow.upper <= base.upper + 1e-12)

    # coefficients whose ±3σ fits inside the physical box are strictly narrowed
    # (betaStar=idx8, a1=idx9 in the all11 ordering).
    names = ps.active_names()
    i_bstar = names.index("betaStar")
    i_a1 = names.index("a1")
    for i in (i_bstar, i_a1):
        assert narrow.lower[i] > base.lower[i] + 1e-6
        assert narrow.upper[i] < base.upper[i] - 1e-6

    # the narrowed box matches mean ± kσ clipped to physical bounds.
    exp_lo = np.maximum(base.means - 3.0 * base.stds, base.lower)
    exp_hi = np.minimum(base.means + 3.0 * base.stds, base.upper)
    assert np.allclose(narrow.lower, exp_lo)
    assert np.allclose(narrow.upper, exp_hi)


def test_sampling_prior_draws_in_physical_bounds(rs):
    ps = rs.InferenceParameterSet.all11()
    phys_lo = np.array(ps.lower_bounds())
    phys_hi = np.array(ps.upper_bounds())
    prior = make_sampling_prior(ps)
    draws = prior.sample(64)
    assert draws.shape == (64, 11)
    assert np.all(draws >= phys_lo - 1e-12)
    assert np.all(draws <= phys_hi + 1e-12)


# --------------------------------------------------------------------------- #
# Nested sweep
# --------------------------------------------------------------------------- #
def test_nested_index_sets_are_nested():
    d2 = set(ss.NESTED_INDICES[2])
    d4 = set(ss.NESTED_INDICES[4])
    d8 = set(ss.NESTED_INDICES[8])
    d11 = set(ss.NESTED_INDICES[11])
    assert d2 < d4 < d8 < d11
    assert d11 == set(range(11))
    # sizes match the keys
    for k in (2, 4, 8, 11):
        assert len(ss.NESTED_INDICES[k]) == k


# --------------------------------------------------------------------------- #
# Harness run / schema / resumability / determinism
# --------------------------------------------------------------------------- #
def _make_study(rs, out_dir, d_thetas=(2, 4), n_total=12, n_test=4, seed=0):
    return ss.SurrogateScalingStudy(
        _fake_case_builder, ss._param_set_factory(),
        d_thetas=d_thetas, n_total=n_total, n_test=n_test,
        rng_seed=seed, out_dir=out_dir,
    )


def test_study_runs_and_emits_schema(rs, tmp_path):
    study = _make_study(rs, tmp_path / "run")
    verdict = study.run()

    # verdict schema
    for key in ("breakdown_d_theta", "rows", "interpretation", "standing_rule"):
        assert key in verdict
    assert "feedback-surrogate-escalation" in verdict["standing_rule"]
    assert len(verdict["rows"]) == 2

    # per-d_theta curve schema
    for d in (2, 4):
        r = study.results[d]
        assert r["curve"], "curve must be non-empty"
        for pt in r["curve"]:
            for k in ("n_train", "rmse", "r2", "coverage_2sigma"):
                assert k in pt

    # artifacts
    assert (tmp_path / "run" / "scaling_summary.json").exists()


def test_ensemble_cache_is_resumable(rs, tmp_path):
    study = _make_study(rs, tmp_path / "cache")
    X1, y1, names1, _ = study._ensemble(2)
    cache = tmp_path / "cache" / "d2_ensemble.npz"
    assert cache.exists()

    # second call loads the cache; sabotage the builder so a re-solve would fail.
    def _exploding_builder(param_set):
        raise AssertionError("must not re-solve when cache exists")

    study.case_builder = _exploding_builder
    X2, y2, names2, _ = study._ensemble(2)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert names1 == names2


def test_lhs_design_is_deterministic(rs, tmp_path):
    s1 = _make_study(rs, tmp_path / "a", seed=7)
    s2 = _make_study(rs, tmp_path / "b", seed=7)
    X1, _, _, _ = s1._ensemble(4)
    X2, _, _, _ = s2._ensemble(4)
    assert np.array_equal(X1, X2)


def _curve(pts):
    return [{"n_train": n, "r2": r2, "rmse": rmse, "coverage_2sigma": cov}
            for (n, r2, rmse, cov) in pts]


def test_breakdown_verdict_flags_correct_d_theta(rs, tmp_path):
    study = _make_study(rs, tmp_path / "v", d_thetas=(2, 4, 8))
    # Crafted curves: d2/d4 reach R²≥0.9 within budget; d8 never does and is
    # below the R² floor at the reference budget (40) -> breakdown at d8.
    study.results = {
        2: {"curve": _curve([(20, 0.95, 1.0, 0.95), (40, 0.98, 0.8, 0.95),
                             (90, 0.99, 0.5, 0.95)])},
        4: {"curve": _curve([(20, 0.60, 5.0, 0.93), (40, 0.92, 2.0, 0.93),
                             (90, 0.97, 1.0, 0.93)])},
        8: {"curve": _curve([(20, 0.10, 12.0, 0.85), (40, 0.30, 9.0, 0.85),
                             (90, 0.45, 7.0, 0.85)])},
    }
    verdict = study.summarize()
    assert verdict["breakdown_d_theta"] == 8
    flags = {r["d_theta"]: r["degraded"] for r in verdict["rows"]}
    assert flags == {2: False, 4: False, 8: True}
    # data-efficiency grows with d_θ (d2 reaches trust at 20, d4 at 40, d8 never)
    assert verdict["data_efficiency"][2] == 20
    assert verdict["data_efficiency"][4] == 40
    assert verdict["data_efficiency"][8] is None
