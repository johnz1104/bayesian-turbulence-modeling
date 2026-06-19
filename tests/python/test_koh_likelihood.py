"""
KOHLikelihood unit tests.

The Kennedy-O'Hagan likelihood is the model-form-uncertainty-aware likelihood
used by BayesianInferenceKOH.  These tests lock in:

  1. Construction validates obs_sigmas and length consistency at __init__ time.
  2. __call__ returns a finite log-likelihood for valid inputs.
  3. __call__ returns -inf for invalid inputs (NaN/Inf eta, non-finite
     hyperparameters, length mismatch) — never NaN, never an exception.
  4. Larger residuals give smaller log-likelihoods (basic monotonicity).
  5. Larger sigma_delta inflates the predictive covariance, so a residual that
     is "many sigma_eps" away becomes more probable as sigma_delta grows.
  6. mode switch: ``diagonal`` and ``physical_gp`` give different
     behaviour and ``n_extra_params`` adapts.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_inference import KOH_MODES, KOHLikelihood


@pytest.fixture
def koh_basic() -> KOHLikelihood:
    """Six-point synthetic observation set with uniform sigma_eps = 0.1."""
    x = np.linspace(0.0, 5.0, 6)
    y = np.array([0.5, 1.0, 1.4, 1.6, 1.5, 1.2])
    sigma_eps = np.full_like(y, 0.1)
    return KOHLikelihood(x, y, sigma_eps)


class TestKOHConstruction:
    def test_basic_construction(self, koh_basic):
        assert koh_basic.n == 6
        assert koh_basic.x.shape == (6, 1)
        assert koh_basic.y.shape == (6,)
        assert koh_basic.sigma_eps.shape == (6,)

    def test_2d_locations_accepted(self):
        x = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        y = np.array([0.1, 0.2, 0.3])
        s = np.full_like(y, 0.05)
        koh = KOHLikelihood(x, y, s)
        assert koh.n == 3
        assert koh.x.shape == (3, 2)

    def test_negative_sigma_rejected(self):
        x = np.array([0.0, 1.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="strictly positive"):
            KOHLikelihood(x, y, np.array([0.1, -0.1]))

    def test_zero_sigma_rejected(self):
        x = np.array([0.0, 1.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="strictly positive"):
            KOHLikelihood(x, y, np.array([0.1, 0.0]))

    def test_nonfinite_sigma_rejected(self):
        x = np.array([0.0, 1.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="finite"):
            KOHLikelihood(x, y, np.array([0.1, np.nan]))

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            KOHLikelihood(
                obs_locations=np.array([0.0, 1.0, 2.0]),
                obs_values=np.array([1.0, 2.0]),
                obs_sigmas=np.array([0.1, 0.1]),
            )

    def test_nonfinite_obs_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            KOHLikelihood(
                obs_locations=np.array([0.0, 1.0]),
                obs_values=np.array([1.0, np.nan]),
                obs_sigmas=np.array([0.1, 0.1]),
            )


class TestKOHValidEvaluation:
    def test_zero_residual_finite(self, koh_basic):
        ll = koh_basic(koh_basic.y, log_sigma_delta=-2.0, log_l_delta=0.0)
        assert np.isfinite(ll)

    def test_residual_decreases_loglik(self, koh_basic):
        eta_match    = koh_basic.y
        eta_offset   = koh_basic.y + 0.5
        eta_far      = koh_basic.y + 2.0
        ll_match     = koh_basic(eta_match,  -2.0, 0.0)
        ll_offset    = koh_basic(eta_offset, -2.0, 0.0)
        ll_far       = koh_basic(eta_far,    -2.0, 0.0)
        assert ll_match > ll_offset > ll_far

    def test_sigma_delta_inflation(self, koh_basic):
        """A residual that is many sigma_eps away should be more probable once
        sigma_delta is allowed to grow toward the residual scale.  This is NOT
        a global monotonicity property — once sigma_delta dominates, the
        log|C| penalty kicks back in — so we only check the small-to-moderate
        regime where adding model-form discrepancy unambiguously helps."""
        eta_offset = koh_basic.y + 0.5  # residual ~ 5 * sigma_eps everywhere
        ll_tiny = koh_basic(eta_offset, log_sigma_delta=-5.0, log_l_delta=0.0)
        ll_med  = koh_basic(eta_offset, log_sigma_delta=-1.0, log_l_delta=0.0)
        assert np.isfinite(ll_tiny) and np.isfinite(ll_med)
        assert ll_tiny < ll_med, (
            f"adding moderate discrepancy did not improve fit: "
            f"ll(sigma_delta≈0.007)={ll_tiny:.3f} >= ll(sigma_delta≈0.37)={ll_med:.3f}"
        )

    def test_sigma_delta_too_large_penalised(self, koh_basic):
        """Sanity: an enormous sigma_delta should hurt, not help, because the
        log|C| determinant penalty dominates.  This is the complement of the
        inflation test and pins down the second half of the optimum."""
        eta_match = koh_basic.y
        ll_med   = koh_basic(eta_match, log_sigma_delta=-2.0, log_l_delta=0.0)
        ll_huge  = koh_basic(eta_match, log_sigma_delta= 5.0, log_l_delta=0.0)
        assert np.isfinite(ll_med) and np.isfinite(ll_huge)
        assert ll_huge < ll_med

    def test_finite_for_extreme_lengthscales(self, koh_basic):
        # Tiny l: kernel ~ I, well conditioned via sigma_eps diagonal.
        # Huge l: kernel ~ all ones, only well posed thanks to jitter+sigma_eps.
        for log_l in (-10.0, -5.0, 0.0, 5.0, 10.0):
            ll = koh_basic(koh_basic.y, log_sigma_delta=-2.0, log_l_delta=log_l)
            assert np.isfinite(ll), f"non-finite ll at log_l_delta={log_l}: {ll}"


class TestKOHInvalidEvaluation:
    """All of these must return -inf, not raise and not return NaN."""

    @pytest.mark.parametrize(
        "broken_eta",
        [
            np.array([np.nan, 1, 2, 3, 4, 5], dtype=float),
            np.array([np.inf, 1, 2, 3, 4, 5], dtype=float),
            np.full(6, np.nan),
            np.full(6, np.inf),
        ],
        ids=["nan-elem", "inf-elem", "all-nan", "all-inf"],
    )
    def test_invalid_eta(self, koh_basic, broken_eta):
        ll = koh_basic(broken_eta, -2.0, 0.0)
        assert ll == -np.inf

    def test_eta_shape_mismatch(self, koh_basic):
        ll = koh_basic(np.zeros(3), -2.0, 0.0)
        assert ll == -np.inf

    @pytest.mark.parametrize(
        "lsd, lld",
        [
            (np.nan, 0.0),
            (np.inf, 0.0),
            (-np.inf, 0.0),
            (0.0, np.nan),
            (0.0, np.inf),
            (0.0, -np.inf),  # exp(-inf)=0 lengthscale is invalid
        ],
        ids=[
            "nan-sigma", "posinf-sigma", "neginf-sigma",
            "nan-l", "posinf-l", "neginf-l",
        ],
    )
    def test_invalid_hyperparams(self, koh_basic, lsd, lld):
        ll = koh_basic(koh_basic.y, lsd, lld)
        assert ll == -np.inf

    def test_no_nan_returned_for_any_input(self, koh_basic):
        """Belt-and-braces: evaluate on a fuzzy grid of edge inputs and
        require either finite or -inf, never NaN."""
        rng = np.random.default_rng(123)
        ll_vals = []
        for _ in range(40):
            eta = koh_basic.y + 0.1 * rng.standard_normal(6)
            lsd = float(rng.normal(-2.0, 5.0))
            lld = float(rng.normal(0.0, 5.0))
            if rng.random() < 0.1:
                eta[0] = np.nan
            ll_vals.append(koh_basic(eta, lsd, lld))
        for v in ll_vals:
            assert (v == -np.inf) or np.isfinite(v), f"got NaN/garbage: {v}"


class TestKOHModeSwitch:
    """KOHLikelihood supports diagonal and physical_gp modes."""

    @pytest.mark.parametrize("mode", KOH_MODES)
    def test_construction_with_each_mode(self, mode):
        x = np.linspace(0.0, 5.0, 5)
        y = np.zeros_like(x)
        s = np.full_like(y, 0.1)
        koh = KOHLikelihood(x, y, s, mode=mode)
        assert koh.mode == mode

    def test_unknown_mode_rejected(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])
        s = np.array([0.1, 0.1])
        with pytest.raises(ValueError, match="mode"):
            KOHLikelihood(x, y, s, mode="not_a_mode")

    def test_n_extra_params(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])
        s = np.array([0.1, 0.1])
        assert KOHLikelihood(x, y, s, mode="diagonal").n_extra_params == 1
        assert KOHLikelihood(x, y, s, mode="physical_gp").n_extra_params == 2

    def test_diagonal_kernel_is_identity(self):
        x = np.linspace(0.0, 5.0, 4)
        y = np.zeros_like(x)
        s = np.full_like(y, 0.1)
        koh = KOHLikelihood(x, y, s, mode="diagonal")
        # _kernel ignores its arg in diagonal mode
        K1 = koh._kernel(0.1)
        K2 = koh._kernel(10.0)
        np.testing.assert_array_equal(K1, np.eye(4))
        np.testing.assert_array_equal(K2, np.eye(4))

    def test_diagonal_loglik_invariant_to_log_l(self):
        """In diagonal mode log_l_delta should not change the log-likelihood."""
        x = np.linspace(0.0, 5.0, 6)
        y = np.array([0.5, 1.0, 1.4, 1.6, 1.5, 1.2])
        s = np.full_like(y, 0.1)
        koh = KOHLikelihood(x, y, s, mode="diagonal")
        ll_a = koh(y, log_sigma_delta=-1.0, log_l_delta=-3.0)
        ll_b = koh(y, log_sigma_delta=-1.0, log_l_delta= 3.0)
        ll_c = koh(y, log_sigma_delta=-1.0, log_l_delta= 0.0)
        assert np.isclose(ll_a, ll_b, rtol=1e-12, atol=1e-12)
        assert np.isclose(ll_a, ll_c, rtol=1e-12, atol=1e-12)

    def test_physical_gp_loglik_does_depend_on_log_l(self):
        x = np.linspace(0.0, 5.0, 6)
        y = np.array([0.5, 1.0, 1.4, 1.6, 1.5, 1.2])
        s = np.full_like(y, 0.1)
        koh = KOHLikelihood(x, y, s, mode="physical_gp")
        eta = y + 0.3
        ll_short = koh(eta, log_sigma_delta=-1.0, log_l_delta=-3.0)
        ll_long  = koh(eta, log_sigma_delta=-1.0, log_l_delta= 3.0)
        # The two should differ; lengthscale matters in physical_gp mode.
        assert not np.isclose(ll_short, ll_long, rtol=1e-3)

    def test_diagonal_recovers_simple_gaussian_when_sigma_delta_is_zero(self):
        """sigma_delta -> 0 should recover the standard Gaussian likelihood."""
        x = np.linspace(0.0, 5.0, 6)
        y = np.array([0.5, 1.0, 1.4, 1.6, 1.5, 1.2])
        s = np.full_like(y, 0.1)
        koh = KOHLikelihood(x, y, s, mode="diagonal")
        eta = y + 0.05
        ll = koh(eta, log_sigma_delta=-30.0, log_l_delta=0.0)
        # Standard log-Gaussian: -0.5 [ Σ (r/σ)² + Σ log 2πσ² ]
        r = y - eta
        ll_expected = -0.5 * (
            np.sum((r / s) ** 2) + np.sum(np.log(2 * np.pi * s ** 2))
        )
        assert np.isclose(ll, ll_expected, rtol=1e-2, atol=1e-2)
