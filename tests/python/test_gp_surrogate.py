"""
GPSurrogate normalization regression tests.

The GP surrogate normalises training targets to zero mean / unit std internally
so the kernel hyperparameter optimisation is well conditioned regardless of
the absolute scale of the log-likelihood (which can span hundreds in
calibration runs).  These tests lock in:

  1. predict() returns mean/variance in the original (un-normalised) scale.
  2. predict_batch() agrees with predict() on the same point.
  3. The surrogate accurately recovers training points it was trained on
     (basic GP self-consistency check).
  4. ARD lengthscales are exposed and have the correct dimension.

Tests use small training sets and `optimize_restarts=1` so each one runs in
~1 s.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_inference import GPSurrogate, latin_hypercube


# Toy ground-truth function used for surrogate training:
#   f(theta) = 50 + 200*sin(2*theta_0) + 100*theta_1
# Output range ~ [O(-300), O(+300)] mimics the log-likelihood scale a real
# calibration run sees.
def _toy_f(X: np.ndarray) -> np.ndarray:
    return 50.0 + 200.0 * np.sin(2.0 * X[:, 0]) + 100.0 * X[:, 1]


@pytest.fixture(scope="module")
def trained_surrogate():
    rng = np.random.default_rng(42)
    np.random.seed(42)  # latin_hypercube uses np.random
    X = latin_hypercube(40, 2, lower=[-1.0, -1.0], upper=[1.0, 1.0])
    y = _toy_f(X) + 0.1 * rng.standard_normal(len(X))
    gp = GPSurrogate()
    gp.train(X, y, optimize_restarts=1)
    return gp, X, y


class TestGPSurrogateNormalization:
    def test_internal_mean_and_std_recorded(self, trained_surrogate):
        gp, _, y = trained_surrogate
        assert gp.trained
        assert np.isclose(gp._y_mean, np.mean(y), rtol=1e-12, atol=1e-12)
        assert np.isclose(gp._y_std,  np.std(y),  rtol=1e-12, atol=1e-12)

    def test_predict_returns_original_scale(self, trained_surrogate):
        """Predicting at training points should approximately match the
        original-scale targets, not the normalised ones."""
        gp, X, y = trained_surrogate
        for i in range(0, len(X), 7):
            mu, var = gp.predict(X[i].tolist())
            assert np.isfinite(mu) and np.isfinite(var)
            assert var >= 0.0
            assert abs(mu - y[i]) < 1.0, (
                f"surrogate prediction {mu} too far from training target {y[i]} "
                f"at index {i} (likely a normalisation bug)"
            )

    def test_predict_and_predict_batch_agree(self, trained_surrogate):
        # GPy's batch and single-point paths take slightly different routes
        # internally, so insist on agreement to a sensible numerical tolerance
        # rather than near-bit-exactness.
        gp, X, _ = trained_surrogate
        thetas = X[:5]
        mus_batch, vars_batch = gp.predict_batch(thetas)
        for i in range(len(thetas)):
            mu_i, var_i = gp.predict(thetas[i].tolist())
            assert np.isclose(mu_i, mus_batch[i], rtol=1e-8, atol=1e-8)
            assert np.isclose(var_i, vars_batch[i], rtol=1e-8, atol=1e-8)

    def test_log_likelihood_alias_matches_predict(self, trained_surrogate):
        gp, X, _ = trained_surrogate
        for i in [0, 5, 10]:
            mu, _ = gp.predict(X[i].tolist())
            assert np.isclose(gp.log_likelihood(X[i].tolist()), mu)

    def test_ard_lengthscales_have_correct_shape(self, trained_surrogate):
        gp, X, _ = trained_surrogate
        ls = gp.lengthscales()
        assert ls is not None
        assert len(ls) == X.shape[1]
        assert np.all(ls > 0)

    def test_rmse_close_to_noise_level(self, trained_surrogate):
        gp, X, y = trained_surrogate
        rmse = gp.rmse(X, y)
        assert rmse < 5.0, (
            f"GP self-RMSE {rmse:.3f} is too high; "
            "normalisation may have broken training"
        )

    def test_constant_target_is_handled(self):
        """If all y values are identical, std=0 path must not blow up."""
        np.random.seed(0)
        X = np.random.rand(15, 2)
        y = np.full(len(X), 7.42)
        gp = GPSurrogate()
        gp.train(X, y, optimize_restarts=1)
        mu, _ = gp.predict(X[0].tolist())
        assert np.isfinite(mu)
        assert abs(mu - 7.42) < 1e-3

    def test_large_offset_does_not_distort(self):
        """A huge bias added to all targets must not break round-trip."""
        np.random.seed(1)
        X = latin_hypercube(30, 2, lower=[0, 0], upper=[1, 1])
        y = _toy_f(X) + 5_000.0  # huge mean offset
        gp = GPSurrogate()
        gp.train(X, y, optimize_restarts=1)
        for i in range(0, len(X), 5):
            mu, _ = gp.predict(X[i].tolist())
            assert abs(mu - y[i]) < 5.0, (
                f"large offset distorted prediction at {i}: mu={mu} y={y[i]}"
            )
