"""Tests for active learning scramjet integration."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PYTHON_DIR   = Path(__file__).resolve().parent.parent.parent / "python"
_EXAMPLES_DIR = _PYTHON_DIR / "examples"
sys.path.insert(0, str(_PYTHON_DIR))
sys.path.insert(0, str(_EXAMPLES_DIR))

from forward_model_interface import (
    forward_model_to_callable, ForwardModelBase, EvaluationResult,
)
from active_learning import ActiveLearner, sample_lhs
from observation_schema import ObservableType, scramjet_synthetic_observation_set
from scramjet_calibration_demo import ScramjetAnalyticForwardModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _QuadFM(ForwardModelBase):
    """Simple FM with known outputs for testing. f(theta) = [a1^2, betaStar^2]."""
    def evaluate(self, theta):
        a1, bs = float(theta[0]), float(theta[1])
        return EvaluationResult([a1**2, bs**2], converged=True)
    def parameter_names(self):
        return ["a1", "betaStar"]


_LOWER = np.array([0.20, 0.05])
_UPPER = np.array([0.50, 0.15])


def _make_scramjet_fm(n_stations=4):
    obs_full, _, _ = scramjet_synthetic_observation_set(
        n_wall_stations=n_stations, rng_seed=0
    )
    obs_cal = obs_full.filter_by_types([
        ObservableType.WALL_PRESSURE_CP,
        ObservableType.SKIN_FRICTION_CF,
    ])
    return ScramjetAnalyticForwardModel(obs_cal)


# ---------------------------------------------------------------------------
# forward_model_to_callable
# ---------------------------------------------------------------------------

class TestForwardModelToCallable:
    def test_returns_callable(self):
        fm   = _QuadFM()
        func = forward_model_to_callable(fm)
        assert callable(func)

    def test_callable_returns_tuple(self):
        fm   = _QuadFM()
        func = forward_model_to_callable(fm)
        result = func([0.31, 0.09])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_ok_flag_true_for_valid(self):
        fm   = _QuadFM()
        func = forward_model_to_callable(fm)
        preds, ok = func([0.31, 0.09])
        assert ok is True

    def test_predictions_correct(self):
        fm   = _QuadFM()
        func = forward_model_to_callable(fm)
        preds, ok = func([0.30, 0.10])
        assert ok
        assert preds == pytest.approx([0.30**2, 0.10**2])

    def test_failed_fm_returns_ok_false(self):
        class _FailFM(ForwardModelBase):
            def evaluate(self, theta):
                return EvaluationResult([], converged=False, status="failed")
            def parameter_names(self):
                return ["a1"]

        func = forward_model_to_callable(_FailFM())
        preds, ok = func([0.31])
        assert not ok
        assert preds is None

    def test_preds_is_ndarray(self):
        fm   = _QuadFM()
        func = forward_model_to_callable(fm)
        preds, ok = func([0.31, 0.09])
        assert isinstance(preds, np.ndarray)

    def test_preds_all_finite(self):
        fm   = _QuadFM()
        func = forward_model_to_callable(fm)
        preds, ok = func([0.31, 0.09])
        assert ok and np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# ActiveLearner with scramjet FM
# ---------------------------------------------------------------------------

class TestActiveLearnerScramjet:
    @pytest.fixture
    def fm(self):
        return _make_scramjet_fm(n_stations=4)

    @pytest.fixture
    def learner(self, fm):
        forward = forward_model_to_callable(fm)
        return ActiveLearner(
            forward=forward,
            lower=_LOWER, upper=_UPPER,
            strategy="random",
            n_candidates=50,
            rng_seed=0,
            verbose=False,
        )

    def test_initialize_builds_training_set(self, learner):
        learner.initialize(n_init=8)
        assert learner.X.shape[0] == 8
        assert learner.Y.shape[1] == 8  # 4 stations × 2 obs types

    def test_run_extends_training_set(self, learner):
        learner.initialize(n_init=6)
        learner.run(n_queries=4)
        assert learner.X.shape[0] == 10

    def test_surrogate_trained_after_init(self, learner):
        learner.initialize(n_init=6)
        assert learner.surrog is not None
        assert learner.surrog.trained

    def test_history_queries_recorded(self, learner):
        learner.initialize(n_init=6)
        learner.run(n_queries=4)
        assert len(learner.history.queries) == 4

    def test_val_set_rmse_recorded(self, fm):
        forward = forward_model_to_callable(fm)
        # Build small val set
        X_val = np.array([[0.25, 0.07], [0.35, 0.10], [0.45, 0.12]])
        Y_val = np.array([forward(x)[0] for x in X_val])
        learner = ActiveLearner(
            forward=forward, lower=_LOWER, upper=_UPPER,
            strategy="random", val_set=(X_val, Y_val),
            rng_seed=0, verbose=False,
        )
        learner.initialize(n_init=8)
        learner.run(n_queries=3)
        assert len(learner.history.rmse_history) > 0
        assert all(np.isfinite(r) for r in learner.history.rmse_history)

    @pytest.mark.parametrize("strategy", ["random", "max_var",
                                           "max_norm_var", "max_min_dist"])
    def test_all_strategies_complete(self, fm, strategy):
        forward = forward_model_to_callable(fm)
        learner = ActiveLearner(
            forward=forward, lower=_LOWER, upper=_UPPER,
            strategy=strategy, n_candidates=50,
            rng_seed=0, verbose=False,
        )
        learner.initialize(n_init=6)
        learner.run(n_queries=4)
        assert learner.X.shape[0] == 10


# ---------------------------------------------------------------------------
# surrogate quality: final RMSE should be small after sufficient queries
# ---------------------------------------------------------------------------

class TestSurrogateQuality:
    def test_final_rmse_small(self):
        fm = _make_scramjet_fm(n_stations=4)
        forward = forward_model_to_callable(fm)
        lower, upper = _LOWER, _UPPER

        # Validation set
        rng = np.random.default_rng(99)
        X_val = rng.uniform(lower, upper, size=(20, 2))
        Y_val = np.array([forward(x)[0] for x in X_val])

        learner = ActiveLearner(
            forward=forward, lower=lower, upper=upper,
            strategy="max_var", n_candidates=100,
            val_set=(X_val, Y_val), rng_seed=0, verbose=False,
        )
        learner.initialize(n_init=12)
        learner.run(n_queries=8)

        final_rmse = learner.history.rmse_history[-1]
        # Analytic model is smooth → surrogate should be excellent
        assert np.isfinite(final_rmse)
        # RMSE should be small relative to output scale
        y_scale = np.std(Y_val)
        assert final_rmse < 0.1 * y_scale, (
            f"RMSE {final_rmse:.4f} > 10% of y_scale {y_scale:.4f}"
        )

    def test_variance_strategy_builds_surrogate(self):
        """SurrogateForwardModel built from active-learned ensemble is usable."""
        from forward_model_interface import SurrogateForwardModel
        from bayesian_inference import MultiOutputSurrogate

        fm = _make_scramjet_fm(n_stations=3)
        forward = forward_model_to_callable(fm)
        learner = ActiveLearner(
            forward=forward, lower=_LOWER, upper=_UPPER,
            strategy="max_var", n_candidates=50,
            rng_seed=0, verbose=False,
        )
        learner.initialize(n_init=10)
        learner.run(n_queries=5)

        # Use the trained surrogate as a SurrogateForwardModel
        surr_fm = SurrogateForwardModel(
            learner.surrog, param_names=fm.parameter_names()
        )
        r = surr_fm.evaluate([0.31, 0.09])
        assert r.converged
        assert len(r.predictions) == 6  # 3 stations × 2 types
        assert all(np.isfinite(p) for p in r.predictions)
