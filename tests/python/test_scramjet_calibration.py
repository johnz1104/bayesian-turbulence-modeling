"""Tests for scramjet_calibration_demo.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the demo importable
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "python" / "examples"
sys.path.insert(0, str(_EXAMPLES_DIR))

from scramjet_calibration_demo import (
    ScramjetAnalyticForwardModel,
    _cp, _cf, _recovery_check,
)
from observation_schema import (
    ObservableType,
    scramjet_synthetic_observation_set,
    koh_from_observation_set,
)
from forward_model_interface import EvaluationResult


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

class TestPhysicsHelpers:
    def test_cp_increases_with_a1(self):
        x = 0.5  # post-shock region
        assert _cp(x, 0.40, 0.09) > _cp(x, 0.20, 0.09)

    def test_cf_increases_with_beta_star(self):
        x = 0.2  # pre-separation region
        assert _cf(x, 0.31, 0.12) > _cf(x, 0.31, 0.06)

    def test_cf_independent_of_a1(self):
        x = 0.2
        assert _cf(x, 0.20, 0.09) == pytest.approx(_cf(x, 0.50, 0.09))

    def test_cp_independent_of_beta_star(self):
        x = 0.5
        assert _cp(x, 0.31, 0.05) == pytest.approx(_cp(x, 0.31, 0.15))


# ---------------------------------------------------------------------------
# ScramjetAnalyticForwardModel
# ---------------------------------------------------------------------------

class TestScramjetAnalyticForwardModel:
    @pytest.fixture
    def obs_cf_cp(self):
        obs_full, _, _ = scramjet_synthetic_observation_set(
            n_wall_stations=5, rng_seed=0
        )
        return obs_full.filter_by_types([
            ObservableType.WALL_PRESSURE_CP,
            ObservableType.SKIN_FRICTION_CF,
        ])

    def test_evaluate_correct_length(self, obs_cf_cp):
        fm = ScramjetAnalyticForwardModel(obs_cf_cp)
        r = fm.evaluate([0.31, 0.09])
        assert len(r.predictions) == obs_cf_cp.n_obs

    def test_evaluate_converged(self, obs_cf_cp):
        fm = ScramjetAnalyticForwardModel(obs_cf_cp)
        r = fm.evaluate([0.31, 0.09])
        assert r.converged

    def test_evaluate_finite(self, obs_cf_cp):
        fm = ScramjetAnalyticForwardModel(obs_cf_cp)
        r = fm.evaluate([0.31, 0.09])
        assert all(np.isfinite(p) for p in r.predictions)

    def test_status_analytic(self, obs_cf_cp):
        fm = ScramjetAnalyticForwardModel(obs_cf_cp)
        r = fm.evaluate([0.31, 0.09])
        assert r.status == "analytic"

    def test_parameter_names(self, obs_cf_cp):
        fm = ScramjetAnalyticForwardModel(obs_cf_cp)
        assert fm.parameter_names() == ["a1", "betaStar"]

    def test_a1_affects_cp_predictions(self, obs_cf_cp):
        fm = ScramjetAnalyticForwardModel(obs_cf_cp)
        r1 = fm.evaluate([0.20, 0.09])
        r2 = fm.evaluate([0.50, 0.09])
        # At least some Cp predictions must differ
        assert not np.allclose(r1.predictions, r2.predictions)

    def test_beta_star_affects_cf_predictions(self, obs_cf_cp):
        fm = ScramjetAnalyticForwardModel(obs_cf_cp)
        r1 = fm.evaluate([0.31, 0.06])
        r2 = fm.evaluate([0.31, 0.14])
        assert not np.allclose(r1.predictions, r2.predictions)

    def test_precomputed_ensemble_none(self, obs_cf_cp):
        fm = ScramjetAnalyticForwardModel(obs_cf_cp)
        assert fm.precomputed_ensemble() is None

    def test_koh_n_matches_forward_model_length(self, obs_cf_cp):
        fm = ScramjetAnalyticForwardModel(obs_cf_cp)
        koh = koh_from_observation_set(obs_cf_cp, mode="diagonal")
        r = fm.evaluate([0.31, 0.09])
        assert len(r.predictions) == koh.n


# ---------------------------------------------------------------------------
# Recovery check helper
# ---------------------------------------------------------------------------

class TestRecoveryCheck:
    def test_passes_within_2sigma(self):
        truth = {"a1": 0.31, "betaStar": 0.09}
        # Fake samples centred on truth with small std
        rng = np.random.default_rng(0)
        samples = rng.normal(
            [0.31, 0.09, -3.0],
            [0.01, 0.005, 0.2],
            size=(1000, 3),
        )
        ok = _recovery_check(samples, truth)
        assert ok

    def test_fails_beyond_2sigma(self):
        truth = {"a1": 0.31, "betaStar": 0.09}
        # Samples far from truth
        rng = np.random.default_rng(0)
        samples = rng.normal(
            [0.50, 0.09, -3.0],
            [0.01, 0.005, 0.2],
            size=(1000, 3),
        )
        ok = _recovery_check(samples, truth)
        assert not ok


# ---------------------------------------------------------------------------
# End-to-end quick calibration
# ---------------------------------------------------------------------------

class TestEndToEndCalibration:
    def test_quick_run_recovers_truth(self):
        """Full pipeline: obs → forward model → KOH → MCMC → recovery."""
        from bayesian_inference import BayesianInferenceKOH, Prior

        obs_full, truth, _ = scramjet_synthetic_observation_set(
            n_wall_stations=5, rng_seed=7
        )
        obs_cal = obs_full.filter_by_types([
            ObservableType.WALL_PRESSURE_CP,
            ObservableType.SKIN_FRICTION_CF,
        ])
        fm  = ScramjetAnalyticForwardModel(obs_cal)
        koh = koh_from_observation_set(obs_cal, mode="diagonal")

        prior = Prior(
            means=[0.31, 0.09],
            stds=[0.05, 0.015],
            lower=[0.20, 0.05],
            upper=[0.50, 0.15],
        )
        param_set = {
            "defaults": [0.31, 0.09],
            "lower": [0.20, 0.05],
            "upper": [0.50, 0.15],
            "names": ["a1", "betaStar"],
        }
        koh_bi = BayesianInferenceKOH(
            forward_model=fm,
            param_set=param_set,
            koh_likelihood=koh,
            theta_prior=prior,
        )
        koh_bi.run_ensemble(n_samples=30, verbose=False)
        koh_bi.train_surrogate(verbose=False)
        koh_bi.run_mcmc(n_steps=200, burn_in=50, verbose=False, rng_seed=7)

        assert koh_bi.samples is not None
        assert koh_bi.samples.shape[1] == 3  # a1, betaStar, log_sigma_delta

        ok = _recovery_check(koh_bi.samples, truth)
        assert ok, (
            f"Recovery failed: truth={truth}, "
            f"a1 post={koh_bi.samples[:,0].mean():.4f}±{koh_bi.samples[:,0].std():.4f}, "
            f"betaStar post={koh_bi.samples[:,1].mean():.4f}±{koh_bi.samples[:,1].std():.4f}"
        )

    def test_posterior_uncertainty_reduced(self):
        """Posterior std should be smaller than prior std (data is informative)."""
        from bayesian_inference import BayesianInferenceKOH, Prior

        obs_full, truth, _ = scramjet_synthetic_observation_set(
            n_wall_stations=6, rng_seed=42
        )
        obs_cal = obs_full.filter_by_types([
            ObservableType.WALL_PRESSURE_CP,
            ObservableType.SKIN_FRICTION_CF,
        ])
        fm  = ScramjetAnalyticForwardModel(obs_cal)
        koh = koh_from_observation_set(obs_cal, mode="diagonal")
        prior = Prior(
            means=[0.31, 0.09],
            stds=[0.05, 0.015],
            lower=[0.20, 0.05],
            upper=[0.50, 0.15],
        )
        param_set = {
            "defaults": [0.31, 0.09],
            "lower": [0.20, 0.05],
            "upper": [0.50, 0.15],
            "names": ["a1", "betaStar"],
        }
        koh_bi = BayesianInferenceKOH(
            forward_model=fm, param_set=param_set,
            koh_likelihood=koh, theta_prior=prior,
        )
        koh_bi.run_ensemble(n_samples=30, verbose=False)
        koh_bi.train_surrogate(verbose=False)
        koh_bi.run_mcmc(n_steps=200, burn_in=50, verbose=False, rng_seed=42)

        post_std_a1   = koh_bi.samples[:, 0].std()
        post_std_beta = koh_bi.samples[:, 1].std()
        assert post_std_a1   < prior.stds[0],   f"a1 not reduced: {post_std_a1:.4f} >= {prior.stds[0]}"
        assert post_std_beta < prior.stds[1], f"betaStar not reduced: {post_std_beta:.4f} >= {prior.stds[1]}"
