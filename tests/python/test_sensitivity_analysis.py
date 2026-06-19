"""Tests for sensitivity_analysis.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PYTHON_DIR   = Path(__file__).resolve().parent.parent.parent / "python"
_EXAMPLES_DIR = _PYTHON_DIR / "examples"
sys.path.insert(0, str(_PYTHON_DIR))
sys.path.insert(0, str(_EXAMPLES_DIR))

from sensitivity_analysis import (
    OATResult, MorrisResult,
    run_oat, run_morris,
    SensitivityAnalyser,
)
from forward_model_interface import ForwardModelBase, EvaluationResult
from observation_schema import ObservableType, scramjet_synthetic_observation_set
from scramjet_calibration_demo import ScramjetAnalyticForwardModel


# ---------------------------------------------------------------------------
# Simple linear test model
#   output_0 = a1 * 10       (sensitive to a1 only)
#   output_1 = betaStar * 5  (sensitive to betaStar only)
# ---------------------------------------------------------------------------

class _LinearFM(ForwardModelBase):
    def evaluate(self, theta):
        a1, bs = float(theta[0]), float(theta[1])
        return EvaluationResult([a1 * 10.0, bs * 5.0], converged=True)

    def parameter_names(self):
        return ["a1", "betaStar"]


_LOWER = np.array([0.20, 0.05])
_UPPER = np.array([0.50, 0.15])
_NOM   = np.array([0.31, 0.09])


# ---------------------------------------------------------------------------
# run_oat
# ---------------------------------------------------------------------------

class TestRunOAT:
    def test_returns_list(self):
        fm  = _LinearFM()
        res = run_oat(fm, _NOM, (_LOWER, _UPPER), ["a1", "betaStar"],
                      n_points=5)
        assert isinstance(res, list)

    def test_correct_n_results(self):
        fm  = _LinearFM()
        res = run_oat(fm, _NOM, (_LOWER, _UPPER), ["a1", "betaStar"],
                      n_points=5)
        # 2 params × 2 outputs = 4
        assert len(res) == 4

    def test_oat_result_type(self):
        fm  = _LinearFM()
        res = run_oat(fm, _NOM, (_LOWER, _UPPER), ["a1", "betaStar"],
                      n_points=5)
        for r in res:
            assert isinstance(r, OATResult)

    def test_a1_sensitivity_output0(self):
        """Output 0 = a1*10 → slope ≈ 10."""
        fm  = _LinearFM()
        res = run_oat(fm, _NOM, (_LOWER, _UPPER), ["a1", "betaStar"],
                      n_points=11)
        r = next(r for r in res if r.param_name == "a1" and r.output_index == 0)
        assert r.sensitivity == pytest.approx(10.0, abs=0.5)

    def test_betastar_zero_sensitivity_output0(self):
        """Output 0 = a1*10 is independent of betaStar → slope ≈ 0."""
        fm  = _LinearFM()
        res = run_oat(fm, _NOM, (_LOWER, _UPPER), ["a1", "betaStar"],
                      n_points=11)
        r = next(r for r in res if r.param_name == "betaStar" and r.output_index == 0)
        assert abs(r.sensitivity) < 0.1

    def test_theta_range_length(self):
        fm  = _LinearFM()
        res = run_oat(fm, _NOM, (_LOWER, _UPPER), ["a1", "betaStar"],
                      n_points=7)
        for r in res:
            assert len(r.theta_range) == 7

    def test_output_range_length(self):
        fm  = _LinearFM()
        res = run_oat(fm, _NOM, (_LOWER, _UPPER), ["a1", "betaStar"],
                      n_points=7)
        for r in res:
            assert len(r.output_range) == 7


# ---------------------------------------------------------------------------
# run_morris
# ---------------------------------------------------------------------------

class TestRunMorris:
    def test_returns_list(self):
        fm  = _LinearFM()
        res = run_morris(fm, (_LOWER, _UPPER), ["a1", "betaStar"],
                         n_trajectories=5, rng_seed=0)
        assert isinstance(res, list)
        assert len(res) == 2

    def test_result_type(self):
        fm  = _LinearFM()
        res = run_morris(fm, (_LOWER, _UPPER), ["a1", "betaStar"],
                         n_trajectories=5, rng_seed=0)
        for r in res:
            assert isinstance(r, MorrisResult)

    def test_sorted_descending_mu_star(self):
        fm  = _LinearFM()
        res = run_morris(fm, (_LOWER, _UPPER), ["a1", "betaStar"],
                         n_trajectories=5, rng_seed=0)
        mu_stars = [r.mu_star for r in res]
        assert mu_stars == sorted(mu_stars, reverse=True)

    def test_a1_higher_mu_star(self):
        """a1 affects output_0 with slope 10; betaStar affects output_1 with slope 5."""
        fm  = _LinearFM()
        res = run_morris(fm, (_LOWER, _UPPER), ["a1", "betaStar"],
                         n_trajectories=10, rng_seed=1)
        a1_r   = next(r for r in res if r.param_name == "a1")
        beta_r = next(r for r in res if r.param_name == "betaStar")
        assert a1_r.mu_star > beta_r.mu_star

    def test_mu_star_per_output_shape(self):
        fm  = _LinearFM()
        res = run_morris(fm, (_LOWER, _UPPER), ["a1", "betaStar"],
                         n_trajectories=5, rng_seed=0)
        for r in res:
            assert r.mu_star_per_output.shape == (2,)

    def test_reproducible_with_seed(self):
        fm  = _LinearFM()
        res1 = run_morris(fm, (_LOWER, _UPPER), ["a1", "betaStar"],
                          n_trajectories=8, rng_seed=42)
        res2 = run_morris(fm, (_LOWER, _UPPER), ["a1", "betaStar"],
                          n_trajectories=8, rng_seed=42)
        for r1, r2 in zip(res1, res2):
            assert r1.mu_star == pytest.approx(r2.mu_star)


# ---------------------------------------------------------------------------
# SensitivityAnalyser
# ---------------------------------------------------------------------------

class TestSensitivityAnalyser:
    @pytest.fixture
    def analyser(self):
        fm = _LinearFM()
        return SensitivityAnalyser(fm, (_LOWER, _UPPER),
                                   ["a1", "betaStar"], nominal=_NOM)

    def test_construction(self, analyser):
        assert analyser.param_names == ["a1", "betaStar"]

    def test_oat_not_run_initially(self, analyser):
        assert analyser.oat_results is None

    def test_morris_not_run_initially(self, analyser):
        assert analyser.morris_results is None

    def test_run_oat_sets_results(self, analyser):
        analyser.run_oat(n_points=5)
        assert analyser.oat_results is not None

    def test_run_morris_sets_results(self, analyser):
        analyser.run_morris(n_trajectories=5, rng_seed=0)
        assert analyser.morris_results is not None

    def test_run_returns_self(self, analyser):
        r1 = analyser.run_oat(n_points=5)
        r2 = analyser.run_morris(n_trajectories=5, rng_seed=0)
        assert r1 is analyser
        assert r2 is analyser

    def test_oat_importance_normalised(self, analyser):
        analyser.run_oat(n_points=7)
        imp = analyser.oat_importance()
        assert max(imp.values()) == pytest.approx(1.0)

    def test_morris_importance_normalised(self, analyser):
        analyser.run_morris(n_trajectories=5, rng_seed=0)
        imp = analyser.morris_importance()
        assert max(imp.values()) == pytest.approx(1.0)

    def test_a1_ranked_first(self, analyser):
        analyser.run_oat(n_points=7).run_morris(n_trajectories=8, rng_seed=0)
        rep = analyser.report()
        assert rep["ranking"][0] == "a1"

    def test_report_keys(self, analyser):
        analyser.run_oat(n_points=5).run_morris(n_trajectories=5, rng_seed=0)
        rep = analyser.report()
        assert "param_names" in rep
        assert "oat_importance" in rep
        assert "morris_importance" in rep
        assert "ranking" in rep

    def test_oat_importance_error_before_run(self):
        fm = _LinearFM()
        sa = SensitivityAnalyser(fm, (_LOWER, _UPPER), ["a1", "betaStar"])
        with pytest.raises(AssertionError):
            sa.oat_importance()

    def test_morris_importance_error_before_run(self):
        fm = _LinearFM()
        sa = SensitivityAnalyser(fm, (_LOWER, _UPPER), ["a1", "betaStar"])
        with pytest.raises(AssertionError):
            sa.morris_importance()

    def test_default_nominal_is_midpoint(self):
        fm = _LinearFM()
        sa = SensitivityAnalyser(fm, (_LOWER, _UPPER), ["a1", "betaStar"])
        expected = 0.5 * (_LOWER + _UPPER)
        assert np.allclose(sa.nominal, expected)


# ---------------------------------------------------------------------------
# Integration: scramjet analytic model
# ---------------------------------------------------------------------------

class TestScramjetSensitivity:
    @pytest.fixture
    def sa_scramjet(self):
        obs_full, _, _ = scramjet_synthetic_observation_set(
            n_wall_stations=5, rng_seed=0
        )
        obs_cal = obs_full.filter_by_types([
            ObservableType.WALL_PRESSURE_CP,
            ObservableType.SKIN_FRICTION_CF,
        ])
        fm = ScramjetAnalyticForwardModel(obs_cal)
        return SensitivityAnalyser(
            fm, (np.array([0.20, 0.05]), np.array([0.50, 0.15])),
            ["a1", "betaStar"], nominal=np.array([0.31, 0.09]),
        )

    def test_oat_runs(self, sa_scramjet):
        sa_scramjet.run_oat(n_points=5)
        assert sa_scramjet.oat_results is not None

    def test_morris_runs(self, sa_scramjet):
        sa_scramjet.run_morris(n_trajectories=5, rng_seed=0)
        assert sa_scramjet.morris_results is not None

    def test_a1_dominant_oat(self, sa_scramjet):
        """a1 controls Cp amplitude → should rank first in OAT."""
        sa_scramjet.run_oat(n_points=9)
        imp = sa_scramjet.oat_importance()
        assert imp["a1"] > imp["betaStar"]

    def test_report_returns_ranking(self, sa_scramjet):
        sa_scramjet.run_oat(n_points=5).run_morris(n_trajectories=5, rng_seed=0)
        rep = sa_scramjet.report()
        assert set(rep["ranking"]) == {"a1", "betaStar"}
        assert len(rep["ranking"]) == 2
