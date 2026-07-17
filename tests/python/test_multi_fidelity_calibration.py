"""Tests for multi_fidelity_calibration.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PYTHON_DIR  = Path(__file__).resolve().parent.parent.parent / "python"
_EXAMPLES_DIR = _PYTHON_DIR / "examples"
sys.path.insert(0, str(_PYTHON_DIR))
sys.path.insert(0, str(_EXAMPLES_DIR))

from multi_fidelity_calibration import FidelityLevel, MultiFidelityStudy, LevelSummary
from bayesian_inference import Prior
from forward_model_interface import ForwardModelBase, EvaluationResult
from observation_schema import (
    ObservableType, scramjet_synthetic_observation_set, ObservationSet
)
from scramjet_calibration_demo import ScramjetAnalyticForwardModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_prior():
    return Prior(
        means=[0.31, 0.09], stds=[0.05, 0.015],
        lower=[0.20, 0.05], upper=[0.50, 0.15],
    )


def _make_obs_and_fm(n_stations, obs_types, rng_seed=0):
    obs_full, truth, _ = scramjet_synthetic_observation_set(
        n_wall_stations=n_stations, rng_seed=rng_seed
    )
    obs = obs_full.filter_by_types(list(obs_types))
    fm  = ScramjetAnalyticForwardModel(obs)
    return obs, fm, truth


# ---------------------------------------------------------------------------
# FidelityLevel
# ---------------------------------------------------------------------------

class TestFidelityLevel:
    def test_n_obs(self):
        obs, fm, _ = _make_obs_and_fm(4, {ObservableType.SKIN_FRICTION_CF})
        lvl = FidelityLevel("lo", obs, fm)
        assert lvl.n_obs == obs.n_obs

    def test_ran_false_before_run(self):
        obs, fm, _ = _make_obs_and_fm(4, {ObservableType.SKIN_FRICTION_CF})
        lvl = FidelityLevel("lo", obs, fm)
        assert not lvl.ran

    def test_ran_true_after_samples_set(self):
        obs, fm, _ = _make_obs_and_fm(4, {ObservableType.SKIN_FRICTION_CF})
        lvl = FidelityLevel("lo", obs, fm)
        lvl.samples = np.ones((10, 3))
        assert lvl.ran


# ---------------------------------------------------------------------------
# MultiFidelityStudy construction
# ---------------------------------------------------------------------------

class TestMultiFidelityStudyConstruction:
    def test_add_level(self):
        study = MultiFidelityStudy(_make_simple_prior(), ["a1", "betaStar"])
        obs, fm, _ = _make_obs_and_fm(4, {ObservableType.SKIN_FRICTION_CF})
        study.add_level("lo", obs, fm)
        assert study.n_levels == 1

    def test_add_multiple_levels(self):
        study = MultiFidelityStudy(_make_simple_prior(), ["a1", "betaStar"])
        for n in [4, 6, 8]:
            obs, fm, _ = _make_obs_and_fm(n, {ObservableType.SKIN_FRICTION_CF,
                                               ObservableType.WALL_PRESSURE_CP})
            study.add_level(f"n{n}", obs, fm)
        assert study.n_levels == 3

    def test_levels_property(self):
        study = MultiFidelityStudy(_make_simple_prior(), ["a1", "betaStar"])
        obs, fm, _ = _make_obs_and_fm(4, {ObservableType.SKIN_FRICTION_CF})
        study.add_level("lo", obs, fm)
        lvls = study.levels
        assert len(lvls) == 1
        assert isinstance(lvls[0], FidelityLevel)

    def test_add_level_returns_self(self):
        study = MultiFidelityStudy(_make_simple_prior(), ["a1", "betaStar"])
        obs, fm, _ = _make_obs_and_fm(4, {ObservableType.SKIN_FRICTION_CF})
        result = study.add_level("lo", obs, fm)
        assert result is study

    def test_default_param_set_built(self):
        prior = _make_simple_prior()
        study = MultiFidelityStudy(prior, ["a1", "betaStar"])
        assert study.param_set["names"] == ["a1", "betaStar"]
        assert study.param_set["defaults"] == pytest.approx(prior.means.tolist())


# ---------------------------------------------------------------------------
# MultiFidelityStudy.run_all
# ---------------------------------------------------------------------------

class TestMultiFidelityStudyRunAll:
    @pytest.fixture(scope="class")
    def ran_two_level(self):
        """The two-level study RUN ONCE; every test inspects this one result
        (the four assertions are properties of a single completed run, not of
        four identical reruns)."""
        prior = _make_simple_prior()
        study = MultiFidelityStudy(prior, ["a1", "betaStar"])

        obs_lo, fm_lo, _ = _make_obs_and_fm(4, {ObservableType.SKIN_FRICTION_CF})
        obs_hi, fm_hi, _ = _make_obs_and_fm(6, {ObservableType.WALL_PRESSURE_CP,
                                                  ObservableType.SKIN_FRICTION_CF})
        study.add_level("lo", obs_lo, fm_lo)
        study.add_level("hi", obs_hi, fm_hi)
        result = study.run_all(n_ensemble=20, n_steps=100, rng_seed=0,
                               verbose=False)
        return study, result

    def test_run_all_sets_samples(self, ran_two_level):
        study, _ = ran_two_level
        for lvl in study.levels:
            assert lvl.ran
            assert lvl.samples is not None

    def test_run_all_samples_shape(self, ran_two_level):
        study, _ = ran_two_level
        for lvl in study.levels:
            # samples shape: (n_mcmc, n_theta + n_extra)
            assert lvl.samples.ndim == 2
            assert lvl.samples.shape[1] == 3  # a1, betaStar, log_sigma_delta

    def test_run_all_returns_self(self, ran_two_level):
        study, result = ran_two_level
        assert result is study

    def test_elapsed_recorded(self, ran_two_level):
        study, _ = ran_two_level
        for lvl in study.levels:
            assert lvl.elapsed_s > 0.0


# ---------------------------------------------------------------------------
# MultiFidelityStudy.comparison_table
# ---------------------------------------------------------------------------

class TestComparisonTable:
    @pytest.fixture(scope="class")
    def ran_study(self):
        prior = _make_simple_prior()
        study = MultiFidelityStudy(prior, ["a1", "betaStar"])
        for n, types in [(4, {ObservableType.SKIN_FRICTION_CF}),
                          (6, {ObservableType.WALL_PRESSURE_CP,
                               ObservableType.SKIN_FRICTION_CF})]:
            obs, fm, _ = _make_obs_and_fm(n, types, rng_seed=0)
            study.add_level(f"n{n}", obs, fm)
        study.run_all(n_ensemble=20, n_steps=100, rng_seed=0, verbose=False)
        return study

    def test_table_has_expected_keys(self, ran_study):
        t = ran_study.comparison_table()
        assert "param_names" in t
        assert "levels" in t
        assert "convergence" in t

    def test_table_n_levels(self, ran_study):
        t = ran_study.comparison_table()
        assert len(t["levels"]) == 2

    def test_level_has_param_stats(self, ran_study):
        t = ran_study.comparison_table()
        for lv in t["levels"]:
            for name in ["a1", "betaStar"]:
                assert name in lv
                s = lv[name]
                assert "mean" in s and "std" in s
                assert "ci_lo" in s and "ci_hi" in s and "ci_width" in s

    def test_ci_width_positive(self, ran_study):
        t = ran_study.comparison_table()
        for lv in t["levels"]:
            for name in ["a1", "betaStar"]:
                assert lv[name]["ci_width"] > 0

    def test_convergence_ratios_listed(self, ran_study):
        t = ran_study.comparison_table()
        for name in ["a1", "betaStar"]:
            assert name in t["convergence"]
            assert len(t["convergence"][name]) == 2

    def test_level_summary_error_if_not_ran(self):
        prior = _make_simple_prior()
        study = MultiFidelityStudy(prior, ["a1", "betaStar"])
        obs, fm, _ = _make_obs_and_fm(4, {ObservableType.SKIN_FRICTION_CF})
        study.add_level("lo", obs, fm)
        with pytest.raises(AssertionError):
            study.level_summary(study.levels[0])


# ---------------------------------------------------------------------------
# End-to-end: betaStar CI tightens from lo to hi fidelity
# ---------------------------------------------------------------------------

class TestFidelityEffect:
    def test_betastar_ci_tightens(self):
        """Hi-fi (Cp+Cf, more stations) gives tighter betaStar CI than lo-fi (Cf only)."""
        prior = _make_simple_prior()
        study = MultiFidelityStudy(prior, ["a1", "betaStar"])

        obs_lo, fm_lo, _ = _make_obs_and_fm(
            4, {ObservableType.SKIN_FRICTION_CF}, rng_seed=1
        )
        obs_hi, fm_hi, _ = _make_obs_and_fm(
            8, {ObservableType.WALL_PRESSURE_CP, ObservableType.SKIN_FRICTION_CF},
            rng_seed=1,
        )
        study.add_level("lo", obs_lo, fm_lo)
        study.add_level("hi", obs_hi, fm_hi)
        study.run_all(n_ensemble=25, n_steps=150, rng_seed=1, verbose=False)

        t = study.comparison_table()
        lo_ci = t["levels"][0]["betaStar"]["ci_width"]
        hi_ci = t["levels"][1]["betaStar"]["ci_width"]
        assert hi_ci < lo_ci, (
            f"Expected hi_ci ({hi_ci:.5f}) < lo_ci ({lo_ci:.5f})"
        )

    def test_a1_ci_tightens_with_cp(self):
        """Adding Cp observations tightens a1 CI (Cf alone is not informative for a1)."""
        prior = _make_simple_prior()
        study = MultiFidelityStudy(prior, ["a1", "betaStar"])

        obs_lo, fm_lo, _ = _make_obs_and_fm(
            5, {ObservableType.SKIN_FRICTION_CF}, rng_seed=2
        )
        obs_hi, fm_hi, _ = _make_obs_and_fm(
            5, {ObservableType.WALL_PRESSURE_CP, ObservableType.SKIN_FRICTION_CF},
            rng_seed=2,
        )
        study.add_level("cf_only", obs_lo, fm_lo)
        study.add_level("cf+cp", obs_hi, fm_hi)
        study.run_all(n_ensemble=25, n_steps=150, rng_seed=2, verbose=False)

        t = study.comparison_table()
        lo_ci = t["levels"][0]["a1"]["ci_width"]
        hi_ci = t["levels"][1]["a1"]["ci_width"]
        assert hi_ci < lo_ci, (
            f"Expected hi_ci ({hi_ci:.5f}) < lo_ci ({lo_ci:.5f})"
        )
