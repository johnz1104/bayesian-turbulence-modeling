"""
KOH identifiability diagnostics.

Uses a synthetic toy forward model so the test runs in a few seconds while
still exercising the BayesianInference / BayesianInferenceKOH switch and the
``koh_diagnostics`` comparison helpers.

Toy model
---------
``ToyForward`` reproduces the C++ ForwardModel API with:
    evaluate(theta) -> EvalResult(predictions, log_lik, status, simple_iters)
    penalized_log_likelihood(theta) -> log_lik

It maps θ = (a, b) to a 4-vector of "observations" ``a*x + b`` evaluated at
``x = [0, 1, 2, 3]`` and uses Gaussian likelihood against a fixed obs vector.

Tests
-----
  * ``run_calibration`` works in all three modes (no_discrepancy, diagonal,
    physical_gp).
  * Diagonal mode samples have one fewer dimension than physical_gp.
  * Posterior recovers the truth within tolerance for each KOH mode.
  * ``write_report`` produces the JSON file (matplotlib plots are best-effort).
  * Identifiability flag lists the expected KOH modes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_inference import (
    Prior, BayesianInference, BayesianInferenceKOH, KOHLikelihood,
    make_prior_from_param_set, _get_param_names,
)
from koh_diagnostics import (
    run_calibration, compare_modes, write_report, make_obs_metadata,
    posterior_widths, posterior_shifts, discrepancy_summary,
)


# ---------------- Toy fixtures ----------------------------------------

@dataclass
class _ToyResult:
    predictions: list
    log_lik: float
    status: str = "Converged"
    simple_iters: int = 1


class ToyForward:
    """Lightweight stand-in for the C++ ForwardModel.

    Maps θ=(a, b) → predictions at x=[0,1,2,3]; the log-likelihood is a
    Gaussian on those predictions vs a fixed obs vector.
    """

    def __init__(self, x_locs, obs, sigmas, theta_true):
        self.x      = np.asarray(x_locs, float)
        self.obs    = np.asarray(obs, float)
        self.sigmas = np.asarray(sigmas, float)
        self.theta_true = np.asarray(theta_true, float)

    def _predict(self, theta):
        a, b = theta
        return a * self.x + b

    def evaluate(self, theta):
        pred = self._predict(theta)
        r    = (self.obs - pred) / self.sigmas
        loglik = -0.5 * float(np.sum(r ** 2)
                              + np.sum(np.log(2 * np.pi * self.sigmas ** 2)))
        return _ToyResult(predictions=[float(v) for v in pred], log_lik=loglik)

    def penalized_log_likelihood(self, theta):
        return self.evaluate(theta).log_lik


class ToyParamSet(Mapping):
    """Supported pure-Python Mapping stand-in for InferenceParameterSet.

    Defaults are deliberately non-zero on both axes so the
    ``make_prior_from_param_set`` 15%-of-mean rule produces a sensible prior
    width on both parameters (otherwise b's default of 0 would yield σ_prior ≈
    1e-6 and the test would be measuring the prior, not the data).
    """

    def __init__(self):
        self._names = ["a", "b"]
        self._defaults = np.array([1.0, 1.0])  # both nonzero -> well-posed prior
        self._lower    = np.array([-2.0, -2.0])
        self._upper    = np.array([ 3.0,  3.0])

    def __getitem__(self, key):
        return {
            "defaults": self._defaults,
            "lower": self._lower,
            "upper": self._upper,
        }[key]
    def __iter__(self):       return iter(("defaults", "lower", "upper"))
    def __len__(self):        return 3

    def n_active(self):       return len(self._names)
    def active_names(self):   return list(self._names)
    def lower_bounds(self):   return list(self._lower)
    def upper_bounds(self):   return list(self._upper)
    def pack(self, _coeffs):  return list(self._defaults)
    def unpack(self, theta):  return list(theta)
    def in_bounds(self, theta):
        t = np.asarray(theta)
        return bool(np.all(t >= self._lower) and np.all(t <= self._upper))


@pytest.fixture(scope="module")
def toy_setup():
    """Toy linear model y = a*x + b with truth one prior-σ from the prior mean
    on each axis."""
    np.random.seed(0)
    x       = np.array([0.0, 1.0, 2.0, 3.0])
    a_true  = 0.85   # 1σ from prior mean of 1.0 (15% std)
    b_true  = 1.15   # 1σ from prior mean of 1.0
    sigmas  = np.full(4, 0.05)
    obs     = a_true * x + b_true + sigmas * np.random.randn(4)

    fwd       = ToyForward(x, obs, sigmas, theta_true=[a_true, b_true])
    param_set = ToyParamSet()
    truth     = np.array([a_true, b_true])
    return SimpleNamespace(
        fwd=fwd, param_set=param_set, x=x, obs=obs, sigmas=sigmas,
        truth=truth,
    )


# ---------------- Tests ------------------------------------------------

@pytest.fixture(scope="module")
def mode_runs(toy_setup):
    """ONE calibration per mode, shared by every assertion in this module.

    The structural assertions (types, modes, sample shapes) are configuration-
    independent, so they inspect the same runs the credible-interval test
    scores instead of reconstructing identical studies per test (the runs are
    the expensive part; this fixture halves the module's calibration work
    while preserving every assertion).
    """
    runs = {}
    for mode in ("no_discrepancy", "diagonal", "physical_gp"):
        runs[mode] = run_calibration(
            toy_setup.fwd, toy_setup.param_set,
            toy_setup.x, toy_setup.obs, toy_setup.sigmas,
            mode=mode, n_ensemble=25, n_steps=300, rng_seed=0, verbose=False,
        )
    return runs


class TestRunCalibrationModes:
    """Each mode constructs the right inference object and recovers truth."""

    def test_no_discrepancy(self, mode_runs):
        bi = mode_runs["no_discrepancy"]
        assert isinstance(bi, BayesianInference)
        assert bi.samples is not None
        assert bi.samples.shape[1] == 2

    def test_diagonal_mode(self, mode_runs):
        bi = mode_runs["diagonal"]
        assert isinstance(bi, BayesianInferenceKOH)
        assert bi.koh.mode == "diagonal"
        assert bi.n_extra == 1
        assert bi.samples.shape[1] == 3   # θ (2) + log σ_δ (1)

    def test_physical_gp_mode(self, mode_runs):
        bi = mode_runs["physical_gp"]
        assert isinstance(bi, BayesianInferenceKOH)
        assert bi.koh.mode == "physical_gp"
        assert bi.n_extra == 2
        assert bi.samples.shape[1] == 4   # θ (2) + log σ_δ + log l_δ

    @pytest.mark.parametrize("mode",
                             ["no_discrepancy", "diagonal", "physical_gp"])
    def test_truth_within_credible_interval(self, toy_setup, mode_runs, mode):
        bi = mode_runs[mode]
        s = bi.posterior_summary()
        for i, name in enumerate(toy_setup.param_set.active_names()):
            mean = s[name]["mean"]
            std  = s[name]["std"]
            err  = abs(mean - toy_setup.truth[i]) / max(std, 1e-9)
            assert err < 4.0, (
                f"mode={mode} {name}: posterior {mean:.3f}±{std:.3f} "
                f"vs truth {toy_setup.truth[i]:.3f} -> err={err:.1f}σ"
            )

    def test_unknown_mode_rejected(self, toy_setup):
        with pytest.raises(ValueError, match="unknown mode"):
            run_calibration(
                toy_setup.fwd, toy_setup.param_set,
                toy_setup.x, toy_setup.obs, toy_setup.sigmas,
                mode="something_else",
                n_ensemble=10, n_steps=50, rng_seed=0, verbose=False,
            )


class TestCompareModes:
    @pytest.fixture(scope="class")
    def comparison(self, toy_setup):
        return compare_modes(
            toy_setup.fwd, toy_setup.param_set,
            toy_setup.x, toy_setup.obs, toy_setup.sigmas,
            obs_metadata=make_obs_metadata([
                {"type": "linear", "location": (float(xi),),
                 "sigma": 0.05, "group": "linear"} for xi in toy_setup.x
            ]),
            n_ensemble=20, n_steps=200, rng_seed=0, verbose=False,
        )

    def test_compare_runs_all_three_modes(self, comparison):
        assert set(comparison["modes"]) == {"no_discrepancy", "diagonal",
                                             "physical_gp"}
        for m in comparison["modes"]:
            assert comparison["timings_s"][m] >= 0.0
            assert m in comparison["summaries"]

    def test_widths_and_shifts_have_all_params(self, comparison):
        names = ["a", "b"]
        widths = posterior_widths(comparison, names)
        shifts = posterior_shifts(comparison, names)
        for m in comparison["modes"]:
            assert set(widths[m].keys()) == set(names)
            assert set(shifts[m].keys()) == set(names)

    def test_discrepancy_summary_skips_no_discrepancy(self, comparison):
        d = discrepancy_summary(comparison)
        assert d["no_discrepancy"]["sigma_delta_mean"] is None
        assert "sigma_delta_mean" in d["diagonal"]
        assert "sigma_delta_mean" in d["physical_gp"]
        assert "l_delta_mean"     in d["physical_gp"]
        assert "l_delta_mean" not in d["diagonal"]

    def test_write_report(self, tmp_path, comparison, toy_setup):
        json_path = write_report(comparison, obs_values=toy_setup.obs,
                                  save_dir=tmp_path, truth=toy_setup.truth)
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "modes" in data
        assert "param_names" in data
        assert "posterior_widths" in data
        assert "discrepancy" in data
        # Identifiability flag must list the two KOH modes explicitly.
        flags = data["identifiability_flag"]
        for m in ("diagonal", "physical_gp"):
            assert m in flags
            for name in ("a", "b"):
                assert "inflation_ratio"   in flags[m][name]
                assert "weakly_identified" in flags[m][name]


class TestKOHDoesNotErasePosteriorWithoutDiscrepancy:
    """Sanity: when truth fits the model perfectly with no model-form error,
    KOH should not blow up the posterior so much that the data is ignored.
    Concretely: posterior std under KOH should remain finite and < 5x the prior std."""

    def test_posterior_widths_finite(self, toy_setup):
        bi_diag = run_calibration(
            toy_setup.fwd, toy_setup.param_set,
            toy_setup.x, toy_setup.obs, toy_setup.sigmas,
            mode="diagonal",
            n_ensemble=25, n_steps=300, rng_seed=0, verbose=False,
        )
        prior = make_prior_from_param_set(toy_setup.param_set)
        s     = bi_diag.posterior_summary()
        for i, name in enumerate(toy_setup.param_set.active_names()):
            sigma_post  = s[name]["std"]
            sigma_prior = prior.stds[i]
            assert np.isfinite(sigma_post)
            assert sigma_post < 5.0 * sigma_prior, (
                f"KOH inflated posterior σ ({sigma_post:.4g}) > 5x prior σ "
                f"({sigma_prior:.4g}) for {name}; KOH is eating data"
            )
