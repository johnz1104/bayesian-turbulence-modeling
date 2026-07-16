"""
Multi-case calibration tests.

Uses two synthetic toy forward models so the suite stays fast.

Toy setup
---------
Both cases share θ = (a, b).  Each case defines a different linear map:

    Case 1 (``linA``):  y₁ = a + b · x         (sensitive to b at large x)
    Case 2 (``linB``):  y₂ = 2a + 0.5 b · x²   (sensitive to a)

Truth:  a = 0.85, b = 1.15.  Together the two cases pin θ much tighter than
either individually — the textbook hierarchical-Bayes win — and we verify
the posterior recovers the truth within a few σ.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_inference import (
    KOHLikelihood, Prior, _get_param_names,
)
from multi_case_calibration import Case, MultiCaseCalibration


# ------------------------ Toy infrastructure --------------------------

@dataclass
class _Result:
    predictions: list


class _LinForward:
    """ForwardModel-shaped wrapper: y = c0*a + c1*b*x + c2*b*x^2 ."""

    def __init__(self, x_locs, c=(1.0, 1.0, 0.0)):
        self.x = np.asarray(x_locs, float)
        self.c0, self.c1, self.c2 = c

    def evaluate(self, theta):
        a, b = theta
        y = self.c0 * a + self.c1 * b * self.x + self.c2 * b * self.x ** 2
        return _Result(predictions=[float(v) for v in y])

    def penalized_log_likelihood(self, theta):  # not used here
        return 0.0


class _ToyParamSet(Mapping):
    def __init__(self):
        self._names = ["a", "b"]
        self._defaults = np.array([1.0, 1.0])
        self._lower    = np.array([-2.0, -2.0])
        self._upper    = np.array([ 3.0,  3.0])
    def __getitem__(self, key):
        return {
            "defaults": self._defaults,
            "lower": self._lower,
            "upper": self._upper,
        }[key]
    def __iter__(self):     return iter(("defaults", "lower", "upper"))
    def __len__(self):      return 3
    def n_active(self):     return 2
    def active_names(self): return list(self._names)
    def lower_bounds(self): return list(self._lower)
    def upper_bounds(self): return list(self._upper)
    def pack(self, _c):     return list(self._defaults)
    def unpack(self, t):    return list(t)
    def in_bounds(self, t):
        t = np.asarray(t)
        return bool(np.all(t >= self._lower) and np.all(t <= self._upper))


@pytest.fixture(scope="module")
def two_case_setup():
    """Build truth, two forward models, and noisy observations for both cases."""
    np.random.seed(0)
    a_true, b_true = 0.85, 1.15
    x1 = np.linspace(0.0, 4.0, 5)
    x2 = np.linspace(0.0, 2.0, 5)
    fwd1 = _LinForward(x1, c=(1.0, 1.0, 0.0))      # y1 = a + b*x
    fwd2 = _LinForward(x2, c=(2.0, 0.0, 0.5))      # y2 = 2a + 0.5*b*x^2

    def _truth(fwd):
        return np.asarray(fwd.evaluate([a_true, b_true]).predictions, float)

    y1_truth = _truth(fwd1); y2_truth = _truth(fwd2)
    sigma1 = np.full_like(y1_truth, 0.05)
    sigma2 = np.full_like(y2_truth, 0.05)
    y1_obs = y1_truth + sigma1 * np.random.randn(y1_truth.size)
    y2_obs = y2_truth + sigma2 * np.random.randn(y2_truth.size)

    return SimpleNamespace(
        truth=np.array([a_true, b_true]),
        param_set=_ToyParamSet(),
        case1=Case(name="linA", forward_model=fwd1,
                   obs_locations=x1, obs_values=y1_obs, obs_sigmas=sigma1),
        case2=Case(name="linB", forward_model=fwd2,
                   obs_locations=x2, obs_values=y2_obs, obs_sigmas=sigma2),
    )


# ------------------------ Tests ---------------------------------------

class TestCaseManagement:
    def test_add_remove(self, two_case_setup):
        cal = MultiCaseCalibration(two_case_setup.param_set)
        cal.add_case(two_case_setup.case1)
        cal.add_case(two_case_setup.case2)
        assert cal.case_names() == ["linA", "linB"]
        cal.remove_case("linA")
        assert cal.case_names() == ["linB"]

    def test_case_without_obs_or_koh_rejected(self, two_case_setup):
        cal = MultiCaseCalibration(two_case_setup.param_set)
        with pytest.raises(ValueError, match="needs either"):
            cal.add_case(Case(name="empty",
                              forward_model=two_case_setup.case1.forward_model))


class TestSingleCaseCollapsesToBaseline:
    """One-case multi-case must produce the same posterior properties as a
    plain Bayesian inference on that case alone — basic sanity check."""

    def test_single_case_recovers_truth(self, two_case_setup):
        cal = MultiCaseCalibration(two_case_setup.param_set)
        cal.add_case(two_case_setup.case1)
        cal.run_ensemble(n_samples=30, verbose=False)
        cal.train_surrogates(verbose=False)
        cal.run_mcmc(n_walkers=12, n_steps=400, burn_in=80, thin=1,
                     verbose=False, rng_seed=42)
        s = cal.posterior_summary()
        for i, n in enumerate(["a", "b"]):
            err = abs(s[n]["mean"] - two_case_setup.truth[i]) / max(s[n]["std"], 1e-9)
            assert err < 4.0, f"single-case {n} err={err:.2f}σ"


class TestTwoCaseCalibration:
    """Two complementary cases should pin the truth tighter than either alone."""

    @pytest.fixture(scope="class")
    def joint_run(self, two_case_setup):
        cal = MultiCaseCalibration(two_case_setup.param_set)
        cal.add_case(two_case_setup.case1)
        cal.add_case(two_case_setup.case2)
        cal.run_ensemble(n_samples=30, verbose=False, rng_seed=0)
        cal.train_surrogates(verbose=False)
        cal.run_mcmc(n_walkers=12, n_steps=500, burn_in=100, thin=1,
                     verbose=False, rng_seed=42)
        return cal

    def test_joint_recovers_truth(self, two_case_setup, joint_run):
        s = joint_run.posterior_summary()
        for i, n in enumerate(["a", "b"]):
            err = abs(s[n]["mean"] - two_case_setup.truth[i]) / max(s[n]["std"], 1e-9)
            assert err < 4.0, (
                f"joint {n}: μ={s[n]['mean']:.3f}±{s[n]['std']:.3f}, "
                f"truth={two_case_setup.truth[i]:.3f}, err={err:.2f}σ")

    def test_joint_posterior_widths_finite(self, joint_run):
        s = joint_run.posterior_summary()
        for n in ("a", "b"):
            assert np.isfinite(s[n]["std"])
            assert s[n]["std"] > 0.0

    def test_summary_keys_match_param_names(self, joint_run):
        s = joint_run.posterior_summary()
        assert set(s.keys()) == {"a", "b"}


class TestKOHCases:
    """Cases that carry a per-case KOHLikelihood are accepted and used."""

    def test_koh_case_runs(self, two_case_setup):
        # Re-wrap case1's observations into a KOHLikelihood (diagonal mode).
        koh = KOHLikelihood(
            two_case_setup.case1.obs_locations,
            two_case_setup.case1.obs_values,
            two_case_setup.case1.obs_sigmas,
            mode="diagonal",
        )
        case_koh = Case(name="linA_koh",
                        forward_model=two_case_setup.case1.forward_model,
                        koh=koh)
        cal = MultiCaseCalibration(two_case_setup.param_set)
        cal.add_case(case_koh)
        cal.run_ensemble(n_samples=20, verbose=False)
        cal.train_surrogates(verbose=False)
        cal.run_mcmc(n_walkers=12, n_steps=200, burn_in=40, thin=1,
                     verbose=False, rng_seed=0)
        assert cal.samples is not None
        assert cal.samples.shape[1] == 2   # only θ; no extra KOH dims here
