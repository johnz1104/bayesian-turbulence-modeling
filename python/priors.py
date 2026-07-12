"""
Priors over SST closure coefficients.

Extracted from bayesian_inference.py (2026-06-12 modularization).
"""

import numpy as np

from solver_bindings import _rs


#  Prior Class: Truncated Normal centered on Menter (1994) defaults

class Prior:
    """
    Truncated Normal prior over SST closure coefficients.

    Each parameter θ_i has:
      - mean                μ_i   (Menter 1994 default)
      - standard deviation  σ_i   (default: 15% of mean)
      - bounds      [lo_i, hi_i]  (physical positivity + stability constraints)

    log p(θ) = -0.5 * Σ_i ((θ_i - μ_i) / σ_i)^2   if all θ_i ∈ [lo_i, hi_i]
             = -inf                               otherwise
    """

    def __init__(self, means, stds, lower, upper):
        self.means = np.asarray(means, dtype=np.float64)
        self.stds  = np.asarray(stds,  dtype=np.float64)
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.ndim  = len(self.means)

    def log_prior(self, theta):
        """Evaluate log-prior at parameter vector theta."""
        theta = np.asarray(theta)
        if np.any(theta < self.lower) or np.any(theta > self.upper):
            return -np.inf
        if np.any(~np.isfinite(theta)):
            return -np.inf
        z = (theta - self.means) / self.stds
        return -0.5 * np.sum(z * z)

    def sample(self, n=1):
        """
        Draw n samples from the truncated normal prior via rejection.
        Rejection sampling is efficient low-dimensional parameter spaces (2-11 dims)
        since the truncation region is wide relative to the prior width.
        """
        samples = np.empty((n, self.ndim))
        for i in range(n):
            for j in range(self.ndim):
                while True:
                    s = np.random.normal(self.means[j], self.stds[j])
                    if self.lower[j] <= s <= self.upper[j]:
                        samples[i, j] = s
                        break
        return samples


def make_prior_from_param_set(param_set, relative_std=0.15):
    """
    Build a Prior from an InferenceParameterSet.
    Accepts either a C++ pybind11 object (has .pack() method) or
    a plain dict with keys 'defaults', 'lower', 'upper' for
    standalone testing without C++ bindings.

    Parameters
    param_set: InferenceParameterSet or dict
        Parameter set defining active SST coefficients.
    relative_std : float
        Prior standard deviation as fraction of mean (default 15%).

    Returns: Prior
    """
    if hasattr(param_set, 'pack'):
        # C++ pybind11 InferenceParameterSet. Note the dependency direction:
        # _rs() imports the compiled binding LAZILY, only on this branch, and
        # reaching this branch presupposes the caller already holds a compiled
        # object (so the binding demonstrably imports in this interpreter).
        # The dict branch below is the designed pure-Python path and never
        # touches the binding; a missing or ABI-mismatched rans_sst_py cannot
        # break dict-based use (pinned by test_priors_dict_path_needs_no_binding).
        defaults = param_set.pack(_rs().SSTCoefficients())
        lo = param_set.lower_bounds()
        hi = param_set.upper_bounds()
    else:
        # dict fallback for standalone testing
        defaults = param_set['defaults']
        lo = param_set['lower']
        hi = param_set['upper']

    means = np.array(defaults)
    stds = np.maximum(relative_std * np.abs(means), 1e-6)
    return Prior(means, stds, np.array(lo), np.array(hi))


def make_sampling_prior(param_set, relative_std=0.15, k_sigma=3.0):
    """
    PHASE 1 prior review for high-dimensional (up to 11-D) calibration.

    ``make_prior_from_param_set`` truncates the Menter-centred normal to the *full*
    physical [lowerBounds, upperBounds] box.  At d_θ = 2 that box is a reasonable
    LHS sampling region, but in high-D it is mostly empty under the prior: a 15%-std
    normal has essentially all its mass within ±3σ, so Latin-hypercube points placed
    across the full physical box waste the ensemble budget on near-zero-prior corners
    and (empirically) diverge the solver.

    This builds the same Menter-centred truncated normal but clips its support to
    ``mean ± k_sigma·σ`` (intersected with the physical bounds).  With the default
    ``k_sigma = 3`` only ~0.3%/dimension of prior mass is removed, while the LHS box
    that ``run_ensemble`` samples shrinks to the physically-defensible region — the
    same treatment is applied uniformly to all 11 coefficients, including the 9 that
    production studies have never exercised.

    Returns a ``Prior`` whose ``lower``/``upper`` are the narrowed support (used both
    as the truncation bounds and as the LHS sampling box).
    """
    base = make_prior_from_param_set(param_set, relative_std=relative_std)
    lo = np.maximum(base.means - k_sigma * base.stds, base.lower)
    hi = np.minimum(base.means + k_sigma * base.stds, base.upper)
    return Prior(base.means, base.stds, lo, hi)
