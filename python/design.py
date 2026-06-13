"""
Experimental design utilities (ensemble sampling plans).

Extracted from bayesian_inference.py (2026-06-12 modularization).
"""

import numpy as np


def latin_hypercube(n, ndim, lower, upper):
    """
    Stratified Latin hypercube in [lower, upper]^ndim.

    Divides each dimension into n equal strata and places exactly one
    sample per stratum, ensuring uniform marginal coverage.  This gives
    much better space-filling than pure random sampling for the same
    budget, which matters when each evaluation is a full CFD solve.

    Parameters:
    n: int
        Number of samples (= number of CFD evaluations in ensemble).
    ndim: int
        Dimensionality (= number of active SST parameters)
    lower, upper : array-like, shape (ndim,)
        Physical bounds for each parameter

    Returns
    samples: ndarray, shape (n, ndim)
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    samples = np.empty((n, ndim))
    for j in range(ndim):
        perm = np.random.permutation(n)
        for i in range(n):
            lo = lower[j] + (upper[j] - lower[j]) * perm[i] / n
            hi = lower[j] + (upper[j] - lower[j]) * (perm[i] + 1) / n
            samples[i, j] = np.random.uniform(lo, hi)
    return samples
