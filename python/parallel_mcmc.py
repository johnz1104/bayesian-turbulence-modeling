"""
PHASE 5 — Parallel-MCMC support utilities.

Provides a thin pool-context helper plus a single ``run_emcee`` driver used by
both ``BayesianInference`` and ``BayesianInferenceKOH``.  Parallelism is built
on top of the ``multiprocess`` library (a ``dill``-backed fork of stdlib
``multiprocessing``) which picks pybind11 / GPy objects much better than the
stock pickle implementation.

Notes
-----
emcee's Pool overhead is non-trivial; for surrogate-based log-posteriors that
take << 1 ms per evaluation, parallel often does not beat serial.  Parallel
MCMC is most useful when the log-posterior is itself expensive (e.g. evaluates
the full CFD forward model directly, ~ seconds per call).

The default of every public entry point is ``parallel=False`` so existing
calls are bit-for-bit identical with the previous behaviour.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


_PYTHON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PYTHON_DIR))


@contextmanager
def get_pool(parallel: bool, n_processes: int | None, externally_provided=None):
    """Yield a ``multiprocess.Pool`` (or None) suitable for emcee.

    ``externally_provided`` lets callers supply their own pool, which we
    forward unchanged.
    """
    if externally_provided is not None:
        yield externally_provided
        return
    if not parallel:
        yield None
        return

    from multiprocess import Pool   # noqa: WPS433
    n = n_processes or max(1, (os.cpu_count() or 1) - 1)
    pool = Pool(n)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()


def run_emcee(log_posterior, prior, n_walkers: int, n_steps: int,
              burn_in: int, thin: int, *,
              parallel: bool = False, pool=None, n_processes: int | None = None,
              progress: bool = True, p0_init=None, rng_seed: int | None = None
              ) -> tuple[np.ndarray, dict]:
    """Run emcee with the requested parallelism.

    Parameters
    ----------
    log_posterior : callable
        The full log-posterior (theta -> float).
    prior : Prior
        Used to initialise walkers near the prior mean if ``p0_init`` is None.
    n_walkers, n_steps, burn_in, thin : int
        Standard emcee budget knobs.
    parallel : bool
        If True, evaluate walker log-posteriors in parallel.  Only useful when
        ``log_posterior`` takes >> 1 ms per call.
    pool : optional
        Externally-managed pool.  Passing this overrides ``parallel``.
    n_processes : int, optional
        Number of pool workers (default: cpu_count - 1).
    p0_init : (n_walkers, ndim) array, optional
        Walker initial state; if not supplied, a small Gaussian ball around
        ``prior.means`` is used.
    rng_seed : int, optional
        Seed numpy's RNG for the walker initialisation step.

    Returns
    -------
    samples : ndarray, shape (n_eff, ndim)
        Posterior samples after burn-in / thinning.
    info : dict
        ``{"elapsed_s", "acceptance_min", "acceptance_max", "acceptance_mean",
            "n_steps", "parallel", "n_processes"}``
    """
    import emcee

    ndim = prior.ndim
    if n_walkers < 2 * ndim:
        n_walkers = 2 * ndim

    if rng_seed is not None:
        np.random.seed(rng_seed)

    if p0_init is None:
        p0 = np.empty((n_walkers, ndim))
        for i in range(n_walkers):
            p0[i] = prior.means + 0.01 * prior.stds * np.random.randn(ndim)
            p0[i] = np.clip(p0[i], prior.lower, prior.upper)
    else:
        p0 = np.asarray(p0_init, float)
        if p0.shape != (n_walkers, ndim):
            raise ValueError("p0_init must have shape (n_walkers, ndim)")

    with get_pool(parallel=parallel,
                   n_processes=n_processes,
                   externally_provided=pool) as effective_pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, log_posterior, pool=effective_pool,
        )
        t0 = time.time()
        sampler.run_mcmc(p0, n_steps, progress=progress)
        elapsed = time.time() - t0

    samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    af = sampler.acceptance_fraction

    info = {
        "elapsed_s":       float(elapsed),
        "acceptance_min":  float(np.min(af)),
        "acceptance_max":  float(np.max(af)),
        "acceptance_mean": float(np.mean(af)),
        "n_steps":         int(n_steps),
        "parallel":        bool(parallel or pool is not None),
        "n_processes":     int(n_processes) if (parallel and n_processes) else None,
        "sampler":         sampler,
    }
    return samples, info
