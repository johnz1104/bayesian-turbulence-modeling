"""
PHASE 8 — Multi-case hierarchical calibration.

Framework for combining multiple cases (e.g. channel + BFS, several Mach
numbers, several Re numbers) into a single Bayesian calibration that shares
the SST coefficient block ``θ`` while allowing per-case discrepancy
hyperparameters in the KOH likelihood.

Design
------
``MultiCaseCalibration`` registers a list of ``Case`` objects.  Each case
contains:

  - a forward model (incompressible or compressible) — must expose
    ``evaluate(theta)`` returning a result with a ``predictions`` list,
  - a label / name,
  - an optional ``KOHLikelihood`` for that case's observations,
  - a per-case weight (defaults to 1.0).

The joint log-posterior is

    log p(θ, {η_i}, {koh_i} | data) = log p(θ)
                                       + Σ_i w_i · log p(data_i | η_i, koh_i)

where the case-i forward model produces η_i.  Each case carries its own
``MultiOutputSurrogate`` so MCMC remains cheap.

Public API
----------
``Case``                 - dataclass-like wrapper around (name, fm, koh, weight)
``MultiCaseCalibration`` - the orchestrator with ``run_ensemble``,
                           ``train_surrogates``, ``run_mcmc``, ``posterior_summary``

This intentionally re-uses ``BayesianInferenceKOH``-style surrogate training
and ``parallel_mcmc.run_emcee`` for sampling, so PHASE 4 / PHASE 5 features
are available transparently.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


_REPO_ROOT  = Path(__file__).resolve().parent.parent
_BUILD_DIR  = _REPO_ROOT / "build"
_PYTHON_DIR = _REPO_ROOT / "python"
sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PYTHON_DIR))

from bayesian_inference import (   # noqa: E402
    Prior, MultiOutputSurrogate, KOHLikelihood,
    latin_hypercube, make_prior_from_param_set, _get_param_names,
)
from parallel_mcmc import run_emcee  # noqa: E402


# ------------------------ Case -----------------------------------------

@dataclass
class Case:
    """One calibration case in a multi-case study.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. ``"channel_Ma01"``, ``"bfs_DS1985"``).
    forward_model : object
        Anything exposing ``evaluate(theta_list) -> result`` where ``result``
        has ``.predictions`` (a finite list/array on success).
    koh : KOHLikelihood, optional
        Per-case KOH likelihood.  If None, a Gaussian likelihood with
        ``obs_sigmas`` is used (no model-form discrepancy).
    obs_locations, obs_values, obs_sigmas : array-like
        Required when ``koh`` is None; otherwise inferred from ``koh``.
    weight : float
        Likelihood weight (default 1.0); useful when one dataset is much
        denser than another and you want to balance influence.
    """
    name:           str
    forward_model:  object
    koh:            Optional[KOHLikelihood] = None
    obs_locations:  Optional[np.ndarray]    = None
    obs_values:     Optional[np.ndarray]    = None
    obs_sigmas:     Optional[np.ndarray]    = None
    weight:         float                   = 1.0
    surrogate:      Optional[MultiOutputSurrogate] = None
    ensemble_X:     Optional[np.ndarray]    = None
    ensemble_Y:     Optional[np.ndarray]    = None

    def n_obs(self) -> int:
        if self.koh is not None:
            return self.koh.n
        return len(self.obs_values)


# ------------------------ Driver ---------------------------------------

class MultiCaseCalibration:
    """
    Multi-case Bayesian calibration with shared θ block.

    Parameters
    ----------
    param_set : InferenceParameterSet
        Shared parameter set across all cases (defines θ).
    theta_prior : Prior, optional
        Prior on θ.  Defaults to the param_set-derived prior.
    """

    def __init__(self, param_set, theta_prior: Optional[Prior] = None):
        self.param_set = param_set
        self.cases: List[Case] = []
        if theta_prior is None:
            theta_prior = make_prior_from_param_set(param_set)
        self.theta_prior = theta_prior
        self.n_theta     = theta_prior.ndim
        self.prior       = theta_prior   # alias: KOH-free version uses θ only
        self.samples     = None
        self.sampler     = None

    # -------- case management --------

    def add_case(self, case: Case) -> None:
        if case.koh is None:
            if case.obs_values is None or case.obs_sigmas is None:
                raise ValueError(
                    f"case {case.name!r}: needs either a koh likelihood or "
                    "obs_values + obs_sigmas")
            case.obs_values = np.asarray(case.obs_values, float)
            case.obs_sigmas = np.asarray(case.obs_sigmas, float)
        self.cases.append(case)

    def remove_case(self, name: str) -> None:
        self.cases = [c for c in self.cases if c.name != name]

    def case_names(self) -> List[str]:
        return [c.name for c in self.cases]

    # -------- ensemble --------

    def run_ensemble(self, n_samples: int = 50, verbose: bool = True,
                     rng_seed: int = 0):
        """Run an LHS ensemble; evaluate every forward model for every θ.

        A θ is kept only if all cases produce the expected number of finite
        predictions; otherwise the θ is dropped from all cases (so the
        surrogates use a consistent training set).
        """
        np.random.seed(rng_seed)
        X = latin_hypercube(n_samples, self.n_theta,
                             self.theta_prior.lower, self.theta_prior.upper)

        # Collect per-case predictions; drop any θ that fails any case.
        valid_X = []
        valid_Y = [[] for _ in self.cases]
        t0 = time.time()
        for i in range(n_samples):
            theta = X[i].tolist()
            preds_per_case = []
            ok = True
            for c in self.cases:
                result = c.forward_model.evaluate(theta)
                pred   = (np.asarray(result.predictions, float)
                           if result.predictions else np.empty(0))
                if len(pred) != c.n_obs() or not np.all(np.isfinite(pred)):
                    ok = False
                    break
                preds_per_case.append(pred)
            if ok:
                valid_X.append(X[i])
                for j, p in enumerate(preds_per_case):
                    valid_Y[j].append(p)
            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  Ensemble {i+1}/{n_samples}  valid={len(valid_X)}  "
                      f"[{(i+1)/elapsed:.1f} eval/s, {len(self.cases)} cases]")

        if not valid_X:
            raise RuntimeError("No valid θ in ensemble — every θ failed some case")

        X_valid = np.asarray(valid_X)
        for j, c in enumerate(self.cases):
            c.ensemble_X = X_valid
            c.ensemble_Y = np.asarray(valid_Y[j])
        if verbose:
            elapsed = time.time() - t0
            print(f"\n  Joint ensemble: {len(X_valid)}/{n_samples} valid "
                  f"[{elapsed:.1f}s, {len(self.cases)} cases]")
        return X_valid

    # -------- surrogate training --------

    def train_surrogates(self, optimize_restarts: int = 3,
                          verbose: bool = True) -> None:
        for c in self.cases:
            if c.ensemble_X is None or c.ensemble_Y is None:
                raise RuntimeError(
                    f"case {c.name!r}: run_ensemble() must precede "
                    "train_surrogates()")
            t0 = time.time()
            c.surrogate = MultiOutputSurrogate()
            c.surrogate.train(c.ensemble_X, c.ensemble_Y,
                              optimize_restarts=optimize_restarts)
            if verbose:
                print(f"  Surrogate trained for {c.name}: "
                      f"{c.ensemble_X.shape[0]} samples × {c.n_obs()} outputs "
                      f"[{time.time()-t0:.2f}s]")

    # -------- joint log-posterior --------

    def log_posterior(self, theta) -> float:
        lp = self.theta_prior.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll_total = 0.0
        for c in self.cases:
            if c.surrogate is None:
                return -np.inf  # ensemble/training not done
            eta, _ = c.surrogate.predict(theta)
            if not np.all(np.isfinite(eta)):
                return -np.inf
            if c.koh is not None:
                ll = c.koh(eta, log_sigma_delta=-3.0, log_l_delta=0.0)
                # If the user wants the KOH hyperparams to be inferred jointly,
                # they should subclass; here we use a moderate fixed σ_δ.
            else:
                r = c.obs_values - eta
                ll = -0.5 * float(
                    np.sum((r / c.obs_sigmas) ** 2)
                    + np.sum(np.log(2.0 * np.pi * c.obs_sigmas ** 2))
                )
            if not np.isfinite(ll):
                return -np.inf
            ll_total += c.weight * ll
        return lp + ll_total

    # -------- MCMC --------

    def run_mcmc(self, n_walkers: int | None = None, n_steps: int = 500,
                 burn_in: int = 100, thin: int = 1, verbose: bool = True,
                 *, parallel: bool = False, pool=None,
                 n_processes: int | None = None, rng_seed: int | None = None):
        """Sample the shared-θ posterior given all currently-registered cases."""
        ndim = self.n_theta
        if n_walkers is None:
            n_walkers = max(16, 2 * ndim)

        if verbose:
            mode = "parallel" if (parallel or pool is not None) else "serial"
            print(f"  MultiCase MCMC ({mode}): {len(self.cases)} cases, "
                  f"{n_walkers} walkers × {n_steps} steps")

        self.samples, info = run_emcee(
            log_posterior=self.log_posterior,
            prior=self.theta_prior,
            n_walkers=n_walkers, n_steps=n_steps,
            burn_in=burn_in, thin=thin,
            parallel=parallel, pool=pool, n_processes=n_processes,
            progress=verbose, rng_seed=rng_seed,
        )
        self.sampler = info["sampler"]

        if verbose:
            print(f"  MCMC complete: {len(self.samples)} samples "
                  f"[{info['elapsed_s']:.1f}s]")
            print(f"  Acceptance: mean={info['acceptance_mean']:.3f}  "
                  f"range=[{info['acceptance_min']:.3f}, "
                  f"{info['acceptance_max']:.3f}]")
        return self.samples

    # -------- posterior summary --------

    def posterior_summary(self) -> dict:
        if self.samples is None:
            raise RuntimeError("run_mcmc() must precede posterior_summary()")
        names = _get_param_names(self.param_set)
        out = {}
        for i, n in enumerate(names):
            s = self.samples[:, i]
            out[n] = {
                "mean":        float(np.mean(s)),
                "std":         float(np.std(s)),
                "ci_2.5":      float(np.percentile(s, 2.5)),
                "ci_97.5":     float(np.percentile(s, 97.5)),
                "prior_mean":  float(self.theta_prior.means[i]),
                "prior_std":   float(self.theta_prior.stds[i]),
                "shift":       float((np.mean(s) - self.theta_prior.means[i])
                                      / max(self.theta_prior.stds[i], 1e-10)),
            }
        return out

    def print_summary(self) -> None:
        s = self.posterior_summary()
        names = _get_param_names(self.param_set)
        print(f"\n  Multi-case calibration ({len(self.cases)} cases): "
              f"{', '.join(self.case_names())}")
        print(f"\n  {'Parameter':>14s}  {'Prior μ':>8s}  {'Posterior μ':>11s}  "
              f"{'±σ':>7s}  {'95% CI':>20s}  {'Shift':>7s}")
        print("  " + "-" * 78)
        for n in names:
            r = s[n]
            print(f"  {n:>14s}  {r['prior_mean']:8.4f}  {r['mean']:11.4f}  "
                  f"{r['std']:7.4f}  [{r['ci_2.5']:.4f}, {r['ci_97.5']:.4f}]  "
                  f"{r['shift']:+7.2f}σ")
