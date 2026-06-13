"""
PHASE 4 — Active learning loop with uncertainty sampling.

Implements a small but rigorous AL driver that wraps any callable
``forward`` returning ``(predictions_array, ok)`` for a θ vector.

Strategies
----------
``"random"``       : LHS or uniform sampling — control baseline.
``"max_var"``      : query the candidate maximizing total surrogate variance
                      Σ_j var_j(θ).  Standard mean-square-error minimizing
                      acquisition; works well in higher dimensions but can
                      over-explore corners in 2-D.
``"max_norm_var"`` : Σ_j var_j(θ) / σ_y_j², compensates for output-channel
                      scale differences.
``"max_min_dist"`` : space-filling acquisition: pick the candidate that
                      maximizes its minimum distance to the current training
                      set.  Surrogate-free; this is the strongest baseline
                      that almost always beats random for the same budget
                      (textbook maximin / greedy LHS).

Public API
----------
sample_lhs(n, lower, upper, rng)          - LHS sampler
ActiveLearner(forward, lower, upper, ...) - Driver class
run_loop(driver, n_init, n_queries)       - Driver method
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


_REPO_ROOT  = Path(__file__).resolve().parent.parent
_PYTHON_DIR = _REPO_ROOT / "python"
sys.path.insert(0, str(_PYTHON_DIR))

from bayesian_inference import (   # noqa: E402
    MultiOutputSurrogate, latin_hypercube,
)
from surrogate_diagnostics import multi_output_diagnostics  # noqa: E402


def sample_lhs(n: int, lower: np.ndarray, upper: np.ndarray,
               rng_seed: int = 0) -> np.ndarray:
    """LHS sampler that delegates to ``bayesian_inference.latin_hypercube``."""
    np.random.seed(rng_seed)
    return latin_hypercube(n, len(lower), lower, upper)


def sample_uniform(n: int, lower, upper,
                   rng_seed: int = 0) -> np.ndarray:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rng = np.random.default_rng(rng_seed)
    u = rng.random((n, len(lower)))
    return lower + u * (upper - lower)


@dataclass
class ALResult:
    iteration:    int
    n_train:      int
    n_valid:      int
    rmse_history: list[float]   = field(default_factory=list)
    elapsed_s:    list[float]   = field(default_factory=list)
    queries:      list[list]    = field(default_factory=list)


class ActiveLearner:
    """
    Multi-output Gaussian-process active learner.

    Parameters
    ----------
    forward : callable
        Forward map: ``forward(theta) -> (predictions, ok_flag)``.  ``ok=False``
        causes the candidate to be skipped without retraining.
    lower, upper : (d,) arrays
        Parameter-space bounds (used by both LHS and acquisition sampling).
    strategy : {"random", "max_var", "max_norm_var"}
    n_candidates : int
        Number of random candidates evaluated per query.
    rng_seed : int
        Seed for the candidate sampler.
    val_set : optional (X_val, Y_val) tuple
        If provided, the learner records validation RMSE after each iteration.
    """

    def __init__(self, forward, lower, upper, *,
                 strategy: str = "max_var",
                 n_candidates: int = 256,
                 rng_seed: int = 0,
                 val_set=None,
                 verbose: bool = True):
        if strategy not in ("random", "max_var", "max_norm_var",
                             "max_min_dist"):
            raise ValueError(f"Unknown strategy {strategy!r}")
        self.forward      = forward
        self.lower        = np.asarray(lower, float)
        self.upper        = np.asarray(upper, float)
        self.strategy     = strategy
        self.n_candidates = n_candidates
        self.rng_seed     = rng_seed
        self.val_set      = val_set
        self.verbose      = verbose

        self.X       = None
        self.Y       = None
        self.surrog  = None
        self.history = ALResult(iteration=0, n_train=0, n_valid=0)

    # ---- ensemble bootstrap ---------------------------------------------

    def initialize(self, n_init: int) -> None:
        """Sample n_init points by LHS, evaluate, and fit surrogate."""
        X0 = sample_lhs(n_init, self.lower, self.upper, rng_seed=self.rng_seed)
        Xs, Ys = [], []
        for k in range(n_init):
            preds, ok = self.forward(X0[k])
            if ok:
                Xs.append(X0[k]); Ys.append(preds)
        if len(Xs) < 3:
            raise RuntimeError(f"initial ensemble has only {len(Xs)} valid "
                                f"points; need at least 3")
        self.X = np.asarray(Xs); self.Y = np.asarray(Ys)
        self.history.n_train = len(self.X)
        self._refit()

    # ---- one query / step -----------------------------------------------

    def _refit(self) -> None:
        # MultiOutputSurrogate has no constructor seed; determinism is
        # already controlled by np.random.seed earlier in the loop.
        self.surrog = MultiOutputSurrogate()
        self.surrog.train(self.X, self.Y)
        if self.val_set is not None:
            X_val, Y_val = self.val_set
            diag = multi_output_diagnostics(self.surrog, X_val, Y_val)
            self.history.rmse_history.append(diag["aggregate"]["mean_rmse"])

    def _propose_candidates(self, rng: np.random.Generator) -> np.ndarray:
        u = rng.random((self.n_candidates, len(self.lower)))
        return self.lower + u * (self.upper - self.lower)

    def _query_next(self, rng: np.random.Generator) -> np.ndarray:
        """Return the next θ to evaluate."""
        if self.strategy == "random":
            cand = self._propose_candidates(rng)
            return cand[0]
        cand = self._propose_candidates(rng)
        if self.strategy == "max_min_dist":
            # Surrogate-free maximin: pick candidate whose min-distance to the
            # current X is largest.  Strong space-filling baseline.
            d2 = np.sum((cand[:, None, :] - self.X[None, :, :]) ** 2, axis=-1)
            score = d2.min(axis=1)
            return cand[int(np.argmax(score))]
        means, vars_ = self.surrog.predict_batch(cand)   # (n_cand, n_out)
        if self.strategy == "max_var":
            score = vars_.sum(axis=1)
        else:  # max_norm_var
            scale = np.maximum(np.std(self.Y, axis=0), 1e-12) ** 2
            score = (vars_ / scale).sum(axis=1)
        return cand[int(np.argmax(score))]

    # ---- driver loop -----------------------------------------------------

    def run(self, n_queries: int):
        """Run n_queries acquisition iterations."""
        rng = np.random.default_rng(self.rng_seed + 1)
        for q in range(n_queries):
            t0 = time.time()
            theta_next = self._query_next(rng)
            preds, ok = self.forward(theta_next)
            if ok:
                self.X = np.vstack([self.X, theta_next])
                self.Y = np.vstack([self.Y, preds])
                self._refit()
                self.history.iteration += 1
                self.history.n_train = len(self.X)
                self.history.queries.append(list(theta_next))
                self.history.elapsed_s.append(time.time() - t0)
                if self.verbose:
                    rmse = self.history.rmse_history[-1] \
                        if self.history.rmse_history else float("nan")
                    print(f"  AL[{q+1}/{n_queries}] strategy={self.strategy} "
                          f"n_train={self.history.n_train} mean_rmse={rmse:.4g} "
                          f"elapsed={time.time()-t0:.2f}s")
            else:
                if self.verbose:
                    print(f"  AL[{q+1}/{n_queries}] forward failed at "
                          f"theta={theta_next}, skipping")
        return self.history


# ---- Convenience: wrap a C++ ForwardModel into the AL forward API ---

def cpp_forward_adapter(forward_model, koh_n=None):
    """Adapter that turns a C++ ForwardModel into the AL forward callable.

    Returns a function ``forward(theta) -> (predictions, ok_flag)``.  The
    forward model is considered ``ok`` only when:

      * the result.predictions list has the expected length (if ``koh_n`` is
        provided),
      * all entries are finite.
    """
    def _fwd(theta):
        result = forward_model.evaluate(list(theta))
        preds  = np.asarray(result.predictions, float) \
            if result.predictions else np.empty(0)
        if koh_n is not None and len(preds) != koh_n:
            return preds, False
        if len(preds) == 0 or not np.all(np.isfinite(preds)):
            return preds, False
        return preds, True
    return _fwd
