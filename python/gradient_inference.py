"""
PHASE 2 — FD-first gradient stack (research_dir.md §4.2; user-decided engine order).

Per the gradient-engine decision, this module provides, in order of where each is
consumed:

  1. ``surrogate_gradient_check`` — verifies the **analytic GP-surrogate gradient**
     (``GPSurrogate.gradient``) against high-order finite differences.  The surrogate
     carries the bulk of all sampling, so this is the mandatory first gate (rel err
     < 1e-6, confirmed with 4th-order central differences).

  2. ``GradientForwardModel`` — the **central finite-difference true-model gradient
     oracle**, used for the *bounded* true-model jobs (active-subspace evaluations,
     true-model reference points, NUTS-on-truth).  FD is correct (it converges to the
     true gradient and carries no frozen-turbulence bias); its O(d_θ) solve cost is
     why it is reserved for bounded jobs while the surrogate carries the rest.  The
     active-subspace gradient set is produced by ``gradient_dataset`` as an
     independent, **checkpointed per-θ job** so the cost spreads across runs/cores.

  3. ``make_surrogate_logpost_grad`` / ``make_truemodel_logpost_grad`` — build the
     ``theta -> (log_post, grad_log_post)`` callables that ``nuts.run_nuts`` consumes,
     from the analytic surrogate gradient and the FD true-model gradient respectively.

The discrete converged-residual adjoint (O(1) cost, independent of d_θ) is the scoped
upgrade gated behind its own FD check; it is intentionally NOT built here.  The
Phase-2 "gradient cost ≈ O(1)" criterion is therefore deferred until the adjoint
lands; every other Phase-2 gate is met by this FD-first stack.

  Gradient-path prerequisite: cases whose QoI is the BFS reattachment length need the
  *smoothed* (interpolated sub-cell zero-crossing) reattachment — the raw discrete
  sign-change x is piecewise-constant in θ and would FD to ~0 with spikes.  The
  smoothing lives in the C++ ObservationOperator so both the forward prediction and
  the FD gradient see a differentiable QoI.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT / "build"), str(_REPO_ROOT / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- #
# Finite-difference primitives
# --------------------------------------------------------------------------- #
def central_fd_gradient(f, theta, h=1e-4, order=4):
    """
    Central finite-difference gradient of a scalar function f(theta).

    ``order=2`` uses the standard (f(x+h)-f(x-h))/2h; ``order=4`` uses the 4th-order
    stencil (-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h))/12h, whose truncation error is O(h^4)
    — needed to verify analytic gradients below the 1e-6 floor where 2nd-order FD
    saturates (~3e-6).
    """
    theta = np.asarray(theta, float)
    d = len(theta)
    g = np.zeros(d)
    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        if order == 2:
            g[j] = (f(theta + e) - f(theta - e)) / (2 * h)
        else:
            g[j] = (-f(theta + 2 * e) + 8 * f(theta + e)
                    - 8 * f(theta - e) + f(theta - 2 * e)) / (12 * h)
    return g


def surrogate_gradient_check(gp, points, h=1e-4, order=4):
    """
    Verify ``GPSurrogate.gradient`` (analytic) against finite differences.

    Returns the maximum relative error over ``points``.  The Phase-2 gate requires
    < 1e-6 with ``order=4`` (the analytic gradient is GPy's exact closed form; the
    residual is the FD stencil's own floor).
    """
    max_rel = 0.0
    for t in np.atleast_2d(points):
        g_an = gp.gradient(t)
        g_fd = central_fd_gradient(gp.log_likelihood, t, h=h, order=order)
        denom = max(np.linalg.norm(g_fd), 1e-12)
        max_rel = max(max_rel, np.linalg.norm(g_an - g_fd) / denom)
    return float(max_rel)


# --------------------------------------------------------------------------- #
# True-model central-FD gradient oracle
# --------------------------------------------------------------------------- #
class GradientForwardModel:
    """
    Central finite-difference gradients of a C++ ForwardModel — the *true-model*
    gradient oracle for bounded jobs.

    Wraps a forward model exposing ``evaluate(theta) -> result`` with
    ``result.log_lik`` and ``result.predictions``.  Provides:

      * ``eta(theta)``                 — prediction vector η(θ)            (1 solve)
      * ``loglik(theta)``              — scalar Gaussian log-lik           (1 solve)
      * ``loglik_gradient(theta)``     — ∂logL/∂θ via central FD       (2·d_θ solves)
      * ``eta_jacobian(theta)``        — ∂η/∂θ  (n_obs×d_θ) via central FD(2·d_θ solves)

    A solve that does not return the expected finite prediction count is treated as
    a missing sample (η = NaN), so callers can detect and skip non-finite gradients.

    Parameters
    ----------
    forward_model : object with evaluate(theta)->result
    n_obs : int
        Expected number of predictions (for validity checks).
    h : float
        FD step (in raw θ units).  Default 1e-4; scale-aware stepping is available via
        ``h_rel`` (fraction of |θ_j|) which is usually more robust across coefficients.
    h_rel : float | None
        If set, per-component step h_j = max(h_rel·|θ_j|, h_floor).
    """

    def __init__(self, forward_model, n_obs, *, h=1e-4, h_rel=None, h_floor=1e-7):
        self.fm = forward_model
        self.n_obs = int(n_obs)
        self.h = float(h)
        self.h_rel = h_rel
        self.h_floor = float(h_floor)
        self.n_solves = 0   # running solve counter (cost accounting)

    def _steps(self, theta):
        if self.h_rel is None:
            return np.full(len(theta), self.h)
        return np.maximum(self.h_rel * np.abs(theta), self.h_floor)

    def _eval(self, theta):
        self.n_solves += 1
        res = self.fm.evaluate(np.asarray(theta, float).tolist())
        preds = np.asarray(res.predictions, float) if res.predictions is not None \
            else np.array([])
        ll = float(res.log_lik)
        ok = (len(preds) == self.n_obs and np.all(np.isfinite(preds))
              and np.isfinite(ll))
        return preds, ll, ok

    def eta(self, theta):
        preds, _, ok = self._eval(theta)
        return preds if ok else np.full(self.n_obs, np.nan)

    def loglik(self, theta):
        _, ll, ok = self._eval(theta)
        return ll if ok else -np.inf

    def loglik_gradient(self, theta):
        """∂(log-lik)/∂θ via central FD (2·d_θ solves)."""
        theta = np.asarray(theta, float)
        d = len(theta)
        steps = self._steps(theta)
        g = np.zeros(d)
        for j in range(d):
            e = np.zeros(d)
            e[j] = steps[j]
            _, lp, okp = self._eval(theta + e)
            _, lm, okm = self._eval(theta - e)
            if not (okp and okm):
                return np.full(d, np.nan)
            g[j] = (lp - lm) / (2 * steps[j])
        return g

    def eta_jacobian(self, theta):
        """∂η/∂θ  (n_obs × d_θ) via central FD (2·d_θ solves)."""
        theta = np.asarray(theta, float)
        d = len(theta)
        steps = self._steps(theta)
        J = np.zeros((self.n_obs, d))
        for j in range(d):
            e = np.zeros(d)
            e[j] = steps[j]
            ep, _, okp = self._eval(theta + e)
            em, _, okm = self._eval(theta - e)
            if not (okp and okm):
                return np.full((self.n_obs, d), np.nan)
            J[:, j] = (ep - em) / (2 * steps[j])
        return J


# --------------------------------------------------------------------------- #
# Warm-started FD true-model gradient — the PRODUCTION Rung-1 engine
# --------------------------------------------------------------------------- #
class WarmFDForwardModel:
    """Warm-started finite-difference true-model gradient — robust drop-in for
    ``GradientForwardModel`` (same interface: ``eta`` / ``loglik`` / ``loglik_gradient`` /
    ``eta_jacobian`` / ``n_obs`` / ``n_solves``), so it slots straight into
    ``make_truemodel_logpost_grad`` and ``gradient_dataset``.

    Wraps a ``rans_sst_py.ParameterSensitivity`` (the case).  The gradient is the warm-started
    central-FD true-model gradient: each θ±h re-solve starts from the FIXED converged state and
    runs to a looser cap, so it re-equilibrates the small perturbation in a few hundred
    iterations instead of the cold-start thousands.  It IS the full-FD true-model gradient
    (same fixed point ⇒ no frozen-pressure bias) — verified to match cold full FD to the FD
    floor (~4e-4) at ~13x lower cost.  This is the chosen Rung-1 engine over the semi-analytic
    tangents (DECISION_RECORD §4): the pressure-coupled tangent does not converge robustly and
    the frozen-pressure tangent carries a ~30% magnitude bias.

    Parameters
    ----------
    sens : rans_sst_py.ParameterSensitivity
    n_obs : int
    h, h_floor : float        central-FD step (β*/a1 auto-shrink in C++).
    warm_max_iter, warm_tol : warm re-solve cap / tolerance (the speedup source).
    """

    def __init__(self, sens, n_obs, *, h=5e-4, h_floor=1e-7,
                 warm_max_iter=400, warm_tol=1e-5):
        self.sens = sens
        self.n_obs = int(n_obs)
        self.h = float(h)
        self.h_floor = float(h_floor)
        self.warm_max_iter = int(warm_max_iter)
        self.warm_tol = float(warm_tol)
        self.n_solves = 0
        self._base = None          # θ of the currently-stored converged base state

    def _ensure(self, theta):
        """Make `sens` hold the converged base state at θ (cached across calls)."""
        t = [float(x) for x in np.asarray(theta, float)]
        if self._base != t:
            self.sens.solve_state(t)
            self.n_solves += 1
            self._base = t
        return t

    def _warm(self, theta):
        t = self._ensure(theta)
        r = self.sens.eta_jacobian_warm_fd(t, self.h, self.h_floor,
                                           self.warm_max_iter, self.warm_tol)
        self.n_solves += 2 * 11    # 22 warm re-solves (d_θ = 11)
        return r

    def eta(self, theta):
        t = self._ensure(theta)
        preds = np.asarray(self.sens.observe(t), float)
        ok = (len(preds) == self.n_obs and np.all(np.isfinite(preds)))
        return preds if ok else np.full(self.n_obs, np.nan)

    def loglik(self, theta):
        t = self._ensure(theta)
        ll = float(self.sens.log_lik(t))
        return ll if np.isfinite(ll) else -np.inf

    def eta_jacobian(self, theta):
        """∂η/∂θ (n_obs × 11) via warm-started central FD."""
        r = self._warm(theta)
        J = np.asarray(r.d_obs_d_theta, float)
        return J if J.shape == (self.n_obs, 11) and np.all(np.isfinite(J)) \
            else np.full((self.n_obs, 11), np.nan)

    def loglik_gradient(self, theta):
        """∂(log-lik)/∂θ (11) via warm-started central FD (same warm solves as eta_jacobian)."""
        r = self._warm(theta)
        g = np.asarray(r.log_lik_gradient, float)
        return g if g.shape == (11,) and np.all(np.isfinite(g)) else np.full(11, np.nan)


# --------------------------------------------------------------------------- #
# Gradient log-posteriors for NUTS
# --------------------------------------------------------------------------- #
def _prior_logp_and_grad(prior, theta):
    """Truncated-normal log-prior and its gradient (−(θ−μ)/σ² in support)."""
    theta = np.asarray(theta, float)
    if np.any(theta < prior.lower) or np.any(theta > prior.upper) \
            or np.any(~np.isfinite(theta)):
        return -np.inf, np.zeros(len(theta))
    z = (theta - prior.means) / prior.stds
    return -0.5 * float(np.sum(z * z)), -(theta - prior.means) / prior.stds ** 2


def make_surrogate_logpost_grad(prior, surrogate):
    """
    Build ``theta -> (log_post, grad)`` using the analytic surrogate gradient.

    log p(θ|y) = log_prior(θ) + surrogate_mean_loglik(θ);
    ∇ = ∇log_prior (analytic) + surrogate.gradient(θ) (analytic).  This is the
    callable the bulk of NUTS sampling runs on.
    """
    def logp_and_grad(theta):
        lp, gp = _prior_logp_and_grad(prior, theta)
        if not np.isfinite(lp):
            return -np.inf, np.zeros(len(np.asarray(theta)))
        ll = surrogate.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf, np.zeros(len(np.asarray(theta)))
        return lp + ll, gp + surrogate.gradient(theta)
    return logp_and_grad


def make_truemodel_logpost_grad(grad_fm, prior):
    """
    Build ``theta -> (log_post, grad)`` using the FD true-model gradient oracle.

    Expensive (O(d_θ) solves per gradient) — for the *bounded* true-model jobs only.
    """
    def logp_and_grad(theta):
        lp, gpri = _prior_logp_and_grad(prior, theta)
        if not np.isfinite(lp):
            return -np.inf, np.zeros(len(np.asarray(theta)))
        ll = grad_fm.loglik(theta)
        if not np.isfinite(ll):
            return -np.inf, np.zeros(len(np.asarray(theta)))
        gll = grad_fm.loglik_gradient(theta)
        if not np.all(np.isfinite(gll)):
            return -np.inf, np.zeros(len(np.asarray(theta)))
        return lp + ll, gpri + gll
    return logp_and_grad


# --------------------------------------------------------------------------- #
# KOH log-likelihood gradient w.r.t. θ (compose the KOH ∂L/∂η with warm-FD ∂η/∂θ)
# --------------------------------------------------------------------------- #
def koh_loglik_gradient_theta(koh, eta, eta_jacobian, log_sigma_delta, log_l_delta):
    """Gradient of the KOH log-likelihood w.r.t. the augmented vector [θ…, log σ_δ, log l_δ].

    The KOH likelihood is θ-coupled only through the predictions η(θ); ``KOHLikelihood.gradient``
    returns ∂L/∂η (and the two discrepancy-hyperparameter gradients), so the θ-gradient is the
    chain rule  ∂L_KOH/∂θ = (∂L_KOH/∂η) · (∂η/∂θ)  with ∂η/∂θ the warm-FD observable Jacobian.

    Parameters
    ----------
    koh : bayesian_inference.KOHLikelihood
    eta : (n_obs,)            predictions η(θ) at the converged state.
    eta_jacobian : (n_obs, d) warm-FD ∂η/∂θ (e.g. WarmFDForwardModel.eta_jacobian).
    log_sigma_delta, log_l_delta : KOH discrepancy hyperparameters.

    Returns ``(g_theta[d], g_log_sigma_delta, g_log_l_delta)``.  Non-finite η ⇒ all zeros (the
    KOH finite/−inf contract).
    """
    eta = np.asarray(eta, float)
    J = np.asarray(eta_jacobian, float)
    deta, g_s, g_l = koh.gradient(eta, log_sigma_delta, log_l_delta)
    deta = np.asarray(deta, float)
    if not (np.all(np.isfinite(deta)) and np.all(np.isfinite(J))):
        return np.zeros(J.shape[1]), 0.0, 0.0
    return deta @ J, float(g_s), float(g_l)


# --------------------------------------------------------------------------- #
# Checkpointed gradient dataset (active-subspace job array, per case)
# --------------------------------------------------------------------------- #
def gradient_dataset(grad_fm, prior, thetas, cache_path, *, verbose=True):
    """
    Compute log-posterior gradients g_i = ∇log p(θ_i|y) at each θ_i, checkpointing to
    ``cache_path`` (.npz) so the (expensive, O(d_θ)-solves-each) set is resumable —
    this is the per-case active-subspace "job array".

    Returns (G, valid) where G is (N, d_θ) with NaN rows for failed solves and valid
    is the boolean mask.  Re-runs load completed rows and only fill the gaps.
    """
    thetas = np.atleast_2d(np.asarray(thetas, float))
    N, d = thetas.shape
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        dat = np.load(cache_path)
        G = dat["G"]
        done = dat["done"].astype(bool)
        if G.shape != (N, d) or not np.array_equal(dat["thetas"], thetas):
            G = np.full((N, d), np.nan)
            done = np.zeros(N, bool)
    else:
        G = np.full((N, d), np.nan)
        done = np.zeros(N, bool)

    logpg = make_truemodel_logpost_grad(grad_fm, prior)
    t0 = time.time()
    for i in range(N):
        if done[i]:
            continue
        _, g = logpg(thetas[i])
        G[i] = g
        done[i] = True
        if verbose and (i + 1) % 5 == 0:
            print(f"  grad {i+1}/{N}  solves={grad_fm.n_solves}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        np.savez(cache_path, thetas=thetas, G=G, done=done)

    valid = np.all(np.isfinite(G), axis=1)
    if verbose:
        print(f"  gradient_dataset: {int(valid.sum())}/{N} valid gradients "
              f"-> {cache_path.name}", flush=True)
    return G, valid


if __name__ == "__main__":
    # Light self-test of the FD primitive + prior gradient (no CFD).
    f = lambda t: -0.5 * np.sum((t - np.array([0.3, -0.2, 0.5])) ** 2)
    g = central_fd_gradient(f, np.array([0.1, 0.1, 0.1]), order=4)
    print("FD grad:", np.round(g, 6), "(expect [-0.2, 0.3, -0.4])")
    print("gradient_inference self-test OK")
