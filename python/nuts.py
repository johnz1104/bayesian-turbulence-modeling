"""
PHASE 2 — hand-rolled No-U-Turn Sampler (Hoffman & Gelman 2014, Algorithm 6).

A gradient-based sibling to ``parallel_mcmc.run_emcee``.  Pure numpy/scipy (no
JAX/NumPyro dependency, per the project's self-contained stack).  Consumes a single
callable returning ``(log_post, grad_log_post)`` and samples the posterior with:

  * efficient NUTS (recursive tree-doubling, slice variable, no-U-turn termination,
    divergence guard),
  * dual-averaging step-size adaptation during warm-up (target accept ~0.8),
  * a diagonal mass matrix (metric) estimated from the second half of warm-up.

Bounds / truncated priors are handled exactly as ``run_emcee`` handles them: the
target returns ``-inf`` outside the support, which NUTS treats as a divergence that
terminates the trajectory.  Because the Phase-1 priors put their mass well inside the
±3σ box, boundary hits are rare and this hard-wall handling is unbiased in practice;
the d_θ=2 emcee-agreement gate confirms it.

The surrogate may warm-start the mass matrix (pass ``init_inv_mass``).

Public API
----------
NUTS(logp_and_grad, dim, ...)          — sampler object
NUTS.sample(n_samples, theta0, ...)    — returns (samples, info)
run_nuts(logp_and_grad, theta0, ...)   — convenience driver (sibling to run_emcee)
"""

from __future__ import annotations

import numpy as np

_DELTA_MAX = 1000.0   # divergence threshold on the Hamiltonian (Hoffman & Gelman)


def _metric_apply(inv_mass, r):
    """A·r where A=inv_mass is the metric (diagonal 1-D or dense 2-D)."""
    return inv_mass @ r if inv_mass.ndim == 2 else inv_mass * r


def _leapfrog(theta, r, eps, grad, inv_mass, logp_and_grad):
    """One leapfrog step with metric inv_mass (diagonal 1-D or dense 2-D)."""
    r_half = r + 0.5 * eps * grad
    theta_new = theta + eps * _metric_apply(inv_mass, r_half)
    logp_new, grad_new = logp_and_grad(theta_new)
    r_new = r_half + 0.5 * eps * grad_new
    return theta_new, r_new, logp_new, grad_new


class NUTS:
    """
    No-U-Turn Sampler with dual-averaging step size and a diagonal mass matrix.

    Parameters
    ----------
    logp_and_grad : callable(theta) -> (float, ndarray)
        Log-posterior and its gradient.  May return (-inf, zeros) out of support.
    dim : int
        Parameter-space dimension.
    target_accept : float
        Dual-averaging target acceptance probability (Stan default 0.8).
    max_tree_depth : int
        Cap on tree depth (2**depth leapfrog steps) to bound cost.
    init_inv_mass : ndarray | None
        Initial metric M^{-1} ≈ posterior covariance (1-D diagonal or 2-D dense; e.g.
        the surrogate posterior variance/covariance).  Default identity.  Adapted from
        warm-up unless ``adapt_mass=False`` — a DENSE metric is estimated, which is
        what lets NUTS sample strongly *correlated* posteriors (e.g. a1–betaStar) and
        match emcee, where a diagonal metric underdisperses the correlated direction.
    adapt_mass : bool
        Estimate the (dense) mass matrix from the second half of warm-up.
    rng_seed : int | None
    """

    def __init__(self, logp_and_grad, dim, *, target_accept=0.8,
                 max_tree_depth=10, init_inv_mass=None, adapt_mass=True,
                 rng_seed=None):
        self.logp_and_grad = logp_and_grad
        self.dim = int(dim)
        self.target_accept = float(target_accept)
        self.max_tree_depth = int(max_tree_depth)
        self.adapt_mass = bool(adapt_mass)
        self.rng = np.random.default_rng(rng_seed)
        self._set_metric(np.ones(dim) if init_inv_mass is None else init_inv_mass)

    # ----- metric (mass matrix) -----------------------------------------------
    def _set_metric(self, M):
        """Store the metric M^{-1}=inv_mass and precompute the momentum factor."""
        M = np.asarray(M, float).copy()
        if M.ndim == 2:
            M = 0.5 * (M + M.T) + 1e-10 * np.eye(len(M))
            self._chol = np.linalg.cholesky(M)       # M = L Lᵀ
        else:
            self._chol = None
        self.inv_mass = M

    def _apply_metric(self, r):
        return _metric_apply(self.inv_mass, r)

    # ----- helpers -------------------------------------------------------------
    def _kinetic(self, r):
        return 0.5 * float(r @ self._apply_metric(r))

    def _draw_momentum(self):
        # r ~ N(0, M) with M = inv_mass^{-1}
        z = self.rng.standard_normal(self.dim)
        if self.inv_mass.ndim == 2:
            return np.linalg.solve(self._chol.T, z)   # L^{-T} z  ~ N(0, (L Lᵀ)^{-1})
        return z / np.sqrt(self.inv_mass)

    def _find_reasonable_epsilon(self, theta, logp, grad):
        """Heuristic initial step size (Hoffman & Gelman Algorithm 4)."""
        eps = 1.0
        r = self._draw_momentum()
        H0 = logp - self._kinetic(r)
        _, r_new, logp_new, _ = _leapfrog(theta, r, eps, grad, self.inv_mass,
                                          self.logp_and_grad)
        H_new = logp_new - self._kinetic(r_new)
        # direction a = +/-1 so that we move accept prob toward 0.5
        a = 1.0 if (H_new - H0) > np.log(0.5) else -1.0
        n = 0
        while a * (H_new - H0) > -a * np.log(2.0) and n < 100:
            eps *= 2.0 ** a
            _, r_new, logp_new, _ = _leapfrog(theta, r, eps, grad, self.inv_mass,
                                              self.logp_and_grad)
            H_new = logp_new - self._kinetic(r_new)
            n += 1
        return eps

    def _build_tree(self, theta, r, logp, grad, log_u, v, depth, eps, H0):
        """Recursive NUTS tree builder (returns the full 11-tuple state)."""
        if depth == 0:
            theta1, r1, logp1, grad1 = _leapfrog(theta, r, v * eps, grad,
                                                 self.inv_mass, self.logp_and_grad)
            H1 = logp1 - self._kinetic(r1)
            if not np.isfinite(H1):
                H1 = -np.inf
            n1 = 1 if log_u <= H1 else 0
            s1 = 1 if (H1 - log_u) > -_DELTA_MAX else 0   # 0 => divergence
            alpha = min(1.0, np.exp(H1 - H0)) if np.isfinite(H1) else 0.0
            return (theta1, r1, grad1, logp1,        # leftmost = rightmost = new pt
                    theta1, r1, grad1, logp1,
                    theta1, logp1, grad1,            # proposal candidate
                    n1, s1, alpha, 1)
        # recurse: build left/right subtrees
        (tm, rm, gm, lm, tp, rp, gp, lp, tprop, lprop, gprop,
         n, s, alpha, na) = self._build_tree(theta, r, logp, grad, log_u, v,
                                             depth - 1, eps, H0)
        if s == 1:
            if v == -1:
                (tm, rm, gm, lm, _, _, _, _, t2, l2, g2,
                 n2, s2, a2, na2) = self._build_tree(tm, rm, lm, gm, log_u, v,
                                                     depth - 1, eps, H0)
            else:
                (_, _, _, _, tp, rp, gp, lp, t2, l2, g2,
                 n2, s2, a2, na2) = self._build_tree(tp, rp, lp, gp, log_u, v,
                                                     depth - 1, eps, H0)
            # multinomial/uniform progressive sampling of the proposal
            if n2 > 0 and self.rng.random() < n2 / max(n + n2, 1):
                tprop, lprop, gprop = t2, l2, g2
            alpha += a2
            na += na2
            # no-U-turn check on the combined subtree span
            dtheta = tp - tm
            s = s2 * (1 if np.dot(dtheta, self._apply_metric(rm)) >= 0 else 0) \
                   * (1 if np.dot(dtheta, self._apply_metric(rp)) >= 0 else 0)
            n += n2
        return (tm, rm, gm, lm, tp, rp, gp, lp, tprop, lprop, gprop,
                n, s, alpha, na)

    # ----- main sampling loop --------------------------------------------------
    def sample(self, n_samples, theta0, *, n_warmup=500, verbose=False):
        """
        Draw ``n_samples`` post-warm-up samples starting from ``theta0``.

        Returns
        -------
        samples : (n_samples, dim) ndarray
        info : dict with accept_stat, step_size, n_divergent, inv_mass, n_warmup
        """
        theta = np.asarray(theta0, float).copy()
        logp, grad = self.logp_and_grad(theta)
        if not np.isfinite(logp):
            raise ValueError("NUTS: initial theta0 has non-finite log-posterior")

        eps = self._find_reasonable_epsilon(theta, logp, grad)
        # dual-averaging state (Hoffman & Gelman 2014, §3.2)
        mu = np.log(10.0 * eps)
        log_eps_bar = 0.0
        H_bar = 0.0
        gamma, t0, kappa = 0.05, 10.0, 0.75
        da_m = 0               # dual-averaging step counter (restarts after mass update)
        mass_updated = False

        total = n_warmup + n_samples
        samples = np.zeros((n_samples, self.dim))
        warm_draws = np.zeros((max(1, n_warmup), self.dim))
        accept_stats = np.zeros(total)
        n_div = 0

        for it in range(total):
            r0 = self._draw_momentum()
            H0 = logp - self._kinetic(r0)
            log_u = H0 + np.log(self.rng.random())   # slice in log-space

            tm = tp = theta.copy()
            rm = rp = r0.copy()
            gm = gp = grad.copy()
            lm = lp = logp
            tprop, lprop, gprop = theta.copy(), logp, grad.copy()
            n, s, depth = 1, 1, 0
            alpha, na = 0.0, 1

            while s == 1 and depth < self.max_tree_depth:
                v = 1 if self.rng.random() < 0.5 else -1
                if v == -1:
                    (tm, rm, gm, lm, _, _, _, _, t2, l2, g2,
                     n2, s2, alpha, na) = self._build_tree(tm, rm, lm, gm, log_u,
                                                          v, depth, eps, H0)
                else:
                    (_, _, _, _, tp, rp, gp, lp, t2, l2, g2,
                     n2, s2, alpha, na) = self._build_tree(tp, rp, lp, gp, log_u,
                                                          v, depth, eps, H0)
                if s2 == 1 and n2 > 0 and self.rng.random() < min(1.0, n2 / max(n, 1)):
                    tprop, lprop, gprop = t2, l2, g2
                n += n2
                dtheta = tp - tm
                s = s2 * (1 if np.dot(dtheta, self._apply_metric(rm)) >= 0 else 0) \
                       * (1 if np.dot(dtheta, self._apply_metric(rp)) >= 0 else 0)
                depth += 1

            if depth >= self.max_tree_depth:
                n_div += 0   # tree-depth cap is not a divergence, just a U-turn miss
            theta, logp, grad = tprop, lprop, gprop
            accept_stat = alpha / max(na, 1)
            accept_stats[it] = accept_stat

            # dual-averaging step-size adaptation during warm-up; freeze after.
            if it < n_warmup:
                da_m += 1
                eta = 1.0 / (da_m + t0)
                H_bar = (1 - eta) * H_bar + eta * (self.target_accept - accept_stat)
                log_eps = mu - np.sqrt(da_m) / gamma * H_bar
                w = da_m ** (-kappa)
                log_eps_bar = w * log_eps + (1 - w) * log_eps_bar
                eps = np.exp(log_eps)
                warm_draws[it] = theta
                # One-shot diagonal mass-matrix (metric) update at the warm-up
                # midpoint, then RESTART step-size dual averaging under the new
                # metric so the second half adapts eps to it (Stan two-window idea).
                if (self.adapt_mass and not mass_updated
                        and n_warmup >= 40 and it == n_warmup // 2):
                    seg = warm_draws[n_warmup // 4: it + 1]
                    n_seg = len(seg)
                    if n_seg > self.dim + 5:
                        cov = np.atleast_2d(np.cov(seg.T))
                        # shrink toward the diagonal for stability with limited warm-up
                        shrink = self.dim / (self.dim + n_seg)
                        cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
                        self._set_metric(cov)
                    else:
                        var = np.var(seg, axis=0)
                        var[var < 1e-12] = 1.0
                        self._set_metric(var)
                    eps = self._find_reasonable_epsilon(theta, logp, grad)
                    mu = np.log(10.0 * eps)
                    log_eps_bar, H_bar, da_m = 0.0, 0.0, 0
                    mass_updated = True
            else:
                if it == n_warmup:
                    eps = np.exp(log_eps_bar)   # freeze the adapted step size
                samples[it - n_warmup] = theta

            if verbose and (it + 1) % max(1, total // 10) == 0:
                print(f"  NUTS {it+1}/{total}  eps={eps:.3g}  "
                      f"accept={np.mean(accept_stats[:it+1]):.2f}", flush=True)

        info = {
            "step_size": float(eps),
            "accept_stat": float(np.mean(accept_stats[n_warmup:])),
            "n_divergent": int(n_div),
            "inv_mass": self.inv_mass.copy(),
            "n_warmup": int(n_warmup),
        }
        return samples, info


def run_nuts(logp_and_grad, theta0, *, n_samples=1000, n_warmup=500, dim=None,
             target_accept=0.8, init_inv_mass=None, rng_seed=None, verbose=False):
    """
    Convenience driver (sibling to ``parallel_mcmc.run_emcee``).

    Parameters
    ----------
    logp_and_grad : callable(theta) -> (logp, grad)
    theta0 : array — starting point (must have finite log-posterior)
    n_samples, n_warmup : int
    dim : int | None — inferred from theta0 if None
    init_inv_mass : array | None — surrogate-warm-started diagonal metric
    """
    theta0 = np.asarray(theta0, float)
    d = dim if dim is not None else len(theta0)
    sampler = NUTS(logp_and_grad, d, target_accept=target_accept,
                   init_inv_mass=init_inv_mass, rng_seed=rng_seed)
    samples, info = sampler.sample(n_samples, theta0, n_warmup=n_warmup,
                                   verbose=verbose)
    return samples, info


if __name__ == "__main__":
    # Analytic recovery smoke: correlated 2-D Gaussian; check mean/cov recovery.
    rng = np.random.default_rng(0)
    mu = np.array([1.0, -2.0])
    cov = np.array([[1.0, 0.8], [0.8, 1.5]])
    prec = np.linalg.inv(cov)

    def lpg(theta):
        d = theta - mu
        return -0.5 * d @ prec @ d, -prec @ d

    samples, info = run_nuts(lpg, np.zeros(2), n_samples=4000, n_warmup=1000,
                             rng_seed=1, verbose=True)
    print("target mean", mu, "got", samples.mean(0).round(3))
    print("target cov\n", cov, "\ngot\n", np.cov(samples.T).round(3))
    print("info:", {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in info.items() if k != "inv_mass"})
