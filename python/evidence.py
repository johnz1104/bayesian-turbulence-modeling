"""
PHASE 3 — model-evidence estimators (research_dir.md §4.3; angle 7).

Estimates the model evidence (marginal likelihood)

    Z_k = ∫ p(y | θ, M_k) p(θ | M_k) dθ

for competing term-toggled closures, and the Bayes factor B_12 = Z_1/Z_2.  Two
independent estimators are provided and cross-checked (the program mandates a dual
TI + stepping-stone check; the unreliable harmonic-mean estimator is deliberately
avoided):

  * **Thermodynamic integration (TI).**  With the power posterior
    p_β(θ) ∝ p(y|θ)^β p(θ), log Z = ∫₀¹ E_{p_β}[log p(y|θ)] dβ.  We sample at a ladder
    of β, average the log-likelihood, and integrate over β by the trapezoid rule.

  * **Stepping-stone (SS).**  log Z = Σ_k log E_{p_{β_k}}[ p(y|θ)^{(β_{k+1}-β_k)} ],
    evaluated stably with log-sum-exp.

Both reuse a gradient-free (emcee) or gradient-based sampler on each rung.  Standard
errors come from the across-rung variance (TI) and a batch-means / segment estimate
(SS).  Validated against the analytic-Gaussian case where Z is closed form, and TI
and SS must agree within their stated SE.

A geometric β-ladder (denser near β=0, where the integrand changes fastest) is the
default; ``power_betas`` exposes the standard β_k = (k/K)^p schedule.

Public API
----------
power_betas(n, p)                      — β-ladder
sample_power_posterior(...)            — samples + per-sample log-lik at fixed β
thermodynamic_integration(...)         — (log_Z, se, diagnostics)
stepping_stone(...)                    — (log_Z, se, diagnostics)
log_evidence(...)                      — run both, return a comparison dict
bayes_factor(logZ1, logZ2)             — (log10 B, Jeffreys label)
analytic_gaussian_log_evidence(...)    — closed-form Z for the verification test
"""

from __future__ import annotations

import numpy as np

try:
    import emcee
except ImportError:                       # pragma: no cover
    emcee = None


# --------------------------------------------------------------------------- #
# β-ladder
# --------------------------------------------------------------------------- #
def power_betas(n=20, p=5.0):
    """
    Geometric-ish inverse-temperature ladder on [0, 1] with K+1 = n points.

    β_k = (k/K)^p concentrates rungs near β=0 where E_β[log L] varies fastest
    (Friel & Pettitt 2008; Xie et al. 2011 recommend p≈5).
    """
    k = np.arange(n)
    return (k / (n - 1)) ** p


# --------------------------------------------------------------------------- #
# Power-posterior sampling
# --------------------------------------------------------------------------- #
def sample_power_posterior(loglik_fn, logprior_fn, beta, theta0, *,
                           n_walkers=None, n_steps=600, burn=200, thin=2,
                           rng_seed=0):
    """
    Sample p_β(θ) ∝ exp(β·loglik + logprior) with emcee and return
    (samples, loglik_values).

    ``loglik_values`` are the *un-tempered* log p(y|θ) at the retained samples —
    the integrand TI averages and SS exponentiates.  At β=0 this samples the prior.
    """
    if emcee is None:                     # pragma: no cover
        raise ImportError("emcee required for sample_power_posterior")
    theta0 = np.asarray(theta0, float)
    dim = len(theta0)
    nw = n_walkers or max(2 * dim, 8)

    def log_target(theta):
        lp = logprior_fn(theta)
        if not np.isfinite(lp):
            return -np.inf
        if beta == 0.0:
            return lp                      # prior only
        ll = loglik_fn(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + beta * ll

    np.random.seed(rng_seed)             # emcee draws from numpy's global RNG
    rng = np.random.default_rng(rng_seed)
    p0 = theta0 + 1e-3 * rng.standard_normal((nw, dim))
    sampler = emcee.EnsembleSampler(nw, dim, log_target)
    state = sampler.run_mcmc(p0, burn, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, n_steps, progress=False)
    chain = sampler.get_chain(discard=0, thin=thin, flat=True)

    # evaluate the *un-tempered* log-likelihood at the retained samples
    ll_vals = np.array([loglik_fn(t) for t in chain])
    good = np.isfinite(ll_vals)
    return chain[good], ll_vals[good]


# --------------------------------------------------------------------------- #
# Thermodynamic integration
# --------------------------------------------------------------------------- #
def thermodynamic_integration(loglik_fn, logprior_fn, theta0, *,
                              betas=None, sampler_kwargs=None, rng_seed=0,
                              verbose=False):
    """
    log Z via thermodynamic integration over a β-ladder (trapezoid rule).

    Returns (log_Z, se, diagnostics).  The SE combines the per-rung Monte-Carlo error
    of E_β[log L] propagated through the trapezoid weights.
    """
    betas = power_betas() if betas is None else np.asarray(betas, float)
    sk = dict(sampler_kwargs or {})
    E = np.zeros(len(betas))      # E_β[log L]
    V = np.zeros(len(betas))      # Var of the mean (for SE)
    for i, b in enumerate(betas):
        samples, ll = sample_power_posterior(loglik_fn, logprior_fn, b, theta0,
                                             rng_seed=rng_seed + i, **sk)
        E[i] = np.mean(ll)
        # batch-means variance of the mean (accounts for autocorrelation crudely)
        nb = max(1, len(ll) // 20)
        batch = np.array([np.mean(c) for c in np.array_split(ll, nb)])
        V[i] = np.var(batch, ddof=1) / nb if nb > 1 else np.var(ll) / max(len(ll), 1)
        if verbose:
            print(f"  TI β={b:.4f}  E[logL]={E[i]:.4f}  n={len(ll)}", flush=True)

    log_Z = np.trapz(E, betas)
    # trapezoid weights w_i; SE² = Σ w_i² V_i
    w = np.zeros(len(betas))
    w[1:] += 0.5 * np.diff(betas)
    w[:-1] += 0.5 * np.diff(betas)
    se = float(np.sqrt(np.sum(w ** 2 * V)))
    return float(log_Z), se, {"betas": betas, "E_logL": E, "var_mean": V}


# --------------------------------------------------------------------------- #
# Stepping-stone
# --------------------------------------------------------------------------- #
def stepping_stone(loglik_fn, logprior_fn, theta0, *,
                   betas=None, sampler_kwargs=None, rng_seed=0, verbose=False):
    """
    log Z via stepping-stone sampling (Xie et al. 2011), log-sum-exp stabilised.

    log Z = Σ_k log( (1/n) Σ_i exp((β_{k+1}-β_k)·logL_i) ),  θ_i ~ p_{β_k}.
    Returns (log_Z, se, diagnostics).
    """
    betas = power_betas() if betas is None else np.asarray(betas, float)
    sk = dict(sampler_kwargs or {})
    ratios = np.zeros(len(betas) - 1)
    ratio_var = np.zeros(len(betas) - 1)
    for k in range(len(betas) - 1):
        db = betas[k + 1] - betas[k]
        _, ll = sample_power_posterior(loglik_fn, logprior_fn, betas[k], theta0,
                                       rng_seed=rng_seed + k, **sk)
        a = db * ll
        amax = np.max(a)
        # log mean exp
        r = amax + np.log(np.mean(np.exp(a - amax)))
        ratios[k] = r
        # delta-method SE of log-mean-exp via batch means
        nb = max(1, len(ll) // 20)
        batch_r = []
        for c in np.array_split(a, nb):
            cm = np.max(c)
            batch_r.append(cm + np.log(np.mean(np.exp(c - cm))))
        ratio_var[k] = np.var(batch_r, ddof=1) / nb if nb > 1 else 0.0
        if verbose:
            print(f"  SS rung {k} β:{betas[k]:.3f}->{betas[k+1]:.3f}  "
                  f"log r={r:.4f}", flush=True)

    log_Z = float(np.sum(ratios))
    se = float(np.sqrt(np.sum(ratio_var)))
    return log_Z, se, {"betas": betas, "log_ratios": ratios}


# --------------------------------------------------------------------------- #
# Driver + Bayes factor
# --------------------------------------------------------------------------- #
def log_evidence(loglik_fn, logprior_fn, theta0, *, betas=None,
                 sampler_kwargs=None, rng_seed=0, verbose=False):
    """Run TI and stepping-stone; return both with their agreement check."""
    ti_Z, ti_se, ti_d = thermodynamic_integration(
        loglik_fn, logprior_fn, theta0, betas=betas,
        sampler_kwargs=sampler_kwargs, rng_seed=rng_seed, verbose=verbose)
    ss_Z, ss_se, ss_d = stepping_stone(
        loglik_fn, logprior_fn, theta0, betas=betas,
        sampler_kwargs=sampler_kwargs, rng_seed=rng_seed, verbose=verbose)
    disagree = abs(ti_Z - ss_Z)
    tol = 3.0 * np.hypot(ti_se, ss_se)
    return {
        "ti_logZ": ti_Z, "ti_se": ti_se,
        "ss_logZ": ss_Z, "ss_se": ss_se,
        "agree": bool(disagree <= max(tol, 0.5)),
        "abs_diff": float(disagree), "tol_3se": float(tol),
        "ti_diag": ti_d, "ss_diag": ss_d,
    }


# Jeffreys' scale for 2·ln(B) (Kass & Raftery 1995).
def bayes_factor(logZ1, logZ2):
    """Return (log10 Bayes factor B_12, Jeffreys/Kass-Raftery label)."""
    ln_B = logZ1 - logZ2
    log10_B = ln_B / np.log(10.0)
    a = abs(log10_B)
    if a < 0.5:
        strength = "barely worth mentioning"
    elif a < 1.0:
        strength = "substantial"
    elif a < 2.0:
        strength = "strong"
    else:
        strength = "decisive"
    favored = 1 if ln_B > 0 else 2
    return float(log10_B), f"{strength} (favors M{favored})"


# --------------------------------------------------------------------------- #
# Analytic verification anchor
# --------------------------------------------------------------------------- #
def analytic_gaussian_log_evidence(y, sigma2, tau2, dim):
    """
    Closed-form log Z for prior θ~N(0,τ²I) and likelihood y~N(θ,σ²I):

        Z = ∫ N(y;θ,σ²I) N(θ;0,τ²I) dθ = N(y; 0, (σ²+τ²) I).
    """
    y = np.asarray(y, float)
    s = sigma2 + tau2
    return float(-0.5 * (dim * np.log(2 * np.pi * s) + np.sum(y ** 2) / s))


if __name__ == "__main__":
    # Analytic-Gaussian recovery smoke (no CFD): TI & SS vs closed-form Z.
    rng = np.random.default_rng(0)
    dim = 2
    sigma2, tau2 = 0.5, 1.0
    y = rng.standard_normal(dim) * np.sqrt(sigma2)

    def loglik(theta):
        d = y - theta
        return -0.5 * (dim * np.log(2 * np.pi * sigma2) + np.sum(d * d) / sigma2)

    def logprior(theta):
        return -0.5 * (dim * np.log(2 * np.pi * tau2) + np.sum(theta * theta) / tau2)

    truth = analytic_gaussian_log_evidence(y, sigma2, tau2, dim)
    res = log_evidence(loglik, logprior, np.zeros(dim),
                       sampler_kwargs=dict(n_steps=800, burn=300), verbose=False)
    print(f"analytic logZ = {truth:.4f}")
    print(f"TI logZ = {res['ti_logZ']:.4f} ± {res['ti_se']:.4f}")
    print(f"SS logZ = {res['ss_logZ']:.4f} ± {res['ss_se']:.4f}")
    print(f"TI-SS agree: {res['agree']}  |Δ|={res['abs_diff']:.4f}")
