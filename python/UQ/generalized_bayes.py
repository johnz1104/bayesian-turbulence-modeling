"""Generalized (calibrated) Bayesian inference for misspecified models.

Standard Bayes contracts on the Kullback-Leibler-closest parameter at the
parametric rate even when the model is wrong, so its posterior grows confident
about a wrong answer as data accumulate (Kleijn and van der Vaart 2012). This is
the overconfidence the framework already exposed incompressibly, and it is severe
for a misspecified RANS closure. The standard repair is a power (Gibbs) posterior

    p_eta(theta | y) proportional to p(theta) * L(y | theta) ** eta,   0 < eta <= 1,

whose learning rate eta tempers the contraction to match the degree of
misspecification (Bissiri, Holmes and Walker 2016; Grunwald and van Ommen 2017).

This module provides the tempered log-posterior (to drive the existing emcee /
NUTS layer), an analytic conjugate-Gaussian power posterior for verification, and
two learning-rate calibrations: a moment-matching rule and a coverage-matching
search.
"""
import numpy as np
from scipy import stats


def tempered_log_posterior(log_likelihood, log_prior, eta):
    """Return a callable theta -> log p(theta) + eta * log L(y | theta).

    Drop-in for the repo's MCMC samplers: pass this in place of the standard
    log-posterior to sample the Gibbs / power posterior at learning rate eta.
    """
    def log_post(theta):
        return log_prior(theta) + eta * log_likelihood(theta)
    return log_post


def power_posterior_gaussian(y, sigma0, eta=1.0, prior_mean=0.0, prior_var=np.inf):
    """Analytic power posterior for theta in a Gaussian location model.

    Model assumes y_i ~ N(theta, sigma0^2). The likelihood raised to eta scales
    the data precision by eta, so

        post precision = 1/prior_var + n eta / sigma0^2
        post mean      = post_var (prior_mean/prior_var + n eta ybar / sigma0^2)

    Returns (posterior_mean, posterior_variance).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    ybar = y.mean()
    prior_prec = 0.0 if np.isinf(prior_var) else 1.0 / prior_var
    post_prec = prior_prec + n * eta / sigma0 ** 2
    post_var = 1.0 / post_prec
    prior_term = 0.0 if np.isinf(prior_var) else prior_mean / prior_var
    post_mean = post_var * (prior_term + n * eta * ybar / sigma0 ** 2)
    return post_mean, post_var


def credible_interval(mean, var, level=0.9):
    """Central Gaussian credible interval at the given level."""
    z = stats.norm.ppf(0.5 + level / 2.0)
    half = z * np.sqrt(var)
    return mean - half, mean + half


def calibrate_eta_moment(y, sigma0):
    """Moment-matching learning rate: temper so the model noise matches the data.

    The model assumes noise variance sigma0^2 but the empirical dispersion is
    s^2; eta = sigma0^2 / s^2 inflates the posterior to the spread the data
    actually shows. Clipped to (0, 1]. For a well-specified model (s^2 = sigma0^2)
    this returns 1, recovering standard Bayes.
    """
    y = np.asarray(y, dtype=float)
    s2 = np.var(y, ddof=1)
    eta = sigma0 ** 2 / max(s2, 1e-300)
    return float(np.clip(eta, 1e-6, 1.0))


def calibrate_eta_coverage(resampler, sigma0, level=0.9, etas=None, n_trials=200,
                           seed=0):
    """Pick the largest eta whose credible intervals reach nominal coverage.

    resampler(rng) -> a fresh data array y. For each candidate eta the posterior
    credible interval is built per dataset and its frequentist coverage of the
    data mean is estimated; the chosen eta is the largest (sharpest, least
    tempered) whose coverage is at least the nominal level. Choosing the largest
    such eta keeps the posterior as sharp as calibration allows.
    """
    if etas is None:
        etas = np.linspace(0.05, 1.0, 20)
    rng = np.random.default_rng(seed)
    datasets = [resampler(rng) for _ in range(n_trials)]
    truth = np.mean([d.mean() for d in datasets])

    best_eta = etas[0]
    for eta in sorted(etas):
        hits = 0
        for d in datasets:
            m, v = power_posterior_gaussian(d, sigma0, eta=eta)
            lo, hi = credible_interval(m, v, level)
            if lo <= truth <= hi:
                hits += 1
        cov = hits / len(datasets)
        if cov >= level:
            best_eta = eta
    return float(best_eta)


def credible_interval_coverage(datasets, truth, sigma0, eta, level=0.9):
    """Frequentist coverage of the (tempered) credible interval over datasets."""
    hits = 0
    for d in datasets:
        m, v = power_posterior_gaussian(d, sigma0, eta=eta)
        lo, hi = credible_interval(m, v, level)
        if lo <= truth <= hi:
            hits += 1
    return hits / len(datasets)
