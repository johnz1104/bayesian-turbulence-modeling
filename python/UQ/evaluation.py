"""Predictive-UQ evaluation harness: the scoring rules and calibration
diagnostics the program uses to earn its claims.

Everything here is distribution-agnostic and works from either predictive
interval endpoints or ensemble samples. Each routine is verified on synthetic
cases with a known answer (a calibrated predictor scores well, a miscalibrated
one does not). Proper scoring rules: CRPS (univariate) and the energy score
(multivariate) in their fair, ensemble-form estimators. Calibration: empirical
coverage, reliability curves, PIT histograms, and simulation-based calibration.
"""
import numpy as np
from scipy import stats


# ---- coverage and sharpness ------------------------------------------------

def empirical_coverage(y_true, lower, upper):
    """Fraction of observations inside the [lower, upper] predictive interval."""
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))))


def interval_sharpness(lower, upper):
    """Mean predictive-interval width (lower is sharper, at matched coverage)."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def coverage_from_samples(y_true, samples, level=0.9):
    """Coverage and mean width of central predictive intervals from ensembles.

    samples: (N, M).  Returns (coverage, sharpness) at the nominal level.
    """
    a = (1.0 - level) / 2.0
    lo = np.quantile(samples, a, axis=1)
    hi = np.quantile(samples, 1.0 - a, axis=1)
    return empirical_coverage(y_true, lo, hi), interval_sharpness(lo, hi)


# ---- proper scoring rules --------------------------------------------------

def crps_ensemble(y_true, samples):
    """Fair ensemble CRPS, averaged over observations.

    CRPS = E|X - y| - 0.5 E|X - X'|, computed with the O(M log M) sorted
    estimator for the second term. samples: (N, M); y_true: (N,).
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)
    N, M = samples.shape
    term1 = np.mean(np.abs(samples - y_true[:, None]), axis=1)
    xs = np.sort(samples, axis=1)
    weights = (2.0 * np.arange(1, M + 1) - M - 1)
    term2 = (xs * weights).sum(axis=1) / (M * M)
    return float(np.mean(term1 - term2))


def energy_score(y_true, samples):
    """Fair ensemble energy score (multivariate CRPS generalisation).

    ES = E||X - y|| - 0.5 E||X - X'||. samples: (N, M, d); y_true: (N, d).
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)
    N, M, d = samples.shape
    diff = samples - y_true[:, None, :]
    term1 = np.mean(np.linalg.norm(diff, axis=2), axis=1)
    # pairwise distances within the ensemble
    pd = samples[:, :, None, :] - samples[:, None, :, :]
    term2 = np.linalg.norm(pd, axis=3).mean(axis=(1, 2))
    return float(np.mean(term1 - 0.5 * term2))


# ---- calibration diagnostics ----------------------------------------------

def pit_values(y_true, samples):
    """Probability-integral-transform values: P(X <= y) under the ensemble.

    For a calibrated predictor these are Uniform(0, 1). samples: (N, M).
    """
    y_true = np.asarray(y_true)
    samples = np.asarray(samples)
    return np.mean(samples <= y_true[:, None], axis=1)


def pit_histogram(pit, bins=10):
    """Histogram of PIT values (counts, edges). Flat = calibrated."""
    return np.histogram(pit, bins=bins, range=(0.0, 1.0))


def pit_uniformity_pvalue(pit):
    """Kolmogorov-Smirnov p-value for PIT ~ Uniform(0,1). Large = calibrated."""
    return float(stats.kstest(np.asarray(pit), "uniform").pvalue)


def reliability_curve(y_true, samples, levels=None):
    """Empirical vs nominal central-interval coverage (the reliability diagram).

    Returns (nominal, empirical) arrays; a calibrated predictor lies on y = x.
    """
    if levels is None:
        levels = np.linspace(0.1, 0.9, 9)
    emp = np.array([coverage_from_samples(y_true, samples, level=lv)[0] for lv in levels])
    return np.asarray(levels), emp


def reliability_error(y_true, samples, levels=None):
    """Mean absolute deviation of empirical coverage from nominal (lower better)."""
    nominal, emp = reliability_curve(y_true, samples, levels)
    return float(np.mean(np.abs(emp - nominal)))


# ---- simulation-based calibration ------------------------------------------

def sbc_ranks(theta_true, posterior_samples):
    """SBC rank statistics for a neural / sampled posterior.

    For each trial i, the rank of the true parameter among that trial's L
    posterior draws. If inference is correct the ranks are Uniform on {0..L}.

    theta_true: (T,) or (T, p); posterior_samples: (T, L) or (T, L, p).
    Returns ranks of shape (T,) (1-D) or (T, p).
    """
    theta_true = np.asarray(theta_true)
    posterior_samples = np.asarray(posterior_samples)
    if theta_true.ndim == 1:
        return np.sum(posterior_samples < theta_true[:, None], axis=1)
    return np.sum(posterior_samples < theta_true[:, None, :], axis=1)


def sbc_uniformity_pvalue(ranks, n_draws):
    """Chi-square p-value that SBC ranks are uniform on {0..n_draws}."""
    ranks = np.asarray(ranks).ravel()
    counts, _ = np.histogram(ranks, bins=min(20, n_draws + 1),
                             range=(0, n_draws + 1))
    expected = np.full(len(counts), counts.sum() / len(counts))
    return float(stats.chisquare(counts, expected).pvalue)
