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
    """Fair ensemble CRPS (Ferro 2014), averaged over observations.

    CRPS_fair = E|X - y| - (1/(2 M (M-1))) sum_{i != j} |X_i - X_j|,
    computed with the O(M log M) sorted estimator for the pair term (the sorted
    weighted sum sum_i (2i - M - 1) x_(i) equals half the off-diagonal pair sum,
    so only the denominator distinguishes fair from biased). Unbiased for the
    infinite-ensemble CRPS, so ensembles of different sizes are comparable; the
    M^2 denominator (crps_ensemble_biased) systematically penalises small M.
    samples: (N, M); y_true: (N,). For M = 1 the pair term has no off-diagonal
    pairs and the score reduces to E|X - y|.

    Convention: use the fair estimator when the M members are an iid SAMPLE of
    an underlying predictive distribution (posterior ensembles, flow draws);
    use the M^2 plug-in when the M members ARE the forecast itself, a finite
    discrete distribution (e.g. a deterministic bounding family read as a
    uniform discrete forecast), for which the plug-in is the exact CRPS, not
    a biased estimate of anything.
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)
    N, M = samples.shape
    term1 = np.mean(np.abs(samples - y_true[:, None]), axis=1)
    if M < 2:
        return float(np.mean(term1))
    xs = np.sort(samples, axis=1)
    weights = (2.0 * np.arange(1, M + 1) - M - 1)
    term2 = (xs * weights).sum(axis=1) / (M * (M - 1))
    return float(np.mean(term1 - term2))


def crps_ensemble_biased(y_true, samples):
    """Ensemble CRPS with the M^2 pair denominator (diagonal included).

    The empirical-CDF plug-in. Two distinct legitimate uses: (1) it is the
    EXACT CRPS of the finite discrete forecast that places mass 1/M on each
    member, so it is the correct (not merely charitable) score when the
    members are themselves the forecast, as with a deterministic bounding
    family; (2) it reproduces previously committed score tables. As an
    estimate of an underlying continuous predictive it is biased against
    small ensembles; use crps_ensemble (fair) for that reading and for any
    cross-ensemble-size comparison of sampled predictives.
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

    ES_fair = E||X - y|| - (1/(2 M (M-1))) sum_{i != j} ||X_i - X_j||: the
    pair term averages off-diagonal pairs only (Ferro 2014), so ensembles of
    different sizes are comparable. samples: (N, M, d); y_true: (N, d).
    For M = 1 the pair term is empty and the score reduces to E||X - y||.
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)
    N, M, d = samples.shape
    diff = samples - y_true[:, None, :]
    term1 = np.mean(np.linalg.norm(diff, axis=2), axis=1)
    if M < 2:
        return float(np.mean(term1))
    # pairwise distances within the ensemble; diagonal is zero, so summing all
    # M^2 entries and dividing by the M(M-1) off-diagonal count is the fair mean
    pd = samples[:, :, None, :] - samples[:, None, :, :]
    term2 = np.linalg.norm(pd, axis=3).sum(axis=(1, 2)) / (M * (M - 1))
    return float(np.mean(term1 - 0.5 * term2))


def energy_score_biased(y_true, samples):
    """Energy score with the M^2 pair denominator (diagonal zeros included).

    Exactly as crps_ensemble_biased: the exact energy score of the finite
    discrete forecast on the M members (the right convention for a
    deterministic bounding family), and the reproduction path for committed
    tables; biased as an estimate of an underlying continuous predictive,
    where energy_score (fair) is the comparable choice across ensemble sizes.
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)
    N, M, d = samples.shape
    diff = samples - y_true[:, None, :]
    term1 = np.mean(np.linalg.norm(diff, axis=2), axis=1)
    pd = samples[:, :, None, :] - samples[:, None, :, :]
    term2 = np.linalg.norm(pd, axis=3).mean(axis=(1, 2))
    return float(np.mean(term1 - 0.5 * term2))


# ---- calibration diagnostics ----------------------------------------------

def pit_values(y_true, samples):
    """Empirical-CDF PIT values: P(X <= y) under the ensemble. samples: (N, M).

    NOTE: these are DISCRETE (multiples of 1/M), so they are a histogram
    diagnostic only; testing them against a continuous Uniform(0,1) (e.g. with
    a KS test) gives invalid p-values. For a formal uniformity test use
    pit_values_randomized, which is exactly Uniform(0,1) under calibration.
    """
    y_true = np.asarray(y_true)
    samples = np.asarray(samples)
    return np.mean(samples <= y_true[:, None], axis=1)


def pit_values_randomized(y_true, samples, seed=0):
    """Randomized PIT for a discrete M-member ensemble.

    u_i = (r_i + V_i (t_i + 1)) / (M + 1) with r_i = #{x_ij < y_i},
    t_i = #{x_ij == y_i} and V_i ~ Uniform(0,1): exactly Uniform(0,1) when the
    observation is exchangeable with the ensemble members, so KS-type tests are
    calibrated on these values. Fixed seed keeps reproductions deterministic.
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)
    N, M = samples.shape
    r_low = np.sum(samples < y_true[:, None], axis=1)
    ties = np.sum(samples == y_true[:, None], axis=1)
    v = np.random.default_rng(seed).uniform(size=N)
    return (r_low + v * (ties + 1.0)) / (M + 1.0)


def pit_histogram(pit, bins=10):
    """Histogram of PIT values (counts, edges). Flat = calibrated."""
    return np.histogram(pit, bins=bins, range=(0.0, 1.0))


def pit_uniformity_pvalue(pit):
    """Kolmogorov-Smirnov p-value for PIT ~ Uniform(0,1). Large = calibrated.

    Only calibrated for CONTINUOUS pit values: pass pit_values_randomized
    output. On the discrete pit_values output the KS null is wrong and the
    p-value is a heuristic at best.
    """
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
