"""Fair (unbiased) ensemble scores and the randomized PIT.

Pins the audit fixes to UQ.evaluation:
  1. crps_ensemble / energy_score use the fair M(M-1) off-diagonal pair
     normalisation (hand-computed two-member case, and Monte-Carlo
     unbiasedness against the Gaussian closed form where the biased
     estimator is visibly off);
  2. the biased variants reproduce the old empirical-CDF values;
  3. energy_score reduces exactly to crps_ensemble in one dimension;
  4. pit_values is discrete (histogram diagnostic only) while
     pit_values_randomized is continuous-uniform under exchangeability, so
     the KS test is calibrated on it.
"""
import numpy as np
from scipy import stats

from UQ import evaluation as ev


def _gaussian_crps_closed(y, mu=0.0, sigma=1.0):
    # CRPS of N(mu, sigma^2) at observation y (Gneiting-Raftery closed form)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * stats.norm.cdf(z) - 1.0)
                    + 2.0 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def test_two_member_hand_computed_values():
    y = np.array([1.0])
    samples = np.array([[0.0, 2.0]])
    # fair: E|X-y| - sum_{i!=j}|xi-xj| / (2 M (M-1)) = 1 - 4/4 = 0
    assert abs(ev.crps_ensemble(y, samples) - 0.0) < 1e-14
    # biased (empirical-CDF): 1 - 4/(2*4) = 0.5
    assert abs(ev.crps_ensemble_biased(y, samples) - 0.5) < 1e-14
    # energy score in 1-D matches CRPS for both estimators
    s3 = samples[:, :, None]
    y2 = y[:, None]
    assert abs(ev.energy_score(y2, s3) - 0.0) < 1e-12
    assert abs(ev.energy_score_biased(y2, s3) - 0.5) < 1e-12


def test_single_member_reduces_to_absolute_error():
    y = np.array([1.0, -2.0])
    samples = np.array([[0.5], [1.0]])
    want = np.mean([0.5, 3.0])
    assert abs(ev.crps_ensemble(y, samples) - want) < 1e-14
    assert abs(ev.energy_score(y[:, None], samples[:, :, None]) - want) < 1e-12


def test_fair_crps_is_unbiased_biased_is_not():
    # K small ensembles (M = 3) from N(0,1) scored against a fixed y: the fair
    # estimator's mean matches the closed-form CRPS of the TRUE forecast; the
    # biased estimator overshoots by ~E|X-X'|/(2M) (about 0.19 here).
    rng = np.random.default_rng(0)
    K, M = 6000, 3
    y = 0.3
    samples = rng.normal(size=(K, M))
    y_arr = np.full(K, y)
    closed = _gaussian_crps_closed(y)
    fair = ev.crps_ensemble(y_arr, samples)
    biased = ev.crps_ensemble_biased(y_arr, samples)
    assert abs(fair - closed) < 0.02, (fair, closed)
    assert biased - closed > 0.12, (biased, closed)


def test_energy_equals_crps_in_one_dimension():
    rng = np.random.default_rng(1)
    y = rng.normal(size=40)
    samples = rng.normal(size=(40, 7))
    a = ev.crps_ensemble(y, samples)
    b = ev.energy_score(y[:, None], samples[:, :, None])
    assert abs(a - b) < 1e-10


def test_pit_discrete_vs_randomized_uniform():
    rng = np.random.default_rng(2)
    N, M = 4000, 8
    y = rng.normal(size=N)
    samples = rng.normal(size=(N, M))

    pit = ev.pit_values(y, samples)
    # discrete: only the M+1 possible rank values appear
    assert len(np.unique(pit)) <= M + 1

    pit_r = ev.pit_values_randomized(y, samples, seed=3)
    # continuous and calibrated: KS against Uniform(0,1) does not reject
    assert ev.pit_uniformity_pvalue(pit_r) > 0.01
    # while the discrete PIT KS p-value is not trustworthy (documented); no
    # assertion on it beyond being a probability
    p_disc = ev.pit_uniformity_pvalue(pit)
    assert 0.0 <= p_disc <= 1.0
