"""Metrics interface (FROZEN, core-v1.0).

Two families are needed for the end-goal findings, which both claim better
statistics WITH quantified uncertainty:

  * a DETERMINISTIC error: does the mean prediction match the truth
    (NormalizedRMSE);
  * a PROBABILISTIC score: is the predicted distribution calibrated
    (GaussianNLL, a proper scoring rule that punishes both bias and
    over/under-confidence).

Out-of-distribution generalization is the same metric evaluated on a held-out
regime; ood_gap reports the in-to-out degradation that every accuracy claim must
carry alongside the simple attached-flow baseline (root CLAUDE.md working rules).
"""

from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """Deterministic metric: a scalar summarising predicted vs observed."""

    greater_is_better: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        ...

    def __call__(self, predicted, observed) -> float:
        return self.evaluate(predicted, observed)


class ProbabilisticMetric(ABC):
    """Probabilistic metric scoring a predictive distribution against truth."""

    greater_is_better: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(self, mean: np.ndarray, std: np.ndarray,
                 observed: np.ndarray) -> float:
        ...

    def __call__(self, mean, std, observed) -> float:
        return self.evaluate(mean, std, observed)


class NormalizedRMSE(Metric):
    """Root-mean-square error normalised by the RMS of the truth (or a scale).

        NRMSE = sqrt(mean((pred - obs)^2)) / scale,
        scale = rms(obs) by default (or a supplied positive constant).

    Zero when the prediction equals the truth; scale-free for cross-case
    comparison. Lower is better.
    """

    def __init__(self, scale: float = None):
        self._scale = scale

    @property
    def name(self) -> str:
        return "nrmse"

    def evaluate(self, predicted, observed) -> float:
        predicted = np.asarray(predicted, dtype=np.float64)
        observed = np.asarray(observed, dtype=np.float64)
        rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
        if self._scale is not None:
            scale = float(self._scale)
        else:
            scale = float(np.sqrt(np.mean(observed ** 2)))
        return rmse / scale if scale > 0.0 else rmse


class GaussianNLL(ProbabilisticMetric):
    """Mean negative log predictive density under a Gaussian forecast.

        NLL = mean_i [ 0.5 log(2 pi std_i^2) + 0.5 (obs_i - mean_i)^2 / std_i^2 ].

    A proper scoring rule: it rewards an accurate mean AND a calibrated std,
    penalising both over-confidence (std too small) and under-confidence (std too
    large). Lower is better. This is the metric the UQ claim is judged on.
    """

    @property
    def name(self) -> str:
        return "gaussian_nll"

    def evaluate(self, mean, std, observed) -> float:
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)
        observed = np.asarray(observed, dtype=np.float64)
        std = np.maximum(std, 1e-12)        # guard the log and the division
        nll = 0.5 * np.log(2.0 * np.pi * std ** 2) + 0.5 * ((observed - mean) / std) ** 2
        return float(np.mean(nll))


def ood_gap(score_in: float, score_out: float, greater_is_better: bool = False) -> float:
    """Out-of-distribution generalization gap.

    Positive means the held-out (out-of-distribution) regime is WORSE than the
    calibration regime. For a lower-is-better metric that is score_out - score_in;
    for a greater-is-better metric it is score_in - score_out.
    """
    return (score_in - score_out) if greater_is_better else (score_out - score_in)
