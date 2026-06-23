"""Metrics interface (core-v1.0): a deterministic error and a probabilistic
(UQ) score, plus the out-of-distribution generalization gap."""

from .base import (
    GaussianNLL,
    Metric,
    NormalizedRMSE,
    ProbabilisticMetric,
    ood_gap,
)

__all__ = [
    "Metric",
    "ProbabilisticMetric",
    "NormalizedRMSE",
    "GaussianNLL",
    "ood_gap",
]
