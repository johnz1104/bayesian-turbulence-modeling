"""A-priori uncertainty-quantification scaffolding for the compressible UQ program.

Clean, synthetic-data-verified modules implementing the techniques scoped in
research.md and compressible_research.md, built up to the point where real DNS
plugs in:

  realizability   barycentric realizability projection (the realizability
                  constraint, kept separate from the invariant construction)
  discrepancy     Reynolds-stress and heat-flux model-form discrepancy plus the
                  Galilean-invariant feature set and integrity basis
  synthetic       synthetic fake-DNS fields with planted, recoverable discrepancy
  generative      conditional generative model-form model (normalizing flow),
                  samples projected into the realizable set
  conformal       split / cross-conformal prediction and conformalized quantile
                  regression
  generalized_bayes  power-likelihood / Gibbs posterior with learning-rate
                  calibration for misspecified models
  evaluation      coverage, sharpness, CRPS, energy score, reliability diagrams,
                  PIT histograms, and simulation-based calibration

Every module is verified on synthetic data with a known answer. None requires
real DNS. The generative model uses PyTorch when available and its test skips
gracefully otherwise; the rest are numpy/scipy only.
"""

from . import realizability, discrepancy, synthetic, conformal
from . import generalized_bayes, evaluation

__all__ = [
    "realizability",
    "discrepancy",
    "synthetic",
    "conformal",
    "generalized_bayes",
    "evaluation",
]
