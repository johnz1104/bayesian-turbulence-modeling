"""
Research-specific code, separated from the universal inference library
(2026-06-12 modularization of bayesian_inference.py).

Modules here may depend on the universal library (priors, gp_surrogate,
koh_likelihood, calibration, koh_calibration, …); the universal library must
never import from this package.
"""

from .synthetic_calibration import BFSSyntheticCalibration, SyntheticCalibration

__all__ = ["BFSSyntheticCalibration", "SyntheticCalibration"]
