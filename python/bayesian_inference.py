"""
RANS-SST Bayesian Inference — backward-compatibility facade.

The original god module was modularized on 2026-06-12.  The universal
inference library now lives in single-concern modules:

    solver_bindings.py   _rs() lazy loader for the C++ rans_sst_py module
    priors.py            Prior, make_prior_from_param_set, make_sampling_prior
    design.py            latin_hypercube
    gp_surrogate.py      GPSurrogate, MultiOutputSurrogate
    param_utils.py       _get_param_names
    koh_likelihood.py    KOH_MODES, KOHLikelihood
    calibration.py       BayesianInference
    koh_calibration.py   BayesianInferenceKOH
    bfs_reference.py     DS1985, _bfs_solver_settings

Research-specific harnesses moved to the research/ package:

    research/synthetic_calibration.py   BFSSyntheticCalibration, SyntheticCalibration

This module re-exports the full historical public surface so existing
imports (`from bayesian_inference import ...`) keep working unchanged.
New code should import from the specific modules directly.  The research
symbols are loaded lazily so the universal library works even when the
research/ package is absent.

As before, the rans_sst_py import is deferred — this module is importable
without the built .so; the error is raised only when a method that needs
the C++ solver is called.
"""

from solver_bindings import _rs
from priors import Prior, make_prior_from_param_set, make_sampling_prior
from design import latin_hypercube
from gp_surrogate import GPSurrogate, MultiOutputSurrogate
from param_utils import _get_param_names
from koh_likelihood import KOH_MODES, KOHLikelihood
from calibration import BayesianInference
from koh_calibration import BayesianInferenceKOH
from bfs_reference import DS1985, _bfs_solver_settings

__all__ = [
    "Prior", "make_prior_from_param_set", "make_sampling_prior",
    "latin_hypercube",
    "GPSurrogate", "MultiOutputSurrogate",
    "KOH_MODES", "KOHLikelihood",
    "BayesianInference", "BayesianInferenceKOH",
    "DS1985",
    "BFSSyntheticCalibration", "SyntheticCalibration",
]

# Research-specific harnesses: resolved lazily (PEP 562) so importing the
# universal facade never requires the research/ package.
_RESEARCH_SYMBOLS = ("BFSSyntheticCalibration", "SyntheticCalibration")


def __getattr__(name):
    if name in _RESEARCH_SYMBOLS:
        from research.synthetic_calibration import (
            BFSSyntheticCalibration, SyntheticCalibration,
        )
        globals()["BFSSyntheticCalibration"] = BFSSyntheticCalibration
        globals()["SyntheticCalibration"] = SyntheticCalibration
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
