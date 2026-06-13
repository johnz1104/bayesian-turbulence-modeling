"""
Parameter-set introspection helpers.

Extracted from bayesian_inference.py (2026-06-12 modularization).  Future
home of the ParameterSpace generalization (see architecture_review.md §2.4-C).
"""


def _get_param_names(param_set):
    """Extract parameter names from param_set (C++ or dict)."""
    if hasattr(param_set, 'active_names'):
        return param_set.active_names()
    if isinstance(param_set, dict) and 'names' in param_set:
        return param_set['names']
    ndim = param_set.n_active() if hasattr(param_set, 'n_active') else 2
    return [f"theta_{i}" for i in range(ndim)]
