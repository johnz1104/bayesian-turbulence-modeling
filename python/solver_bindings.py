"""
Lazy loader for the C++ extension module (rans_sst_py).

Extracted from bayesian_inference.py (2026-06-12 modularization).  Modules
using the C++ solver import `_rs` from here, so everything stays importable
without the built .so; the error is raised only when a method that actually
needs the C++ solver is called.
"""

_rans_sst_py = None


def _rs():
    global _rans_sst_py
    if _rans_sst_py is None:
        try:
            import rans_sst_py as _m
            _rans_sst_py = _m
        except ImportError:
            raise ImportError(
                "Cannot import rans_sst_py. Build the project first:\n"
                "  cd build && cmake --build .\n"
                "Then add the build directory to sys.path before importing:\n"
                "  import sys; sys.path.insert(0, 'build/')\n"
                "Or use bayesian_inference.py in standalone mode with dict param_sets."
            ) from None
    return _rans_sst_py
