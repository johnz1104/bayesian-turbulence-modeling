"""Closure interfaces (core-v1.0).

The Closure abstract base is binding-free. The SSTClosure reference adapter pulls
in the compiled rans_sst_py binding, so import it explicitly from
``research.shared.closures.sst`` to keep this package importable without a build.
"""

from .base import Closure

__all__ = ["Closure"]
