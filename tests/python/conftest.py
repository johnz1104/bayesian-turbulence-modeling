"""
pytest configuration for QBTM Python tests.

Responsibilities:
1. Put the build/ directory on sys.path so `import rans_sst_py` works without
   the user having to set PYTHONPATH manually.
2. Put python/ on sys.path so `import bayesian_inference` works.
3. Provide a session-scoped fixture that seeds numpy and Python's `random`
   module with a fixed seed, so all tests are deterministic.
4. Skip tests that require the C++ extension when it cannot be imported.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT  = Path(__file__).resolve().parents[2]
BUILD_DIR  = REPO_ROOT / "build"
PYTHON_DIR = REPO_ROOT / "python"


def _prepend_unique(p: Path) -> None:
    """Insert path at front of sys.path if not already present."""
    s = str(p)
    if s in sys.path:
        return
    sys.path.insert(0, s)


# Make the build .so and python sources importable regardless of cwd.
_prepend_unique(BUILD_DIR)
_prepend_unique(PYTHON_DIR)

# Make the repo root importable so the frozen research.shared interfaces resolve.
# Appended (not prepended) so it cannot shadow build/ or python/.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """
    Reset RNGs before every test so order-of-execution cannot leak state.

    Fixed seed 0 is the project default for tests; individual tests that need a
    different seed should reseed inside the test body.
    """
    np.random.seed(0)
    random.seed(0)
    yield


@pytest.fixture(scope="session")
def rs():
    """The compiled rans_sst_py module, or skip the test if unavailable."""
    try:
        import rans_sst_py as _rs
    except ImportError as exc:
        pytest.skip(
            "rans_sst_py extension not built; run cmake --build build first "
            f"({exc})"
        )
    return _rs


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def build_dir() -> Path:
    return BUILD_DIR


def pytest_report_header(config):
    extras = []
    so = list(BUILD_DIR.glob("rans_sst_py*.so")) + list(BUILD_DIR.glob("rans_sst_py*.pyd"))
    extras.append(f"build dir: {BUILD_DIR}  (extension found: {bool(so)})")
    extras.append(f"python source: {PYTHON_DIR}")
    return extras
