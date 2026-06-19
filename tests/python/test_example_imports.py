"""
Smoke tests verifying that existing public Python entry points still work.

These tests guard that existing incompressible/KOH
examples still import and run.  They are intentionally lightweight: each
example is parsed for syntax, imported as a module (so its top-level code runs
but `main()` does NOT, since `__name__` will not be `__main__`), and the public
classes the examples rely on are checked for existence.

Tests do NOT execute the actual MCMC pipeline (those are smoke-tested at the
C++ level for compressible and tracked separately for incompressible/BFS as
slower regression jobs).
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT     = Path(__file__).resolve().parents[2]
EXAMPLES_DIR  = REPO_ROOT / "python" / "examples"

EXAMPLES = [
    "channel_flow_example.py",
    "bfs_example.py",
    "koh_example.py",
    # Infrastructure dry-run examples (no CFD required)
    "precomputed_high_mach_ensemble_demo.py",
    "high_mach_data_pathway_dry_run.py",
    "observable_ablation_demo.py",
    "external_cfd_queue_dry_run.py",
    "koh_stress_test.py",
]


def _import_module_from_path(path: Path):
    """Load a python file as a module by absolute path; return the module."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)  # runs top-level code only (not main())
    return mod


@pytest.fixture(scope="module")
def bayesian_inference():
    return importlib.import_module("bayesian_inference")


class TestBayesianInferencePublicSurface:
    """Lock the public surface used by examples and downstream notebooks."""

    @pytest.mark.parametrize("symbol", [
        "Prior",
        "GPSurrogate",
        "MultiOutputSurrogate",
        "BayesianInference",
        "BayesianInferenceKOH",
        "KOHLikelihood",
        "BFSSyntheticCalibration",
        "SyntheticCalibration",
        "DS1985",
        "latin_hypercube",
        "make_prior_from_param_set",
    ])
    def test_symbol_exposed(self, bayesian_inference, symbol):
        assert hasattr(bayesian_inference, symbol), (
            f"bayesian_inference is missing public symbol '{symbol}'"
        )

    def test_DS1985_keys(self, bayesian_inference):
        ds = bayesian_inference.DS1985
        for key in ("Re_h", "h_s", "H", "Lu", "Ld", "xr_h",
                    "Cf_stations", "U_profiles"):
            assert key in ds, f"DS1985 missing key {key!r}"
        assert ds["Re_h"] == pytest.approx(37600.0)


class TestVisualizationImports:
    def test_visualization_imports_without_pyvista(self):
        # visualization.py defers pyvista imports inside methods; importing the
        # module itself must work even if pyvista is not installed.
        vis = importlib.import_module("visualization")
        assert hasattr(vis, "FlowData")
        assert hasattr(vis, "FlowVisualizer")

    def test_inference_visualizer_imports(self):
        iv = importlib.import_module("inference_visualizer")
        assert hasattr(iv, "InferenceVisualizer")


class TestExampleScriptsParse:
    @pytest.mark.parametrize("name", EXAMPLES)
    def test_syntax_valid(self, name):
        path = EXAMPLES_DIR / name
        assert path.is_file(), f"missing example: {path}"
        ast.parse(path.read_text(), filename=str(path))


class TestExampleScriptsImport:
    """
    Importing each example file must succeed without invoking main().

    This catches:
      - Missing module-level dependencies
      - Path setup that breaks under conftest's PYTHONPATH
      - Symbols imported from bayesian_inference that have been renamed/removed
    """

    @pytest.fixture(autouse=True)
    def _purge_module_cache(self):
        # Each test gets a clean import slate so the module's top-level code
        # actually runs again.
        for name in [p.replace(".py", "") for p in EXAMPLES]:
            sys.modules.pop(name, None)
        yield

    @pytest.mark.parametrize("name", EXAMPLES)
    def test_imports_cleanly(self, name, rs):
        # `rs` fixture ensures rans_sst_py is importable; the example uses it.
        path = EXAMPLES_DIR / name
        mod = _import_module_from_path(path)
        # Every example exposes a `main` callable; verify and don't invoke it.
        assert callable(getattr(mod, "main", None)), (
            f"{name} should expose a main() callable"
        )
