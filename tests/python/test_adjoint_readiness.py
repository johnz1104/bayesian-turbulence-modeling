"""Tests for adjoint_readiness.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PYTHON_DIR = Path(__file__).resolve().parent.parent.parent / "python"
sys.path.insert(0, str(_PYTHON_DIR))

from adjoint_readiness import (
    AdjointReadinessDiagnostic,
    ReadinessResult,
    run_diagnostic,
    DEFAULT_SENS_THRESHOLD,
    DEFAULT_ENSEMBLE_THRESHOLD,
    DEFAULT_DIM_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helper: make a sensitivity report
# ---------------------------------------------------------------------------

def _make_sens_report(oat=None, morris=None):
    """Build a minimal sensitivity report dict."""
    report = {"param_names": ["a1", "betaStar"], "ranking": ["a1", "betaStar"]}
    if oat is not None:
        report["oat_importance"] = oat
    if morris is not None:
        report["morris_importance"] = morris
    return report


# ---------------------------------------------------------------------------
# ReadinessResult
# ---------------------------------------------------------------------------

class TestReadinessResult:
    def test_str_contains_recommendation(self):
        r = ReadinessResult(
            adjoint_warranted=False,
            criteria={"max_sensitivity": False, "n_ensemble": False, "d_theta": False},
            evidence={"max_sensitivity": 0.3, "n_ensemble": 40.0, "d_theta": 2.0},
            thresholds={"max_sensitivity": 0.5, "n_ensemble": 500.0, "d_theta": 5.0},
            ranking=["a1", "betaStar"],
            recommendation="Not warranted",
        )
        assert "Not warranted" in str(r)

    def test_adjoint_warranted_false_by_default(self):
        r = ReadinessResult(
            adjoint_warranted=False,
            criteria={}, evidence={}, thresholds={}, ranking=[], recommendation=""
        )
        assert not r.adjoint_warranted


# ---------------------------------------------------------------------------
# AdjointReadinessDiagnostic — criteria
# ---------------------------------------------------------------------------

class TestDiagnosticCriteria:
    def test_all_fail(self):
        rep = _make_sens_report(oat={"a1": 0.3, "betaStar": 0.1})
        diag = AdjointReadinessDiagnostic(rep, n_ensemble=100, d_theta=2)
        result = diag.run()
        assert not result.adjoint_warranted
        assert not result.criteria["max_sensitivity"]
        assert not result.criteria["n_ensemble"]
        assert not result.criteria["d_theta"]

    def test_all_pass(self):
        rep = _make_sens_report(oat={"a1": 0.9, "betaStar": 0.5})
        diag = AdjointReadinessDiagnostic(rep, n_ensemble=600, d_theta=8)
        result = diag.run()
        assert result.adjoint_warranted

    def test_sensitivity_pass_only(self):
        rep = _make_sens_report(oat={"a1": 1.0, "betaStar": 0.5})
        diag = AdjointReadinessDiagnostic(rep, n_ensemble=50, d_theta=2)
        result = diag.run()
        assert result.criteria["max_sensitivity"]
        assert not result.criteria["n_ensemble"]
        assert not result.criteria["d_theta"]
        assert not result.adjoint_warranted

    def test_ensemble_threshold_boundary(self):
        rep = _make_sens_report(oat={"a1": 0.1})
        diag = AdjointReadinessDiagnostic(rep, n_ensemble=501, d_theta=2,
                                          ensemble_threshold=500)
        result = diag.run()
        assert result.criteria["n_ensemble"]

    def test_dim_threshold_boundary(self):
        rep = _make_sens_report(oat={"a1": 0.1})
        diag = AdjointReadinessDiagnostic(rep, n_ensemble=100, d_theta=6,
                                          dim_threshold=5)
        result = diag.run()
        assert result.criteria["d_theta"]

    def test_uses_max_across_oat_and_morris(self):
        """Max of OAT and Morris importance should be used."""
        rep = _make_sens_report(
            oat={"a1": 0.4, "betaStar": 0.2},     # max OAT = 0.4  (fails)
            morris={"a1": 0.6, "betaStar": 0.3},   # max Morris = 0.6 (passes)
        )
        diag = AdjointReadinessDiagnostic(rep, n_ensemble=100, d_theta=2,
                                          sens_threshold=0.5)
        result = diag.run()
        assert result.criteria["max_sensitivity"]  # Morris pushed it over
        assert result.evidence["max_sensitivity"] == pytest.approx(0.6)

    def test_custom_thresholds(self):
        rep = _make_sens_report(oat={"a1": 0.8})
        diag = AdjointReadinessDiagnostic(
            rep, n_ensemble=50, d_theta=3,
            sens_threshold=0.7,
            ensemble_threshold=40,
            dim_threshold=2,
        )
        result = diag.run()
        assert result.adjoint_warranted

    def test_ranking_sorted_by_importance(self):
        rep = _make_sens_report(
            oat={"a1": 0.9, "betaStar": 0.2, "kappa": 0.5}
        )
        diag = AdjointReadinessDiagnostic(rep, n_ensemble=100, d_theta=3)
        result = diag.run()
        assert result.ranking[0] == "a1"
        assert result.ranking[-1] == "betaStar"

    def test_evidence_keys(self):
        rep = _make_sens_report(oat={"a1": 0.5})
        diag = AdjointReadinessDiagnostic(rep, n_ensemble=100, d_theta=2)
        result = diag.run()
        assert "max_sensitivity" in result.evidence
        assert "n_ensemble" in result.evidence
        assert "d_theta" in result.evidence

    def test_thresholds_match_config(self):
        rep = _make_sens_report(oat={"a1": 0.5})
        diag = AdjointReadinessDiagnostic(
            rep, n_ensemble=100, d_theta=2,
            sens_threshold=0.3, ensemble_threshold=200, dim_threshold=4,
        )
        result = diag.run()
        assert result.thresholds["max_sensitivity"] == pytest.approx(0.3)
        assert result.thresholds["n_ensemble"] == pytest.approx(200.0)
        assert result.thresholds["d_theta"] == pytest.approx(4.0)

    def test_empty_report_no_crash(self):
        """Empty sensitivity report should not crash."""
        rep = {"param_names": [], "ranking": []}
        diag = AdjointReadinessDiagnostic(rep, n_ensemble=100, d_theta=2)
        result = diag.run()
        assert not result.adjoint_warranted

    def test_recommendation_in_result(self):
        rep = _make_sens_report(oat={"a1": 0.5})
        diag = AdjointReadinessDiagnostic(rep, n_ensemble=100, d_theta=2)
        result = diag.run()
        assert len(result.recommendation) > 0


# ---------------------------------------------------------------------------
# run_diagnostic convenience function
# ---------------------------------------------------------------------------

class TestRunDiagnostic:
    def test_returns_readiness_result(self):
        rep = _make_sens_report(oat={"a1": 0.5})
        result = run_diagnostic(rep, n_ensemble=100, d_theta=2, verbose=False)
        assert isinstance(result, ReadinessResult)

    def test_not_warranted_for_2d_problem(self):
        """The current 2-param scramjet problem should not warrant adjoints."""
        rep = _make_sens_report(
            oat={"a1": 1.0, "betaStar": 0.03}
        )
        result = run_diagnostic(
            rep,
            n_ensemble=150,
            d_theta=2,
            verbose=False,
        )
        # Ensemble=150 < 500, d_theta=2 < 5 → should fail
        assert not result.adjoint_warranted

    def test_warranted_for_11d_large_ensemble(self):
        """The 11-parameter all11 problem with large ensemble should warrant."""
        rep = _make_sens_report(
            oat={f"p{i}": float(i) / 11 for i in range(1, 12)}
        )
        # normalise
        mx = max(rep["oat_importance"].values())
        rep["oat_importance"] = {k: v/mx for k, v in rep["oat_importance"].items()}
        result = run_diagnostic(
            rep,
            n_ensemble=600,
            d_theta=11,
            verbose=False,
        )
        assert result.adjoint_warranted


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

class TestDefaultThresholds:
    def test_default_sens_threshold(self):
        assert DEFAULT_SENS_THRESHOLD == 0.5

    def test_default_ensemble_threshold(self):
        assert DEFAULT_ENSEMBLE_THRESHOLD == 500

    def test_default_dim_threshold(self):
        assert DEFAULT_DIM_THRESHOLD == 5
