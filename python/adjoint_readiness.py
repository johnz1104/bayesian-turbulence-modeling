"""
POST-PHASE 10: Adjoint-readiness diagnostic.

Evaluates whether the current calibration problem justifies implementing
analytic adjoint gradients in the forward model, based on three thresholds
set in Theory/text/DECISION_RECORD.md §4:

  1. At least one parameter has normalised sensitivity score > SENS_THRESHOLD
     (OAT or Morris).
  2. Required ensemble size exceeds ENSEMBLE_THRESHOLD (surrogate convergence
     requires too many evaluations without gradients).
  3. Parameter space dimensionality exceeds DIM_THRESHOLD (high-d surrogates
     are data-hungry; adjoints become competitive).

All three criteria must be met for adjoint to be recommended.

Public API
----------
AdjointReadinessDiagnostic(...)  — configurable diagnostic object
run_diagnostic(...)              — convenience function
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# Default thresholds (from DECISION_RECORD.md §4)
DEFAULT_SENS_THRESHOLD     = 0.5   # normalised importance > 0.5
DEFAULT_ENSEMBLE_THRESHOLD = 500   # ensemble larger than this → adjoints help
DEFAULT_DIM_THRESHOLD      = 5     # d_theta > 5 → curse of dimensionality kicks in


@dataclass
class ReadinessResult:
    """Result of the adjoint-readiness diagnostic."""
    adjoint_warranted: bool
    criteria: Dict[str, bool]          # per-criterion pass/fail
    evidence: Dict[str, float]         # measured values
    thresholds: Dict[str, float]       # configured thresholds
    ranking: List[str]                 # parameters sorted by importance
    recommendation: str                # human-readable conclusion

    def __str__(self) -> str:
        lines = [
            "=== Adjoint-Readiness Diagnostic ===",
            f"  Recommendation: {self.recommendation}",
            "",
            "  Criteria:",
        ]
        for name, passed in self.criteria.items():
            val = self.evidence.get(name, float("nan"))
            thr = self.thresholds.get(name, float("nan"))
            tag = "PASS" if passed else "FAIL"
            lines.append(f"    [{tag}] {name}: {val:.3g} {'>' if passed else '<='} {thr:.3g}")
        lines += [
            "",
            f"  Parameter ranking (sensitivity): {' > '.join(self.ranking)}",
            f"  Adjoint warranted: {self.adjoint_warranted}",
        ]
        return "\n".join(lines)


class AdjointReadinessDiagnostic:
    """
    Diagnose whether adjoint gradients are justified for the current problem.

    Parameters
    ----------
    sensitivity_report : dict
        Output of ``SensitivityAnalyser.report()``.
    n_ensemble : int
        Ensemble size currently used (or planned).
    d_theta : int
        Number of calibration parameters.
    sens_threshold : float
        Min normalised importance to flag a parameter as adjoint-relevant.
    ensemble_threshold : int
        Ensemble size above which adjoints become competitive.
    dim_threshold : int
        Dimensionality above which surrogate coverage requires too many evals.
    """

    def __init__(
        self,
        sensitivity_report: Dict,
        n_ensemble: int,
        d_theta: int,
        sens_threshold: float = DEFAULT_SENS_THRESHOLD,
        ensemble_threshold: int = DEFAULT_ENSEMBLE_THRESHOLD,
        dim_threshold: int = DEFAULT_DIM_THRESHOLD,
    ):
        self.sens_report        = sensitivity_report
        self.n_ensemble         = n_ensemble
        self.d_theta            = d_theta
        self.sens_threshold     = sens_threshold
        self.ensemble_threshold = ensemble_threshold
        self.dim_threshold      = dim_threshold

    def run(self) -> ReadinessResult:
        """Evaluate all three criteria and return a ReadinessResult."""
        # --- Criterion 1: sensitivity -----------------------------------
        # Use whichever sources are available in the report
        scores: Dict[str, float] = {}
        for src in ("oat_importance", "morris_importance"):
            if src in self.sens_report:
                for name, val in self.sens_report[src].items():
                    scores[name] = max(scores.get(name, 0.0), float(val))

        max_sens = max(scores.values()) if scores else 0.0
        ranking  = sorted(scores, key=lambda n: -scores[n])
        crit_sens = max_sens > self.sens_threshold

        # --- Criterion 2: ensemble size ---------------------------------
        crit_ensemble = self.n_ensemble > self.ensemble_threshold

        # --- Criterion 3: dimensionality --------------------------------
        crit_dim = self.d_theta > self.dim_threshold

        adjoint = crit_sens and crit_ensemble and crit_dim

        criteria = {
            "max_sensitivity":  crit_sens,
            "n_ensemble":       crit_ensemble,
            "d_theta":          crit_dim,
        }
        evidence = {
            "max_sensitivity":  max_sens,
            "n_ensemble":       float(self.n_ensemble),
            "d_theta":          float(self.d_theta),
        }
        thresholds = {
            "max_sensitivity":  self.sens_threshold,
            "n_ensemble":       float(self.ensemble_threshold),
            "d_theta":          float(self.dim_threshold),
        }

        if adjoint:
            rec = (
                "Adjoint gradients RECOMMENDED. "
                "High sensitivity, large ensemble, and high dimensionality "
                "all suggest adjoints will reduce calibration cost."
            )
        else:
            failing = [k for k, v in criteria.items() if not v]
            rec = (
                "Adjoint gradients NOT YET WARRANTED. "
                f"Failing criteria: {', '.join(failing)}. "
                "Continue with GP surrogate approach."
            )

        return ReadinessResult(
            adjoint_warranted=adjoint,
            criteria=criteria,
            evidence=evidence,
            thresholds=thresholds,
            ranking=ranking,
            recommendation=rec,
        )


def run_diagnostic(
    sensitivity_report: Dict,
    n_ensemble: int,
    d_theta: int,
    sens_threshold: float = DEFAULT_SENS_THRESHOLD,
    ensemble_threshold: int = DEFAULT_ENSEMBLE_THRESHOLD,
    dim_threshold: int = DEFAULT_DIM_THRESHOLD,
    verbose: bool = True,
) -> ReadinessResult:
    """
    Convenience wrapper: build diagnostic, run it, optionally print result.

    Parameters
    ----------
    sensitivity_report : dict
        From ``SensitivityAnalyser.report()``.
    n_ensemble : int
    d_theta : int
    ... (threshold kwargs)
    verbose : bool
        If True, print the result table.

    Returns
    -------
    ReadinessResult
    """
    diag = AdjointReadinessDiagnostic(
        sensitivity_report=sensitivity_report,
        n_ensemble=n_ensemble,
        d_theta=d_theta,
        sens_threshold=sens_threshold,
        ensemble_threshold=ensemble_threshold,
        dim_threshold=dim_threshold,
    )
    result = diag.run()
    if verbose:
        print(result)
    return result
