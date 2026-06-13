"""
POST-PHASE 7: Sensitivity analysis before adjoints.

Provides variance-based (Sobol) and one-at-a-time (OAT) sensitivity analysis
for black-box forward models.  Used to rank SST parameters by their influence
on scramjet observables BEFORE committing to adjoint implementation.

Typical usage::

    from sensitivity_analysis import SensitivityAnalyser
    from forward_model_interface import ForwardModelBase

    sa = SensitivityAnalyser(
        forward_model=fm,
        bounds=(lower, upper),
        param_names=["a1", "betaStar"],
    )
    sa.run_oat(n_points=15)
    sa.run_morris(n_trajectories=20)
    report = sa.report()
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from forward_model_interface import ForwardModelBase


# ---------------------------------------------------------------------------
# OAT sensitivity
# ---------------------------------------------------------------------------

@dataclass
class OATResult:
    """One-at-a-time sensitivity for a single parameter and output."""
    param_name: str
    output_index: int
    theta_range: np.ndarray   # parameter values swept
    output_range: np.ndarray  # corresponding output values
    sensitivity: float        # Δoutput / Δparam (finite-difference slope)


def run_oat(
    forward_model: ForwardModelBase,
    nominal: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    param_names: List[str],
    n_points: int = 11,
    verbose: bool = False,
) -> List[OATResult]:
    """
    One-at-a-time sweep: vary one parameter across its range while holding all
    others at nominal.  Computes a finite-difference slope for each
    (parameter, output) pair.

    Parameters
    ----------
    forward_model : ForwardModelBase
    nominal : array-like, shape (d,)
        Reference parameter vector.
    bounds : (lower, upper), each shape (d,)
    param_names : list[str]
    n_points : int
        Number of sweep points per parameter.

    Returns
    -------
    list of OATResult (one per parameter × output combination, then collapsed
    into a flat list)
    """
    nominal = np.asarray(nominal, dtype=float)
    lower   = np.asarray(bounds[0], dtype=float)
    upper   = np.asarray(bounds[1], dtype=float)
    d = len(nominal)
    results = []

    for j in range(d):
        sweep = np.linspace(lower[j], upper[j], n_points)
        outs  = []
        for v in sweep:
            theta = nominal.copy()
            theta[j] = v
            r = forward_model.evaluate(theta.tolist())
            preds = np.array(r.predictions) if r.converged and r.predictions else None
            outs.append(preds)

        # Skip if any evaluation failed
        valid = [o for o in outs if o is not None]
        if not valid:
            continue
        n_out = len(valid[0])
        outs_arr = np.array([o if o is not None else np.full(n_out, np.nan)
                              for o in outs])   # (n_points, n_out)

        for k in range(n_out):
            y = outs_arr[:, k]
            finite = np.isfinite(y)
            if finite.sum() < 2:
                slope = 0.0
            else:
                slope = float(np.polyfit(sweep[finite], y[finite], 1)[0])
            results.append(OATResult(
                param_name=param_names[j],
                output_index=k,
                theta_range=sweep,
                output_range=y,
                sensitivity=slope,
            ))

        if verbose:
            print(f"  OAT: {param_names[j]} done")

    return results


# ---------------------------------------------------------------------------
# Morris screening (elementary effects)
# ---------------------------------------------------------------------------

@dataclass
class MorrisResult:
    """Morris elementary effects for one parameter."""
    param_name: str
    mu_star: float    # mean |elementary effect| over outputs (averaged)
    sigma: float      # std of elementary effect over trajectories
    mu_star_per_output: np.ndarray   # shape (n_out,)


def run_morris(
    forward_model: ForwardModelBase,
    bounds: Tuple[np.ndarray, np.ndarray],
    param_names: List[str],
    n_trajectories: int = 10,
    n_levels: int = 4,
    rng_seed: Optional[int] = None,
    verbose: bool = False,
) -> List[MorrisResult]:
    """
    Morris (1991) elementary effects screening.

    Generates ``n_trajectories`` trajectories in parameter space, each a
    sequence of one-parameter-at-a-time perturbations.  The mean absolute
    elementary effect μ* ranks parameters by global importance.

    Parameters
    ----------
    forward_model : ForwardModelBase
    bounds : (lower, upper)
    param_names : list[str]
    n_trajectories : int
    n_levels : int
        Grid levels for the Morris grid (must be even).
    rng_seed : int, optional

    Returns
    -------
    list of MorrisResult, one per parameter, sorted by descending μ*
    """
    rng   = np.random.default_rng(rng_seed)
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    d     = len(lower)
    delta = (n_levels / 2) / (n_levels - 1)  # Morris Δ

    all_ee: Dict[int, List[np.ndarray]] = {j: [] for j in range(d)}  # param → list of EE vectors

    for t in range(n_trajectories):
        # Starting point on the Morris grid
        x0 = lower + rng.integers(0, n_levels // 2, size=d) * (upper - lower) / (n_levels - 1)

        perm = rng.permutation(d)
        theta = x0.copy()
        r0 = forward_model.evaluate(theta.tolist())
        y0 = np.array(r0.predictions) if r0.converged and r0.predictions else None

        for j in perm:
            theta_new = theta.copy()
            theta_new[j] = np.clip(theta[j] + delta * (upper[j] - lower[j]),
                                   lower[j], upper[j])
            r1 = forward_model.evaluate(theta_new.tolist())
            y1 = np.array(r1.predictions) if r1.converged and r1.predictions else None

            if y0 is not None and y1 is not None:
                ee = (y1 - y0) / (delta * (upper[j] - lower[j]))
                all_ee[j].append(ee)

            theta = theta_new
            y0    = y1

        if verbose and (t + 1) % max(1, n_trajectories // 5) == 0:
            print(f"  Morris: trajectory {t+1}/{n_trajectories}")

    results = []
    for j in range(d):
        ees = all_ee[j]
        if not ees:
            results.append(MorrisResult(param_names[j], 0.0, 0.0,
                                        np.zeros(1)))
            continue
        ees_arr = np.array(ees)              # (n_traj, n_out)
        mu_star_per_out = np.mean(np.abs(ees_arr), axis=0)
        sigma = float(np.std(ees_arr.ravel()))
        results.append(MorrisResult(
            param_name=param_names[j],
            mu_star=float(np.mean(mu_star_per_out)),
            sigma=sigma,
            mu_star_per_output=mu_star_per_out,
        ))

    results.sort(key=lambda r: -r.mu_star)
    return results


# ---------------------------------------------------------------------------
# SensitivityAnalyser — orchestrator
# ---------------------------------------------------------------------------

class SensitivityAnalyser:
    """
    Orchestrates OAT and Morris sensitivity analyses on a ForwardModelBase.

    Parameters
    ----------
    forward_model : ForwardModelBase
    bounds : (lower, upper)
    param_names : list[str]
    nominal : array-like, optional
        Nominal parameter vector for OAT sweeps.  Defaults to midpoint of bounds.
    """

    def __init__(
        self,
        forward_model: ForwardModelBase,
        bounds: Tuple[np.ndarray, np.ndarray],
        param_names: List[str],
        nominal: Optional[np.ndarray] = None,
    ):
        self.forward_model = forward_model
        self.lower  = np.asarray(bounds[0], dtype=float)
        self.upper  = np.asarray(bounds[1], dtype=float)
        self.param_names = list(param_names)
        self.nominal = (np.asarray(nominal, dtype=float)
                        if nominal is not None
                        else 0.5 * (self.lower + self.upper))

        self.oat_results: Optional[List[OATResult]] = None
        self.morris_results: Optional[List[MorrisResult]] = None

    def run_oat(self, n_points: int = 11, verbose: bool = False) -> "SensitivityAnalyser":
        self.oat_results = run_oat(
            self.forward_model, self.nominal,
            (self.lower, self.upper), self.param_names,
            n_points=n_points, verbose=verbose,
        )
        return self

    def run_morris(self, n_trajectories: int = 15, n_levels: int = 4,
                   rng_seed: Optional[int] = None,
                   verbose: bool = False) -> "SensitivityAnalyser":
        self.morris_results = run_morris(
            self.forward_model, (self.lower, self.upper),
            self.param_names,
            n_trajectories=n_trajectories, n_levels=n_levels,
            rng_seed=rng_seed, verbose=verbose,
        )
        return self

    def oat_importance(self) -> Dict[str, float]:
        """
        Return per-parameter OAT importance: mean |sensitivity| across outputs,
        normalised so the largest = 1.
        """
        assert self.oat_results is not None, "Run run_oat() first."
        raw: Dict[str, List[float]] = {n: [] for n in self.param_names}
        for r in self.oat_results:
            raw[r.param_name].append(abs(r.sensitivity))
        scores = {n: float(np.mean(v)) for n, v in raw.items() if v}
        max_s  = max(scores.values()) if scores else 1.0
        return {n: s / max(max_s, 1e-10) for n, s in scores.items()}

    def morris_importance(self) -> Dict[str, float]:
        """Return per-parameter Morris μ* normalised to [0, 1]."""
        assert self.morris_results is not None, "Run run_morris() first."
        scores = {r.param_name: r.mu_star for r in self.morris_results}
        max_s  = max(scores.values()) if scores else 1.0
        return {n: s / max(max_s, 1e-10) for n, s in scores.items()}

    def report(self) -> Dict:
        """
        Combined sensitivity report as a plain dict.

        Keys: ``"param_names"``, ``"oat_importance"`` (if run),
        ``"morris_importance"`` (if run), ``"ranking"``.
        """
        out: Dict = {"param_names": self.param_names}
        if self.oat_results is not None:
            out["oat_importance"] = self.oat_importance()
        if self.morris_results is not None:
            out["morris_importance"] = self.morris_importance()

        # Consensus ranking: average available scores
        combined: Dict[str, List[float]] = {n: [] for n in self.param_names}
        for src in ("oat_importance", "morris_importance"):
            if src in out:
                for name, val in out[src].items():
                    combined[name].append(val)
        ranking = sorted(
            combined.keys(),
            key=lambda n: -np.mean(combined[n]) if combined[n] else 0.0,
        )
        out["ranking"] = ranking
        return out

    def print_report(self) -> None:
        """Print a formatted sensitivity table."""
        rep = self.report()
        print(f"\n  {'Parameter':>14s}  {'OAT':>8s}  {'Morris μ*':>10s}  Rank")
        print("  " + "-" * 46)
        for i, name in enumerate(rep["ranking"], 1):
            oat_s   = rep.get("oat_importance",    {}).get(name, float("nan"))
            morr_s  = rep.get("morris_importance", {}).get(name, float("nan"))
            print(f"  {name:>14s}  {oat_s:8.4f}  {morr_s:10.4f}  #{i}")
