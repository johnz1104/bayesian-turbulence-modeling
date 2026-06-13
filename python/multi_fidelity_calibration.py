"""
POST-PHASE 6: Multi-fidelity Bayesian calibration study.

Provides tools to run KOH calibration at multiple fidelity levels (defined
by the number/type of observations and forward-model resolution) and compare
how the posterior tightens as fidelity increases.

The "fidelity" notion here is purely data-driven: each level is characterised
by a different ObservationSet + ForwardModelBase pair.  Increasing fidelity
typically means more observation stations, more observable types, or a
higher-resolution solver.

Typical usage::

    from multi_fidelity_calibration import FidelityLevel, MultiFidelityStudy
    from bayesian_inference import Prior

    study = MultiFidelityStudy(
        prior=Prior([0.31, 0.09], [0.05, 0.015], [0.20, 0.05], [0.50, 0.15]),
        param_names=["a1", "betaStar"],
    )
    study.add_level("lo", obs_lo, fm_lo)
    study.add_level("hi", obs_hi, fm_hi)
    study.run_all(n_ensemble=40, n_steps=300)
    report = study.comparison_table()
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from bayesian_inference import BayesianInferenceKOH, Prior
from forward_model_interface import ForwardModelBase
from observation_schema import ObservationSet, koh_from_observation_set


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FidelityLevel:
    """One calibration problem at a given fidelity."""
    label: str
    obs_set: ObservationSet
    forward_model: ForwardModelBase
    koh_mode: str = "diagonal"
    samples: Optional[np.ndarray] = field(default=None, repr=False)
    elapsed_s: float = 0.0

    @property
    def n_obs(self) -> int:
        return self.obs_set.n_obs

    @property
    def ran(self) -> bool:
        return self.samples is not None


@dataclass
class LevelSummary:
    """Posterior statistics for one fidelity level."""
    label: str
    n_obs: int
    param_stats: Dict[str, Dict]    # {name: {mean, std, ci_lo, ci_hi}}
    elapsed_s: float


# ---------------------------------------------------------------------------
# MultiFidelityStudy
# ---------------------------------------------------------------------------

class MultiFidelityStudy:
    """
    Run BayesianInferenceKOH at multiple fidelity levels and compare posteriors.

    Parameters
    ----------
    prior : Prior
        Common parameter prior (same bounds for all levels).
    param_names : list[str]
        Names of the calibration parameters.
    param_set : dict, optional
        Dict-style param_set for BayesianInferenceKOH (defaults to one built
        from ``prior`` and ``param_names``).
    log_sigma_delta_prior : (mean, std)
        Prior on KOH discrepancy amplitude.
    """

    def __init__(
        self,
        prior: Prior,
        param_names: List[str],
        param_set: Optional[Dict] = None,
        log_sigma_delta_prior: Tuple[float, float] = (-2.0, 2.0),
    ):
        self.prior = prior
        self.param_names = list(param_names)
        self.param_set = param_set or {
            "defaults": prior.means.tolist(),
            "lower":    prior.lower.tolist(),
            "upper":    prior.upper.tolist(),
            "names":    list(param_names),
        }
        self.log_sigma_delta_prior = log_sigma_delta_prior
        self._levels: List[FidelityLevel] = []

    def add_level(
        self,
        label: str,
        obs_set: ObservationSet,
        forward_model: ForwardModelBase,
        koh_mode: str = "diagonal",
    ) -> "MultiFidelityStudy":
        """Append a fidelity level (call in ascending fidelity order)."""
        self._levels.append(
            FidelityLevel(label=label, obs_set=obs_set,
                          forward_model=forward_model, koh_mode=koh_mode)
        )
        return self

    def run_all(
        self,
        n_ensemble: int = 60,
        n_steps: int = 500,
        burn_in: Optional[int] = None,
        rng_seed: Optional[int] = None,
        verbose: bool = True,
    ) -> "MultiFidelityStudy":
        """
        Run KOH calibration at every registered fidelity level.

        Parameters
        ----------
        n_ensemble : int
            Ensemble size for surrogate training.
        n_steps : int
            MCMC steps per level.
        burn_in : int, optional
            MCMC burn-in steps (default: n_steps // 5).
        rng_seed : int, optional
            Base random seed (each level gets seed + level_index).
        verbose : bool
        """
        burn_in = burn_in or max(50, n_steps // 5)

        for i, lvl in enumerate(self._levels):
            seed_i = None if rng_seed is None else rng_seed + i
            if verbose:
                print(f"\n--- Level '{lvl.label}'  "
                      f"n_obs={lvl.n_obs}  mode={lvl.koh_mode} ---")

            koh = koh_from_observation_set(lvl.obs_set, mode=lvl.koh_mode)
            koh_bi = BayesianInferenceKOH(
                forward_model=lvl.forward_model,
                param_set=self.param_set,
                koh_likelihood=koh,
                theta_prior=self.prior,
                log_sigma_delta_prior=self.log_sigma_delta_prior,
            )

            t0 = time.time()
            koh_bi.run_ensemble(n_samples=n_ensemble, verbose=False)
            koh_bi.train_surrogate(verbose=False)
            koh_bi.run_mcmc(n_steps=n_steps, burn_in=burn_in,
                            verbose=verbose, rng_seed=seed_i)
            elapsed = time.time() - t0

            lvl.samples  = koh_bi.samples
            lvl.elapsed_s = elapsed

        return self

    # ---- Results ----------------------------------------------------------

    def level_summary(self, lvl: FidelityLevel) -> LevelSummary:
        """Compute posterior statistics for one level."""
        assert lvl.ran, f"Level '{lvl.label}' has not been run yet."
        stats = {}
        for i, name in enumerate(self.param_names):
            s = lvl.samples[:, i]
            stats[name] = {
                "mean":   float(np.mean(s)),
                "std":    float(np.std(s)),
                "ci_lo":  float(np.percentile(s, 2.5)),
                "ci_hi":  float(np.percentile(s, 97.5)),
                "ci_width": float(np.percentile(s, 97.5) - np.percentile(s, 2.5)),
            }
        return LevelSummary(
            label=lvl.label,
            n_obs=lvl.n_obs,
            param_stats=stats,
            elapsed_s=lvl.elapsed_s,
        )

    def comparison_table(self) -> Dict:
        """
        Return a structured dict comparing all levels.

        Keys: "param_names", "levels" (list of LevelSummary dicts),
              "convergence" (per-param CI-width reduction ratio vs first level).
        """
        summaries = [self.level_summary(lvl) for lvl in self._levels]

        levels_out = []
        for s in summaries:
            entry = {"label": s.label, "n_obs": s.n_obs,
                     "elapsed_s": s.elapsed_s}
            for name in self.param_names:
                entry[name] = s.param_stats[name]
            levels_out.append(entry)

        # CI-width reduction ratio (vs level 0)
        convergence = {}
        if len(summaries) > 1:
            baseline = summaries[0].param_stats
            for name in self.param_names:
                w0 = baseline[name]["ci_width"]
                convergence[name] = [
                    float(s.param_stats[name]["ci_width"] / max(w0, 1e-10))
                    for s in summaries
                ]

        return {
            "param_names": self.param_names,
            "levels": levels_out,
            "convergence": convergence,
        }

    def print_comparison(self, truth: Optional[Dict] = None) -> None:
        """Print a formatted per-parameter, per-level comparison table."""
        table = self.comparison_table()
        levels = table["levels"]

        for name in self.param_names:
            tv = truth[name] if (truth and name in truth) else None
            print(f"\n  {name}")
            print(f"  {'Level':>12s}  {'n_obs':>6s}  {'mean':>8s}  "
                  f"{'std':>7s}  {'95% CI width':>14s}")
            print("  " + "-" * 60)
            for lv in levels:
                s = lv[name]
                line = (f"  {lv['label']:>12s}  {lv['n_obs']:>6d}  "
                        f"{s['mean']:8.5f}  {s['std']:7.5f}  "
                        f"{s['ci_width']:14.5f}")
                if tv is not None:
                    err = abs(s["mean"] - tv) / max(s["std"], 1e-10)
                    line += f"  (truth={tv:.4f}  |err|={err:.1f}σ)"
                print(line)

    @property
    def levels(self) -> List[FidelityLevel]:
        return list(self._levels)

    @property
    def n_levels(self) -> int:
        return len(self._levels)
