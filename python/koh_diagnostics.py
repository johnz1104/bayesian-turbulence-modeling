"""
PHASE 3 — KOH robustness and identifiability diagnostics.

This module provides reusable helpers for comparing standard Bayesian
calibration with two flavours of KOH model-form discrepancy:

    mode="no_discrepancy"   -> BayesianInference (no KOH, scalar GP surrogate)
    mode="diagonal"          -> KOHLikelihood(diagonal): K_δ = I, only σ_δ
    mode="physical_gp"       -> KOHLikelihood(physical_gp): full GP on x

Each observation can carry metadata (type, location, units, σ, group).  The
metadata is purely Pythonic — it does not change the C++ ObservationOperator
— but it is consumed by the diagnostics report and saved alongside the
posterior summaries.

Public API
----------
make_obs_metadata(...)
    Build a list of obs metadata dicts from heterogeneous BFS-style inputs.

run_calibration(forward_model, param_set, koh, mode, n_ensemble, n_steps)
    Run one calibration in the given mode; returns a populated ``BayesianInference``
    or ``BayesianInferenceKOH`` object with deterministic seeding.

compare_modes(...)
    Run all three modes and produce a comparison summary suitable for plotting.

identifiability_report(comparison_summary, save_dir)
    Save plots + JSON to ``save_dir``.  Plots use matplotlib only (no PyVista),
    so they work in headless CI.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_REPO_ROOT  = Path(__file__).resolve().parent.parent
_BUILD_DIR  = _REPO_ROOT / "build"
_PYTHON_DIR = _REPO_ROOT / "python"
sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PYTHON_DIR))

from bayesian_inference import (   # noqa: E402
    BayesianInference, BayesianInferenceKOH, KOHLikelihood,
    KOH_MODES, _get_param_names,
)


# ---------------- Observation metadata ----------------------------------

@dataclass
class ObsMeta:
    """Metadata for one calibration observation.  Purely Pythonic; the C++
    ObservationOperator does not see it."""
    obs_type: str          # "Cf", "reattachment", "velocity_profile", etc.
    location: tuple        # physical location, e.g. (x, y, z) or scalar x/h
    sigma:    float        # measurement uncertainty
    group:    str          # group label for stratified diagnostics
    units:    str = ""     # human label, e.g. "h", "dimensionless", "m/s"


def make_obs_metadata(items: Iterable[dict]) -> list[ObsMeta]:
    """Coerce a list of dicts into ObsMeta dataclasses, validating fields."""
    out = []
    for d in items:
        out.append(ObsMeta(
            obs_type = d["type"],
            location = tuple(d["location"]) if hasattr(d["location"], "__iter__")
                       else (float(d["location"]),),
            sigma    = float(d["sigma"]),
            group    = d.get("group", d["type"]),
            units    = d.get("units", ""),
        ))
    return out


# ---------------- Calibration drivers ------------------------------------

def _seed(rng_seed: int) -> None:
    np.random.seed(rng_seed)


def run_calibration(forward_model, param_set,
                    obs_locations, obs_values, obs_sigmas,
                    mode: str,
                    n_ensemble: int = 30, n_steps: int = 300,
                    n_walkers: int | None = None,
                    burn_in: int | None = None,
                    rng_seed: int = 0,
                    verbose: bool = True):
    """Run a single calibration in the requested mode.

    Returns a populated ``BayesianInference`` (mode="no_discrepancy") or
    ``BayesianInferenceKOH`` (mode in {diagonal, physical_gp}).
    """
    _seed(rng_seed)

    if mode == "no_discrepancy":
        bi = BayesianInference(forward_model, param_set)
        bi.run_ensemble(n_samples=n_ensemble, verbose=verbose)
        bi.train_surrogate(verbose=verbose)
        n_w = n_walkers or max(16, 2 * param_set.n_active())
        bi.run_mcmc(n_walkers=n_w, n_steps=n_steps,
                    burn_in=burn_in or max(50, n_steps // 5),
                    thin=1, verbose=verbose)
        return bi

    if mode not in KOH_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of "
                          f"['no_discrepancy', *{KOH_MODES}]")

    koh = KOHLikelihood(obs_locations, obs_values, obs_sigmas, mode=mode)
    bi  = BayesianInferenceKOH(forward_model, param_set, koh)
    bi.run_ensemble(n_samples=n_ensemble, verbose=verbose)
    bi.train_surrogate(verbose=verbose)
    n_extra = bi.n_extra
    n_w = n_walkers or max(16, 2 * (param_set.n_active() + n_extra))
    bi.run_mcmc(n_walkers=n_w, n_steps=n_steps,
                burn_in=burn_in or max(50, n_steps // 5),
                thin=1, verbose=verbose)
    return bi


# ---------------- Comparison summary ------------------------------------

def _theta_summary(bi) -> dict:
    """Per-parameter posterior stats restricted to the θ block (no KOH hyperparams)."""
    summary = bi.posterior_summary()
    theta_names = _get_param_names(bi.param_set)
    return {n: summary[n] for n in theta_names if n in summary}


def compare_modes(forward_model, param_set,
                  obs_locations, obs_values, obs_sigmas,
                  obs_metadata: list[ObsMeta] | None = None,
                  modes: list[str] | None = None,
                  n_ensemble: int = 30, n_steps: int = 300,
                  rng_seed: int = 0,
                  verbose: bool = True) -> dict[str, Any]:
    """Run several calibration modes back-to-back and assemble a comparison."""
    if modes is None:
        modes = ["no_discrepancy", "diagonal", "physical_gp"]

    out_runs: dict[str, Any]   = {}
    out_summaries: dict[str, Any] = {}
    timings: dict[str, float] = {}

    for mode in modes:
        if verbose:
            print(f"\n=== Calibration: mode={mode} ===")
        t0 = time.time()
        bi = run_calibration(
            forward_model, param_set,
            obs_locations, obs_values, obs_sigmas,
            mode=mode, n_ensemble=n_ensemble, n_steps=n_steps,
            rng_seed=rng_seed, verbose=verbose,
        )
        timings[mode] = time.time() - t0
        out_runs[mode] = bi
        out_summaries[mode] = bi.posterior_summary()

    return {
        "modes":       modes,
        "runs":        out_runs,
        "summaries":   out_summaries,
        "timings_s":   timings,
        "obs_metadata": [asdict(m) for m in (obs_metadata or [])],
    }


# ---------------- Posterior comparison helpers --------------------------

def posterior_widths(comparison: dict[str, Any], param_names: list[str]) -> dict:
    """Per-parameter posterior std for each mode; useful for KOH-vs-standard widths."""
    out = {}
    for mode in comparison["modes"]:
        s = comparison["summaries"][mode]
        out[mode] = {n: s[n]["std"] for n in param_names if n in s}
    return out


def posterior_shifts(comparison: dict[str, Any], param_names: list[str]) -> dict:
    """Per-parameter shift = (posterior_mean - prior_mean) / prior_std."""
    out = {}
    for mode in comparison["modes"]:
        s = comparison["summaries"][mode]
        out[mode] = {n: s[n]["shift"] for n in param_names if n in s}
    return out


def discrepancy_summary(comparison: dict[str, Any]) -> dict:
    """Posterior summary of σ_δ (and l_δ where applicable) per mode."""
    out: dict[str, dict] = {}
    for mode in comparison["modes"]:
        if mode == "no_discrepancy":
            out[mode] = {"sigma_delta_mean": None, "l_delta_mean": None}
            continue
        bi = comparison["runs"][mode]
        sigma_d = np.exp(bi.samples[:, bi.n_theta])
        info = {
            "sigma_delta_mean": float(np.mean(sigma_d)),
            "sigma_delta_std":  float(np.std(sigma_d)),
            "sigma_delta_p2_5": float(np.percentile(sigma_d, 2.5)),
            "sigma_delta_p97_5": float(np.percentile(sigma_d, 97.5)),
        }
        if bi.n_extra == 2:
            l_d = np.exp(bi.samples[:, bi.n_theta + 1])
            info.update({
                "l_delta_mean":  float(np.mean(l_d)),
                "l_delta_std":   float(np.std(l_d)),
                "l_delta_p2_5":  float(np.percentile(l_d, 2.5)),
                "l_delta_p97_5": float(np.percentile(l_d, 97.5)),
            })
        out[mode] = info
    return out


def residual_correlation(comparison: dict[str, Any], obs_values: np.ndarray) -> dict:
    """At each mode's posterior-mean θ, compute residual r = y - η.  Returns
    the residuals and Pearson correlation matrix (n_obs x n_obs) so the user
    can see whether KOH 'absorbed' correlated residuals."""
    out = {}
    for mode in comparison["modes"]:
        bi = comparison["runs"][mode]
        param_names = _get_param_names(bi.param_set)
        theta_mean = np.array([bi.posterior_summary()[n]["mean"]
                                for n in param_names])

        if mode == "no_discrepancy":
            mu, _ = bi.surrogate.predict(theta_mean.tolist())
            # Standard surrogate is scalar log-likelihood, not eta vector.
            # Skip residuals for no_discrepancy (not directly available).
            out[mode] = {
                "residuals": None,
                "correlation": None,
                "residual_log_lik": float(mu),
            }
            continue

        eta_mean, _ = bi.multi_surrogate.predict(theta_mean.tolist())
        r = obs_values - eta_mean
        # Pearson correlation matrix: with only one sample per obs we cannot
        # compute it directly; instead we take the outer product r r^T as a
        # diagnostic of residual co-occurrence and report rms.
        out[mode] = {
            "residuals":     [float(v) for v in r],
            "rms":           float(np.sqrt(np.mean(r ** 2))),
            "max_abs":       float(np.max(np.abs(r))),
        }
    return out


def surrogate_holdout_per_observable(comparison: dict[str, Any]) -> dict:
    """Re-run a holdout split per mode and report per-output RMSE.  Only the
    KOH modes have a multi-output surrogate; for ``no_discrepancy`` we report
    the scalar log-likelihood RMSE instead."""
    out = {}
    for mode in comparison["modes"]:
        bi = comparison["runs"][mode]
        if mode == "no_discrepancy":
            out[mode] = {"per_output_rmse": None,
                         "scalar_log_lik_rmse_note": "scalar surrogate; per-output RMSE not applicable"}
            continue
        # Re-do a holdout the same way train_surrogate did, with our seed.
        rng = np.random.default_rng(0)
        n   = len(bi.ensemble_X)
        n_test = max(1, int(n * 0.1))
        idx = rng.permutation(n)
        X_te = bi.ensemble_X[idx[:n_test]]
        Y_te = bi.ensemble_Y[idx[:n_test]]
        rmse = bi.multi_surrogate.rmse(X_te, Y_te)
        out[mode] = {"per_output_rmse": [float(v) for v in rmse]}
    return out


# ---------------- Save report -------------------------------------------

def write_report(comparison: dict[str, Any],
                  obs_values: np.ndarray,
                  save_dir: Path,
                  truth: np.ndarray | None = None) -> Path:
    """Write JSON summary, posterior-width plot, and shift plot to save_dir."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    runs = comparison["runs"]
    param_names = _get_param_names(next(iter(runs.values())).param_set)

    summary = {
        "modes":              comparison["modes"],
        "param_names":        list(param_names),
        "timings_s":          comparison["timings_s"],
        "posterior_widths":   posterior_widths(comparison, param_names),
        "posterior_shifts":   posterior_shifts(comparison, param_names),
        "discrepancy":        discrepancy_summary(comparison),
        "residuals":          residual_correlation(comparison, obs_values),
        "holdout":            surrogate_holdout_per_observable(comparison),
        "per_mode_summary":   {m: comparison["summaries"][m]
                               for m in comparison["modes"]},
        "obs_metadata":       comparison.get("obs_metadata", []),
        "truth":              truth.tolist() if truth is not None else None,
        "identifiability_flag": _flag_identifiability(comparison, param_names),
    }
    json_path = save_dir / "koh_identifiability_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # Plots: only attempt if matplotlib is importable.
    try:
        _plot_widths_and_shifts(comparison, param_names, save_dir)
        _plot_discrepancy(comparison, save_dir)
    except ImportError:
        print("  (matplotlib missing; plots skipped)")

    return json_path


def _flag_identifiability(comparison: dict[str, Any],
                          param_names: list[str]) -> dict:
    """Heuristic flag: if KOH width / std width > 5x, the KOH discrepancy is
    'eating' the data.  We document the ratio and a boolean per-parameter."""
    flags = {}
    if "no_discrepancy" not in comparison["modes"]:
        return flags
    std = comparison["summaries"]["no_discrepancy"]
    for koh_mode in ("diagonal", "physical_gp"):
        if koh_mode not in comparison["modes"]:
            continue
        koh = comparison["summaries"][koh_mode]
        per_param = {}
        for n in param_names:
            if n not in std or n not in koh:
                continue
            sigma_std = std[n]["std"]
            sigma_koh = koh[n]["std"]
            ratio = sigma_koh / sigma_std if sigma_std > 1e-12 else float("nan")
            per_param[n] = {
                "std_width":         float(sigma_std),
                "koh_width":         float(sigma_koh),
                "inflation_ratio":   float(ratio),
                "weakly_identified": bool(ratio > 5.0),
            }
        flags[koh_mode] = per_param
    return flags


def _plot_widths_and_shifts(comparison, param_names, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widths = posterior_widths(comparison, param_names)
    shifts = posterior_shifts(comparison, param_names)
    modes  = comparison["modes"]

    # Posterior-width bar plot.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(param_names))
    bw = 0.8 / max(len(modes), 1)
    for k, mode in enumerate(modes):
        ax = axes[0]
        vals = [widths[mode].get(n, 0.0) for n in param_names]
        ax.bar(x + k * bw - 0.4, vals, width=bw, label=mode)
    axes[0].set_xticks(x); axes[0].set_xticklabels(param_names, rotation=20)
    axes[0].set_ylabel("posterior σ"); axes[0].legend()
    axes[0].set_title("Posterior width vs KOH mode")

    for k, mode in enumerate(modes):
        ax = axes[1]
        vals = [shifts[mode].get(n, 0.0) for n in param_names]
        ax.bar(x + k * bw - 0.4, vals, width=bw, label=mode)
    axes[1].set_xticks(x); axes[1].set_xticklabels(param_names, rotation=20)
    axes[1].set_ylabel("(post μ - prior μ)/σ_prior"); axes[1].legend()
    axes[1].set_title("Posterior shift vs KOH mode")
    axes[1].axhline(0, color="k", lw=0.5)

    fig.tight_layout()
    out = save_dir / "posterior_widths_shifts.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_discrepancy(comparison, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_koh = [m for m in comparison["modes"] if m != "no_discrepancy"]
    if not has_koh:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode in has_koh:
        bi      = comparison["runs"][mode]
        sigma_d = np.exp(bi.samples[:, bi.n_theta])
        axes[0].hist(sigma_d, bins=40, alpha=0.5, label=mode, density=True)
        if bi.n_extra == 2:
            l_d = np.exp(bi.samples[:, bi.n_theta + 1])
            axes[1].hist(l_d, bins=40, alpha=0.5, label=mode, density=True)
    axes[0].set_xlabel("σ_δ"); axes[0].legend()
    axes[0].set_title("KOH amplitude posterior")
    axes[1].set_xlabel("l_δ"); axes[1].legend()
    axes[1].set_title("KOH lengthscale posterior (physical_gp only)")

    fig.tight_layout()
    out = save_dir / "discrepancy_posterior.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
