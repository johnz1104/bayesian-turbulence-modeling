"""
PHASE 4 — Surrogate validation diagnostics.

Provides train/validation splitting, per-output RMSE, and predictive-variance
calibration metrics for both ``GPSurrogate`` (scalar log-lik) and
``MultiOutputSurrogate`` (vector η) flavors.

Public API
----------
train_val_split(X, Y, val_frac, rng_seed)
    Reproducible random split.

scalar_diagnostics(gp, X_train, y_train, X_val, y_val)
    Per-fold RMSE, R^2, mean log predictive variance, σ-coverage at 1σ/2σ/3σ.

multi_output_diagnostics(mo, X_train, Y_train, X_val, Y_val)
    Per-output RMSE, R^2, σ-coverage at 1σ/2σ/3σ; one-line summary.

plot_predicted_vs_true(diag, save_path)
    Saves a per-output predicted-vs-true scatter plot (matplotlib).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_REPO_ROOT  = Path(__file__).resolve().parent.parent
_PYTHON_DIR = _REPO_ROOT / "python"
sys.path.insert(0, str(_PYTHON_DIR))


def train_val_split(X: np.ndarray, Y: np.ndarray, *,
                    val_frac: float = 0.2, rng_seed: int = 0,
                    min_val: int = 1) -> tuple[np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray]:
    """Reproducible random train/validation split.

    Parameters
    ----------
    X : (n, d) array
    Y : (n, …) array
    val_frac : float in (0, 1)
        Fraction of points to hold out.
    min_val : int
        At least this many validation points (defaults to 1).
    """
    X = np.asarray(X); Y = np.asarray(Y)
    n = len(X)
    if n < 2:
        raise ValueError("train_val_split needs n >= 2")
    n_val = max(min_val, int(round(val_frac * n)))
    n_val = min(n_val, n - 1)  # keep at least 1 training sample

    rng = np.random.default_rng(rng_seed)
    idx = rng.permutation(n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx]


def _coverage(residual: np.ndarray, sigma: np.ndarray, k: float) -> float:
    """Fraction of residuals within ±k σ."""
    sigma_safe = np.maximum(sigma, 1e-30)
    z = np.abs(residual) / sigma_safe
    return float(np.mean(z <= k))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, or NaN if y_true is constant."""
    var = float(np.var(y_true, ddof=0))
    if var < 1e-30:
        return float("nan")
    sse = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - sse / (var * len(y_true))


def scalar_diagnostics(gp, X_val, y_val) -> dict:
    """Validation diagnostics for a scalar surrogate (e.g. ``GPSurrogate``)."""
    mu, var = gp.predict_batch(X_val)
    sigma   = np.sqrt(np.maximum(var, 0.0))
    res     = y_val - mu
    return {
        "n_val":          int(len(X_val)),
        "rmse":           float(np.sqrt(np.mean(res ** 2))),
        "mae":            float(np.mean(np.abs(res))),
        "r2":             _r2(y_val, mu),
        "mean_log_var":   float(np.mean(np.log(np.maximum(var, 1e-300)))),
        "coverage_1sigma": _coverage(res, sigma, 1.0),
        "coverage_2sigma": _coverage(res, sigma, 2.0),
        "coverage_3sigma": _coverage(res, sigma, 3.0),
    }


def multi_output_diagnostics(mo, X_val, Y_val) -> dict:
    """Per-output diagnostics for ``MultiOutputSurrogate``.

    Returns
    -------
    dict with keys:
        ``per_output``: list[dict], one per output channel
        ``aggregate``: dict with mean RMSE, mean R^2, mean coverage
    """
    means, vars_ = mo.predict_batch(X_val)   # shapes (n, n_out)
    sigmas       = np.sqrt(np.maximum(vars_, 0.0))
    res          = Y_val - means

    per_output = []
    for j in range(mo.n_outputs):
        rj = res[:, j]
        sj = sigmas[:, j]
        per_output.append({
            "output":          j,
            "rmse":            float(np.sqrt(np.mean(rj ** 2))),
            "mae":             float(np.mean(np.abs(rj))),
            "r2":              _r2(Y_val[:, j], means[:, j]),
            "mean_sigma":      float(np.mean(sj)),
            "coverage_1sigma": _coverage(rj, sj, 1.0),
            "coverage_2sigma": _coverage(rj, sj, 2.0),
            "coverage_3sigma": _coverage(rj, sj, 3.0),
        })

    return {
        "per_output": per_output,
        "aggregate": {
            "n_val":            int(len(X_val)),
            "mean_rmse":        float(np.mean([p["rmse"]            for p in per_output])),
            "mean_r2":          float(np.nanmean([p["r2"]            for p in per_output])),
            "mean_cov_1sigma":  float(np.mean([p["coverage_1sigma"] for p in per_output])),
            "mean_cov_2sigma":  float(np.mean([p["coverage_2sigma"] for p in per_output])),
            "mean_cov_3sigma":  float(np.mean([p["coverage_3sigma"] for p in per_output])),
        },
    }


def plot_predicted_vs_true(mo, X_val, Y_val, save_path: Path):
    """Per-output scatter; save_path is a PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    means, vars_ = mo.predict_batch(X_val)
    sigmas       = np.sqrt(np.maximum(vars_, 0.0))
    n_out        = mo.n_outputs
    cols         = min(3, n_out)
    rows         = (n_out + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows),
                              squeeze=False)
    for j in range(n_out):
        ax = axes[j // cols][j % cols]
        ax.errorbar(Y_val[:, j], means[:, j], yerr=2.0 * sigmas[:, j],
                    fmt="o", ms=4, alpha=0.7)
        lo = min(np.min(Y_val[:, j]), np.min(means[:, j]))
        hi = max(np.max(Y_val[:, j]), np.max(means[:, j]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"true Y[{j}]"); ax.set_ylabel(f"pred Y[{j}] ± 2σ")
    for k in range(n_out, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True
