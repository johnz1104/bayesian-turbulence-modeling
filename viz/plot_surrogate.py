"""
GP surrogate diagnostics on held-out ensemble points: predicted vs true
log-likelihood (with the GP's own predictive uncertainty as error bars) and the
residuals.  Source: viz/artifacts/<param_set>/surrogate_holdout.npz.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _common import load_holdout, have, set_style, save, plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("param_set", nargs="?", default="a1_betaStar")
    args = parser.parse_args()

    if not have(args.param_set):
        print(f"  no artifacts for {args.param_set}; run run_calibration.py first")
        return

    set_style()
    h = load_holdout(args.param_set)
    y_true = h["y_true"]
    y_pred = h["y_pred"]
    y_sd = np.sqrt(np.maximum(h["y_var"], 0.0))
    rmse = float(h["rmse"])

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="ideal")
    ax.errorbar(y_true, y_pred, yerr=y_sd, fmt="o", color="#3b6ea5",
                ecolor="#9bb7d4", capsize=3, ms=6, label="held-out points")
    ax.set_xlabel("true log-likelihood (CFD)")
    ax.set_ylabel("GP-predicted log-likelihood")
    ax.set_title(f"Surrogate accuracy on holdout\nRMSE = {rmse:.2f}, "
                 f"$R^2$ = {r2:.3f}  (n = {len(y_true)})")
    ax.legend(frameon=False)

    ax2 = axes[1]
    resid = y_pred - y_true
    ax2.axhline(0, color="k", lw=1)
    ax2.stem(np.arange(len(resid)), resid, basefmt=" ", linefmt="#3b6ea5",
             markerfmt="o")
    ax2.set_xlabel("holdout point index")
    ax2.set_ylabel("residual (pred - true)")
    ax2.set_title("Surrogate residuals")

    fig.suptitle(f"GP surrogate diagnostics - {args.param_set}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, f"surrogate_{args.param_set}.png")


if __name__ == "__main__":
    main()
