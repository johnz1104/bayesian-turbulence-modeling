"""
MCMC convergence diagnostics from a real emcee chain: per-coefficient walker
traces (with split-R-hat annotated) and the chain autocorrelation function.
Source: viz/artifacts/<param_set>/chain.npz.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _common import load_chain, load_summary, have, set_style, save, plt


def autocorr(x, max_lag):
    """Normalised autocorrelation of a 1-D series (mean over walkers)."""
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    lags = np.arange(0, max_lag)
    ac = np.array([np.dot(x[: n - k], x[k:]) / (n - k) for k in lags]) / var
    return lags, ac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("param_set", nargs="?", default="a1_betaStar")
    args = parser.parse_args()

    if not have(args.param_set):
        print(f"  no artifacts for {args.param_set}; run run_calibration.py first")
        return

    set_style()
    meta = load_summary(args.param_set)
    npz = load_chain(args.param_set)
    chain = npz["chain"]            # (n_steps, n_walkers, ndim)
    rhat = npz["rhat"]
    names = meta["names"]
    ndim = chain.shape[2]

    fig, axes = plt.subplots(ndim, 2, figsize=(11, 2.4 * ndim + 0.5),
                             squeeze=False)
    for i, name in enumerate(names):
        # traces (subsample walkers for legibility)
        ax = axes[i][0]
        nw = chain.shape[1]
        for w in range(0, nw, max(1, nw // 12)):
            ax.plot(chain[:, w, i], lw=0.6, alpha=0.5)
        ax.axhline(meta["posterior"][name]["mean"], color="k", lw=1.2, ls="--")
        ax.set_ylabel(name)
        ax.set_title(f"{name} trace   (split-$\\hat{{R}}$ = {rhat[i]:.3f})",
                     fontsize=10)
        if i == ndim - 1:
            ax.set_xlabel("step")

        # autocorrelation of the walker-mean series
        ax2 = axes[i][1]
        series = chain[:, :, i].mean(axis=1)
        max_lag = min(150, len(series) // 2)
        lags, ac = autocorr(series, max_lag)
        ax2.plot(lags, ac, color="#3b6ea5")
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_ylabel("autocorr")
        ax2.set_title(f"{name} autocorrelation", fontsize=10)
        if i == ndim - 1:
            ax2.set_xlabel("lag")

    af = float(npz["acceptance"].mean())
    fig.suptitle(f"MCMC convergence - {args.param_set}   "
                 f"(mean acceptance {af:.2f}, all $\\hat{{R}}$ < 1.05 = converged)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, f"convergence_{args.param_set}.png")


if __name__ == "__main__":
    main()
