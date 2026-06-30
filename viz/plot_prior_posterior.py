"""
Prior-versus-posterior comparison per coefficient: the truncated-normal prior
(Menter mean, 15% std) overlaid on the posterior marginal histogram from the
real chain.  Shows which coefficients the data moved and tightened, and which
stayed at the prior.  Source: viz/artifacts/<param_set>/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _common import (load_chain, load_summary, have, set_style, save, plt,
                     PRIOR_REL_STD)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("param_set", nargs="?", default="a1_betaStar")
    args = parser.parse_args()

    if not have(args.param_set):
        print(f"  no artifacts for {args.param_set}; run run_calibration.py first")
        return

    set_style()
    meta = load_summary(args.param_set)
    flat = load_chain(args.param_set)["flat"]
    names = meta["names"]
    ndim = len(names)

    fig, axes = plt.subplots(1, ndim, figsize=(4.2 * ndim, 3.8), squeeze=False)
    for i, name in enumerate(names):
        ax = axes[0][i]
        s = flat[:, i]
        post = meta["posterior"][name]
        mu0 = post["prior_mean"]
        sd0 = PRIOR_REL_STD * abs(mu0)

        ax.hist(s, bins=45, density=True, color="#3b6ea5", alpha=0.55,
                label="posterior")
        grid = np.linspace(min(s.min(), mu0 - 3 * sd0),
                           max(s.max(), mu0 + 3 * sd0), 400)
        prior_pdf = np.exp(-0.5 * ((grid - mu0) / sd0) ** 2) / (sd0 * np.sqrt(2 * np.pi))
        ax.plot(grid, prior_pdf, color="#c44", lw=1.8, label="prior")
        ax.axvline(mu0, color="#c44", ls=":", lw=1)
        ax.axvline(post["mean"], color="#1b3a5b", ls="--", lw=1.2)

        tighten = 100.0 * (1.0 - post["std"] / sd0)
        ax.set_title(f"{name}\nposterior {post['mean']:.4f} ± {post['std']:.4f}\n"
                     f"shift {post['shift']:+.2f}σ, width {tighten:+.0f}% vs prior",
                     fontsize=9)
        ax.set_xlabel(name)
        if i == 0:
            ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f"Prior vs posterior - {args.param_set} (channel $C_f$, Dean 1978)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, f"prior_posterior_{args.param_set}.png")


if __name__ == "__main__":
    main()
