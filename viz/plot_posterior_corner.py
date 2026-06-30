"""
Posterior corner plot of the calibrated SST coefficients, with the prior mean
(Menter 1994 default) marked on each panel.  Source: a real MCMC chain from
viz/artifacts/<param_set>/ (default a1_betaStar).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import corner
from _common import load_chain, load_summary, have, set_style, FIGURES, plt


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
    prior_means = [meta["posterior"][n]["prior_mean"] for n in names]

    fig = corner.corner(
        flat, labels=names, truths=prior_means, truth_color="#c44",
        show_titles=True, title_fmt=".4f", quantiles=[0.025, 0.5, 0.975],
        color="#3b6ea5", hist_kwargs={"color": "#3b6ea5"},
    )
    fig.suptitle(f"Posterior over SST coefficients - {args.param_set} "
                 f"(channel $C_f$, Dean 1978);  red line = prior mean (Menter 1994)",
                 fontsize=10, y=1.06)

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"posterior_corner_{args.param_set}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
