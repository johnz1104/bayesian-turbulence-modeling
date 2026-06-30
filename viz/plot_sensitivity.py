"""
Parameter-influence (sensitivity) bar chart from the surrogate's ARD-RBF
lengthscales.  Relevance is taken as 1/lengthscale: a short lengthscale means
the log-likelihood varies fast with that coefficient (influential); a long one
means the data barely sees it.  Source: viz/artifacts/<param_set>/summary.json.

Defaults to the richest available set (near_wall4 if present, else a1_betaStar).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _common import load_summary, have, set_style, save, plt


def pick_default():
    for ps in ("near_wall4", "all11", "a1_betaStar"):
        if have(ps):
            return ps
    return "a1_betaStar"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("param_set", nargs="?", default=None)
    args = parser.parse_args()
    param_set = args.param_set or pick_default()

    if not have(param_set):
        print(f"  no artifacts for {param_set}; run run_calibration.py first")
        return

    set_style()
    meta = load_summary(param_set)
    names = meta["names"]
    ls = np.array([meta["surrogate"]["ard_lengthscales"][n] for n in names])
    relevance = 1.0 / ls
    relevance = relevance / relevance.max()         # normalise to the top driver

    order = np.argsort(relevance)
    fig, ax = plt.subplots(figsize=(6.6, 0.7 * len(names) + 2.2))
    ax.barh(np.array(names)[order], relevance[order], color="#3b6ea5")
    for y, (n, r, l) in enumerate(zip(np.array(names)[order],
                                      relevance[order], ls[order])):
        ax.text(r + 0.01, y, f"  1/ℓ rel={r:.2f}  (ℓ={l:.2f})",
                va="center", fontsize=8)
    ax.set_xlabel("normalised relevance  (1 / ARD lengthscale)")
    ax.set_xlim(0, 1.25)
    ax.set_title(f"SST coefficient influence on channel $C_f$ - {param_set}\n"
                 "(longer bar = data constrains it more)")
    save(fig, f"sensitivity_{param_set}.png")


if __name__ == "__main__":
    main()
