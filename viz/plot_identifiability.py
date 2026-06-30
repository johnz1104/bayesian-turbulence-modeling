"""
Identifiability view: the posterior correlation matrix of the calibrated
coefficients.  Strong off-diagonal correlation means the data constrains a
combination of coefficients rather than each one separately (a non-identifiable
direction).  Source: viz/artifacts/<param_set>/summary.json.

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
    names = meta["posterior_correlation"]["names"]
    C = np.array(meta["posterior_correlation"]["matrix"])
    n = len(names)

    fig, ax = plt.subplots(figsize=(0.9 * n + 2.6, 0.9 * n + 2.2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(C[i, j]) > 0.55 else "black", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="posterior correlation")
    ax.set_title(f"Posterior correlation - {param_set}\n"
                 "(off-diagonal magnitude = non-identifiable directions)")
    fig.tight_layout()
    save(fig, f"identifiability_{param_set}.png")


if __name__ == "__main__":
    main()
