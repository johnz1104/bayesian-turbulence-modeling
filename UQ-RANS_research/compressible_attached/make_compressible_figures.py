"""Figures for the compressible attached-flow evidence package.

Reads results/compressible/finding_numbers.json (the fixed-seed production
output of reproduce_compressible.py) and writes the four evidence figures.
Every plotted number traces to that JSON; nothing is computed here beyond
layout.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from UQ.datasets import GVChannelDNS

LEVEL = 0.9


def load(results):
    with open(os.path.join(results, "finding_numbers.json")) as fh:
        return json.load(fh)


def mach_of(tag):
    return GVChannelDNS.parse_tag(tag)[1]


def fig_in_distribution(numbers, outdir):
    rec = numbers["in_distribution"]
    tags = sorted(rec, key=mach_of)
    mach = [mach_of(t) for t in tags]
    std = [rec[t]["standard_thermal_coverage"] for t in tags]
    tmp = [rec[t]["tempered_thermal_coverage"] for t in tags]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(mach, std, "o", color="#b3443c", label="standard Bayes")
    ax.plot(mach, tmp, "s", color="#2a6f97", label="generalized Bayes")
    ax.axhline(LEVEL, color="k", lw=0.8, ls="--", label=f"nominal {LEVEL}")
    ax.set_xlabel("centreline Mach number M_CLx")
    ax.set_ylabel("held-out thermal coverage")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("In-distribution coverage of the held-out thermal block")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "in_distribution_thermal_coverage.png"),
                dpi=160)
    plt.close(fig)


def fig_cross_mach(numbers, outdir):
    rec = numbers["cross_mach"]["primary"]
    tags = sorted(rec, key=mach_of)
    mach = [mach_of(t) for t in tags]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, style, color, label in (
            ("standard_thermal_coverage", "o", "#b3443c", "standard"),
            ("tempered_thermal_coverage", "s", "#2a6f97",
             "generalized Bayes"),
            ("conformal_thermal_coverage", "^", "#3a7d44", "conformal")):
        ax.plot(mach, [rec[t][key] for t in tags], style, color=color,
                label=label)
    ax.axhline(LEVEL, color="k", lw=0.8, ls="--")
    ax.set_xlabel("held-out centreline Mach number")
    ax.set_ylabel("held-out thermal coverage")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Cross-Mach transfer: calibrate M < 1, predict M >= 1.47")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "crossmach_thermal_coverage.png"),
                dpi=160)
    plt.close(fig)


def fig_prt(numbers, outdir):
    plate = numbers["plate"]
    post = plate["prt_posterior"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axvspan(post["q05"], post["q95"], color="#2a6f97", alpha=0.25,
               label="pooled Pr_t posterior (5-95)")
    ax.axvline(post["mean"], color="#2a6f97", lw=1.5)
    ax.axvline(0.9, color="k", lw=0.8, ls="--", label="fixed 0.9 convention")
    y = 0
    for tag, rec in plate.items():
        if not isinstance(rec, dict) or "measured_prt" not in rec:
            continue
        m = rec["measured_prt"]
        ax.plot([m["q25"], m["q75"]], [y, y], color="#3a7d44", lw=3,
                alpha=0.8)
        ax.plot(m["median"], y, "o", color="#3a7d44")
        ax.text(1.52, y, f"{tag} (Tw/Tr {rec['tw_tr']:g})", fontsize=8,
                va="center")
        y += 1
    ax.set_xlabel("turbulent Prandtl number")
    ax.set_yticks([])
    ax.set_xlim(0.45, 1.55)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Pr_t posterior against the measured plate profiles")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "prt_posterior_vs_measured.png"),
                dpi=160)
    plt.close(fig)


def fig_plate(numbers, outdir):
    plate = numbers["plate"]
    rows = [(tag, rec) for tag, rec in plate.items()
            if isinstance(rec, dict) and "standard_coverage" in rec]
    rows.sort(key=lambda r: r[1]["tw_tr"])
    x = [r[1]["tw_tr"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, [r[1]["standard_coverage"] for r in rows], "o",
            color="#b3443c", label="standard")
    ax.plot(x, [r[1]["tempered_coverage"] for r in rows], "s",
            color="#2a6f97", label="generalized Bayes")
    ax.axhline(LEVEL, color="k", lw=0.8, ls="--")
    for xi, (tag, rec) in zip(x, rows):
        ax.annotate(f"M{rec['m_inf']:g}", (xi, 1.0), fontsize=8,
                    ha="center")
    ax.set_xlabel("wall-to-recovery temperature ratio Tw/Tr")
    ax.set_ylabel("heat-flux-profile coverage")
    ax.set_ylim(-0.03, 1.1)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Wall-cooling axis: plate heat-flux profile coverage")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "plate_wallcooling_coverage.png"),
                dpi=160)
    plt.close(fig)


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else "results/compressible"
    outdir = sys.argv[2] if len(sys.argv) > 2 else \
        "UQ-RANS_research/compressible_attached/figures"
    os.makedirs(outdir, exist_ok=True)
    numbers = load(results)
    fig_in_distribution(numbers, outdir)
    fig_cross_mach(numbers, outdir)
    fig_prt(numbers, outdir)
    fig_plate(numbers, outdir)
    print(f"figures written to {outdir}")


if __name__ == "__main__":
    main()
