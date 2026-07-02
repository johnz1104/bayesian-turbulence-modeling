"""Figures for the backward-facing-step model-form evidence memo.

Reads the curated finding_numbers.json in this directory and writes the three
memo figures to figures/. Colors are the validated categorical slots (blue,
aqua, yellow; worst adjacent CVD deltaE 47.2 on the light surface); the aqua
and yellow slots sit below 3:1 contrast on white, so every interval carries a
visible direct label and every quoted value also lives in the memo tables.

Run from the repo root:
  python3 UQ-RANS_research/separated_modelform/make_bfs_figures.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

C_FLOW, C_GAUSS, C_EIGEN = "#2a78d6", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"


def main():
    os.makedirs(FIG, exist_ok=True)
    d = json.load(open(os.path.join(HERE, "finding_numbers.json")))
    po = d["aposteriori"]
    ap = d["apriori"]
    truth = po["baseline"]["reattachment_dns"]
    base = po["baseline"]["reattachment"]

    # ---- figure 1: reattachment predictive intervals across methods --------
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rows = []
    fr = po["ensembles"]["flow"]["reattachment"]
    gr = po["ensembles"]["gauss"]["reattachment"]
    e5 = po["eigenspace"]["0.5"]
    e1 = po["eigenspace"]["1.0"]
    rows.append(("generative flow (90% band)", fr["band"], fr["mean"], C_FLOW,
                 [m["reattachment"] for m in po["ensembles"]["flow"]["members"]
                  if m["status"] == "Converged"]))
    rows.append(("Gaussian model-form (90% band)", gr["band"], gr["mean"],
                 C_GAUSS,
                 [m["reattachment"] for m in po["ensembles"]["gauss"]["members"]
                  if m["status"] == "Converged"]))
    rows.append(("eigenspace envelope, Delta_B=0.5", e5["envelope"], None,
                 C_EIGEN, list(e5["corners"].values())))
    rows.append(("eigenspace envelope, Delta_B=1.0", e1["envelope"], None,
                 C_EIGEN, list(e1["corners"].values())))

    for i, (name, band, mean, color, pts) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot(band, [y, y], color=color, lw=3, solid_capstyle="round",
                zorder=2)
        ax.scatter(pts, np.full(len(pts), y), s=14, color=color, alpha=0.55,
                   edgecolors="white", linewidths=0.4, zorder=3)
        if mean is not None:
            ax.scatter([mean], [y], s=46, color=color, edgecolors="white",
                       linewidths=0.8, zorder=4)
        ax.text(band[1] + 0.12, y, f"{name}  [{band[0]:.2f}, {band[1]:.2f}]",
                va="center", fontsize=8.5, color=INK)

    ax.axvline(truth, color=INK, lw=1.2, ls="--", zorder=1)
    ax.text(truth, len(rows) - 0.35, f"DNS {truth:.2f}", fontsize=8.5,
            color=INK, ha="center")
    ax.axvline(base, color=MUTED, lw=1.2, ls=":", zorder=1)
    ax.text(base, -0.62, f"baseline SST {base:.2f}", fontsize=8.5, color=MUTED,
            ha="center")
    ax.set_yticks([])
    ax.set_xlim(4.2, 11.4)
    ax.set_ylim(-0.9, len(rows) - 0.1)
    ax.set_xlabel("reattachment length x_r / h", color=INK)
    ax.set_title("Propagated reattachment: predictive bands and envelopes vs "
                 "the DNS truth", fontsize=10, color=INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bfs_reattachment_intervals.png"), dpi=150)
    plt.close(fig)

    # ---- figure 2: a-priori station-held-out coverage -----------------------
    loso = ap["leave_one_station_out"]
    xs = sorted(float(k) for k in loso)
    cov_f = [np.mean(list(loso[str(x)]["flow"]["coverage"].values())) for x in xs]
    cov_g = [np.mean(list(loso[str(x)]["gauss"]["coverage"].values())) for x in xs]
    idx = np.arange(len(xs))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axhline(0.9, color=INK, lw=1.0, ls="--")
    ax.text(len(xs) - 0.52, 0.905, "nominal 0.90", fontsize=8.5, color=INK,
            ha="right")
    ax.plot(idx, cov_f, "-o", color=C_FLOW, lw=2, ms=6, label="generative flow")
    ax.plot(idx, cov_g, "-o", color=C_GAUSS, lw=2, ms=6,
            label="Gaussian model-form")
    for i, (cf_, cg_) in enumerate(zip(cov_f, cov_g)):
        ax.annotate(f"{cf_:.2f}", (i, cf_), textcoords="offset points",
                    xytext=(0, 7), fontsize=7.5, color=C_FLOW, ha="center")
        ax.annotate(f"{cg_:.2f}", (i, cg_), textcoords="offset points",
                    xytext=(0, -13), fontsize=7.5, color="#0e7a52",
                    ha="center")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel("held-out station x/h", color=INK)
    ax.set_ylabel("coverage at nominal 0.90", color=INK)
    ax.set_ylim(0.35, 1.05)
    ax.set_title("A-priori anisotropy coverage, leave-one-station-out",
                 fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bfs_apriori_coverage.png"), dpi=150)
    plt.close(fig)

    print(f"wrote figures to {FIG}/")


if __name__ == "__main__":
    main()
