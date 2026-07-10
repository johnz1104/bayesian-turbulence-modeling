"""Figures for the heat-flux model-form evidence package.

Consumes the production outputs of python/UQ/reproduce_heatflux_modelform.py
(results/heatflux/finding_numbers.json and figure_arrays.npz, fixed seeds) and
writes the committed figures next to this script. Run from the repo root:

    python3 UQ-RANS_research/heatflux_modelform/make_heatflux_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
# anchor the run-output directory to this script's own repo root, not the
# process cwd: both the primary checkout and the worktree accumulate a
# gitignored results/heatflux, so a cwd-relative path could silently rebuild
# the committed figures from the OTHER checkout's run
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO, "results", "heatflux")

KIND_STYLE = {
    "flow": dict(color="tab:blue", marker="o", label="conditional flow"),
    "gauss": dict(color="tab:orange", marker="s",
                  label="conditional Gaussian"),
    "pooled": dict(color="tab:gray", marker="^",
                   label="pooled unconditional (diagnostic)"),
}


def mean(rec, key):
    return rec[key]["mean"]


def fig_crossmach_coverage(d):
    """Held-out coverage at nominal 0.90 against the case Mach number."""
    asm = d["assembly"]
    gv16 = d["config"]["high_mach"]
    cm = d["cross_mach"]["wall_normal"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(0.9, color="k", lw=0.8, ls="--", label="nominal 0.90")
    ax.axhspan(0.80, 0.98, color="green", alpha=0.08,
               label="pre-registered band")
    for kind, st in KIND_STYLE.items():
        x = [asm[t]["m_nom"] for t in gv16]
        y = [mean(cm[kind][t], "coverage_y_0.9") for t in gv16]
        ax.plot(x, y, ls="none", ms=6, alpha=0.85, **st)
        # the in-distribution control (leave-one-case-out) as the anchor
        loco = np.mean([mean(v, "coverage_y_0.9")
                        for v in d["loco"][kind].values()])
        ax.plot([0.55], [loco], marker=st["marker"], color=st["color"],
                ms=9, mfc="none", mew=2)
    ax.annotate("open: LOCO control\n(low-Mach, in distribution)",
                xy=(0.57, 0.95), fontsize=8, color="0.3")
    ax.set_xlabel("case centreline Mach number M_CLx")
    ax.set_ylabel("per-case coverage of dq_y at nominal 0.90")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("Cross-Mach transfer of the conditional heat-flux models\n"
                 "(train M_CLx < 1.0, seed means)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "heatflux_crossmach_coverage.png"),
                dpi=160)
    plt.close(fig)


def fig_conformal_scores(d):
    """The secondary axis: normalized conformal re-scoring."""
    asm = d["assembly"]
    gv16 = sorted(d["config"]["high_mach"],
                  key=lambda t: asm[t]["m_nom"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    ax = axes[0]
    scores = ("absolute", "bq_base", "semilocal")
    width = 0.35
    for i, rel in enumerate(("rel0.005", "rel0.01")):
        vals = [d["conformal"][rel][s]["level0.9"]["mean_coverage"]
                for s in scores]
        ax.bar(np.arange(3) + (i - 0.5) * width, vals, width,
               label=f"sigma level {rel[3:]}", alpha=0.85)
    ax.axhline(0.9, color="k", lw=0.8, ls="--")
    ax.axhspan(0.80, 0.98, color="green", alpha=0.08)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["absolute\n(control)", "B_q-normalized\n(primary)",
                        "semi-local\n(sensitivity)"], fontsize=8)
    ax.set_ylabel("held-out thermal conformal coverage at 0.90")
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=8)
    ax.set_title("Score normalization alone: mean over the\n16 held-out"
                 " cases (global-correction residuals)", fontsize=9)

    ax = axes[1]
    for rel, c, mk in (("rel0.005", "tab:green", "o"),
                       ("rel0.01", "tab:olive", "s")):
        per = d["conformal"][rel]["bq_base"]["level0.9"]["per_case"]
        x = [asm[t]["m_nom"] for t in gv16]
        y = [per[t]["coverage"] for t in gv16]
        ax.plot(x, y, marker=mk, ls="none", color=c, ms=6, alpha=0.85,
                label=f"B_q-normalized, {rel[3:]}")
    ax.axhline(0.9, color="k", lw=0.8, ls="--")
    ax.set_xlabel("case centreline Mach number M_CLx")
    ax.set_ylabel("per-case thermal coverage at 0.90")
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=8)
    ax.set_title("The residual failure after scale normalization\n"
                 "grows with Mach (shape, not scale)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "heatflux_conformal_scores.png"), dpi=160)
    plt.close(fig)


def fig_family_scores(d):
    """The family clause on the joint leg: flow against Gaussian per case."""
    j = d["cross_mach"]["joint_xy"]
    gv16 = d["config"]["high_mach"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
    for ax, metric, label in ((axes[0], "energy_score", "energy score"),
                              (axes[1], "crps_y", "wall-normal CRPS")):
        f = np.array([mean(j["flow"][t], metric) for t in gv16])
        g = np.array([mean(j["gauss"][t], metric) for t in gv16])
        lim = (0, 1.1 * max(f.max(), g.max()))
        ax.plot(lim, lim, color="k", lw=0.8, ls="--")
        ax.plot(g, f, "o", color="tab:blue", ms=6, alpha=0.85)
        ax.set_xlabel(f"Gaussian {label}")
        ax.set_ylabel(f"flow {label}")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        wins = int(np.sum(f < g))
        ax.set_title(f"{label}: flow better on {wins}/16 held-out cases"
                     " (below the diagonal)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "heatflux_family_scores.png"), dpi=160)
    plt.close(fig)


def fig_pit(d, arrays):
    """Wall-normal PIT histograms: in sample, the LOCO control, cross-Mach."""
    gv = set(d["assembly"]["gv"])
    panels = (("in_sample_wall_normal", "in sample (train on all)", None),
              ("loco", "leave-one-case-out control", None),
              ("cross_mach_wall_normal", "cross-Mach held out (GV)", gv))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, (stage, title, keep) in zip(axes, panels):
        for kind, st in (("flow", KIND_STYLE["flow"]),
                         ("gauss", KIND_STYLE["gauss"])):
            pit = [v for k, v in arrays.items()
                   if k.startswith(f"{stage}/{kind}/seed0/")
                   and (keep is None or k.split("/")[3] in keep)]
            if not pit:
                continue
            pit = np.concatenate(pit)
            ax.hist(pit, bins=10, range=(0, 1), density=True,
                    histtype="step", lw=1.8, color=st["color"],
                    label=st["label"])
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("PIT of dq_y")
    axes[0].set_ylabel("density (flat = calibrated)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "heatflux_pit.png"), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(RESULTS, "finding_numbers.json")) as fh:
        d = json.load(fh)
    arrays = dict(np.load(os.path.join(RESULTS, "figure_arrays.npz")))
    fig_crossmach_coverage(d)
    fig_conformal_scores(d)
    fig_family_scores(d)
    fig_pit(d, arrays)
    print(f"wrote 4 figures to {FIGS}")


if __name__ == "__main__":
    main()
