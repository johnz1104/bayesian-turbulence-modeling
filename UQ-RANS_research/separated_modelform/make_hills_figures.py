"""Curation and figures for the periodic-hills and cross-geometry evidence.

Reads the regenerable production outputs (results/separated/
crossgeom_numbers.json and hills_aposteriori_numbers.json), writes the curated
hills_numbers.json in this directory (per-member probe vectors stripped; the
probe scores they feed are kept), and renders the memo figures to figures/.
The within-first-geometry anchors come from the committed finding_numbers.json.
Colors are the validated categorical slots (blue, aqua, yellow); every interval
carries a direct label and every quoted value also lives in the memo tables.

Run from the repo root, after the two production scripts:
  python3 UQ-RANS_research/separated_modelform/make_hills_figures.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "..", "..", "results", "separated")

C_FLOW, C_GAUSS, C_EIGEN = "#2a78d6", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"


def curate():
    """Assemble the curated numbers file every quoted hills value traces to."""
    cg = json.load(open(os.path.join(RES, "crossgeom_numbers.json")))
    po = json.load(open(os.path.join(RES, "hills_aposteriori_numbers.json")))
    for section in (po.get("ensembles", {}), po.get("cross_geometry", {})):
        for rec in section.values():
            for leg in ([rec] + list(rec.values())
                        if isinstance(rec, dict) else [rec]):
                if isinstance(leg, dict) and "members" in leg:
                    for m in leg["members"]:
                        m.pop("u_probe", None)
    out = {
        "note": "curated production numbers for the periodic-hills and "
                "cross-geometry legs; regenerate via "
                "reproduce_separated_crossgeom.py and "
                "reproduce_separated_hills_aposteriori.py (fixed seed 0)",
        "crossgeom": cg,
        "aposteriori": po,
    }
    path = os.path.join(HERE, "hills_numbers.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out, path


def _cov(rec):
    return float(np.mean(list(rec["coverage"].values())))


def _shp(rec):
    return float(np.mean(list(rec["sharpness"].values())))


def fig_lobo(d):
    """Within-hills leave-one-band-out coverage, flow vs Gaussian."""
    lobo = d["crossgeom"]["within_hills"]["leave_one_band_out"]
    bands = sorted(lobo, key=float)
    cov_f = [_cov(lobo[b]["flow"]) for b in bands]
    cov_g = [_cov(lobo[b]["gauss"]) for b in bands]
    idx = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axhline(0.9, color=INK, lw=1.0, ls="--")
    ax.text(len(bands) - 0.52, 0.905, "nominal 0.90", fontsize=8.5, color=INK,
            ha="right")
    ax.plot(idx, cov_f, "-o", color=C_FLOW, lw=2, ms=6, label="generative flow")
    ax.plot(idx, cov_g, "-o", color=C_GAUSS, lw=2, ms=6,
            label="Gaussian model-form")
    for i, (cf_, cg_) in enumerate(zip(cov_f, cov_g)):
        ax.annotate(f"{cf_:.2f}", (i, cf_), textcoords="offset points",
                    xytext=(0, 7), fontsize=7.5, color=C_FLOW, ha="center")
        ax.annotate(f"{cg_:.2f}", (i, cg_), textcoords="offset points",
                    xytext=(0, -13), fontsize=7.5, color="#0e7a52", ha="center")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{int(float(b))}" for b in bands])
    ax.set_xlabel("held-out streamwise band", color=INK)
    ax.set_ylabel("coverage at nominal 0.90", color=INK)
    ax.set_ylim(0.35, 1.05)
    ax.set_title("Periodic hills: a-priori anisotropy coverage, "
                 "leave-one-band-out", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "hills_apriori_coverage.png"), dpi=150)
    plt.close(fig)


def fig_transfer(d):
    """Cross-geometry a-priori transfer: coverage with interval width labels.

    Left panel scores on the hills points (within-hills against the
    first-geometry-trained transfer); right panel scores on the first
    geometry's points (within against the hills-trained transfer). The width
    labels carry the widening-versus-coverage reading.
    """
    bfs = json.load(open(os.path.join(HERE, "finding_numbers.json")))
    wi_h = d["crossgeom"]["within_hills"]["in_distribution"]
    wi_b = bfs["apriori"]["in_distribution"]
    cg = d["crossgeom"]["cross_geometry"]

    panels = [
        ("scored on the hills points",
         [("within-hills", wi_h),
          ("transfer", {k: cg[k]["bfs_to_hills"] for k in ("flow", "gauss")})]),
        ("scored on the step points",
         [("within-step", wi_b),
          ("transfer", {k: cg[k]["hills_to_bfs"] for k in ("flow", "gauss")})]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    for ax, (title, cols) in zip(axes, panels):
        ax.axhline(0.9, color=INK, lw=1.0, ls="--")
        for j, (label, recs) in enumerate(cols):
            for kind, color, dx in (("flow", C_FLOW, -0.09),
                                    ("gauss", C_GAUSS, 0.09)):
                r = recs[kind]
                ax.scatter([j + dx], [_cov(r)], s=52, color=color,
                           edgecolors="white", linewidths=0.8, zorder=3)
                ax.annotate(f"{_cov(r):.2f}\nw {_shp(r):.3f}", (j + dx, _cov(r)),
                            textcoords="offset points", xytext=(0, 8),
                            fontsize=7, color=color, ha="center")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c[0] for c in cols], fontsize=9)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_ylim(0.25, 1.1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("coverage at nominal 0.90", color=INK)
    axes[0].scatter([], [], color=C_FLOW, label="generative flow")
    axes[0].scatter([], [], color=C_GAUSS, label="Gaussian model-form")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Cross-geometry a-priori transfer (w = mean 90 percent "
                 "interval width)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "hills_crossgeom_transfer.png"), dpi=150)
    plt.close(fig)


def fig_intervals(d):
    """Propagated hills reattachment: bands, envelopes and transfer vs DNS."""
    po = d["aposteriori"]
    truth = po["baseline"]["truths_dns"]["reattachment"]
    base = po["baseline"]["reattachment"]
    rows = []
    for kind, color, label in (("flow", C_FLOW, "generative flow"),
                               ("gauss", C_GAUSS, "Gaussian model-form")):
        r = po["ensembles"][kind]["reattachment"]
        pts = [m["reattachment"] for m in po["ensembles"][kind]["members"]
               if m["status"] == "Converged"
               and np.isfinite(m["reattachment"])]
        rows.append((f"{label} (90% band)", r.get("band"), r.get("mean"),
                     color, pts))
    for delta in ("0.5", "1.0"):
        e = po["eigenspace"][delta]["reattachment"]
        rows.append((f"eigenspace envelope, Delta_B={delta}",
                     e.get("envelope"), None, C_EIGEN,
                     list(e.get("corners", {}).values())))
    for kind, color, label in (("flow", C_FLOW, "step-trained flow"),
                               ("gauss", C_GAUSS, "step-trained Gaussian")):
        r = po["cross_geometry"][kind]["bfs_to_hills"]["reattachment"]
        pts = [m["reattachment"]
               for m in po["cross_geometry"][kind]["bfs_to_hills"]["members"]
               if m["status"] == "Converged"
               and np.isfinite(m["reattachment"])]
        rows.append((f"{label} (transfer, 90% band)", r.get("band"),
                     r.get("mean"), color, pts))

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    kept = [row for row in rows if row[1]]
    for i, (name, band, mean, color, pts) in enumerate(kept):
        y = len(kept) - 1 - i
        ax.plot(band, [y, y], color=color, lw=3, solid_capstyle="round",
                zorder=2)
        ax.scatter(pts, np.full(len(pts), y), s=14, color=color, alpha=0.55,
                   edgecolors="white", linewidths=0.4, zorder=3)
        if mean is not None:
            ax.scatter([mean], [y], s=46, color=color, edgecolors="white",
                       linewidths=0.8, zorder=4)
        ax.text(band[1] + 0.10, y, f"{name}  [{band[0]:.2f}, {band[1]:.2f}]",
                va="center", fontsize=8, color=INK)
    ax.axvline(truth, color=INK, lw=1.2, ls="--", zorder=1)
    ax.text(truth, len(kept) - 0.3, f"DNS {truth:.2f}", fontsize=8.5,
            color=INK, ha="center")
    ax.axvline(base, color=MUTED, lw=1.2, ls=":", zorder=1)
    ax.text(base, -0.62, f"baseline SST {base:.2f}", fontsize=8.5, color=MUTED,
            ha="center")
    ax.set_yticks([])
    ax.set_xlim(1.0, 13.0)
    ax.set_ylim(-0.9, len(kept) - 0.1)
    ax.set_xlabel("reattachment length x_r / h", color=INK)
    ax.set_title("Periodic hills: propagated reattachment vs the DNS truth",
                 fontsize=10, color=INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "hills_reattachment_intervals.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    d, path = curate()
    fig_lobo(d)
    fig_transfer(d)
    fig_intervals(d)
    print(f"curated {path}")
    print(f"wrote figures to {FIG}/")


if __name__ == "__main__":
    main()
