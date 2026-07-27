"""Evidence-package figures for the shock-interaction phase.

Reads only the study outputs (the numbers JSONs, the cached wall records and
the a-posteriori member files) and the DNS records; writes the package
figures. Panels degrade gracefully: a panel whose inputs are absent is
skipped with a note, so the maker runs on partial (or quick) results.

    python3 python/UQ/make_sbli_figures.py --results results/sbli \
        --out UQ-RANS_research/shock_interaction/figures
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "build"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from UQ.reproduce_sbli_apriori import (_all_records, _wall_path,
                                       GATE_A_CF, GATE_A_STATION)
from UQ.reproduce_sbli_aposteriori import (_member_path, _member_config,
                                           _member_current,
                                           _corner_member_path,
                                           _load_member, KINDS, FOLDS,
                                           MODEL_SEEDS,
                                           fold_score_lineage_ok)
from UQ.datasets.sbli_aposteriori import STATIONS

KIND_COLORS = {"flow": "tab:blue", "gauss": "tab:orange",
               "pooled": "tab:green"}
LEGS = ("db", "dq_y", "dq_joint")


def _seed_dim_mean(per_seed, key):
    vals = []
    for s in per_seed:
        v = np.asarray(s[key], dtype=float)
        vals.append(float(np.mean(v)))
    return float(np.mean(vals))


def fig_gates(numbers, records, results_dir, out):
    """Gate A attached skin friction and the gate-B baseline wall pressure
    against the records."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    p = _wall_path(results_dir, "gate_a_attached")
    if os.path.isfile(p):
        w = np.load(p)
        ax.plot(w["x_star"], 1e3 * w["Cf"], "-", color="k",
                label="baseline (attached)")
        ax.plot([GATE_A_STATION], [1e3 * GATE_A_CF], "o", color="tab:red",
                label="measured incoming layer")
        ga = numbers.get("gates", {}).get("A", {})
        if "cf_at_station" in ga:
            ax.plot([GATE_A_STATION], [1e3 * ga["cf_at_station"]], "s",
                    color="tab:blue",
                    label=f"solve ({100 * ga['cf_rel_error']:.1f}% off)")
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$10^3\, C_f$")
        ax.legend(fontsize=8)
        ax.set_title("gate A: attached configuration")

    ax = axes[1]
    plotted = False
    for i, case in enumerate(FOLDS):
        p = _wall_path(results_dir, case)
        rec = records.get(case)
        if not os.path.isfile(p) or rec is None or rec.series.cp is None:
            continue
        w = np.load(p)
        c = plt.cm.viridis(i / max(len(FOLDS) - 1, 1))
        ax.plot(w["x_star"], w["Cp"], "-", color=c, label=f"{case} solve")
        ax.plot(rec.series.x, rec.series.cp, ":", color=c, lw=1.2,
                label=f"{case} DNS")
        plotted = True
    if plotted:
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$C_p$")
        ax.legend(fontsize=7)
        ax.set_title("gate B: interaction baselines")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_gates.png"), dpi=160)
    plt.close(fig)
    print("wrote fig1_gates.png")


def fig_loso(numbers, out):
    """Held-out coverage at nominal 0.9 per leg, fold and model."""
    loso = numbers.get("loso")
    if not loso:
        print("skip fig2 (no loso block)")
        return
    legs = [l for l in LEGS if l in loso]
    fig, axes = plt.subplots(1, len(legs), figsize=(4.2 * len(legs), 3.6),
                             squeeze=False)
    for j, leg in enumerate(legs):
        ax = axes[0][j]
        folds = sorted(loso[leg])
        xs = np.arange(len(folds))
        for kind in ("flow", "gauss", "pooled"):
            ys = []
            for held in folds:
                per_seed = loso[leg][held]["models"].get(kind)
                ys.append(_seed_dim_mean(per_seed, "coverage_0.9")
                          if per_seed else np.nan)
            ax.plot(xs, ys, "o-", color=KIND_COLORS[kind], label=kind,
                    ms=4)
        ax.axhline(0.9, color="k", lw=0.8, ls="--")
        ax.axhspan(0.80, 0.98, color="k", alpha=0.06)
        ax.set_xticks(xs)
        ax.set_xticklabels(folds, rotation=45, fontsize=7)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"LOso: {leg}")
        if j == 0:
            ax.set_ylabel("held-out coverage @0.9")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_apriori_loso.png"), dpi=160)
    plt.close(fig)
    print("wrote fig2_apriori_loso.png")


def fig_regions(numbers, out):
    """Region-resolved coverage (the interaction zone against the attached
    stretches), seed-and-fold mean."""
    loso = numbers.get("loso")
    if not loso:
        print("skip fig3 (no loso block)")
        return
    legs = [l for l in LEGS if l in loso]
    fig, axes = plt.subplots(1, len(legs), figsize=(4.2 * len(legs), 3.6),
                             squeeze=False)
    for j, leg in enumerate(legs):
        ax = axes[0][j]
        region_names = None
        for kind in ("flow", "gauss"):
            acc = {}
            for held, block in loso[leg].items():
                per_seed = block["models"].get(kind)
                if not per_seed:
                    continue
                for s in per_seed:
                    rc = s.get("region_coverage_0.9", {})
                    for r, v in rc.items():
                        acc.setdefault(r, []).append(float(np.mean(v)))
            if not acc:
                continue
            region_names = sorted(acc)
            xs = np.arange(len(region_names))
            ax.plot(xs, [np.mean(acc[r]) for r in region_names], "s-",
                    color=KIND_COLORS[kind], label=kind)
        if region_names:
            ax.axhline(0.9, color="k", lw=0.8, ls="--")
            ax.set_xticks(np.arange(len(region_names)))
            ax.set_xticklabels(region_names, fontsize=8)
            ax.set_ylim(0.0, 1.02)
            ax.set_title(f"regions: {leg}")
            if j == 0:
                ax.set_ylabel("coverage @0.9")
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_apriori_regions.png"), dpi=160)
    plt.close(fig)
    print("wrote fig3_apriori_regions.png")


def fig_bands(records, results_dir, fold, n_members, out):
    """A-posteriori wall bands: the flow and Gaussian 5-95 bands and the
    corner envelope against the record."""
    rec = records.get(fold)
    if rec is None:
        print(f"skip fig5 {fold} (no record)")
        return
    # per-seed ensembles, identity-gated (stale or unfingerprinted member
    # caches never render), drawn per model seed and never pooled across
    # seeds (the seed-resolved protocol)
    ensembles = {}
    for kind in KINDS:
        per_seed = {}
        for ms in MODEL_SEEDS:
            members = []
            for i in range(n_members):
                p = _member_path(results_dir, fold, kind, i, model_seed=ms)
                c = _member_config(results_dir, fold, kind=kind, index=i,
                                   model_seed=ms)
                if _member_current(p, c):
                    m = _load_member(p)
                    if "Converged" in m["status"]:
                        members.append(m)
            if len(members) >= 2:
                per_seed[ms] = members
        if per_seed:
            ensembles[kind] = per_seed
    corners = []
    for lab in ("1C_d1", "2C_d1", "3C_d1", "1C_d0.5", "2C_d0.5", "3C_d0.5"):
        p = _corner_member_path(results_dir, fold, lab)
        if _member_current(p, _member_config(results_dir, fold, corner=lab)):
            m = _load_member(p)
            if "Converged" in m["status"]:
                corners.append(m)
    if not ensembles and not corners:
        print(f"skip fig5 {fold} (no converged members)")
        return

    series = rec.series
    quantities = [("Cf", series.cf, 1e3, r"$10^3\, C_f$")]
    if series.St is not None and not np.all(np.isnan(series.St)):
        quantities.append(("St", series.St, 1e3, r"$10^3\, St$"))
    fig, axes = plt.subplots(1, len(quantities),
                             figsize=(5.5 * len(quantities), 4),
                             squeeze=False)
    for j, (q, truth, fac, lab) in enumerate(quantities):
        ax = axes[0][j]
        xg = np.linspace(series.x[0], series.x[-1], 300)
        for kind, per_seed in ensembles.items():
            counts = ",".join(str(len(per_seed[ms]))
                              for ms in sorted(per_seed))
            for j_ms, ms in enumerate(sorted(per_seed)):
                ens = np.stack([np.interp(xg, m["wall"]["x_star"],
                                          m["wall"][q])
                                for m in per_seed[ms]])
                lo = np.quantile(ens, 0.05, axis=0)
                hi = np.quantile(ens, 0.95, axis=0)
                ax.fill_between(xg, fac * lo, fac * hi,
                                alpha=0.25 / max(1, len(per_seed)) * 2,
                                color=KIND_COLORS[kind],
                                label=(f"{kind} 5-95 per seed ({counts})"
                                       if j_ms == 0 else None))
        if len(corners) >= 2:
            ens = np.stack([np.interp(xg, m["wall"]["x_star"],
                                      m["wall"][q]) for m in corners])
            ax.plot(xg, fac * ens.min(axis=0), "-", color="gray", lw=0.9)
            ax.plot(xg, fac * ens.max(axis=0), "-", color="gray", lw=0.9,
                    label=f"corner envelope ({len(corners)})")
        ax.plot(series.x, fac * truth, "k.", ms=2.5, label="DNS")
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(lab)
        ax.legend(fontsize=7)
        ax.set_title(f"{fold}: coupled wall {q}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig5_aposteriori_{fold}.png"), dpi=160)
    plt.close(fig)
    print(f"wrote fig5_aposteriori_{fold}.png")


def fig_attached(apo_numbers, results_dir, out):
    """Preserve-attached control: member skin frictions at the incoming
    station against the baseline and the measurement."""
    folds = apo_numbers.get("folds", {})
    rows = []
    for fold, block in folds.items():
        # transitive gate: a fold whose recorded member and target hashes
        # no longer match the caches on disk never renders
        if not fold_score_lineage_ok(results_dir, fold):
            print(f"skip fig6 rows for {fold} (fold-score lineage stale)")
            continue
        att = block.get("attached_control", {})
        for kind in KINDS:
            per_seed = att.get(kind, {}).get("per_seed", {})
            for ms in sorted(per_seed):
                if per_seed[ms].get("cf_members"):
                    rows.append((f"{fold} ms{ms}", kind,
                                 per_seed[ms]["cf_members"],
                                 att.get("cf_baseline")))
    if not rows:
        print("skip fig6 (no attached-control members)")
        return
    fig, ax = plt.subplots(figsize=(1.1 * len(rows) + 2.5, 4))
    for i, (fold, kind, cfs, cf_base) in enumerate(rows):
        x = np.full(len(cfs), i, dtype=float)
        x += np.linspace(-0.15, 0.15, len(cfs))
        ax.plot(x, 1e3 * np.asarray(cfs), "o", ms=4,
                color=KIND_COLORS[kind], alpha=0.7)
    if rows[0][3] is not None:
        ax.axhline(1e3 * rows[0][3], color="k", lw=1.0,
                   label="baseline solve")
    ax.axhline(1e3 * GATE_A_CF, color="tab:red", lw=1.0, ls="--",
               label="measured")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([f"{f}\n{k}" for f, k, _, _ in rows], fontsize=7)
    ax.set_ylabel(r"$10^3\, C_f$ at $x^*=-7.65$")
    ax.set_title("preserve-attached control")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_attached_control.png"), dpi=160)
    plt.close(fig)
    print("wrote fig6_attached_control.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/sbli")
    ap.add_argument("--out", default=os.path.join(
        _HERE, "..", "..", "UQ-RANS_research", "shock_interaction",
        "figures"))
    ap.add_argument("--n-members", type=int, default=24)
    ap.add_argument("--allow-partial", action="store_true",
                    dest="allow_partial",
                    help="render interim figures from unvalidated or "
                         "incomplete numbers (NEVER for the evidence "
                         "package; every panel is labeled by its inputs)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    records = _all_records()
    numbers = {}
    p = os.path.join(args.results, "apriori_numbers.json")
    if os.path.isfile(p):
        # evidence figures refuse invalid or incomplete numbers outright
        # (the review's completion finding); --allow-partial is the labeled
        # interim escape hatch
        from UQ.reproduce_sbli_apriori import validate_apriori_numbers
        ok, why = validate_apriori_numbers(args.results, strict=True)
        if ok or args.allow_partial:
            if not ok:
                print(f"NOTE: rendering PARTIAL evidence ({why}); never "
                      f"package these figures")
            numbers = json.load(open(p))
        else:
            print(f"apriori numbers refused ({why}); rerun with "
                  f"--allow-partial for labeled interim figures")
            sys.exit(1)
    apo_numbers = {}
    p = os.path.join(args.results, "aposteriori_numbers.json")
    if os.path.isfile(p):
        apo_numbers = json.load(open(p))

    fig_gates(numbers, records, args.results, args.out)
    fig_loso(numbers, args.out)
    fig_regions(numbers, args.out)
    for fold in FOLDS:
        fig_bands(records, args.results, fold, args.n_members, args.out)
    fig_attached(apo_numbers, args.results, args.out)
    print("figures in", args.out)


if __name__ == "__main__":
    main()
