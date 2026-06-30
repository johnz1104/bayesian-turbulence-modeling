"""
Validation figure: simulated skin friction against the empirical correlation
the solver is validated on, for the channel (Dean 1978) and flat plate
(Schoenherr).  Source: viz/artifacts/cpp_validation.json (the solver CLI's own
reported numbers, captured by run_cpp_validation.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _common import ARTIFACTS, load_json, set_style, save, plt


def main():
    path = ARTIFACTS / "cpp_validation.json"
    if not path.exists():
        print(f"  missing {path}; run: python3 viz/run_cpp_validation.py")
        return
    data = load_json(path)
    set_style()

    cases = [data["channel"], data["plate"]]
    labels = ["Channel\n(vs Dean 1978)", "Flat plate\n(vs Schoenherr)"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(cases))
    w = 0.36
    sim = [c["cf_sim"] for c in cases]
    ref = [c["cf_ref"] for c in cases]

    b1 = ax.bar(x - w / 2, sim, w, label="SST simulation", color="#3b6ea5")
    b2 = ax.bar(x + w / 2, ref, w, label="Reference correlation", color="#9c9c9c")

    for c, xi in zip(cases, x):
        top = max(c["cf_sim"], c["cf_ref"])
        ax.annotate(f"{c['rel_error_pct']:.1f}% error",
                    xy=(xi, top), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Skin-friction coefficient $C_f$")
    ax.set_title("Solver validation: SST $C_f$ vs empirical correlations")
    ax.legend(frameon=False)
    ax.bar_label(b1, fmt="%.5f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.5f", padding=2, fontsize=8)
    ax.set_ylim(0, max(max(sim), max(ref)) * 1.25)

    save(fig, "validation_cf.png")


if __name__ == "__main__":
    main()
