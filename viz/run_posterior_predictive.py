"""
Posterior predictive check: re-run the real solver at a sample of posterior
coefficient vectors and compare the predicted skin friction against the Dean
(1978) observation the calibration targeted.  This is the honest end-to-end
check, with no surrogate in the loop: each prediction is a full CFD solve.

It is the one figure that needs fresh solves (about 18 s each), so it is a
separate, optional step.  Reads the real posterior from
viz/artifacts/a1_betaStar/chain.npz; writes predictions to
viz/artifacts/a1_betaStar/posterior_predictive.npz and the figure to
viz/figures/posterior_predictive_a1_betaStar.png.

Usage:
    python3 viz/run_posterior_predictive.py --n 24
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "build"))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "python"))

import numpy as np
import rans_sst_py as rs
from run_calibration import build_channel_case
from _common import set_style, save, plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=24, help="posterior samples to solve")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    art = _SCRIPT_DIR / "artifacts" / "a1_betaStar"
    if not (art / "chain.npz").exists():
        print("  run: python3 viz/run_calibration.py a1_betaStar")
        return

    flat = np.load(art / "chain.npz")["flat"]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(flat), size=args.n, replace=False)

    ps = rs.InferenceParameterSet.a1_betaStar()
    _, fm, cf_dean = build_channel_case(ps)

    preds = []
    for k, i in enumerate(idx):
        r = fm.evaluate(flat[i].tolist())
        if r.log_lik > -1e5 and len(r.predictions) > 0:
            preds.append(float(r.predictions[0]))   # predicted Cf at the station
        if (k + 1) % 6 == 0:
            print(f"  {k+1}/{args.n} solves")
    preds = np.array(preds)
    print(f"  {len(preds)} valid predictions; Cf_obs (Dean) = {cf_dean:.6f}")

    np.savez(art / "posterior_predictive.npz", cf_pred=preds, cf_obs=cf_dean)

    set_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.hist(preds, bins=12, color="#3b6ea5", alpha=0.7, label="posterior predictive $C_f$")
    ax.axvline(cf_dean, color="#c44", lw=2, label="Dean 1978 observation")
    lo, hi = np.percentile(preds, [2.5, 97.5])
    ax.axvspan(lo, hi, color="#3b6ea5", alpha=0.12, label="95% predictive interval")
    ax.set_xlabel("predicted skin friction $C_f$ (real CFD at posterior samples)")
    ax.set_ylabel("count")
    ax.set_title(f"Posterior predictive check - channel $C_f$\n"
                 f"mean {preds.mean():.5f}, obs {cf_dean:.5f}  (n={len(preds)} solves)")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "posterior_predictive_a1_betaStar.png")


if __name__ == "__main__":
    main()
