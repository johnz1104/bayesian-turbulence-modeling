"""
PHASE 6 / V.5 driver — irreducible model-form discrepancy δ(x) along the wall.

Reconstructs the posterior-mean discrepancy at the wall observation locations from the
physical_gp KOH posterior, for cases with spatial Cf profiles (channel, flat_plate).

Usage:
    python3 discrepancy_study.py                 # channel + flat_plate
    python3 discrepancy_study.py flat_plate --variant 2   # in the evidence-preferred model
    python3 discrepancy_study.py --quick
"""

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT.parent.parent / "build"))
sys.path.insert(0, str(_SCRIPT.parent))

from run_discrepancy import run_case_discrepancy


def main():
    ap = argparse.ArgumentParser(description="Discrepancy δ(x) study (angle 1 / V.5)")
    ap.add_argument("cases", nargs="*", default=["channel", "flat_plate"])
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("-o", "--out-dir", default="results/discrepancy")
    args = ap.parse_args()
    # n_ens=150 + noise_floor=1e-2 is the V.1-validated trustworthy-surrogate config
    n_ens = 60 if args.quick else 150
    for case in args.cases:
        run_case_discrepancy(case, variant=args.variant, n_ens=n_ens,
                             out_dir=args.out_dir, noise_floor=1e-2)


if __name__ == "__main__":
    main()
