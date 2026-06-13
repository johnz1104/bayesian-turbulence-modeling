"""
PHASE 4 / V.3 driver — high-D identifiability (active subspace + ARD + posterior cov).

Usage:
    python3 identifiability_study.py                  # all three cases
    python3 identifiability_study.py bfs
    python3 identifiability_study.py --quick          # fewer FD gradients
"""

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT.parent.parent / "build"))
sys.path.insert(0, str(_SCRIPT.parent))

from run_identifiability import run_case_identifiability


def main():
    ap = argparse.ArgumentParser(description="Identifiability study (angle 3)")
    ap.add_argument("cases", nargs="*", default=["channel", "flat_plate", "bfs"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("-o", "--out-dir", default="results/identifiability")
    args = ap.parse_args()

    n_grad = 12 if args.quick else 24
    n_ens = 60 if args.quick else 120
    for case in args.cases:
        run_case_identifiability(case, n_grad=n_grad, n_ens=n_ens,
                                 out_dir=args.out_dir)


if __name__ == "__main__":
    main()
