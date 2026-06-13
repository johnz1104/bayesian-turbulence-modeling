"""
PHASE 3 / V.4 driver — model evidence + Bayes factors for term-toggled closures.

Estimates log Z (TI + stepping-stone) for M_full / M_nolim / M_kw on the requested
validation case(s), with Bayes factors vs. M_full and the mandatory prior sweep.

Usage:
    python3 model_evidence_study.py                 # all three cases
    python3 model_evidence_study.py bfs             # one case
    python3 model_evidence_study.py --quick         # smaller ensembles
    python3 model_evidence_study.py -o results/model_evidence
"""

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT.parent.parent / "build"))
sys.path.insert(0, str(_SCRIPT.parent))

from model_evidence import run_case_evidence, summarize


def main():
    ap = argparse.ArgumentParser(description="Model-evidence study (angle 7)")
    ap.add_argument("cases", nargs="*", default=["channel", "flat_plate", "bfs"])
    ap.add_argument("--quick", action="store_true", help="smaller ensembles")
    ap.add_argument("-o", "--out-dir", default="results/model_evidence")
    args = ap.parse_args()

    # n_ens=150 + noise_floor=1e-2 is the V.1-validated trustworthy-surrogate config
    n_ens = 48 if args.quick else 150
    results = []
    for case in args.cases:
        print(f"\n===== model evidence: {case} =====", flush=True)
        results.append(run_case_evidence(case, n_ens=n_ens, out_dir=args.out_dir,
                                         noise_floor=1e-2))
    summarize(results)


if __name__ == "__main__":
    main()
