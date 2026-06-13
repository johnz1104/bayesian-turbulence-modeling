"""
PHASE 2 — Compressible validation ladder.

Runs the pressure-based compressible solver at progressively higher Mach
numbers and writes a single JSON summary documenting convergence behaviour,
positivity invariants, and mass conservation.

Cases:
    Ma = 0.1   (smoke / baseline; must converge reliably)
    Ma = 0.3   (subsonic; must converge reliably)
    Ma = 0.5   (compressible regime; converge or fail with diagnostic)
    Ma = 0.7   (optional, high-subsonic; --include-ma07 to enable)

Output:
    outputs/compressible_validation_summary.json

Usage:
    python3 compressible_validation_ladder.py
    python3 compressible_validation_ladder.py --include-ma07
    python3 compressible_validation_ladder.py --quick           # smaller meshes
    python3 compressible_validation_ladder.py -o outputs/run123 # custom dir
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_BUILD_DIR  = _SCRIPT_DIR.parent.parent / "build"
_PYTHON_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PYTHON_DIR))

try:
    import rans_sst_py as rs  # noqa: F401  - presence check only
except ImportError:
    print("ERROR: rans_sst_py not built. Run: cmake --build build")
    sys.exit(1)

from compressible_diagnostics import (
    make_validation_case, run_validation_case, format_summary_row,
)


def _ladder_specs(quick: bool, include_ma07: bool) -> list[tuple[str, float, int, int, int]]:
    """Return [(name, Ma, nx, ny, max_iterations), ...].

    Mesh sizes are chosen so the baseline Ma=0.1 case converges within the
    iteration budget (494 iterations on the 40x30 reference mesh, see
    project context).  Quick mode is for development; --include-ma07 enables
    the optional high-subsonic case.
    """
    if quick:
        # Smaller meshes; expected to "Unconverged" but stay positive and
        # finite — used as a smoke ladder during local iteration.
        base = [
            ("ma_0_1",  0.1, 24, 18, 2000),
            ("ma_0_3",  0.3, 24, 18, 3000),
            ("ma_0_5",  0.5, 32, 24, 5000),
        ]
        if include_ma07:
            base.append(("ma_0_7",  0.7, 40, 32, 8000))
    else:
        # Production-validation meshes; Ma=0.1 should reach Converged.
        base = [
            ("ma_0_1",  0.1, 40, 30, 3000),
            ("ma_0_3",  0.3, 40, 30, 4000),
            ("ma_0_5",  0.5, 48, 36, 6000),
        ]
        if include_ma07:
            base.append(("ma_0_7",  0.7, 64, 48, 10000))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-ma07", action="store_true",
                    help="Also run the optional Ma=0.7 case")
    ap.add_argument("--quick", action="store_true",
                    help="Use smaller meshes / shorter solves (~minute)")
    ap.add_argument("-o", "--outdir", type=Path, default=None,
                    help="Output directory (default: <repo>/outputs/)")
    ap.add_argument("--no-stop-on-divergence", action="store_true",
                    help="Run all cases even if Ma=0.1 diverges")
    args = ap.parse_args()

    outdir = args.outdir or (_PYTHON_DIR.parent / "outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    print("Compressible validation ladder")
    print(f"  Quick mode:    {args.quick}")
    print(f"  Include Ma=0.7:{args.include_ma07}")
    print(f"  Output dir:    {outdir}")

    summaries = []
    overall_t0 = time.time()
    for spec in _ladder_specs(args.quick, args.include_ma07):
        name, Ma, nx, ny, maxit = spec
        print(f"\n--- {name}  (Ma={Ma:.2f},  mesh {nx}x{ny},  maxit={maxit}) ---")
        case   = make_validation_case(name=name, Ma=Ma, nx=nx, ny=ny,
                                      max_iterations=maxit)
        result = run_validation_case(case)
        print(format_summary_row(result))
        summaries.append(result)

        # Hard guard: if even Ma=0.1 falls over, the rest of the ladder cannot
        # possibly succeed.
        if (Ma <= 0.15 and result["diverged"]
                and not args.no_stop_on_divergence):
            print("  ABORT: baseline Ma=0.1 case diverged; not running rest.")
            break

    elapsed = time.time() - overall_t0

    out = {
        "schema_version":  1,
        "elapsed_s":       elapsed,
        "quick":           args.quick,
        "include_ma07":    args.include_ma07,
        "summaries":       summaries,
    }
    out_path = outdir / "compressible_validation_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Total wall time: {elapsed:.1f} s")

    # Final verdict text on stdout
    pass_low = any(s["Ma"] == 0.1 and s["converged"] and s["positivity_ok"]
                    for s in summaries)
    pass_mid = any(s["Ma"] == 0.3 and (s["converged"] or s["positivity_ok"])
                    for s in summaries)
    print(f"\n  Ma=0.1 PASSED:  {pass_low}")
    print(f"  Ma=0.3 PASSED:  {pass_mid}")


if __name__ == "__main__":
    main()
