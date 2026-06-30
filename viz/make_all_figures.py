"""
Regenerate every figure from whatever artifacts are present in viz/artifacts/.

This does NOT run any calibration; it only plots existing real output.  Produce
the artifacts first:

    python3 viz/run_cpp_validation.py
    python3 viz/run_calibration.py a1_betaStar
    python3 viz/run_calibration.py near_wall4        # optional, richer sensitivity/identifiability
    python3 viz/run_surrogate_benchmark.py           # optional, acceleration number

then:

    python3 viz/make_all_figures.py
"""

import runpy
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from _common import have, ARTIFACTS


def _run(script, argv):
    sys.argv = [script] + argv
    print(f"[{script} {' '.join(argv)}]")
    runpy.run_path(str(_SCRIPT_DIR / script), run_name="__main__")


def main():
    if (ARTIFACTS / "cpp_validation.json").exists():
        _run("plot_validation.py", [])

    # Per-parameter-set figures for every calibrated set that exists.
    for ps in ("a1_betaStar", "near_wall4", "all11"):
        if have(ps):
            _run("plot_posterior_corner.py", [ps])
            _run("plot_convergence.py", [ps])
            _run("plot_surrogate.py", [ps])
            _run("plot_prior_posterior.py", [ps])

    # Sensitivity and identifiability default to the richest available set.
    _run("plot_sensitivity.py", [])
    _run("plot_identifiability.py", [])

    print("\nDone. Figures in viz/figures/.")


if __name__ == "__main__":
    main()
