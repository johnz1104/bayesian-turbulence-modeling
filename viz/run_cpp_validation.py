"""
Run the C++ solver's built-in validations and capture the reported numbers
into a JSON artifact the validation figure consumes.

The solver CLI prints the simulated skin-friction coefficient alongside the
empirical correlation it is validated against (Dean 1978 for the channel,
Schoenherr for the flat plate).  This script invokes the CLI, parses those
numbers, and writes them to ``viz/artifacts/cpp_validation.json``.  No value is
hand-transcribed: the figure is built from exactly what the solver reported on
this machine.

Usage:
    python3 viz/run_cpp_validation.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_CLI = _REPO_ROOT / "build" / "rans_sst"


def _run(flag):
    out = subprocess.run([str(_CLI), flag], capture_output=True, text=True).stdout
    return out


def parse_channel(text):
    sim = float(re.search(r"Cf \(simulation\):\s*([0-9.eE+-]+)", text).group(1))
    ref = float(re.search(r"Cf \(Dean 1978\):\s*([0-9.eE+-]+)", text).group(1))
    err = float(re.search(r"Relative error:\s*([0-9.eE+-]+)%", text).group(1))
    return {"case": "Turbulent channel (Re_b=6800)", "reference": "Dean 1978",
            "cf_sim": sim, "cf_ref": ref, "rel_error_pct": err}


def parse_plate(text):
    sim = float(re.search(r"Cf \(simulation\):\s*([0-9.eE+-]+)", text).group(1))
    ref = float(re.search(r"Cf \(Schoenherr\):\s*([0-9.eE+-]+)", text).group(1))
    err = float(re.search(r"Relative error:\s*([0-9.eE+-]+)%", text).group(1))
    rex = re.search(r"Re_x:\s*([0-9.eE+-]+)", text)
    return {"case": "Flat-plate boundary layer (x/L=0.8)", "reference": "Schoenherr",
            "cf_sim": sim, "cf_ref": ref, "rel_error_pct": err,
            "Re_x": float(rex.group(1)) if rex else None}


def main():
    if not _CLI.exists():
        print(f"ERROR: solver CLI not found at {_CLI}. Build first:")
        print("  cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON && cmake --build build -j")
        sys.exit(1)

    results = {"channel": parse_channel(_run("--validate-channel")),
               "plate": parse_plate(_run("--validate-plate"))}

    out_dir = _SCRIPT_DIR / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cpp_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    for v in results.values():
        print(f"  {v['case']:42s}  Cf_sim={v['cf_sim']:.6f}  "
              f"Cf_ref={v['cf_ref']:.6f}  err={v['rel_error_pct']:.2f}%  (vs {v['reference']})")
    print(f"\n  artifact -> {out_dir / 'cpp_validation.json'}")


if __name__ == "__main__":
    main()
