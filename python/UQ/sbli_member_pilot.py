"""Compatibility entry point for the lineage-clean Phase-P SBLI pilot.

The implementation lives in reproduce_sbli_aposteriori so the pilot and
matrix share the same manifest, gate, Phase-A, target, restart, member and
cache-identity contracts. It runs the three registered pilot solves plus
the independent abort-off panel, writes ``aposteriori/pilot.json``, and
stops for review. It never authorizes or launches Phase M.

Usage (capture stdout: it is the verbose residual trace named by the pilot
artifact):

    QBTM_DNS_DATA=<root> PYTHONPATH=build:python \
      python3.11 python/UQ/sbli_member_pilot.py \
      --results results/sbli > phase_p.log 2>&1
"""
import sys

from UQ.reproduce_sbli_aposteriori import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--stage", "pilot"]
    main()
