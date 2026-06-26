"""Generate and cache one Reynolds number's calibration ensemble.

A thin wrapper so the five per-Re ensembles (the expensive forward-solve step of
reproduce_channel.py) can be generated in parallel, one process per Re, then the
reproduce script runs the analysis from the caches. Uses the same CONFIG as
reproduce_channel so the caches are interchangeable.

Usage: PYTHONPATH=build:python python3 python/UQ/gen_one_ensemble.py <re_tau_nominal>
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "build"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from UQ.datasets import ChannelDNS
from UQ.datasets.channel_calibration import ChannelCalibration
from UQ.reproduce_channel import CONFIG, OUT


def main():
    n = int(sys.argv[1])
    os.makedirs(OUT, exist_ok=True)
    dns = ChannelDNS.load(n)
    c = ChannelCalibration(dns, param_set=CONFIG["param_set"],
                           n_stations=CONFIG["n_stations"], cfg=CONFIG["cfg"],
                           sigma_floor=CONFIG["sigma_floor"])
    nv = c.run_ensemble(n=CONFIG["n_ensemble"], seed=CONFIG["seed"], verbose=True)
    path = os.path.join(OUT, f"ensemble_{n}.npz")
    np.savez(path, **c.to_cache())
    print(f"[Re_tau {n}] {nv}/{CONFIG['n_ensemble']} valid -> {path}", flush=True)


if __name__ == "__main__":
    main()
