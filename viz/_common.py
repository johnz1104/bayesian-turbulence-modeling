"""
Shared helpers for the QBTM figure scripts: artifact paths, loaders, and a
consistent matplotlib style.  Every figure script reads from ``viz/artifacts/``
(produced by ``run_calibration.py`` / ``run_cpp_validation.py``) and writes to
``viz/figures/``.  Nothing here generates data; it only loads and styles.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS = SCRIPT_DIR / "artifacts"
FIGURES = SCRIPT_DIR / "figures"

# Prior standard deviation as a fraction of the mean (matches
# make_prior_from_param_set's default relative_std).
PRIOR_REL_STD = 0.15


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "figure.facecolor": "white",
    })


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_summary(param_set):
    return load_json(ARTIFACTS / param_set / "summary.json")


def load_chain(param_set):
    return np.load(ARTIFACTS / param_set / "chain.npz")


def load_holdout(param_set):
    return np.load(ARTIFACTS / param_set / "surrogate_holdout.npz")


def have(param_set):
    return (ARTIFACTS / param_set / "summary.json").exists()


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / name
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    return out
