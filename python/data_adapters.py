"""
POST-PHASE 3 — External high-Mach data adapter.

Provides source-agnostic loaders that convert tabular wall/profile data into
ObservationSet objects consumable by the Bayesian/KOH pipeline.

Supported formats
-----------------
CSV       : CSVLoader   — one observable per column, rows = spatial stations
JSON      : JSONLoader  — list of observation dicts or a {metadata, observations} blob
NPZ       : NPZLoader   — NumPy archive with named arrays

Adapter helpers
---------------
load_wall_data(path, ...)         - auto-detect format, return ObservationSet
load_profile_data(path, ...)      - load wall-normal profile
make_synthetic_high_mach_npz(...) - generate and save a synthetic .npz dataset
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from observation_schema import (
    CaseMetadata, CaseData, Observation, ObservationSet,
    GeometryType, WallCondition, DataSource, ObservableType,
)


# ---------------------------------------------------------------------------
# Known column-name → ObservableType map
# ---------------------------------------------------------------------------

_COL_TO_TYPE: Dict[str, ObservableType] = {
    # pressure / Cp — all aliases lower-cased at lookup time
    "cp":                    ObservableType.WALL_PRESSURE_CP,
    "cp_w":                  ObservableType.WALL_PRESSURE_CP,
    "wall_pressure_cp":      ObservableType.WALL_PRESSURE_CP,
    "pressure_coefficient":  ObservableType.WALL_PRESSURE_CP,
    "p":                     ObservableType.WALL_PRESSURE_CP,
    "pressure":              ObservableType.WALL_PRESSURE_CP,
    # skin friction
    "cf":                    ObservableType.SKIN_FRICTION_CF,
    "cf_w":                  ObservableType.SKIN_FRICTION_CF,
    "skin_friction_cf":      ObservableType.SKIN_FRICTION_CF,
    "skin_friction":         ObservableType.SKIN_FRICTION_CF,
    # heat flux
    "qw":                    ObservableType.WALL_HEAT_FLUX,
    "q_w":                   ObservableType.WALL_HEAT_FLUX,
    "wall_heat_flux":        ObservableType.WALL_HEAT_FLUX,
    "heat_flux":             ObservableType.WALL_HEAT_FLUX,
    "wall_heat_transfer":    ObservableType.WALL_HEAT_FLUX,
    # profiles
    "mach":                  ObservableType.MACH_PROFILE,
    "ma":                    ObservableType.MACH_PROFILE,
    "m":                     ObservableType.MACH_PROFILE,    # "M" → lower → "m"
    "mach_number":           ObservableType.MACH_PROFILE,
    "temperature":           ObservableType.TEMPERATURE_PROFILE,
    "t":                     ObservableType.TEMPERATURE_PROFILE,  # "T" → lower
    "wall_temperature":      ObservableType.TEMPERATURE_PROFILE,
    "t_w":                   ObservableType.TEMPERATURE_PROFILE,
    "velocity":              ObservableType.VELOCITY_PROFILE,
    "u":                     ObservableType.VELOCITY_U,
    # scalar locations
    "shock_location":        ObservableType.SHOCK_LOCATION,
    "separation_location":   ObservableType.SEPARATION_LOCATION,
    "reattachment":          ObservableType.REATTACHMENT_LOCATION,
    "xr":                    ObservableType.REATTACHMENT_LENGTH,
    "x_r":                   ObservableType.REATTACHMENT_LENGTH,
    "pressure_recovery":     ObservableType.PRESSURE_RECOVERY,
    "mass_flow":             ObservableType.MASS_FLOW_RATE,
}

_UNCERTAINTY_SUFFIXES = ("_sigma", "_unc", "_err", "_std", "_uncertainty")
_LOCATION_COLS = ("x", "x_loc", "xloc", "station", "x_h", "x_l",
                  "x_over_h", "x_mm", "x_m", "x/h", "x/d", "x/l",
                  "streamwise", "x_coord")
_YLOCATION_COLS = ("y", "y_loc", "yloc", "y_h", "y_l")


def _infer_obs_type(col_name: str) -> Optional[ObservableType]:
    key = col_name.strip().lower()
    if key in _COL_TO_TYPE:
        return _COL_TO_TYPE[key]
    # Allow prefix matching only for aliases with 3+ characters to avoid
    # single-char aliases (u, t, m, p) matching unrelated column names.
    for alias, ot in _COL_TO_TYPE.items():
        if len(alias) >= 3 and key.startswith(alias):
            return ot
    return None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DataLoader(ABC):
    """Abstract loader: subclasses implement ``load`` → ObservationSet."""

    @abstractmethod
    def load(
        self,
        path: Union[str, Path],
        case_id: str = "external",
        default_uncertainty_frac: float = 0.05,
        group: str = "",
        units: str = "",
    ) -> ObservationSet:
        """Load data from *path* and return an ObservationSet."""

    @staticmethod
    def _uncertainty_col(col_name: str, columns: Sequence[str]) -> Optional[str]:
        """Find an uncertainty column paired with a data column."""
        for suffix in _UNCERTAINTY_SUFFIXES:
            candidate = col_name + suffix
            if candidate in columns:
                return candidate
        return None


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

class CSVLoader(DataLoader):
    """Load wall or profile data from a CSV file.

    Expected format
    ---------------
    Header row with column names.  At minimum one location column (``x`` or
    ``x_loc``) and one or more observable columns recognised by the name map.

    Example header:
        x, cp, cp_sigma, cf, cf_sigma, qw, qw_sigma
        x, mach, mach_sigma, temperature, temperature_sigma

    Columns whose names cannot be mapped to an ObservableType are silently
    skipped (useful for ignoring auxiliary metadata columns).
    """

    def load(
        self,
        path: Union[str, Path],
        case_id: str = "external",
        default_uncertainty_frac: float = 0.05,
        group: str = "",
        units: str = "",
    ) -> ObservationSet:
        path = Path(path)
        rows: List[dict] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            col_names = reader.fieldnames or []
            for row in reader:
                rows.append({k: v.strip() for k, v in row.items()})

        # Identify location column
        x_col = next((c for c in col_names
                       if c.strip().lower() in _LOCATION_COLS), None)
        y_col = next((c for c in col_names
                       if c.strip().lower() in _YLOCATION_COLS), None)

        obs_set = ObservationSet(case_id)
        sigma_cols = {c for c in col_names
                      if any(c.lower().endswith(s) for s in _UNCERTAINTY_SUFFIXES)}

        # Identify observable columns (everything that is not location/sigma)
        obs_cols = [c for c in col_names
                    if c != x_col and c != y_col and c not in sigma_cols
                    and _infer_obs_type(c) is not None]
        if not obs_cols:
            recognised = [c for c in col_names if c != x_col and c != y_col
                          and c not in sigma_cols]
            raise ValueError(
                f"CSVLoader: no recognisable observable columns found in {path!r}. "
                f"Non-location columns present: {recognised}. "
                f"Recognised aliases include: cf, cp, qw, mach, temperature, "
                f"skin_friction, pressure_coefficient, heat_flux, etc."
            )

        for col in obs_cols:
            obs_type = _infer_obs_type(col)
            if obs_type is None:
                continue
            unc_col = self._uncertainty_col(col.lower(), [c.lower() for c in col_names])
            # Map unc_col back to original case
            unc_orig = next((c for c in col_names if c.lower() == unc_col), None) \
                if unc_col else None
            for row in rows:
                try:
                    val = float(row[col])
                except (ValueError, KeyError):
                    continue
                x_val = float(row[x_col]) if x_col and x_col in row else 0.0
                y_val = float(row[y_col]) if y_col and y_col in row else 0.0
                if unc_orig and unc_orig in row:
                    try:
                        sig = abs(float(row[unc_orig]))
                        if not (sig > 0):
                            sig = abs(val) * default_uncertainty_frac + 1e-10
                    except ValueError:
                        sig = abs(val) * default_uncertainty_frac + 1e-10
                else:
                    sig = abs(val) * default_uncertainty_frac + 1e-10
                obs_set.add(Observation(
                    observable_type=obs_type,
                    value=val, uncertainty=sig,
                    x=x_val, y=y_val,
                    units=units,
                    group=group or obs_type.value,
                    case_id=case_id,
                ))
        return obs_set


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------

class JSONLoader(DataLoader):
    """Load from a JSON file.

    Accepted formats
    ----------------
    1. Full CaseData dict: ``{"metadata": {...}, "observations": {...}}``
    2. ObservationSet dict: ``{"case_id": "...", "observations": [...]}``
    3. List of Observation dicts: ``[{"observable_type": "cf", ...}, ...]``
    4. Column-oriented dict with optional ``x`` key:
       ``{"x": [...], "cf": [...], "cf_sigma": [...]}``
    """

    def load(
        self,
        path: Union[str, Path],
        case_id: str = "external",
        default_uncertainty_frac: float = 0.05,
        group: str = "",
        units: str = "",
    ) -> ObservationSet:
        data = json.loads(Path(path).read_text())

        # Format 1: full CaseData
        if isinstance(data, dict) and "metadata" in data and "observations" in data:
            cd = CaseData.from_dict(data)
            return cd.observations

        # Format 2: ObservationSet dict
        if isinstance(data, dict) and "observations" in data \
                and isinstance(data["observations"], list) \
                and data["observations"] \
                and isinstance(data["observations"][0], dict) \
                and "observable_type" in data["observations"][0]:
            return ObservationSet.from_dict(data)

        # Format 3: list of Observation dicts
        if isinstance(data, list) and data and isinstance(data[0], dict) \
                and "observable_type" in data[0]:
            obs_set = ObservationSet(case_id)
            for d in data:
                obs_set.add(Observation.from_dict(d))
            return obs_set

        # Format 4: column-oriented
        if isinstance(data, dict):
            return self._load_column_oriented(
                data, case_id, default_uncertainty_frac, group, units
            )

        raise ValueError(f"JSONLoader: unrecognised format in {path}")

    @staticmethod
    def _load_column_oriented(
        data: dict, case_id: str,
        default_uncertainty_frac: float, group: str, units: str
    ) -> ObservationSet:
        xs = np.asarray(data.get("x", data.get("x_loc", [0.0])), dtype=float)
        ys = np.asarray(data.get("y", data.get("y_loc", [0.0] * len(xs))), dtype=float)
        if len(ys) == 1 and len(xs) > 1:
            ys = np.full(len(xs), ys[0])

        obs_set = ObservationSet(case_id)
        skip = {"x", "x_loc", "y", "y_loc"}
        sigma_keys = {k for k in data
                      if any(k.lower().endswith(s) for s in _UNCERTAINTY_SUFFIXES)}

        for col, arr in data.items():
            if col in skip or col in sigma_keys:
                continue
            obs_type = _infer_obs_type(col)
            if obs_type is None:
                continue
            vals = np.asarray(arr, dtype=float)
            unc_key = next((k for k in data if k.lower() == col.lower() + "_sigma"
                            or k.lower() == col.lower() + "_unc"), None)
            sigs = (np.asarray(data[unc_key], dtype=float) if unc_key
                    else np.abs(vals) * default_uncertainty_frac + 1e-10)
            for i, (val, sig) in enumerate(zip(vals, sigs)):
                x_val = xs[i] if i < len(xs) else 0.0
                y_val = ys[i] if i < len(ys) else 0.0
                obs_set.add(Observation(
                    observable_type=obs_type, value=float(val),
                    uncertainty=float(sig), x=float(x_val), y=float(y_val),
                    units=units, group=group or obs_type.value, case_id=case_id,
                ))
        return obs_set


# ---------------------------------------------------------------------------
# NPZ loader
# ---------------------------------------------------------------------------

class NPZLoader(DataLoader):
    """Load from a NumPy .npz archive.

    Expected keys
    -------------
    ``x``           : 1-D array of streamwise locations (required if wall data)
    ``y``           : 1-D array of wall-normal locations (optional)
    ``cp``          : Cp(x) values
    ``cp_sigma``    : Cp uncertainties (optional)
    ``cf``          : Cf(x) values
    ``cf_sigma``    : Cf uncertainties
    ``qw``          : qw(x) values
    ``qw_sigma``    : qw uncertainties
    ``mach``        : Mach profile M(y)
    (any other column name recognised by the _COL_TO_TYPE map)

    If ``*_sigma`` arrays are absent, uncertainty defaults to
    ``default_uncertainty_frac × |value| + 1e-10``.
    """

    def load(
        self,
        path: Union[str, Path],
        case_id: str = "external",
        default_uncertainty_frac: float = 0.05,
        group: str = "",
        units: str = "",
    ) -> ObservationSet:
        arc = np.load(str(path), allow_pickle=False)
        keys = list(arc.keys())

        xs = arc["x"].ravel() if "x" in keys else np.array([0.0])
        ys = arc["y"].ravel() if "y" in keys else np.zeros(len(xs))
        if len(ys) == 1 and len(xs) > 1:
            ys = np.full(len(xs), ys[0])

        sigma_keys = {k for k in keys
                      if any(k.lower().endswith(s) for s in _UNCERTAINTY_SUFFIXES)}
        skip = {"x", "y", "x_loc", "y_loc"} | sigma_keys

        obs_set = ObservationSet(case_id)
        for col in keys:
            if col in skip or col.startswith("_meta_"):
                continue
            obs_type = _infer_obs_type(col)
            if obs_type is None:
                continue
            vals = arc[col].ravel()
            unc_key = next((k for k in keys
                             if k.lower() in (col.lower() + "_sigma",
                                              col.lower() + "_unc",
                                              col.lower() + "_err")), None)
            sigs = (arc[unc_key].ravel() if unc_key
                    else np.abs(vals) * default_uncertainty_frac + 1e-10)
            for i, (val, sig) in enumerate(zip(vals, sigs)):
                x_val = xs[i] if i < len(xs) else 0.0
                y_val = ys[i] if i < len(ys) else 0.0
                obs_set.add(Observation(
                    observable_type=obs_type, value=float(val),
                    uncertainty=float(max(float(sig), 1e-10)),
                    x=float(x_val), y=float(y_val),
                    units=units, group=group or obs_type.value, case_id=case_id,
                ))
        return obs_set


# ---------------------------------------------------------------------------
# Auto-detect convenience wrapper
# ---------------------------------------------------------------------------

def load_wall_data(
    path: Union[str, Path],
    case_id:                  str   = "external",
    default_uncertainty_frac: float = 0.05,
    group:                    str   = "",
    units:                    str   = "",
) -> ObservationSet:
    """Load wall data from CSV, JSON, or NPZ; format detected from extension.

    Parameters
    ----------
    path : path-like
        Data file path.  Extension must be .csv, .json, or .npz.
    case_id : str
        Case identifier tag.
    default_uncertainty_frac : float
        Fractional uncertainty used when no ``*_sigma`` column is present.
    group : str
        KOH group tag applied to all loaded observations.
    units : str
        Units string applied to all loaded observations.

    Returns
    -------
    ObservationSet
    """
    path = Path(path)
    ext  = path.suffix.lower()
    loader: DataLoader
    if ext == ".csv":
        loader = CSVLoader()
    elif ext == ".json":
        loader = JSONLoader()
    elif ext == ".npz":
        loader = NPZLoader()
    else:
        raise ValueError(f"load_wall_data: unsupported extension {ext!r} "
                         f"(expected .csv, .json, .npz)")
    return loader.load(path, case_id=case_id,
                       default_uncertainty_frac=default_uncertainty_frac,
                       group=group, units=units)


def load_profile_data(
    path: Union[str, Path],
    case_id:                  str   = "external",
    default_uncertainty_frac: float = 0.05,
    group:                    str   = "profile",
) -> ObservationSet:
    """Convenience wrapper for profile data (wall-normal y column required)."""
    return load_wall_data(path, case_id=case_id,
                          default_uncertainty_frac=default_uncertainty_frac,
                          group=group)


# ---------------------------------------------------------------------------
# Synthetic high-Mach dataset generator
# ---------------------------------------------------------------------------

def make_synthetic_high_mach_npz(
    path: Union[str, Path],
    n_stations:  int   = 15,
    mach:        float = 2.0,
    reynolds:    float = 1e6,
    a1:          float = 0.31,
    beta_star:   float = 0.09,
    noise_frac:  float = 0.05,
    rng_seed:    int   = 42,
) -> Path:
    """Generate a synthetic high-Mach wall dataset and save as .npz.

    The dataset mimics Cp, Cf, qw distributions through a shock–boundary-layer
    interaction.  Parameters (a1, beta_star) control the distributions so the
    Bayesian pipeline can attempt to recover them.

    This is PURELY SYNTHETIC and does not represent any real flow.

    Parameters
    ----------
    path        : Output .npz file path.
    n_stations  : Number of wall stations.
    mach / reynolds : Flow conditions stored as metadata in the archive.
    a1 / beta_star  : True SST parameters used to generate data.
    noise_frac  : Fractional Gaussian noise added to each observable.
    rng_seed    : NumPy seed for reproducibility.

    Returns
    -------
    Path to the written .npz file.
    """
    rng = np.random.default_rng(rng_seed)
    xs  = np.linspace(0.0, 1.0, n_stations)
    x_shock = 0.40
    x_sep   = 0.35
    x_reat  = 0.65

    cp = (0.3 + 0.4 * a1 / 0.31) * (0.5 * (1 + np.tanh((xs - x_shock) / 0.05))) \
         - 0.05 * np.exp(-((xs - x_sep) / 0.05) ** 2)
    cf = 0.002 * beta_star / 0.09 \
         - 0.0015 * np.exp(-((xs - x_sep)  / 0.06) ** 2) \
         + 0.0010 * np.exp(-((xs - x_reat) / 0.08) ** 2)
    qw = 50_000.0 * beta_star / 0.09 \
         + 80_000.0 * np.exp(-((xs - x_reat) / 0.07) ** 2)

    def noisy(arr):
        sig = np.abs(arr) * noise_frac + 1e-6
        return arr + rng.standard_normal(len(arr)) * sig, sig

    cp_noisy, cp_sig = noisy(cp)
    cf_noisy, cf_sig = noisy(cf)
    qw_noisy, qw_sig = noisy(qw)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Metadata keys use _meta_ prefix so they are not matched as observables
    np.savez(
        str(path),
        x=xs,
        cp=cp_noisy, cp_sigma=cp_sig,
        cf=cf_noisy, cf_sigma=cf_sig,
        qw=qw_noisy, qw_sigma=qw_sig,
        _meta_mach=np.array([mach]),
        _meta_reynolds=np.array([reynolds]),
        _meta_a1_true=np.array([a1]),
        _meta_beta_star_true=np.array([beta_star]),
        _meta_x_shock=np.array([x_shock]),
        _meta_x_reat=np.array([x_reat]),
    )
    return path


def make_synthetic_high_mach_csv(
    path: Union[str, Path],
    n_stations: int   = 15,
    mach:       float = 2.0,
    a1:         float = 0.31,
    beta_star:  float = 0.09,
    noise_frac: float = 0.05,
    rng_seed:   int   = 42,
) -> Path:
    """Generate a synthetic high-Mach wall dataset and save as .csv."""
    rng = np.random.default_rng(rng_seed)
    xs  = np.linspace(0.0, 1.0, n_stations)
    x_shock, x_sep, x_reat = 0.40, 0.35, 0.65

    cp = (0.3 + 0.4 * a1 / 0.31) * (0.5 * (1 + np.tanh((xs - x_shock) / 0.05))) \
         - 0.05 * np.exp(-((xs - x_sep) / 0.05) ** 2)
    cf = 0.002 * beta_star / 0.09 \
         - 0.0015 * np.exp(-((xs - x_sep)  / 0.06) ** 2) \
         + 0.0010 * np.exp(-((xs - x_reat) / 0.08) ** 2)
    qw = 50_000.0 * beta_star / 0.09 \
         + 80_000.0 * np.exp(-((xs - x_reat) / 0.07) ** 2)

    def noisy(arr):
        sig = np.abs(arr) * noise_frac + 1e-6   # 1e-6 floor survives .8f formatting
        return arr + rng.standard_normal(len(arr)) * sig, sig

    cp_n, cp_s = noisy(cp)
    cf_n, cf_s = noisy(cf)
    qw_n, qw_s = noisy(qw)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "cp", "cp_sigma", "cf", "cf_sigma", "qw", "qw_sigma"])
        for i in range(n_stations):
            writer.writerow([
                f"{xs[i]:.6f}", f"{cp_n[i]:.8f}", f"{cp_s[i]:.8f}",
                f"{cf_n[i]:.8f}", f"{cf_s[i]:.8f}",
                f"{qw_n[i]:.4f}", f"{qw_s[i]:.4f}",
            ])
    return path
