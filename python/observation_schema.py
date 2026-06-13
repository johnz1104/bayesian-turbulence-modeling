"""
POST-PHASE 2 — Generalized scramjet-relevant observation schema.

Provides a solver-agnostic data model for representing calibration targets
across incompressible, low-Mach, and (future) high-Mach scramjet cases.

Classes
-------
GeometryType   - enum of supported geometry categories
WallCondition  - enum of wall thermal boundary conditions
DataSource     - enum of data provenance labels
ObservableType - enum of measurable quantities
CaseMetadata   - case-level metadata (geometry, Mach, Re, stagnation state, etc.)
Observation    - one scalar observable with location, uncertainty, and group tags
ObservationSet - ordered collection of Observation objects with query helpers
CaseData       - top-level container pairing CaseMetadata with ObservationSet

Factory helpers
---------------
bfs_to_observation_set(cal, ...)   - convert BFSSyntheticCalibration obs to schema
channel_to_observation_set(...)    - convert channel Cf obs to schema
koh_from_observation_set(obs_set)  - build KOHLikelihood from an ObservationSet
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class GeometryType(str, Enum):
    CHANNEL            = "channel"
    FLAT_PLATE         = "flat_plate"
    BACKWARD_FACING_STEP = "backward_facing_step"
    RAMP               = "ramp"
    SCRAMJET_INLET     = "scramjet_inlet"
    SCRAMJET_COMBUSTOR = "scramjet_combustor"
    NOZZLE             = "nozzle"
    GENERIC            = "generic"


class WallCondition(str, Enum):
    ADIABATIC    = "adiabatic"
    ISOTHERMAL   = "isothermal"
    COOLED_WALL  = "cooled_wall"
    HEATED_WALL  = "heated_wall"


class DataSource(str, Enum):
    INTERNAL_INCOMPRESSIBLE = "internal_incompressible"
    INTERNAL_LOW_MACH       = "internal_low_mach"
    EXTERNAL_CFD            = "external_cfd"
    EXPERIMENT              = "experiment"
    SYNTHETIC               = "synthetic"
    PRECOMPUTED_ENSEMBLE    = "precomputed_ensemble"


class ObservableType(str, Enum):
    # Wall-distributed quantities (have streamwise x-location)
    WALL_PRESSURE_CP        = "wall_pressure_cp"
    SKIN_FRICTION_CF        = "skin_friction_cf"
    WALL_HEAT_FLUX          = "wall_heat_flux"
    # Scalar locations
    SHOCK_LOCATION          = "shock_location"
    SEPARATION_LOCATION     = "separation_location"
    REATTACHMENT_LOCATION   = "reattachment_location"
    REATTACHMENT_LENGTH     = "reattachment_length"  # legacy BFS alias
    # Scalar integrals / bulk
    PRESSURE_RECOVERY       = "pressure_recovery"
    MASS_FLOW_RATE          = "mass_flow_rate"
    DRAG                    = "drag"
    # Profile quantities (have wall-normal y-location)
    MACH_PROFILE            = "mach_profile"
    TEMPERATURE_PROFILE     = "temperature_profile"
    VELOCITY_PROFILE        = "velocity_profile"
    VELOCITY_U              = "velocity_u"

    @property
    def is_wall_distributed(self) -> bool:
        return self in {
            ObservableType.WALL_PRESSURE_CP,
            ObservableType.SKIN_FRICTION_CF,
            ObservableType.WALL_HEAT_FLUX,
        }

    @property
    def is_profile(self) -> bool:
        return self in {
            ObservableType.MACH_PROFILE,
            ObservableType.TEMPERATURE_PROFILE,
            ObservableType.VELOCITY_PROFILE,
            ObservableType.VELOCITY_U,
        }

    @property
    def is_scalar_location(self) -> bool:
        return self in {
            ObservableType.SHOCK_LOCATION,
            ObservableType.SEPARATION_LOCATION,
            ObservableType.REATTACHMENT_LOCATION,
            ObservableType.REATTACHMENT_LENGTH,
        }


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CaseMetadata:
    """Case-level descriptors independent of the observation values."""
    case_id:               str
    geometry_type:         GeometryType
    mach:                  float
    reynolds:              float
    stagnation_pressure:   Optional[float]   = None   # Pa
    stagnation_temperature: Optional[float]  = None   # K
    wall_condition:        WallCondition     = WallCondition.ADIABATIC
    fuel_on:               bool              = False
    data_source:           DataSource        = DataSource.SYNTHETIC
    notes:                 str               = ""

    def to_dict(self) -> dict:
        return {
            "case_id":               self.case_id,
            "geometry_type":         self.geometry_type.value,
            "mach":                  self.mach,
            "reynolds":              self.reynolds,
            "stagnation_pressure":   self.stagnation_pressure,
            "stagnation_temperature": self.stagnation_temperature,
            "wall_condition":        self.wall_condition.value,
            "fuel_on":               self.fuel_on,
            "data_source":           self.data_source.value,
            "notes":                 self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CaseMetadata":
        return cls(
            case_id               = d["case_id"],
            geometry_type         = GeometryType(d["geometry_type"]),
            mach                  = float(d["mach"]),
            reynolds              = float(d["reynolds"]),
            stagnation_pressure   = d.get("stagnation_pressure"),
            stagnation_temperature = d.get("stagnation_temperature"),
            wall_condition        = WallCondition(d.get("wall_condition", "adiabatic")),
            fuel_on               = bool(d.get("fuel_on", False)),
            data_source           = DataSource(d.get("data_source", "synthetic")),
            notes                 = d.get("notes", ""),
        )


@dataclass
class Observation:
    """One scalar observable with full metadata."""
    observable_type:      ObservableType
    value:                float
    uncertainty:          float        # 1-sigma
    x:                    float = 0.0  # streamwise location (nondim by ref length)
    y:                    float = 0.0  # wall-normal location (nondim)
    reference_length:     float = 1.0
    reference_velocity:   Optional[float] = None
    reference_pressure:   Optional[float] = None
    reference_temperature: Optional[float] = None
    units:                str   = ""
    group:                str   = ""   # KOH discrepancy group tag
    case_id:              str   = ""

    def to_dict(self) -> dict:
        return {
            "observable_type":      self.observable_type.value,
            "value":                self.value,
            "uncertainty":          self.uncertainty,
            "x":                    self.x,
            "y":                    self.y,
            "reference_length":     self.reference_length,
            "reference_velocity":   self.reference_velocity,
            "reference_pressure":   self.reference_pressure,
            "reference_temperature": self.reference_temperature,
            "units":                self.units,
            "group":                self.group,
            "case_id":              self.case_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        return cls(
            observable_type      = ObservableType(d["observable_type"]),
            value                = float(d["value"]),
            uncertainty          = float(d["uncertainty"]),
            x                    = float(d.get("x", 0.0)),
            y                    = float(d.get("y", 0.0)),
            reference_length     = float(d.get("reference_length", 1.0)),
            reference_velocity   = d.get("reference_velocity"),
            reference_pressure   = d.get("reference_pressure"),
            reference_temperature = d.get("reference_temperature"),
            units                = d.get("units", ""),
            group                = d.get("group", ""),
            case_id              = d.get("case_id", ""),
        )


class ObservationSet:
    """Ordered collection of Observation objects with query and conversion helpers.

    Parameters
    ----------
    case_id : str
        Identifier that must match the parent CaseMetadata.

    Notes
    -----
    KOH spatial discrepancy uses ``locations_x()`` / ``locations_xy()`` to build
    the kernel matrix; set the ``x`` (and optionally ``y``) fields on each
    Observation to enable physical-GP mode.
    """

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self._obs: List[Observation] = []

    # ---- construction helpers ---------------------------------------------

    def add(self, obs: Observation) -> "ObservationSet":
        """Append one observation. Returns self for chaining."""
        self._obs.append(obs)
        return self

    def add_all(self, observations: Sequence[Observation]) -> "ObservationSet":
        for o in observations:
            self._obs.append(o)
        return self

    # ---- bulk accessors ---------------------------------------------------

    @property
    def n_obs(self) -> int:
        return len(self._obs)

    def __len__(self) -> int:
        return len(self._obs)

    def __iter__(self):
        return iter(self._obs)

    def __getitem__(self, idx):
        return self._obs[idx]

    def values(self) -> np.ndarray:
        return np.array([o.value for o in self._obs])

    def uncertainties(self) -> np.ndarray:
        return np.array([o.uncertainty for o in self._obs])

    def locations_x(self) -> np.ndarray:
        """Streamwise x-locations for all observations (nondim)."""
        return np.array([o.x for o in self._obs])

    def locations_y(self) -> np.ndarray:
        """Wall-normal y-locations for all observations (nondim)."""
        return np.array([o.y for o in self._obs])

    def locations_xy(self) -> np.ndarray:
        """Shape (n, 2) array of (x, y) for 2-D GP kernels."""
        return np.column_stack([self.locations_x(), self.locations_y()])

    def types(self) -> List[ObservableType]:
        return [o.observable_type for o in self._obs]

    def groups(self) -> List[str]:
        return [o.group for o in self._obs]

    # ---- filtering --------------------------------------------------------

    def filter_by_type(self, obs_type: ObservableType) -> "ObservationSet":
        out = ObservationSet(self.case_id)
        out._obs = [o for o in self._obs if o.observable_type == obs_type]
        return out

    def filter_by_group(self, group: str) -> "ObservationSet":
        out = ObservationSet(self.case_id)
        out._obs = [o for o in self._obs if o.group == group]
        return out

    def filter_by_types(self, obs_types: Sequence[ObservableType]) -> "ObservationSet":
        types_set = set(obs_types)
        out = ObservationSet(self.case_id)
        out._obs = [o for o in self._obs if o.observable_type in types_set]
        return out

    # ---- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "case_id":      self.case_id,
            "observations": [o.to_dict() for o in self._obs],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ObservationSet":
        obs_set = cls(d["case_id"])
        for od in d.get("observations", []):
            obs_set.add(Observation.from_dict(od))
        return obs_set

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path) -> "ObservationSet":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---- KOH compatibility ------------------------------------------------

    def to_koh_arrays(self, use_xy: bool = False):
        """Return (locations, values, sigmas) arrays consumed by KOHLikelihood.

        Parameters
        ----------
        use_xy : bool
            If True, return (n,2) xy locations for 2-D GP kernels.
            If False (default), return (n,) x-only locations.

        Returns
        -------
        locations : ndarray, shape (n,) or (n,2)
        values    : ndarray, shape (n,)
        sigmas    : ndarray, shape (n,)
        """
        locs = self.locations_xy() if use_xy else self.locations_x()
        return locs, self.values(), self.uncertainties()

    def __repr__(self) -> str:
        type_counts: dict = {}
        for o in self._obs:
            type_counts[o.observable_type.value] = \
                type_counts.get(o.observable_type.value, 0) + 1
        parts = ", ".join(f"{k}×{v}" for k, v in type_counts.items())
        return f"ObservationSet(case_id={self.case_id!r}, n={self.n_obs}, [{parts}])"


@dataclass
class CaseData:
    """Top-level container: one CaseMetadata + one ObservationSet."""
    metadata:     CaseMetadata
    observations: ObservationSet

    @property
    def case_id(self) -> str:
        return self.metadata.case_id

    def to_dict(self) -> dict:
        return {
            "metadata":     self.metadata.to_dict(),
            "observations": self.observations.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CaseData":
        return cls(
            metadata     = CaseMetadata.from_dict(d["metadata"]),
            observations = ObservationSet.from_dict(d["observations"]),
        )

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path) -> "CaseData":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Factory helpers — convert existing calibration objects to schema
# ---------------------------------------------------------------------------

def bfs_to_observation_set(
    xr_h: float,
    xr_sigma: float,
    cf_stations: Sequence[float],
    cf_values: Sequence[float],
    cf_sigmas: Sequence[float],
    Re_h: float = 37600.0,
    h_s: float  = 1.0,
    case_id: str = "bfs_DS1985",
) -> ObservationSet:
    """Build an ObservationSet for a BFS calibration case.

    Parameters
    ----------
    xr_h       : reattachment length x_r/h (nondim)
    xr_sigma   : uncertainty on x_r/h
    cf_stations: list of x/h locations where Cf is observed
    cf_values  : Cf at each station
    cf_sigmas  : Cf measurement uncertainty at each station

    Returns
    -------
    ObservationSet with one REATTACHMENT_LENGTH + N SKIN_FRICTION_CF observations.
    """
    obs_set = ObservationSet(case_id)
    obs_set.add(Observation(
        observable_type  = ObservableType.REATTACHMENT_LENGTH,
        value            = float(xr_h),
        uncertainty      = float(xr_sigma),
        x                = float(xr_h),   # location estimate = value
        reference_length = float(h_s),
        units            = "x/h",
        group            = "reattachment",
        case_id          = case_id,
    ))
    for x_loc, cf_val, cf_sig in zip(cf_stations, cf_values, cf_sigmas):
        obs_set.add(Observation(
            observable_type  = ObservableType.SKIN_FRICTION_CF,
            value            = float(cf_val),
            uncertainty      = float(cf_sig),
            x                = float(x_loc),
            reference_length = float(h_s),
            units            = "-",
            group            = "cf_wall",
            case_id          = case_id,
        ))
    return obs_set


def channel_to_observation_set(
    cf_stations: Sequence[float],
    cf_values:   Sequence[float],
    cf_sigmas:   Sequence[float],
    mach:        float = 0.0,
    reynolds:    float = 6800.0,
    case_id:     str   = "channel",
) -> ObservationSet:
    """Build an ObservationSet for a channel-flow calibration case."""
    obs_set = ObservationSet(case_id)
    for x_loc, cf_val, cf_sig in zip(cf_stations, cf_values, cf_sigmas):
        obs_set.add(Observation(
            observable_type  = ObservableType.SKIN_FRICTION_CF,
            value            = float(cf_val),
            uncertainty      = float(cf_sig),
            x                = float(x_loc),
            units            = "-",
            group            = "cf_wall",
            case_id          = case_id,
        ))
    return obs_set


def koh_from_observation_set(obs_set: ObservationSet,
                              mode: str = "physical_gp",
                              use_xy: bool = False):
    """Construct a KOHLikelihood directly from an ObservationSet.

    Imports KOHLikelihood lazily to avoid a hard dependency on bayesian_inference
    at module import time.

    Parameters
    ----------
    obs_set : ObservationSet
    mode    : "diagonal" or "physical_gp"
    use_xy  : If True, use 2-D (x,y) kernel locations; otherwise x-only.

    Returns
    -------
    KOHLikelihood instance
    """
    from bayesian_inference import KOHLikelihood  # lazy import
    locs, vals, sigs = obs_set.to_koh_arrays(use_xy=use_xy)
    return KOHLikelihood(locs, vals, sigs, mode=mode)


def scramjet_synthetic_observation_set(
    n_wall_stations: int = 10,
    x_range: tuple = (0.0, 1.0),
    mach: float = 2.0,
    reynolds: float = 1e6,
    case_id: str = "scramjet_synthetic",
    noise_frac: float = 0.05,
    rng_seed: int = 42,
) -> tuple:
    """Generate a synthetic scramjet-like wall observation set.

    Produces Cp(x), Cf(x), and qw(x) at evenly spaced wall stations using a
    parametric model inspired by shock-boundary-layer interaction physics:
    - Cp rises through a simulated shock, then plateaus
    - Cf drops in the separation bubble, recovers downstream
    - qw peaks at reattachment

    This is PURELY SYNTHETIC — it does not claim to represent any real
    scramjet flow.  It serves as a test bed for the Bayesian pipeline before
    real high-Mach data is available.

    Returns
    -------
    obs_set  : ObservationSet  with Cp, Cf, qw observations
    truth    : dict            with the "true" parameter values used
    metadata : CaseMetadata
    """
    rng = np.random.default_rng(rng_seed)
    xs = np.linspace(x_range[0], x_range[1], n_wall_stations)

    # Parametric model: shock at x_shock = 0.4, reattachment at x_r = 0.65
    x_shock = 0.40
    x_sep   = 0.35
    x_reat  = 0.65

    def _cp(x, a1, beta_star):
        """Cp rises through shock, controlled by a1."""
        shock_strength = 0.5 * (1.0 + np.tanh((x - x_shock) / 0.05))
        sep_dip        = -0.05 * np.exp(-((x - x_sep) / 0.05) ** 2)
        return (0.3 + 0.4 * a1 / 0.31) * shock_strength + sep_dip

    def _cf(x, a1, beta_star):
        """Cf drops in separation, recovers after reattachment."""
        base  = 0.002 * beta_star / 0.09
        sep   = -0.0015 * np.exp(-((x - x_sep) / 0.06) ** 2)
        reat  = 0.0010 * np.exp(-((x - x_reat) / 0.08) ** 2)
        return base + sep + reat

    def _qw(x, a1, beta_star):
        """qw peaks at reattachment."""
        base  = 50_000.0 * beta_star / 0.09
        peak  = 80_000.0 * np.exp(-((x - x_reat) / 0.07) ** 2)
        return base + peak

    # "True" parameters (perturbed from Menter defaults)
    a1_true       = 0.31 * (1.0 + 0.10 * rng.standard_normal())
    betaStar_true = 0.09 * (1.0 + 0.10 * rng.standard_normal())
    truth = {"a1": float(a1_true), "betaStar": float(betaStar_true),
             "x_shock": x_shock, "x_sep": x_sep, "x_reat": x_reat,
             "mach": mach, "reynolds": reynolds}

    obs_set = ObservationSet(case_id)

    for x in xs:
        cp_val = float(_cp(x, a1_true, betaStar_true))
        cf_val = float(_cf(x, a1_true, betaStar_true))
        qw_val = float(_qw(x, a1_true, betaStar_true))

        # Add noise
        cp_sig = abs(cp_val) * noise_frac + 1e-4
        cf_sig = abs(cf_val) * noise_frac + 1e-6
        qw_sig = abs(qw_val) * noise_frac + 100.0

        obs_set.add(Observation(
            observable_type = ObservableType.WALL_PRESSURE_CP,
            value           = cp_val + float(rng.standard_normal() * cp_sig),
            uncertainty     = cp_sig,
            x               = float(x),
            units           = "-",
            group           = "cp_wall",
            case_id         = case_id,
        ))
        obs_set.add(Observation(
            observable_type = ObservableType.SKIN_FRICTION_CF,
            value           = cf_val + float(rng.standard_normal() * cf_sig),
            uncertainty     = cf_sig,
            x               = float(x),
            units           = "-",
            group           = "cf_wall",
            case_id         = case_id,
        ))
        obs_set.add(Observation(
            observable_type = ObservableType.WALL_HEAT_FLUX,
            value           = qw_val + float(rng.standard_normal() * qw_sig),
            uncertainty     = qw_sig,
            x               = float(x),
            units           = "W/m^2",
            group           = "qw_wall",
            case_id         = case_id,
        ))

    # Add scalar location observations
    obs_set.add(Observation(
        observable_type = ObservableType.SHOCK_LOCATION,
        value           = x_shock + float(rng.standard_normal() * 0.01),
        uncertainty     = 0.01,
        x               = x_shock,
        units           = "x/L",
        group           = "locations",
        case_id         = case_id,
    ))
    obs_set.add(Observation(
        observable_type = ObservableType.REATTACHMENT_LOCATION,
        value           = x_reat + float(rng.standard_normal() * 0.02),
        uncertainty     = 0.02,
        x               = x_reat,
        units           = "x/L",
        group           = "locations",
        case_id         = case_id,
    ))

    metadata = CaseMetadata(
        case_id               = case_id,
        geometry_type         = GeometryType.SCRAMJET_INLET,
        mach                  = mach,
        reynolds              = reynolds,
        stagnation_pressure   = 101325.0 * (1 + 0.4 * mach**2) ** 3.5,
        stagnation_temperature = 300.0  * (1 + 0.2 * mach**2),
        wall_condition        = WallCondition.ADIABATIC,
        data_source           = DataSource.SYNTHETIC,
        notes                 = "Parametric synthetic SBLI case — not a real flow",
    )

    return obs_set, truth, metadata
