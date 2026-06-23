"""Benchmark loader interface (FROZEN, core-v1.0).

A Benchmark supplies high-fidelity truth for one case: the observation locations,
values, and uncertainties the likelihood compares predictions against, optional
mean fields, and the provenance metadata recorded in data/README.md (fidelity,
geometry, Mach, Reynolds number, normalization, separation/reattachment
definitions, coordinate origin, ...).

Thread A benchmarks are cheap synthetic truth (Kuramoto-Sivashinsky, Burgers,
Lorenz-96); Thread B benchmarks are frozen DNS / wall-resolved LES fields.
Loaders own fetching or regenerating from the gitignored data/ tree, keyed by the
manifest; no large artifact is committed (root CLAUDE.md working rules).
"""

from abc import ABC, abstractmethod

import numpy as np

# Provenance keys a Thread B benchmark should carry; mirrors data/README.md so a
# loader and the manifest cannot drift. Advisory, checked by required_provenance().
PROVENANCE_FIELDS = (
    "source", "fidelity", "geometry", "mach", "reynolds",
    "normalization", "reference_state", "separation_def", "reattachment_def",
    "coordinate_origin", "license",
)


class BenchmarkData:
    """Loaded truth for one case.

    Parameters
    ----------
    name : str
    observable : str
        Kind of the observations (for example "Cf", "St", "reattachment",
        "reynolds_stress", "energy_spectrum").
    locations : array (n,) or (n, d)
        Coordinates of the observations (for example x/h along a wall).
    values : array (n,)
        Truth values at those locations.
    sigmas : array (n,)
        Per-observation uncertainty (measurement + truth), strictly positive.
    fields : dict, optional
        Named mean fields / Reynolds stresses kept for a-priori tests.
    metadata : dict, optional
        Provenance, keyed as in data/README.md (see PROVENANCE_FIELDS).
    """

    def __init__(self, name, observable, locations, values, sigmas,
                 fields=None, metadata=None):
        self.name = str(name)
        self.observable = str(observable)
        loc = np.asarray(locations, dtype=np.float64)
        self.locations = loc.reshape(-1, 1) if loc.ndim == 1 else loc
        self.values = np.asarray(values, dtype=np.float64)
        self.sigmas = np.asarray(sigmas, dtype=np.float64)
        self.fields = dict(fields) if fields else {}
        self.metadata = dict(metadata) if metadata else {}
        n = len(self.values)
        if self.locations.shape[0] != n or self.sigmas.shape[0] != n:
            raise ValueError(
                "BenchmarkData: locations, values, sigmas must share length "
                f"({self.locations.shape[0]}, {n}, {self.sigmas.shape[0]})"
            )
        if np.any(self.sigmas <= 0.0) or not np.all(np.isfinite(self.sigmas)):
            raise ValueError("BenchmarkData: sigmas must be finite and positive")

    @property
    def n_obs(self) -> int:
        return len(self.values)

    def missing_provenance(self) -> list:
        """Provenance keys from PROVENANCE_FIELDS absent from metadata (advisory)."""
        return [k for k in PROVENANCE_FIELDS if k not in self.metadata]


class Benchmark(ABC):
    """A loadable benchmark case."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier; matches the data/README.md manifest entry."""

    @abstractmethod
    def load(self) -> BenchmarkData:
        """Fetch or regenerate the truth and return it as BenchmarkData."""

    @abstractmethod
    def observables(self) -> list:
        """Observable kinds this benchmark provides."""


class InMemoryBenchmark(Benchmark):
    """Benchmark backed by in-memory arrays.

    Used for synthetic Thread A truth and to wrap an already-built case's
    observations (for example a CaseSpec) without touching the data/ tree.
    """

    def __init__(self, data: BenchmarkData):
        self._data = data

    @property
    def name(self) -> str:
        return self._data.name

    def load(self) -> BenchmarkData:
        return self._data

    def observables(self) -> list:
        return [self._data.observable]
