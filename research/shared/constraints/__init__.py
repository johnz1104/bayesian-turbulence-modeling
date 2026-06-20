"""Physics constraints (core-v1.0): two separate entry points, realizability
projection and the Galilean-invariant integrity basis, never conflated."""

from .base import (
    BarycentricRealizability,
    GalileanInvariantBasis,
    IntegrityBasis,
    RealizabilityProjection,
)

__all__ = [
    "RealizabilityProjection",
    "BarycentricRealizability",
    "GalileanInvariantBasis",
    "IntegrityBasis",
]
