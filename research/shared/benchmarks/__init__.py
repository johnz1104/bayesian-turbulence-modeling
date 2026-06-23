"""Benchmark loader interface (core-v1.0): truth + provenance for one case."""

from .base import (
    PROVENANCE_FIELDS,
    Benchmark,
    BenchmarkData,
    InMemoryBenchmark,
)

__all__ = [
    "Benchmark",
    "BenchmarkData",
    "InMemoryBenchmark",
    "PROVENANCE_FIELDS",
]
