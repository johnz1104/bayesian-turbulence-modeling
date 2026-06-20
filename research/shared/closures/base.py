"""Closure abstract base class (FROZEN, core-v1.0).

A Closure is a turbulence-closure model. It exposes the inference handshake
(parameter spec + prior + a likelihood hook bound to data) and declares its
structural character, so a study can compare a memoryless, deterministic, local
baseline (SST) against a generalized Langevin closure that adds memory, a
stochastic forcing, and non-locality.

The physics constraints a closure needs (the realizability projection and the
Galilean-invariant basis) are kept in a SEPARATE module, research.shared.
constraints, and are not methods on the closure: the two constraints are
enforced separately and must not be conflated (see the root CLAUDE.md, and
research.shared.constraints.base for why conflating them diverges the solver).
"""

from abc import ABC, abstractmethod

from ..inference.handshake import ClosurePrior, Likelihood, ParameterSpec


class Closure(ABC):
    """Abstract closure exposing the frozen inference handshake.

    Structural-character flags default to the SST baseline (all False). A
    Mori-Zwanzig / generalized Langevin closure overrides them: has_memory marks
    a memory integral, is_stochastic marks a fluctuating forcing, is_nonlocal
    marks dependence on the strain history along a streamline (Thread B) or in
    time (Thread A).
    """

    has_memory: bool = False
    is_stochastic: bool = False
    is_nonlocal: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in provenance and result tables."""

    @abstractmethod
    def parameter_spec(self) -> ParameterSpec:
        """Names, defaults, and realizability + stability bounds of the coefficients."""

    @abstractmethod
    def prior(self) -> ClosurePrior:
        """Prior over the coefficients, including the FD coupling hook (inert for
        a memoryless closure)."""

    @abstractmethod
    def likelihood(self, benchmark) -> Likelihood:
        """Bind this closure to a benchmark (the data) to obtain the likelihood
        hook theta -> predicted statistics."""
