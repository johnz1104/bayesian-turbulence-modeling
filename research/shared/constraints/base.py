"""Physics constraints (FROZEN, core-v1.0): TWO separate entry points.

Realizability and Galilean (frame) invariance are distinct constraints requiring
distinct enforcement, and conflating them is a direct cause of solver divergence
(root CLAUDE.md, cross-cutting science constraints):

  * GalileanInvariantBasis delivers INVARIANCE. The integrity-basis construction
    expresses the anisotropy as b = sum_n g_n(I1..I5) T^(n) in tensors built from
    the strain-rate S and rotation-rate W (velocity-gradient quantities), so it is
    invariant under a Galilean boost and covariant under frame rotation. It does
    NOT deliver realizability.

  * RealizabilityProjection delivers REALIZABILITY, separately, by projecting the
    anisotropy eigenvalues into the barycentric (Lumley) triangle. The basis
    construction above can produce an unrealizable b; this projection repairs it.

These are two entry points on purpose. A closure applies the basis to build an
invariant anisotropy, THEN projects it to the realizable set. They are never one
call.
"""

from abc import ABC, abstractmethod

import numpy as np


# ---------------------------------------------------------------------------
# Entry point 1: realizability
# ---------------------------------------------------------------------------

class RealizabilityProjection(ABC):
    """Project a Reynolds-stress anisotropy tensor into the realizable set."""

    @abstractmethod
    def project(self, b: np.ndarray) -> np.ndarray:
        """Return a realizable 3x3 symmetric traceless anisotropy near ``b``."""

    @abstractmethod
    def is_realizable(self, b: np.ndarray, tol: float = 1e-9) -> bool:
        """True if ``b`` already lies in the realizable (barycentric) triangle."""


class BarycentricRealizability(RealizabilityProjection):
    """Reference realizability projection by barycentric eigenvalue clipping.

    For the anisotropy b = R/(2k) - I/3 (symmetric, traceless), order the
    eigenvalues l1 >= l2 >= l3. The barycentric weights

        C1c = l1 - l2          (one-component limit)
        C2c = 2 (l2 - l3)      (two-component limit)
        C3c = 3 l3 + 1         (isotropic limit)

    sum to 1 (since l1 + l2 + l3 = 0), and b is realizable iff all three are
    non-negative. The projection clips the weights onto the probability simplex
    (set negatives to zero, renormalise) and inverts back to eigenvalues

        l3 = (C3c - 1) / 3,   l2 = C2c / 2 + l3,   l1 = C1c + l2,

    then rebuilds b from the same eigenvectors. It is idempotent on realizable
    inputs and always returns a realizable tensor.
    """

    def _symmetric_traceless(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=np.float64)
        if b.shape != (3, 3):
            raise ValueError("anisotropy must be a 3x3 matrix")
        b = 0.5 * (b + b.T)
        return b - (np.trace(b) / 3.0) * np.eye(3)

    def barycentric(self, b: np.ndarray) -> np.ndarray:
        """Barycentric weights (C1c, C2c, C3c) of ``b``."""
        b0 = self._symmetric_traceless(b)
        w = np.linalg.eigvalsh(b0)          # ascending
        l1, l2, l3 = w[2], w[1], w[0]       # descending
        return np.array([l1 - l2, 2.0 * (l2 - l3), 3.0 * l3 + 1.0])

    def is_realizable(self, b: np.ndarray, tol: float = 1e-9) -> bool:
        c = self.barycentric(b)
        return bool(np.all(c >= -tol))

    def project(self, b: np.ndarray) -> np.ndarray:
        b0 = self._symmetric_traceless(b)
        w, V = np.linalg.eigh(b0)           # w ascending, columns are eigenvectors
        l1, l2, l3 = w[2], w[1], w[0]
        c = np.array([l1 - l2, 2.0 * (l2 - l3), 3.0 * l3 + 1.0])
        # clip onto the simplex {C >= 0, sum C = 1}
        c = np.maximum(c, 0.0)
        s = c.sum()
        c = c / s if s > 0.0 else np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        l3p = (c[2] - 1.0) / 3.0
        l2p = c[1] / 2.0 + l3p
        l1p = c[0] + l2p
        lam_ascending = np.array([l3p, l2p, l1p])   # match eigh ascending order
        return (V * lam_ascending) @ V.T


# ---------------------------------------------------------------------------
# Entry point 2: Galilean / frame invariance
# ---------------------------------------------------------------------------

class GalileanInvariantBasis(ABC):
    """Integrity basis delivering invariance (NOT realizability)."""

    @abstractmethod
    def invariants(self, S: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Scalar invariants of the (S, W) pair."""

    @abstractmethod
    def tensors(self, S: np.ndarray, W: np.ndarray) -> list:
        """Integrity-basis tensors in which the anisotropy is expanded."""

    @staticmethod
    def strain_rotation(grad_u: np.ndarray):
        """Split a velocity gradient G_ij = du_i/dx_j into (S, W).

        S = 0.5 (G + G^T) and W = 0.5 (G - G^T). Both depend only on the velocity
        gradient, so a Galilean boost u -> u + c leaves them unchanged; this is the
        Galilean half of the invariance the basis provides.
        """
        G = np.asarray(grad_u, dtype=np.float64)
        if G.shape != (3, 3):
            raise ValueError("velocity gradient must be a 3x3 matrix")
        return 0.5 * (G + G.T), 0.5 * (G - G.T)


class IntegrityBasis(GalileanInvariantBasis):
    """Reference integrity basis (Pope 1975), minimal subset.

    Five invariants of the normalised strain S and rotation W,

        I1 = tr(S^2),  I2 = tr(W^2),  I3 = tr(S^3),
        I4 = tr(S W^2), I5 = tr(S^2 W^2),

    and the first three of Pope's ten basis tensors,

        T1 = S,  T2 = S W - W S,  T3 = S^2 - (1/3) tr(S^2) I.

    Each invariant is a trace of products of frame-covariant tensors, so it is
    unchanged under a rotation Q: invariants(Q S Q^T, Q W Q^T) = invariants(S, W).
    The anisotropy is b = sum_n g_n(I1..I5) T^(n); the coefficient functions g_n
    are what a closure learns. The full basis is ten tensors; this reference keeps
    the first three for the foundation interface.
    """

    def _check(self, S, W):
        S = np.asarray(S, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        if S.shape != (3, 3) or W.shape != (3, 3):
            raise ValueError("S and W must be 3x3 matrices")
        return S, W

    def invariants(self, S: np.ndarray, W: np.ndarray) -> np.ndarray:
        S, W = self._check(S, W)
        S2 = S @ S
        W2 = W @ W
        return np.array([
            np.trace(S2),
            np.trace(W2),
            np.trace(S2 @ S),
            np.trace(S @ W2),
            np.trace(S2 @ W2),
        ])

    def tensors(self, S: np.ndarray, W: np.ndarray) -> list:
        S, W = self._check(S, W)
        T1 = S
        T2 = S @ W - W @ S
        T3 = S @ S - (np.trace(S @ S) / 3.0) * np.eye(3)
        return [T1, T2, T3]
