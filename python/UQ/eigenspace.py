"""Eigenspace perturbation of the Reynolds-stress anisotropy (the comparison
baseline of the separated-flow model-form study).

The method of Emory, Larsson and Iaccarino (2013): move the anisotropy
eigenvalues toward the one-, two- and three-component limiting states of the
barycentric triangle while keeping the eigenvectors, giving a small set of
perturbed closures whose solves bound the quantity-of-interest response. It is
a deterministic bounding envelope, not a probability distribution; how it is
scored against probabilistic methods is pre-registered in
UQ-RANS_research/separated_modelform/METHODS_OPERATIONALIZATION.md (the
envelope check, plus a uniform-ensemble reading for CRPS and energy-score
comparability, labeled charitable wherever used).

Barycentric map (Banerjee et al. 2007), as in ``realizability``:
    C1c = l1 - l2,  C2c = 2 (l2 - l3),  C3c = 3 l3 + 1,  sum = 1.
A perturbation toward a corner is the convex move C* = C + Delta_B (C_corner -
C); for Delta_B in [0, 1] and a realizable input the result stays inside the
simplex, so the perturbed anisotropy is realizable by construction.
"""
import numpy as np

from . import realizability as rz

# barycentric coordinates of the limiting states: one-component (rod-like),
# two-component (disk-like), three-component (isotropic)
CORNERS = {
    "1C": np.array([1.0, 0.0, 0.0]),
    "2C": np.array([0.0, 1.0, 0.0]),
    "3C": np.array([0.0, 0.0, 1.0]),
}


class EigenspacePerturbation:
    """Barycentric eigenvalue perturbation toward the limiting states."""

    @staticmethod
    def perturb(b, corner, delta_b=1.0):
        """Move the eigenvalues of b toward a barycentric corner.

        b is an (N, 3, 3) anisotropy batch, corner one of "1C", "2C", "3C",
        delta_b in [0, 1] the relative perturbation magnitude (1 = full
        projection to the corner). Eigenvectors are preserved (the 2013
        eigenvalue-only formulation). Returns the perturbed (N, 3, 3) batch.
        """
        target = CORNERS[corner]
        b = np.asarray(b, float)
        w, v = np.linalg.eigh(b)                     # ascending
        w = w[..., ::-1]                             # descending l1 >= l2 >= l3
        v = v[..., ::-1]
        l1, l2, l3 = w[..., 0], w[..., 1], w[..., 2]
        C = np.stack([l1 - l2, 2.0 * (l2 - l3), 3.0 * l3 + 1.0], axis=-1)

        # convex move toward the corner in barycentric coordinates
        Cp = C + float(delta_b) * (target - C)

        # invert the barycentric map back to eigenvalues
        l3p = (Cp[..., 2] - 1.0) / 3.0
        l2p = l3p + 0.5 * Cp[..., 1]
        l1p = l2p + Cp[..., 0]
        lp = np.stack([l1p, l2p, l3p], axis=-1)

        bp = np.einsum("...ij,...j,...kj->...ik", v, lp, v)
        return 0.5 * (bp + np.swapaxes(bp, -1, -2))  # symmetrise round-off

    @staticmethod
    def corner_set(b, delta_b=1.0, corners=("1C", "2C", "3C")):
        """The perturbed-anisotropy family, one entry per corner.

        This is the set of closures the a-posteriori envelope is built from:
        each entry is propagated through the same Reynolds-stress injection as
        the generative and Gaussian methods, and the per-quantity [min, max]
        over the solves is the envelope.
        """
        return {c: EigenspacePerturbation.perturb(b, c, delta_b) for c in corners}

    @staticmethod
    def is_realizable_family(family, tol=1e-8):
        """All members realizable (true for delta_b <= 1 and realizable input)."""
        ok = True
        for bp in family.values():
            R = 2.0 * (bp + np.eye(3) / 3.0)
            ok = ok and bool(np.all(rz.is_realizable(R, tol=tol)))
        return ok
