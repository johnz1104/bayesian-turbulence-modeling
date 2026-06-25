"""Barycentric realizability projection for the Reynolds-stress anisotropy.

This is the realizability constraint of the project, kept strictly separate from
the Galilean-invariant construction (which lives in ``discrepancy``). Conflating
the two is a documented cause of solver divergence, so this module touches only
the anisotropy eigenvalues and leaves the (invariant) eigenvector frame intact.

It mirrors the C++ ``RealizabilityProjection`` so the Python UQ layer can project
learned-discrepancy samples into the realizable set without a build dependency.

Anisotropy:    b_ij = R_ij/(2k) - delta_ij/3,  k = 0.5 R_ii
Barycentric map (Banerjee et al. 2007) from sorted eigenvalues l1>=l2>=l3:
    C1c = l1 - l2,   C2c = 2(l2 - l3),   C3c = 3 l3 + 1,   sum = 1
The realizable states are exactly the simplex {C_i >= 0, sum C_i = 1}; projection
is the Euclidean projection onto that simplex followed by reconstruction.

All functions are vectorised over a leading batch dimension. Tensors are
(..., 3, 3) symmetric arrays.
"""
import numpy as np

# barycentric triangle corner coordinates (1C, 2C, 3C), for plotting / distance
CORNER_1C = np.array([1.0, 0.0])
CORNER_2C = np.array([0.0, 0.0])
CORNER_3C = np.array([0.5, np.sqrt(3.0) / 2.0])


def turbulent_energy(R):
    """k = 0.5 trace(R)."""
    return 0.5 * np.trace(R, axis1=-2, axis2=-1)


def anisotropy(R, k_floor=1e-12):
    """Normalised anisotropy b = R/(2k) - I/3 (batched)."""
    k = turbulent_energy(R)
    twoK = np.maximum(2.0 * k, k_floor)
    b = R / twoK[..., None, None] - np.eye(3) / 3.0
    return b


def _sorted_eigh(b):
    """Eigenvalues (descending) and eigenvectors of a batch of symmetric 3x3."""
    w, v = np.linalg.eigh(b)            # ascending
    w = w[..., ::-1]
    v = v[..., ::-1]
    return w, v


def barycentric_coords(R):
    """Return (C1c, C2c, C3c), each shape (...,), from a Reynolds stress batch."""
    b = anisotropy(R)
    w, _ = _sorted_eigh(b)
    l1, l2, l3 = w[..., 0], w[..., 1], w[..., 2]
    c1 = l1 - l2
    c2 = 2.0 * (l2 - l3)
    c3 = 3.0 * l3 + 1.0
    return c1, c2, c3


def is_realizable(R, tol=1e-9):
    """True where all barycentric coordinates are non-negative."""
    c1, c2, c3 = barycentric_coords(R)
    return (c1 >= -tol) & (c2 >= -tol) & (c3 >= -tol)


def _project_simplex(C):
    """Euclidean projection of rows of C (..., 3) onto the probability simplex.

    Wang and Carreira-Perpinan (2013), vectorised over the batch.
    """
    n = C.shape[-1]
    u = np.sort(C, axis=-1)[..., ::-1]
    cssv = np.cumsum(u, axis=-1) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=-1)           # support size
    theta = np.take_along_axis(cssv, (rho - 1)[..., None], axis=-1)[..., 0] / rho
    return np.maximum(C - theta[..., None], 0.0)


def project_anisotropy(b):
    """Project an anisotropy tensor batch into the realizable set.

    Eigenvectors are preserved; only the eigenvalues move. Returns the projected
    anisotropy and the per-sample L2 eigenvalue shift (the projection distance).
    """
    w, v = _sorted_eigh(b)
    l1, l2, l3 = w[..., 0], w[..., 1], w[..., 2]
    C = np.stack([l1 - l2, 2.0 * (l2 - l3), 3.0 * l3 + 1.0], axis=-1)
    Cp = _project_simplex(C)
    l3p = (Cp[..., 2] - 1.0) / 3.0
    l2p = l3p + 0.5 * Cp[..., 1]
    l1p = l2p + Cp[..., 0]
    lp = np.stack([l1p, l2p, l3p], axis=-1)
    dist = np.sqrt(np.sum((lp - w) ** 2, axis=-1))
    # rebuild b = V diag(lp) V^T
    bp = np.einsum("...ij,...j,...kj->...ik", v, lp, v)
    # symmetrise against round-off
    bp = 0.5 * (bp + np.swapaxes(bp, -1, -2))
    return bp, dist


def project_reynolds_stress(R):
    """Project a Reynolds-stress batch into the realizable set, preserving 2k."""
    k = turbulent_energy(R)
    twoK = np.maximum(2.0 * k, 1e-30)
    b = R / twoK[..., None, None] - np.eye(3) / 3.0
    bp, _ = project_anisotropy(b)
    Rp = twoK[..., None, None] * (bp + np.eye(3) / 3.0)
    return Rp


def barycentric_xy(R):
    """Map a Reynolds-stress batch to (x, y) points in the barycentric triangle."""
    c1, c2, c3 = barycentric_coords(R)
    x = c1 * CORNER_1C[0] + c2 * CORNER_2C[0] + c3 * CORNER_3C[0]
    y = c1 * CORNER_1C[1] + c2 * CORNER_2C[1] + c3 * CORNER_3C[1]
    return np.stack([x, y], axis=-1)
