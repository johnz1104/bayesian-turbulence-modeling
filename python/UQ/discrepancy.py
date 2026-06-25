"""Model-form discrepancy extraction and the Galilean-invariant feature set.

Given a high-fidelity (DNS) field this forms the two model-form discrepancies the
compressible program is built around and the invariant features a learned closure
is conditioned on:

  - Reynolds-stress discrepancy: the DNS anisotropy minus the Boussinesq (linear
    eddy-viscosity) anisotropy,  db = b_DNS - b_Boussinesq.
  - turbulent-heat-flux discrepancy: the DNS heat flux minus the
    gradient-diffusion-hypothesis flux,  dq = q_DNS + (nu_t/Pr_t) grad(T).

The conditioning features are Galilean invariant by construction: they are built
from the velocity-gradient tensor (invariant under a constant velocity shift),
normalised by a turbulence timescale, via Pope's (1975) integrity basis. This
delivers invariance; it does NOT deliver realizability, which is enforced
separately in ``realizability``. Keeping the two apart is a project rule.

All routines are vectorised over a leading sample axis N. Tensors are (N, 3, 3),
vectors (N, 3).
"""
import numpy as np

I3 = np.eye(3)


def strain_rotation(grad_u, timescale):
    """Non-dimensional strain S and rotation W from the velocity gradient.

    grad_u[n,i,j] = d u_i / d x_j.  The timescale (e.g. k/epsilon or 1/omega)
    non-dimensionalises them so the invariants are pure numbers. The velocity
    gradient is Galilean invariant, hence so are S and W.
    """
    ts = np.asarray(timescale)[..., None, None]
    gt = np.swapaxes(grad_u, -1, -2)
    S = 0.5 * (grad_u + gt) * ts
    W = 0.5 * (grad_u - gt) * ts
    return S, W


def _mm(A, B):
    return np.einsum("nij,njk->nik", A, B)


def _trace(A):
    return np.trace(A, axis1=-2, axis2=-1)


def invariants(S, W):
    """Pope's five scalar invariants of (S, W) (Galilean invariant features).

    lambda1 = tr(S^2), lambda2 = tr(W^2), lambda3 = tr(S^3),
    lambda4 = tr(W^2 S), lambda5 = tr(W^2 S^2)
    """
    S2 = _mm(S, S)
    W2 = _mm(W, W)
    l1 = _trace(S2)
    l2 = _trace(W2)
    l3 = _trace(_mm(S2, S))
    l4 = _trace(_mm(W2, S))
    l5 = _trace(_mm(W2, S2))
    return np.stack([l1, l2, l3, l4, l5], axis=-1)


def _dev(T):
    """Deviatoric (traceless) part of a tensor batch."""
    return T - (_trace(T) / 3.0)[..., None, None] * I3


def integrity_basis(S, W):
    """Pope's ten-tensor integrity basis T^(1..10), each (N,3,3) traceless.

    The anisotropy of any Galilean-invariant eddy-viscosity-type closure can be
    written b = sum_n g_n(invariants) T^(n), so a learned discrepancy expressed
    in this basis inherits the invariance.
    """
    S2 = _mm(S, S)
    W2 = _mm(W, W)
    SW = _mm(S, W)
    WS = _mm(W, S)
    T = []
    T.append(_dev(S))                                      # T1 (deviatoric: compressible-safe)
    T.append(SW - WS)                                      # T2
    T.append(_dev(S2))                                     # T3
    T.append(_dev(W2))                                     # T4
    T.append(_mm(W, S2) - _mm(S2, W))                      # T5
    T.append(_dev(_mm(W2, S) + _mm(S, W2)))                # T6
    T.append(_mm(_mm(W, S), W2) - _mm(_mm(W2, S), W))      # T7
    T.append(_mm(_mm(S, W), S2) - _mm(_mm(S2, W), S))      # T8
    T.append(_dev(_mm(W2, S2) + _mm(S2, W2)))              # T9
    T.append(_mm(_mm(W, S2), W2) - _mm(_mm(W2, S2), W))    # T10
    return np.stack(T, axis=1)                             # (N, 10, 3, 3)


def boussinesq_anisotropy(S):
    """Linear eddy-viscosity (Boussinesq) anisotropy in non-dimensional form.

    With the eddy-viscosity model -u'u' = 2 nu_t S - (2/3) k I, the normalised
    anisotropy is b = -(nu_t/k) S.  In the non-dimensional convention used here
    (S already carries the turbulence timescale and nu_t ~ C_mu k timescale) the
    linear term is b = -C_mu S with C_mu = 0.09; this is the T^(1) coefficient.
    """
    Cmu = 0.09
    return -Cmu * _dev(S)   # deviatoric strain (traceless even when div u != 0)


def reynolds_anisotropy(R, k_floor=1e-12):
    """DNS anisotropy b = R/(2k) - I/3 from the Reynolds-stress tensor."""
    k = 0.5 * _trace(R)
    twoK = np.maximum(2.0 * k, k_floor)
    return R / twoK[..., None, None] - I3 / 3.0


def reynolds_discrepancy(R_dns, S):
    """Anisotropy discrepancy db = b_DNS - b_Boussinesq (the learnable target)."""
    return reynolds_anisotropy(R_dns) - boussinesq_anisotropy(S)


def gdh_heat_flux(grad_T, nu_t, Pr_t=0.9):
    """Gradient-diffusion-hypothesis turbulent heat flux  q = -(nu_t/Pr_t) grad T."""
    coeff = (np.asarray(nu_t) / Pr_t)[..., None]
    return -coeff * grad_T


def heatflux_discrepancy(q_dns, grad_T, nu_t, Pr_t=0.9):
    """Heat-flux discrepancy dq = q_DNS - q_GDH (Reynolds-analogy breakdown)."""
    return q_dns - gdh_heat_flux(grad_T, nu_t, Pr_t)


def feature_set(grad_u, timescale, extra=None):
    """Assemble the Galilean-invariant conditioning features for a learned model.

    Returns the five Pope invariants, optionally concatenated with extra
    compressibility-aware features (e.g. turbulent Mach number) supplied as an
    (N, m) array. The features are the inputs a conditional discrepancy model is
    conditioned on.
    """
    S, W = strain_rotation(grad_u, timescale)
    feats = invariants(S, W)
    if extra is not None:
        feats = np.concatenate([feats, np.asarray(extra)], axis=-1)
    return feats
