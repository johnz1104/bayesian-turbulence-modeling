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


def basis_coefficients(T_basis, db, ridge=1e-12):
    """Objective (frame-equivariant) representation of a tensor discrepancy:
    per-sample least-squares coefficients of db in the integrity basis.

    Solves min_g || sum_n g_n T^(n) - db ||_F^2 (+ ridge ||g||^2) per sample.
    Because the T^(n) rotate with the frame while g and the conditioning
    invariants do not, a model that PREDICTS g (instead of raw tensor
    components) reconstructs a discrepancy that transforms correctly under
    rotation: the objectivity the raw-component parameterisation lacks. The
    ridge term regularises the rank deficiency of the basis on 2-D mean flows
    (where only three tensors are independent) without changing the
    reconstruction on the achievable subspace.

    T_basis: (N, nb, 3, 3) from integrity_basis (or a truncation of it);
    db: (N, 3, 3) symmetric traceless. Returns g: (N, nb).
    """
    T_basis = np.asarray(T_basis, float)
    db = np.asarray(db, float)
    N, nb = T_basis.shape[0], T_basis.shape[1]
    A = T_basis.reshape(N, nb, 9)                      # rows: basis tensors
    y = db.reshape(N, 9)
    # normal equations per sample: (A A^T + ridge I) g = A y
    G = np.einsum("nip,njp->nij", A, A) + ridge * np.eye(nb)
    rhs = np.einsum("nip,np->ni", A, y)
    return np.linalg.solve(G, rhs)


def basis_reconstruct(T_basis, g):
    """Reconstruct the tensor discrepancy from integrity-basis coefficients.

    db = sum_n g_n T^(n). T_basis: (N, nb, 3, 3); g: (N, nb) -> (N, 3, 3).
    """
    return np.einsum("nb,nbij->nij", np.asarray(g, float),
                     np.asarray(T_basis, float))


def boussinesq_anisotropy(S):
    """Linear eddy-viscosity (Boussinesq) anisotropy in non-dimensional form.

    With the eddy-viscosity model -u'u' = 2 nu_t S - (2/3) k I, the normalised
    anisotropy is b = -(nu_t/k) S.  In the non-dimensional convention used here
    (S already carries the turbulence timescale and nu_t ~ C_mu k timescale) the
    linear term is b = -C_mu S with C_mu = 0.09; this is the T^(1) coefficient.

    This form assumes nu_t = k/omega exactly. Where a baseline limits nu_t below
    that (the SST Bradshaw limiter binds in separated shear layers), use
    boussinesq_anisotropy_actual with the solved nu_t instead, so the baseline
    anisotropy is the stress the solver actually applies.
    """
    Cmu = 0.09
    return -Cmu * _dev(S)   # deviatoric strain (traceless even when div u != 0)


def boussinesq_anisotropy_actual(S, nu_t, k_baseline, timescale,
                                 k_floor=1e-30, coeff_cap=0.09):
    """Boussinesq anisotropy from the baseline's ACTUAL eddy viscosity.

    b = -(nu_t / k) S_dim = -(nu_t / (k timescale)) S_nondim, with the baseline's
    own k and the same timescale that non-dimensionalised S. This reproduces the
    stress the running solver applies, including the SST Bradshaw limiter (which
    binds in separated shear layers and REDUCES the coefficient below C_mu). The
    injected a-posteriori closure adds the learned db on top of this same
    baseline, so training target and injection stay consistent where the limiter
    is active.

    coeff_cap removes the solver's numerical nu_t floor (0.1 nu everywhere): near
    a wall the physical k/omega falls orders of magnitude below that floor, and
    dividing the floored nu_t by the vanishing local k would inflate b_B far
    beyond the realizable bound. In the tau = 1/(C_mu omega) convention the
    unfloored coefficient nu_t/(k tau) never exceeds C_mu (the limiter only
    lowers it), so capping at C_mu = 0.09 reproduces exactly min(nu_t, k/omega),
    the physical SST viscosity.
    """
    coeff = np.asarray(nu_t) / np.maximum(np.asarray(k_baseline)
                                          * np.asarray(timescale), k_floor)
    coeff = np.minimum(coeff, coeff_cap)
    return -coeff[..., None, None] * _dev(S)


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
