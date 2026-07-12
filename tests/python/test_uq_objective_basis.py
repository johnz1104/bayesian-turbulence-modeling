"""Objective (integrity-basis) representation of tensor discrepancies.

Pins the audit addition to UQ.discrepancy: predicting integrity-basis
coefficients instead of raw lab-frame components makes the learned tensor
rotation-equivariant. The basis over-spans the five-dimensional
symmetric-traceless space (ten tensors, Cayley-Hamilton relations), so
COEFFICIENTS are not unique and are never asserted; the physically meaningful
properties are (a) exact reconstruction on the achievable space and (b)
equivariance of the reconstruction under frame rotation, while the
conditioning invariants stay fixed.
"""
import numpy as np

from UQ import discrepancy as dc


def _random_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _random_sym_traceless(rng, n):
    A = rng.normal(size=(n, 3, 3))
    S = 0.5 * (A + np.swapaxes(A, -1, -2))
    tr = np.trace(S, axis1=-2, axis2=-1)
    return S - tr[:, None, None] * np.eye(3) / 3.0


def test_reconstruction_is_exact_on_generic_states():
    rng = np.random.default_rng(0)
    grad_u = rng.normal(size=(40, 3, 3))
    S, W = dc.strain_rotation(grad_u, np.ones(40))
    T = dc.integrity_basis(S, W)
    db = _random_sym_traceless(rng, 40)
    g = dc.basis_coefficients(T, db)
    recon = dc.basis_reconstruct(T, g)
    assert np.allclose(recon, db, atol=1e-7), "basis must span the target space"


def test_reconstruction_round_trip_from_planted_coefficients():
    rng = np.random.default_rng(1)
    grad_u = rng.normal(size=(25, 3, 3))
    S, W = dc.strain_rotation(grad_u, np.ones(25))
    T = dc.integrity_basis(S, W)
    g_true = 0.1 * rng.normal(size=(25, 10))
    db = dc.basis_reconstruct(T, g_true)
    g_est = dc.basis_coefficients(T, db)
    # coefficients need not match (over-complete basis); the tensor must
    assert np.allclose(dc.basis_reconstruct(T, g_est), db, atol=1e-8)


def test_reconstruction_is_rotation_equivariant_and_invariants_fixed():
    rng = np.random.default_rng(2)
    R = _random_rotation(rng)
    grad_u = rng.normal(size=(30, 3, 3))
    ts = np.ones(30)

    S, W = dc.strain_rotation(grad_u, ts)
    T = dc.integrity_basis(S, W)
    db = _random_sym_traceless(rng, 30)
    recon = dc.basis_reconstruct(T, dc.basis_coefficients(T, db))

    grad_u_rot = np.einsum("ij,njk,lk->nil", R, grad_u, R)
    S_r, W_r = dc.strain_rotation(grad_u_rot, ts)
    T_r = dc.integrity_basis(S_r, W_r)
    db_rot = np.einsum("ij,njk,lk->nil", R, db, R)
    recon_rot = dc.basis_reconstruct(T_r, dc.basis_coefficients(T_r, db_rot))

    want = np.einsum("ij,njk,lk->nil", R, recon, R)
    assert np.allclose(recon_rot, want, atol=1e-7), \
        "objective reconstruction must rotate with the frame"

    # the conditioning features are invariant under the same rotation
    assert np.allclose(dc.invariants(S, W), dc.invariants(S_r, W_r), atol=1e-9)


def test_raw_component_representation_is_not_equivariant():
    # the defect the objective option fixes: identical invariant inputs, yet
    # the raw-component target changes under rotation, so a raw-component
    # model CANNOT be equivariant (documents why the option exists)
    rng = np.random.default_rng(3)
    R = _random_rotation(rng)
    grad_u = rng.normal(size=(5, 3, 3))
    S, W = dc.strain_rotation(grad_u, np.ones(5))
    S_r, W_r = dc.strain_rotation(
        np.einsum("ij,njk,lk->nil", R, grad_u, R), np.ones(5))
    db = _random_sym_traceless(rng, 5)
    db_rot = np.einsum("ij,njk,lk->nil", R, db, R)
    assert np.allclose(dc.invariants(S, W), dc.invariants(S_r, W_r), atol=1e-9)
    assert not np.allclose(db, db_rot, atol=1e-3)
