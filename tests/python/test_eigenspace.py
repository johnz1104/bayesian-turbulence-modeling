"""Eigenspace-perturbation baseline: barycentric eigenvalue moves.

Pure-numpy properties of the Emory/Iaccarino perturbation used as the
comparison baseline of the separated-flow model-form study:
  1. delta_b = 1 lands exactly on the requested barycentric corner;
  2. delta_b = 0 is the identity;
  3. eigenvectors are preserved (the perturbed tensor commutes with the input);
  4. realizability is preserved for delta_b in [0, 1] on realizable input;
  5. the perturbed anisotropy stays symmetric and traceless.
"""
import numpy as np

from UQ import realizability as rz
from UQ.eigenspace import EigenspacePerturbation, CORNERS


def _random_realizable_b(n=200, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 3, 3)) * 0.4
    b = 0.5 * (a + np.swapaxes(a, 1, 2))
    b -= np.trace(b, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3.0
    bp, _ = rz.project_anisotropy(b)
    return bp


def _bary(b):
    R = 2.0 * (b + np.eye(3) / 3.0)
    c1, c2, c3 = rz.barycentric_coords(R)
    return np.stack([c1, c2, c3], axis=-1)


def test_full_perturbation_reaches_each_corner():
    b = _random_realizable_b()
    for name, corner in CORNERS.items():
        bp = EigenspacePerturbation.perturb(b, name, delta_b=1.0)
        C = _bary(bp)
        assert np.allclose(C, corner[None, :], atol=1e-8), name


def test_zero_perturbation_is_identity():
    b = _random_realizable_b()
    bp = EigenspacePerturbation.perturb(b, "1C", delta_b=0.0)
    assert np.allclose(bp, b, atol=1e-10)


def test_eigenvectors_preserved():
    # eigenvalue-only perturbation: b and its perturbation share eigenvectors,
    # so they commute
    b = _random_realizable_b(n=100, seed=5)
    for name in CORNERS:
        bp = EigenspacePerturbation.perturb(b, name, delta_b=0.6)
        comm = np.einsum("nij,njk->nik", b, bp) - np.einsum("nij,njk->nik", bp, b)
        assert np.max(np.abs(comm)) < 1e-9, name


def test_realizability_preserved_and_family_builds():
    b = _random_realizable_b(n=150, seed=8)
    for delta in (0.25, 0.5, 1.0):
        family = EigenspacePerturbation.corner_set(b, delta_b=delta)
        assert set(family) == {"1C", "2C", "3C"}
        assert EigenspacePerturbation.is_realizable_family(family)


def test_symmetric_traceless():
    b = _random_realizable_b(n=80, seed=11)
    bp = EigenspacePerturbation.perturb(b, "2C", delta_b=0.7)
    assert np.allclose(bp, np.swapaxes(bp, 1, 2), atol=1e-12)
    assert np.max(np.abs(np.trace(bp, axis1=1, axis2=2))) < 1e-9
