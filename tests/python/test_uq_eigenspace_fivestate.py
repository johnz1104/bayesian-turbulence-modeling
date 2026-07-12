"""Five-state eigenspace perturbation (Iaccarino, Mishra and Ghili 2017).

Pins the audit extension to UQ.eigenspace: the two eigenvector-perturbed
states must (a) keep the corner-perturbed EIGENVALUES exactly (same
barycentric point, hence realizability by construction), and (b) extremise
the anisotropy contribution to production b : S over all eigenvalue-to-
strain-eigenvector pairings (rearrangement inequality), with vmax minimising
b : S (production has the opposite sign) and vmin maximising it.
"""
import itertools

import numpy as np

from UQ.eigenspace import EigenspacePerturbation
from UQ import realizability as rz


def _random_realizable_b(rng, n):
    A = rng.normal(size=(n, 3, 3))
    R = np.einsum("nij,nkj->nik", A, A)          # SPD Reynolds tensors
    k = 0.5 * np.trace(R, axis1=-2, axis2=-1)
    return R / (2.0 * k)[:, None, None] - np.eye(3) / 3.0


def _random_strain(rng, n):
    A = rng.normal(size=(n, 3, 3))
    S = 0.5 * (A + np.swapaxes(A, -1, -2))
    tr = np.trace(S, axis1=-2, axis2=-1)
    return S - tr[:, None, None] * np.eye(3) / 3.0


def test_five_state_family_keys_and_realizability():
    rng = np.random.default_rng(0)
    b = _random_realizable_b(rng, 20)
    S = _random_strain(rng, 20)
    fam = EigenspacePerturbation.five_state_set(b, S, delta_b=1.0)
    assert set(fam) == {"1C", "2C", "3C", "1C_vmax", "1C_vmin"}
    assert EigenspacePerturbation.is_realizable_family(fam)


def test_eigenvector_states_preserve_corner_eigenvalues():
    rng = np.random.default_rng(1)
    b = _random_realizable_b(rng, 15)
    S = _random_strain(rng, 15)
    for delta in (0.5, 1.0):
        corner = EigenspacePerturbation.perturb(b, "1C", delta)
        states = EigenspacePerturbation.production_extremal_states(
            b, S, delta_b=delta)
        lc = np.sort(np.linalg.eigvalsh(corner), axis=-1)
        for bp in states.values():
            lp = np.sort(np.linalg.eigvalsh(bp), axis=-1)
            assert np.allclose(lp, lc, atol=1e-10), \
                "eigenvector perturbation must not move the eigenvalues"


def test_production_extremality_over_all_pairings():
    rng = np.random.default_rng(2)
    b = _random_realizable_b(rng, 10)
    S = _random_strain(rng, 10)
    states = EigenspacePerturbation.production_extremal_states(b, S, delta_b=1.0)
    corner = EigenspacePerturbation.perturb(b, "1C", 1.0)
    lp = np.linalg.eigvalsh(corner)[..., ::-1]       # descending
    gS, vS = np.linalg.eigh(S)                       # ascending eigenvalues

    bs_max = np.einsum("nij,nij->n", states["1C_vmax"], S)
    bs_min = np.einsum("nij,nij->n", states["1C_vmin"], S)

    # enumerate every pairing of eigenvalues onto strain eigenvectors
    for n in range(b.shape[0]):
        contractions = []
        for perm in itertools.permutations(range(3)):
            bp = sum(lp[n, i] * np.outer(vS[n][:, perm[i]], vS[n][:, perm[i]])
                     for i in range(3))
            contractions.append(np.sum(bp * S[n]))
        contractions = np.array(contractions)
        # production P ~ -2k b:S, so vmax (max production) minimises b:S
        assert abs(bs_max[n] - contractions.min()) < 1e-10
        assert abs(bs_min[n] - contractions.max()) < 1e-10


def test_three_corner_set_unchanged():
    # the 2013 three-corner family is untouched by the extension
    rng = np.random.default_rng(3)
    b = _random_realizable_b(rng, 8)
    fam3 = EigenspacePerturbation.corner_set(b, delta_b=1.0)
    assert set(fam3) == {"1C", "2C", "3C"}
    fam5 = EigenspacePerturbation.five_state_set(b, _random_strain(rng, 8), 1.0)
    for c in ("1C", "2C", "3C"):
        assert np.allclose(fam3[c], fam5[c], atol=0.0)
