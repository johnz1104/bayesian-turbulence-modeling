"""Eigenspace perturbation of the Reynolds-stress anisotropy (the comparison
baseline of the separated-flow model-form study).

Two variants, named precisely because they differ in strength:

- THREE-CORNER EIGENVALUE-ONLY (Emory, Larsson and Iaccarino 2013), corner_set:
  move the anisotropy eigenvalues toward the one-, two- and three-component
  limiting states of the barycentric triangle while keeping the eigenvectors.
  Three perturbed closures.
- FIVE-STATE (Iaccarino, Mishra and Ghili 2017, Phys. Rev. Fluids 2, 024605),
  five_state_set: the 1C and 2C eigenvalue corners EACH paired with BOTH
  production-extremal eigenvector alignments (the perturbed stress realigned
  with the mean strain-rate eigenframe to maximise or minimise turbulence
  production), plus the isotropic 3C corner, where the eigenvalues are equal
  and the alignment is immaterial: {(1C, vmax), (1C, vmin), (2C, vmax),
  (2C, vmin), 3C}. Five perturbed closures; the established practice reported
  to improve bounds over eigenvalue-only perturbation.

Both are deterministic bounding envelopes, not probability distributions; how
they are scored against probabilistic methods is pre-registered in
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
        """The THREE-CORNER EIGENVALUE-ONLY family (Emory et al. 2013).

        This is the set of closures the a-posteriori envelope is built from:
        each entry is propagated through the same Reynolds-stress injection as
        the generative and Gaussian methods, and the per-quantity [min, max]
        over the solves is the envelope. Eigenvectors are preserved, so this is
        the weaker 2013 variant; see five_state_set for the 2017 extension.
        """
        return {c: EigenspacePerturbation.perturb(b, c, delta_b) for c in corners}

    @staticmethod
    def production_extremal_states(b, strain, delta_b=1.0, base_corner="1C"):
        """The two eigenvector-perturbed states of the 2017 five-state method.

        The anisotropy contribution to turbulence production is
        P_b = -2 k b : S = -2 k sum_i lambda_i gamma_pair(i) once b's
        eigenvectors are realigned with the strain eigenframe, so by the
        rearrangement inequality production is maximised by pairing the
        DESCENDING anisotropy eigenvalues with the ASCENDING strain eigenvalues
        (largest anisotropy on the most compressive direction) and minimised by
        the descending-descending pairing. Both states reuse the corner-
        perturbed eigenvalues (base_corner at delta_b, default the 1C state,
        where the anisotropy magnitude and hence the production bound is
        widest), so they sit at the same barycentric point as that corner and
        realizability is preserved by construction.

        b, strain: (N, 3, 3) batches (strain = symmetric mean rate of strain).
        Returns {"<base_corner>_vmax": (N, 3, 3), "<base_corner>_vmin": ...}.
        """
        b = np.asarray(b, float)
        strain = np.asarray(strain, float)

        # corner-perturbed eigenvalues, descending (reuse the 2013 move)
        bp = EigenspacePerturbation.perturb(b, base_corner, delta_b)
        lp = np.linalg.eigvalsh(bp)[..., ::-1]       # l1 >= l2 >= l3

        # strain eigenframe, ascending gamma1 <= gamma2 <= gamma3
        gS, vS = np.linalg.eigh(0.5 * (strain + np.swapaxes(strain, -1, -2)))

        # vmax: descending lambda on ascending gamma (most compressive first);
        # vmin: descending lambda on descending gamma
        vmax = np.einsum("...ij,...j,...kj->...ik", vS, lp, vS)
        vS_desc = vS[..., ::-1]
        vmin = np.einsum("...ij,...j,...kj->...ik", vS_desc, lp, vS_desc)
        sym = lambda a: 0.5 * (a + np.swapaxes(a, -1, -2))
        return {f"{base_corner}_vmax": sym(vmax), f"{base_corner}_vmin": sym(vmin)}

    @staticmethod
    def five_state_set(b, strain, delta_b=1.0):
        """The FIVE-STATE family (Iaccarino, Mishra and Ghili 2017).

        The documented construction: the 1C and 2C eigenvalue corners each
        paired with both production-extremal eigenvector alignments, plus the
        isotropic 3C corner (equal eigenvalues, so the eigenvector pairing is
        immaterial there; at delta_b = 1 the 3C member is exactly b = 0), all
        at the same delta_b. Propagated exactly like corner_set members.
        """
        family = {}
        for corner in ("1C", "2C"):
            family.update(EigenspacePerturbation.production_extremal_states(
                b, strain, delta_b, base_corner=corner))
        family["3C"] = EigenspacePerturbation.perturb(b, "3C", delta_b)
        return family

    @staticmethod
    def is_realizable_family(family, tol=1e-8):
        """All members realizable (true for delta_b <= 1 and realizable input)."""
        ok = True
        for bp in family.values():
            R = 2.0 * (bp + np.eye(3) / 3.0)
            ok = ok and bool(np.all(rz.is_realizable(R, tol=tol)))
        return ok
