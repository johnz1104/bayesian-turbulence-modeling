"""Plane-Couette DNS loader (Pirozzoli et al. 2014) verified on the real raw files.

These tests run only where the gitignored DNS_data is present (locally); on CI
without the bulk data they skip. They check the loader against the per-case
friction Reynolds number printed in each file and the internal consistency of the
canonical record (tensor assembly from the rms columns, k, wall units), plus the
data-only physics identity that anchors the modeled observation uncertainty:
plane Couette has exactly constant total stress, dU^+/dy^+ - <u'v'>^+ = 1.
"""
import numpy as np
import pytest

from UQ.datasets import CouetteDNS, COUETTE_CASES

# Per-case parameters: friction Reynolds number (the RETAU title value) and the
# wall-normal point count, read from the compiled files (DNS_data/README.md).
KNOWN = {
    171: (171.0, 128),
    260: (260.0, 128),
    507: (507.0, 192),
    986: (986.0, 256),
}

pytestmark = pytest.mark.skipif(
    not CouetteDNS.is_available(986),
    reason="Couette DNS_data not present (bulk data is local/gitignored)",
)


@pytest.fixture(scope="module")
def cases():
    return {n: CouetteDNS.load(n) for n in COUETTE_CASES if CouetteDNS.is_available(n)}


@pytest.mark.parametrize("n", COUETTE_CASES)
def test_re_tau_and_size_match_file(cases, n):
    """Re_tau from the RETAU title line and the point count match the file."""
    if n not in cases:
        pytest.skip(f"Re_tau={n} not present")
    c = cases[n]
    re_tau, npts = KNOWN[n]
    assert c.re_tau == pytest.approx(re_tau, abs=1e-6)
    assert c.re_tau_nominal == n
    assert c.n == npts


@pytest.mark.parametrize("n", COUETTE_CASES)
def test_tensor_from_rms_and_k(cases, n):
    """R is symmetric, the normal stresses are the squared rms columns, the shear
    maps to R_xy, the spanwise off-diagonals vanish, and k = 0.5 tr(R)."""
    if n not in cases:
        pytest.skip(f"Re_tau={n} not present")
    c = cases[n]
    assert c.R.shape == (c.n, 3, 3)
    assert np.allclose(c.R, np.swapaxes(c.R, 1, 2))             # symmetric
    assert np.allclose(c.R[:, 0, 0], c.u_rms ** 2)             # uu = u'^2
    assert np.allclose(c.R[:, 1, 1], c.v_rms ** 2)             # vv = v'^2
    assert np.allclose(c.R[:, 2, 2], c.w_rms ** 2)             # ww = w'^2
    assert np.allclose(c.R[:, 0, 2], 0.0)                       # u'w' vanishes
    assert np.allclose(c.R[:, 1, 2], 0.0)                       # v'w' vanishes
    k_from_R = 0.5 * np.trace(c.R, axis1=1, axis2=2)
    assert np.allclose(k_from_R, c.k, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n", COUETTE_CASES)
def test_wall_units_and_half_gap(cases, n):
    """U^+(0)=0, monotone y^+, the profile spans wall to centerline (y/h: 0 -> 1),
    and y^+ reaches ~Re_tau at the centerline."""
    if n not in cases:
        pytest.skip(f"Re_tau={n} not present")
    c = cases[n]
    # the file's first node sits just off the wall (no exact y^+ = 0 row), so the
    # viscous-sublayer law U^+ ~ y^+ holds there rather than U^+ = 0 exactly
    assert c.yplus[0] < 0.3
    assert c.U[0] == pytest.approx(c.yplus[0], rel=0.05)
    assert np.all(np.diff(c.yplus) > 0)            # monotone increasing
    assert c.y_outer[0] == pytest.approx(0.0, abs=2e-3)
    assert c.y_outer[-1] == pytest.approx(1.0, abs=0.05)
    assert c.yplus[-1] == pytest.approx(c.re_tau, rel=0.02)
    # Couette is a shear flow: the primary shear stress is negative across the gap
    assert np.all(c.uv[1:] < 0.0)


@pytest.mark.parametrize("n", COUETTE_CASES)
def test_constant_total_stress_identity(cases, n):
    """Plane Couette has exactly constant total stress: dU^+/dy^+ - <u'v'>^+ = 1.

    This is the data-only physics anchor for the modeled observation sigma. The
    finite-differenced gradient is least accurate at the wall and the centerline,
    so the rms is taken over the interior, where it measures the DNS's own
    statistical convergence (a fraction of a percent).
    """
    if n not in cases:
        pytest.skip(f"Re_tau={n} not present")
    c = cases[n]
    total = c.total_stress_plus()
    interior = (c.yplus > 5.0) & (c.y_outer < 0.95)
    rms = float(np.sqrt(np.mean((total[interior] - 1.0) ** 2)))
    assert rms < 3.0e-3                            # well under 0.3% across all Re


def test_velocity_gradient_is_pure_streamwise_shear(cases):
    """Couette has no mean spanwise velocity, so grad_u has only the dU/dy entry."""
    c = next(iter(cases.values()))
    g = c.velocity_gradient()
    assert g.shape == (c.n, 3, 3)
    assert np.allclose(g[:, 0, 1], c.dUdy())
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 1] = False
    assert np.allclose(g[:, mask], 0.0)
    assert c.W is None


def test_builds_realizable_dnsfield(cases):
    """The record builds a UQ DNSField whose Reynolds stress is realizable DNS."""
    from UQ import realizability
    c = cases[986] if 986 in cases else next(iter(cases.values()))
    fd = c.to_dnsfield(timescale_plus=np.ones(c.n))
    assert fd.grad_u.shape == (c.n, 3, 3)
    assert fd.R.shape == (c.n, 3, 3)
    assert np.all(realizability.is_realizable(c.R[1:]))   # skip exact-zero wall row
