"""Channel DNS loader (Lee and Moser 2015) verified on the real raw files.

These tests run only where the gitignored DNS_data is present (locally); on CI
without the bulk data they skip. They check the loader against the per-case
parameters published in DNS_data/README.md and the internal consistency of the
canonical record (tensor assembly, k, wall boundary conditions, wall units).
"""
import numpy as np
import pytest

from UQ.datasets import ChannelDNS, CHANNEL_CASES

# Published per-case parameters (DNS_data/README.md): actual Re_tau, nu, u_tau,
# wall-normal point count.
KNOWN = {
    180:  (182.088,  3.50e-04, 0.0637309, 96),
    550:  (543.496,  1.00e-04, 0.0543496, 192),
    1000: (1000.512, 5.00e-05, 0.0500256, 256),
    2000: (1994.756, 2.30e-05, 0.0458794, 384),
    5200: (5185.897, 8.00e-06, 0.0414872, 768),
}

pytestmark = pytest.mark.skipif(
    not ChannelDNS.is_available(180),
    reason="channel DNS_data not present (bulk data is local/gitignored)",
)


@pytest.fixture(scope="module")
def cases():
    return {n: ChannelDNS.load(n) for n in CHANNEL_CASES if ChannelDNS.is_available(n)}


@pytest.mark.parametrize("n", CHANNEL_CASES)
def test_friction_parameters_match_published(cases, n):
    """u_tau, nu, Re_tau, half-height, and point count match the manifest."""
    if n not in cases:
        pytest.skip(f"Re_tau={n} not present")
    c = cases[n]
    re_tau, nu, u_tau, npts = KNOWN[n]
    assert c.re_tau == pytest.approx(re_tau, abs=1e-2)
    assert c.nu == pytest.approx(nu, rel=1e-6)
    assert c.u_tau == pytest.approx(u_tau, rel=1e-6)
    assert c.delta == pytest.approx(1.0, abs=1e-9)
    assert c.n == npts
    assert c.re_tau_nominal == n


@pytest.mark.parametrize("n", CHANNEL_CASES)
def test_tensor_and_k_consistency(cases, n):
    """R is symmetric, holds the six DNS components, and k = 0.5 tr(R)."""
    if n not in cases:
        pytest.skip(f"Re_tau={n} not present")
    c = cases[n]
    assert c.R.shape == (c.n, 3, 3)
    assert np.allclose(c.R, np.swapaxes(c.R, 1, 2))          # symmetric
    assert np.allclose(c.R[:, 0, 0], c.uu)                   # streamwise normal
    assert np.allclose(c.R[:, 0, 1], c.uv)                   # primary shear
    assert np.allclose(c.R[:, 1, 2], c.vw)                   # off-diagonal
    k_from_R = 0.5 * np.trace(c.R, axis1=1, axis2=2)
    assert np.allclose(k_from_R, c.k, rtol=1e-6, atol=1e-8)  # k column vs 0.5 tr(R)


@pytest.mark.parametrize("n", CHANNEL_CASES)
def test_wall_boundary_and_monotonic_grid(cases, n):
    """Wall units sanity: U^+(0)=0, dU/dy^+(0)=1, monotone y^+ from 0 to Re_tau."""
    if n not in cases:
        pytest.skip(f"Re_tau={n} not present")
    c = cases[n]
    assert abs(c.U[0]) < 1e-6                 # no-slip wall
    assert c.dUdy[0] == pytest.approx(1.0, abs=2e-3)   # viscous sublayer slope
    assert np.all(np.diff(c.yplus) > 0)       # monotone increasing
    assert c.yplus[0] == pytest.approx(0.0, abs=1e-6)
    # outer edge reaches the centerline: y^+_max ~ Re_tau, y/delta_max ~ 1
    assert c.yplus[-1] == pytest.approx(c.re_tau, rel=0.05)
    assert c.y_delta[-1] == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("n", CHANNEL_CASES)
def test_stdev_profiles_present_and_nonnegative(cases, n):
    """Every observed quantity carries a same-length, non-negative stdev profile."""
    if n not in cases:
        pytest.skip(f"Re_tau={n} not present")
    c = cases[n]
    for name in ("U", "dUdy", "uu", "vv", "ww", "uv", "k"):
        std = getattr(c, name + "_std")
        assert std.shape == (c.n,)
        assert np.all(std >= 0.0)


def test_derived_bulk_and_friction_physical(cases):
    """Bulk velocity grows and Cf falls with Re_tau, and Cf tracks Dean to a few %.

    Re_b is the published-consistent bulk Reynolds number (Re_tau 182 -> Re_b
    ~5700, the canonical Moser-Kim-Mansour channel condition).
    """
    present = [n for n in CHANNEL_CASES if n in cases]
    ub = [cases[n].bulk_velocity() for n in present]
    cf = [cases[n].skin_friction() for n in present]
    assert np.all(np.diff(ub) > 0)            # U_b^+ increases with Re_tau
    assert np.all(np.diff(cf) < 0)            # Cf decreases with Re_tau
    # Cf within ~6% of Dean (1978) Cf = 0.073 Re_b^-0.25 across the range
    for n in present:
        c = cases[n]
        cf_dean = 0.073 * c.reynolds_bulk() ** (-0.25)
        assert c.skin_friction() == pytest.approx(cf_dean, rel=0.06)


def test_velocity_gradient_is_pure_shear(cases):
    """grad_u has only the d U / d y entry, matching the parallel-shear geometry."""
    c = next(iter(cases.values()))
    g = c.velocity_gradient()
    assert g.shape == (c.n, 3, 3)
    assert np.allclose(g[:, 0, 1], c.dUdy)
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 1] = False
    assert np.allclose(g[:, mask], 0.0)       # every other entry is zero


def test_to_dnsfield_roundtrips_into_uq_schema(cases):
    """The record builds a UQ DNSField whose anisotropy is realizable DNS truth."""
    from UQ import realizability
    c = next(iter(cases.values()))
    # unit baseline timescale is enough to exercise the schema wiring here
    fd = c.to_dnsfield(timescale_plus=np.ones(c.n))
    assert fd.grad_u.shape == (c.n, 3, 3)
    assert fd.R.shape == (c.n, 3, 3)
    # the DNS Reynolds stress is realizable everywhere (it is real turbulence)
    assert np.all(realizability.is_realizable(c.R[1:]))   # skip exact-zero wall row
