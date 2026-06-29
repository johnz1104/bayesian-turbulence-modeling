"""Pipe-flow DNS loader (Pirozzoli 2024) verified on the real raw files.

Skips where the gitignored DNS_data is absent (CI). Checks the loader against the
header parameters and, centrally, the linear-total-stress identity dU^+/dy^+ -
R_xy = 1 - y^+/Re_tau, which only holds when the cylindrical shear u_z u_r is
mapped to R_xy with the radial-to-wall-normal sign flip; getting the flip wrong
inflates the residual to order one.
"""
import numpy as np
import pytest

from UQ.datasets import PipeDNS, PIPE_CASES

# Per-case header parameters (DNS_data/README.md): actual friction Reynolds
# number, bulk Reynolds number, point count.
KNOWN = {
    500:   (495.6,   1.7e4,  95),
    1140:  (1132.2,  4.4e4,  163),
    2000:  (1972.0,  8.25e4, 242),
    3000:  (3027.3,  1.33e5, 326),
    6000:  (6007.9,  2.85e5, 545),
    12000: (12054.6, 6.12e5, 1023),
}

pytestmark = pytest.mark.skipif(
    not PipeDNS.is_available(2000),
    reason="pipe DNS_data not present (bulk data is local/gitignored)",
)


@pytest.fixture(scope="module")
def cases():
    return {n: PipeDNS.load(n) for n in PIPE_CASES if PipeDNS.is_available(n)}


@pytest.mark.parametrize("n", PIPE_CASES)
def test_header_parameters_match(cases, n):
    """Actual Re_tau, bulk Re, and point count match the header."""
    if n not in cases:
        pytest.skip(f"Re_tau={n} not present")
    c = cases[n]
    re_tau, re_bulk, npts = KNOWN[n]
    assert c.re_tau == pytest.approx(re_tau, rel=1e-3)
    assert c.re_bulk == pytest.approx(re_bulk, rel=1e-3)
    assert c.re_tau_nominal == n
    assert c.n == npts


@pytest.mark.parametrize("n", PIPE_CASES)
def test_cylindrical_to_cartesian_mapping(cases, n):
    """Variances map u_z^2->R_xx, u_r^2->R_yy, u_t^2->R_zz; R_xy = -u_z u_r < 0."""
    if n not in cases:
        pytest.skip(f"Re_tau={n} not present")
    c = cases[n]
    assert np.allclose(c.R, np.swapaxes(c.R, 1, 2))            # symmetric
    assert np.allclose(c.R[:, 0, 1], -c.uzur)                  # shear sign flip
    assert np.all(c.uv[1:] < 0.0)                              # R_xy negative
    assert np.allclose(c.R[:, 0, 2], 0.0)                      # u'w' vanishes
    assert np.allclose(c.R[:, 1, 2], 0.0)                      # v'w' vanishes
    k_from_R = 0.5 * np.trace(c.R, axis1=1, axis2=2)
    assert np.allclose(k_from_R, c.k, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n", PIPE_CASES)
def test_linear_total_stress_identity(cases, n):
    """dU^+/dy^+ - R_xy = 1 - y^+/Re_tau across the radius (the sign-flip anchor)."""
    if n not in cases:
        pytest.skip(f"Re_tau={n} not present")
    c = cases[n]
    total = c.total_stress_plus()
    target = c.total_stress_target()
    interior = (c.yplus > 5.0) & (c.y_outer < 0.9)
    rms = float(np.sqrt(np.mean((total[interior] - target[interior]) ** 2)))
    assert rms < 5.0e-3                                        # under 0.5% for all Re
    # the un-flipped mapping would miss the identity by order one
    wrong = c.dUdy() + c.uv                                    # dU/dy - (+u_z u_r)
    rms_wrong = float(np.sqrt(np.mean((wrong[interior] - target[interior]) ** 2)))
    assert rms_wrong > 0.5


@pytest.mark.parametrize("n", PIPE_CASES)
def test_outer_coordinate_and_monotonic(cases, n):
    """y/R = y^+/Re_tau runs from the wall to ~1 at the axis, monotone in y^+."""
    if n not in cases:
        pytest.skip(f"Re_tau={n} not present")
    c = cases[n]
    assert np.all(np.diff(c.yplus) > 0)
    assert np.allclose(c.y_outer, c.yplus / c.re_tau)
    assert c.y_outer[-1] > 0.95                                # reaches into the core


def test_friction_factor_derived_when_header_omits_it(cases):
    """The lowest-Re file omits the friction factor; it is derived from
    f = 8 (2 Re_tau / Re_bulk)^2 and is self-consistent."""
    if 500 not in cases:
        pytest.skip("Re_tau=500 not present")
    c = cases[500]
    expected = 8.0 * (2.0 * c.re_tau / c.re_bulk) ** 2
    assert c.friction_factor == pytest.approx(expected, rel=1e-9)
    # a header that carries the factor (2000) is read, not derived
    if 2000 in cases:
        assert cases[2000].friction_factor == pytest.approx(1.828e-2, rel=2e-3)


def test_builds_realizable_dnsfield(cases):
    """The record builds a UQ DNSField whose Reynolds stress is realizable DNS."""
    from UQ import realizability
    c = cases[2000] if 2000 in cases else next(iter(cases.values()))
    fd = c.to_dnsfield(timescale_plus=np.ones(c.n))
    assert fd.R.shape == (c.n, 3, 3)
    assert np.all(realizability.is_realizable(c.R[1:]))
