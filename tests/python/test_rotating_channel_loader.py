"""Streamwise-rotating channel DNS loader (Univ. Manitoba) on the real raw files.

Skips where the gitignored DNS_data is absent (CI). Exercises the format quirks
(CRLF line endings, the Ro_tau 7.5 file with no .txt suffix, Ro_tau defined in a
nomenclature block above its flow-conditions value) and the rotation physics that
makes this the model-stress-test companion: a nonzero mean spanwise velocity and
nonzero out-of-plane shear stresses that a non-rotating Boussinesq baseline
cannot produce.
"""
import numpy as np
import pytest

from UQ.datasets import RotatingChannelDNS, ROTATING_CASES


pytestmark = pytest.mark.skipif(
    not RotatingChannelDNS.is_available(15),
    reason="rotating-channel DNS_data not present (bulk data is local/gitignored)",
)


@pytest.fixture(scope="module")
def cases():
    return {ro: RotatingChannelDNS.load(ro) for ro in ROTATING_CASES
            if RotatingChannelDNS.is_available(ro)}


@pytest.mark.parametrize("ro", ROTATING_CASES)
def test_flow_conditions_parsed(cases, ro):
    """Re_tau = 180 and the rotation number match the flow-conditions block, not
    the nomenclature definition above it; the Ro_tau 7.5 file (no .txt) loads."""
    if ro not in cases:
        pytest.skip(f"Ro_tau={ro} not present")
    c = cases[ro]
    assert c.re_tau == pytest.approx(180.0, abs=1e-6)
    assert c.ro_tau == pytest.approx(ro, rel=1e-6)


@pytest.mark.parametrize("ro", ROTATING_CASES)
def test_full_channel_grid(cases, ro):
    """The profile spans the full channel: y/h from +1 to -1, y^+ = (y/h) Re_tau."""
    if ro not in cases:
        pytest.skip(f"Ro_tau={ro} not present")
    c = cases[ro]
    assert c.n == 129
    assert c.y_outer[0] == pytest.approx(1.0, abs=1e-6)
    assert c.y_outer[-1] == pytest.approx(-1.0, abs=1e-6)
    assert np.allclose(c.yplus, c.y_outer * c.re_tau)
    assert np.all(np.diff(c.yplus) < 0)                        # monotone decreasing


@pytest.mark.parametrize("ro", ROTATING_CASES)
def test_full_anisotropy_and_k(cases, ro):
    """The Reynolds-stress tensor carries the rotation-induced out-of-plane shear,
    is symmetric, and the TKE column equals 0.5 tr(R)."""
    if ro not in cases:
        pytest.skip(f"Ro_tau={ro} not present")
    c = cases[ro]
    assert np.allclose(c.R, np.swapaxes(c.R, 1, 2))
    assert np.allclose(c.R[:, 0, 2], c.uw)                     # <uw> present
    assert np.allclose(c.R[:, 1, 2], c.vw)                     # <vw> present
    assert np.max(np.abs(c.uw)) > 1e-3                         # nonzero (rotation)
    k_from_R = 0.5 * np.trace(c.R, axis1=1, axis2=2)
    assert np.allclose(k_from_R, c.k, atol=1e-5)               # TKE column vs 0.5 tr(R)


@pytest.mark.parametrize("ro", ROTATING_CASES)
def test_mean_spanwise_velocity_and_gradient(cases, ro):
    """Streamwise rotation drives a mean W^+, so grad_u carries dU/dy and dW/dy."""
    if ro not in cases:
        pytest.skip(f"Ro_tau={ro} not present")
    c = cases[ro]
    assert c.W is not None
    assert np.max(np.abs(c.W)) > 1e-2
    g = c.velocity_gradient()
    assert np.allclose(g[:, 0, 1], c.dUdy())                   # dU/dy
    assert np.allclose(g[:, 2, 1], np.gradient(c.W, c.yplus))  # dW/dy nonzero
    assert np.max(np.abs(g[:, 2, 1])) > 0.0


def test_rotation_symmetry(cases):
    """Rotation symmetry of the streamwise-rotating channel: U^+ even and W^+ odd
    about the centerline (the two walls are mirror images, not identical)."""
    c = cases[15] if 15 in cases else next(iter(cases.values()))
    mid = c.n // 2
    for i in range(2, mid - 1):
        j = c.n - 1 - i
        assert c.U[i] == pytest.approx(c.U[j], rel=1e-3, abs=1e-3)
        assert c.W[i] == pytest.approx(-c.W[j], rel=1e-3, abs=1e-3)


def test_out_of_plane_shear_is_a_rotation_signature(cases):
    """A non-rotating channel has only <uv>; here a substantial fraction of the
    shear stress lives in the rotation-induced <uw>, <vw>, which the Boussinesq
    baseline structurally cannot represent."""
    for c in cases.values():
        assert c.out_of_plane_fraction() > 0.1


@pytest.mark.parametrize("ro", ROTATING_CASES)
def test_budget_residual_present_and_small(cases, ro):
    """The RSTE budget residual columns res_*^+ are loaded (N, 6) and are a small
    data-only convergence indicator."""
    if ro not in cases:
        pytest.skip(f"Ro_tau={ro} not present")
    c = cases[ro]
    assert c.budget_residual.shape == (c.n, 6)
    assert c.budget_residual_level() < 1e-2                    # well-converged budget


def test_builds_dnsfield(cases):
    """The record builds a UQ DNSField with the full active anisotropy."""
    c = cases[15] if 15 in cases else next(iter(cases.values()))
    fd = c.to_dnsfield(timescale_plus=np.ones(c.n))
    assert fd.grad_u.shape == (c.n, 3, 3)
    assert fd.R.shape == (c.n, 3, 3)
    assert np.max(np.abs(fd.grad_u[:, 2, 1])) > 0.0           # dW/dy is active
